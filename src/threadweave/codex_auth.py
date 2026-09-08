"""Codex-owned authentication via the official structured app-server protocol.

Never read auth.json, export credentials, or persist raw account/config responses.
This connection is for account control and model discovery, not agent execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from .models import HarnessError


class CodexControl:
    def __init__(self, executable="codex", *, timeout=30):
        self.executable = executable
        self.timeout = timeout
        self.process = None
        self.sequence = 0
        self.pending = {}
        self.notifications = asyncio.Queue()
        self.reader = None

    async def __aenter__(self):
        if not shutil.which(self.executable):
            raise HarnessError(
                "provider", "CODEX_NOT_INSTALLED", "Install the official Codex CLI first."
            )
        # Explicitly prohibit API-key fallback, including keys inherited by the daemon.
        environment = dict(os.environ)
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
            environment.pop(key, None)
        self.process = await asyncio.create_subprocess_exec(
            self.executable,
            "app-server",
            "--stdio",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=8 * 1024 * 1024,
        )
        self.reader = asyncio.create_task(self._read())
        try:
            await self.call(
                "initialize",
                {
                    "clientInfo": {"name": "threadweave", "version": "0.2.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_):
        if self.process and self.process.stdin:
            self.process.stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await self.process.stdin.wait_closed()
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 3)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.reader:
            self.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader

    async def _read(self):
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if "id" in message:
                    future = self.pending.get(message["id"])
                    if future and not future.done():
                        if "error" in message:
                            # Server messages can contain config values: do not log them.
                            future.set_exception(
                                HarnessError(
                                    "provider",
                                    "CODEX_RPC_ERROR",
                                    "Codex rejected the account request",
                                )
                            )
                        else:
                            future.set_result(message.get("result", {}))
                elif "method" in message:
                    await self.notifications.put(message)
        except (ValueError, OSError):
            pass
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(
                        HarnessError("provider", "CODEX_DISCONNECTED", "Codex transport closed")
                    )

    async def call(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            self.process.stdin.write(
                (json.dumps({"id": request_id, "method": method, "params": params}) + "\n").encode()
            )
            await self.process.stdin.drain()
            return await asyncio.wait_for(future, self.timeout)
        except TimeoutError as exc:
            raise HarnessError(
                "provider", "CODEX_TIMEOUT", "Codex account request timed out"
            ) from exc
        finally:
            self.pending.pop(request_id, None)

    async def status(self):
        # Status is not a forced token rotation. Inference's official AuthManager
        # performs age/expiry checks and unauthorized recovery when necessary.
        response = await self.call("account/read", {"refreshToken": False})
        account = response.get("account") or {}
        authenticated = account.get("type") == "chatgpt"
        return {
            "provider": "codex_subscription",
            "authentication": "ChatGPT" if authenticated else "none",
            "logged_in": authenticated,
            "plan": account.get("planType") if authenticated else None,
            "token_state": "managed_by_codex" if authenticated else "AUTH_REQUIRED",
            "credential_storage": "official Codex credential store (shared)",
        }

    async def models(self):
        models, cursor = [], None
        while True:
            page = await self.call("model/list", {"limit": 100, "cursor": cursor})
            for model in page.get("data", []):
                models.append(
                    {
                        key: model.get(key)
                        for key in (
                            "model",
                            "isDefault",
                            "defaultReasoningEffort",
                            "supportedReasoningEfforts",
                        )
                    }
                )
            cursor = page.get("nextCursor")
            if not cursor:
                return models

    async def settings(self):
        """Return an allowlisted non-secret projection, not the raw Codex configuration."""
        config = (await self.call("config/read", {"includeLayers": False})).get("config", {})
        if config.get("model_provider") not in (None, "openai"):
            raise HarnessError(
                "provider",
                "AUTH_CONFIGURATION",
                "Codex must select its OpenAI subscription provider",
            )
        if config.get("forced_login_method") == "api":
            raise HarnessError(
                "provider",
                "AUTH_CONFIGURATION",
                "Codex policy requires API authentication; subscription mode is unavailable",
            )
        workspaces = config.get("forced_chatgpt_workspace_id")
        if isinstance(workspaces, str):
            workspaces = [workspaces]
        return {
            "codex_home": os.environ.get("CODEX_HOME", str(Path.home() / ".codex")),
            "store": config.get("cli_auth_credentials_store") or "file",
            "keyring": config.get("auth_keyring_backend")
            or ("secrets" if os.name == "nt" else "direct"),
            "workspaces": workspaces,
            "respect_system_proxy": bool(config.get("respect_system_proxy")),
            "model": config.get("model"),
            "reasoning_effort": config.get("model_reasoning_effort"),
        }

    async def login(self, present: Callable, *, device=False):
        status = await self.status()
        if status["logged_in"]:
            return status
        reply = await self.call(
            "account/login/start", {"type": "chatgptDeviceCode" if device else "chatgpt"}
        )
        # Authorization URLs/codes are shown only to the human, never added to trajectories.
        present(reply)
        try:
            async with asyncio.timeout(600):
                while True:
                    event = await self.notifications.get()
                    if event["method"] != "account/login/completed":
                        continue
                    if event.get("params", {}).get("success"):
                        return await self.status()
                    raise HarnessError(
                        "provider",
                        "AUTH_REQUIRED",
                        "ChatGPT login failed; run threadweave auth login",
                    )
        except BaseException:
            if reply.get("loginId"):
                with contextlib.suppress(Exception):
                    await self.call("account/login/cancel", {"loginId": reply["loginId"]})
            raise

    async def logout(self):
        await self.call("account/logout", {})
        return await self.status()
