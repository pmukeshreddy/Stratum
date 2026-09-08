"""Bounded standard LSP queries against installed language servers.

Optional compiler enrichment; syntax/module indexing never depends on a server.
Server results retain UTF-16 ranges and source provenance, not lexical guesses.
"""

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from .execution import environment

SOURCE_ROOT = str(Path(__file__).resolve().parent.parent)

SERVERS = {
    "clangd": ["clangd", "--background-index=false", "--clang-tidy=false"],
    "rust-analyzer": ["rust-analyzer"],
    "gopls": ["gopls"],
    "typescript-language-server": ["typescript-language-server", "--stdio"],
    "pyright-langserver": ["pyright-langserver", "--stdio"],
}


async def query(context, path, line, column, operation="definition", server=None):
    source = context.path(path)
    config = context.runtime.store.config(context.session_id)
    if source.stat().st_size > 2_000_000:
        raise ValueError("LSP source exceeds the 2 MB interactive request bound")
    language = {
        ".py": "python",
        ".rs": "rust",
        ".go": "go",
        ".ts": "typescript",
        ".js": "javascript",
        ".c": "c",
        ".cu": "cpp",
        ".hpp": "cpp",
    }.get(source.suffix, "cpp")
    server = server or {
        "python": "pyright-langserver",
        "rust": "rust-analyzer",
        "go": "gopls",
        "typescript": "typescript-language-server",
        "javascript": "typescript-language-server",
    }.get(language, "clangd")
    if server not in SERVERS or not shutil.which(server):
        raise ValueError(
            f"Language server {server!r} is not installed; use structural repo queries or install/configure this server"
        )
    if (
        config.execution.command_allowlist is not None
        and server not in config.execution.command_allowlist
    ):
        raise PermissionError("Language server executable is not allowlisted")
    if operation not in {"definition", "declaration", "references", "implementation"}:
        raise ValueError("operation must be definition, declaration, references, or implementation")
    text = source.read_text()
    lines = text.splitlines()
    if not 1 <= line <= len(lines) or not 1 <= column <= len(lines[line - 1]) + 1:
        raise ValueError("Use one-based source line and column within the file")
    character = len(lines[line - 1][: column - 1].encode("utf-16-le")) // 2
    root = Path(context.session.workspace.path)
    bootstrap = f"import sys; sys.path.insert(0, {SOURCE_ROOT!r}); from threadweave.process_worker import main; main()"
    with (
        tempfile.TemporaryDirectory(prefix="threadweave-lsp-") as scratch,
        tempfile.TemporaryFile() as log,
    ):
        prefix = []
        if config.execution.read_only:
            from .isolation import readonly_worker

            prefix = readonly_worker(Path(scratch))
        process = await asyncio.create_subprocess_exec(
            *prefix,
            sys.executable,
            "-c",
            bootstrap,
            *SERVERS[server],
            cwd=root,
            env={**environment(config.execution), "TMPDIR": scratch},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )

        async def send(method, params, identifier=None):
            packet = {"jsonrpc": "2.0", "method": method, "params": params}
            if identifier is not None:
                packet["id"] = identifier
            data = json.dumps(packet).encode()
            process.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
            await process.stdin.drain()

        async def response(identifier):
            while True:
                size = None
                for _ in range(30):
                    header = await process.stdout.readline()
                    if not header:
                        raise ValueError(
                            "Language server exited before completing the request; stderr retained"
                        )
                    if header == b"\r\n":
                        break
                    if header.lower().startswith(b"content-length:"):
                        size = int(header.split(b":", 1)[1])
                if size is None or not 0 <= size <= 8_000_000:
                    raise ValueError("Invalid/unbounded LSP message")
                packet = json.loads(await process.stdout.readexactly(size))
                if packet.get("id") == identifier and "method" not in packet:
                    if "error" in packet:
                        raise ValueError(
                            "Language server error: " + json.dumps(packet["error"])[:1000]
                        )
                    return packet.get("result")
                if "id" in packet and "method" in packet:
                    reply = {
                        "jsonrpc": "2.0",
                        "id": packet["id"],
                        "result": [] if packet["method"] == "workspace/configuration" else None,
                    }
                    data = json.dumps(reply).encode()
                    process.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
                    await process.stdin.drain()

        try:
            async with asyncio.timeout(min(20, config.limits.tool_timeout_seconds)):
                await send(
                    "initialize",
                    {
                        "processId": os.getpid(),
                        "rootUri": root.as_uri(),
                        "capabilities": {"general": {"positionEncodings": ["utf-16"]}},
                    },
                    1,
                )
                initialized = await response(1)
                await send("initialized", {})
                await send(
                    "textDocument/didOpen",
                    {
                        "textDocument": {
                            "uri": source.as_uri(),
                            "languageId": language,
                            "version": 1,
                            "text": text,
                        }
                    },
                )
                params = {
                    "textDocument": {"uri": source.as_uri()},
                    "position": {"line": line - 1, "character": character},
                }
                if operation == "references":
                    params["context"] = {"includeDeclaration": True}
                await send("textDocument/" + operation, params, 2)
                result = await response(2)
            records = result if isinstance(result, list) else [result] if result else []
            return {
                "matches": [
                    {
                        "path": unquote(urlparse(r.get("uri", r.get("targetUri", ""))).path),
                        "range": r.get("range", r.get("targetSelectionRange")),
                        "relationship": operation,
                        "quality": "compiler/LSP semantic",
                        "rank_reason": f"{server} source-position resolution",
                        "coordinate_system": "zero-based UTF-16 LSP range",
                    }
                    for r in records[:50]
                ],
                "server": initialized.get("serverInfo", {"name": server}),
                "total": len(records),
            }
        finally:
            process.stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 0.5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            log.seek(0)
            artifact = context.runtime.artifacts.put_stream(
                context.session_id, log, source_event=context.source_event
            )
            context.runtime.store.event(
                context.session_id,
                "lsp_query",
                {
                    "server": server,
                    "path": path,
                    "operation": operation,
                    "stderr_artifact": artifact,
                },
                parent=context.source_event,
            )
