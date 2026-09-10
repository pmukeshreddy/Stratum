"""Duplex request/response transport over an owned Docker exec's stdio."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid

from ..models import HarnessError


class Peer:
    def __init__(self, reader, send, handler):
        self.reader, self.send, self.handler = reader, send, handler
        self.pending, self.handlers = {}, {}
        self.write_lock = asyncio.Lock()
        self.pump = asyncio.create_task(self.read())

    async def write(self, value):
        async with self.write_lock:
            await self.send((json.dumps(value, ensure_ascii=True) + "\n").encode())

    async def call(self, method, **args):
        identifier = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.write({"id": identifier, "method": method, "args": args})
            return await future
        except asyncio.CancelledError:
            with contextlib.suppress(BrokenPipeError, ConnectionError):
                await self.write({"cancel": identifier})
            raise
        finally:
            self.pending.pop(identifier, None)

    async def handle(self, packet):
        identifier = packet["id"]
        try:
            result = await self.handler(packet["method"], packet["args"])
            reply = {"id": identifier, "result": result}
        except asyncio.CancelledError:
            return
        except Exception as exc:
            failure = (
                exc.failure.model_dump()
                if isinstance(exc, HarnessError)
                else {"category": "environment", "code": type(exc).__name__, "message": str(exc)}
            )
            reply = {"id": identifier, "error": failure}
        finally:
            self.handlers.pop(identifier, None)
        await self.write(reply)

    async def read(self):
        try:
            while line := await self.reader.readline():
                packet = json.loads(line)
                if "cancel" in packet:
                    if task := self.handlers.get(packet["cancel"]):
                        task.cancel()
                elif "method" in packet:
                    self.handlers[packet["id"]] = asyncio.create_task(self.handle(packet))
                elif (future := self.pending.get(packet["id"])) and not future.done():
                    if "error" in packet:
                        future.set_exception(HarnessError(**packet["error"]))
                    else:
                        future.set_result(packet["result"])
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("Buffalo worker transport closed"))
            for task in list(self.handlers.values()):
                task.cancel()

    async def close(self):
        self.pump.cancel()
        tasks = [self.pump, *self.handlers.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
