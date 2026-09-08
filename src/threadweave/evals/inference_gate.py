"""Shared inference admission for Buffalo root, descendant, and auxiliary calls."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from ..models import HarnessError
from .schema import save, timestamp


class InferenceGate:
    def __init__(self, directory, capacity=16):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.initial_capacity = capacity
        self.active = self.peak = 0
        self.condition = asyncio.Condition()
        self.adjustments = []
        self.sequence = 0
        self.last_reduction = float("-inf")
        self.stopped = set()

    async def stop_admission(self, owner):
        """Block new calls after a terminal observation without interrupting a response."""
        async with self.condition:
            self.stopped.add(owner)
            self.condition.notify_all()

    def journal(self, **event):
        with (self.directory / "inference-events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": timestamp(), **event}) + "\n")

    @asynccontextmanager
    async def permit(self, owner):
        async with self.condition:
            await self.condition.wait_for(
                lambda: owner in self.stopped or self.active < self.capacity
            )
            if owner in self.stopped:
                raise HarnessError("environment", "environment_terminal", "Game already ended")
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.sequence += 1
            ticket = self.sequence
            self.journal(
                event="acquire",
                ticket=ticket,
                owner=owner,
                active=self.active,
                capacity=self.capacity,
            )
        try:
            yield ticket
        finally:
            async with self.condition:
                self.active -= 1
                self.journal(
                    event="release",
                    ticket=ticket,
                    owner=owner,
                    active=self.active,
                    capacity=self.capacity,
                )
                self.condition.notify_all()

    async def outcome(self, unstable=False):
        async with self.condition:
            # A 429 reduces immediately; transport bursts use this same explicit event.
            if unstable and self.capacity > 1 and time.monotonic() - self.last_reduction >= 2:
                previous = self.capacity
                self.capacity -= 1
                self.last_reduction = time.monotonic()
                change = {
                    "from": previous,
                    "to": self.capacity,
                    "reason": "provider throttling or transport failure",
                    "time": timestamp(),
                }
                self.adjustments.append(change)
                self.journal(event="capacity_reduced", **change)
                self.condition.notify_all()

    async def close(self):
        save(self.directory / "concurrency.json", self.summary())

    def summary(self):
        return {
            "initial_limit": self.initial_capacity,
            "final_limit": self.capacity,
            "peak_inflight": self.peak,
            "active": self.active,
            "adjustments": self.adjustments,
            "scope": "all Buffalo root/descendant/compaction/refinement calls",
        }
