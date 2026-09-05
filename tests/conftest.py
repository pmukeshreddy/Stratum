from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from threadweave.models import Action, ModelResponse, RunConfig, Usage
from threadweave.runtime import Runtime


def response(tool_name: str, /, **arguments):
    return ModelResponse(
        actions=[Action(name=tool_name, arguments=arguments)],
        usage=Usage(input_tokens=20, output_tokens=10),
    )


async def eventually(predicate, seconds=10):
    async with asyncio.timeout(seconds):
        while True:
            result = predicate()
            if result:
                return result
            await asyncio.sleep(0.01)


@pytest.fixture
def config():
    return RunConfig(
        provider={"name": "mock", "model": "deterministic", "max_output_tokens": 128},
        retry={"initial_delay": 0, "max_delay": 0},
        limits={"max_turns": 50, "wall_seconds": 30},
    )


@pytest_asyncio.fixture
async def runtime(tmp_path):
    manager = Runtime(tmp_path / "data")
    try:
        yield manager
    finally:
        await manager.shutdown()
