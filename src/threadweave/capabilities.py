"""Adapter-extensible capability and child-profile descriptions."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityProvider:
    name: str
    register_tools: Callable
    namespace_factory: str | None = None
    instructions: Callable | None = None
    service: object | None = None


@dataclass(frozen=True)
class ChildProfile:
    isolate: bool | None = None
    require_isolation: bool = False
    read_only: bool = False
    instruction: str = ""
    configure: Callable | None = None


def default_adapters():
    from .coding_adapter import CodingAdapter
    from .tasks import WorkspaceTask

    return {"workspace": WorkspaceTask(), "coding": CodingAdapter()}
