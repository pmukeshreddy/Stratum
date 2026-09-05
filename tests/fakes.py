import asyncio

from pydantic import Field

from threadweave.models import Action, ModelRequest, ModelResponse, ProviderConfig
from threadweave.models import RunConfig as ProductionConfig


class TestConfig(ProductionConfig):
    __test__ = False
    provider: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(name="mock", model="deterministic")
    )


class ScriptedProvider:
    """Deterministic test provider. The persisted turn index selects each response."""

    def __init__(self, scripts: dict[str, list], *, delay=0):
        self.scripts, self.delay = scripts, delay
        self.requests: list[ModelRequest] = []
        self.active = 0
        self.peak_active = 0

    async def invoke(self, request, emit):
        self.requests.append(request)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            sequence = self.scripts.get(request.name, self.scripts.get("*", []))
            item = (
                sequence[request.turn]
                if request.turn < len(sequence)
                else ModelResponse(
                    actions=[Action(name="finish", arguments={"result": "Script complete"})]
                )
            )
            if isinstance(item, Exception):
                raise item
            if callable(item):
                item = item(request)
                if hasattr(item, "__await__"):
                    item = await item
            return item if isinstance(item, ModelResponse) else ModelResponse.model_validate(item)
        finally:
            self.active -= 1
