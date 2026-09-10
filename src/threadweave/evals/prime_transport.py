"""Prime evaluation transport: actual Codex provider, no agent or refinement logic."""

import asyncio
import json
import sys

from ..models import ModelRequest
from ..subscription import SubscriptionProvider
from .harness import discard


async def main():
    request = ModelRequest.model_validate(json.load(sys.stdin))
    provider = SubscriptionProvider()
    request.config, _ = await provider.resolve(
        request.config, reasoning_off=request.reasoning_mode == "off"
    )
    response = await provider.invoke(request, discard)
    payload = response.model_dump(mode="json")
    if request.metadata.get("retain_provider_continuation"):
        payload["provider_items"] = response.provider_items
    print(json.dumps(payload))


if __name__ == "__main__":
    asyncio.run(main())
