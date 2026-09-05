"""Test-only provider for terminal/daemon subprocess acceptance tests."""

from threadweave.models import ModelResponse

from .conftest import response


class ChatScenario:
    async def invoke(self, request, emit):
        actions = [
            response("repo_search", query="def add"),
            ModelResponse(text="First conversation reply: add currently subtracts."),
            response("python", code="chat_marker = 73\nprint(chat_marker)"),
            ModelResponse(text="Second conversation reply: marker is retained."),
            response("python", code="assert chat_marker == 73\nprint(chat_marker)"),
            ModelResponse(text="Recovered conversation reply: marker is still 73."),
        ]
        result = actions[min(request.turn, len(actions) - 1)]
        if result.text:
            await emit(result.text[:12])
            await emit(result.text[12:])
        return result


def install(runtime):
    runtime.providers["chat_scenario"] = ChatScenario()
