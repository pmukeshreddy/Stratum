"""A non-coding extension exercised through the public runtime and real kernel."""

from threadweave.capabilities import CapabilityProvider, ChildProfile
from threadweave.tasks import WorkspaceTask
from threadweave.tools import Empty, Tool


def namespaces(bridge, values, metadata, host):
    from threadweave.kernel_api import Capability

    values["observatory"] = Capability(bridge, {"sample": ("sample_observation", [])})


def install(runtime):
    async def sample(context, args):
        return {"observation": "measured", "session_id": context.session_id}

    def register(registry):
        registry.register(
            Tool(
                "sample_observation",
                "Sample an observation",
                Empty,
                sample,
            )
        )

    class Observatory(WorkspaceTask):
        capabilities = ("observatory",)
        profiles = {
            "survey": ChildProfile(isolate=False, instruction="Return measured observations.")
        }

    runtime.adapters["observatory"] = Observatory()
    runtime.environment.register_capability(
        CapabilityProvider(
            "observatory",
            register,
            "tests.generality_extension:namespaces",
            lambda config: "Use observatory.sample() to obtain measured observations.",
        )
    )
