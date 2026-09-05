from threadweave.models import Action, ModelResponse, Outcome
from threadweave.providers import ScriptedProvider

from .conftest import response


async def test_two_python_workers_reach_a_barrier_concurrently(runtime, tmp_path, config):
    def worker_code(own, peer):
        return (
            "import time\n"
            f"(workspace / '{own}').touch()\n"
            "deadline = time.monotonic() + 3\n"
            f"while not (workspace / '{peer}').exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            f"assert (workspace / '{peer}').exists(), 'Workers did not run concurrently'\n"
            "'Both workers crossed the barrier'"
        )

    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [
                ModelResponse(
                    actions=[
                        Action(
                            name="agent_spawn",
                            arguments={"instruction": "First worker", "name": "left"},
                        ),
                        Action(
                            name="agent_spawn",
                            arguments={"instruction": "Second worker", "name": "right"},
                        ),
                    ]
                ),
                response("agent_wait", seconds=0.3),
                response("finish", result="Concurrent computation"),
            ],
            "left": [
                response("python", code=worker_code("left-ready", "right-ready")),
                response("finish", result="Left crossed"),
            ],
            "right": [
                response("python", code=worker_code("right-ready", "left-ready")),
                response("finish", result="Right crossed"),
            ],
        }
    )
    root = runtime.create("Actual parallel computation", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    for child in runtime.store.sessions(root_id=root.id)[1:]:
        assert runtime.store.events(child.id, kind="python_result")
        assert not runtime.store.events(child.id, kind="python_error")


async def test_example_adapter_uses_custom_verifier_name(runtime, tmp_path, config):
    config.extensions = ["examples.extension:install"]
    config.task.adapter = "measurement"
    config.task.verifier = "measurement"
    config.task.verifier_options = {"maximum_error": 0.01}
    config.task.require_verifier = True
    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [
                response("workspace_write", path="measurement.json", content='{"error":0.001}')
            ],
        }
    )
    root = runtime.create("Custom experiment", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    assert runtime.store.events(root.id, kind="verifier_result")[-1]["payload"]["passed"]
