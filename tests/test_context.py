import pytest

from threadweave.context import Context
from threadweave.models import Workspace
from threadweave.storage import Store

from .fakes import TestConfig as RunConfig


def test_compaction_preserves_full_history_and_complete_tool_pairs(tmp_path):
    store = Store(tmp_path / "db")
    config = RunConfig(
        provider={"max_output_tokens": 128},
        context={"max_tokens": 6000, "summary_chars": 600, "recent_blocks": 2},
    )
    root = store.create("Task", Workspace(path=str(tmp_path)), config)
    ids = []
    for i in range(20):
        eid = store.event(root.id, "observation", {"text": f"detail-{i}-" + "x" * 2000})
        ids.append(eid)
        store.add_context(
            root.id,
            eid,
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(i), "content": "x" * 2000},
            ],
        )
    context = Context(store)
    messages, bound = context.assemble(root.id, [])
    assert bound <= config.context.max_tokens - 128
    assert store.session(root.id).summary
    assert store.events(root.id, kind="context_compaction")
    for eid in ids:
        assert len(store.event_by_id(eid)["payload"]["text"]) > 2000
    for i, message in enumerate(messages):
        if message["role"] == "tool":
            assert messages[i - 1]["tool_calls"][0]["id"] == message["tool_call_id"]
    store.close()
    store = Store(tmp_path / "db")
    assert Context(store).messages(root.id) == messages
    assert store.records.count("compactions") > 0
    store.close()


def test_unknown_resume_does_not_create_session(runtime):
    with pytest.raises(KeyError, match="Unknown session"):
        runtime.resume("does-not-exist")
    assert runtime.store.sessions() == []
