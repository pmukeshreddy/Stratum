import json

import pytest
from pydantic import ValidationError

from threadweave.models import Lifecycle, Usage, Workspace
from threadweave.storage import Store

from .fakes import TestConfig as RunConfig


def make(store, tmp_path, **kwargs):
    return store.create("Task", Workspace(path=str(tmp_path)), RunConfig(), **kwargs)


def test_lifecycle_events_identity_and_restart(tmp_path):
    store = Store(tmp_path / "db")
    root = make(store, tmp_path)
    child = make(store, tmp_path, parent_id=root.id, name="child")
    store.transition(root.id, Lifecycle.RUNNING)
    with pytest.raises(ValueError):
        store.transition(root.id, Lifecycle.ADMITTED)
    store.transition(root.id, Lifecycle.IDLE)
    store.transition(root.id, Lifecycle.INACTIVE)
    eid = store.event(child.id, "observation", {"value": 42}, parent=store.events(root.id)[0]["id"])
    config_id = root.config_id
    store.close()
    restored = Store(tmp_path / "db")
    assert restored.session(root.id).lifecycle == Lifecycle.INACTIVE
    assert restored.session(child.id).parent_id == root.id
    assert restored.session(child.id).root_id == root.id
    assert restored.session(root.id).config_id == config_id
    assert restored.config(root.id).model_dump() == RunConfig().model_dump()
    event = restored.event_by_id(eid)
    assert event["payload"] == {"value": 42}
    assert event["timestamp"] > 0 and event["parent_event_id"]
    assert json.loads((restored.directory / "store.json").read_text())["format"] == 1
    restored.close()


def test_append_only_and_atomic_event_transaction(tmp_path):
    store = Store(tmp_path / "db")
    root = make(store, tmp_path)
    with pytest.raises(ValueError, match="append-only"):
        store.records.delete("events")
    with pytest.raises(ValueError, match="append-only"):
        store.records.update("events", {"type": "fake"})
    before = len(store.events(root.id))
    with pytest.raises(ValueError), store.transaction():
        store.event(root.id, "rolled_back", {})
        store.update(root.id, name="bad")
        raise ValueError("rollback")
    assert len(store.events(root.id)) == before
    assert store.session(root.id).name == "root"
    store.close()


def test_messages_delivered_atomically_and_only_once(tmp_path):
    store = Store(tmp_path / "db")
    root = make(store, tmp_path)
    child = make(store, tmp_path, parent_id=root.id)
    message = store.send(root.id, child.id, "persistent message")
    store.close()
    store = Store(tmp_path / "db")
    assert store.messages(child.id, pending=True)[0]["id"] == message
    delivered = store.receive(child.id, lambda m: m["body"])
    assert len(delivered) == 1
    assert store.receive(child.id, lambda m: m["body"]) == []
    assert "persistent message" in str(store.session(child.id).context)
    assert len(store.events(child.id, kind="agent_message_received")) == 1
    store.close()


def test_recursive_usage_includes_all_descendants(tmp_path):
    store = Store(tmp_path / "db")
    root = make(store, tmp_path)
    child = make(store, tmp_path, parent_id=root.id)
    grandchild = make(store, tmp_path, parent_id=child.id)
    store.charge(root.id, Usage(input_tokens=10, model_calls=1, cost=0.1))
    store.charge(child.id, Usage(output_tokens=20, python_executions=1, tool_calls=2))
    store.charge(grandchild.id, Usage(input_tokens=30, retries=1, verifier_calls=1, wall_seconds=2))
    total = store.usage(root.id, tree=True)
    assert total.input_tokens == 40 and total.output_tokens == 20
    assert total.subagent_count == 2 and total.retries == 1
    assert total.python_executions == 1 and total.tool_calls == 2
    assert total.cost == 0.1 and total.wall_seconds == 2
    assert total == store.usage(child.id, tree=True)
    store.close()


def test_config_validation_and_roundtrip():
    config = RunConfig()
    assert RunConfig.model_validate_json(config.model_dump_json()) == config
    for invalid in (
        {"unknown": True},
        {"limits": {"concurrency": 0}},
        {"task": {"require_verifier": True}},
        {"limits": {"cost_budget": 2}},
        {"provider": {"max_output_tokens": 100000}},
    ):
        with pytest.raises(ValidationError):
            RunConfig.model_validate(invalid)
