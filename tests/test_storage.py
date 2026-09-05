import sqlite3

import pytest
from pydantic import ValidationError

from threadweave.models import Lifecycle, StateEdit, Usage, Workspace
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
    assert restored.db.execute("SELECT version FROM schema_migrations").fetchone()[0] == 1
    restored.close()


def test_append_only_and_atomic_event_transaction(tmp_path):
    store = Store(tmp_path / "db")
    root = make(store, tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.db.execute("DELETE FROM events")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.db.execute("UPDATE events SET type='fake'")
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


def test_refinement_versioning_delete_rollback_and_conflict(tmp_path):
    store = Store(tmp_path / "db")
    root = make(store, tmp_path)
    evidence = store.event(root.id, "observation", {"temperature": 20})
    edit = StateEdit(
        title="Observation",
        content={"text": "20 C"},
        source_events=[evidence],
        intended_effect="Retain measured temperature",
        select=True,
    )
    store.queue_refinement(root.id, edit)
    assert store.states(root.id) == []
    entry_id = store.apply_refinements(root.id)[0]
    assert store.state(root.id, entry_id)["version"] == 1
    assert entry_id in store.session(root.id).selected_state
    store.queue_refinement(
        root.id,
        edit.model_copy(
            update={"entry_id": entry_id, "content": {"text": "22 C"}, "expected_version": 1}
        ),
    )
    store.apply_refinements(root.id)
    assert store.state(root.id, entry_id)["content"]["text"] == "22 C"
    assert store.state(root.id, entry_id, 1)["content"]["text"] == "20 C"
    store.queue_refinement(
        root.id, edit.model_copy(update={"entry_id": entry_id, "expected_version": 1})
    )
    assert store.apply_refinements(root.id) == []
    assert store.state(root.id, entry_id)["version"] == 2
    store.queue_refinement(
        root.id, edit.model_copy(update={"entry_id": entry_id, "operation": "delete"})
    )
    store.apply_refinements(root.id)
    assert store.states(root.id) == []
    store.queue_refinement(
        root.id,
        edit.model_copy(
            update={"entry_id": entry_id, "operation": "rollback", "rollback_version": 1}
        ),
    )
    store.apply_refinements(root.id)
    recovered = store.state(root.id, entry_id)
    assert recovered["version"] == 4 and not recovered["deleted"]
    assert recovered["content"]["text"] == "20 C"
    assert recovered["provenance"]["rollback_version"] == 1
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        store.db.execute("DELETE FROM state_versions")
    store.close()


@pytest.mark.parametrize(
    "kind,content",
    [
        ("memory", {"text": "A fact"}),
        ("prompt_note", {"text": "Check evidence"}),
        ("skill", {"name": "answer", "description": "Compute answer", "code": "answer = 42"}),
        ("subagent_spec", {"instruction": "Review artifacts"}),
    ],
)
def test_typed_state_scope_provenance_and_global_permission(tmp_path, kind, content):
    store = Store(tmp_path / "db")
    root = make(store, tmp_path)
    other = make(store, tmp_path)
    event = store.event(root.id, "evidence", {})
    edit = StateEdit(kind=kind, content=content, source_events=[event], intended_effect="Reuse")
    store.queue_refinement(root.id, edit)
    eid = store.apply_refinements(root.id)[0]
    with pytest.raises(KeyError):
        store.state(other.id, eid)
    with pytest.raises(PermissionError):
        store.queue_refinement(root.id, edit.model_copy(update={"scope": "global"}))
    with pytest.raises(PermissionError):
        store.queue_refinement(other.id, edit)
    assert store.state(root.id, eid)["provenance"]["source_events"] == [event]
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
