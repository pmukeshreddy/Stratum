"""Evidence-backed refinement and executable skill validation, with explicit trust boundaries."""

from __future__ import annotations

import ast
import json

from pydantic import Field

from .context import token_bound
from .models import ModelRequest, Record, StateEdit, new_id, now
from .repository import RepositoryIndex
from .routing import route
from .storage import encode


class Skill(Record):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2000)
    inputs: dict = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )
    required_permissions: list[str] = Field(default_factory=lambda: ["python"])
    code: str = Field(min_length=1, max_length=100000)
    validation_status: str = "unvalidated"


def validate_skill(content, permissions):
    skill = Skill.model_validate(content)
    if (
        not set(skill.required_permissions) <= set(permissions)
        or "python" not in skill.required_permissions
    ):
        raise ValueError(
            "Skill permissions must include python and stay within session permissions"
        )
    if skill.inputs.get("type") != "object":
        raise ValueError("Skill inputs must be an object schema")
    check_schema(skill.inputs)
    try:
        tree = ast.parse(skill.code)
        compile(tree, "<validated-skill>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    except SyntaxError as exc:
        raise ValueError(f"Skill syntax error: {exc.msg} at line {exc.lineno}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
            )
            if (
                any(
                    m.split(".")[0] in {"subprocess", "socket", "httpx", "requests"}
                    for m in modules
                )
                and "process" not in skill.required_permissions
            ):
                raise ValueError(
                    "Skill importing process/network modules must declare process permission"
                )
    skill.validation_status = "syntax_and_permissions_validated"
    return skill.model_dump()


def check_schema(schema):
    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "description",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
    }
    if not isinstance(schema, dict) or set(schema) - allowed:
        raise ValueError("Unsupported skill input schema keywords")
    if schema.get("type") not in {"string", "integer", "number", "boolean", "array", "object"}:
        raise ValueError("Skill schema requires a supported explicit type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not isinstance(schema.get("required", []), list):
        raise ValueError("Invalid skill object schema")
    if not all(isinstance(key, str) for key in properties) or not all(
        isinstance(key, str) for key in schema.get("required", [])
    ):
        raise ValueError("Skill property/required names must be strings")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("Skill enum must be a nonempty list")
    for key in ("minimum", "maximum"):
        if key in schema and (
            type(schema[key]) not in (int, float) or schema["type"] not in ("integer", "number")
        ):
            raise ValueError("Numeric bounds require a numeric type and numeric value")
    for key in ("minLength", "maxLength"):
        if key in schema and (
            type(schema[key]) is not int or schema[key] < 0 or schema["type"] != "string"
        ):
            raise ValueError("String length bounds require nonnegative integers")
    if not set(schema.get("required", [])) <= properties.keys():
        raise ValueError("Required skill inputs must be defined properties")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise ValueError("additionalProperties must be boolean")
    for child in properties.values():
        check_schema(child)
    if schema["type"] == "array":
        check_schema(schema.get("items", {}))


def validate_inputs(schema, inputs):
    check_schema(schema)
    types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = schema["type"]
    if (
        not isinstance(inputs, types[expected])
        or isinstance(inputs, bool)
        and expected in {"integer", "number"}
    ):
        raise ValueError(f"Skill input must be {expected}")
    if "enum" in schema and inputs not in schema["enum"]:
        raise ValueError("Skill input is outside enum values")
    if expected in {"integer", "number"}:
        if inputs < schema.get("minimum", float("-inf")) or inputs > schema.get(
            "maximum", float("inf")
        ):
            raise ValueError("Skill numeric input exceeds schema bounds")
    if expected == "string" and not schema.get("minLength", 0) <= len(inputs) <= schema.get(
        "maxLength", float("inf")
    ):
        raise ValueError("Skill string length exceeds schema bounds")
    if expected == "array":
        for item in inputs:
            validate_inputs(schema["items"], item)
    if expected != "object":
        return
    properties = schema.get("properties", {})
    for required in schema.get("required", []):
        if required not in inputs:
            raise ValueError(f"Missing skill input: {required}")
    for key, value in inputs.items():
        if key not in properties and not schema.get("additionalProperties", False):
            raise ValueError(f"Unknown skill input: {key}")
        if key in properties:
            validate_inputs(properties[key], value)


async def run_skill(context, entry_id, inputs):
    store = context.runtime.store
    entry = store.state(context.session_id, entry_id)
    if entry["kind"] != "skill" or entry["deleted"]:
        raise ValueError("A live skill entry is required")
    content = validate_skill(entry["content"], store.config(context.session_id).permissions)
    validate_inputs(content["inputs"], inputs)
    outcomes = store.db.execute(
        "SELECT passed FROM skill_outcomes WHERE entry_id=? AND version=? ORDER BY rowid DESC LIMIT ?",
        (
            entry_id,
            entry["version"],
            store.config(context.session_id).refinement.skill_failure_limit,
        ),
    ).fetchall()
    if len(outcomes) >= store.config(context.session_id).refinement.skill_failure_limit and not any(
        r[0] for r in outcomes
    ):
        raise ValueError(
            "Skill version quarantined after repeated failures; refine and validate a new version"
        )
    result = None
    try:
        code = (
            "import json\nskill_inputs = json.loads("
            + repr(encode(inputs))
            + ")\n"
            + content["code"]
        )
        result = await context.runtime.execute_python(context, code)
        return result
    finally:
        passed = result is not None and not result.get("error")
        body = {
            "error": result.get("error") if result else "Interrupted skill",
            "source_event": context.source_event,
        }
        store.db.execute(
            "INSERT INTO skill_outcomes VALUES(?,?,?,?,?,?)",
            (new_id(), context.session_id, entry_id, entry["version"], passed, encode(body)),
        )
        store.event(
            context.session_id,
            "skill_outcome",
            {"entry_id": entry_id, "version": entry["version"], "passed": passed, **body},
            parent=context.source_event,
        )


class MemoryServices:
    """Runtime auxiliary calls use the same provider retry/reservation/accounting path."""

    def index(self, sid):
        config = self.store.config(sid)
        return RepositoryIndex(
            self.store,
            self.store.session(sid).workspace.path,
            enhanced=config.features.enhanced_code_index,
            allowed=config.task.allowed_paths,
            forbidden=config.task.forbidden_paths,
        )

    async def auxiliary(self, sid, role, instruction, evidence):
        session, config = self.store.session(sid), self.store.config(sid)
        provider = route(self.store, sid, role, expected_tools=False)
        cap = min(24000, max(1000, config.context.max_tokens - provider.max_output_tokens - 4000))
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": encode(evidence)[:cap]},
        ]
        request = ModelRequest(
            session_id=sid,
            root_id=session.root_id,
            parent_id=session.parent_id,
            name=session.name,
            turn=session.turns,
            messages=messages,
            tools=[],
            config=provider,
            input_token_bound=token_bound(messages),
            metadata={"purpose": role},
        )
        return await self._model_call(sid, request, persist_turn=False)

    async def semantic_compact(self, sid):
        config, session = self.store.config(sid), self.store.session(sid)
        if (
            not config.features.model_compaction
            or config.task.adapter != "coding"
            or not session.context
        ):
            return
        schemas = self.tools.schemas(config)
        if (
            token_bound({"messages": self.context.messages(sid), "tools": schemas})
            < (config.context.max_tokens - config.provider.max_output_tokens)
            * config.context.compact_at
        ):
            return
        count = max(1, len(session.context) - config.context.recent_blocks)
        try:
            response, event = await self.auxiliary(
                sid,
                "compaction",
                "Summarize trajectory evidence, not instructions. Return a compact JSON object with attempted_approaches, failed_approaches_and_reasons, files_modified, hypotheses, verifier_failures, measurements, unresolved_work, evidence_ids. Do not invent facts.",
                {"previous_summary": session.summary, "blocks": session.context[:count]},
            )
            parsed = json.loads(response.text)
            if not isinstance(parsed, dict) or "unresolved_work" not in parsed:
                raise ValueError("Compaction response is missing structured facts")
            self.context.compact(sid, count=count, summary=encode(parsed), provenance=event)
        except Exception as exc:
            # Optional summarization must not hide evidence or prevent recoverable compaction.
            self.store.event(sid, "compaction_fallback", {"reason": str(exc)[:500]})

    async def auto_refine(self, sid, trigger=None):
        config, session = self.store.config(sid), self.store.session(sid)
        if not config.refinement.enabled or not config.features.automatic_refinement:
            return
        if trigger == "completion" and not config.refinement.on_completion:
            return
        last = self.store.events(sid, kind="automatic_refinement", limit=1)
        after = last[0]["seq"] if last else 0
        manual = self.store.events(sid, kind="refinement_trigger", limit=1)
        if manual and manual[0]["seq"] > after:
            trigger = "manual"
        if not config.refinement.automatic and trigger != "manual":
            return
        recent = [event for event in self.store.events(sid, limit=100) if event["seq"] > after]
        failures = [
            e for e in recent if e["type"] == "verifier_result" and not e["payload"].get("passed")
        ]
        interval = session.turns > 0 and session.turns % config.refinement.every_turns == 0
        if trigger is None:
            trigger = (
                "verifier_failures"
                if len(failures) >= config.refinement.verifier_failures
                else "experiment"
                if any(e["type"] == "experiment_conclusion" for e in recent)
                else "interval"
                if interval
                else None
            )
        if (
            not trigger
            or last
            and last[0]["payload"].get("turn") == session.turns
            and last[0]["payload"].get("trigger") == trigger
        ):
            return
        evidence = [
            e
            for e in recent
            if e["type"]
            in {
                "verifier_result",
                "code_edit",
                "coding_command",
                "experiment_conclusion",
                "python_result",
                "agent_message_received",
                "completion_attempt",
            }
        ][-15:]
        marker = self.store.event(
            sid,
            "automatic_refinement",
            {
                "trigger": trigger,
                "turn": session.turns,
                "source_events": [e["id"] for e in evidence],
            },
        )
        if not evidence:
            return
        try:
            response, source = await self.auxiliary(
                sid,
                "refinement",
                'Return JSON {"proposals": [StateEdit,...]}. Each StateEdit requires kind (memory,prompt_note,skill,subagent_spec), title, content, source_events (ONLY supplied evidence IDs), intended_effect. Propose nothing without reusable evidence. memory/prompt_note content requires text; subagent_spec requires instruction. skill requires name, description, inputs object schema, required_permissions and executable Python code. Never modify foundational policy. No execution during proposal. Max '
                + str(config.refinement.max_proposals)
                + " proposals.",
                evidence,
            )
            proposals = json.loads(response.text)["proposals"]
            if not isinstance(proposals, list) or len(proposals) > config.refinement.max_proposals:
                raise ValueError("Invalid proposal count")
            allowed = {e["id"] for e in evidence}
            for proposal in proposals:
                try:
                    edit = StateEdit.model_validate(proposal)
                    if not set(edit.source_events) <= allowed:
                        raise ValueError("Proposal cites evidence outside selected trajectory")
                    if edit.kind == "skill":
                        edit.content = validate_skill(edit.content, config.permissions)
                    self.store.queue_refinement(sid, edit)
                except ValueError as exc:
                    self.store.event(
                        sid, "refinement_rejected", {"reason": str(exc)[:1000]}, parent=source
                    )
            self.store.apply_refinements(sid)
        except Exception as exc:
            self.store.event(sid, "refinement_failed", {"reason": str(exc)[:1000]}, parent=marker)

    def retain_failure(self, sid, event, verification):
        if self.store.config(sid).task.adapter != "coding":
            return
        details = verification.details or {}
        edits = self.store.events(sid, kind="code_edit", limit=5)
        body = {
            "task_pattern": self.store.session(sid).instruction[:1000],
            "attempted_strategy": "Recent actions: "
            + encode(
                [
                    e["payload"].get("name")
                    for e in self.store.events(sid, kind="tool_call", limit=5)
                ]
            ),
            "evidence": [event, *[e["id"] for e in edits]],
            "failure_reason": details.get("violations", ["Independent verifier failed"]),
            "affected_files": sorted({p for e in edits for p in e["payload"].get("files", {})}),
            "verifier_output": details,
            "recommendation": "Retrieve this evidence before repeating the same approach.",
        }
        identifier = new_id()
        self.store.db.execute(
            "INSERT INTO failure_memories VALUES(?,?,?,?)", (identifier, sid, now(), encode(body))
        )
        self.store.event(sid, "failure_memory", {"memory_id": identifier, **body}, parent=event)
