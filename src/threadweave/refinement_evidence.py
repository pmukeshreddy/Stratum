"""Cheap, persistent novelty filter. Ambiguous evidence is admitted for semantic review."""

import ast
import hashlib
import re
import shlex

from .storage import encode

SIGNALS = {
    "python_result",
    "python_error",
    "execution_failure_observed",
    "workspace_effects",
    "code_edit",
    "coding_command",
    "experiment_conclusion",
    "agent_message_received",
    "verification_result",
    "verification_evidence",
    "verifier_result",
    "skill_outcome",
    "semantic_state_updated",
}
NOISE_KEYS = {
    "timestamp",
    "duration",
    "duration_seconds",
    "execution_id",
    "action_id",
    "window_id",
    "stdout_artifact",
    "stderr_artifact",
    "snapshot_metrics",
    "variables",
    "state_manifest_artifact",
    "structured_artifact",
    "test_report_artifact",
    "process_effects",
}


def stable(value):
    if isinstance(value, dict):
        return {
            k: stable(v)
            for k, v in value.items()
            if k not in NOISE_KEYS and not k.endswith("_path")
        }
    if isinstance(value, list):
        return [stable(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\b[0-9a-f]{32,64}\b", "<identity>", value)
    return value


def classify(event, store):
    payload = event["payload"]
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        return True, "new structured evidence", stable(payload)
    content = stable(result)
    if event["type"] == "python_result":
        parent = event.get("parent_event_id")
        code = store.event_by_id(parent)["payload"].get("code", "") if parent else ""
        content = {"code": code, "result": content}
        if code:
            try:
                tree = ast.parse(code)
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
                shells = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "bash"]
                if shells and all(
                    n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)
                    and all(
                        part.strip() and shlex.split(part)[0] in {"pwd", "ls", "true", "echo"}
                        for part in n.args[0].value.split(";")
                    )
                    for n in shells
                ):
                    return False, "trivial shell observation", content
                trivial = (
                    len(tree.body) == 1
                    and isinstance(tree.body[0], ast.Expr)
                    and all(
                        isinstance(n.func, ast.Name)
                        and n.func.id in {"print", "len", "str", "repr"}
                        for n in calls
                    )
                )
                if trivial and not result.get("error") and len(str(result.get("stdout", ""))) < 160:
                    return False, "trivial observation with no new workflow", content
            except SyntaxError:
                pass  # IPython syntax is evidence too.
    if event["type"] == "verification_result" and result.get("passed") and not result.get("files"):
        return False, "unchanged continuous check", content
    if event["type"] == "coding_command" and result.get("passed"):
        command = result.get("requested_command", result.get("command", []))
        if command and command[0] in {"pwd", "ls", "true", "echo"}:
            return False, "trivial command success", content
    if event["type"] == "workspace_effects" and not result.get("file_count"):
        return False, "no workspace change", content
    return True, "new success, failure or discovery may improve later work", content


def prefilter(runtime, sid, *, retry=False):
    store = runtime.store
    seen = list(store.iter_events(sid, kind="refinement_event_seen"))
    fingerprints = {e["payload"]["fingerprint"] for e in seen}
    after = max((e["payload"]["source_seq"] for e in seen), default=0)
    if retry:
        fingerprints = set()
        after = 0
    candidates = [e for e in store.iter_events(sid, after=after) if e["type"] in SIGNALS]
    admitted = []
    for event in candidates:
        passed, reason, content = classify(event, store)
        fingerprint = hashlib.sha256(encode([event["type"], content]).encode()).hexdigest()
        if fingerprint in fingerprints:
            passed, reason = False, "identical evidence already inspected"
        fingerprints.add(fingerprint)
        info = {
            "source_event": event["id"],
            "source_seq": event["seq"],
            "fingerprint": fingerprint,
            "reason": reason,
        }
        marker = store.event(sid, "refinement_event_seen", info, parent=event["id"])
        store.event(
            sid,
            "refinement_prefilter_pass" if passed else "refinement_prefilter_reject",
            info,
            parent=marker,
        )
        if passed:
            admitted.append(event["id"])
    return admitted


def bounded_evidence(runtime, sid, value, budget):
    """Remove repeated raw payloads before auxiliary inference, retaining source IDs.

    The entire input is archived. Small complete inputs pass through unchanged;
    oversized observations carry head/tail excerpts and explicit archive handles.
    """
    from .context import token_bound
    from .models import HarnessError

    model = runtime.store.config(sid).provider.model
    if token_bound(value, model) <= budget:
        return value
    artifact = runtime.artifacts.put(sid, value)

    def excerpt(text, cap):
        if len(text) <= cap:
            return text
        return (
            text[: cap // 2]
            + f"\n[omitted middle: artifacts.load({artifact!r})]\n"
            + text[-cap // 2 :]
        )

    def reduce(item, cap, list_cap=1000):
        if (
            isinstance(item, dict)
            and "id" in item
            and "type" in item
            and ("body" in item or "payload" in item)
        ):
            # One event gets one allocation. Recursing into every nested field
            # multiplies the allowance and can exceed the budget even at tiny caps.
            body = item.get("body", item.get("payload"))
            return {
                **{
                    k: item[k]
                    for k in ("id", "seq", "type", "role", "timestamp", "session_id")
                    if k in item
                },
                "body": excerpt(body if isinstance(body, str) else encode(body), cap),
                "complete_record_artifact": artifact,
            }
        if isinstance(item, str) and len(item) > cap:
            try:
                return encode(reduce(__import__("json").loads(item), cap, list_cap))
            except ValueError:
                return (
                    item[: cap // 2]
                    + f"\n[omitted middle: artifacts.load({artifact!r})]\n"
                    + item[-cap // 2 :]
                )
        if isinstance(item, list):
            chosen = (
                item
                if len(item) <= list_cap
                else item[: list_cap // 4] + item[-(list_cap - list_cap // 4) :]
            )
            return [reduce(v, cap, list_cap) for v in chosen]
        if isinstance(item, dict):
            return {
                k: reduce(v, cap, list_cap)
                for k, v in item.items()
                if k not in {"snapshot_metrics", "process_effects", "variables", "tests"}
            }
        return item

    for cap, list_cap in ((3000, 1000), (1500, 32), (600, 12), (300, 6)):
        candidate = reduce(value, cap, list_cap)
        candidate["original_task"] = value["original_task"]
        candidate["complete_evidence_artifact"] = artifact
        candidate["evidence_policy"] = (
            "Source identifiers and observation excerpts; full raw evidence archived. Do not infer facts from omitted content."
        )
        if token_bound(candidate, model) <= budget:
            runtime.store.event(
                sid,
                "refinement_evidence_bounded",
                {"artifact": artifact, "tokens": token_bound(candidate, model), "budget": budget},
            )
            return candidate
    raise HarnessError(
        "runtime",
        "refinement_evidence_capacity",
        "Structured evidence exceeds auxiliary budget; retained for a later checkpoint",
    )
