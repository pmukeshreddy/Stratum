"""Explicit fixed/role routing; every choice has inspectable inputs and reasons."""

from .models import new_id, now


def route(store, sid, role="agent", *, context_size=0, expected_tools=True, latency="normal"):
    config = store.config(sid)
    alias = config.routing.default
    reason = "fixed configured provider"
    if config.routing.policy == "role_based" and role in config.routing.roles:
        alias = config.routing.roles[role]
        reason = f"role_based mapping for {role}"
    elif alias:
        reason = "configured default model alias"
    provider = config.models[alias] if alias else config.provider
    decision = {
        "role": role,
        "alias": alias,
        "provider": provider.name,
        "model": provider.model,
        "reason": reason,
        "context_size": context_size,
        "expected_tools": expected_tools,
        "latency_preference": latency,
        "task_type": config.task.adapter,
        "remaining_cost": None
        if config.limits.cost_budget is None
        else config.limits.cost_budget - store.usage(sid, tree=True).cost,
        "policy": config.routing.policy,
    }
    identifier = new_id()
    store.records.insert(
        "routing_decisions",
        {"id": identifier, "session_id": sid, "created_at": now(), "body": decision},
    )
    store.event(sid, "model_routing", {"decision_id": identifier, **decision})
    return provider
