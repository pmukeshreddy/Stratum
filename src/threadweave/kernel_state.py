"""L2 lifecycle: per-value checkpoints, durable offloads and explicit reconstruction.

All manifests/blobs are trusted local state, scoped to one kernel. Namespace changes
are committed only after their durable manifest succeeds. No history is replayed.
"""

from __future__ import annotations

import ast
import json
import sys
import time
from itertools import chain
from pathlib import Path

from .artifacts import atomic_write
from .models import KernelStatePolicy
from .snapshots import deadline


class StateHandle:
    """An explicit handle, never a transparent proxy with surprising side effects."""

    def __init__(self, state, name, metadata):
        self.state, self.name, self.metadata = state, name, metadata

    def load(self):
        return self.state.rehydrate(self.name)

    def __repr__(self):
        return (
            f"StateHandle({self.name!r}, action={self.metadata['action']!r}; load() to rehydrate)"
        )


def memory_size(value, max_nodes):
    seen, pending, total = set(), [iter((value,))], 0
    while pending and len(seen) < max_nodes:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += sys.getsizeof(item, 0)
        if type(item) is dict:
            pending.append(chain(item.keys(), item.values()))
        elif type(item) in (list, tuple, set, frozenset):
            pending.append(iter(item))
        elif type(item).__module__.startswith("pandas") and hasattr(item, "memory_usage"):
            usage = item.memory_usage(deep=True)
            total = max(total, int(usage.sum() if hasattr(usage, "sum") else usage))
    return total, bool(pending)


class KernelState:
    def __init__(self, worker, metadata):
        self.worker = worker
        self.owner = metadata.get("session_id", worker.directory.name)
        self.policy = KernelStatePolicy.model_validate(metadata.get("kernel_state", {}))
        self.cell = 0
        self.used, self.pinned, self.manifest = {}, set(), {}
        self.force = None

    def observe(self, tree):
        self.cell += 1
        # Loads inside functions are conservatively counted too. Dynamic lookup can
        # be pinned explicitly; automatic retirement always retains recovery data.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                self.used[node.id] = self.cell

    def retain(self, *names):
        self.pinned.update(names)

    def describe(self):
        return self.manifest

    def rehydrate(self, name, _loading=None):
        worker = self.worker
        loading = set() if _loading is None else _loading
        if name in loading:
            raise ValueError(f"Cyclic reconstruction dependency: {name}")
        row = self.manifest[name]
        if row.get("owner") != self.owner:
            raise ValueError("Kernel state owner mismatch")
        alias = row.get("alias_of")
        if alias and alias in self.manifest:
            value = worker.values.get(alias)
            if alias not in worker.values or isinstance(value, StateHandle):
                value = self.rehydrate(alias, loading | {name})
            worker.values[name] = value
        elif row.get("record") is not None:
            from .kernel_worker import unpack

            value = worker.blobs.decode(row["record"], lambda v: unpack(v, worker.host))
            worker.values[name] = value
        else:
            recipe = row.get("recipe")
            if not recipe:
                raise ValueError(f"No recovery path for {name}: {row.get('reason')}")
            for dep in recipe.get("dependencies", []):
                if dep not in worker.values or isinstance(worker.values[dep], StateHandle):
                    self.rehydrate(dep, loading | {name})
            if recipe.get("source_file") and recipe.get("sha256"):
                import hashlib

                if (
                    hashlib.sha256(Path(recipe["source_file"]).read_bytes()).hexdigest()
                    != recipe["sha256"]
                ):
                    raise ValueError("Reconstruction source changed")
            worker.values.pop(name, None)
            exec(compile(recipe["code"], "<recovery-recipe>", "exec"), worker.values)
            if name not in worker.values:
                raise ValueError(f"Recipe did not recreate {name}")
            value = worker.values[name]
        self.used[name] = self.cell
        # Observability only: keep successful explicit restores out of model context,
        # checkpoint state and tool results. Recovery already has a runtime receipt.
        if getattr(worker, "active_id", None):
            try:
                with (worker.directory / "restore-events.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "timestamp": time.time(),
                                "session_id": self.owner,
                                "execution_id": worker.active_id,
                                "name": name,
                                "action": row.get("action"),
                            }
                        )
                        + "\n"
                    )
            except OSError:
                pass  # Measurement failure must not change a task's execution.
        return value

    def restore(self):
        worker = self.worker
        restored, missing, reconstructed = [], {}, []
        data = None
        for path in (worker.checkpoint, worker.directory / "checkpoint.previous.json"):
            if not path.exists():
                continue
            try:
                candidate = json.loads(path.read_text())
                if candidate.get("version") not in (1, 2):
                    raise ValueError("Unsupported checkpoint version")
                if candidate.get("owner", self.owner) != self.owner:
                    raise ValueError("Kernel state owner mismatch")
                data = candidate
                break
            except Exception as exc:
                missing["__checkpoint__"] = f"{path.name}: {exc}"
        if data is None:
            return {"restored": restored, "missing": missing, "reconstructed": reconstructed}
        self.cell = data.get("cell", 0)
        self.used = data.get("used", {})
        self.pinned = set(data.get("pinned", []))
        worker.receipt = data.get("receipt")
        worker.recipes = {
            n: {"code": r} if isinstance(r, str) else r for n, r in data.get("recipes", {}).items()
        }
        missing.update(data.get("missing", {}))
        self.manifest = data.get("manifest", {})
        # Read existing version-1 checkpoints through the same restore path.
        for name, record in data.get("values", {}).items():
            self.manifest.setdefault(
                name, {"owner": self.owner, "action": "keep_live", "record": record}
            )
        for name, recipe in worker.recipes.items():
            self.manifest.setdefault(
                name, {"owner": self.owner, "action": "keep_live", "recipe": recipe}
            )
        for name, row in self.manifest.items():
            if name in worker.protected:
                continue
            try:
                if row["action"] == "prune":
                    continue
                if row["action"] in ("offload", "reconstruct"):
                    worker.values[name] = StateHandle(self, name, row)
                    restored.append(name)
                elif row.get("record") is not None or row.get("recipe"):
                    self.rehydrate(name)
                    (reconstructed if row.get("recipe") else restored).append(name)
                    missing.pop(name, None)
            except BaseException as exc:
                missing[name] = str(exc)
        return {"restored": restored, "missing": missing, "reconstructed": reconstructed}

    def snapshot(self, receipt, *, reason=None):
        worker, policy = self.worker, self.policy
        started = worker.blobs.begin()
        candidates, total = {}, 0
        for name, value in list(worker.values.items()):
            if name in worker.protected or name.startswith("__"):
                continue
            try:
                size, approximate = memory_size(value, policy.size_scan_nodes)
            except Exception:
                size, approximate = sys.getsizeof(value, 0), True
            total += size
            candidates[name] = (value, size, approximate)
        reason = reason or self.force
        self.force = None
        pressure = total > policy.memory_bytes or any(
            s > policy.variable_bytes for _, s, _ in candidates.values()
        )
        compact = bool(reason or pressure or self.cell % policy.checkpoint_cells == 0)
        reason = reason or (
            "memory_pressure" if pressure else "checkpoint" if compact else "execution"
        )
        manifest = {n: r for n, r in self.manifest.items() if r["action"] == "prune"}
        values, missing, changes, used = {}, {}, {}, 0
        identities = {}
        projected = total
        # Important/recent values get first claim on the checkpoint allowance.
        ordered = sorted(candidates, key=lambda n: (n not in self.pinned, -self.used.get(n, 0)))
        for name in ordered:
            value, size, approximate = candidates[name]
            if isinstance(value, StateHandle):
                row = dict(value.metadata)
                row["last_use_cell"] = self.used.get(name, row.get("last_use_cell", self.cell))
                if (
                    compact
                    and name not in self.pinned
                    and row.get("action") == "offload"
                    and self.cell - row["last_use_cell"] >= policy.stale_cells
                ):
                    row["action"] = "prune"
                    changes[name] = None
                manifest[name] = row
                continue
            if id(value) in identities:
                canonical = identities[id(value)]
                row = {**manifest[canonical], "name": name, "alias_of": canonical}
                manifest[name] = row
                if canonical in changes:
                    changes[name] = StateHandle(self, name, row) if changes[canonical] else None
                    projected -= size
                elif row.get("record") is not None:
                    values[name] = row["record"]
                continue
            if type(value) not in (str, bytes, int, float, bool, type(None)):
                identities[id(value)] = name
            row = {
                "name": name,
                "owner": self.owner,
                "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "memory_bytes": size,
                "size_is_estimate": approximate,
                "serialized_bytes": None,
                "last_use_cell": self.used.get(name, self.cell),
                "importance": "pinned" if name in self.pinned else "normal",
                "artifact_dependencies": [],
                "serializable": False,
                "reconstructible": name in worker.recipes,
                "action": "keep_live",
            }
            large = size > policy.variable_bytes or (
                projected > policy.memory_bytes and size > policy.inline_bytes
            )
            stale = self.cell - row["last_use_cell"] >= policy.stale_cells
            retire = compact and large and name not in self.pinned
            try:
                remaining = policy.snapshot_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise ValueError("Snapshot time budget exhausted")
                if name in worker.recipes:
                    row["recipe"] = worker.recipes[name]
                    row["artifact_dependencies"] = row["recipe"].get("artifacts", [])
                    if retire:
                        row["action"] = "reconstruct"
                        changes[name] = StateHandle(self, name, row)
                else:
                    with deadline(min(policy.variable_seconds, remaining)):
                        record, serialized = worker.blobs.encode(name, value, worker.pack)
                    row.update(record=record, serialized_bytes=serialized, serializable=True)
                    if record[0] == "blob":
                        row["artifact_dependencies"] = [record[1]["sha256"]]
                    if serialized > policy.variable_bytes and name not in self.pinned:
                        retire = True
                    if used + serialized > policy.snapshot_bytes:
                        retire = name not in self.pinned
                        record = worker.blobs.offload(record)
                        row["record"] = record
                        row["artifact_dependencies"] = [record[1]["sha256"]]
                    else:
                        used += serialized
                    if retire:
                        row["record"] = worker.blobs.offload(record)
                        row["artifact_dependencies"] = [row["record"][1]["sha256"]]
                        row["action"] = "prune" if stale else "offload"
                        changes[name] = None if stale else StateHandle(self, name, row)
                    else:
                        row["action"] = "snapshot_inline" if record[0] != "blob" else "keep_live"
                        values[name] = record
            except Exception as exc:
                row.update(action="skip", reason=str(exc))
                missing[name] = str(exc)
            manifest[name] = row
            if name in changes:
                projected -= size
        metrics = {
            **worker.blobs.stats,
            "seconds": time.monotonic() - started,
            "saved_variables": sum(r.get("serializable", False) for r in manifest.values()),
            "missing_variables": len(missing),
            "memory_bytes": total,
            "projected_live_bytes": projected,
            "reason": reason,
            "offloaded": [
                n for n, r in manifest.items() if n in changes and r["action"] == "offload"
            ],
            "pruned": [n for n, r in manifest.items() if n in changes and r["action"] == "prune"],
            "reconstructible": [n for n, r in manifest.items() if r["action"] == "reconstruct"],
        }
        data = {
            "version": 2,
            "owner": self.owner,
            "cell": self.cell,
            "used": self.used,
            "pinned": sorted(self.pinned),
            "values": values,
            "manifest": manifest,
            "missing": missing,
            "recipes": worker.recipes,
            "receipt": receipt,
            "reference_identity": {
                "top_level_aliases": "preserved",
                "cross_variable_nested_aliases": "independent values; not guaranteed",
                "allocation_policy": "streamed typed values and protocol-5 buffers; native user reducers execute trusted code",
                "stream_chunk_bytes": policy.stream_chunk_bytes,
                "serialized_operation_limit": policy.artifact_bytes,
            },
        }
        receipt["result"]["snapshot_metrics"] = metrics
        # Commit before eviction. A failed write leaves live objects intact.
        try:
            if worker.checkpoint.exists():
                previous = worker.checkpoint.read_bytes()
                try:
                    json.loads(previous)
                except ValueError:
                    previous = None
                if previous is not None:
                    atomic_write(worker.directory / "checkpoint.previous.json", previous)
            atomic_write(worker.checkpoint, json.dumps(data, allow_nan=False).encode())
        except Exception as exc:
            missing["__checkpoint__"] = str(exc)
            metrics["commit_failed"] = True
            return missing
        self.manifest = manifest
        for name, replacement in changes.items():
            if replacement is None:
                worker.values.pop(name, None)
            else:
                worker.values[name] = replacement
        # Serialization caches must not secretly retain evicted objects or old values.
        for cache in (worker.blobs.cache, worker.blobs.shadows, worker.blobs.array_shadows):
            for name in list(cache):
                if name not in candidates or name in changes:
                    cache.pop(name, None)
        receipt["result"]["kernel_state"] = {
            n: {k: v for k, v in r.items() if k != "record"} for n, r in manifest.items()
        }
        return missing
