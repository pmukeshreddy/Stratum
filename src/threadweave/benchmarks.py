"""Measured performance only: correctness, warmups, repetitions, distribution and comparison."""

import math
import re
import statistics

from .models import new_id, now


def summarize(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Measurements must be nonempty finite numbers")
    ordered = sorted(values)
    return {
        "median": statistics.median(values),
        "p50": statistics.median(values),
        "p95": ordered[max(0, math.ceil(0.95 * len(values)) - 1)],
        "p99": ordered[max(0, math.ceil(0.99 * len(values)) - 1)],
        "min": min(values),
        "max": max(values),
        "count": len(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "variance": statistics.variance(values) if len(values) > 1 else 0,
        "percentile_method": "nearest rank; median uses midpoint",
    }


def compare(reference, candidate, config):
    if (
        not reference
        or not candidate
        or reference.get("correct") is False
        or candidate.get("correct") is False
    ):
        raise ValueError("Performance comparison requires correctness-passing measurements")
    base, value = reference["median"], candidate["median"]
    if base == 0:
        raise ValueError("Relative improvement needs a nonzero baseline")
    improvement = (
        (base - value) / abs(base)
        if config.direction == "lower_is_better"
        else (value - base) / abs(base)
    )
    return {
        "improvement": improvement,
        "required_improvement": config.required_improvement,
        "noise_tolerance": config.noise_tolerance,
        "passed": improvement + config.noise_tolerance >= config.required_improvement,
    }


async def run_benchmark(
    context, config, *, reference=None, label="candidate", correctness_passed=None
):
    from .coding import run_command

    checks = [
        await run_command(
            context, c, kind="benchmark_correctness", timeout_seconds=config.timeout_seconds
        )
        for c in config.correctness_commands
    ]
    correct = all(r["passed"] for r in checks) and correctness_passed is not False
    if not checks and correctness_passed is None:
        raise ValueError(
            "Benchmark requires correctness_commands or an independently supplied correctness result"
        )
    result = {
        "label": label,
        "config": config.model_dump(),
        "correctness": checks,
        "correct": correct,
        "measurements": [],
        "runs": [],
        "passed": False,
    }
    from .machine import metadata

    result["environment"] = metadata()
    if correct:
        pattern = re.compile(config.metric_regex)
        if pattern.groups != 1:
            raise ValueError("metric_regex needs exactly one numeric capture group")
        for i in range(config.warmups + config.repetitions):
            run = await run_command(
                context, config.command, kind="benchmark", timeout_seconds=config.timeout_seconds
            )
            run["warmup"] = i < config.warmups
            result["runs"].append(run)
            if not run["passed"]:
                result["error"] = "Benchmark command failed"
                break
            raw = context.runtime.artifacts.load(context.session_id, run["stdout_artifact"])
            matches = pattern.findall(raw)
            if len(matches) != 1:
                result["error"] = "metric_regex must match exactly one measurement per run"
                break
            metric = float(matches[0][0] if isinstance(matches[0], tuple) else matches[0])
            if not math.isfinite(metric):
                raise ValueError("Benchmark emitted a non-finite metric")
            if not run["warmup"]:
                result["measurements"].append(metric)
        if len(result["measurements"]) == config.repetitions:
            selected = result["measurements"]
            result["excluded_measurements"] = []
            if config.outlier_policy == "iqr" and len(selected) >= 4:
                q1, _, q3 = statistics.quantiles(selected, n=4, method="inclusive")
                lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
                result["excluded_measurements"] = [x for x in selected if not lower <= x <= upper]
                selected = [x for x in selected if lower <= x <= upper]
            result.update(summarize(selected))
            result["passed"] = True
            if reference:
                if not reference.get("passed") or "median" not in reference:
                    result.update(passed=False, error="No valid baseline measurement")
                else:
                    result["comparison"] = compare(reference, result, config)
                    result["passed"] = result["comparison"]["passed"]
    aid = context.runtime.artifacts.put(
        context.session_id, result, source_event=context.source_event
    )
    result["raw_artifact"] = aid
    identifier = new_id()
    context.runtime.store.records.insert(
        "benchmark_measurements",
        {"id": identifier, "session_id": context.session_id, "created_at": now(), "body": result},
    )
    context.runtime.store.event(
        context.session_id,
        "benchmark_result",
        {"measurement_id": identifier, **result},
        parent=context.source_event,
    )
    return result
