"""Example: a JSON measurement benchmark using the ordinary core session loop."""

import json

from threadweave.models import Record, Verification
from threadweave.tools import Tool


class MeasurementArgs(Record):
    path: str = "measurement.json"


async def read_measurement(context, arguments):
    path = context.path(arguments.path)
    if path.stat().st_size > 1000000:
        raise ValueError("Measurement exceeds 1 MB")
    return json.loads(path.read_text())


class MeasurementTask:
    async def prepare(self, context, task):
        return {"specification": task.specification, "success_metrics": task.success_metrics}

    async def verify(self, context, task):
        path = context.path(task.verifier_options.get("path", "measurement.json"))
        if not path.exists():
            return Verification(passed=False, details="No measurement has been produced")
        measurement = await read_measurement(context, MeasurementArgs(path=str(path)))
        error = measurement["error"]
        return Verification(
            passed=error <= task.verifier_options["maximum_error"],
            details=measurement,
            metrics={"error": error},
        )


def install(runtime):
    runtime.adapters["measurement"] = MeasurementTask()
    runtime.tools.register(
        Tool(
            "measurement_read",
            "Read a bounded JSON measurement artifact.",
            MeasurementArgs,
            read_measurement,
            ("workspace.read",),
        )
    )
