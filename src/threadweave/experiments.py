"""Durable hypotheses and measured experiment runs bound to source checkpoints."""

from .benchmarks import compare, run_benchmark
from .coding import run_command
from .coding_config import BenchmarkConfig
from .gitops import GitWorkspace
from .models import new_id, now


class Experiments:
    def __init__(self, context):
        self.context, self.store = context, context.runtime.store

    def create(self, hypothesis, changes, metric=None, verifier=None):
        if not hypothesis.strip() or not verifier:
            raise ValueError("Experiments require a hypothesis and correctness verifier commands")
        identifier = new_id()
        body = {
            "hypothesis": hypothesis,
            "changes": changes,
            "metric": metric,
            "verifier": verifier,
            "source_checkpoint": GitWorkspace(self.context).snapshot_tree("experiment-input"),
            "conclusion": None,
        }
        self.store.records.insert(
            "experiments",
            {
                "id": identifier,
                "session_id": self.context.session_id,
                "created_at": now(),
                "status": "created",
                "body": body,
            },
        )
        self.store.event(
            self.context.session_id,
            "experiment_created",
            {"experiment_id": identifier, **body},
            parent=self.context.source_event,
        )
        return {"experiment_id": identifier, **body}

    def get(self, identifier):
        row = self.store.records.first("experiments", id=identifier)
        if not row or self.store.session(row["session_id"]).root_id not in self.store.history_roots(
            self.context.session_id
        ):
            raise KeyError("Unknown experiment in this trajectory")
        return {
            **dict(row),
            "body": row["body"],
            "runs": [
                r["body"]
                for r in self.store.records.select(
                    "experiment_runs",
                    experiment_id=identifier,
                    order=(("created_at", False),),
                    fields=("body",),
                )
            ],
        }

    def list(self):
        return [
            self.get(row["id"])
            for row in self.store.records.select(
                "experiments",
                session_id=self.context.session_id,
                order=(("created_at", False),),
                fields=("id",),
            )
        ]

    async def run(self, identifier, conclusion=None):
        experiment = self.get(identifier)
        if experiment["session_id"] != self.context.session_id:
            raise PermissionError("Only the owning session can execute an experiment")
        body = experiment["body"]
        self.store.records.update("experiments", {"status": "running"}, id=identifier)
        run_id = new_id()
        self.store.event(
            self.context.session_id,
            "experiment_run_started",
            {"experiment_id": identifier, "run_id": run_id},
            parent=self.context.source_event,
        )
        try:
            results = [
                await run_command(self.context, cmd, kind="experiment_correctness")
                for cmd in body["verifier"]
            ]
            correct = all(r["passed"] for r in results)
            metric = (
                await run_benchmark(
                    self.context,
                    BenchmarkConfig.model_validate(body["metric"]),
                    correctness_passed=correct,
                )
                if body["metric"] and correct
                else None
            )
            patch = GitWorkspace(self.context).diff(body["source_checkpoint"])
            result = {
                "id": run_id,
                "correctness": results,
                "metrics": metric,
                "passed": correct and (metric is None or metric["passed"]),
                "patch_artifact": self.context.runtime.artifacts.put_bytes(
                    self.context.session_id, patch.encode(), "text/x-diff"
                ),
                "conclusion": conclusion,
                "conclusion_source": "agent" if conclusion else None,
            }
            self.store.records.insert(
                "experiment_runs",
                {"id": run_id, "experiment_id": identifier, "created_at": now(), "body": result},
            )
            self.store.records.update("experiments", {"status": "concluded"}, id=identifier)
            self.store.event(
                self.context.session_id,
                "experiment_conclusion",
                {"experiment_id": identifier, **result},
                parent=self.context.source_event,
            )
            return result
        except BaseException:
            self.store.records.update("experiments", {"status": "interrupted"}, id=identifier)
            raise

    def compare(self, first, second):
        a, b = self.get(first), self.get(second)
        if not a["runs"] or not b["runs"]:
            raise ValueError("Both experiments must have measured runs")
        if a["body"]["metric"] != b["body"]["metric"]:
            raise ValueError(
                "Metric configurations differ; these measurements are not directly comparable"
            )
        config = BenchmarkConfig.model_validate(a["body"]["metric"])
        if not a["runs"][-1]["passed"] or not b["runs"][-1]["passed"]:
            raise ValueError("Both experiment correctness gates must pass before comparison")
        return compare(a["runs"][-1]["metrics"], b["runs"][-1]["metrics"], config)
