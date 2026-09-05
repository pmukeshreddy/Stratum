"""Adapter safety checks; these are not benchmark/model results."""

import json
import sys

import pytest

from threadweave.manyih_evaluation import evaluate_manyih
from threadweave.models import RunConfig


async def test_missing_official_artifacts_fail_before_model_invocation(tmp_path):
    with pytest.raises(ValueError, match="official ManyIH checkout"):
        await evaluate_manyih(tmp_path / "absent", sys.executable, RunConfig(), tmp_path / "out")


async def test_existing_trajectory_config_is_never_overwritten(tmp_path, monkeypatch):
    source = tmp_path / "metadata-fixture"
    data = source / "manyih/data"
    data.mkdir(parents=True)
    # Never formatted, executed or graded: only exercises the pre-inference guard.
    (data / "coding.json").write_text(json.dumps({"data": [{"prompt": "unused"}]}))
    output = tmp_path / "existing"
    output.mkdir()
    record = output / "run_config.json"
    record.write_text('{"original": true}')

    async def resolved(config):
        return config

    monkeypatch.setattr("threadweave.manyih_evaluation.resolved_config", resolved)
    with pytest.raises(ValueError, match="preserve prior trajectories"):
        await evaluate_manyih(source, sys.executable, RunConfig(), output)
    assert record.read_text() == '{"original": true}'
