import sys

import pytest

from threadweave.gitops import git
from threadweave.models import RunConfig


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mathops.py").write_text(
        "def add(a, b):\n    return a - b\n\ndef twice(value):\n    return add(value, value)\n"
    )
    (root / "tests").mkdir()
    (root / "tests/test_mathops.py").write_text(
        "from mathops import add, twice\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_twice():\n    assert twice(3) == 6\n"
    )
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@localhost",
        "commit",
        "-qm",
        "Real failing input",
    )
    return root


@pytest.fixture
def coding_config():
    return RunConfig(
        provider={"name": "test", "model": "test-only", "max_output_tokens": 256},
        task={
            "adapter": "coding",
            "test_commands": [[sys.executable, "-m", "pytest", "-q"]],
            "protect_tests": True,
        },
        permissions=["workspace.read", "workspace.write", "python", "process", "agents", "state"],
        context={"max_tokens": 96000},
        limits={"token_budget": 3_000_000, "wall_seconds": 90, "max_turns": 60},
        features={"model_compaction": False},
        retry={"initial_delay": 0, "max_delay": 0},
    )
