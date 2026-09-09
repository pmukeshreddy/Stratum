import pytest

from threadweave.kernel import Kernel
from threadweave.models import HarnessError, new_id


async def bridge(name, args):
    assert name == "sample"
    return list(range(args["count"]))


async def test_repl_state_large_values_tool_bridge_and_structured_output(tmp_path):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        first = await kernel.execute(
            new_id(),
            "rows = await tools.acall('sample', count=10000)\nprint('hello')\nlen(rows)",
            5,
        )
        assert first["value"] == "10000" and first["stdout"] == "hello\n"
        second = await kernel.execute(
            new_id(),
            "import os, sys\nos.write(1, b'native\\n')\nprint('warning', file=sys.stderr)\nsum(rows)",
            5,
        )
        assert second["value"] == "49995000"
        assert "native" in second["stdout"] and "warning" in second["stderr"]
        error = await kernel.execute(new_id(), "1 / 0", 5)
        assert error["error"]["code"] == "ZeroDivisionError"
        assert error["error"]["traceback"]
        assert (await kernel.execute(new_id(), "len(rows)", 5))["value"] == "10000"
    finally:
        await kernel.close()


async def test_checkpoint_restores_values_and_recipes_without_replaying_side_effects(tmp_path):
    directory = tmp_path / "kernel"
    kernel = Kernel(directory, tmp_path, bridge)
    try:
        result = await kernel.execute(
            new_id(),
            """
from pathlib import Path
marker = workspace / 'executions.txt'
marker.write_text(marker.read_text() + 'x' if marker.exists() else 'x')
answer = {'items': [1, 2, 3], 'tuple': (4, 5), 'bytes': b'abc'}
double = lambda x: x * 2
remember_recipe('double', 'double = lambda x: x * 2')
lost = iter([1, 2, 3])
""",
            5,
        )
        assert "lost" in result["not_checkpointed"]
    finally:
        await kernel.close()
    restored = Kernel(directory, tmp_path, bridge)
    try:
        await restored.start()
        assert "answer" in restored.recovery["restored"]
        assert "double" in restored.recovery["reconstructed"]
        assert "lost" in restored.recovery["missing"]
        result = await restored.execute(new_id(), "double(sum(answer['items']))", 5)
        assert result["value"] == "12"
        assert (tmp_path / "executions.txt").read_text() == "x"
        await restored.execute(new_id(), "forget('answer', 'double', 'lost')", 5)
    finally:
        await restored.close()
    again = Kernel(directory, tmp_path, bridge)
    try:
        await again.start()
        assert "answer" not in again.recovery["restored"]
        assert "double" not in again.recovery["reconstructed"]
    finally:
        await again.close()


async def test_timeout_interrupts_execution_preserving_partial_namespace(tmp_path):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        await kernel.execute(new_id(), "value = 42", 5)
        pid = kernel.process.pid
        with pytest.raises(HarnessError) as caught:
            await kernel.execute(new_id(), "value = 99\nwhile True: pass", 0.1)
        assert caught.value.failure.code == "python_timeout"
        assert caught.value.failure.uncertain
        assert kernel.process.pid == pid
        assert (await kernel.execute(new_id(), "value", 5))["value"] == "99"
    finally:
        await kernel.close()


async def test_execution_receipt_prevents_duplicate_side_effect(tmp_path):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    execution_id = new_id()
    code = "path = workspace / 'once.txt'\npath.write_text(path.read_text() + 'x' if path.exists() else 'x')"
    try:
        await kernel.execute(execution_id, code, 5)
        assert kernel.receipt(execution_id)
        await kernel.execute(execution_id, code, 5)
        assert (tmp_path / "once.txt").read_text() == "x"
    finally:
        await kernel.close()


async def test_broken_checkpoint_reports_warning_and_continues(tmp_path):
    directory = tmp_path / "kernel"
    directory.mkdir()
    (directory / "checkpoint.json").write_text("broken JSON")
    kernel = Kernel(directory, tmp_path, bridge)
    try:
        await kernel.start()
        assert "__checkpoint__" in kernel.recovery["missing"]
        result = await kernel.execute("recovered", "value = 42\nvalue", 10)
        assert result["error"] is None
        assert not result["snapshot_metrics"].get("commit_failed")
    finally:
        await kernel.close()


async def test_workspace_module_import_recovery_captures_native_startup_output(tmp_path):
    (tmp_path / "local_module.py").write_text(
        "import os\nprint('importing module')\nos.write(1, b'native import output\\n')\nanswer = 42\n"
    )
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        assert (await kernel.execute(new_id(), "import local_module\nlocal_module.answer", 5))[
            "value"
        ] == "42"
    finally:
        await kernel.close()
    restored = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        result = await restored.execute(new_id(), "local_module.answer", 5)
        assert result["value"] == "42"
        assert "native import output" in (tmp_path / "kernel/recovery.log").read_text()
    finally:
        await restored.close()
