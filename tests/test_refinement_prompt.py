"""Root refinement guidance is self-contained; no external checkout is needed."""

from pathlib import Path

from threadweave.context import python_instructions


def test_refinement_guidance_is_root_only(python_config):
    prompt = python_instructions(python_config)
    child = python_instructions(python_config, child=True)
    assert "Treat continual harness refinement" in prompt
    assert "await refine.run()" in prompt
    assert "Treat continual harness refinement" not in child


def test_builtin_refine_guide_is_available_without_inlining_examples(python_config):
    guide = Path(__file__).resolve().parents[1] / "src/threadweave/builtin_skills/refine/SKILL.md"
    content = guide.read_text()
    assert "await refine.status()" in content and "await refine.run()" in content
    prompt = python_instructions(python_config)
    assert str(guide) in prompt
    for text in (
        "create a memory about always checking git status",
        "promote the error-handling pattern",
        "One request per turn is enough",
        "Planning may overlap tools",
        "25 assistant turns",
        "20 minute cooldown",
        "Create normal module artifacts first",
    ):
        assert text not in prompt
