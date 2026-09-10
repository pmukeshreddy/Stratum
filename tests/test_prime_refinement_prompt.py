"""Root guidance is checked against the supplied Prime source, not firing counts."""

from pathlib import Path

from threadweave.context import python_instructions

PRIME = Path(__file__).resolve().parents[2] / "prime-agent-main/packages/coding-agent"


def test_root_refine_paragraph_is_prime_verbatim(python_config):
    source = (PRIME / "src/core/prompts/rlm.ts").read_text()
    prefix = '"Treat continual harness refinement'
    paragraph = source[source.index(prefix) + 1 :].split('",', 1)[0]
    prompt = python_instructions(python_config)
    assert paragraph in prompt
    assert paragraph not in python_instructions(python_config, child=True)


def test_skill_guide_is_prime_verbatim_and_not_inlined(python_config):
    guide = Path(__file__).resolve().parents[1] / "src/threadweave/builtin_skills/refine/SKILL.md"
    assert guide.read_text() == (PRIME / "skills/refine/SKILL.md").read_text()
    prompt = python_instructions(python_config)
    assert str(guide) in prompt
    for text in (
        "create a memory about always checking git status",
        "promote the error-handling pattern",
        "One request per turn is enough",
        "Planning may overlap tools",
        "25 assistant turns",
        "20 minute cooldown",
        "ManyIH",
        "Create normal module artifacts first",
    ):
        assert text not in prompt
