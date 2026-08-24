from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_NEURAL_AND_CORPUS_ROOTS = (
    ROOT / "runtime",
    ROOT / "training",
    ROOT / "curator",
    ROOT / "Cortext",
)

FORBIDDEN_ARCHITECTURE_LIMITS = (
    "max_source_chars",
    "max_input_chars",
    "max_target_chars",
    "max_output_chars",
    "max_pages",
    "active_tail_chars",
)
FORBIDDEN_DESTRUCTIVE_SLICES = (
    "text = text[:",
    "content = content[:",
    "summary = summary[:",
)


def _active_python_sources() -> tuple[Path, ...]:
    return tuple(
        path
        for root in ACTIVE_NEURAL_AND_CORPUS_ROOTS
        for path in root.rglob("*.py")
        if "archive" not in path.parts and "__pycache__" not in path.parts
    )


def test_active_neural_and_corpus_paths_have_no_fixed_character_architecture_limits() -> None:
    violations: list[str] = []
    for path in _active_python_sources():
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_ARCHITECTURE_LIMITS + FORBIDDEN_DESTRUCTIVE_SLICES:
            if term in text:
                violations.append(f"{path.relative_to(ROOT)} contains {term!r}")
    assert not violations, "\n".join(violations)
