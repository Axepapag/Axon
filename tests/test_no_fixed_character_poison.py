from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_NEURAL_AND_CORPUS_ROOTS = (
    ROOT / "runtime",
    ROOT / "training",
    ROOT / "curator",
    ROOT / "Cortext",
)

# Quarantined historical evidence: the legacy typed-reasoning module is kept
# inside training/ only as a frozen migration donor for state_dict bridging
# and transition tests.  It is not part of the active English-native
# runtime/training surface, so hygiene scans of active sources skip it.
QUARANTINED_LEGACY_MODULE_NAMES = frozenset({"legacy_typed_reasoning_d64.py"})

FORBIDDEN_ARCHITECTURE_LIMITS = (
    "max_source_chars",
    "max_input_chars",
    "max_target_chars",
    "max_output_chars",
    "max_pages",
    "active_tail_chars",
    "max_query_length",
    "max_item_chars",
    "max_item_bytes",
    "pending_cap",
    "max_consecutive_failures",
    "recall_limit",
    "recall_max_chars",
    "recall_max_item_chars",
    "inference_budget_chars",
    "max_dialects",
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
        if "archive" not in path.parts
        and "__pycache__" not in path.parts
        and path.name not in QUARANTINED_LEGACY_MODULE_NAMES
    )


def test_active_neural_and_corpus_paths_have_no_fixed_character_architecture_limits() -> None:
    violations: list[str] = []
    for path in _active_python_sources():
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_ARCHITECTURE_LIMITS + FORBIDDEN_DESTRUCTIVE_SLICES:
            if term in text:
                violations.append(f"{path.relative_to(ROOT)} contains {term!r}")
    assert not violations, "\n".join(violations)


def test_retired_reasoning_identity_marker_is_the_only_legacy_budget_name() -> None:
    term = "inference_budget_transport_units"
    occurrences: list[str] = []
    for path in _active_python_sources():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if term in line:
                occurrences.append(f"{path.relative_to(ROOT).as_posix()}:{line_number}")
    assert occurrences == []

    legacy_module = ROOT / "training" / "legacy_typed_reasoning_d64.py"
    legacy_occurrences = [
        line_number
        for line_number, line in enumerate(legacy_module.read_text(encoding="utf-8").splitlines(), 1)
        if term in line
    ]
    assert len(legacy_occurrences) == 1
