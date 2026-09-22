from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_ROOTS = (ROOT / "runtime", ROOT / "training", ROOT / "Cortext")
LEGACY_PREFIXES = (
    "training.legacy_typed_reasoning_d64",
    "training.legacy_typed_reasoning_curriculum",
    ".legacy_typed_reasoning_d64",
    ".legacy_typed_reasoning_curriculum",
)


def test_active_runtime_and_trainer_do_not_import_retired_typed_reasoning() -> None:
    violations: list[str] = []
    for root in ACTIVE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "archive" in path.parts or path.name.startswith("legacy_typed_reasoning"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("training.legacy_typed_reasoning"):
                            violations.append(f"{path.relative_to(ROOT)} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    rendered = ("." * node.level) + module
                    if any(rendered.startswith(prefix) for prefix in LEGACY_PREFIXES):
                        violations.append(f"{path.relative_to(ROOT)} imports {rendered}")
    assert not violations, "\n".join(violations)
