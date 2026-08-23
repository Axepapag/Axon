from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_authority_mirrors_are_byte_identical() -> None:
    assert (ROOT / "SOURCE_OF_TRUTH.md").read_bytes() == (
        ROOT / "docs" / "SOURCE_OF_TRUTH.md"
    ).read_bytes()
    assert (ROOT / "WORKING_CONTRACT.md").read_bytes() == (
        ROOT / "docs" / "WORKING_CONTRACT.md"
    ).read_bytes()


def test_day_zero_active_python_surface_is_narrow() -> None:
    runtime_dirs = {
        path.name
        for path in (ROOT / "runtime").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert runtime_dirs == {"axon_runtime", "field", "dormant", "heart", "trainer"}

    field_files = {path.name for path in (ROOT / "runtime" / "field").glob("*.py")}
    assert field_files == {
        "__init__.py",
        "schema.py",
        "delta.py",
        "compiler_d64.py",
        "semantic_d64.py",
        "state_branch.py",
    }

    runtime_files = {
        path.name for path in (ROOT / "runtime" / "axon_runtime").glob("*.py")
    }
    assert runtime_files == {"__init__.py", "d64_adapter.py"}

    dormant_files = {path.name for path in (ROOT / "runtime" / "dormant").glob("*.py")}
    assert dormant_files == {
        "__init__.py",
        "evidence_bridge.py",
        "evaluation.py",
        "generations.py",
        "incremental.py",
        "relevance.py",
    }

    cortext_files = {path.name for path in (ROOT / "Cortext").glob("*.py")}
    assert cortext_files == {"__init__.py", "contracts.py", "evaluation.py"}

    heart_files = {path.name for path in (ROOT / "runtime" / "heart").glob("*.py")}
    assert heart_files == {
        "__init__.py",
        "errors.py",
        "authority.py",
        "registry.py",
        "tick.py",
        "board.py",
        "transaction.py",
        "ingress_queue.py",
        "coordinator.py",
        "durable_ingress.py",
        "health.py",
        "host.py",
        "identity.py",
        "lease.py",
        "valve.py",
    }

    trainer_files = {path.name for path in (ROOT / "runtime" / "trainer").glob("*.py")}
    assert trainer_files == {
        "__init__.py",
        "contracts.py",
        "registry.py",
        "authority.py",
        "telemetry.py",
        "store.py",
        "host.py",
    }

    training_files = {path.name for path in (ROOT / "training").glob("*.py")}
    assert training_files == {
        "__init__.py",
        "canonical_d64.py",
        "complete_field_64d.py",
        "train_complete_field_64d.py",
    }


def test_parallel_pre_day_zero_bodies_are_not_live() -> None:
    forbidden_paths = (
        "cores",
        "ops",
        "kaggle",
        "review_only",
        "codex-turn-state.md",
        "TRAIN_COMPLETE_FIELD_64D_R0.bat",
        "table.bat",
        "runtime/bus",
        "runtime/table",
        "runtime/council",
        "runtime/tick_loop.py",
        "runtime/core_proposer.py",
        "runtime/multi_tick_refiner.py",
        "runtime/field/view.py",
        "runtime/field/schedule.py",
        "runtime/field/charslot.py",
        "runtime/field/schema_v2.py",
        "runtime/field/serde_v2.py",
        "runtime/field/view_v2.py",
    )
    for relative in forbidden_paths:
        assert not (ROOT / relative).exists(), relative

    assert (ROOT / "archive" / "day_zero_legacy_2026-08-20" / "README.md").is_file()


def test_active_d64_code_cannot_import_archived_anatomy() -> None:
    forbidden_fragments = (
        "runtime.council",
        "runtime.tick_loop",
        "runtime.core_proposer",
        "runtime.multi_tick_refiner",
        "runtime.field.view",
        "runtime.field.schedule",
        "runtime.field.charslot",
        "runtime.field.schema_v2",
        "runtime.field.serde_v2",
        "runtime.field.view_v2",
        "runtime.axon_runtime.exact_driver",
        "runtime.axon_runtime.bootstrap",
        "cores.core",
        "cores.soul_v2",
        "legacy-record-direct",
    )
    roots = (
        ROOT / "runtime" / "field",
        ROOT / "runtime" / "axon_runtime",
        ROOT / "runtime" / "dormant",
        ROOT / "runtime" / "heart",
        ROOT / "training",
    )
    for source_root in roots:
        for path in source_root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for fragment in forbidden_fragments:
                assert fragment not in text, f"{path.relative_to(ROOT)} contains {fragment}"


def test_council_state_is_not_live() -> None:
    assert not (ROOT / "State" / "active" / "council_field.json").exists()
    assert not (ROOT / "State" / "active" / "council_field.test.json").exists()
    assert not (ROOT / "State" / "dormant" / "council_field_tails.jsonl").exists()
    assert not (ROOT / "State" / "souls" / "council").exists()
    assert not (ROOT / "State" / "souls" / "council_test").exists()


def test_active_training_has_no_detached_escape_hatch() -> None:
    model_text = (ROOT / "training" / "complete_field_64d.py").read_text(encoding="utf-8")
    trainer_text = (ROOT / "training" / "train_complete_field_64d.py").read_text(encoding="utf-8")
    for fragment in (
        "class CompleteFieldPager",
        "def read_field_with_memory",
        "def forward_transaction",
        "def run_transaction",
    ):
        assert fragment not in model_text
    for fragment in ("--legacy-record-direct", "--canonical-d64", "legacy-record-direct"):
        assert fragment not in trainer_text
