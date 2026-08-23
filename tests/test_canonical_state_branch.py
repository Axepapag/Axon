from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.field import (
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    BranchAuthorityError,
    BranchIntegrityError,
    CanonicalStateBranch,
    D64FieldCompiler,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
    StaleDeltaError,
    replacement_delta,
)


def test_branch_initializes_loads_and_commits_canonical_delta(tmp_path: Path) -> None:
    snapshot = SharedFieldSnapshot.from_texts(
        {"user_input": "hello", "scratch": "old"}, tick_id=4
    )
    branch = CanonicalStateBranch(
        tmp_path / "branch",
        branch_id="test-branch",
        authority_root=tmp_path,
    )
    head = branch.initialize(snapshot)
    assert head.generation == 0
    assert branch.load_head().field_id == snapshot.field_id

    compiled = D64FieldCompiler().compile(snapshot)
    delta = replacement_delta(
        snapshot,
        compiled,
        region="scratch",
        text="new",
        author_core_id="core64",
        pass_id=1,
    )
    successor = branch.commit(delta)
    assert successor.region("scratch").text == "new"
    assert successor.parent_field_id == snapshot.field_id
    assert branch.load_head().field_id == successor.field_id
    assert branch.load_head_record().generation == 1
    assert (branch.snapshots_dir / f"{snapshot.field_id}.json").exists()
    assert (branch.snapshots_dir / f"{successor.field_id}.json").exists()
    assert (branch.deltas_dir / f"{delta.delta_id}.json").exists()
    events = [json.loads(line) for line in branch.journal_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["initialize", "commit"]


def test_legacy_v1_snapshot_preserves_historical_cortex_name_and_hash(tmp_path: Path) -> None:
    snapshot = SharedFieldSnapshot(
        tick_id=2,
        regions=(RegionState.from_text(LogicalRegion.CORTEX, "legacy exact memory"),),
        schema_version=LEGACY_SCHEMA_VERSION,
    )
    historical_id = snapshot.field_id
    branch = CanonicalStateBranch(
        tmp_path / "legacy",
        branch_id="legacy",
        authority_root=tmp_path,
    )
    branch.initialize(snapshot)

    serialized = json.loads(
        (branch.snapshots_dir / f"{historical_id}.json").read_text(encoding="utf-8")
    )
    assert serialized["schema"] == LEGACY_SCHEMA_VERSION
    assert serialized["regions"][2]["name"] == "structured_knowledge"

    reloaded = branch.load_head()
    assert reloaded.schema_version == LEGACY_SCHEMA_VERSION
    assert reloaded.field_id == historical_id
    assert reloaded.region(LogicalRegion.CORTEX).text == "legacy exact memory"


def test_branch_migrates_v1_head_to_v2_cortex_without_rewriting_history(tmp_path: Path) -> None:
    legacy = SharedFieldSnapshot(
        tick_id=7,
        regions=(RegionState.from_text(LogicalRegion.CORTEX, "exact remembered text"),),
        source_manifest_ids=("legacy-manifest",),
        schema_version=LEGACY_SCHEMA_VERSION,
    )
    branch = CanonicalStateBranch(
        tmp_path / "migrate",
        branch_id="migrate",
        authority_root=tmp_path,
    )
    branch.initialize(legacy)
    legacy_path = branch.snapshots_dir / f"{legacy.field_id}.json"
    legacy_bytes = legacy_path.read_bytes()

    migrated = branch.migrate_to_current_schema()

    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.tick_id == legacy.tick_id + 1
    assert migrated.parent_field_id == legacy.field_id
    assert migrated.source_manifest_ids == legacy.source_manifest_ids
    assert migrated.region(LogicalRegion.CORTEX).text == legacy.region(LogicalRegion.CORTEX).text
    assert legacy_path.read_bytes() == legacy_bytes
    assert branch.load_head_record().generation == 1

    serialized = json.loads(
        (branch.snapshots_dir / f"{migrated.field_id}.json").read_text(encoding="utf-8")
    )
    assert serialized["schema"] == SCHEMA_VERSION
    assert serialized["regions"][2]["name"] == "cortex"

    events = [json.loads(line) for line in branch.journal_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["initialize", "schema_migration"]
    migration = events[-1]
    assert migration["from_schema"] == LEGACY_SCHEMA_VERSION
    assert migration["to_schema"] == SCHEMA_VERSION
    assert migration["region_rename"] == {"structured_knowledge": "cortex"}

    compiled = D64FieldCompiler().compile(migrated)
    compiled.verify_roundtrip(migrated)
    assert compiled.region_text(LogicalRegion.CORTEX) == "exact remembered text"


def test_branch_rejects_stale_commit(tmp_path: Path) -> None:
    snapshot = SharedFieldSnapshot.from_texts({"scratch": "one"})
    branch = CanonicalStateBranch(
        tmp_path / "branch",
        branch_id="stale",
        authority_root=tmp_path,
    )
    branch.initialize(snapshot)
    compiled = D64FieldCompiler().compile(snapshot)
    first = replacement_delta(
        snapshot,
        compiled,
        region="scratch",
        text="two",
        author_core_id="core64",
        pass_id=1,
    )
    branch.commit(first)
    stale = replacement_delta(
        snapshot,
        compiled,
        region="scratch",
        text="three",
        author_core_id="core64",
        pass_id=2,
    )
    with pytest.raises(StaleDeltaError):
        branch.commit(stale)


def test_branch_load_detects_snapshot_hash_tampering(tmp_path: Path) -> None:
    snapshot = SharedFieldSnapshot.from_texts({"user_input": "hello"})
    branch = CanonicalStateBranch(
        tmp_path / "branch",
        branch_id="tamper",
        authority_root=tmp_path,
    )
    branch.initialize(snapshot)
    path = branch.snapshots_dir / f"{snapshot.field_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["regions"][1]["spans"][0]["text"] = "changed"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(BranchIntegrityError, match="hash mismatch"):
        branch.load_head()


def test_training_constructor_stays_under_state_training_branches(tmp_path: Path) -> None:
    branch = CanonicalStateBranch.training("r0", state_root=tmp_path / "State")
    expected = (tmp_path / "State" / "training" / "branches" / "r0").resolve()
    assert branch.root == expected


def test_branch_authority_rejects_outside_root(tmp_path: Path) -> None:
    authority = tmp_path / "allowed"
    with pytest.raises(BranchAuthorityError):
        CanonicalStateBranch(
            tmp_path / "outside",
            branch_id="bad",
            authority_root=authority,
        )
