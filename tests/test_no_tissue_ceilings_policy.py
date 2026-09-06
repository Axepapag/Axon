from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from runtime.field import CanonicalStateBranch, LogicalRegion, SharedFieldSnapshot
from runtime.heart import BeatCoordinator, CoreRegistry
from runtime.source_of_truth import (
    CAPACITY_POLICY_RELATIVE_PATH,
    EXPECTED_CAPACITY_POLICY_SHA256,
    SourceOfTruthPolicyError,
    load_capacity_policy,
)
from training.complete_field_64d import ReaderConfig
from training.first_form_curriculum import _safe_target
from training.heart_preflight import scan_active_capacity_poison
from training.living_reasoning_d64 import candidate_a_config

ROOT = Path(__file__).resolve().parent.parent


def test_protected_capacity_policy_affirms_every_no_ceiling_law() -> None:
    policy = load_capacity_policy()
    assert policy.policy_sha256 == EXPECTED_CAPACITY_POLICY_SHA256
    assert all(policy.laws.values())
    assert all(control.source_preserved for control in policy.controls.values())
    assert not any(
        control.affects_tissue_identity for control in policy.controls.values()
    )


def test_capacity_policy_binding_fails_closed_after_unratified_change(
    tmp_path: Path,
) -> None:
    source = ROOT / CAPACITY_POLICY_RELATIVE_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["controls"]["heart.global_items_per_beat"]["value"] += 1
    changed = tmp_path / "capacity_policy.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SourceOfTruthPolicyError, match="hash does not match"):
        load_capacity_policy(changed)


def test_active_capacity_firewall_has_no_poison() -> None:
    result = scan_active_capacity_poison(ROOT)
    assert result["passed"], "\n".join(result["violations"])


def test_work_slices_are_not_part_of_current_d64_tissue_config() -> None:
    assert "inference_budget_chars" not in {item.name for item in fields(ReaderConfig)}
    for heads, expected in (
        (1, "living-d64-675b5ec0f0053cd54c0fbda6"),
        (2, "living-d64-40b4ad197df1810d910de15d"),
        (4, "living-d64-e0a0ad21a7bbfe1f1a18d704"),
    ):
        config = candidate_a_config(n_heads=heads, dropout=0.0)
        assert config.architecture_id == expected
        assert "inference_budget_transport_units" not in config.to_canonical_dict()
        assert "generate_gate_bias" not in config.to_canonical_dict()
        fair = candidate_a_config(n_heads=heads, dropout=0.0, generate_gate_bias=0.0)
        assert fair.architecture_id == expected


def test_recall_query_and_curriculum_target_survive_old_512_boundary(
    tmp_path: Path,
) -> None:
    query = "language " * 1_000
    branch = CanonicalStateBranch(
        tmp_path / "branch",
        branch_id="no-query-ceiling",
        authority_root=tmp_path,
    )
    coordinator = BeatCoordinator(branch, CoreRegistry(), state_root=tmp_path)
    field = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: query})
    assert coordinator._extract_recall_query(field) == query.strip()
    assert _safe_target("response " * 1_000)
