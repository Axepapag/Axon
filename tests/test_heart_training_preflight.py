from __future__ import annotations

from pathlib import Path

from runtime.heart.translation_core import (
    HeartTranslationCore,
    HeartTranslationCoreConfig,
    heart_translation_architecture_id,
)
from runtime.trainer import (
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationPlan,
    ParameterRegistry,
    PreflightEvidenceKind,
)
from training.heart_preflight import build_heart_training_preflight, scan_active_capacity_poison
from training.heart_translation import (
    HeartTranslationTrainingObjective,
    build_heart_decoder_mechanism_curriculum,
    build_heart_translation_curriculum,
)


ROOT = Path(__file__).resolve().parent.parent


def _preflight_inputs():
    config = HeartTranslationCoreConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        ffn_dim=128,
        dropout=0.0,
        source_page_chars=64,
    )
    model = HeartTranslationCore(config)
    curriculum = build_heart_translation_curriculum()
    objective = HeartTranslationTrainingObjective()
    descriptor = ParameterModuleDescriptor(
        module_id="heart-preflight-test",
        organ_kind=OrganKind.HEART_TRANSLATION_CORE,
        generation_id="heart-base-v2",
        architecture=heart_translation_architecture_id(config),
        d_model=config.d_model,
    )
    registry = ParameterRegistry()
    registry.declare_expected((descriptor,))
    registry.register(descriptor, model)
    inventory = registry.capture_inventory(exact_value_hashes=True)
    manifest = inventory.module(descriptor.module_id)
    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id=descriptor.module_id,
        base_generation_id=descriptor.generation_id,
        candidate_generation_id="heart-candidate-v2",
        tensor_names=tuple(item.name for item in manifest.tensors if item.requires_grad),
        optimizer_name="AdamW",
        learning_rate=3e-4,
        max_steps=2,
        source_manifest_ids=(curriculum.train_manifest_id, objective.objective_id),
        holdout_manifest_ids=(curriculum.heldout_manifest_id,),
    )
    return model, curriculum, inventory, plan


def test_heart_preflight_binds_all_required_evidence_before_training(tmp_path: Path) -> None:
    model, curriculum, inventory, plan = _preflight_inputs()
    receipt = build_heart_training_preflight(
        model=model,
        curriculum=curriculum,
        inventory=inventory,
        plan=plan,
        batch_size=2,
        state_root=tmp_path,
        repo_root=ROOT,
    )

    assert receipt.passed
    assert {item.kind for item in receipt.evidence} == set(PreflightEvidenceKind)
    assert receipt.contract.complete_field_required
    assert receipt.contract.exact_source_required
    assert {item.name for item in receipt.contract.declared_bounds} == {
        "heart_source_page_chars",
        "optimizer_steps",
        "training_batch_size",
    }
    evidence_dir = tmp_path / "training" / "heart" / "preflight_evidence"
    assert len(tuple(evidence_dir.glob("*.json"))) == len(PreflightEvidenceKind)


def test_heart_preflight_accepts_separate_complete_field_decoder_mechanism_curriculum(
    tmp_path: Path,
) -> None:
    model, _semantic_curriculum, inventory, original_plan = _preflight_inputs()
    curriculum = build_heart_decoder_mechanism_curriculum()
    plan = ParameterMutationPlan(
        base_inventory_id=original_plan.base_inventory_id,
        module_id=original_plan.module_id,
        base_generation_id=original_plan.base_generation_id,
        candidate_generation_id="heart-decoder-mechanism-candidate-v1",
        tensor_names=original_plan.tensor_names,
        optimizer_name=original_plan.optimizer_name,
        learning_rate=original_plan.learning_rate,
        max_steps=original_plan.max_steps,
        source_manifest_ids=(curriculum.train_manifest_id,),
        holdout_manifest_ids=(curriculum.heldout_manifest_id,),
    )
    receipt = build_heart_training_preflight(
        model=model,
        curriculum=curriculum,
        inventory=inventory,
        plan=plan,
        batch_size=2,
        state_root=tmp_path,
        repo_root=ROOT,
    )

    assert receipt.passed
    curriculum_evidence = next(
        item for item in receipt.evidence if item.kind is PreflightEvidenceKind.CURRICULUM_DISTRIBUTION
    )
    assert "Exact-copy" in curriculum_evidence.summary


def test_semantic_capacity_scanner_detects_finite_learned_position_table(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "poisoned.py").write_text(
        "from torch import nn\n"
        "class Poison(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.position_embedding = nn.Embedding(192, 64)\n",
        encoding="utf-8",
    )

    result = scan_active_capacity_poison(tmp_path)
    assert result["passed"] is False
    assert any("finite learned position/page embedding" in item for item in result["violations"])
