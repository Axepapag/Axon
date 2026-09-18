"""Base identity must be a property of the lineage, not of the local BLAS build.

``registry.parameter_value_sha256`` hashes raw tensor bytes, so
``ParameterInventory.inventory_id`` -- and through it
``ParameterMutationPlanV2.plan_id`` -- was a function of the machine's BLAS and
reduction order.  The measured consequence: the step-600 v6 candidate trained on
Kaggle could not be resumed on the local GPU at all, because 15 of the 110 base
records (14 parameters plus ``bank16``) differ by roughly one ULP between the
two environments and ``TrainerExecutionError: checkpoint plan lineage
mismatch`` fired before a single step.  The measured split is 15 of the 95
recorded base tensors (14 parameters plus ``bank16``).

``substrate/substrate.py`` already ratified the correct treatment for exactly
this hazard: materialize the computed artifact once, then verify later rebuilds
with ``np.allclose(..., atol=1e-5, rtol=0.0)``.  These tests pin that treatment
for the base module, and -- just as importantly -- pin the two ways it must
still refuse: a real initialization edit, and a resumed lineage whose recorded
base cannot be reproduced here at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from runtime.trainer.base_artifact import (
    BASE_ARTIFACT_ATOL,
    BASE_ARTIFACT_DIRNAME,
    BaseArtifactError,
    base_artifact_path,
    resolve_base_module,
)
from runtime.trainer.contracts import OrganKind, ParameterInventory, ParameterModuleDescriptor
from runtime.trainer.registry import capture_module_manifest

MODULE_ID = "reasoning-d64-base-artifact-probe"
GENERATION_ID = f"{MODULE_ID}-untrained-base-v1"
CANDIDATE_GENERATION_ID = "r64v3-probe"
ARCHITECTURE = "living-d64-probe"


class _ProbeCore(nn.Module):
    """A core-shaped module: one initialized parameter plus one derived buffer."""

    def __init__(self, *, seed: int = 7) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.decoder_init = nn.Parameter(torch.randn(3, 4, generator=generator) * 0.1)
        self.register_buffer("bank16", torch.randn(4, 5, generator=generator) * 0.25)

    def forward(self, value: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unused
        return value


def _descriptor(*, architecture: str = ARCHITECTURE) -> ParameterModuleDescriptor:
    return ParameterModuleDescriptor(
        module_id=MODULE_ID,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=GENERATION_ID,
        architecture=architecture,
        d_model=4,
    )


def _inventory_id(module: nn.Module, descriptor: ParameterModuleDescriptor) -> str:
    manifest = capture_module_manifest(descriptor, module, exact_value_hashes=True)
    inventory = ParameterInventory(
        manifests=(manifest,),
        expected_module_ids=(descriptor.module_id,),
        missing_module_ids=(),
    )
    return inventory.inventory_id


def _one_ulp(tensor: torch.Tensor) -> torch.Tensor:
    return tensor + torch.where(
        tensor == 0, torch.full_like(tensor, torch.finfo(torch.float32).tiny), tensor.abs()
    ) * torch.finfo(torch.float32).eps


def _drift_derived_buffer(module: _ProbeCore) -> None:
    """Perturb only the derived buffer, the way a different BLAS build does."""

    with torch.no_grad():
        module.bank16.copy_(_one_ulp(module.bank16))


def _record_lineage(state_root: Path, module: nn.Module) -> str:
    """Write the recorded plan/inventory/checkpoint chain a resume would read."""

    descriptor = _descriptor()
    manifest = capture_module_manifest(descriptor, module, exact_value_hashes=True)
    inventory = ParameterInventory(
        manifests=(manifest,),
        expected_module_ids=(MODULE_ID,),
        missing_module_ids=(),
    )
    trainer_root = state_root / "training" / "trainer"
    (trainer_root / "inventories").mkdir(parents=True, exist_ok=True)
    (trainer_root / "plans").mkdir(parents=True, exist_ok=True)
    candidate_root = trainer_root / "candidates" / MODULE_ID / CANDIDATE_GENERATION_ID
    candidate_root.mkdir(parents=True, exist_ok=True)
    (trainer_root / "inventories" / f"{inventory.inventory_id}.json").write_text(
        json.dumps(inventory.to_canonical_dict()), encoding="utf-8"
    )
    plan_id = "plan-probe"
    (trainer_root / "plans" / f"{plan_id}.json").write_text(
        json.dumps({"plan_id": plan_id, "base_inventory_id": inventory.inventory_id}),
        encoding="utf-8",
    )
    (candidate_root / "latest_checkpoint.json").write_text(
        json.dumps({"plan_id": plan_id, "step": 600}), encoding="utf-8"
    )
    return inventory.inventory_id


def test_a_fresh_generation_persists_its_exact_bytes_and_replays_them(tmp_path: Path) -> None:
    module = _ProbeCore()
    before = _inventory_id(module, _descriptor())

    created = resolve_base_module(
        module,
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )

    assert created.action == "created"
    assert created.artifact_path.is_file()
    assert created.artifact_path.parent.name == BASE_ARTIFACT_DIRNAME
    assert created.reconciled_records == ()

    rebuilt = _ProbeCore()
    assert _inventory_id(rebuilt, _descriptor()) == before
    adopted = resolve_base_module(
        rebuilt,
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )
    assert adopted.action == "adopted"
    assert adopted.reconciled_records == ()
    assert _inventory_id(rebuilt, _descriptor()) == before


def test_derived_float_drift_is_reconciled_and_base_identity_becomes_stable(tmp_path: Path) -> None:
    reference = _ProbeCore()
    resolve_base_module(
        reference,
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )
    recorded_id = _inventory_id(reference, _descriptor())

    drifted = _ProbeCore()
    _drift_derived_buffer(drifted)
    # The trap this repair closes: without the artifact the drifted rebuild is a
    # different base identity, which is exactly what locked the v6 lineage to
    # the machine that built it.
    assert _inventory_id(drifted, _descriptor()) != recorded_id

    disposition = resolve_base_module(
        drifted,
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )

    assert disposition.action == "adopted"
    assert disposition.reconciled_records == ("bank16",)
    assert 0.0 < disposition.max_abs_delta < BASE_ARTIFACT_ATOL
    assert torch.equal(drifted.bank16, reference.bank16)
    assert _inventory_id(drifted, _descriptor()) == recorded_id


def test_a_real_initialization_change_fails_closed(tmp_path: Path) -> None:
    reference = _ProbeCore()
    resolve_base_module(
        reference,
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )

    edited = _ProbeCore()
    with torch.no_grad():
        edited.decoder_init.add_(1e-3)

    with pytest.raises(BaseArtifactError) as error:
        resolve_base_module(
            edited,
            state_root=tmp_path,
            descriptor=_descriptor(),
            candidate_generation_id=CANDIDATE_GENERATION_ID,
        )

    assert "beyond doctrine tolerance" in str(error.value)
    assert "decoder_init" in str(error.value)
    assert "re-versioned" in str(error.value)


def test_a_changed_architecture_cannot_adopt_another_bases_bytes(tmp_path: Path) -> None:
    resolve_base_module(
        _ProbeCore(),
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )

    with pytest.raises(BaseArtifactError) as error:
        resolve_base_module(
            _ProbeCore(),
            state_root=tmp_path,
            descriptor=_descriptor(architecture="living-d64-something-else"),
            candidate_generation_id=CANDIDATE_GENERATION_ID,
        )

    assert "different base module" in str(error.value)


def test_a_resumed_lineage_refuses_a_drifted_rebuild_instead_of_an_opaque_mismatch(
    tmp_path: Path,
) -> None:
    recorded_id = _record_lineage(tmp_path, _ProbeCore())

    drifted = _ProbeCore()
    _drift_derived_buffer(drifted)
    with pytest.raises(BaseArtifactError) as error:
        resolve_base_module(
            drifted,
            state_root=tmp_path,
            descriptor=_descriptor(),
            candidate_generation_id=CANDIDATE_GENERATION_ID,
        )

    message = str(error.value)
    assert "not bit-identical" in message
    assert recorded_id in message
    assert "bank16" in message
    assert not base_artifact_path(
        tmp_path, module_id=MODULE_ID, generation_id=GENERATION_ID
    ).exists()


def test_a_stale_artifact_cannot_override_the_lineage_recorded_base(tmp_path: Path) -> None:
    """Artifact-first ordering must not lock in bytes the lineage never used."""

    pre_record = _ProbeCore()
    resolve_base_module(
        pre_record,
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )
    # The lineage later records a different base than the one this machine
    # happened to capture first, which is exactly the ordering hazard.
    recorded_id = _record_lineage(tmp_path, _ProbeCore(seed=11))

    with pytest.raises(BaseArtifactError) as error:
        resolve_base_module(
            _ProbeCore(),
            state_root=tmp_path,
            descriptor=_descriptor(),
            candidate_generation_id=CANDIDATE_GENERATION_ID,
        )

    message = str(error.value)
    assert "was captured before the lineage recorded base inventory" in message
    assert recorded_id in message


def test_a_resumed_lineage_persists_its_artifact_when_the_base_is_reproducible(
    tmp_path: Path,
) -> None:
    recorded_id = _record_lineage(tmp_path, _ProbeCore())

    disposition = resolve_base_module(
        _ProbeCore(),
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )

    assert disposition.action == "created-verified"
    assert disposition.recorded_base_inventory_id == recorded_id
    assert disposition.artifact_path.is_file()

    adopted = resolve_base_module(
        _ProbeCore(),
        state_root=tmp_path,
        descriptor=_descriptor(),
        candidate_generation_id=CANDIDATE_GENERATION_ID,
    )
    assert adopted.action == "adopted"
    assert adopted.recorded_base_inventory_id == recorded_id
