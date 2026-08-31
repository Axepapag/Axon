"""Durable transparent state for Axon's Trainer control plane."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from runtime.field import canonical_json_bytes

from .activation import (
    ActiveGenerationPointer,
    GenerationSnapshotRecord,
    ParameterActivationReceipt,
    ParameterRollbackReceipt,
)
from .authority import AuthorizedParameterMutation
from .contracts import (
    ParameterInventory,
    ParameterModuleDescriptor,
    ParameterMutationPlanLike,
    ParameterPromotionProposal,
    is_parameter_mutation_plan,
)
from .episodes import RuntimeEpisodeSessionManifest
from .gates import EvaluationObservation, PromotionGateDecision
from .learning import GovernedLearningPolicy
from .lifecycle import (
    CandidateCheckpointRecord,
    CandidateLifecycleEvent,
    LearningMicrostepReceipt,
    OptimizationStepReceipt,
)
from .preflight import TrainingPreflightReceipt
from .registry import capture_module_manifest
from .sessions import TrainerSessionManifest
from .telemetry import ParameterTelemetryFrame


class TrainerStoreError(RuntimeError):
    pass


class TrainerStateStore:
    """Immutable lineage artifacts plus append-only live parameter telemetry."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve(strict=False)
        self.inventories_dir = self.root / "inventories"
        self.plans_dir = self.root / "plans"
        self.capacity_contracts_dir = self.root / "capacity_contracts"
        self.preflight_receipts_dir = self.root / "preflight_receipts"
        self.sessions_dir = self.root / "sessions"
        self.authorizations_dir = self.root / "authorizations"
        self.promotions_dir = self.root / "promotion_proposals"
        self.learning_policies_dir = self.root / "learning_policies"
        self.latest_learning_policy_path = self.root / "latest_learning_policy.json"
        self.candidates_dir = self.root / "candidates"
        self.active_generations_dir = self.root / "active_generations"
        self.generation_snapshots_dir = self.root / "generation_snapshots"
        self.activation_receipts_dir = self.root / "activation_receipts"
        self.rollback_receipts_dir = self.root / "rollback_receipts"
        self.lifecycle_path = self.root / "candidate_lifecycle.jsonl"
        self.latest_lifecycle_path = self.root / "latest_candidate_lifecycle.json"
        self.microsteps_path = self.root / "learning_microsteps.jsonl"
        self.latest_microstep_path = self.root / "latest_learning_microstep.json"
        self.steps_path = self.root / "optimization_steps.jsonl"
        self.latest_step_path = self.root / "latest_optimization_step.json"
        self.evaluations_dir = self.root / "evaluations"
        self.gate_decisions_dir = self.root / "gate_decisions"
        self.telemetry_path = self.root / "telemetry.jsonl"
        self.latest_telemetry_path = self.root / "latest_telemetry.json"

    @classmethod
    def active(cls, *, state_root: Path | str = Path(r"D:\Axon\State")) -> "TrainerStateStore":
        state = Path(state_root).resolve(strict=False)
        return cls(state / "training" / "trainer")

    def write_inventory(self, inventory: ParameterInventory) -> Path:
        if not isinstance(inventory, ParameterInventory):
            raise TypeError("inventory must be ParameterInventory")
        path = self.inventories_dir / f"{inventory.inventory_id}.json"
        self._write_immutable(path, inventory.to_canonical_dict())
        return path

    def write_plan(self, plan: ParameterMutationPlanLike) -> Path:
        if not is_parameter_mutation_plan(plan):
            raise TypeError("plan must be a governed parameter mutation plan")
        path = self.plans_dir / f"{plan.plan_id}.json"
        self._write_immutable(path, plan.to_canonical_dict())
        return path

    def write_preflight_receipt(self, receipt: TrainingPreflightReceipt) -> Path:
        if not isinstance(receipt, TrainingPreflightReceipt):
            raise TypeError("receipt must be TrainingPreflightReceipt")
        contract_path = self.capacity_contracts_dir / f"{receipt.contract.contract_id}.json"
        self._write_immutable(contract_path, receipt.contract.to_canonical_dict())
        path = self.preflight_receipts_dir / f"{receipt.receipt_id}.json"
        self._write_immutable(path, receipt.to_canonical_dict())
        return path

    def write_learning_policy(self, policy: GovernedLearningPolicy) -> Path:
        if not isinstance(policy, GovernedLearningPolicy):
            raise TypeError("policy must be GovernedLearningPolicy")
        path = self.learning_policies_dir / f"{policy.policy_id}.json"
        value = policy.to_canonical_dict()
        self._write_immutable(path, value)
        self._atomic_json(self.latest_learning_policy_path, value)
        return path

    def write_session(self, session: TrainerSessionManifest) -> Path:
        if not isinstance(session, TrainerSessionManifest):
            raise TypeError("session must be a TrainerSessionManifest")
        path = self.sessions_dir / session.session_id / "manifest.json"
        self._write_immutable(path, session.to_canonical_dict())
        return path

    def write_runtime_episode_session(self, session: RuntimeEpisodeSessionManifest) -> Path:
        if not isinstance(session, RuntimeEpisodeSessionManifest):
            raise TypeError("session must be a RuntimeEpisodeSessionManifest")
        path = self.sessions_dir / session.session_id / "manifest.json"
        self._write_immutable(path, session.to_canonical_dict())
        return path

    def write_authorization(self, authorization: AuthorizedParameterMutation) -> Path:
        if not isinstance(authorization, AuthorizedParameterMutation):
            raise TypeError("authorization must be AuthorizedParameterMutation")
        path = self.authorizations_dir / f"{authorization.authorization_id}.json"
        self._write_immutable(path, authorization.to_canonical_dict())
        return path

    def read_authorization(self, authorization_id: str) -> AuthorizedParameterMutation:
        path = self.authorizations_dir / f"{authorization_id}.json"
        if not path.is_file():
            raise TrainerStoreError("candidate checkpoint authorization record is missing")
        return AuthorizedParameterMutation.from_mapping(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def write_promotion_proposal(self, proposal: ParameterPromotionProposal) -> Path:
        if not isinstance(proposal, ParameterPromotionProposal):
            raise TypeError("proposal must be ParameterPromotionProposal")
        path = self.promotions_dir / f"{proposal.proposal_id}.json"
        self._write_immutable(path, proposal.to_canonical_dict())
        return path

    def append_telemetry(self, frame: ParameterTelemetryFrame) -> None:
        if not isinstance(frame, ParameterTelemetryFrame):
            raise TypeError("frame must be ParameterTelemetryFrame")
        value = frame.to_canonical_dict()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._atomic_json(self.latest_telemetry_path, value)

    def append_candidate_lifecycle(self, event: CandidateLifecycleEvent) -> None:
        if not isinstance(event, CandidateLifecycleEvent):
            raise TypeError("event must be CandidateLifecycleEvent")
        self._append_jsonl(self.lifecycle_path, event.to_canonical_dict())
        self._atomic_json(self.latest_lifecycle_path, event.to_canonical_dict())

    def append_learning_microstep(self, receipt: LearningMicrostepReceipt) -> None:
        if not isinstance(receipt, LearningMicrostepReceipt):
            raise TypeError("receipt must be LearningMicrostepReceipt")
        self._append_jsonl(self.microsteps_path, receipt.to_canonical_dict())
        self._atomic_json(self.latest_microstep_path, receipt.to_canonical_dict())

    def append_optimization_step(self, receipt: OptimizationStepReceipt) -> None:
        if not isinstance(receipt, OptimizationStepReceipt):
            raise TypeError("receipt must be OptimizationStepReceipt")
        self._append_jsonl(self.steps_path, receipt.to_canonical_dict())
        self._atomic_json(self.latest_step_path, receipt.to_canonical_dict())

    def write_evaluation_observation(self, observation: EvaluationObservation) -> Path:
        if not isinstance(observation, EvaluationObservation):
            raise TypeError("observation must be EvaluationObservation")
        path = self.evaluations_dir / f"{observation.observation_id}.json"
        self._write_immutable(path, observation.to_canonical_dict())
        return path

    def write_gate_decision(self, decision: PromotionGateDecision) -> Path:
        if not isinstance(decision, PromotionGateDecision):
            raise TypeError("decision must be PromotionGateDecision")
        path = self.gate_decisions_dir / f"{decision.decision_id}.json"
        self._write_immutable(path, decision.to_canonical_dict())
        return path

    def save_candidate_checkpoint(
        self,
        *,
        module: nn.Module,
        descriptor: ParameterModuleDescriptor,
        base_generation_id: str,
        plan_id: str,
        authorization_id: str,
        learning_policy: GovernedLearningPolicy,
        step: int,
        micro_step: int,
        accumulation_index: int,
        current_learning_rate: float,
        accumulated_loss_sum: float,
        optimizer: torch.optim.Optimizer | None = None,
        scaler_state: Mapping[str, Any] | None = None,
        gradient_state: Mapping[str, torch.Tensor | None] | None = None,
        previous_checkpoint_id: str | None = None,
    ) -> CandidateCheckpointRecord:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be torch.nn.Module")
        if not isinstance(descriptor, ParameterModuleDescriptor):
            raise TypeError("descriptor must be ParameterModuleDescriptor")
        if not isinstance(learning_policy, GovernedLearningPolicy):
            raise TypeError("learning_policy must be GovernedLearningPolicy")
        if descriptor.generation_id == base_generation_id:
            raise TrainerStoreError("candidate descriptor generation must differ from base generation")
        for label, value in (("step", step), ("micro_step", micro_step), ("accumulation_index", accumulation_index)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        current_learning_rate = float(current_learning_rate)
        if not (0.0 < current_learning_rate < float("inf")):
            raise ValueError("current_learning_rate must be positive and finite")

        manifest = capture_module_manifest(descriptor, module, exact_value_hashes=True)
        candidate_root = self.candidates_dir / descriptor.module_id / descriptor.generation_id
        checkpoints_dir = candidate_root / "checkpoints"
        staging_dir = candidate_root / "checkpoint_staging"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        gradients = None
        if gradient_state is not None:
            gradients = {
                str(name): None if gradient is None else gradient.detach().to(device="cpu").clone()
                for name, gradient in gradient_state.items()
            }
        payload = {
            "schema": "axon-trainer-candidate-checkpoint-payload-v2",
            "descriptor": descriptor.to_canonical_dict(),
            "base_generation_id": base_generation_id,
            "plan_id": plan_id,
            "authorization_id": authorization_id,
            "learning_policy_id": learning_policy.policy_id,
            "learning_policy": learning_policy.to_canonical_dict(),
            "step": step,
            "micro_step": micro_step,
            "accumulation_index": accumulation_index,
            "current_learning_rate": current_learning_rate,
            "accumulated_loss_sum": float(accumulated_loss_sum),
            "parameter_manifest_id": manifest.manifest_id,
            "parameter_manifest": manifest.to_canonical_dict(),
            "module_state_dict": module.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "scaler_state_dict": None if scaler_state is None else dict(scaler_state),
            "gradient_state_dict": gradients,
        }
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"step_{step:09d}_micro_{micro_step:012d}_",
            suffix=".pt.tmp",
            dir=staging_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = self._file_sha256(temporary)
        artifact_path = checkpoints_dir / f"{digest}.pt"
        if artifact_path.exists():
            if self._file_sha256(artifact_path) != digest:
                raise TrainerStoreError("content-addressed checkpoint path collision")
            temporary.unlink()
        else:
            os.replace(temporary, artifact_path)
        size = artifact_path.stat().st_size
        relpath = artifact_path.relative_to(self.root).as_posix()
        record = CandidateCheckpointRecord(
            module_id=descriptor.module_id,
            base_generation_id=base_generation_id,
            candidate_generation_id=descriptor.generation_id,
            plan_id=plan_id,
            authorization_id=authorization_id,
            learning_policy_id=learning_policy.policy_id,
            step=step,
            micro_step=micro_step,
            accumulation_index=accumulation_index,
            parameter_manifest_id=manifest.manifest_id,
            artifact_relpath=relpath,
            artifact_sha256=digest,
            artifact_bytes=size,
            optimizer_included=optimizer is not None,
            gradient_state_included=gradient_state is not None,
            scaler_included=scaler_state is not None,
            current_learning_rate=current_learning_rate,
            accumulated_loss_sum=float(accumulated_loss_sum),
            previous_checkpoint_id=previous_checkpoint_id,
        )
        record_path = candidate_root / "checkpoint_records" / f"{record.checkpoint_id}.json"
        self._write_immutable(record_path, record.to_canonical_dict())
        self._atomic_json(candidate_root / "latest_checkpoint.json", record.to_canonical_dict())
        return record

    def load_verified_candidate_checkpoint(self, record: CandidateCheckpointRecord) -> dict[str, Any]:
        if not isinstance(record, CandidateCheckpointRecord):
            raise TypeError("record must be CandidateCheckpointRecord")
        artifact_path = (self.root / record.artifact_relpath).resolve(strict=False)
        try:
            artifact_path.relative_to(self.root)
        except ValueError as exc:
            raise TrainerStoreError("candidate checkpoint escapes Trainer state root") from exc
        if not artifact_path.is_file():
            raise TrainerStoreError(f"candidate checkpoint missing at {artifact_path}")
        if artifact_path.stat().st_size != record.artifact_bytes:
            raise TrainerStoreError("candidate checkpoint byte length disagrees with record")
        if self._file_sha256(artifact_path) != record.artifact_sha256:
            raise TrainerStoreError("candidate checkpoint SHA256 disagrees with record")
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "axon-trainer-candidate-checkpoint-payload-v2":
            raise TrainerStoreError("candidate checkpoint payload schema mismatch")
        descriptor = payload.get("descriptor", {})
        checks = {
            "module_id": descriptor.get("module_id"),
            "candidate_generation_id": descriptor.get("generation_id"),
            "base_generation_id": payload.get("base_generation_id"),
            "plan_id": payload.get("plan_id"),
            "authorization_id": payload.get("authorization_id"),
            "learning_policy_id": payload.get("learning_policy_id"),
            "step": payload.get("step"),
            "micro_step": payload.get("micro_step"),
            "accumulation_index": payload.get("accumulation_index"),
            "parameter_manifest_id": payload.get("parameter_manifest_id"),
            "current_learning_rate": payload.get("current_learning_rate"),
            "accumulated_loss_sum": payload.get("accumulated_loss_sum"),
        }
        expected = {
            "module_id": record.module_id,
            "candidate_generation_id": record.candidate_generation_id,
            "base_generation_id": record.base_generation_id,
            "plan_id": record.plan_id,
            "authorization_id": record.authorization_id,
            "learning_policy_id": record.learning_policy_id,
            "step": record.step,
            "micro_step": record.micro_step,
            "accumulation_index": record.accumulation_index,
            "parameter_manifest_id": record.parameter_manifest_id,
            "current_learning_rate": record.current_learning_rate,
            "accumulated_loss_sum": record.accumulated_loss_sum,
        }
        if checks != expected:
            raise TrainerStoreError("candidate checkpoint lineage metadata disagrees with record")
        manifest = payload.get("parameter_manifest")
        if not isinstance(manifest, dict) or manifest.get("manifest_id") != record.parameter_manifest_id:
            raise TrainerStoreError("candidate checkpoint embedded manifest disagrees with record")
        policy = payload.get("learning_policy")
        if not isinstance(policy, dict) or policy.get("policy_id") != record.learning_policy_id:
            raise TrainerStoreError("candidate checkpoint embedded learning policy disagrees with record")
        if bool(payload.get("optimizer_state_dict") is not None) != bool(record.optimizer_included):
            raise TrainerStoreError("candidate checkpoint optimizer-state flag mismatch")
        if bool(payload.get("gradient_state_dict") is not None) != bool(record.gradient_state_included):
            raise TrainerStoreError("candidate checkpoint gradient-state flag mismatch")
        if bool(payload.get("scaler_state_dict") is not None) != bool(record.scaler_included):
            raise TrainerStoreError("candidate checkpoint scaler-state flag mismatch")
        return payload

    def save_generation_snapshot(
        self,
        *,
        module: nn.Module,
        descriptor: ParameterModuleDescriptor,
        source_inventory_id: str,
        previous_pointer_id: str | None = None,
    ) -> GenerationSnapshotRecord:
        """Persist one exact restorable model-state generation outside the live module."""

        if not isinstance(module, nn.Module):
            raise TypeError("module must be torch.nn.Module")
        if not isinstance(descriptor, ParameterModuleDescriptor):
            raise TypeError("descriptor must be ParameterModuleDescriptor")
        manifest = capture_module_manifest(descriptor, module, exact_value_hashes=True)
        generation_root = self.generation_snapshots_dir / descriptor.module_id / descriptor.generation_id
        generation_root.mkdir(parents=True, exist_ok=True)
        lineage_suffix = hashlib.sha256(source_inventory_id.encode("utf-8")).hexdigest()[:16]
        artifact_path = generation_root / f"{manifest.manifest_id}.{lineage_suffix}.pt"
        if not artifact_path.exists():
            temporary = artifact_path.with_name(artifact_path.name + ".tmp")
            payload = {
                "schema": "axon-trainer-generation-snapshot-payload-v1",
                "descriptor": descriptor.to_canonical_dict(),
                "source_inventory_id": source_inventory_id,
                "parameter_manifest_id": manifest.manifest_id,
                "module_state_dict": module.state_dict(),
            }
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, artifact_path)
        digest = self._file_sha256(artifact_path)
        record = GenerationSnapshotRecord(
            module_id=descriptor.module_id,
            generation_id=descriptor.generation_id,
            parameter_manifest_id=manifest.manifest_id,
            artifact_relpath=artifact_path.relative_to(self.root).as_posix(),
            artifact_sha256=digest,
            artifact_bytes=artifact_path.stat().st_size,
            source_inventory_id=source_inventory_id,
            previous_pointer_id=previous_pointer_id,
        )
        record_path = generation_root / f"{record.snapshot_id}.json"
        self._write_immutable(record_path, record.to_canonical_dict())
        return record

    def load_verified_generation_snapshot(self, record: GenerationSnapshotRecord) -> dict[str, Any]:
        if not isinstance(record, GenerationSnapshotRecord):
            raise TypeError("record must be GenerationSnapshotRecord")
        artifact_path = (self.root / record.artifact_relpath).resolve(strict=False)
        try:
            artifact_path.relative_to(self.root)
        except ValueError as exc:
            raise TrainerStoreError("generation snapshot escapes Trainer state root") from exc
        if not artifact_path.is_file():
            raise TrainerStoreError(f"generation snapshot missing at {artifact_path}")
        if artifact_path.stat().st_size != record.artifact_bytes:
            raise TrainerStoreError("generation snapshot byte length disagrees with record")
        if self._file_sha256(artifact_path) != record.artifact_sha256:
            raise TrainerStoreError("generation snapshot SHA256 disagrees with record")
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "axon-trainer-generation-snapshot-payload-v1":
            raise TrainerStoreError("generation snapshot payload schema mismatch")
        descriptor = payload.get("descriptor", {})
        if descriptor.get("module_id") != record.module_id or descriptor.get("generation_id") != record.generation_id:
            raise TrainerStoreError("generation snapshot descriptor lineage mismatch")
        if payload.get("source_inventory_id") != record.source_inventory_id:
            raise TrainerStoreError("generation snapshot inventory lineage mismatch")
        if payload.get("parameter_manifest_id") != record.parameter_manifest_id:
            raise TrainerStoreError("generation snapshot manifest lineage mismatch")
        return payload

    def write_generation_snapshot_record(self, record: GenerationSnapshotRecord) -> Path:
        if not isinstance(record, GenerationSnapshotRecord):
            raise TypeError("record must be GenerationSnapshotRecord")
        path = self.generation_snapshots_dir / record.module_id / record.generation_id / f"{record.snapshot_id}.json"
        self._write_immutable(path, record.to_canonical_dict())
        return path

    def read_active_pointer(self, module_id: str) -> ActiveGenerationPointer | None:
        path = self.active_generations_dir / module_id / "pointer.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainerStoreError(f"invalid active-generation pointer at {path}") from exc
        if value.get("schema") != "axon-trainer-active-generation-pointer-v1":
            raise TrainerStoreError("unsupported active-generation pointer schema")
        stored_id = value.pop("pointer_id", None)
        value.pop("schema", None)
        pointer = ActiveGenerationPointer(**value)
        if stored_id != pointer.pointer_id:
            raise TrainerStoreError("active-generation pointer hash mismatch")
        return pointer

    def publish_active_pointer(self, pointer: ActiveGenerationPointer) -> Path:
        if not isinstance(pointer, ActiveGenerationPointer):
            raise TypeError("pointer must be ActiveGenerationPointer")
        path = self.active_generations_dir / pointer.module_id / "pointer.json"
        self._atomic_json(path, pointer.to_canonical_dict())
        history = self.active_generations_dir / pointer.module_id / "history" / f"{pointer.pointer_id}.json"
        self._write_immutable(history, pointer.to_canonical_dict())
        return path

    def write_activation_receipt(self, receipt: ParameterActivationReceipt) -> Path:
        if not isinstance(receipt, ParameterActivationReceipt):
            raise TypeError("receipt must be ParameterActivationReceipt")
        path = self.activation_receipts_dir / receipt.module_id / f"{receipt.receipt_id}.json"
        self._write_immutable(path, receipt.to_canonical_dict())
        return path

    def write_rollback_receipt(self, receipt: ParameterRollbackReceipt) -> Path:
        if not isinstance(receipt, ParameterRollbackReceipt):
            raise TypeError("receipt must be ParameterRollbackReceipt")
        path = self.rollback_receipts_dir / receipt.module_id / f"{receipt.receipt_id}.json"
        self._write_immutable(path, receipt.to_canonical_dict())
        return path

    def read_generation_snapshot_record(self, module_id: str, generation_id: str, snapshot_id: str) -> GenerationSnapshotRecord:
        path = self.generation_snapshots_dir / module_id / generation_id / f"{snapshot_id}.json"
        if not path.is_file():
            raise TrainerStoreError(f"generation snapshot record missing at {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "axon-trainer-generation-snapshot-v1":
            raise TrainerStoreError("generation snapshot record schema mismatch")
        stored_id = value.pop("snapshot_id", None)
        value.pop("schema", None)
        record = GenerationSnapshotRecord(**value)
        if stored_id != record.snapshot_id:
            raise TrainerStoreError("generation snapshot record hash mismatch")
        return record

    def _append_jsonl(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _file_sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _write_immutable(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json_bytes(existing) != canonical_json_bytes(dict(value)):
                raise TrainerStoreError(f"immutable Trainer artifact disagrees at {path}")
            return
        self._atomic_json(path, value)


__all__ = ["TrainerStateStore", "TrainerStoreError"]
