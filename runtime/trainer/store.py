"""Durable transparent state for Axon's Trainer control plane."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from runtime.field import canonical_json_bytes

from .authority import AuthorizedParameterMutation
from .contracts import ParameterInventory, ParameterModuleDescriptor, ParameterMutationPlan, ParameterPromotionProposal
from .gates import EvaluationObservation, PromotionGateDecision
from .lifecycle import CandidateCheckpointRecord, CandidateLifecycleEvent, OptimizationStepReceipt
from .registry import capture_module_manifest
from .telemetry import ParameterTelemetryFrame


class TrainerStoreError(RuntimeError):
    pass


class TrainerStateStore:
    """Immutable lineage artifacts plus append-only live parameter telemetry."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve(strict=False)
        self.inventories_dir = self.root / "inventories"
        self.plans_dir = self.root / "plans"
        self.authorizations_dir = self.root / "authorizations"
        self.promotions_dir = self.root / "promotion_proposals"
        self.candidates_dir = self.root / "candidates"
        self.lifecycle_path = self.root / "candidate_lifecycle.jsonl"
        self.latest_lifecycle_path = self.root / "latest_candidate_lifecycle.json"
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

    def write_plan(self, plan: ParameterMutationPlan) -> Path:
        if not isinstance(plan, ParameterMutationPlan):
            raise TypeError("plan must be ParameterMutationPlan")
        path = self.plans_dir / f"{plan.plan_id}.json"
        self._write_immutable(path, plan.to_canonical_dict())
        return path

    def write_authorization(self, authorization: AuthorizedParameterMutation) -> Path:
        if not isinstance(authorization, AuthorizedParameterMutation):
            raise TypeError("authorization must be AuthorizedParameterMutation")
        path = self.authorizations_dir / f"{authorization.authorization_id}.json"
        self._write_immutable(path, authorization.to_canonical_dict())
        return path

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
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
        previous_checkpoint_id: str | None = None,
    ) -> CandidateCheckpointRecord:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be torch.nn.Module")
        if not isinstance(descriptor, ParameterModuleDescriptor):
            raise TypeError("descriptor must be ParameterModuleDescriptor")
        if descriptor.generation_id == base_generation_id:
            raise TrainerStoreError("candidate descriptor generation must differ from base generation")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")

        manifest = capture_module_manifest(descriptor, module, exact_value_hashes=True)
        candidate_root = self.candidates_dir / descriptor.module_id / descriptor.generation_id
        checkpoints_dir = candidate_root / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = checkpoints_dir / f"step_{step:09d}.pt"
        temporary = artifact_path.with_name(artifact_path.name + ".tmp")
        payload = {
            "schema": "axon-trainer-candidate-checkpoint-payload-v1",
            "descriptor": descriptor.to_canonical_dict(),
            "base_generation_id": base_generation_id,
            "plan_id": plan_id,
            "authorization_id": authorization_id,
            "step": step,
            "parameter_manifest_id": manifest.manifest_id,
            "module_state_dict": module.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        }
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, artifact_path)
        digest = self._file_sha256(artifact_path)
        size = artifact_path.stat().st_size
        relpath = artifact_path.relative_to(self.root).as_posix()
        record = CandidateCheckpointRecord(
            module_id=descriptor.module_id,
            base_generation_id=base_generation_id,
            candidate_generation_id=descriptor.generation_id,
            plan_id=plan_id,
            authorization_id=authorization_id,
            step=step,
            parameter_manifest_id=manifest.manifest_id,
            artifact_relpath=relpath,
            artifact_sha256=digest,
            artifact_bytes=size,
            optimizer_included=optimizer is not None,
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
        if payload.get("schema") != "axon-trainer-candidate-checkpoint-payload-v1":
            raise TrainerStoreError("candidate checkpoint payload schema mismatch")
        descriptor = payload.get("descriptor", {})
        checks = {
            "module_id": descriptor.get("module_id"),
            "candidate_generation_id": descriptor.get("generation_id"),
            "base_generation_id": payload.get("base_generation_id"),
            "plan_id": payload.get("plan_id"),
            "authorization_id": payload.get("authorization_id"),
            "step": payload.get("step"),
            "parameter_manifest_id": payload.get("parameter_manifest_id"),
        }
        expected = {
            "module_id": record.module_id,
            "candidate_generation_id": record.candidate_generation_id,
            "base_generation_id": record.base_generation_id,
            "plan_id": record.plan_id,
            "authorization_id": record.authorization_id,
            "step": record.step,
            "parameter_manifest_id": record.parameter_manifest_id,
        }
        if checks != expected:
            raise TrainerStoreError("candidate checkpoint lineage metadata disagrees with record")
        return payload

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


__all__ = ["TrainerStoreError", "TrainerStateStore"]
