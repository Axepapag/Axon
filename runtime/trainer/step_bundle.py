"""Crash-consistent parameter-checkpoint plus private-Soul step boundaries.

An optimizer receipt, parameter checkpoint, and three causal Soul transitions
are individually durable, but none is an accepted reasoning-training step by
itself.  This module records an immutable intent before Soul publication,
finishes any interrupted transition chain idempotently, and publishes one
atomic pointer plus completion sentinel only after every artifact verifies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Mapping

from runtime.field import canonical_json_bytes, canonical_sha256
from runtime.soul import SoulCommitReceipt, SoulIntegrityError, SoulTransition, apply_soul_transition

from .lifecycle import CandidateCheckpointRecord, OptimizationStepReceipt
from .soul_candidates import CandidateSoulManifest, CandidateSoulWorkspace
from .store import TrainerStateStore, TrainerStoreError

TRAINING_STEP_INTENT_SCHEMA = "axon-reasoning-training-step-intent-v1"
ACCEPTED_TRAINING_STEP_SCHEMA = "axon-accepted-reasoning-training-step-v1"
ACCEPTED_STEP_POINTER_SCHEMA = "axon-accepted-training-step-pointer-v1"
ACCEPTED_STEP_SENTINEL_SCHEMA = "axon-accepted-training-step-sentinel-v1"
_PHASES = ("first", "refined", "consolidated")


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()


def _content_id(value: str, label: str) -> str:
    value = _required(value, label).lower()
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA256 identity")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json_bytes(existing) != canonical_json_bytes(dict(value)):
            raise TrainerStoreError(f"immutable accepted-step artifact disagrees at {path}")
        return
    _atomic_json(path, value)


@dataclass(frozen=True, slots=True)
class CandidateTrainingStepIntent:
    module_id: str
    candidate_generation_id: str
    core_id: str
    plan_id: str
    authorization_id: str
    learning_policy_id: str
    step: int
    optimization_receipt_id: str
    checkpoint_id: str
    candidate_soul_manifest_id: str
    before_soul_id: str
    expected_after_soul_id: str
    transitions: tuple[SoulTransition, ...]
    previous_bundle_id: str | None
    intent_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "candidate_generation_id",
            "core_id",
            "plan_id",
            "authorization_id",
            "learning_policy_id",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("accepted training step must be positive")
        for label in (
            "optimization_receipt_id",
            "checkpoint_id",
            "candidate_soul_manifest_id",
            "before_soul_id",
            "expected_after_soul_id",
        ):
            object.__setattr__(self, label, _content_id(getattr(self, label), label))
        if self.previous_bundle_id is not None:
            object.__setattr__(
                self,
                "previous_bundle_id",
                _content_id(self.previous_bundle_id, "previous_bundle_id"),
            )
        transitions = tuple(self.transitions)
        phases = tuple(item.phase for item in transitions)
        if not transitions or len(transitions) % len(_PHASES):
            raise ValueError("reasoning checkpoint segment requires complete three-phase steps")
        if any(
            phases[index : index + len(_PHASES)] != _PHASES
            for index in range(0, len(phases), len(_PHASES))
        ):
            raise ValueError(
                "every reasoning checkpoint segment step requires FIRST, REFINED, CONSOLIDATED"
            )
        if transitions[0].before_soul_id != self.before_soul_id:
            raise ValueError("training step does not begin at before_soul_id")
        for previous, current in pairwise(transitions):
            if current.before_generation != previous.before_generation + 1:
                raise ValueError("training step Soul generations are discontinuous")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "intent_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINING_STEP_INTENT_SCHEMA,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "core_id": self.core_id,
            "plan_id": self.plan_id,
            "authorization_id": self.authorization_id,
            "learning_policy_id": self.learning_policy_id,
            "step": self.step,
            "optimization_receipt_id": self.optimization_receipt_id,
            "checkpoint_id": self.checkpoint_id,
            "candidate_soul_manifest_id": self.candidate_soul_manifest_id,
            "before_soul_id": self.before_soul_id,
            "expected_after_soul_id": self.expected_after_soul_id,
            "transitions": [item.to_canonical_dict() for item in self.transitions],
            "previous_bundle_id": self.previous_bundle_id,
        }
        if include_id:
            value["intent_id"] = self.intent_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateTrainingStepIntent":
        item = dict(value)
        if item.pop("schema", None) != TRAINING_STEP_INTENT_SCHEMA:
            raise TrainerStoreError("unsupported training-step intent schema")
        intent_id = item.pop("intent_id", None)
        try:
            item["transitions"] = tuple(
                SoulTransition.from_mapping(raw) for raw in item["transitions"]
            )
            intent = cls(**item)
        except (KeyError, TypeError, ValueError, SoulIntegrityError) as exc:
            raise TrainerStoreError("training-step intent is invalid") from exc
        if intent.intent_id != intent_id:
            raise TrainerStoreError("training-step intent identity mismatch")
        return intent


@dataclass(frozen=True, slots=True)
class AcceptedTrainingStepBundle:
    intent_id: str
    module_id: str
    candidate_generation_id: str
    core_id: str
    step: int
    optimization_receipt_id: str
    checkpoint_id: str
    candidate_soul_manifest_id: str
    before_soul_id: str
    after_soul_id: str
    soul_receipt_ids: tuple[str, ...]
    previous_bundle_id: str | None
    bundle_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in ("module_id", "candidate_generation_id", "core_id"):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("accepted training step must be positive")
        for label in (
            "intent_id",
            "optimization_receipt_id",
            "checkpoint_id",
            "candidate_soul_manifest_id",
            "before_soul_id",
            "after_soul_id",
        ):
            object.__setattr__(self, label, _content_id(getattr(self, label), label))
        receipts = tuple(_content_id(item, "soul_receipt_id") for item in self.soul_receipt_ids)
        if (
            not receipts
            or len(receipts) % len(_PHASES)
            or len(set(receipts)) != len(receipts)
        ):
            raise ValueError(
                "accepted reasoning checkpoint segment requires complete unique Soul receipt triples"
            )
        object.__setattr__(self, "soul_receipt_ids", receipts)
        if self.previous_bundle_id is not None:
            object.__setattr__(
                self,
                "previous_bundle_id",
                _content_id(self.previous_bundle_id, "previous_bundle_id"),
            )
        object.__setattr__(self, "bundle_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": ACCEPTED_TRAINING_STEP_SCHEMA,
            "intent_id": self.intent_id,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "core_id": self.core_id,
            "step": self.step,
            "optimization_receipt_id": self.optimization_receipt_id,
            "checkpoint_id": self.checkpoint_id,
            "candidate_soul_manifest_id": self.candidate_soul_manifest_id,
            "before_soul_id": self.before_soul_id,
            "after_soul_id": self.after_soul_id,
            "soul_receipt_ids": list(self.soul_receipt_ids),
            "previous_bundle_id": self.previous_bundle_id,
        }
        if include_id:
            value["bundle_id"] = self.bundle_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcceptedTrainingStepBundle":
        item = dict(value)
        if item.pop("schema", None) != ACCEPTED_TRAINING_STEP_SCHEMA:
            raise TrainerStoreError("unsupported accepted-step bundle schema")
        bundle_id = item.pop("bundle_id", None)
        try:
            item["soul_receipt_ids"] = tuple(item["soul_receipt_ids"])
            bundle = cls(**item)
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainerStoreError("accepted-step bundle is invalid") from exc
        if bundle.bundle_id != bundle_id:
            raise TrainerStoreError("accepted-step bundle identity mismatch")
        return bundle


@dataclass(frozen=True, slots=True)
class AcceptedStepPointer:
    module_id: str
    candidate_generation_id: str
    current_bundle_id: str
    current_step: int
    rolling_bundle_ids: tuple[str, ...]
    pointer_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in ("module_id", "candidate_generation_id"):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        object.__setattr__(self, "current_bundle_id", _content_id(self.current_bundle_id, "current_bundle_id"))
        if isinstance(self.current_step, bool) or not isinstance(self.current_step, int) or self.current_step < 1:
            raise ValueError("accepted-step pointer step must be positive")
        rolling = tuple(_content_id(item, "rolling_bundle_id") for item in self.rolling_bundle_ids)
        if not rolling or len(rolling) > 3 or rolling[-1] != self.current_bundle_id:
            raise ValueError("accepted-step pointer must retain one to three bundles ending at current")
        object.__setattr__(self, "rolling_bundle_ids", rolling)
        object.__setattr__(self, "pointer_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": ACCEPTED_STEP_POINTER_SCHEMA,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "current_bundle_id": self.current_bundle_id,
            "current_step": self.current_step,
            "rolling_bundle_ids": list(self.rolling_bundle_ids),
        }
        if include_id:
            value["pointer_id"] = self.pointer_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcceptedStepPointer":
        item = dict(value)
        if item.pop("schema", None) != ACCEPTED_STEP_POINTER_SCHEMA:
            raise TrainerStoreError("unsupported accepted-step pointer schema")
        pointer_id = item.pop("pointer_id", None)
        try:
            item["rolling_bundle_ids"] = tuple(item["rolling_bundle_ids"])
            pointer = cls(**item)
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainerStoreError("accepted-step pointer is invalid") from exc
        if pointer.pointer_id != pointer_id:
            raise TrainerStoreError("accepted-step pointer identity mismatch")
        return pointer


class CandidateStepBundleCoordinator:
    """Publish or recover accepted candidate parameter+Soul step bundles."""

    def __init__(self, state_root: Path | str) -> None:
        self.state_root = Path(state_root).resolve(strict=False)
        self.trainer_store = TrainerStateStore.active(state_root=self.state_root)
        self.souls = CandidateSoulWorkspace(self.state_root)

    def _root(self, module_id: str, candidate_generation_id: str) -> Path:
        return (
            self.trainer_store.candidates_dir
            / _required(module_id, "module_id")
            / _required(candidate_generation_id, "candidate_generation_id")
            / "accepted_steps"
        )

    def _paths(self, module_id: str, candidate_generation_id: str) -> tuple[Path, Path, Path, Path]:
        root = self._root(module_id, candidate_generation_id)
        return root / "intents", root / "bundles", root / "pointer.json", root / "checkpoint_done.json"

    def _rebuild_pointer_from_bundles(
        self,
        module_id: str,
        candidate_generation_id: str,
    ) -> AcceptedStepPointer | None:
        _intents, bundles_dir, pointer_path, sentinel_path = self._paths(
            module_id,
            candidate_generation_id,
        )
        evidence = {
            "schema": "axon-accepted-step-pointer-recovery-evidence-v1",
            "module_id": module_id,
            "candidate_generation_id": candidate_generation_id,
            "pointer": (
                None if not pointer_path.exists() else pointer_path.read_text(encoding="utf-8")
            ),
            "sentinel": (
                None if not sentinel_path.exists() else sentinel_path.read_text(encoding="utf-8")
            ),
        }
        evidence["evidence_id"] = canonical_sha256(evidence)
        recovery_path = self._root(module_id, candidate_generation_id) / "recovery_evidence"
        _immutable_json(recovery_path / f"{evidence['evidence_id']}.json", evidence)
        bundles = tuple(
            AcceptedTrainingStepBundle.from_mapping(json.loads(path.read_text(encoding="utf-8")))
            for path in bundles_dir.glob("*.json")
        ) if bundles_dir.exists() else ()
        if not bundles:
            return None
        by_previous: dict[str | None, AcceptedTrainingStepBundle] = {}
        for bundle in bundles:
            if bundle.previous_bundle_id in by_previous:
                raise TrainerStoreError("accepted-step bundle history forks")
            by_previous[bundle.previous_bundle_id] = bundle
        chain: list[AcceptedTrainingStepBundle] = []
        cursor: str | None = None
        while cursor in by_previous:
            bundle = by_previous[cursor]
            if chain and bundle.step <= chain[-1].step:
                raise TrainerStoreError("accepted-step bundle history is not step-monotonic")
            chain.append(bundle)
            cursor = bundle.bundle_id
        if len(chain) != len(bundles):
            raise TrainerStoreError("accepted-step bundles are not one complete linear history")
        current = chain[-1]
        pointer = AcceptedStepPointer(
            module_id=module_id,
            candidate_generation_id=candidate_generation_id,
            current_bundle_id=current.bundle_id,
            current_step=current.step,
            rolling_bundle_ids=tuple(item.bundle_id for item in chain[-3:]),
        )
        sentinel = {
            "schema": ACCEPTED_STEP_SENTINEL_SCHEMA,
            "pointer_id": pointer.pointer_id,
            "bundle_id": pointer.current_bundle_id,
            "step": pointer.current_step,
        }
        _atomic_json(sentinel_path, sentinel)
        _atomic_json(pointer_path, pointer.to_canonical_dict())
        _atomic_json(sentinel_path, sentinel)
        return pointer

    def _durable_optimization_receipt(self, receipt_id: str) -> OptimizationStepReceipt:
        if not self.trainer_store.steps_path.exists():
            raise TrainerStoreError("optimization-step journal is missing")
        found: OptimizationStepReceipt | None = None
        for line in self.trainer_store.steps_path.read_text(encoding="utf-8").splitlines():
            receipt = OptimizationStepReceipt.from_mapping(json.loads(line))
            if receipt.receipt_id == receipt_id:
                found = receipt
        if found is None:
            raise TrainerStoreError("optimization receipt is not durable")
        return found

    def _checkpoint_record(
        self,
        module_id: str,
        candidate_generation_id: str,
        checkpoint_id: str,
    ) -> CandidateCheckpointRecord:
        path = (
            self.trainer_store.candidates_dir
            / module_id
            / candidate_generation_id
            / "checkpoint_records"
            / f"{checkpoint_id}.json"
        )
        if not path.is_file():
            raise TrainerStoreError("candidate checkpoint record is missing")
        record = CandidateCheckpointRecord.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        self.trainer_store.load_verified_candidate_checkpoint(record)
        return record

    def _soul_manifest(self, intent: CandidateTrainingStepIntent) -> CandidateSoulManifest:
        path = (
            self.state_root
            / "training"
            / "soul_candidates"
            / intent.candidate_generation_id
            / intent.core_id
            / "manifest.json"
        )
        if not path.is_file():
            raise TrainerStoreError("candidate Soul manifest is missing")
        manifest = CandidateSoulManifest.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        if manifest.manifest_id != intent.candidate_soul_manifest_id:
            raise TrainerStoreError("candidate Soul manifest identity mismatch")
        return manifest

    def latest_pointer(self, module_id: str, candidate_generation_id: str) -> AcceptedStepPointer | None:
        _intents, bundles, pointer_path, sentinel_path = self._paths(module_id, candidate_generation_id)
        if not pointer_path.exists() and not sentinel_path.exists():
            return None
        if not pointer_path.is_file() or not sentinel_path.is_file():
            raise TrainerStoreError("accepted-step pointer/sentinel publication is incomplete")
        pointer = AcceptedStepPointer.from_mapping(json.loads(pointer_path.read_text(encoding="utf-8")))
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        expected_sentinel = {
            "schema": ACCEPTED_STEP_SENTINEL_SCHEMA,
            "pointer_id": pointer.pointer_id,
            "bundle_id": pointer.current_bundle_id,
            "step": pointer.current_step,
        }
        if sentinel != expected_sentinel:
            raise TrainerStoreError("accepted-step completion sentinel disagrees with pointer")
        self.load_bundle(module_id, candidate_generation_id, pointer.current_bundle_id)
        for bundle_id in pointer.rolling_bundle_ids:
            if not (bundles / f"{bundle_id}.json").is_file():
                raise TrainerStoreError("rolling accepted-step bundle is missing")
        return pointer

    def load_bundle(
        self,
        module_id: str,
        candidate_generation_id: str,
        bundle_id: str,
    ) -> AcceptedTrainingStepBundle:
        _intents, bundles, _pointer, _sentinel = self._paths(module_id, candidate_generation_id)
        path = bundles / f"{bundle_id}.json"
        if not path.is_file():
            raise TrainerStoreError("accepted-step bundle is missing")
        bundle = AcceptedTrainingStepBundle.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        if bundle.module_id != module_id or bundle.candidate_generation_id != candidate_generation_id:
            raise TrainerStoreError("accepted-step bundle belongs to another candidate")
        return bundle

    def latest_bundle(
        self,
        module_id: str,
        candidate_generation_id: str,
    ) -> AcceptedTrainingStepBundle | None:
        pointer = self.latest_pointer(module_id, candidate_generation_id)
        if pointer is None:
            return None
        return self.load_bundle(module_id, candidate_generation_id, pointer.current_bundle_id)

    def prepare_intent(
        self,
        *,
        optimization_receipt: OptimizationStepReceipt,
        checkpoint: CandidateCheckpointRecord,
        soul_manifest: CandidateSoulManifest,
        transitions: tuple[SoulTransition, ...],
    ) -> CandidateTrainingStepIntent:
        if not isinstance(optimization_receipt, OptimizationStepReceipt):
            raise TypeError("optimization_receipt must be OptimizationStepReceipt")
        if not isinstance(checkpoint, CandidateCheckpointRecord):
            raise TypeError("checkpoint must be CandidateCheckpointRecord")
        if not isinstance(soul_manifest, CandidateSoulManifest):
            raise TypeError("soul_manifest must be CandidateSoulManifest")
        durable = self._durable_optimization_receipt(optimization_receipt.receipt_id)
        if durable != optimization_receipt:
            raise TrainerStoreError("optimization receipt differs from its durable journal entry")
        verified_checkpoint = self._checkpoint_record(
            checkpoint.module_id,
            checkpoint.candidate_generation_id,
            checkpoint.checkpoint_id,
        )
        if verified_checkpoint != checkpoint:
            raise TrainerStoreError("checkpoint differs from its durable record")
        if (
            checkpoint.step != optimization_receipt.step
            or checkpoint.module_id != optimization_receipt.module_id
            or checkpoint.candidate_generation_id != optimization_receipt.candidate_generation_id
            or checkpoint.plan_id != optimization_receipt.plan_id
            or checkpoint.authorization_id != optimization_receipt.authorization_id
            or checkpoint.learning_policy_id != optimization_receipt.learning_policy_id
        ):
            raise TrainerStoreError("checkpoint and optimization receipt lineage disagree")
        if soul_manifest.candidate_id != checkpoint.candidate_generation_id:
            raise TrainerStoreError("candidate Soul and parameter generations disagree")
        branch = self.souls.branch(soul_manifest.candidate_id, soul_manifest.core_id)
        before = branch.load_head()
        cursor = before
        for transition in transitions:
            cursor = apply_soul_transition(cursor, transition)
        pointer = self.latest_pointer(checkpoint.module_id, checkpoint.candidate_generation_id)
        previous_bundle_id = None if pointer is None else pointer.current_bundle_id
        prior_step = 0 if pointer is None else pointer.current_step
        if checkpoint.step <= prior_step:
            raise TrainerStoreError("accepted checkpoint segment does not advance optimizer state")
        intent = CandidateTrainingStepIntent(
            module_id=checkpoint.module_id,
            candidate_generation_id=checkpoint.candidate_generation_id,
            core_id=soul_manifest.core_id,
            plan_id=checkpoint.plan_id,
            authorization_id=checkpoint.authorization_id,
            learning_policy_id=checkpoint.learning_policy_id,
            step=checkpoint.step,
            optimization_receipt_id=optimization_receipt.receipt_id,
            checkpoint_id=checkpoint.checkpoint_id,
            candidate_soul_manifest_id=soul_manifest.manifest_id,
            before_soul_id=before.soul_id,
            expected_after_soul_id=cursor.soul_id,
            transitions=tuple(transitions),
            previous_bundle_id=previous_bundle_id,
        )
        intents, _bundles, _pointer, _sentinel = self._paths(
            intent.module_id,
            intent.candidate_generation_id,
        )
        _immutable_json(intents / f"{intent.intent_id}.json", intent.to_canonical_dict())
        return intent

    def finalize_intent(self, intent: CandidateTrainingStepIntent) -> AcceptedTrainingStepBundle:
        intents, bundles, pointer_path, sentinel_path = self._paths(
            intent.module_id,
            intent.candidate_generation_id,
        )
        durable_intent = CandidateTrainingStepIntent.from_mapping(
            json.loads((intents / f"{intent.intent_id}.json").read_text(encoding="utf-8"))
        )
        if durable_intent != intent:
            raise TrainerStoreError("training-step intent differs from durable intent")
        existing_paths = tuple(bundles.glob("*.json")) if bundles.exists() else ()
        for path in existing_paths:
            existing = AcceptedTrainingStepBundle.from_mapping(json.loads(path.read_text(encoding="utf-8")))
            if existing.intent_id == intent.intent_id:
                self._publish(existing, pointer_path, sentinel_path)
                return existing

        receipt = self._durable_optimization_receipt(intent.optimization_receipt_id)
        checkpoint = self._checkpoint_record(
            intent.module_id,
            intent.candidate_generation_id,
            intent.checkpoint_id,
        )
        manifest = self._soul_manifest(intent)
        if (
            receipt.step != intent.step
            or checkpoint.step != intent.step
            or receipt.plan_id != intent.plan_id
            or checkpoint.plan_id != intent.plan_id
            or manifest.candidate_id != intent.candidate_generation_id
        ):
            raise TrainerStoreError("accepted-step intent artifact lineage disagrees")

        branch = self.souls.branch(intent.candidate_generation_id, intent.core_id)
        snapshots = [branch.load_snapshot(intent.before_soul_id)]
        for transition in intent.transitions:
            snapshots.append(apply_soul_transition(snapshots[-1], transition))
        if snapshots[-1].soul_id != intent.expected_after_soul_id:
            raise TrainerStoreError("training-step intent expected Soul identity is wrong")
        head = branch.load_head()
        ids = [item.soul_id for item in snapshots]
        if head.soul_id not in ids:
            raise TrainerStoreError("candidate Soul HEAD is outside the recoverable step chain")
        completed = ids.index(head.soul_id)
        soul_receipts: list[SoulCommitReceipt] = []
        for index, transition in enumerate(intent.transitions):
            if index < completed:
                durable_receipt = branch.load_receipt(transition.transition_id)
            else:
                durable_receipt = branch.commit_transition(
                    transition,
                    commit_binding=f"trainer-step-intent:{intent.intent_id}",
                )
            if (
                durable_receipt.before_soul_id != snapshots[index].soul_id
                or durable_receipt.after_soul_id != snapshots[index + 1].soul_id
                or durable_receipt.commit_binding != f"trainer-step-intent:{intent.intent_id}"
            ):
                raise TrainerStoreError("candidate Soul receipt disagrees with accepted-step intent")
            soul_receipts.append(durable_receipt)
        if branch.load_head().soul_id != intent.expected_after_soul_id:
            raise TrainerStoreError("candidate Soul HEAD did not reach accepted-step successor")

        bundle = AcceptedTrainingStepBundle(
            intent_id=intent.intent_id,
            module_id=intent.module_id,
            candidate_generation_id=intent.candidate_generation_id,
            core_id=intent.core_id,
            step=intent.step,
            optimization_receipt_id=intent.optimization_receipt_id,
            checkpoint_id=intent.checkpoint_id,
            candidate_soul_manifest_id=intent.candidate_soul_manifest_id,
            before_soul_id=intent.before_soul_id,
            after_soul_id=intent.expected_after_soul_id,
            soul_receipt_ids=tuple(item.receipt_id for item in soul_receipts),
            previous_bundle_id=intent.previous_bundle_id,
        )
        _immutable_json(bundles / f"{bundle.bundle_id}.json", bundle.to_canonical_dict())
        self._publish(bundle, pointer_path, sentinel_path)
        return bundle

    def _publish(self, bundle: AcceptedTrainingStepBundle, pointer_path: Path, sentinel_path: Path) -> None:
        prior: AcceptedStepPointer | None = None
        if pointer_path.exists() and sentinel_path.exists():
            prior = self.latest_pointer(bundle.module_id, bundle.candidate_generation_id)
        if prior is not None and prior.current_bundle_id == bundle.bundle_id:
            return
        observed_previous = None if prior is None else prior.current_bundle_id
        if observed_previous != bundle.previous_bundle_id:
            raise TrainerStoreError("accepted-step pointer moved beyond this intent")
        rolling = (() if prior is None else prior.rolling_bundle_ids) + (bundle.bundle_id,)
        pointer = AcceptedStepPointer(
            module_id=bundle.module_id,
            candidate_generation_id=bundle.candidate_generation_id,
            current_bundle_id=bundle.bundle_id,
            current_step=bundle.step,
            rolling_bundle_ids=rolling[-3:],
        )
        sentinel = {
            "schema": ACCEPTED_STEP_SENTINEL_SCHEMA,
            "pointer_id": pointer.pointer_id,
            "bundle_id": pointer.current_bundle_id,
            "step": pointer.current_step,
        }
        _atomic_json(sentinel_path, sentinel)
        _atomic_json(pointer_path, pointer.to_canonical_dict())
        # The reader requires both exact values. Rewriting the sentinel after
        # pointer publication closes either single-file crash window.
        _atomic_json(sentinel_path, sentinel)

    def accept_step(
        self,
        *,
        optimization_receipt: OptimizationStepReceipt,
        checkpoint: CandidateCheckpointRecord,
        soul_manifest: CandidateSoulManifest,
        transitions: tuple[SoulTransition, ...],
    ) -> AcceptedTrainingStepBundle:
        intent = self.prepare_intent(
            optimization_receipt=optimization_receipt,
            checkpoint=checkpoint,
            soul_manifest=soul_manifest,
            transitions=transitions,
        )
        return self.finalize_intent(intent)

    def recover_pending(
        self,
        module_id: str,
        candidate_generation_id: str,
    ) -> tuple[AcceptedTrainingStepBundle, ...]:
        intents, _bundles, _pointer_path, _sentinel_path = self._paths(
            module_id,
            candidate_generation_id,
        )
        try:
            self.latest_pointer(module_id, candidate_generation_id)
        except TrainerStoreError:
            # A crash may separate or corrupt the two mutable publications.
            # Preserve their exact bytes, then rebuild solely from immutable
            # accepted bundles.  No historical bundle or intent is removed.
            self._rebuild_pointer_from_bundles(module_id, candidate_generation_id)
        pending = tuple(
            CandidateTrainingStepIntent.from_mapping(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(intents.glob("*.json"))
        ) if intents.exists() else ()
        recovered: list[AcceptedTrainingStepBundle] = []
        for intent in sorted(pending, key=lambda item: item.step):
            pointer = self.latest_pointer(module_id, candidate_generation_id)
            current_step = 0 if pointer is None else pointer.current_step
            if intent.step <= current_step:
                continue
            expected_previous = None if pointer is None else pointer.current_bundle_id
            if intent.previous_bundle_id != expected_previous:
                raise TrainerStoreError("pending accepted-step intent does not extend current history")
            recovered.append(self.finalize_intent(intent))
        return tuple(recovered)

    def checkpoint_for_bundle(self, bundle: AcceptedTrainingStepBundle) -> CandidateCheckpointRecord:
        return self._checkpoint_record(
            bundle.module_id,
            bundle.candidate_generation_id,
            bundle.checkpoint_id,
        )


__all__ = [
    "ACCEPTED_STEP_POINTER_SCHEMA",
    "ACCEPTED_STEP_SENTINEL_SCHEMA",
    "ACCEPTED_TRAINING_STEP_SCHEMA",
    "TRAINING_STEP_INTENT_SCHEMA",
    "AcceptedStepPointer",
    "AcceptedTrainingStepBundle",
    "CandidateStepBundleCoordinator",
    "CandidateTrainingStepIntent",
]
