"""Renewable resource tranches and additive continuation receipts.

A tranche is a bounded execution allowance for ONE segment of one candidate
lineage.  It is deliberately separate from the immutable v1 mutation plan:
plan identity, candidate-generation identity, learning-policy identity, and
checkpoint lineage never include tranche size, wall-time, cost, provider,
device, or cadence.  Reaching a bound pauses work at an exact accepted
checkpoint; a later tranche resumes the same lineage at global step N+1 from
that exact parent bundle.

Historical v1 plans keep their ``max_steps`` as immutable truth — the
envelope under which their accepted steps ran.  A tranche may continue a
candidate beyond that envelope when the parent checkpoint, optimizer state,
and private-Soul HEAD are verified exact and the continuation is receipted.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from runtime.field import canonical_json_bytes, canonical_sha256

RESOURCE_TRANCHE_SCHEMA = "axon-trainer-resource-tranche-v1"
TRANCHE_CONTINUATION_SCHEMA = "axon-trainer-tranche-continuation-v1"


class TrancheError(RuntimeError):
    """Fail-closed tranche/continuation validation failure."""


def _content_id(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TrancheError(f"{label} must be a 64-char content identity")
    try:
        int(value, 16)
    except ValueError as exc:
        raise TrancheError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrancheError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ResourceTranche:
    """One renewable execution allowance for a single candidate lineage.

    ``steps`` bounds the optimizer steps this tranche grants beyond the
    tranche's own base global step.  ``base_global_step`` is the accepted
    global optimizer step the tranche starts from; it is an execution fact,
    not tissue identity.  Learning-rate scheduling is deliberately absent:
    changing resource allowance must never change the learning recipe.
    """

    module_id: str
    candidate_generation_id: str
    plan_id: str
    learning_policy_id: str
    base_global_step: int
    steps: int
    parent_bundle_id: str | None = None
    purpose: str = "bounded continuation tranche"
    tranche_id: str = field(init=False)

    def __post_init__(self) -> None:
        _required(self.module_id, "module_id")
        _required(self.candidate_generation_id, "candidate_generation_id")
        for label in ("plan_id", "learning_policy_id"):
            object.__setattr__(self, label, _content_id(getattr(self, label), label))
        for label, value in (("base_global_step", self.base_global_step), ("steps", self.steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TrancheError(f"{label} must be a non-negative integer")
        if self.steps < 1:
            raise TrancheError("a tranche must grant at least one optimizer step")
        if self.base_global_step == 0 and self.parent_bundle_id is not None:
            raise TrancheError("a base-zero tranche cannot name a parent bundle")
        if self.base_global_step > 0 and self.parent_bundle_id is None:
            raise TrancheError("a continuation tranche requires an exact parent bundle")
        if self.parent_bundle_id is not None:
            object.__setattr__(
                self,
                "parent_bundle_id",
                _content_id(self.parent_bundle_id, "parent_bundle_id"),
            )
        _required(self.purpose, "purpose")
        object.__setattr__(
            self,
            "tranche_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    @property
    def final_global_step(self) -> int:
        return self.base_global_step + self.steps

    def admits_step(self, global_step: int) -> bool:
        """One-based global step admissibility within this tranche."""

        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise TrancheError("global_step must be an integer")
        return self.base_global_step < global_step <= self.final_global_step

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": RESOURCE_TRANCHE_SCHEMA,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "plan_id": self.plan_id,
            "learning_policy_id": self.learning_policy_id,
            "base_global_step": self.base_global_step,
            "steps": self.steps,
            "final_global_step": self.final_global_step,
            "parent_bundle_id": self.parent_bundle_id,
            "purpose": self.purpose,
        }
        if include_id:
            value["tranche_id"] = self.tranche_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourceTranche":
        item = dict(value)
        if item.pop("schema", None) != RESOURCE_TRANCHE_SCHEMA:
            raise TrancheError("unsupported resource tranche schema")
        observed = item.pop("tranche_id", None)
        item.pop("final_global_step", None)
        tranche = cls(**item)
        if tranche.tranche_id != observed:
            raise TrancheError("resource tranche identity mismatch")
        return tranche


@dataclass(frozen=True, slots=True)
class TrancheContinuation:
    """Immutable receipt binding one tranche to its exact accepted parent.

    Names the parent accepted-step bundle (parameter checkpoint, optimizer
    receipt, Soul HEAD) and the prior tranche chain, proving the lineage
    continued rather than restarted.
    """

    tranche_id: str
    module_id: str
    candidate_generation_id: str
    plan_id: str
    learning_policy_id: str
    parent_bundle_id: str
    parent_checkpoint_id: str
    parent_optimizer_receipt_id: str
    parent_soul_id: str
    parent_global_step: int
    prior_tranche_id: str | None
    continuation_id: str = field(init=False)

    def __post_init__(self) -> None:
        tranche_id = _content_id(self.tranche_id, "tranche_id")
        object.__setattr__(self, "tranche_id", tranche_id)
        _required(self.module_id, "module_id")
        _required(self.candidate_generation_id, "candidate_generation_id")
        for label in (
            "plan_id",
            "learning_policy_id",
            "parent_bundle_id",
            "parent_checkpoint_id",
            "parent_optimizer_receipt_id",
            "parent_soul_id",
        ):
            object.__setattr__(self, label, _content_id(getattr(self, label), label))
        if isinstance(self.parent_global_step, bool) or not isinstance(self.parent_global_step, int) or self.parent_global_step < 0:
            raise TrancheError("parent_global_step must be a non-negative integer")
        if self.prior_tranche_id is not None:
            _content_id(self.prior_tranche_id, "prior_tranche_id")
        object.__setattr__(
            self,
            "continuation_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRANCHE_CONTINUATION_SCHEMA,
            "tranche_id": self.tranche_id,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "plan_id": self.plan_id,
            "learning_policy_id": self.learning_policy_id,
            "parent_bundle_id": self.parent_bundle_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "parent_optimizer_receipt_id": self.parent_optimizer_receipt_id,
            "parent_soul_id": self.parent_soul_id,
            "parent_global_step": self.parent_global_step,
            "prior_tranche_id": self.prior_tranche_id,
        }
        if include_id:
            value["continuation_id"] = self.continuation_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrancheContinuation":
        item = dict(value)
        if item.pop("schema", None) != TRANCHE_CONTINUATION_SCHEMA:
            raise TrancheError("unsupported tranche continuation schema")
        observed = item.pop("continuation_id", None)
        continuation = cls(**item)
        if continuation.continuation_id != observed:
            raise TrancheError("tranche continuation identity mismatch")
        return continuation


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json_bytes(existing) != canonical_json_bytes(dict(value)):
            raise TrancheError(f"immutable tranche artifact disagrees at {path}")
        return
    _atomic_json(path, value)


class TrancheStore:
    """Durable tranche and continuation records beneath the Trainer store."""

    def __init__(self, trainer_root: Path | str) -> None:
        self.root = Path(trainer_root).resolve()
        self.tranches_dir = self.root / "tranches"
        self.continuations_dir = self.root / "tranches" / "continuations"

    # -- tranche records -------------------------------------------------

    def write_tranche(self, tranche: ResourceTranche) -> Path:
        if not isinstance(tranche, ResourceTranche):
            raise TrancheError("write_tranche requires ResourceTranche")
        path = self.tranches_dir / f"{tranche.tranche_id}.json"
        _immutable_json(path, tranche.to_canonical_dict())
        return path

    def read_tranche(self, tranche_id: str) -> ResourceTranche:
        path = self.tranches_dir / f"{_content_id(tranche_id, 'tranche_id')}.json"
        if not path.is_file():
            raise TrancheError(f"resource tranche record is missing: {tranche_id}")
        return ResourceTranche.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def tranches_for(self, module_id: str, candidate_generation_id: str) -> tuple[ResourceTranche, ...]:
        records = []
        for path in sorted(self.tranches_dir.glob("*.json")):
            body = json.loads(path.read_text(encoding="utf-8"))
            if body.get("module_id") == module_id and body.get("candidate_generation_id") == candidate_generation_id:
                records.append(ResourceTranche.from_mapping(body))
        return tuple(sorted(records, key=lambda item: (item.base_global_step, item.tranche_id)))

    # -- continuation receipts -------------------------------------------

    def write_continuation(self, continuation: TrancheContinuation) -> Path:
        if not isinstance(continuation, TrancheContinuation):
            raise TrancheError("write_continuation requires TrancheContinuation")
        tranche = self.read_tranche(continuation.tranche_id)
        expected = (
            tranche.module_id,
            tranche.candidate_generation_id,
            tranche.plan_id,
            tranche.learning_policy_id,
            tranche.parent_bundle_id,
            tranche.base_global_step,
        )
        observed = (
            continuation.module_id,
            continuation.candidate_generation_id,
            continuation.plan_id,
            continuation.learning_policy_id,
            continuation.parent_bundle_id,
            continuation.parent_global_step,
        )
        if observed != expected:
            raise TrancheError("continuation receipt disagrees with its resource tranche")
        if continuation.prior_tranche_id is not None:
            prior = self.read_tranche(continuation.prior_tranche_id)
            if (
                prior.module_id != continuation.module_id
                or prior.candidate_generation_id != continuation.candidate_generation_id
                or prior.final_global_step != continuation.parent_global_step
            ):
                raise TrancheError("prior tranche does not reach this continuation parent")
        path = self.continuations_dir / f"{continuation.continuation_id}.json"
        _immutable_json(path, continuation.to_canonical_dict())
        return path

    def read_continuation(self, continuation_id: str) -> TrancheContinuation:
        path = self.continuations_dir / f"{_content_id(continuation_id, 'continuation_id')}.json"
        if not path.is_file():
            raise TrancheError(f"tranche continuation record is missing: {continuation_id}")
        return TrancheContinuation.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def continuations_for(self, module_id: str, candidate_generation_id: str) -> tuple[TrancheContinuation, ...]:
        records = []
        for path in sorted(self.continuations_dir.glob("*.json")):
            body = json.loads(path.read_text(encoding="utf-8"))
            if body.get("module_id") == module_id and body.get("candidate_generation_id") == candidate_generation_id:
                records.append(TrancheContinuation.from_mapping(body))
        return tuple(sorted(records, key=lambda item: item.parent_global_step))


__all__ = [
    "RESOURCE_TRANCHE_SCHEMA",
    "TRANCHE_CONTINUATION_SCHEMA",
    "ResourceTranche",
    "TrancheContinuation",
    "TrancheError",
    "TrancheStore",
]
