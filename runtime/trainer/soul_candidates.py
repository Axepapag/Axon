"""Isolated candidate-Soul branches for runtime-faithful learning.

The Trainer may fork a core's exact live Soul into a candidate workspace, but
it never mutates or interprets the live payload.  If lived experience advances
the live branch while training runs, promotion is blocked until the candidate
has replayed the exact intervening receipt chain and passed its gates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from runtime.field import canonical_sha256
from runtime.soul import (
    SoulBranch,
    SoulIntegrityError,
    SoulSnapshot,
    SoulStore,
    SoulTemperature,
)

SOUL_CANDIDATE_MANIFEST_SCHEMA = "axon-trainer-soul-candidate-manifest-v1"
SOUL_CANDIDATE_PROMOTION_PLAN_SCHEMA = "axon-trainer-soul-promotion-plan-v1"


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()


def _safe_component(value: str, label: str) -> str:
    value = _required(value, label)
    if value in {".", ".."} or any(char in value for char in '\\/:*?"<>|'):
        raise ValueError(f"{label} is not a safe path component")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class CandidateSoulManifest:
    candidate_id: str
    core_id: str
    branch_id: str
    architecture_id: str
    base_live_parameter_generation: str
    candidate_parameter_generation: str
    base_live_soul_id: str
    base_live_generation: int
    candidate_initial_soul_id: str
    deep_cold_payload_sha256: str
    runtime_episode_session_id: str
    whole_episode_split: str
    soul_trajectory_ids: tuple[str, ...]
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "candidate_id",
            "core_id",
            "branch_id",
            "architecture_id",
            "base_live_parameter_generation",
            "candidate_parameter_generation",
            "runtime_episode_session_id",
            "whole_episode_split",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        for label in (
            "base_live_soul_id",
            "candidate_initial_soul_id",
            "deep_cold_payload_sha256",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{label} must be a SHA256 identity")
        if (
            isinstance(self.base_live_generation, bool)
            or not isinstance(self.base_live_generation, int)
            or self.base_live_generation < 0
        ):
            raise ValueError("base_live_generation must be non-negative")
        trajectories = tuple(sorted(set(map(str, self.soul_trajectory_ids))))
        if not trajectories or any(len(item) != 64 for item in trajectories):
            raise ValueError("candidate Soul requires one or more causal trajectory identities")
        object.__setattr__(self, "soul_trajectory_ids", trajectories)
        object.__setattr__(self, "manifest_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": SOUL_CANDIDATE_MANIFEST_SCHEMA,
            "candidate_id": self.candidate_id,
            "core_id": self.core_id,
            "branch_id": self.branch_id,
            "architecture_id": self.architecture_id,
            "base_live_parameter_generation": self.base_live_parameter_generation,
            "candidate_parameter_generation": self.candidate_parameter_generation,
            "base_live_soul_id": self.base_live_soul_id,
            "base_live_generation": self.base_live_generation,
            "candidate_initial_soul_id": self.candidate_initial_soul_id,
            "deep_cold_payload_sha256": self.deep_cold_payload_sha256,
            "runtime_episode_session_id": self.runtime_episode_session_id,
            "whole_episode_split": self.whole_episode_split,
            "soul_trajectory_ids": list(self.soul_trajectory_ids),
        }
        if include_id:
            value["manifest_id"] = self.manifest_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateSoulManifest":
        item = dict(value)
        if item.pop("schema", None) != SOUL_CANDIDATE_MANIFEST_SCHEMA:
            raise SoulIntegrityError("unsupported candidate Soul manifest schema")
        manifest_id = item.pop("manifest_id", None)
        try:
            item["soul_trajectory_ids"] = tuple(item["soul_trajectory_ids"])
            manifest = cls(**item)
        except (KeyError, TypeError, ValueError) as exc:
            raise SoulIntegrityError("candidate Soul manifest is invalid") from exc
        if manifest.manifest_id != manifest_id:
            raise SoulIntegrityError("candidate Soul manifest identity mismatch")
        return manifest


@dataclass(frozen=True, slots=True)
class CandidateSoulPromotionPlan:
    manifest_id: str
    core_id: str
    candidate_id: str
    base_live_soul_id: str
    observed_live_soul_id: str
    candidate_soul_id: str
    replay_receipt_ids: tuple[str, ...]
    status: str
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.status not in {"exact_base_ready_for_gates", "live_replay_required"}:
            raise ValueError("unsupported candidate Soul promotion status")
        receipts = tuple(self.replay_receipt_ids)
        if self.status == "exact_base_ready_for_gates" and receipts:
            raise ValueError("exact-base candidate cannot require live replay")
        if self.status == "live_replay_required" and not receipts:
            raise ValueError("advanced live Soul requires an exact replay chain")
        object.__setattr__(self, "replay_receipt_ids", receipts)
        object.__setattr__(self, "plan_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": SOUL_CANDIDATE_PROMOTION_PLAN_SCHEMA,
            "manifest_id": self.manifest_id,
            "core_id": self.core_id,
            "candidate_id": self.candidate_id,
            "base_live_soul_id": self.base_live_soul_id,
            "observed_live_soul_id": self.observed_live_soul_id,
            "candidate_soul_id": self.candidate_soul_id,
            "replay_receipt_ids": list(self.replay_receipt_ids),
            "status": self.status,
        }
        if include_id:
            value["plan_id"] = self.plan_id
        return value


class CandidateSoulWorkspace:
    """Trainer-owned factory and auditor for private candidate Soul branches."""

    def __init__(self, state_root: Path | str) -> None:
        self.state_root = Path(state_root).resolve()
        self.root = self.state_root / "training" / "soul_candidates"
        self.live_store = SoulStore.active(self.state_root)

    def branch(self, candidate_id: str, core_id: str) -> SoulBranch:
        safe_candidate = _safe_component(candidate_id, "candidate_id")
        safe_core = _safe_component(core_id, "core_id")
        return SoulBranch(
            self.root / safe_candidate / safe_core / "branch",
            core_id=safe_core,
            branch_id=safe_candidate,
        )

    def prepare(
        self,
        *,
        candidate_id: str,
        core_id: str,
        runtime_episode_session_id: str,
        whole_episode_split: str,
        soul_trajectory_ids: Iterable[str],
        candidate_parameter_generation: str | None = None,
    ) -> CandidateSoulManifest:
        candidate_id = _safe_component(candidate_id, "candidate_id")
        core_id = _safe_component(core_id, "core_id")
        path = self.root / candidate_id / core_id / "manifest.json"
        if path.exists():
            existing = CandidateSoulManifest.from_mapping(
                json.loads(path.read_text(encoding="utf-8"))
            )
            requested_trajectories = tuple(sorted(set(map(str, soul_trajectory_ids))))
            if (
                existing.runtime_episode_session_id != runtime_episode_session_id
                or existing.whole_episode_split != whole_episode_split
                or existing.soul_trajectory_ids != requested_trajectories
                or (
                    candidate_parameter_generation is not None
                    and existing.candidate_parameter_generation
                    != candidate_parameter_generation
                )
            ):
                raise SoulIntegrityError("candidate Soul manifest is immutable")
            self.branch(candidate_id, core_id).load_head()
            return existing
        live = self.live_store.branch(core_id)
        live_head = live.load_head()
        candidate = self.branch(candidate_id, core_id)
        candidate_generation = (
            live_head.parameter_generation
            if candidate_parameter_generation is None
            else _required(candidate_parameter_generation, "candidate_parameter_generation")
        )
        if candidate_generation == live_head.parameter_generation:
            fork = live.fork_to(candidate)
        else:
            fork = candidate.initialize(
                architecture_id=live_head.architecture_id,
                parameter_generation=candidate_generation,
                snapshot=SoulSnapshot(
                    core_id=live_head.core_id,
                    architecture_id=live_head.architecture_id,
                    parameter_generation=candidate_generation,
                    generation=0,
                    parent_soul_id=None,
                    layers=live_head.layers,
                ),
                forked_from={
                    "core_id": live_head.core_id,
                    "branch_id": live.branch_id,
                    "soul_id": live_head.soul_id,
                    "generation": live_head.generation,
                    "parameter_generation": live_head.parameter_generation,
                    "inheritance": "exact_layers_rebound_to_candidate_parameter_generation",
                },
            )
        manifest = CandidateSoulManifest(
            candidate_id=candidate_id,
            core_id=core_id,
            branch_id=candidate.branch_id,
            architecture_id=fork.architecture_id,
            base_live_parameter_generation=live_head.parameter_generation,
            candidate_parameter_generation=fork.parameter_generation,
            base_live_soul_id=live_head.soul_id,
            base_live_generation=live_head.generation,
            candidate_initial_soul_id=fork.soul_id,
            deep_cold_payload_sha256=fork.layer(SoulTemperature.DEEP_COLD).payload_sha256,
            runtime_episode_session_id=runtime_episode_session_id,
            whole_episode_split=whole_episode_split,
            soul_trajectory_ids=tuple(soul_trajectory_ids),
        )
        _atomic_json(path, manifest.to_canonical_dict())
        return manifest

    def promotion_plan(self, manifest: CandidateSoulManifest) -> CandidateSoulPromotionPlan:
        live = self.live_store.branch(manifest.core_id)
        candidate = self.branch(manifest.candidate_id, manifest.core_id)
        live_head = live.load_head()
        candidate_head = candidate.load_head()
        if (
            candidate_head.architecture_id != manifest.architecture_id
            or candidate_head.parameter_generation != manifest.candidate_parameter_generation
        ):
            raise SoulIntegrityError("candidate Soul architecture changed after preparation")
        if live_head.soul_id == manifest.base_live_soul_id:
            receipts = ()
            status = "exact_base_ready_for_gates"
        else:
            receipts = tuple(
                receipt.receipt_id
                for receipt in live.receipts_after(manifest.base_live_soul_id)
            )
            status = "live_replay_required"
        return CandidateSoulPromotionPlan(
            manifest_id=manifest.manifest_id,
            core_id=manifest.core_id,
            candidate_id=manifest.candidate_id,
            base_live_soul_id=manifest.base_live_soul_id,
            observed_live_soul_id=live_head.soul_id,
            candidate_soul_id=candidate_head.soul_id,
            replay_receipt_ids=receipts,
            status=status,
        )


__all__ = [
    "SOUL_CANDIDATE_MANIFEST_SCHEMA",
    "SOUL_CANDIDATE_PROMOTION_PLAN_SCHEMA",
    "CandidateSoulManifest",
    "CandidateSoulPromotionPlan",
    "CandidateSoulWorkspace",
]
