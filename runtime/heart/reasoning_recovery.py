"""Crash-consistent preparation and recovery for reasoning autobiography."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from runtime.field import (
    CANONICAL_REGION_ORDER,
    CanonicalStateBranch,
    CompiledD64Field,
    apply_delta,
    canonical_json_bytes,
    canonical_sha256,
    field_delta_from_canonical_dict,
    snapshot_from_canonical_dict,
)
from runtime.soul import SoulCommitReceipt, SoulStore, SoulTransition

from .authority import AuthorityGrant
from .tick import TickIdentity
from .transaction import HeartCommit

REASONING_RECOVERY_PREPARATION_SCHEMA = "axon-reasoning-recovery-preparation-v1"
REASONING_RECOVERY_COMPLETION_SCHEMA = "axon-reasoning-recovery-completion-v1"
_REASONING_CIRCULATION_SCHEMA = "axon-reasoning-circulation-v2"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"reasoning recovery resource must be an object: {path}")
    return value


def attention_view_from_surface(
    surface: CompiledD64Field,
    *,
    view_id: str,
) -> dict[str, Any]:
    return {
        "schema": "axon-runtime-attention-view-v1",
        "source_field_id": surface.source_field_id,
        "source_tick_id": surface.source_tick_id,
        "view_id": view_id,
        "rail_id": surface.rail_id,
        "regions": {
            region.value: [
                {"start": start, "end": end}
                for start, end in sorted(
                    {
                        (
                            address.attended_interval_start,
                            address.attended_interval_end,
                        )
                        for address in surface.region_character_addresses(region)
                    }
                )
            ]
            for region in CANONICAL_REGION_ORDER
        },
    }


@dataclass(frozen=True, slots=True)
class RecoveredReasoningEpisode:
    preparation_id: str
    result_id: str
    event_id: str
    response_text: str
    occurred_at: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ReasoningRecoveryPreparation:
    body: Mapping[str, Any]
    preparation_id: str = field(init=False)

    def __post_init__(self) -> None:
        body = dict(self.body)
        if body.get("schema") != REASONING_RECOVERY_PREPARATION_SCHEMA:
            raise ValueError("unsupported reasoning recovery preparation schema")
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "preparation_id", canonical_sha256(body))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {**dict(self.body), "preparation_id": self.preparation_id}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReasoningRecoveryPreparation":
        body = dict(value)
        observed = body.pop("preparation_id", None)
        preparation = cls(body)
        if observed != preparation.preparation_id:
            raise ValueError("reasoning recovery preparation identity mismatch")
        return preparation


class ReasoningAutobiographyRecoveryStore:
    """Immutable precommit spool; completion markers never erase evidence."""

    def __init__(self, state_root: Path | str) -> None:
        self.root = Path(state_root).resolve(strict=False) / "active" / "heart" / "reasoning_recovery"
        self.prepared_dir = self.root / "prepared"
        self.completed_dir = self.root / "completed"

    def prepare(self, body: Mapping[str, Any]) -> ReasoningRecoveryPreparation:
        preparation = ReasoningRecoveryPreparation(body)
        path = self.prepared_dir / f"{preparation.preparation_id}.json"
        value = preparation.to_canonical_dict()
        if path.exists():
            if canonical_json_bytes(_read_json(path)) != canonical_json_bytes(value):
                raise ValueError("reasoning recovery preparation identity collision")
        else:
            _atomic_json(path, value)
        return preparation

    def pending(self) -> tuple[ReasoningRecoveryPreparation, ...]:
        preparations = []
        for path in sorted(self.prepared_dir.glob("*.json")):
            item = ReasoningRecoveryPreparation.from_mapping(_read_json(path))
            if not (self.completed_dir / f"{item.preparation_id}.json").exists():
                preparations.append(item)
        return tuple(preparations)

    def for_materialized_delta(self, delta_id: str) -> ReasoningRecoveryPreparation:
        matches = tuple(
            item
            for item in self.pending()
            if item.body.get("materialized_delta_id") == delta_id
        )
        if len(matches) != 1:
            raise ValueError("reasoning result does not bind exactly one pending recovery preparation")
        return matches[0]

    def mark_completed(
        self,
        preparation_id: str,
        *,
        result_id: str,
        event_id: str,
        record_id: str,
        import_id: str,
    ) -> None:
        body = {
            "schema": REASONING_RECOVERY_COMPLETION_SCHEMA,
            "preparation_id": preparation_id,
            "result_id": result_id,
            "event_id": event_id,
            "record_id": record_id,
            "import_id": import_id,
        }
        body["completion_id"] = canonical_sha256(body)
        path = self.completed_dir / f"{preparation_id}.json"
        if path.exists():
            if canonical_json_bytes(_read_json(path)) != canonical_json_bytes(body):
                raise ValueError("reasoning recovery completion disagrees with durable evidence")
            return
        _atomic_json(path, body)

    @staticmethod
    def _is_canonical_ancestor(branch: CanonicalStateBranch, field_id: str) -> bool:
        cursor = branch.load_head()
        while True:
            if cursor.field_id == field_id:
                return True
            if cursor.parent_field_id is None:
                return False
            cursor = branch.load_snapshot(cursor.parent_field_id)

    def reconstruct_if_committed(
        self,
        preparation: ReasoningRecoveryPreparation,
        *,
        branch: CanonicalStateBranch,
        soul_store: SoulStore,
        conversation_id: str,
    ) -> RecoveredReasoningEpisode | None:
        body = preparation.body
        base = snapshot_from_canonical_dict(body["pre_action_field"])
        materialized = field_delta_from_canonical_dict(body["materialized_delta"])
        successor = apply_delta(
            base,
            materialized,
            permitted_regions=AuthorityGrant.consolidator().governed_regions,
        )
        if not self._is_canonical_ancestor(branch, successor.field_id):
            return None

        transition = SoulTransition.from_mapping(body["consolidator_soul_transition"])
        soul_branch = soul_store.branch(transition.core_id)
        try:
            consolidator_receipt = soul_branch.finalize_transition(
                transition.transition_id,
                commit_binding=f"canonical-field:{successor.field_id}",
            )
        except Exception:
            consolidator_receipt = soul_branch.load_receipt(transition.transition_id)
            if consolidator_receipt.commit_binding != f"canonical-field:{successor.field_id}":
                raise

        prior_receipts = tuple(
            SoulCommitReceipt.from_mapping(item) for item in body["prior_soul_receipts"]
        )
        receipts = (*prior_receipts, consolidator_receipt)
        initial_souls = dict(body["initial_souls"])
        lineages = []
        for core_id in sorted(initial_souls):
            cursor = str(initial_souls[core_id])
            receipt_ids = []
            for receipt in receipts:
                if receipt.core_id != core_id:
                    continue
                if receipt.before_soul_id != cursor:
                    raise ValueError("recovered private-soul trajectory is discontinuous")
                cursor = receipt.after_soul_id
                receipt_ids.append(receipt.receipt_id)
            lineage = {
                "schema": "axon-reasoning-soul-lineage-v1",
                "core_id": core_id,
                "initial_soul_id": str(initial_souls[core_id]),
                "final_soul_id": cursor,
                "transition_receipt_ids": receipt_ids,
            }
            lineage["lineage_id"] = canonical_sha256(lineage)
            lineages.append(lineage)

        identity = dict(dict(body["image"])["identity"])
        tick = TickIdentity(
            tick_sequence=int(identity["tick_sequence"]),
            heartbeat_id=int(identity["heartbeat_id"]),
            base_field_id=str(identity["base_field_id"]),
            base_tick_id=int(identity["base_tick_id"]),
        )
        circulation_metadata = {
            **dict(body["circulation_metadata"]),
            "recovery_preparation_id": preparation.preparation_id,
        }
        commit = HeartCommit(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            successor=successor,
            delta=materialized,
            grant=AuthorityGrant.consolidator(),
            tick=tick,
            valve_provenance={
                "authority_class": "consolidator",
                "tick_uid": tick.tick_uid,
                **circulation_metadata,
            },
        )
        circulation = {
            "schema": _REASONING_CIRCULATION_SCHEMA,
            "image": body["image"],
            "first_records": body["first_records"],
            "refined_records": body["refined_records"],
            "first_workspace": body["first_workspace"],
            "refined_workspace": body["refined_workspace"],
            "first_emissions": body["first_emissions"],
            "refined_emissions": body["refined_emissions"],
            "soul_transition_receipts": [item.to_canonical_dict() for item in receipts],
            "soul_lineages": lineages,
            "consolidator_core_id": body["consolidator_core_id"],
            "consolidator_emission": body["consolidator_emission"],
            "source_delta": body["source_delta"],
            "materialized_delta": body["materialized_delta"],
            "finalization_receipt": body["finalization_receipt"],
            "commit": commit.to_canonical_dict(),
        }
        result_id = canonical_sha256(circulation)
        circulation["result_id"] = result_id
        response_text = successor.region("response_draft").text
        payload = {
            "schema": "axon-runtime-reasoning-episode-v3",
            "conversation_id": conversation_id,
            "pre_action_field": base.to_dict(),
            "attention_view": body["attention_view"],
            "circulation": circulation,
            "response_text_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            "outcome_quality": "observed",
            "outcome_evidence_ids": [],
        }
        return RecoveredReasoningEpisode(
            preparation_id=preparation.preparation_id,
            result_id=result_id,
            event_id=f"reasoning-{result_id}",
            response_text=response_text,
            occurred_at=str(body["occurred_at"]),
            payload=payload,
        )


__all__ = [
    "REASONING_RECOVERY_COMPLETION_SCHEMA",
    "REASONING_RECOVERY_PREPARATION_SCHEMA",
    "ReasoningAutobiographyRecoveryStore",
    "ReasoningRecoveryPreparation",
    "RecoveredReasoningEpisode",
    "attention_view_from_surface",
]
