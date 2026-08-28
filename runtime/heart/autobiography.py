"""Exact autobiographical deposits made at the Heart commit boundary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from runtime.dormant import (
    DormantExperienceStore,
    ExperienceImportManifest,
    ExperienceRecord,
)
from runtime.field import LogicalRegion, canonical_json_bytes

from .durable_ingress import IngressRecord

LIVE_INGRESS_SOURCE_SCHEMA = "axon-heart-lived-ingress-source-v1"
LIVE_EVENT_SOURCE_SCHEMA = "axon-heart-lived-event-source-v1"


@dataclass(frozen=True, slots=True)
class HeartAutobiographyReceipt:
    event_id: str
    record_id: str
    import_id: str


class HeartAutobiography:
    """Freezes every accepted external event as exact Dormant experience.

    The caller must establish canonical acceptance before invoking this class.
    Publication is content-addressed, so recovery after a commit-before-ack
    crash produces the same source, record, and import identities.
    """

    def __init__(self, state_root: Path | str) -> None:
        self.store = DormantExperienceStore(state_root)

    @staticmethod
    def _source_value(record: IngressRecord) -> dict[str, object]:
        return {
            "schema": LIVE_INGRESS_SOURCE_SCHEMA,
            "event_id": record.event_id,
            "valve_id": record.valve_id,
            "valve_version": record.valve_version,
            "source_id": record.source_id,
            "envelope_type": record.envelope_type,
            "payload": record.payload,
            "provenance": record.provenance,
            "enqueued_at": record.enqueued_at,
        }

    def deposit_ingress(
        self,
        record: IngressRecord,
        *,
        canonical_region: LogicalRegion,
    ) -> HeartAutobiographyReceipt:
        if not isinstance(record, IngressRecord):
            raise TypeError("deposit_ingress requires an IngressRecord")
        if not isinstance(canonical_region, LogicalRegion):
            raise TypeError("canonical_region must be a LogicalRegion")
        source_name = f"heart_ingress:{record.event_id}"
        source_bytes = canonical_json_bytes(self._source_value(record)) + b"\n"
        binding = self.store.publish_inline_source_snapshot(
            source_name=source_name,
            original_path=f"axon://heart/ingress/{record.event_id}",
            exact_bytes=source_bytes,
        )
        experience = ExperienceRecord(
            record_kind="accepted_ingress",
            source_name=binding.source_name,
            source_sha256=binding.sha256,
            source_pointer=f"heart-ingress-event:{record.event_id}",
            sequence=0,
            exact_text=record.payload,
            payload={
                "event_id": record.event_id,
                "valve_id": record.valve_id,
                "valve_version": record.valve_version,
                "source_id": record.source_id,
                "envelope_type": record.envelope_type,
                "provenance": record.provenance,
                "enqueued_at": record.enqueued_at,
                "canonical_region": canonical_region.value,
            },
            occurred_at=record.enqueued_at,
            evidence_class="accepted_lived_evidence",
            lifecycle="accepted",
        )
        manifest: ExperienceImportManifest = self.store.publish_import(
            (experience,),
            sources=(binding,),
            label=f"live-heart-ingress:{record.event_id}",
        )
        return HeartAutobiographyReceipt(
            event_id=record.event_id,
            record_id=experience.record_id,
            import_id=manifest.import_id,
        )

    def deposit_event(
        self,
        *,
        event_id: str,
        record_kind: str,
        exact_text: str,
        payload: Mapping[str, Any],
        occurred_at: str,
        evidence_class: str = "accepted_lived_evidence",
        lifecycle: str = "accepted",
    ) -> HeartAutobiographyReceipt:
        """Deposit any completed runtime event through the exact evidence path."""

        for name, value in (("event_id", event_id), ("record_kind", record_kind)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(exact_text, str):
            raise TypeError("exact_text must be a string")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        source_value = {
            "schema": LIVE_EVENT_SOURCE_SCHEMA,
            "event_id": event_id,
            "record_kind": record_kind,
            "exact_text": exact_text,
            "payload": dict(payload),
            "occurred_at": str(occurred_at),
            "evidence_class": str(evidence_class),
            "lifecycle": str(lifecycle),
        }
        source_name = f"heart_event:{event_id}"
        binding = self.store.publish_inline_source_snapshot(
            source_name=source_name,
            original_path=f"axon://heart/events/{event_id}",
            exact_bytes=canonical_json_bytes(source_value) + b"\n",
        )
        experience = ExperienceRecord(
            record_kind=record_kind,
            source_name=binding.source_name,
            source_sha256=binding.sha256,
            source_pointer=f"heart-event:{event_id}",
            sequence=0,
            exact_text=exact_text,
            payload={"event_id": event_id, **dict(payload)},
            occurred_at=str(occurred_at),
            evidence_class=evidence_class,
            lifecycle=lifecycle,
        )
        manifest = self.store.publish_import(
            (experience,),
            sources=(binding,),
            label=f"live-heart-event:{event_id}",
        )
        return HeartAutobiographyReceipt(
            event_id=event_id,
            record_id=experience.record_id,
            import_id=manifest.import_id,
        )

    def deposit_reasoning_episode(
        self,
        *,
        event_id: str,
        response_text: str,
        payload: Mapping[str, Any],
        occurred_at: str,
    ) -> HeartAutobiographyReceipt:
        return self.deposit_event(
            event_id=event_id,
            record_kind="runtime_reasoning_episode",
            exact_text=response_text,
            payload=payload,
            occurred_at=occurred_at,
        )

    def deposit_episode_outcome(
        self,
        *,
        event_id: str,
        episode_event_id: str,
        outcome_quality: str,
        evidence_ids: tuple[str, ...],
        detail: str,
        occurred_at: str,
    ) -> HeartAutobiographyReceipt:
        if not evidence_ids:
            raise ValueError("episode outcomes require explicit evidence_ids")
        return self.deposit_event(
            event_id=event_id,
            record_kind="runtime_episode_outcome",
            exact_text=detail,
            payload={
                "episode_event_id": episode_event_id,
                "outcome_quality": outcome_quality,
                "evidence_ids": list(evidence_ids),
            },
            occurred_at=occurred_at,
            evidence_class="explicit_outcome_evidence",
        )

    def deposit_tool_invocation(
        self,
        *,
        event_id: str,
        exact_text: str,
        payload: Mapping[str, Any],
        occurred_at: str,
    ) -> HeartAutobiographyReceipt:
        return self.deposit_event(
            event_id=event_id,
            record_kind="runtime_tool_invocation",
            exact_text=exact_text,
            payload=payload,
            occurred_at=occurred_at,
        )

    def deposit_trainer_outcome(
        self,
        *,
        event_id: str,
        exact_text: str,
        payload: Mapping[str, Any],
        occurred_at: str,
    ) -> HeartAutobiographyReceipt:
        return self.deposit_event(
            event_id=event_id,
            record_kind="runtime_trainer_outcome",
            exact_text=exact_text,
            payload=payload,
            occurred_at=occurred_at,
        )


__all__ = [
    "LIVE_EVENT_SOURCE_SCHEMA",
    "LIVE_INGRESS_SOURCE_SCHEMA",
    "HeartAutobiography",
    "HeartAutobiographyReceipt",
]
