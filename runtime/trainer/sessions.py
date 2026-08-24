"""Durable training sessions compiled from Axon's exact lived experience."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from runtime.dormant import DormantExperienceStore, ExperienceRecord
from runtime.field import canonical_sha256


TRAINER_SESSION_POLICY_SCHEMA = "axon-trainer-session-policy-v1"
TRAINER_SESSION_EXAMPLE_SCHEMA = "axon-trainer-session-example-v1"
TRAINER_SESSION_SCHEMA = "axon-trainer-session-v1"


def _split(record_id: str) -> str:
    bucket = int(record_id[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "heldout"
    return "regression"


@dataclass(frozen=True, slots=True)
class TrainerSessionPolicy:
    session_kind: str
    target_organ: str
    source_record_kinds: tuple[str, ...]
    evidence_basis: str
    serving_promotion_eligible: bool = False
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("session_kind", "target_organ", "evidence_basis"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        kinds = tuple(sorted(set(str(item) for item in self.source_record_kinds)))
        if not kinds or any(not item for item in kinds):
            raise ValueError("source_record_kinds must be non-empty")
        object.__setattr__(self, "source_record_kinds", kinds)
        object.__setattr__(self, "policy_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": TRAINER_SESSION_POLICY_SCHEMA,
            "session_kind": self.session_kind,
            "target_organ": self.target_organ,
            "source_record_kinds": list(self.source_record_kinds),
            "evidence_basis": self.evidence_basis,
            "serving_promotion_eligible": self.serving_promotion_eligible,
            "split_policy": "sha256-bucket-v1:80-train-10-heldout-10-regression",
            "source_capacity": "complete-record-no-character-truncation",
        }
        if include_id:
            value["policy_id"] = self.policy_id
        return value


@dataclass(frozen=True, slots=True)
class TrainerSessionExample:
    source_record_ids: tuple[str, ...]
    input_record_id: str
    target_record_id: str
    input_text_sha256: str
    target_text_sha256: str
    split: str
    context_source_name: str
    context_sequence_start: int
    context_sequence_end: int
    target_basis: str
    example_id: str = field(init=False)

    def __post_init__(self) -> None:
        ids = tuple(self.source_record_ids)
        if not ids or self.input_record_id not in ids or self.target_record_id not in ids:
            raise ValueError("session example source identities are incomplete")
        if self.split not in {"train", "heldout", "regression"}:
            raise ValueError("unsupported session split")
        if self.context_sequence_start < 0 or self.context_sequence_end < self.context_sequence_start:
            raise ValueError("invalid complete-context sequence range")
        for name in ("input_text_sha256", "target_text_sha256"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a SHA256 digest")
        object.__setattr__(self, "example_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": TRAINER_SESSION_EXAMPLE_SCHEMA,
            "source_record_ids": list(self.source_record_ids),
            "input_record_id": self.input_record_id,
            "target_record_id": self.target_record_id,
            "input_text_sha256": self.input_text_sha256,
            "target_text_sha256": self.target_text_sha256,
            "split": self.split,
            "context": {
                "source_name": self.context_source_name,
                "sequence_start": self.context_sequence_start,
                "sequence_end": self.context_sequence_end,
                "complete_prefix": self.context_sequence_start == 0,
            },
            "target_basis": self.target_basis,
        }
        if include_id:
            value["example_id"] = self.example_id
        return value


@dataclass(frozen=True, slots=True)
class TrainerSessionManifest:
    policy: TrainerSessionPolicy
    source_import_ids: tuple[str, ...]
    examples: tuple[TrainerSessionExample, ...]
    excluded_counts: tuple[tuple[str, int], ...]
    session_id: str = field(init=False)

    def __post_init__(self) -> None:
        imports = tuple(sorted(set(str(item) for item in self.source_import_ids)))
        if not imports or any(len(item) != 64 for item in imports):
            raise ValueError("source_import_ids must contain content identities")
        object.__setattr__(self, "source_import_ids", imports)
        examples = tuple(self.examples)
        if not examples or len({item.example_id for item in examples}) != len(examples):
            raise ValueError("session examples must be non-empty and unique")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "excluded_counts", tuple(sorted(self.excluded_counts)))
        object.__setattr__(self, "session_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def split_counts(self) -> tuple[tuple[str, int], ...]:
        counts = {name: 0 for name in ("train", "heldout", "regression")}
        for example in self.examples:
            counts[example.split] += 1
        return tuple(sorted(counts.items()))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": TRAINER_SESSION_SCHEMA,
            "policy": self.policy.to_canonical_dict(),
            "source_import_ids": list(self.source_import_ids),
            "example_count": len(self.examples),
            "split_counts": [list(item) for item in self.split_counts],
            "excluded_counts": [list(item) for item in self.excluded_counts],
            "examples": [item.to_canonical_dict() for item in self.examples],
        }
        if include_id:
            value["session_id"] = self.session_id
        return value


class LivedExperienceSessionCompiler:
    """Build source-reference sessions without duplicating or truncating memories."""

    def __init__(self, experience: DormantExperienceStore) -> None:
        if not isinstance(experience, DormantExperienceStore):
            raise TypeError("experience must be a DormantExperienceStore")
        self.experience = experience

    def _records(self, import_ids: Sequence[str] | None) -> tuple[tuple[str, ...], tuple[ExperienceRecord, ...]]:
        selected = tuple(import_ids) if import_ids is not None else tuple(
            item.import_id for item in self.experience.import_manifests()
        )
        if not selected:
            raise ValueError("no exact Dormant experience imports are available")
        return selected, tuple(self.experience.iter_records(selected))

    @staticmethod
    def _example(
        input_record: ExperienceRecord,
        target_record: ExperienceRecord,
        *,
        target_basis: str,
        context_start: int = 0,
    ) -> TrainerSessionExample:
        return TrainerSessionExample(
            source_record_ids=tuple(dict.fromkeys((input_record.record_id, target_record.record_id))),
            input_record_id=input_record.record_id,
            target_record_id=target_record.record_id,
            input_text_sha256=input_record.exact_text_sha256,
            target_text_sha256=target_record.exact_text_sha256,
            split=_split(target_record.record_id),
            context_source_name=input_record.source_name,
            context_sequence_start=context_start,
            context_sequence_end=target_record.sequence + 1,
            target_basis=target_basis,
        )

    def compile_heart_grounding(
        self, import_ids: Sequence[str] | None = None
    ) -> TrainerSessionManifest:
        selected, records = self._records(import_ids)
        policy = TrainerSessionPolicy(
            session_kind="exact_lived_record_grounding",
            target_organ="heart_translation_conduction",
            source_record_kinds=tuple(sorted({item.record_kind for item in records})),
            evidence_basis="exact recovered record roundtrip",
            serving_promotion_eligible=False,
        )
        examples: list[TrainerSessionExample] = []
        empty = 0
        for record in records:
            if not record.exact_text:
                empty += 1
                continue
            examples.append(
                self._example(
                    record,
                    record,
                    target_basis="exact source-record identity roundtrip",
                    context_start=record.sequence,
                )
            )
        return TrainerSessionManifest(
            policy=policy,
            source_import_ids=selected,
            examples=tuple(examples),
            excluded_counts=(("empty_exact_text_preserved_but_not_grounding_target", empty),) if empty else (),
        )

    def compile_observed_conversation(
        self, import_ids: Sequence[str] | None = None
    ) -> TrainerSessionManifest:
        selected, records = self._records(import_ids)
        policy = TrainerSessionPolicy(
            session_kind="observed_conversation_response",
            target_organ="reasoning_core_candidate",
            source_record_kinds=("message", "runtime_conversation_message"),
            evidence_basis="observed historical response without outcome endorsement",
            serving_promotion_eligible=False,
        )
        by_source: dict[str, list[ExperienceRecord]] = {}
        for record in records:
            if record.record_kind in policy.source_record_kinds:
                by_source.setdefault(record.source_name, []).append(record)
        examples: list[TrainerSessionExample] = []
        unpaired = 0
        for source_name in sorted(by_source):
            ordered = sorted(by_source[source_name], key=lambda item: item.sequence)
            for index, record in enumerate(ordered):
                role = str(record.payload.get("role", "")).casefold()
                if role != "user":
                    continue
                if index + 1 >= len(ordered):
                    unpaired += 1
                    continue
                response = ordered[index + 1]
                if str(response.payload.get("role", "")).casefold() != "assistant":
                    unpaired += 1
                    continue
                if not record.exact_text or not response.exact_text:
                    unpaired += 1
                    continue
                examples.append(
                    self._example(
                        record,
                        response,
                        target_basis="observed assistant response; correctness not inferred",
                        context_start=0,
                    )
                )
        return TrainerSessionManifest(
            policy=policy,
            source_import_ids=selected,
            examples=tuple(examples),
            excluded_counts=(("unpaired_or_empty_user_message", unpaired),) if unpaired else (),
        )


__all__ = [
    "TRAINER_SESSION_POLICY_SCHEMA",
    "TRAINER_SESSION_EXAMPLE_SCHEMA",
    "TRAINER_SESSION_SCHEMA",
    "TrainerSessionPolicy",
    "TrainerSessionExample",
    "TrainerSessionManifest",
    "LivedExperienceSessionCompiler",
]
