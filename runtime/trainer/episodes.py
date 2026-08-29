"""Runtime-faithful Trainer episodes reconstructed from lived circulation.

Historical adjacency can describe what was observed, but it cannot reproduce a
reasoning tick or establish answer quality.  These contracts accept only exact
``runtime_reasoning_episode`` deposits, keep every conversation in one split,
rebuild the pre-action canonical field and accepted delta, and treat serving
eligibility as false until explicit outcome evidence says otherwise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from runtime.dormant import DormantExperienceStore, ExperienceRecord
from runtime.field import (
    FieldDelta,
    LogicalRegion,
    SharedFieldSnapshot,
    apply_delta,
    canonical_json_bytes,
    canonical_sha256,
    field_delta_from_canonical_dict,
    snapshot_from_canonical_dict,
)
from runtime.heart import (
    AuthorityGrant,
    CategoricalTextFrame,
    ReasoningEmission,
    frame_completed_turn,
)
from runtime.soul import SoulCommitReceipt, SoulStore

RUNTIME_EPISODE_EXAMPLE_SCHEMA = "axon-runtime-episode-example-v2"
RUNTIME_EPISODE_SESSION_SCHEMA = "axon-runtime-episode-session-v2"
RUNTIME_REASONING_EPISODE_SCHEMA = "axon-runtime-reasoning-episode-v2"


class EpisodeOutcomeQuality(str, Enum):
    UNKNOWN = "unknown"
    OBSERVED = "observed"
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTED = "corrected"
    ENDORSED = "endorsed"


SERVING_QUALITIES = frozenset(
    {
        EpisodeOutcomeQuality.SUCCESS,
        EpisodeOutcomeQuality.CORRECTED,
        EpisodeOutcomeQuality.ENDORSED,
    }
)


def whole_episode_split(conversation_id: str) -> str:
    """Assign an entire conversation to one deterministic 80/10/10 split."""

    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("conversation_id must be non-empty")
    bucket = int(hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "heldout"
    return "regression"


def _validated_soul_lineages(
    circulation: Mapping[str, Any],
    soul_store: SoulStore,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    """Verify every serialized receipt and exact per-core causal soul chain."""

    try:
        raw_receipts = tuple(circulation["soul_transition_receipts"])
        raw_lineages = tuple(circulation["soul_lineages"])
    except (KeyError, TypeError) as exc:
        raise ValueError("runtime episode lacks private-soul lineage") from exc
    receipts = tuple(SoulCommitReceipt.from_mapping(item) for item in raw_receipts)
    receipt_map = {item.receipt_id: item for item in receipts}
    if len(receipt_map) != len(receipts):
        raise ValueError("runtime episode contains duplicate soul receipts")
    lineages: list[Mapping[str, Any]] = []
    consumed: set[str] = set()
    for raw in raw_lineages:
        lineage = dict(raw)
        required = {
            "schema",
            "core_id",
            "initial_soul_id",
            "final_soul_id",
            "transition_receipt_ids",
            "lineage_id",
        }
        if set(lineage) != required or lineage["schema"] != "axon-reasoning-soul-lineage-v1":
            raise ValueError("runtime soul lineage fields are invalid")
        body = dict(lineage)
        lineage_id = body.pop("lineage_id")
        if canonical_sha256(body) != lineage_id:
            raise ValueError("runtime soul lineage identity mismatch")
        core_id = str(lineage["core_id"])
        branch = soul_store.branch(core_id)
        initial = branch.load_snapshot(str(lineage["initial_soul_id"]))
        cursor = initial.soul_id
        for receipt_id in tuple(lineage["transition_receipt_ids"]):
            receipt = receipt_map.get(str(receipt_id))
            if receipt is None or receipt.core_id != core_id or receipt.before_soul_id != cursor:
                raise ValueError("runtime soul receipt chain is missing, foreign, or discontinuous")
            durable = branch.load_receipt(receipt.transition_id)
            if durable.receipt_id != receipt.receipt_id:
                raise ValueError("runtime soul receipt disagrees with durable private state")
            branch.load_snapshot(receipt.after_soul_id)
            cursor = receipt.after_soul_id
            consumed.add(receipt.receipt_id)
        if cursor != lineage["final_soul_id"]:
            raise ValueError("runtime soul lineage final snapshot mismatch")
        lineages.append(lineage)
    if consumed != set(receipt_map):
        raise ValueError("runtime episode contains unbound soul transition receipts")
    ordered = tuple(sorted(lineages, key=lambda item: str(item["core_id"])))
    return ordered, canonical_sha256(ordered)


@dataclass(frozen=True, slots=True)
class RuntimeEpisodeExample:
    source_import_id: str
    source_record_id: str
    episode_event_id: str
    conversation_id: str
    base_field_id: str
    base_tick_id: int
    circulation_result_id: str
    soul_trajectory_id: str
    response_text_sha256: str
    outcome_quality: EpisodeOutcomeQuality
    outcome_evidence_ids: tuple[str, ...]
    outcome_record_id: str | None
    split: str
    serving_promotion_eligible: bool
    example_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_import_id",
            "source_record_id",
            "episode_event_id",
            "conversation_id",
            "base_field_id",
            "circulation_result_id",
            "soul_trajectory_id",
            "response_text_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "source_import_id",
            "source_record_id",
            "base_field_id",
            "circulation_result_id",
            "soul_trajectory_id",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a content identity")
        if len(self.response_text_sha256) != 64:
            raise ValueError("response_text_sha256 must be a SHA256 digest")
        if isinstance(self.base_tick_id, bool) or not isinstance(self.base_tick_id, int) or self.base_tick_id < 0:
            raise ValueError("base_tick_id must be non-negative")
        quality = (
            self.outcome_quality
            if isinstance(self.outcome_quality, EpisodeOutcomeQuality)
            else EpisodeOutcomeQuality(self.outcome_quality)
        )
        evidence = tuple(sorted(set(map(str, self.outcome_evidence_ids))))
        object.__setattr__(self, "outcome_quality", quality)
        object.__setattr__(self, "outcome_evidence_ids", evidence)
        if self.outcome_record_id is not None and len(self.outcome_record_id) != 64:
            raise ValueError("outcome_record_id must be None or a content identity")
        if self.split not in {"train", "heldout", "regression"}:
            raise ValueError("unsupported episode split")
        eligible = quality in SERVING_QUALITIES and bool(evidence) and self.outcome_record_id is not None
        if self.serving_promotion_eligible != eligible:
            raise ValueError("serving eligibility does not match explicit outcome evidence")
        object.__setattr__(self, "example_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": RUNTIME_EPISODE_EXAMPLE_SCHEMA,
            "source_import_id": self.source_import_id,
            "source_record_id": self.source_record_id,
            "episode_event_id": self.episode_event_id,
            "conversation_id": self.conversation_id,
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "circulation_result_id": self.circulation_result_id,
            "soul_trajectory_id": self.soul_trajectory_id,
            "response_text_sha256": self.response_text_sha256,
            "outcome_quality": self.outcome_quality.value,
            "outcome_evidence_ids": list(self.outcome_evidence_ids),
            "outcome_record_id": self.outcome_record_id,
            "split": self.split,
            "serving_promotion_eligible": self.serving_promotion_eligible,
        }
        if include_id:
            value["example_id"] = self.example_id
        return value


@dataclass(frozen=True, slots=True)
class RuntimeEpisodeSessionManifest:
    source_import_ids: tuple[str, ...]
    examples: tuple[RuntimeEpisodeExample, ...]
    excluded_counts: tuple[tuple[str, int], ...] = ()
    session_id: str = field(init=False)

    def __post_init__(self) -> None:
        imports = tuple(sorted(set(map(str, self.source_import_ids))))
        if not imports or any(len(item) != 64 for item in imports):
            raise ValueError("source_import_ids must contain content identities")
        examples = tuple(self.examples)
        if not examples or len({item.example_id for item in examples}) != len(examples):
            raise ValueError("runtime episode examples must be non-empty and unique")
        if any(item.source_import_id not in imports for item in examples):
            raise ValueError("runtime episode example names an unbound import")
        object.__setattr__(self, "source_import_ids", imports)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "excluded_counts", tuple(sorted(self.excluded_counts)))
        object.__setattr__(self, "session_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def split_counts(self) -> tuple[tuple[str, int], ...]:
        counts = {name: 0 for name in ("train", "heldout", "regression")}
        for example in self.examples:
            counts[example.split] += 1
        return tuple(sorted(counts.items()))

    @property
    def serving_eligible_count(self) -> int:
        return sum(item.serving_promotion_eligible for item in self.examples)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": RUNTIME_EPISODE_SESSION_SCHEMA,
            "source_import_ids": list(self.source_import_ids),
            "split_policy": "whole-conversation-sha256-v1:80-train-10-heldout-10-regression",
            "source_capacity": "complete-runtime-episode-no-character-truncation",
            "quality_policy": "explicit-outcome-evidence-required-for-serving-promotion",
            "soul_policy": "exact-private-soul-lineage-and-causal-whole-trajectory-v1",
            "example_count": len(self.examples),
            "serving_eligible_count": self.serving_eligible_count,
            "split_counts": [list(item) for item in self.split_counts],
            "excluded_counts": [list(item) for item in self.excluded_counts],
            "examples": [item.to_canonical_dict() for item in self.examples],
        }
        if include_id:
            value["session_id"] = self.session_id
        return value


@dataclass(frozen=True, slots=True)
class LoadedRuntimeEpisode:
    example: RuntimeEpisodeExample
    record: ExperienceRecord
    pre_action_field: SharedFieldSnapshot
    source_delta: FieldDelta
    materialized_delta: FieldDelta
    successor_field: SharedFieldSnapshot
    circulation: Mapping[str, Any]
    soul_lineages: tuple[Mapping[str, Any], ...]


class RuntimeEpisodeSessionCompiler:
    """Compile complete live reasoning episodes with explicit quality labels."""

    def __init__(self, experience: DormantExperienceStore) -> None:
        if not isinstance(experience, DormantExperienceStore):
            raise TypeError("experience must be DormantExperienceStore")
        self.experience = experience
        self.soul_store = SoulStore.active(experience.state_root)

    def _selected_records(
        self,
        import_ids: Sequence[str] | None,
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ExperienceRecord]]]:
        selected = tuple(import_ids) if import_ids is not None else tuple(
            item.import_id for item in self.experience.import_manifests()
        )
        if not selected:
            raise ValueError("no exact Dormant experience imports are available")
        records: dict[str, tuple[str, ExperienceRecord]] = {}
        for import_id in selected:
            for record in self.experience.iter_records((import_id,)):
                records.setdefault(record.record_id, (import_id, record))
        return selected, records

    def compile(
        self,
        import_ids: Sequence[str] | None = None,
    ) -> RuntimeEpisodeSessionManifest:
        selected, records = self._selected_records(import_ids)
        outcomes: dict[str, list[ExperienceRecord]] = {}
        for _, record in records.values():
            if record.record_kind != "runtime_episode_outcome":
                continue
            episode_id = str(record.payload.get("episode_event_id", ""))
            if episode_id:
                outcomes.setdefault(episode_id, []).append(record)

        examples: list[RuntimeEpisodeExample] = []
        excluded: dict[str, int] = {}
        for import_id, record in records.values():
            if record.record_kind != "runtime_reasoning_episode":
                continue
            payload = record.payload
            try:
                if payload.get("schema") != RUNTIME_REASONING_EPISODE_SCHEMA:
                    raise ValueError("schema")
                event_id = str(payload["event_id"])
                conversation_id = str(payload["conversation_id"])
                pre_action = dict(payload["pre_action_field"])
                circulation = dict(payload["circulation"])
                response_sha256 = str(payload["response_text_sha256"])
                base_field_id = str(pre_action["field_id"])
                base_tick_id = int(pre_action["tick_id"])
                result_id = str(circulation["result_id"])
                _, soul_trajectory_id = _validated_soul_lineages(
                    circulation,
                    self.soul_store,
                )
            except (KeyError, TypeError, ValueError):
                excluded["incomplete_runtime_episode"] = excluded.get("incomplete_runtime_episode", 0) + 1
                continue

            outcome_record: ExperienceRecord | None = None
            if event_id in outcomes:
                outcome_record = sorted(
                    outcomes[event_id],
                    key=lambda item: (item.occurred_at, item.record_id),
                )[-1]
            if outcome_record is None:
                try:
                    quality = EpisodeOutcomeQuality(str(payload.get("outcome_quality", "observed")))
                except ValueError:
                    quality = EpisodeOutcomeQuality.UNKNOWN
                evidence_ids = tuple(map(str, payload.get("outcome_evidence_ids", ())))
            else:
                try:
                    quality = EpisodeOutcomeQuality(str(outcome_record.payload["outcome_quality"]))
                except (KeyError, ValueError):
                    quality = EpisodeOutcomeQuality.UNKNOWN
                evidence_ids = tuple(map(str, outcome_record.payload.get("evidence_ids", ())))
            eligible = quality in SERVING_QUALITIES and bool(evidence_ids) and outcome_record is not None
            examples.append(
                RuntimeEpisodeExample(
                    source_import_id=import_id,
                    source_record_id=record.record_id,
                    episode_event_id=event_id,
                    conversation_id=conversation_id,
                    base_field_id=base_field_id,
                    base_tick_id=base_tick_id,
                    circulation_result_id=result_id,
                    soul_trajectory_id=soul_trajectory_id,
                    response_text_sha256=response_sha256,
                    outcome_quality=quality,
                    outcome_evidence_ids=evidence_ids,
                    outcome_record_id=(None if outcome_record is None else outcome_record.record_id),
                    split=whole_episode_split(conversation_id),
                    serving_promotion_eligible=eligible,
                )
            )
        if not examples:
            raise ValueError("no complete runtime reasoning episodes were found")
        by_conversation: dict[str, list[RuntimeEpisodeExample]] = {}
        for example in examples:
            by_conversation.setdefault(example.conversation_id, []).append(example)
        record_by_id = {record.record_id: record for _, record in records.values()}
        for conversation_examples in by_conversation.values():
            prior_final: dict[str, str] = {}
            for example in sorted(
                conversation_examples,
                key=lambda item: (item.base_tick_id, item.episode_event_id),
            ):
                circulation = dict(record_by_id[example.source_record_id].payload["circulation"])
                lineages, observed_id = _validated_soul_lineages(circulation, self.soul_store)
                if observed_id != example.soul_trajectory_id:
                    raise ValueError("runtime episode soul trajectory identity changed")
                for lineage in lineages:
                    core_id = str(lineage["core_id"])
                    if core_id in prior_final and prior_final[core_id] != lineage["initial_soul_id"]:
                        raise ValueError(
                            "runtime conversation contains a discontinuous or future-leaked soul trajectory"
                        )
                    prior_final[core_id] = str(lineage["final_soul_id"])
        return RuntimeEpisodeSessionManifest(
            source_import_ids=selected,
            examples=tuple(sorted(examples, key=lambda item: item.example_id)),
            excluded_counts=tuple(excluded.items()),
        )


class RuntimeEpisodeLoader:
    """Reconstruct and verify the exact runtime state/action pair for training."""

    def __init__(self, experience: DormantExperienceStore) -> None:
        if not isinstance(experience, DormantExperienceStore):
            raise TypeError("experience must be DormantExperienceStore")
        self.experience = experience
        self.soul_store = SoulStore.active(experience.state_root)

    @staticmethod
    def _verify_workspace(
        value: Mapping[str, Any],
        *,
        image_id: str,
        tick_uid: str,
    ) -> None:
        workspace = dict(value)
        required = {
            "schema",
            "image_id",
            "tick_uid",
            "pass_kind",
            "entries",
            "workspace_id",
            "rendered_rails",
        }
        if set(workspace) != required:
            raise ValueError("runtime proposal workspace fields are invalid")
        if workspace["image_id"] != image_id or workspace["tick_uid"] != tick_uid:
            raise ValueError("runtime proposal workspace is stale for its tick image")
        identity_body = {
            key: workspace[key]
            for key in ("schema", "image_id", "tick_uid", "pass_kind", "entries")
        }
        if canonical_sha256(identity_body) != workspace["workspace_id"]:
            raise ValueError("runtime proposal workspace identity mismatch")
        readable = canonical_json_bytes(
            {**identity_body, "workspace_id": workspace["workspace_id"]}
        ).decode("utf-8")
        rails = tuple(workspace["rendered_rails"])
        if not rails:
            raise ValueError("runtime proposal workspace has no rendered rails")
        for raw_rail in rails:
            rail = dict(raw_rail)
            if rail.get("source_workspace_id") != workspace["workspace_id"]:
                raise ValueError("runtime proposal rail is stale for its workspace")
            frame = CategoricalTextFrame.from_mapping(rail["text_frame"])
            if frame.d_model != rail.get("d_model") or frame.text != readable:
                raise ValueError("runtime proposal rail failed exact readable roundtrip")
            rail_body = dict(rail)
            rail_id = rail_body.pop("rail_id", None)
            if rail_id != canonical_sha256(rail_body):
                raise ValueError("runtime proposal rail identity mismatch")

    def load(
        self,
        session: RuntimeEpisodeSessionManifest,
        *,
        split: str | None = None,
        serving_eligible_only: bool = False,
    ) -> tuple[LoadedRuntimeEpisode, ...]:
        if not isinstance(session, RuntimeEpisodeSessionManifest):
            raise TypeError("session must be RuntimeEpisodeSessionManifest")
        if split is not None and split not in {"train", "heldout", "regression"}:
            raise ValueError("unsupported episode split")
        record_map = {
            record.record_id: record
            for record in self.experience.iter_records(session.source_import_ids)
        }
        loaded: list[LoadedRuntimeEpisode] = []
        for example in session.examples:
            if split is not None and example.split != split:
                continue
            if serving_eligible_only and not example.serving_promotion_eligible:
                continue
            try:
                record = record_map[example.source_record_id]
            except KeyError as exc:
                raise ValueError("runtime episode record is missing from its bound imports") from exc
            if record.record_kind != "runtime_reasoning_episode":
                raise ValueError("runtime episode example references the wrong record kind")
            payload = record.payload
            if payload.get("event_id") != example.episode_event_id:
                raise ValueError("runtime episode event identity mismatch")
            if payload.get("conversation_id") != example.conversation_id:
                raise ValueError("runtime episode conversation identity mismatch")
            if example.outcome_record_id is not None:
                try:
                    outcome = record_map[example.outcome_record_id]
                except KeyError as exc:
                    raise ValueError("runtime episode outcome evidence record is missing") from exc
                if (
                    outcome.record_kind != "runtime_episode_outcome"
                    or outcome.payload.get("episode_event_id") != example.episode_event_id
                    or outcome.payload.get("outcome_quality") != example.outcome_quality.value
                    or tuple(sorted(map(str, outcome.payload.get("evidence_ids", ()))))
                    != example.outcome_evidence_ids
                ):
                    raise ValueError("runtime episode outcome evidence disagrees with the session")
            if whole_episode_split(example.conversation_id) != example.split:
                raise ValueError("runtime episode split is not conversation-stable")
            pre_action = snapshot_from_canonical_dict(dict(payload["pre_action_field"]))
            if pre_action.field_id != example.base_field_id or pre_action.tick_id != example.base_tick_id:
                raise ValueError("runtime episode base snapshot identity mismatch")
            circulation = dict(payload["circulation"])
            observed_result_id = str(circulation.get("result_id", ""))
            result_body = dict(circulation)
            result_body.pop("result_id", None)
            if observed_result_id != canonical_sha256(result_body):
                raise ValueError("runtime reasoning circulation identity mismatch")
            if observed_result_id != example.circulation_result_id:
                raise ValueError("session example circulation identity mismatch")
            soul_lineages, soul_trajectory_id = _validated_soul_lineages(
                circulation,
                self.soul_store,
            )
            if soul_trajectory_id != example.soul_trajectory_id:
                raise ValueError("session example soul trajectory identity mismatch")
            image_identity = dict(dict(circulation["image"])["identity"])
            image_id = canonical_sha256(dict(circulation["image"]))
            tick_uid = canonical_sha256(image_identity)
            if (
                image_identity.get("base_field_id") != pre_action.field_id
                or image_identity.get("base_tick_id") != pre_action.tick_id
            ):
                raise ValueError("runtime tick image is stale for the reconstructed base")
            self._verify_workspace(
                dict(circulation["first_workspace"]),
                image_id=image_id,
                tick_uid=tick_uid,
            )
            self._verify_workspace(
                dict(circulation["refined_workspace"]),
                image_id=image_id,
                tick_uid=tick_uid,
            )
            first_emissions = tuple(
                ReasoningEmission.from_mapping(item) for item in circulation["first_emissions"]
            )
            refined_emissions = tuple(
                ReasoningEmission.from_mapping(item) for item in circulation["refined_emissions"]
            )
            for emission, expected_pass in (
                *((item, "first") for item in first_emissions),
                *((item, "refined") for item in refined_emissions),
            ):
                if (
                    emission.base_field_id != pre_action.field_id
                    or emission.base_tick_id != pre_action.tick_id
                    or emission.pass_id != expected_pass
                ):
                    raise ValueError("runtime core emission is stale or names the wrong pass")
            source_delta = field_delta_from_canonical_dict(dict(circulation["source_delta"]))
            materialized_delta = field_delta_from_canonical_dict(dict(circulation["materialized_delta"]))
            if source_delta.base_field_id != pre_action.field_id or materialized_delta.base_field_id != pre_action.field_id:
                raise ValueError("runtime episode deltas are stale for the reconstructed base")
            consolidator_emission = ReasoningEmission.from_mapping(
                dict(circulation["consolidator_emission"])
            )
            decoded_source = consolidator_emission.decode_delta(
                pre_action,
                AuthorityGrant.consolidator(),
            )
            if decoded_source is None or decoded_source.delta_id != source_delta.delta_id:
                raise ValueError("runtime consolidator emission does not reproduce its source delta")
            successor = apply_delta(
                pre_action,
                materialized_delta,
                permitted_regions=AuthorityGrant.consolidator().governed_regions,
            )
            commit = dict(circulation["commit"])
            if commit.get("delta_id") != materialized_delta.delta_id:
                raise ValueError("runtime episode commit names the wrong materialized delta")
            if (
                commit.get("successor_field_id") != successor.field_id
                or commit.get("successor_tick_id") != successor.tick_id
            ):
                raise ValueError("runtime episode successor cannot be reconstructed exactly")
            response_text = record.exact_text
            response_sha256 = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
            if response_sha256 != payload.get("response_text_sha256") or response_sha256 != example.response_text_sha256:
                raise ValueError("runtime episode response hash mismatch")
            if successor.region(LogicalRegion.RESPONSE_DRAFT).text != response_text:
                raise ValueError("runtime episode exact response differs from the accepted successor")
            receipt = circulation.get("finalization_receipt")
            user_text = pre_action.region(LogicalRegion.USER_INPUT).text
            if user_text:
                if not isinstance(receipt, Mapping):
                    raise ValueError("runtime user turn lacks a finalization receipt")
                framed = frame_completed_turn(user_text, response_text)
                expected_receipt = {
                    "base_field_id": pre_action.field_id,
                    "source_delta_id": source_delta.delta_id,
                    "user_text_sha256": hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
                    "response_text_sha256": response_sha256,
                    "framed_turn_sha256": hashlib.sha256(framed.encode("utf-8")).hexdigest(),
                    "framed_turn_chars": len(framed),
                }
                if any(receipt.get(key) != value for key, value in expected_receipt.items()):
                    raise ValueError("runtime turn finalization receipt cannot be reproduced")
            loaded.append(
                LoadedRuntimeEpisode(
                    example=example,
                    record=record,
                    pre_action_field=pre_action,
                    source_delta=source_delta,
                    materialized_delta=materialized_delta,
                    successor_field=successor,
                    circulation=circulation,
                    soul_lineages=soul_lineages,
                )
            )
        return tuple(loaded)


__all__ = [
    "RUNTIME_EPISODE_EXAMPLE_SCHEMA",
    "RUNTIME_EPISODE_SESSION_SCHEMA",
    "RUNTIME_REASONING_EPISODE_SCHEMA",
    "SERVING_QUALITIES",
    "EpisodeOutcomeQuality",
    "LoadedRuntimeEpisode",
    "RuntimeEpisodeExample",
    "RuntimeEpisodeLoader",
    "RuntimeEpisodeSessionCompiler",
    "RuntimeEpisodeSessionManifest",
    "whole_episode_split",
]
