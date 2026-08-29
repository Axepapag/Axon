"""Runtime-faithful FFCS-E sequential ticks with exact private-Soul carriage."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Mapping

import torch

from runtime.dormant import DormantExperienceStore
from runtime.field import (
    D64FieldCompiler,
    FieldDelta,
    InsertText,
    LogicalRegion,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    canonical_json_bytes,
    canonical_sha256,
    field_delta_from_canonical_dict,
    snapshot_from_canonical_dict,
)
from runtime.heart import (
    AuthorityGrant,
    IngressChannel,
    ProposalPass,
    ProposalWorkspace,
    ProposalWorkspaceEntry,
    ReasoningDecision,
    ReasoningOperationKind,
    TurnFinalizationReceipt,
    materialize_completed_turn,
)
from runtime.soul import SoulSnapshot

from .first_form_curriculum import _episode_from_dict, _episode_to_dict
from .living_reasoning_curriculum import (
    LivingReasoningEpisode,
    LivingReasoningTarget,
    living_episode_objective,
)
from .living_reasoning_d64 import CausalLivingUnroll, LivingReasoningCoreD64

FFCS_E_TICK_SCHEMA = "axon-first-form-sequential-tick-v1"
FFCS_E_CASE_SCHEMA = "axon-first-form-sequential-case-v1"
FFCS_E_MANIFEST_SCHEMA = "axon-first-form-sequential-curriculum-v1"
DEFAULT_E_SPLIT_COUNTS = (48, 6, 6)
_SPLITS = ("train", "heldout", "regression")


def _receipt_from_mapping(value: Mapping[str, Any] | None) -> TurnFinalizationReceipt | None:
    if value is None:
        return None
    item = dict(value)
    if item.pop("schema", None) != "axon-heart-turn-finalization-v1":
        raise ValueError("unsupported turn-finalization receipt schema")
    observed = item.pop("receipt_id", None)
    receipt = TurnFinalizationReceipt(**item)
    if receipt.receipt_id != observed:
        raise ValueError("turn-finalization receipt identity mismatch")
    return receipt


@dataclass(frozen=True, slots=True)
class SequentialFirstFormTick:
    episode: LivingReasoningEpisode
    source_delta: FieldDelta
    materialized_delta: FieldDelta
    finalization_receipt: TurnFinalizationReceipt | None
    successor: SharedFieldSnapshot
    next_ingress_delta: FieldDelta | None = None
    next_snapshot: SharedFieldSnapshot | None = None
    tick_case_id: str = field(init=False)

    def __post_init__(self) -> None:
        expected_delta, expected_receipt = materialize_completed_turn(
            self.episode.snapshot,
            self.source_delta,
        )
        if expected_delta.delta_id != self.materialized_delta.delta_id:
            raise ValueError("sequential tick materialized delta differs from Heart finalization")
        if expected_receipt != self.finalization_receipt:
            raise ValueError("sequential tick finalization receipt mismatch")
        replayed = apply_delta(
            self.episode.snapshot,
            self.materialized_delta,
            permitted_regions=AuthorityGrant.consolidator().governed_regions,
        )
        if replayed.field_id != self.successor.field_id:
            raise ValueError("sequential tick successor is not the exact Heart replay")
        if (self.next_ingress_delta is None) != (self.next_snapshot is None):
            raise ValueError("next ingress delta and snapshot must be present together")
        if self.next_ingress_delta is not None:
            next_snapshot = apply_delta(
                self.successor,
                self.next_ingress_delta,
                permitted_regions=AuthorityGrant.ingress(IngressChannel.USER).governed_regions,
            )
            if next_snapshot.field_id != self.next_snapshot.field_id:
                raise ValueError("sequential next snapshot is not the exact ingress replay")
        compiler = D64FieldCompiler()
        compiler.compile(self.episode.snapshot).verify_roundtrip(self.episode.snapshot)
        compiler.compile(self.successor).verify_roundtrip(self.successor)
        object.__setattr__(self, "tick_case_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": FFCS_E_TICK_SCHEMA,
            "episode": _episode_to_dict(self.episode),
            "source_delta": self.source_delta.to_canonical_dict(),
            "source_delta_id": self.source_delta.delta_id,
            "materialized_delta": self.materialized_delta.to_canonical_dict(),
            "materialized_delta_id": self.materialized_delta.delta_id,
            "finalization_receipt": (
                None
                if self.finalization_receipt is None
                else self.finalization_receipt.to_canonical_dict()
            ),
            "successor": self.successor.to_dict(),
            "next_ingress_delta": (
                None if self.next_ingress_delta is None else self.next_ingress_delta.to_canonical_dict()
            ),
            "next_ingress_delta_id": (
                None if self.next_ingress_delta is None else self.next_ingress_delta.delta_id
            ),
            "next_snapshot": None if self.next_snapshot is None else self.next_snapshot.to_dict(),
        }
        if include_id:
            value["tick_case_id"] = self.tick_case_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SequentialFirstFormTick":
        item = dict(value)
        if item.pop("schema", None) != FFCS_E_TICK_SCHEMA:
            raise ValueError("unsupported sequential tick schema")
        observed = item.pop("tick_case_id", None)
        source_id = item.pop("source_delta_id", None)
        materialized_id = item.pop("materialized_delta_id", None)
        ingress_id = item.pop("next_ingress_delta_id", None)
        item["episode"] = _episode_from_dict(item["episode"])
        item["source_delta"] = field_delta_from_canonical_dict(item["source_delta"])
        item["materialized_delta"] = field_delta_from_canonical_dict(item["materialized_delta"])
        item["finalization_receipt"] = _receipt_from_mapping(item["finalization_receipt"])
        item["successor"] = snapshot_from_canonical_dict(item["successor"])
        if item["next_ingress_delta"] is not None:
            item["next_ingress_delta"] = field_delta_from_canonical_dict(
                item["next_ingress_delta"]
            )
        if item["next_snapshot"] is not None:
            item["next_snapshot"] = snapshot_from_canonical_dict(item["next_snapshot"])
        tick = cls(**item)
        if (
            tick.tick_case_id != observed
            or tick.source_delta.delta_id != source_id
            or tick.materialized_delta.delta_id != materialized_id
            or (None if tick.next_ingress_delta is None else tick.next_ingress_delta.delta_id)
            != ingress_id
        ):
            raise ValueError("sequential tick identity mismatch")
        return tick


@dataclass(frozen=True, slots=True)
class SequentialFirstFormCase:
    label: str
    split: str
    lineage_id: str
    ticks: tuple[SequentialFirstFormTick, ...]
    procedural_depth: int
    case_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.split not in _SPLITS or not self.label or not self.lineage_id:
            raise ValueError("sequential case identity and split must be valid")
        ticks = tuple(self.ticks)
        if not 2 <= len(ticks) <= 4:
            raise ValueError("FFCS-E cases require two to four real ticks")
        if self.procedural_depth != len(ticks):
            raise ValueError("sequential procedural depth must equal tick count")
        if any(item.episode.split != self.split for item in ticks):
            raise ValueError("all sequential ticks must inherit the case split")
        for current, following in pairwise(ticks):
            if current.next_snapshot is None:
                raise ValueError("non-final sequential tick requires exact next ingress")
            if current.next_snapshot.field_id != following.episode.snapshot.field_id:
                raise ValueError("sequential tick chain skips a canonical successor")
        if ticks[-1].next_snapshot is not None:
            raise ValueError("final sequential tick may not leak a future field")
        object.__setattr__(self, "ticks", ticks)
        object.__setattr__(self, "case_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": FFCS_E_CASE_SCHEMA,
            "label": self.label,
            "split": self.split,
            "lineage_id": self.lineage_id,
            "procedural_depth": self.procedural_depth,
            "ticks": [item.to_canonical_dict() for item in self.ticks],
            "soul_boundary_law": (
                "same core and exact architecture/parameter generation; serialize and re-inhale "
                "at every phase and tick boundary; runtime gradient break; no future leakage"
            ),
        }
        if include_id:
            value["case_id"] = self.case_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SequentialFirstFormCase":
        item = dict(value)
        if item.pop("schema", None) != FFCS_E_CASE_SCHEMA:
            raise ValueError("unsupported sequential case schema")
        observed = item.pop("case_id", None)
        item.pop("soul_boundary_law", None)
        item["ticks"] = tuple(SequentialFirstFormTick.from_mapping(row) for row in item["ticks"])
        case = cls(**item)
        if case.case_id != observed:
            raise ValueError("sequential case identity mismatch")
        return case


@dataclass(frozen=True, slots=True)
class SequentialFirstFormCurriculum:
    cases: tuple[SequentialFirstFormCase, ...]
    source_import_ids: tuple[str, ...]
    requested_split_counts: tuple[int, int, int]
    identity_text_sha256: str
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases or len({item.case_id for item in cases}) != len(cases):
            raise ValueError("sequential curriculum cases must be nonempty and unique")
        object.__setattr__(self, "cases", cases)
        imports = tuple(sorted(set(self.source_import_ids)))
        if not imports or any(len(item) != 64 for item in imports):
            raise ValueError("sequential curriculum requires exact source import identities")
        object.__setattr__(self, "source_import_ids", imports)
        counts = tuple(sum(item.split == split for item in cases) for split in _SPLITS)
        if counts != tuple(self.requested_split_counts):
            raise ValueError("sequential curriculum did not satisfy its declared split budget")
        if len(self.identity_text_sha256) != 64:
            raise ValueError("sequential Identity hash must be SHA256")
        object.__setattr__(self, "manifest_id", canonical_sha256(self.to_canonical_dict(False)))

    def split(self, name: str) -> tuple[SequentialFirstFormCase, ...]:
        return tuple(item for item in self.cases if item.split == name)

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": FFCS_E_MANIFEST_SCHEMA,
            "source_import_ids": list(self.source_import_ids),
            "requested_split_counts": list(self.requested_split_counts),
            "actual_split_counts": [len(self.split(name)) for name in _SPLITS],
            "identity_text_sha256": self.identity_text_sha256,
            "case_count": len(self.cases),
            "cases": [item.to_canonical_dict() for item in self.cases],
        }
        if include_id:
            value["manifest_id"] = self.manifest_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SequentialFirstFormCurriculum":
        item = dict(value)
        if item.pop("schema", None) != FFCS_E_MANIFEST_SCHEMA:
            raise ValueError("unsupported sequential curriculum schema")
        observed = item.pop("manifest_id", None)
        actual = item.pop("actual_split_counts", None)
        case_count = item.pop("case_count", None)
        item["source_import_ids"] = tuple(item["source_import_ids"])
        item["requested_split_counts"] = tuple(item["requested_split_counts"])
        item["cases"] = tuple(SequentialFirstFormCase.from_mapping(row) for row in item["cases"])
        curriculum = cls(**item)
        if (
            curriculum.manifest_id != observed
            or case_count != len(curriculum.cases)
            or actual != [len(curriculum.split(name)) for name in _SPLITS]
        ):
            raise ValueError("sequential curriculum identity/count mismatch")
        return curriculum


def _workspace(tick_uid: str, pass_kind: ProposalPass, detail: str) -> str:
    return ProposalWorkspace(
        image_id=canonical_sha256({"tick_uid": tick_uid, "kind": "ffcs-e-image"}),
        tick_uid=tick_uid,
        pass_kind=pass_kind,
        entries=(
            ProposalWorkspaceEntry(
                core_id="ffcs-e-brother",
                d_model=64,
                state="returned",
                detail=detail,
                emission_id=None,
                delta=None,
            ),
        ),
    ).readable_text()


def _episode(
    *,
    label: str,
    split: str,
    snapshot: SharedFieldSnapshot,
    region: LogicalRegion,
    payload: str,
    end: int,
) -> LivingReasoningEpisode:
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=_workspace(label, ProposalPass.FIRST, "Inspect the complete current field."),
        refined_workspace_text=_workspace(
            label,
            ProposalPass.REFINED,
            "Retain only the proposal supported by this tick's frozen field.",
        ),
        targets=(
            LivingReasoningTarget("first", ReasoningDecision.NO_OP, supervision_weight=0.0),
            LivingReasoningTarget("refined", ReasoningDecision.NO_OP, supervision_weight=0.0),
            LivingReasoningTarget(
                "consolidated",
                ReasoningDecision.DELTA,
                operation=ReasoningOperationKind.REPLACE,
                region=region,
                start=0,
                end=end,
                payload=payload,
            ),
        ),
        mechanism_tags=("ffcs_e", "sequential_tick", "soul_carriage", "canonical_successor"),
        outcome_quality="verified_runtime_replay_target",
        source_example_id=canonical_sha256(
            {"label": label, "field_id": snapshot.field_id, "region": region.value, "payload": payload}
        ),
        target_basis="exact typed delta followed by deterministic Heart successor semantics",
    )


def _tick(
    episode: LivingReasoningEpisode,
    *,
    core_id: str,
    next_user: str | None,
) -> SequentialFirstFormTick:
    target = episode.targets[-1]
    source = FieldDelta(
        base_field_id=episode.snapshot.field_id,
        base_tick_id=episode.snapshot.tick_id,
        author_core_id=core_id,
        pass_id="consolidated",
        operations=(
            ReplaceText(
                region=target.region,
                start=target.start,
                end=target.end,
                text=target.payload,
                provenance=f"ffcs-e:{episode.episode_id}",
            ),
        ),
        evidence=(episode.episode_id,),
    )
    materialized, receipt = materialize_completed_turn(episode.snapshot, source)
    successor = apply_delta(
        episode.snapshot,
        materialized,
        permitted_regions=AuthorityGrant.consolidator().governed_regions,
    )
    ingress = None
    next_snapshot = None
    if next_user is not None:
        ingress = FieldDelta(
            base_field_id=successor.field_id,
            base_tick_id=successor.tick_id,
            author_core_id="ffcs-e-user-ingress",
            pass_id="ingress",
            operations=(
                InsertText(
                    region=LogicalRegion.USER_INPUT,
                    offset=0,
                    text=next_user,
                    provenance=f"ffcs-e-next:{episode.episode_id}",
                ),
            ),
            evidence=(episode.episode_id,),
        )
        next_snapshot = apply_delta(
            successor,
            ingress,
            permitted_regions=AuthorityGrant.ingress(IngressChannel.USER).governed_regions,
        )
    return SequentialFirstFormTick(
        episode=episode,
        source_delta=source,
        materialized_delta=materialized,
        finalization_receipt=receipt,
        successor=successor,
        next_ingress_delta=ingress,
        next_snapshot=next_snapshot,
    )


def compile_sequential_first_form(
    experience: DormantExperienceStore,
    *,
    identity_text: str,
    requested_split_counts: tuple[int, int, int] = DEFAULT_E_SPLIT_COUNTS,
) -> SequentialFirstFormCurriculum:
    if not identity_text:
        raise ValueError("FFCS-E requires ratified canonical Identity")
    cases = []
    for split, required in zip(_SPLITS, requested_split_counts, strict=True):
        for index in range(required):
            lineage = f"ffcs-e-sequential:{split}:{index:04d}"
            note = f"carry-note-{index:04d}-λ"
            first_user = f"Read the private carried note for episode {index:04d}."
            first_answer = f"noted:{note}"
            second_user = "What exact note persisted before our previous turn?"
            second_answer = f"persisted:{note}"
            base = SharedFieldSnapshot.from_texts(
                {
                    LogicalRegion.IDENTITY: identity_text,
                    LogicalRegion.CORTEX: "Prepare a private working note, then answer later user turns from it.",
                    LogicalRegion.SCRATCH: "",
                    LogicalRegion.RESPONSE_DRAFT: "",
                },
                source_manifest_ids=(canonical_sha256({"lineage": lineage}),),
            )
            episode0 = _episode(
                label=f"ffcs-e-{split}-{index:03d}-tick-0",
                split=split,
                snapshot=base,
                region=LogicalRegion.SCRATCH,
                payload=note,
                end=0,
            )
            tick0 = _tick(episode0, core_id="ffcs-e-core", next_user=first_user)
            episode1 = _episode(
                label=f"ffcs-e-{split}-{index:03d}-tick-1",
                split=split,
                snapshot=tick0.next_snapshot,
                region=LogicalRegion.RESPONSE_DRAFT,
                payload=first_answer,
                end=0,
            )
            tick1 = _tick(episode1, core_id="ffcs-e-core", next_user=second_user)
            episode2 = _episode(
                label=f"ffcs-e-{split}-{index:03d}-tick-2",
                split=split,
                snapshot=tick1.next_snapshot,
                region=LogicalRegion.RESPONSE_DRAFT,
                payload=second_answer,
                end=len(first_answer),
            )
            tick2 = _tick(episode2, core_id="ffcs-e-core", next_user=None)
            cases.append(
                SequentialFirstFormCase(
                    label=f"ffcs-e-{split}-{index:03d}",
                    split=split,
                    lineage_id=lineage,
                    ticks=(tick0, tick1, tick2),
                    procedural_depth=3,
                )
            )
    manifests = experience.import_manifests()
    return SequentialFirstFormCurriculum(
        cases=tuple(cases),
        source_import_ids=tuple(item.import_id for item in manifests),
        requested_split_counts=requested_split_counts,
        identity_text_sha256=hashlib.sha256(identity_text.encode()).hexdigest(),
    )


def publish_sequential_first_form(
    curriculum: SequentialFirstFormCurriculum,
    *,
    state_root: Path | str,
) -> Path:
    final = (
        Path(state_root).resolve()
        / "training"
        / "curricula"
        / "ffcs_e_v1"
        / curriculum.manifest_id
        / "manifest.json"
    )
    body = curriculum.to_canonical_dict()
    if final.exists():
        observed = json.loads(final.read_text(encoding="utf-8"))
        if canonical_json_bytes(observed) != canonical_json_bytes(body):
            raise ValueError("immutable FFCS-E manifest mismatch")
        SequentialFirstFormCurriculum.from_mapping(observed)
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(final.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(body, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final)
    return final


def load_sequential_first_form(path: Path | str) -> SequentialFirstFormCurriculum:
    return SequentialFirstFormCurriculum.from_mapping(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def sequential_living_objective(
    model: LivingReasoningCoreD64,
    case: SequentialFirstFormCase,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
) -> tuple[torch.Tensor, tuple[CausalLivingUnroll, ...], SoulSnapshot]:
    """Train exact ticks in order, carrying only serialized same-core Soul."""

    soul = initial_soul
    losses = []
    unrolls = []
    for tick in case.ticks:
        loss, unroll, _metrics = living_episode_objective(
            model,
            tick.episode,
            soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
        )
        losses.append(loss)
        unrolls.append(unroll)
        soul = unroll.souls[-1]
    return torch.stack(losses).mean(), tuple(unrolls), soul


__all__ = [
    "DEFAULT_E_SPLIT_COUNTS",
    "FFCS_E_CASE_SCHEMA",
    "FFCS_E_MANIFEST_SCHEMA",
    "FFCS_E_TICK_SCHEMA",
    "SequentialFirstFormCase",
    "SequentialFirstFormCurriculum",
    "SequentialFirstFormTick",
    "compile_sequential_first_form",
    "load_sequential_first_form",
    "publish_sequential_first_form",
    "sequential_living_objective",
]
