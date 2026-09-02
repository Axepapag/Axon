"""First-Form Curriculum Stack compiled from exact Dormant evidence.

FFCS v1 is a governed campaign, not a permanent capacity limit.  Case-count
budgets select complete lessons; they never clip an admitted field or target.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from runtime.dormant import DormantExperienceStore, ExperienceRecord
from runtime.field import (
    D64FieldCompiler,
    FieldDelta,
    LogicalRegion,
    ReplaceText,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
    snapshot_from_canonical_dict,
)
from runtime.heart import (
    ProposalPass,
    ProposalWorkspace,
    ProposalWorkspaceEntry,
    ReasoningDecision,
    ReasoningOperationKind,
)

from .living_reasoning_curriculum import (
    LIVING_REASONING_TARGET_SCHEMA,
    LivingReasoningCurriculum,
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

FFCS_MANIFEST_SCHEMA = "axon-first-form-curriculum-v1"
FFCS_CASE_SCHEMA = "axon-first-form-curriculum-case-v1"
DEFAULT_FAMILY_SPLIT_COUNTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("A", (32, 4, 4)),
    ("B", (96, 12, 12)),
    ("C", (80, 10, 10)),
)
DEFAULT_DF_SPLIT_COUNTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("D", (64, 8, 8)),
    ("F", (48, 6, 6)),
)
_SPLITS = ("train", "heldout", "regression")
_MESSAGE_KINDS = {"message", "runtime_conversation_message"}
_QUOTE_KINDS = {"diary", "episode", "mission", "objective"}
_SECRET_SHAPES = (
    "authorization:",
    "bearer ",
    "api_key",
    "api-key",
    "private key-----",
    "client_secret",
)


class TeachingEligibility(str, Enum):
    VERIFIED_TARGET = "verified_target"
    PROCESS_EVIDENCE = "process_evidence"
    OBSERVED_ONLY = "observed_only"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class FirstFormCase:
    family: str
    competency: str
    eligibility: TeachingEligibility | str
    lineage_id: str
    source_record_ids: tuple[str, ...]
    episode: LivingReasoningEpisode
    transport_pages: int
    procedural_depth: int
    derived: bool
    case_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.family not in {
            "A", "B", "C", "D", "E", "F", "C1", "L0", "L1", "L2", "L3", "L4",
        }:
            raise ValueError("unsupported FFCS family")
        if not self.competency or not self.lineage_id:
            raise ValueError("FFCS competency and lineage must be non-empty")
        eligibility = (
            self.eligibility
            if isinstance(self.eligibility, TeachingEligibility)
            else TeachingEligibility(self.eligibility)
        )
        object.__setattr__(self, "eligibility", eligibility)
        source_ids = tuple(dict.fromkeys(map(str, self.source_record_ids)))
        if any(len(item) != 64 for item in source_ids):
            raise ValueError("FFCS source record ids must be content identities")
        object.__setattr__(self, "source_record_ids", source_ids)
        if self.transport_pages < 1 or self.procedural_depth < 1:
            raise ValueError("FFCS difficulty measurements must be positive")
        object.__setattr__(self, "case_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": FFCS_CASE_SCHEMA,
            "family": self.family,
            "competency": self.competency,
            "eligibility": self.eligibility.value,
            "lineage_id": self.lineage_id,
            "source_record_ids": list(self.source_record_ids),
            "transport_pages": self.transport_pages,
            "procedural_depth": self.procedural_depth,
            "derived": self.derived,
            "episode": _episode_to_dict(self.episode),
        }
        if include_id:
            value["case_id"] = self.case_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FirstFormCase":
        item = dict(value)
        if item.pop("schema", None) != FFCS_CASE_SCHEMA:
            raise ValueError("unsupported FFCS case schema")
        observed = item.pop("case_id", None)
        item["source_record_ids"] = tuple(item["source_record_ids"])
        item["episode"] = _episode_from_dict(item["episode"])
        case = cls(**item)
        if case.case_id != observed:
            raise ValueError("FFCS case identity mismatch")
        return case


@dataclass(frozen=True, slots=True)
class FirstFormCurriculum:
    cases: tuple[FirstFormCase, ...]
    source_import_ids: tuple[str, ...]
    requested_family_split_counts: tuple[tuple[str, tuple[int, int, int]], ...]
    excluded_counts: tuple[tuple[str, int], ...]
    identity_text_sha256: str
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases or len({item.case_id for item in cases}) != len(cases):
            raise ValueError("FFCS cases must be non-empty and unique")
        object.__setattr__(self, "cases", cases)
        imports = tuple(sorted(set(self.source_import_ids)))
        if not imports or any(len(item) != 64 for item in imports):
            raise ValueError("FFCS source import ids must be SHA256 identities")
        object.__setattr__(self, "source_import_ids", imports)
        counts = tuple(
            (str(family), tuple(map(int, split_counts)))
            for family, split_counts in self.requested_family_split_counts
        )
        if any(len(split_counts) != 3 for _, split_counts in counts):
            raise ValueError("FFCS budgets require train/heldout/regression counts")
        object.__setattr__(self, "requested_family_split_counts", counts)
        object.__setattr__(self, "excluded_counts", tuple(sorted(self.excluded_counts)))
        if len(self.identity_text_sha256) != 64:
            raise ValueError("FFCS identity text hash must be SHA256")
        living = self.living_curriculum
        if len(living.episodes) != len(cases):
            raise ValueError("FFCS living curriculum lost cases")
        object.__setattr__(self, "manifest_id", canonical_sha256(self.to_canonical_dict(False)))

    @property
    def living_curriculum(self) -> LivingReasoningCurriculum:
        return LivingReasoningCurriculum(tuple(item.episode for item in self.cases))

    @property
    def teaching_cases(self) -> tuple[FirstFormCase, ...]:
        """Cases whose exact targets are authorized to drive optimizer loss.

        Published manifests retain every evidence class and their historical
        identities.  This derived view is the permanent Trainer boundary:
        PROCESS_EVIDENCE, OBSERVED_ONLY, and QUARANTINED material may remain
        inspectable, but cannot silently become exact-string supervision.
        """

        return tuple(
            item
            for item in self.cases
            if item.eligibility is TeachingEligibility.VERIFIED_TARGET
        )

    @property
    def teaching_living_curriculum(self) -> LivingReasoningCurriculum:
        """Content-addressed optimizer/evaluation view of verified targets."""

        cases = self.teaching_cases
        if not cases:
            raise ValueError("FFCS manifest contains no VERIFIED_TARGET teaching cases")
        return LivingReasoningCurriculum(tuple(item.episode for item in cases))

    @property
    def eligibility_counts(self) -> tuple[tuple[str, int], ...]:
        """Deterministic evidence-class inventory for reports and gates."""

        return tuple(
            (
                eligibility.value,
                sum(item.eligibility is eligibility for item in self.cases),
            )
            for eligibility in TeachingEligibility
        )

    @property
    def actual_family_split_counts(self) -> tuple[tuple[str, tuple[int, int, int]], ...]:
        rows = []
        for family, _ in self.requested_family_split_counts:
            rows.append(
                (
                    family,
                    tuple(
                        sum(
                            item.family == family and item.episode.split == split
                            for item in self.cases
                        )
                        for split in _SPLITS
                    ),
                )
            )
        return tuple(rows)

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": FFCS_MANIFEST_SCHEMA,
            "source_import_ids": list(self.source_import_ids),
            "identity_text_sha256": self.identity_text_sha256,
            "requested_family_split_counts": [
                [family, list(counts)]
                for family, counts in self.requested_family_split_counts
            ],
            "actual_family_split_counts": [
                [family, list(counts)] for family, counts in self.actual_family_split_counts
            ],
            "excluded_counts": [list(item) for item in self.excluded_counts],
            "case_count": len(self.cases),
            "living_curriculum_id": self.living_curriculum.curriculum_id,
            "cases": [item.to_canonical_dict() for item in self.cases],
            "budget_law": (
                "revisable campaign admission budget; complete admitted lessons are never clipped"
            ),
        }
        if include_id:
            value["manifest_id"] = self.manifest_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FirstFormCurriculum":
        item = dict(value)
        if item.pop("schema", None) != FFCS_MANIFEST_SCHEMA:
            raise ValueError("unsupported FFCS manifest schema")
        observed = item.pop("manifest_id", None)
        actual = item.pop("actual_family_split_counts", None)
        living_id = item.pop("living_curriculum_id", None)
        case_count = int(item.pop("case_count", -1))
        item.pop("budget_law", None)
        item["source_import_ids"] = tuple(item["source_import_ids"])
        item["requested_family_split_counts"] = tuple(
            (str(family), tuple(counts))
            for family, counts in item["requested_family_split_counts"]
        )
        item["excluded_counts"] = tuple(tuple(row) for row in item["excluded_counts"])
        item["cases"] = tuple(FirstFormCase.from_mapping(row) for row in item["cases"])
        curriculum = cls(**item)
        if curriculum.manifest_id != observed:
            raise ValueError("FFCS manifest identity mismatch")
        if case_count != len(curriculum.cases):
            raise ValueError("FFCS manifest case count mismatch")
        if actual != [
            [family, list(counts)] for family, counts in curriculum.actual_family_split_counts
        ]:
            raise ValueError("FFCS actual family counts mismatch")
        if living_id != curriculum.living_curriculum.curriculum_id:
            raise ValueError("FFCS living curriculum identity mismatch")
        return curriculum


def _episode_to_dict(episode: LivingReasoningEpisode) -> dict[str, Any]:
    return {
        "label": episode.label,
        "split": episode.split,
        "snapshot": episode.snapshot.to_dict(),
        "first_workspace_text": episode.first_workspace_text,
        "refined_workspace_text": episode.refined_workspace_text,
        "targets": [item.to_canonical_dict() for item in episode.targets],
        "mechanism_tags": list(episode.mechanism_tags),
        "outcome_quality": episode.outcome_quality,
        "source_example_id": episode.source_example_id,
        "outcome_evidence_ids": list(episode.outcome_evidence_ids),
        "target_basis": episode.target_basis,
        "episode_id": episode.episode_id,
    }


def _episode_from_dict(value: Mapping[str, Any]) -> LivingReasoningEpisode:
    item = dict(value)
    observed = item.pop("episode_id", None)
    item["snapshot"] = snapshot_from_canonical_dict(item["snapshot"])
    item["mechanism_tags"] = tuple(item["mechanism_tags"])
    item["outcome_evidence_ids"] = tuple(item["outcome_evidence_ids"])
    targets = []
    for raw in item["targets"]:
        target = dict(raw)
        if target.pop("schema", None) != LIVING_REASONING_TARGET_SCHEMA:
            raise ValueError("unsupported living reasoning target schema")
        observed_target_id = target.pop("target_id", None)
        restored_target = LivingReasoningTarget(**target)
        if restored_target.target_id != observed_target_id:
            raise ValueError("FFCS target identity mismatch")
        targets.append(restored_target)
    item["targets"] = tuple(targets)
    episode = LivingReasoningEpisode(**item)
    if episode.episode_id != observed:
        raise ValueError("FFCS episode identity mismatch")
    return episode


def _workspace(pass_kind: ProposalPass, detail: str) -> str:
    return ProposalWorkspace(
        image_id="ffcs-v1-derived-image",
        tick_uid="ffcs-v1-derived-tick",
        pass_kind=pass_kind,
        entries=(
            ProposalWorkspaceEntry(
                core_id="ffcs-brother-fixture",
                d_model=64,
                state="returned",
                detail=detail,
                emission_id=None,
                delta=None,
            ),
        ),
    ).readable_text()


def _proposal_workspace(
    *,
    snapshot: SharedFieldSnapshot,
    pass_kind: ProposalPass,
    proposals: tuple[tuple[str, str, str], ...],
    tick_uid: str,
) -> str:
    entries = []
    for core_id, payload, detail in proposals:
        delta = FieldDelta(
            base_field_id=snapshot.field_id,
            base_tick_id=snapshot.tick_id,
            author_core_id=core_id,
            pass_id=pass_kind.value,
            operations=(
                ReplaceText(
                    region=LogicalRegion.RESPONSE_DRAFT,
                    start=0,
                    end=len(snapshot.region(LogicalRegion.RESPONSE_DRAFT).text),
                    text=payload,
                    provenance=f"ffcs-{pass_kind.value}:{core_id}",
                ),
            ),
            evidence=(canonical_sha256({"detail": detail}),),
        )
        entries.append(
            ProposalWorkspaceEntry(
                core_id=core_id,
                d_model=64,
                state="returned",
                detail=detail,
                emission_id=canonical_sha256(
                    {"core_id": core_id, "pass": pass_kind.value, "delta_id": delta.delta_id}
                ),
                delta=delta.to_canonical_dict(),
            )
        )
    return ProposalWorkspace(
        image_id=canonical_sha256(
            {"schema": "axon-ffcs-derived-image-v1", "field_id": snapshot.field_id}
        ),
        tick_uid=tick_uid,
        pass_kind=pass_kind,
        entries=tuple(entries),
    ).readable_text()


def _copy_episode(
    *,
    label: str,
    split: str,
    identity_text: str,
    user_text: str,
    evidence_text: str,
    answer: str,
    source_example_id: str,
    source_record_ids: tuple[str, ...],
    tags: tuple[str, ...],
) -> LivingReasoningEpisode:
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: identity_text,
            LogicalRegion.USER_INPUT: user_text,
            LogicalRegion.CORTEX: evidence_text,
            LogicalRegion.SCRATCH: "",
            LogicalRegion.RESPONSE_DRAFT: "",
        }
    )
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=_workspace(
            ProposalPass.FIRST,
            "Inspect exact canonical evidence before proposing a response.",
        ),
        refined_workspace_text=_workspace(
            ProposalPass.REFINED,
            "Retain only the exact evidence-bound response.",
        ),
        targets=(
            LivingReasoningTarget(
                phase="first",
                decision=ReasoningDecision.NO_OP,
                supervision_weight=0.0,
            ),
            LivingReasoningTarget(
                phase="refined",
                decision=ReasoningDecision.NO_OP,
                supervision_weight=0.0,
            ),
            LivingReasoningTarget(
                phase="consolidated",
                decision=ReasoningDecision.DELTA,
                operation=ReasoningOperationKind.REPLACE,
                region=LogicalRegion.RESPONSE_DRAFT,
                start=0,
                end=0,
                payload=answer,
            ),
        ),
        mechanism_tags=tags,
        outcome_quality="verified_copy_target",
        source_example_id=source_example_id,
        outcome_evidence_ids=source_record_ids,
        target_basis="exact visible evidence copy; semantic quality is not inferred",
    )


def _case(
    *,
    family: str,
    competency: str,
    lineage_id: str,
    source_record_ids: tuple[str, ...],
    episode: LivingReasoningEpisode,
    derived: bool,
    procedural_depth: int = 1,
) -> FirstFormCase:
    compiled = D64FieldCompiler().compile(episode.snapshot)
    compiled.verify_roundtrip(episode.snapshot)
    pages = len(tuple(compiled.iter_character_pages(32)))
    return FirstFormCase(
        family=family,
        competency=competency,
        eligibility=TeachingEligibility.VERIFIED_TARGET,
        lineage_id=lineage_id,
        source_record_ids=source_record_ids,
        episode=episode,
        transport_pages=pages,
        procedural_depth=procedural_depth,
        derived=derived,
    )


def _split(lineage_id: str) -> str:
    bucket = int(hashlib.sha256(lineage_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "heldout" if bucket == 8 else "regression"


def _role(record: ExperienceRecord) -> str:
    return str(record.payload.get("role", "")).casefold()


def _safe_target(text: str) -> bool:
    folded = text.casefold()
    return (
        bool(text.strip())
        and not any(shape in folded for shape in _SECRET_SHAPES)
    )


def _time(record: ExperienceRecord) -> datetime | None:
    try:
        return datetime.fromisoformat(record.occurred_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


class FirstFormCurriculumCompiler:
    def __init__(self, experience: DormantExperienceStore) -> None:
        self.experience = experience

    def compile_abc(
        self,
        *,
        identity_text: str,
        requested_counts: tuple[
            tuple[str, tuple[int, int, int]], ...
        ] = DEFAULT_FAMILY_SPLIT_COUNTS,
    ) -> FirstFormCurriculum:
        if not identity_text:
            raise ValueError("FFCS requires the ratified canonical Identity")
        budgets = {family: dict(zip(_SPLITS, counts, strict=True)) for family, counts in requested_counts}
        if set(budgets) != {"A", "B", "C"}:
            raise ValueError("compile_abc requires exact A/B/C budgets")
        selected = {family: {split: 0 for split in _SPLITS} for family in budgets}
        cases: list[FirstFormCase] = []
        excluded: dict[str, int] = {}

        identity_lines = tuple(line for line in identity_text.splitlines() if line.strip())
        for split, required in budgets["A"].items():
            for index in range(required):
                answer = identity_lines[index % len(identity_lines)]
                lineage = f"canonical-identity-v1:{split}:{index}"
                episode = _copy_episode(
                    label=f"ffcs-a-{split}-{index:03d}",
                    split=split,
                    identity_text=identity_text,
                    user_text="Copy the designated exact Identity evidence into response_draft.",
                    evidence_text=f"[DESIGNATED IDENTITY EVIDENCE]\n{answer}\n[/DESIGNATED IDENTITY EVIDENCE]",
                    answer=answer,
                    source_example_id=canonical_sha256({"lineage": lineage, "answer": answer}),
                    source_record_ids=(),
                    tags=("ffcs_a", "identity", "home_rail", "exact_copy"),
                )
                cases.append(
                    _case(
                        family="A",
                        competency="identity_and_home_rail_presence",
                        lineage_id=lineage,
                        source_record_ids=(),
                        episode=episode,
                        derived=True,
                    )
                )
                selected["A"][split] += 1

        previous_message: dict[str, tuple[str, ExperienceRecord]] = {}
        window: dict[str, int] = {}
        last_time: dict[str, datetime] = {}
        manifests = self.experience.import_manifests()
        for record in self.experience.iter_records(tuple(item.import_id for item in manifests)):
            source = record.source_name
            observed_time = _time(record)
            if (
                observed_time is not None
                and source in last_time
                and abs((observed_time - last_time[source]).total_seconds()) > 1800
            ):
                window[source] = window.get(source, 0) + 1
                previous_message.pop(source, None)
            if observed_time is not None:
                last_time[source] = observed_time
            lineage = canonical_sha256(
                {
                    "schema": "axon-ffcs-source-window-v1",
                    "source_name": source,
                    "window": window.get(source, 0),
                }
            )
            split = _split(lineage)

            if record.record_kind in _MESSAGE_KINDS:
                if _role(record) == "assistant":
                    prior = previous_message.get(source)
                    if prior is not None and prior[0] == lineage and _role(prior[1]) == "user":
                        if selected["B"][split] >= budgets["B"][split]:
                            excluded["b_split_budget_filled"] = excluded.get("b_split_budget_filled", 0) + 1
                        elif not _safe_target(record.exact_text):
                            excluded["b_target_failed_content_policy"] = excluded.get("b_target_failed_content_policy", 0) + 1
                        else:
                            source_ids = (prior[1].record_id, record.record_id)
                            example_id = canonical_sha256(
                                {"schema": "axon-ffcs-b-source-pair-v1", "record_ids": source_ids}
                            )
                            evidence = (
                                f"[EXACT RECOVERED RESPONSE record_id={record.record_id}]\n"
                                f"{record.exact_text}\n[/EXACT RECOVERED RESPONSE]"
                            )
                            episode = _copy_episode(
                                label=f"ffcs-b-{record.record_id[:16]}",
                                split=split,
                                identity_text=identity_text,
                                user_text=prior[1].exact_text,
                                evidence_text=evidence,
                                answer=record.exact_text,
                                source_example_id=example_id,
                                source_record_ids=source_ids,
                                tags=("ffcs_b", "observed_dialogue", "exact_turn_copy"),
                            )
                            cases.append(
                                _case(
                                    family="B",
                                    competency="copy_grounded_turn_authorship",
                                    lineage_id=lineage,
                                    source_record_ids=source_ids,
                                    episode=episode,
                                    derived=True,
                                )
                            )
                            selected["B"][split] += 1
                previous_message[source] = (lineage, record)

            if record.record_kind in _QUOTE_KINDS:
                if selected["C"][split] >= budgets["C"][split]:
                    excluded["c_split_budget_filled"] = excluded.get("c_split_budget_filled", 0) + 1
                elif not _safe_target(record.exact_text):
                    excluded["c_target_failed_content_policy"] = excluded.get("c_target_failed_content_policy", 0) + 1
                else:
                    source_ids = (record.record_id,)
                    example_id = canonical_sha256(
                        {"schema": "axon-ffcs-c-source-record-v1", "record_id": record.record_id}
                    )
                    evidence = (
                        f"[EXACT DORMANT EVIDENCE record_id={record.record_id}]\n"
                        f"{record.exact_text}\n[/EXACT DORMANT EVIDENCE]"
                    )
                    episode = _copy_episode(
                        label=f"ffcs-c-{record.record_id[:16]}",
                        split=split,
                        identity_text=identity_text,
                        user_text="Quote the exact Dormant evidence shown in Cortex.",
                        evidence_text=evidence,
                        answer=record.exact_text,
                        source_example_id=example_id,
                        source_record_ids=source_ids,
                        tags=("ffcs_c", "dormant_grounding", "exact_quote"),
                    )
                    cases.append(
                        _case(
                            family="C",
                            competency="grounded_exact_memory_quote",
                            lineage_id=lineage,
                            source_record_ids=source_ids,
                            episode=episode,
                            derived=True,
                        )
                    )
                    selected["C"][split] += 1

            if all(
                selected[family][split_name] >= budgets[family][split_name]
                for family in ("B", "C")
                for split_name in _SPLITS
            ):
                break

        missing = {
            f"{family}:{split}": budgets[family][split] - selected[family][split]
            for family in budgets
            for split in _SPLITS
            if selected[family][split] != budgets[family][split]
        }
        if missing:
            raise ValueError(f"FFCS source evidence cannot satisfy declared budgets: {missing}")
        return FirstFormCurriculum(
            cases=tuple(cases),
            source_import_ids=tuple(item.import_id for item in manifests),
            requested_family_split_counts=requested_counts,
            excluded_counts=tuple(excluded.items()),
            identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
        )

    def compile_df(
        self,
        *,
        identity_text: str,
        requested_counts: tuple[
            tuple[str, tuple[int, int, int]], ...
        ] = DEFAULT_DF_SPLIT_COUNTS,
    ) -> FirstFormCurriculum:
        """Compile derived society and long-field cases without source clipping."""

        if not identity_text:
            raise ValueError("FFCS requires the ratified canonical Identity")
        budgets = {
            family: dict(zip(_SPLITS, counts, strict=True))
            for family, counts in requested_counts
        }
        if set(budgets) != {"D", "F"}:
            raise ValueError("compile_df requires exact D/F budgets")
        cases: list[FirstFormCase] = []
        colors = ("AMBER", "BLUE", "GREEN", "VIOLET", "WHITE", "SILVER")

        for split, required in budgets["D"].items():
            for index in range(required):
                current = colors[index % len(colors)]
                stale = colors[(index + 1) % len(colors)]
                lineage = f"ffcs-d-society:{split}:{index // 4:04d}"
                tick_uid = f"ffcs-d:{split}:{index:04d}"
                snapshot = SharedFieldSnapshot.from_texts(
                    {
                        LogicalRegion.IDENTITY: identity_text,
                        LogicalRegion.USER_INPUT: "Report the current signal after inspecting the brothers' board.",
                        LogicalRegion.CORTEX: (
                            f"[CURRENT CANONICAL EVIDENCE] signal={current} [/CURRENT CANONICAL EVIDENCE]"
                        ),
                        LogicalRegion.SCRATCH: "",
                        LogicalRegion.RESPONSE_DRAFT: "",
                    },
                    source_manifest_ids=(canonical_sha256({"lineage": lineage}),),
                )
                first = _proposal_workspace(
                    snapshot=snapshot,
                    pass_kind=ProposalPass.FIRST,
                    proposals=(
                        ("brother-current", current, "Matches current canonical evidence."),
                        ("brother-stale", stale, "Plausible but contradicted by current canonical evidence."),
                    ),
                    tick_uid=tick_uid,
                )
                refined = _proposal_workspace(
                    snapshot=snapshot,
                    pass_kind=ProposalPass.REFINED,
                    proposals=(
                        ("brother-current", current, "Retained after comparison with the frozen field."),
                        ("brother-stale", current, "Corrected after seeing the complete first board."),
                    ),
                    tick_uid=tick_uid,
                )
                episode = LivingReasoningEpisode(
                    label=f"ffcs-d-{split}-{index:03d}",
                    split=split,
                    snapshot=snapshot,
                    first_workspace_text=first,
                    refined_workspace_text=refined,
                    targets=(
                        LivingReasoningTarget(
                            phase="first",
                            decision=ReasoningDecision.DELTA,
                            operation=ReasoningOperationKind.REPLACE,
                            region=LogicalRegion.SCRATCH,
                            start=0,
                            end=0,
                            payload=f"field_signal={current}",
                        ),
                        LivingReasoningTarget(
                            phase="refined",
                            decision=ReasoningDecision.DELTA,
                            operation=ReasoningOperationKind.REPLACE,
                            region=LogicalRegion.SCRATCH,
                            start=0,
                            end=0,
                            payload=f"board_verified_signal={current}",
                        ),
                        LivingReasoningTarget(
                            phase="consolidated",
                            decision=ReasoningDecision.DELTA,
                            operation=ReasoningOperationKind.REPLACE,
                            region=LogicalRegion.RESPONSE_DRAFT,
                            start=0,
                            end=0,
                            payload=current,
                        ),
                    ),
                    mechanism_tags=(
                        "ffcs_d",
                        "society",
                        "proposal_refinement",
                        "field_authority",
                        "consolidator_duty",
                    ),
                    outcome_quality="verified_derived_society_target",
                    source_example_id=canonical_sha256(
                        {"lineage": lineage, "variant": index, "current": current, "stale": stale}
                    ),
                    target_basis="constructed board fixture resolved against exact visible canonical evidence",
                )
                cases.append(
                    _case(
                        family="D",
                        competency="board_inspection_and_consolidator_duty",
                        lineage_id=lineage,
                        source_record_ids=(),
                        episode=episode,
                        derived=True,
                        procedural_depth=3,
                    )
                )

        for split, required in budgets["F"].items():
            for index in range(required):
                left = f"L{index:03d}λ"
                right = f"R{index:03d}🧠"
                distractor = "irrelevant observation; " * (5 + index % 5)
                evidence = (
                    f"[FACT_A] left_code={left} [/FACT_A]\n"
                    + distractor
                    + "\n[BRIDGE] Combine FACT_A then FACT_B with a vertical bar. [/BRIDGE]\n"
                    + distractor[::-1]
                    + f"\n[FACT_B] right_code={right} [/FACT_B]"
                )
                answer = f"{left}|{right}"
                lineage = f"ffcs-f-long-field:{split}:{index // 3:04d}"
                snapshot = SharedFieldSnapshot.from_texts(
                    {
                        LogicalRegion.IDENTITY: identity_text,
                        LogicalRegion.USER_INPUT: "Combine the two exact distant codes as instructed by the bridge.",
                        LogicalRegion.CORTEX: evidence,
                        LogicalRegion.SCRATCH: "",
                        LogicalRegion.RESPONSE_DRAFT: "",
                    },
                    source_manifest_ids=(canonical_sha256({"lineage": lineage}),),
                )
                episode = LivingReasoningEpisode(
                    label=f"ffcs-f-{split}-{index:03d}",
                    split=split,
                    snapshot=snapshot,
                    first_workspace_text=_workspace(
                        ProposalPass.FIRST,
                        f"Brother located FACT_A as {left}; verify against the field.",
                    ),
                    refined_workspace_text=_workspace(
                        ProposalPass.REFINED,
                        f"Brother located FACT_B as {right}; combine only after complete coverage.",
                    ),
                    targets=(
                        LivingReasoningTarget(
                            phase="first",
                            decision=ReasoningDecision.DELTA,
                            operation=ReasoningOperationKind.REPLACE,
                            region=LogicalRegion.SCRATCH,
                            start=0,
                            end=0,
                            payload=left,
                        ),
                        LivingReasoningTarget(
                            phase="refined",
                            decision=ReasoningDecision.DELTA,
                            operation=ReasoningOperationKind.REPLACE,
                            region=LogicalRegion.SCRATCH,
                            start=0,
                            end=0,
                            payload=answer,
                        ),
                        LivingReasoningTarget(
                            phase="consolidated",
                            decision=ReasoningDecision.DELTA,
                            operation=ReasoningOperationKind.REPLACE,
                            region=LogicalRegion.RESPONSE_DRAFT,
                            start=0,
                            end=0,
                            payload=answer,
                        ),
                    ),
                    mechanism_tags=(
                        "ffcs_f",
                        "long_field",
                        "multi_page",
                        "cross_page_composition",
                        "unicode",
                        "head",
                        "middle",
                        "tail",
                    ),
                    outcome_quality="verified_derived_long_field_target",
                    source_example_id=canonical_sha256(
                        {"lineage": lineage, "variant": index, "answer": answer}
                    ),
                    target_basis="exact composition of two visible distant evidence spans",
                )
                cases.append(
                    _case(
                        family="F",
                        competency="complete_long_field_cross_page_composition",
                        lineage_id=lineage,
                        source_record_ids=(),
                        episode=episode,
                        derived=True,
                        procedural_depth=3,
                    )
                )

        manifests = self.experience.import_manifests()
        return FirstFormCurriculum(
            cases=tuple(cases),
            source_import_ids=tuple(item.import_id for item in manifests),
            requested_family_split_counts=requested_counts,
            excluded_counts=(),
            identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
        )


def publish_first_form_curriculum(
    curriculum: FirstFormCurriculum,
    *,
    state_root: Path | str,
) -> Path:
    root = Path(state_root).resolve() / "training" / "curricula" / "ffcs_v1"
    final = root / curriculum.manifest_id / "manifest.json"
    body = curriculum.to_canonical_dict()
    if final.exists():
        observed = json.loads(final.read_text(encoding="utf-8"))
        if canonical_json_bytes(observed) != canonical_json_bytes(body):
            raise ValueError("immutable FFCS manifest disagrees with existing artifact")
        FirstFormCurriculum.from_mapping(observed)
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(final.name + f".{os.getpid()}.tmp")
    data = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final)
    FirstFormCurriculum.from_mapping(json.loads(final.read_text(encoding="utf-8")))
    return final


def load_first_form_curriculum(path: Path | str) -> FirstFormCurriculum:
    return FirstFormCurriculum.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "DEFAULT_DF_SPLIT_COUNTS",
    "DEFAULT_FAMILY_SPLIT_COUNTS",
    "FFCS_CASE_SCHEMA",
    "FFCS_MANIFEST_SCHEMA",
    "FirstFormCase",
    "FirstFormCurriculum",
    "FirstFormCurriculumCompiler",
    "TeachingEligibility",
    "load_first_form_curriculum",
    "publish_first_form_curriculum",
]
