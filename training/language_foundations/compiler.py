"""Compiler and publisher for the L0-L4 Language Foundations curriculum.

Assembles the authored band fixtures into an immutable
``axon-first-form-curriculum-v1`` manifest with families L0-L4, reusing the
exact episode shape and publication semantics of C1/FFCS.  Compile-time gates:
budget satisfaction (shortfall fails closed), whole-lineage splits, hidden
target leakage, visibility consistency, and transfer-item disjointness.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from runtime.field import (
    D64FieldCompiler,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)
from runtime.heart import (
    ProposalPass,
    ProposalWorkspace,
    ProposalWorkspaceEntry,
    ReasoningDecision,
    ReasoningOperationKind,
)

from ..first_form_curriculum import (
    FirstFormCase,
    FirstFormCurriculum,
    TeachingEligibility,
)
from ..living_reasoning_curriculum import (
    LivingReasoningEpisode,
    LivingReasoningTarget,
)
from .characters import l0_lesson_specs
from .contracts import (
    LANGUAGE_FOUNDATIONS_AUTHORED_SOURCE_ID,
    LANGUAGE_FOUNDATIONS_FAMILIES,
    TRANSFER_ITEM_PREFIX,
    VISIBILITY_HIDDEN_TAG,
    VISIBILITY_VISIBLE_TAG,
    LanguageLessonSpec,
    TargetVisibility,
)
from .grammar import l4_lesson_specs
from .morphology import l3_lesson_specs
from .orthography import l1_lesson_specs
from .validators import (
    verify_language_no_target_leakage,
    verify_language_transfer_disjointness,
    verify_language_visibility_consistency,
)
from .vocabulary import l2_lesson_specs

DEFAULT_LANGUAGE_SPLIT_COUNTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("L0", (38, 10, 10)),
    ("L1", (40, 10, 10)),
    ("L2", (52, 11, 10)),
    ("L3", (50, 12, 11)),
    ("L4", (60, 13, 12)),
)
_SPLITS = ("train", "heldout", "regression")


def all_language_lesson_specs() -> tuple[LanguageLessonSpec, ...]:
    """Every authored L0-L4 lesson fixture, in deterministic order."""

    return (
        *l0_lesson_specs(),
        *l1_lesson_specs(),
        *l2_lesson_specs(),
        *l3_lesson_specs(),
        *l4_lesson_specs(),
    )


def _workspace(pass_kind: ProposalPass, competency: str) -> str:
    detail = (
        f"Brother studies the {competency} exercise and plans the exact "
        "response; the exact wording is reserved for the consolidated commit."
    )
    return ProposalWorkspace(
        image_id="language-foundations-v1-derived-image",
        tick_uid="language-foundations-v1-derived-tick",
        pass_kind=pass_kind,
        entries=(
            ProposalWorkspaceEntry(
                core_id="language-foundations-brother-fixture",
                d_model=64,
                state="returned",
                detail=detail,
                emission_id=None,
                delta=None,
            ),
        ),
    ).readable_text()


def _consolidated_target(spec: LanguageLessonSpec) -> LivingReasoningTarget:
    if spec.edit is not None:
        kind, start, end = spec.edit
        return LivingReasoningTarget(
            phase="consolidated",
            decision=ReasoningDecision.DELTA,
            operation=kind,
            region=LogicalRegion.RESPONSE_DRAFT,
            start=start,
            end=end,
            payload=spec.answer,
        )
    if not spec.answer:
        return LivingReasoningTarget(
            phase="consolidated",
            decision=ReasoningDecision.NO_OP,
            supervision_weight=1.0,
        )
    return LivingReasoningTarget(
        phase="consolidated",
        decision=ReasoningDecision.DELTA,
        operation=ReasoningOperationKind.REPLACE,
        region=LogicalRegion.RESPONSE_DRAFT,
        start=0,
        end=len(spec.draft_starter),
        payload=spec.answer,
    )


def _episode(
    *,
    spec: LanguageLessonSpec,
    identity_text: str,
    lineage: str,
    label: str,
) -> LivingReasoningEpisode:
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: identity_text,
            LogicalRegion.CONVERSATION_HISTORY: spec.conversation_history,
            LogicalRegion.USER_INPUT: spec.prompt,
            LogicalRegion.CORTEX: spec.cortex_text,
            LogicalRegion.SCRATCH: "",
            LogicalRegion.RESPONSE_DRAFT: spec.draft_starter,
        },
        source_manifest_ids=(LANGUAGE_FOUNDATIONS_AUTHORED_SOURCE_ID,),
    )
    tags = [
        "language_foundations",
        f"band_{spec.family.lower()}",
        spec.competency,
        spec.validator,
        (
            VISIBILITY_HIDDEN_TAG
            if spec.visibility is TargetVisibility.HIDDEN
            else VISIBILITY_VISIBLE_TAG
        ),
    ]
    if spec.transfer_item is not None:
        tags.append(f"{TRANSFER_ITEM_PREFIX}{spec.transfer_item}")
    return LivingReasoningEpisode(
        label=label,
        split=spec.split,
        snapshot=snapshot,
        first_workspace_text=_workspace(ProposalPass.FIRST, spec.competency),
        refined_workspace_text=_workspace(ProposalPass.REFINED, spec.competency),
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
            _consolidated_target(spec),
        ),
        mechanism_tags=tuple(tags),
        outcome_quality="authored_objectively_checkable_target",
        source_example_id=canonical_sha256(
            {"schema": "axon-language-foundations-lesson-v1", "lineage": lineage}
        ),
        target_basis=spec.target_basis,
    )


def _case(
    *,
    spec: LanguageLessonSpec,
    identity_text: str,
    sequence: int,
) -> FirstFormCase:
    lineage = f"{spec.family.lower()}-{spec.competency}:{spec.split}:{sequence:03d}"
    episode = _episode(
        spec=spec,
        identity_text=identity_text,
        lineage=lineage,
        label=f"{spec.family.lower()}-{spec.competency}-{spec.split}-{sequence:03d}",
    )
    compiled = D64FieldCompiler().compile(episode.snapshot)
    compiled.verify_roundtrip(episode.snapshot)
    pages = len(tuple(compiled.iter_character_pages(32)))
    return FirstFormCase(
        family=spec.family,
        competency=spec.competency,
        eligibility=TeachingEligibility.VERIFIED_TARGET,
        lineage_id=lineage,
        source_record_ids=(),
        episode=episode,
        transport_pages=pages,
        procedural_depth=spec.procedural_depth,
        derived=True,
    )


class LanguageFoundationsCompiler:
    """Compiles the authored L0-L4 fixture corpus into a FirstFormCurriculum."""

    def compile(
        self,
        *,
        identity_text: str,
        requested_counts: tuple[
            tuple[str, tuple[int, int, int]], ...
        ] = DEFAULT_LANGUAGE_SPLIT_COUNTS,
    ) -> FirstFormCurriculum:
        if not identity_text:
            raise ValueError("language foundations requires the ratified canonical Identity")
        budgets = {
            family: dict(zip(_SPLITS, counts, strict=True))
            for family, counts in requested_counts
        }
        if set(budgets) != set(LANGUAGE_FOUNDATIONS_FAMILIES):
            raise ValueError("language compile requires exact L0-L4 budgets")
        specs = all_language_lesson_specs()
        available: dict[str, dict[str, int]] = {
            family: {split: 0 for split in _SPLITS} for family in budgets
        }
        for spec in specs:
            available[spec.family][spec.split] += 1
        shortfall = {
            f"{family}:{split}": budgets[family][split] - available[family][split]
            for family in budgets
            for split in _SPLITS
            if budgets[family][split] > available[family][split]
        }
        if shortfall:
            raise ValueError(
                f"language authored fixtures cannot satisfy declared budgets: {shortfall}"
            )

        selected = {family: {split: 0 for split in _SPLITS} for family in budgets}
        sequences: dict[tuple[str, str], int] = {}
        cases: list[FirstFormCase] = []
        for spec in specs:
            if selected[spec.family][spec.split] >= budgets[spec.family][spec.split]:
                continue
            key = (spec.competency, spec.split)
            sequence = sequences.get(key, 0)
            sequences[key] = sequence + 1
            cases.append(_case(spec=spec, identity_text=identity_text, sequence=sequence))
            selected[spec.family][spec.split] += 1
        excluded = sum(
            available[family][split] - selected[family][split]
            for family in budgets
            for split in _SPLITS
        )

        lineage_splits: dict[str, set[str]] = {}
        for case in cases:
            lineage_splits.setdefault(case.lineage_id, set()).add(case.episode.split)
        if any(len(splits) != 1 for splits in lineage_splits.values()):
            raise ValueError("language lineage crosses a train/heldout/regression split")
        verify_language_visibility_consistency(cases)
        verify_language_no_target_leakage(cases)
        verify_language_transfer_disjointness(cases)
        return FirstFormCurriculum(
            cases=tuple(cases),
            source_import_ids=(LANGUAGE_FOUNDATIONS_AUTHORED_SOURCE_ID,),
            requested_family_split_counts=requested_counts,
            excluded_counts=(
                () if excluded == 0 else (("language_fixtures_not_selected", excluded),)
            ),
            identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
        )


def publish_language_foundations_curriculum(
    curriculum: FirstFormCurriculum,
    *,
    state_root: Path | str,
) -> Path:
    root = Path(state_root).resolve() / "training" / "curricula" / "language_l0_l4"
    final = root / curriculum.manifest_id / "manifest.json"
    body = curriculum.to_canonical_dict()
    if final.exists():
        observed = json.loads(final.read_text(encoding="utf-8"))
        if canonical_json_bytes(observed) != canonical_json_bytes(body):
            raise ValueError("immutable language manifest disagrees with existing artifact")
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


__all__ = [
    "DEFAULT_LANGUAGE_SPLIT_COUNTS",
    "LanguageFoundationsCompiler",
    "all_language_lesson_specs",
    "publish_language_foundations_curriculum",
]
