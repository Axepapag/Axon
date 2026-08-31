"""C1 hidden-target single-turn communication curriculum.

Implements the first implementation shot of
``docs/COMMUNICATION_FIRST_CURRICULUM.md``: a small, diverse, provenance-
governed family of single-turn conversational lessons whose response targets
are hidden from every visible surface.

Binding lesson laws honored here:

1. Hidden targets.  The consolidated response payload never appears verbatim
   in any visible field region, proposal workspace text, metadata string, or
   sibling case.  ``verify_c1_no_target_leakage`` enforces this over the
   complete case surface and runs at compile time.
4. Whole-lineage splits.  Every case carries a unique ``lineage_id`` scoped by
   its split; train/heldout/regression therefore never share a lineage.
5-6. Eligibility labeling.  All v1 cases are authored fixtures with explicit
   synthetic provenance (``C1_AUTHORED_SOURCE_ID``); no recovered material
   supervises.  ``VERIFIED_TARGET`` is used only where the target is
   objectively checkable against the visible field: grounded short answers
   paraphrase an exact Cortex fact, clarification requests respond to a
   planted material ambiguity, uncertainty admissions face absent evidence,
   and correction acceptances echo an explicit user correction.  Purely
   conversational targets (greeting, acknowledgement, ordinary turn-taking)
   are labeled ``PROCESS_EVIDENCE``: their quality is asserted by authorship
   and contract verification, not by an objective machine check, and they are
   not recovered observations, so ``OBSERVED_ONLY`` would misstate provenance.

No C1 target asserts Axon identity claims, personal history, or quotes
Identity region content; the leakage verifier checks the Identity region like
every other visible surface.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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

from .first_form_curriculum import (
    FirstFormCase,
    FirstFormCurriculum,
    TeachingEligibility,
)
from .living_reasoning_curriculum import (
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

C1_FAMILY = "C1"
C1_COMPETENCIES: tuple[str, ...] = (
    "greeting",
    "acknowledgement",
    "grounded_short_answer",
    "clarification_request",
    "correction_acceptance",
    "uncertainty_admission",
    "turn_taking",
)
DEFAULT_C1_SPLIT_COUNTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    (C1_FAMILY, (24, 6, 6)),
)
C1_AUTHORED_SOURCE_ID = canonical_sha256(
    {"schema": "axon-c1-authored-fixture-corpus-v1", "fixture_version": 1}
)
_SPLITS = ("train", "heldout", "regression")
_PROCESS_COMPETENCIES = frozenset({"greeting", "acknowledgement", "turn_taking"})

_WORKSPACE_DETAILS: Mapping[str, str] = {
    "greeting": (
        "Brother recognizes an opening greeting and plans a brief warm reply; "
        "the exact wording is reserved for the consolidated commit."
    ),
    "acknowledgement": (
        "Brother registers the user's status update and plans a short "
        "confirmation; the exact wording is reserved for the consolidated commit."
    ),
    "grounded_short_answer": (
        "Brother locates the governing fact inside the Cortex evidence and plans "
        "one concise answer sentence; the exact wording is reserved for the "
        "consolidated commit."
    ),
    "clarification_request": (
        "Brother detects a material ambiguity in the request and plans a single "
        "targeted clarifying question; the exact wording is reserved for the "
        "consolidated commit."
    ),
    "correction_acceptance": (
        "Brother sees the user's correction override the earlier assistant claim "
        "and plans an accepting acknowledgement; the exact wording is reserved "
        "for the consolidated commit."
    ),
    "uncertainty_admission": (
        "Brother finds no supporting evidence anywhere in the visible field and "
        "plans an honest admission of uncertainty; the exact wording is reserved "
        "for the consolidated commit."
    ),
    "turn_taking": (
        "Brother recognizes a routine turn-taking beat and plans a brief "
        "courteous close; the exact wording is reserved for the consolidated "
        "commit."
    ),
}

_TARGET_BASIS: Mapping[str, str] = {
    "greeting": (
        "authored conversational fixture; quality asserted by authorship and "
        "contract verification, not machine-checked"
    ),
    "acknowledgement": (
        "authored conversational fixture; quality asserted by authorship and "
        "contract verification, not machine-checked"
    ),
    "grounded_short_answer": (
        "authored fixture; the answer paraphrases only the exact Cortex fact "
        "visible in the field and is objectively checkable against it"
    ),
    "clarification_request": (
        "authored fixture; the request carries a planted material ambiguity with "
        "no resolving evidence anywhere in the visible field"
    ),
    "correction_acceptance": (
        "authored fixture; the user turn supplies an explicit correction that "
        "supersedes the earlier assistant claim in the visible history"
    ),
    "uncertainty_admission": (
        "authored fixture; no region of the visible field contains evidence that "
        "answers the question"
    ),
    "turn_taking": (
        "authored conversational fixture; quality asserted by authorship and "
        "contract verification, not machine-checked"
    ),
}


@dataclass(frozen=True, slots=True)
class _C1Fixture:
    competency: str
    split: str
    user_text: str
    answer: str
    conversation_history: str = ""
    cortex_text: str = ""
    procedural_depth: int = 1


_C1_FIXTURES: tuple[_C1Fixture, ...] = (
    # greeting (3 train / 1 heldout / 1 regression)
    _C1Fixture(
        "greeting", "train", "Hello.", "Hello! What can I do for you today?"
    ),
    _C1Fixture(
        "greeting",
        "train",
        "Good morning.",
        "Morning! I hope your day is off to a good start. What's on the agenda?",
    ),
    _C1Fixture(
        "greeting", "train", "hey, are you there?", "Hey! Yes, I'm here — what's up?"
    ),
    _C1Fixture(
        "greeting", "heldout", "Good evening!", "Evening! How can I help you tonight?"
    ),
    _C1Fixture(
        "greeting", "regression", "hi", "Hi there! What would you like to work on?"
    ),
    # acknowledgement (3 train / 1 heldout / 1 regression)
    _C1Fixture(
        "acknowledgement",
        "train",
        "I finished the report you asked about.",
        "Got it — thanks for the update on the report.",
    ),
    _C1Fixture(
        "acknowledgement",
        "train",
        "Just so you know, the meeting moved to 3 PM.",
        "Understood — noted that the meeting is now at 3 PM.",
    ),
    _C1Fixture(
        "acknowledgement",
        "train",
        "FYI, I sent the invoice yesterday.",
        "Thanks for letting me know the invoice is on its way.",
    ),
    _C1Fixture(
        "acknowledgement",
        "heldout",
        "The package finally arrived.",
        "Good to hear — glad the package made it.",
    ),
    _C1Fixture(
        "acknowledgement",
        "regression",
        "I already restarted the server.",
        "Understood, thanks — the server restart is done.",
    ),
    # grounded_short_answer (5 train / 1 heldout / 1 regression)
    _C1Fixture(
        "grounded_short_answer",
        "train",
        "How much water does the tank hold?",
        "It holds up to 40 liters.",
        cortex_text="[FACT] The water tank can hold 40 liters. [/FACT]",
    ),
    _C1Fixture(
        "grounded_short_answer",
        "train",
        "When does the library close on weekdays?",
        "Weekday closing time is 8 PM.",
        cortex_text="[FACT] The library closes at 8 PM on weekdays. [/FACT]",
    ),
    _C1Fixture(
        "grounded_short_answer",
        "train",
        "What time does Marta's bakery open?",
        "Doors open at 6:30 AM.",
        cortex_text="[FACT] Marta's bakery opens at 6:30 in the morning. [/FACT]",
    ),
    _C1Fixture(
        "grounded_short_answer",
        "train",
        "Which wire is the ground?",
        "Ground is the blue wire.",
        cortex_text=(
            "[FACT_A] The red wire carries power. [/FACT_A]\n"
            "[FACT_B] The blue wire is the ground line. [/FACT_B]"
        ),
        procedural_depth=2,
    ),
    _C1Fixture(
        "grounded_short_answer",
        "train",
        "Which platform do I need for the Rostock train?",
        "It leaves from platform 3.",
        cortex_text="[FACT] The train to Rostock departs from platform 3. [/FACT]",
    ),
    _C1Fixture(
        "grounded_short_answer",
        "heldout",
        "Can I visit the museum on Monday?",
        "No — it stays closed on Mondays.",
        cortex_text="[FACT] The museum is closed on Mondays. [/FACT]",
    ),
    _C1Fixture(
        "grounded_short_answer",
        "regression",
        "How long is the ferry ride?",
        "Roughly 25 minutes each way.",
        cortex_text="[FACT] The ferry crossing takes about 25 minutes. [/FACT]",
    ),
    # clarification_request (4 train / 1 heldout / 1 regression)
    _C1Fixture(
        "clarification_request",
        "train",
        "Please move it to the other folder.",
        "Which item should I move, and which folder do you mean?",
    ),
    _C1Fixture(
        "clarification_request",
        "train",
        "Set a reminder for the appointment.",
        "Sure — what date and time is the appointment?",
    ),
    _C1Fixture(
        "clarification_request",
        "train",
        "Order more of the usual, please.",
        "Happy to — what exactly counts as the usual order?",
    ),
    _C1Fixture(
        "clarification_request",
        "train",
        "Compare the two options for me.",
        "Which two options do you mean? I don't see any in our conversation.",
    ),
    _C1Fixture(
        "clarification_request",
        "heldout",
        "Send the draft to Alex.",
        "Do you mean Alex M. or Alex R.?",
        conversation_history=(
            "Earlier in this conversation:\n"
            "Alex M. asked about the draft budget.\n"
            "Alex R. asked about the draft schedule."
        ),
    ),
    _C1Fixture(
        "clarification_request",
        "regression",
        "Turn it up a bit, please.",
        "What should I turn up — the volume, the heat, or something else?",
    ),
    # correction_acceptance (3 train / 0 heldout / 1 regression)
    _C1Fixture(
        "correction_acceptance",
        "train",
        "Actually, the deadline is March 8, not the 5th.",
        "Thanks for the correction — the deadline is March 8.",
        conversation_history="Earlier assistant reply: The deadline is March 5.",
    ),
    _C1Fixture(
        "correction_acceptance",
        "train",
        "That's wrong — it's Canberra.",
        "You're right, thank you — the capital is Canberra.",
        conversation_history="Earlier assistant reply: The capital of Australia is Sydney.",
    ),
    _C1Fixture(
        "correction_acceptance",
        "train",
        "No, it opens at ten.",
        "Understood — I've corrected it: opening time is ten.",
        conversation_history="Earlier assistant reply: The shop opens at nine.",
    ),
    _C1Fixture(
        "correction_acceptance",
        "regression",
        "You're mistaken; the server room is on the second floor.",
        "Thanks for catching that — the server room is on the second floor.",
        conversation_history="Earlier assistant reply: The server room is on the third floor.",
    ),
    # uncertainty_admission (3 train / 1 heldout / 1 regression)
    _C1Fixture(
        "uncertainty_admission",
        "train",
        "What will the weather be in Oslo next Tuesday?",
        "I don't have any weather data here, so I can't say.",
    ),
    _C1Fixture(
        "uncertainty_admission",
        "train",
        "What did my colleague say about the budget?",
        "I don't have any record of that conversation, so I can't tell you.",
    ),
    _C1Fixture(
        "uncertainty_admission",
        "train",
        "How many tickets were sold yesterday?",
        "I don't know — I have no sales figures in front of me.",
    ),
    _C1Fixture(
        "uncertainty_admission",
        "heldout",
        "Who won the match last night?",
        "I don't have that information, so I can't say who won.",
    ),
    _C1Fixture(
        "uncertainty_admission",
        "regression",
        "What's the gate code for the storage unit?",
        "I don't know the code, and there's no record of it in front of me.",
    ),
    # turn_taking (3 train / 1 heldout / 0 regression)
    _C1Fixture(
        "turn_taking",
        "train",
        "Thanks, that's all for now.",
        "You're welcome! Have a good one.",
    ),
    _C1Fixture(
        "turn_taking",
        "train",
        "ok, bye",
        "Goodbye! Come back whenever you need.",
    ),
    _C1Fixture(
        "turn_taking",
        "train",
        "ha, that actually worked",
        "Glad it worked! What's next?",
    ),
    _C1Fixture(
        "turn_taking",
        "heldout",
        "great, talk later",
        "Sounds good — talk to you later!",
    ),
)


@dataclass(frozen=True, slots=True)
class C1TargetLeak:
    """One observed verbatim occurrence of a hidden target on a visible surface."""

    case_id: str
    surface: str
    detail: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "surface": self.surface, "detail": self.detail}


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _visible_surfaces(case: FirstFormCase) -> tuple[tuple[str, str], ...]:
    episode = case.episode
    surfaces = [
        (f"field:{region.name.value}", region.text) for region in episode.snapshot.regions
    ]
    surfaces.append(("workspace:first", episode.first_workspace_text))
    surfaces.append(("workspace:refined", episode.refined_workspace_text))
    surfaces.extend(
        (
            ("metadata:label", episode.label),
            ("metadata:competency", case.competency),
            ("metadata:lineage_id", case.lineage_id),
            ("metadata:mechanism_tags", " ".join(episode.mechanism_tags)),
            ("metadata:outcome_quality", episode.outcome_quality),
            ("metadata:target_basis", episode.target_basis),
        )
    )
    return tuple(surfaces)


def find_c1_target_leaks(cases: Iterable[FirstFormCase]) -> tuple[C1TargetLeak, ...]:
    """Find every verbatim target occurrence on any visible C1 surface."""

    items = tuple(cases)
    leaks: list[C1TargetLeak] = []
    for index, case in enumerate(items):
        targets = tuple(
            target.payload
            for target in case.episode.targets
            if target.decision is ReasoningDecision.DELTA and target.payload
        )
        own_surfaces = _visible_surfaces(case)
        sibling_surfaces = tuple(
            (f"sibling:{other.case_id[:16]}:{name}", text)
            for other_index, other in enumerate(items)
            if other_index != index
            for name, text in _visible_surfaces(other)
        )
        for payload in targets:
            folded = _normalize(payload)
            for surface, text in (*own_surfaces, *sibling_surfaces):
                if folded in _normalize(text):
                    leaks.append(
                        C1TargetLeak(
                            case_id=case.case_id,
                            surface=surface,
                            detail="hidden target payload appears verbatim on a visible surface",
                        )
                    )
    return tuple(leaks)


def verify_c1_no_target_leakage(cases: Iterable[FirstFormCase]) -> None:
    """Raise unless every C1 hidden target is absent from every visible surface."""

    leaks = find_c1_target_leaks(cases)
    if leaks:
        rendered = [
            {"case_id": leak.case_id[:16], "surface": leak.surface} for leak in leaks
        ]
        raise ValueError(f"C1 hidden-target leakage detected: {rendered}")


def _workspace(pass_kind: ProposalPass, detail: str) -> str:
    return ProposalWorkspace(
        image_id="c1-v1-derived-image",
        tick_uid="c1-v1-derived-tick",
        pass_kind=pass_kind,
        entries=(
            ProposalWorkspaceEntry(
                core_id="c1-brother-fixture",
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
    identity_text: str,
    fixture: _C1Fixture,
    lineage: str,
) -> LivingReasoningEpisode:
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: identity_text,
            LogicalRegion.CONVERSATION_HISTORY: fixture.conversation_history,
            LogicalRegion.USER_INPUT: fixture.user_text,
            LogicalRegion.CORTEX: fixture.cortex_text,
            LogicalRegion.SCRATCH: "",
            LogicalRegion.RESPONSE_DRAFT: "",
        },
        source_manifest_ids=(C1_AUTHORED_SOURCE_ID,),
    )
    detail = _WORKSPACE_DETAILS[fixture.competency]
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=_workspace(ProposalPass.FIRST, detail),
        refined_workspace_text=_workspace(ProposalPass.REFINED, detail),
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
                payload=fixture.answer,
            ),
        ),
        mechanism_tags=(
            "c1",
            "communication_first",
            "hidden_target",
            "single_turn",
            fixture.competency,
        ),
        outcome_quality=(
            "authored_conversational_process_evidence"
            if fixture.competency in _PROCESS_COMPETENCIES
            else "authored_objectively_checkable_target"
        ),
        source_example_id=canonical_sha256(
            {"schema": "axon-c1-fixture-v1", "lineage": lineage}
        ),
        target_basis=_TARGET_BASIS[fixture.competency],
    )


def _case(
    *,
    fixture: _C1Fixture,
    identity_text: str,
    sequence: int,
) -> FirstFormCase:
    lineage = f"c1-{fixture.competency}:{fixture.split}:{sequence:03d}"
    episode = _episode(
        label=f"c1-{fixture.competency}-{fixture.split}-{sequence:03d}",
        split=fixture.split,
        identity_text=identity_text,
        fixture=fixture,
        lineage=lineage,
    )
    compiled = D64FieldCompiler().compile(episode.snapshot)
    compiled.verify_roundtrip(episode.snapshot)
    pages = len(tuple(compiled.iter_character_pages(32)))
    return FirstFormCase(
        family=C1_FAMILY,
        competency=fixture.competency,
        eligibility=(
            TeachingEligibility.PROCESS_EVIDENCE
            if fixture.competency in _PROCESS_COMPETENCIES
            else TeachingEligibility.VERIFIED_TARGET
        ),
        lineage_id=lineage,
        source_record_ids=(),
        episode=episode,
        transport_pages=pages,
        procedural_depth=fixture.procedural_depth,
        derived=True,
    )


class C1CurriculumCompiler:
    """Compiles the authored C1 fixture corpus into a FirstFormCurriculum."""

    def compile(
        self,
        *,
        identity_text: str,
        requested_counts: tuple[
            tuple[str, tuple[int, int, int]], ...
        ] = DEFAULT_C1_SPLIT_COUNTS,
    ) -> FirstFormCurriculum:
        if not identity_text:
            raise ValueError("C1 requires the ratified canonical Identity")
        budgets = {family: dict(zip(_SPLITS, counts, strict=True)) for family, counts in requested_counts}
        if set(budgets) != {C1_FAMILY}:
            raise ValueError("C1 compile requires exact C1 budgets")
        available = {split: 0 for split in _SPLITS}
        for fixture in _C1_FIXTURES:
            available[fixture.split] += 1
        requested = budgets[C1_FAMILY]
        shortfall = {
            split: requested[split] - available[split]
            for split in _SPLITS
            if requested[split] > available[split]
        }
        if shortfall:
            raise ValueError(f"C1 authored fixtures cannot satisfy declared budgets: {shortfall}")

        selected = {split: 0 for split in _SPLITS}
        cases: list[FirstFormCase] = []
        for sequence, fixture in enumerate(_C1_FIXTURES):
            if selected[fixture.split] >= requested[fixture.split]:
                continue
            cases.append(_case(fixture=fixture, identity_text=identity_text, sequence=sequence))
            selected[fixture.split] += 1
        excluded = sum(available[split] - requested[split] for split in _SPLITS)

        competencies = {case.competency for case in cases}
        if requested == available and competencies != set(C1_COMPETENCIES):
            raise ValueError(f"C1 budget selection dropped lesson classes: {competencies}")
        lineage_splits: dict[str, set[str]] = {}
        for case in cases:
            lineage_splits.setdefault(case.lineage_id, set()).add(case.episode.split)
        if any(len(splits) != 1 for splits in lineage_splits.values()):
            raise ValueError("C1 lineage crosses a train/heldout/regression split")
        verify_c1_no_target_leakage(cases)
        return FirstFormCurriculum(
            cases=tuple(cases),
            source_import_ids=(C1_AUTHORED_SOURCE_ID,),
            requested_family_split_counts=requested_counts,
            excluded_counts=(() if excluded == 0 else (("c1_fixtures_not_selected", excluded),)),
            identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
        )


def publish_c1_curriculum(
    curriculum: FirstFormCurriculum,
    *,
    state_root: Path | str,
) -> Path:
    root = Path(state_root).resolve() / "training" / "curricula" / "c1"
    final = root / curriculum.manifest_id / "manifest.json"
    body = curriculum.to_canonical_dict()
    if final.exists():
        observed = json.loads(final.read_text(encoding="utf-8"))
        if canonical_json_bytes(observed) != canonical_json_bytes(body):
            raise ValueError("immutable C1 manifest disagrees with existing artifact")
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
    "C1_AUTHORED_SOURCE_ID",
    "C1_COMPETENCIES",
    "C1_FAMILY",
    "DEFAULT_C1_SPLIT_COUNTS",
    "C1CurriculumCompiler",
    "C1TargetLeak",
    "find_c1_target_leaks",
    "publish_c1_curriculum",
    "verify_c1_no_target_leakage",
]
