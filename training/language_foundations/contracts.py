"""Shared contracts for the L0-L4 Language Foundations curriculum.

Implements the first bounded campaign of the Language Foundations Schoolhouse
proposal (``roundtable/proposals/CHATGPT_LANGUAGE_FOUNDATIONS_SCHOOLHOUSE_PROPOSAL_2026-08-31.md``
section 17): objective, tightly structured lessons for character/transport
fluency (L0), spelling/punctuation (L1), vocabulary (L2), morphology (L3) and
grammar mechanics (L4).

Design laws honored here:

- Runtime fidelity.  Every lesson is a normal ``SharedFieldSnapshot`` compiled
  through the unchanged ``D64FieldCompiler`` and supervised through the same
  three-phase ``LivingReasoningEpisode`` shape as FFCS/C1.  No tokenizer, no
  parallel text path, no training-only surface.
- Objective targets only.  Every v1 case is ``VERIFIED_TARGET`` with an exact
  transport, categorical-choice, exact addressed-edit, or no-op-decision
  target.  Semantic-content and PROCESS_EVIDENCE targets are deliberately
  absent.  The Trainer derives a content-addressed ``VERIFIED_TARGET`` teaching
  view and rejects any other eligibility class at its optimizer boundary.
- Rule transfer over memorization.  Heldout and regression items use unseen
  words, stems, and sentence subjects; the compiler proves split disjointness.
- Budgets are campaign discipline, never doctrine.  All counts are revisable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from runtime.field import canonical_sha256
from runtime.heart import ReasoningOperationKind

L0_FAMILY = "L0"
L1_FAMILY = "L1"
L2_FAMILY = "L2"
L3_FAMILY = "L3"
L4_FAMILY = "L4"
LANGUAGE_FOUNDATIONS_FAMILIES: tuple[str, ...] = (
    L0_FAMILY,
    L1_FAMILY,
    L2_FAMILY,
    L3_FAMILY,
    L4_FAMILY,
)

LANGUAGE_FOUNDATIONS_AUTHORED_SOURCE_ID = canonical_sha256(
    {"schema": "axon-language-foundations-authored-fixture-corpus-v1", "fixture_version": 1}
)

_SPLITS = ("train", "heldout", "regression")

# Validator kinds are carried in episode mechanism_tags (never as new
# FirstFormCase fields, which would change case_id hashes of published
# manifests).  All four are objective and machine-checkable.
VALIDATOR_EXACT_TRANSPORT = "target_validator:exact_transport"
VALIDATOR_CATEGORICAL_CHOICE = "target_validator:categorical_choice"
VALIDATOR_EXACT_ADDRESSED_EDIT = "target_validator:exact_addressed_edit"
VALIDATOR_NO_OP_DECISION = "target_validator:no_op_decision"
VALIDATOR_TAGS: tuple[str, ...] = (
    VALIDATOR_EXACT_TRANSPORT,
    VALIDATOR_CATEGORICAL_CHOICE,
    VALIDATOR_EXACT_ADDRESSED_EDIT,
    VALIDATOR_NO_OP_DECISION,
)

VISIBILITY_HIDDEN_TAG = "visibility:hidden"
VISIBILITY_VISIBLE_TAG = "visibility:visible_by_design"
TRANSFER_ITEM_PREFIX = "transfer_item:"


class TargetVisibility(str, Enum):
    """Whether the exact target is readable from the lesson's visible field.

    HIDDEN means the exact target string must not occur on any visible
    surface; the compiler enforces this.  VISIBLE_BY_DESIGN means the target
    is an exact copy, an option selection, or a transformation of visible
    text, and the compiler asserts it really is visible.  The visibility flag
    records the actual verified outcome; it never weakens the check.
    """

    HIDDEN = "hidden"
    VISIBLE_BY_DESIGN = "visible_by_design"


# Developmental prerequisites (proposal section 16, Hermes layer): a competency
# graph, not a hard architecture ladder.  Keys are competency IDs used by the
# band modules; values are competencies that should be stable first.
COMPETENCY_PREREQUISITES: Mapping[str, tuple[str, ...]] = {
    # L0 — character and transport fluency
    "exact_copy": (),
    "minimal_pair_selection": ("exact_copy",),
    "capitalization_pair": ("exact_copy",),
    "punctuation_copy": ("exact_copy",),
    "no_op_distinction": (),
    "addressed_edit": ("exact_copy",),
    # L1 — spelling, capitalization, punctuation
    "spelling_repair": ("exact_copy", "minimal_pair_selection"),
    "plural_spelling": ("exact_copy",),
    "capitalization": ("capitalization_pair",),
    "terminal_punctuation": ("punctuation_copy",),
    "apostrophe_contraction": ("punctuation_copy",),
    "sentence_spacing": ("punctuation_copy",),
    # L2 — vocabulary and lexical relations
    "definition_to_word": ("exact_copy",),
    "word_to_definition": ("exact_copy",),
    "synonym": ("word_to_definition",),
    "antonym": ("word_to_definition",),
    "category_membership": ("word_to_definition",),
    "context_selection": ("word_to_definition",),
    "axon_domain_vocabulary": ("definition_to_word", "word_to_definition"),
    # L3 — morphology
    "plural_regular": ("plural_spelling",),
    "past_tense_regular": ("spelling_repair",),
    "progressive": ("spelling_repair",),
    "third_person": ("spelling_repair",),
    "comparative_superlative": ("spelling_repair",),
    "derivation": ("word_to_definition",),
    "irregular": ("past_tense_regular", "plural_regular"),
    # L4 — grammar mechanics
    "subject_verb_agreement": ("third_person", "plural_regular"),
    "tense_choice": ("past_tense_regular", "irregular"),
    "pronoun_reference": ("word_to_definition",),
    "articles": ("exact_copy",),
    "negation": ("subject_verb_agreement",),
    "question_formation": ("subject_verb_agreement",),
    "conjunction_join": ("subject_verb_agreement",),
    "sentence_ordering": ("exact_copy", "subject_verb_agreement"),
    "grammatical_choice": ("subject_verb_agreement",),
    "sentence_repair": ("spelling_repair", "subject_verb_agreement"),
}


@dataclass(frozen=True, slots=True)
class LanguageLessonSpec:
    """One authored L0-L4 lesson before compilation into a FirstFormCase."""

    family: str
    competency: str
    split: str
    prompt: str
    answer: str
    visibility: TargetVisibility
    validator: str
    target_basis: str
    cortex_text: str = ""
    conversation_history: str = ""
    draft_starter: str = ""
    edit: tuple[ReasoningOperationKind, int, int] | None = None
    transfer_item: str | None = None
    procedural_depth: int = 1

    def __post_init__(self) -> None:
        if self.family not in LANGUAGE_FOUNDATIONS_FAMILIES:
            raise ValueError("language lesson family must be L0-L4")
        if self.split not in _SPLITS:
            raise ValueError("language lesson split must be train, heldout, or regression")
        if not self.competency or not self.prompt:
            raise ValueError("language lesson competency and prompt must be non-empty")
        if self.competency not in COMPETENCY_PREREQUISITES:
            raise ValueError(f"unregistered language competency: {self.competency}")
        if self.validator not in VALIDATOR_TAGS:
            raise ValueError("language lesson validator must be a declared validator tag")
        if self.validator == VALIDATOR_NO_OP_DECISION and (
            self.answer or self.edit is not None or not self.draft_starter
        ):
            raise ValueError("no-op lessons carry no answer/edit and a completed draft")
        is_delete_edit = (
            self.edit is not None and self.edit[0] is ReasoningOperationKind.DELETE
        )
        if self.validator != VALIDATOR_NO_OP_DECISION and not self.answer and not is_delete_edit:
            raise ValueError("supervised language lessons require a non-empty answer")
        if self.edit is not None:
            kind, start, end = self.edit
            if not isinstance(kind, ReasoningOperationKind):
                raise ValueError("edit kind must be a ReasoningOperationKind")
            if start < 0 or end < start:
                raise ValueError("edit address must be ordered and non-negative")
            if self.draft_starter and end > len(self.draft_starter):
                raise ValueError("edit address must stay inside the draft starter")
        if self.procedural_depth < 1:
            raise ValueError("procedural depth must be positive")


__all__ = [
    "COMPETENCY_PREREQUISITES",
    "L0_FAMILY",
    "L1_FAMILY",
    "L2_FAMILY",
    "L3_FAMILY",
    "L4_FAMILY",
    "LANGUAGE_FOUNDATIONS_AUTHORED_SOURCE_ID",
    "LANGUAGE_FOUNDATIONS_FAMILIES",
    "TRANSFER_ITEM_PREFIX",
    "VALIDATOR_CATEGORICAL_CHOICE",
    "VALIDATOR_EXACT_ADDRESSED_EDIT",
    "VALIDATOR_EXACT_TRANSPORT",
    "VALIDATOR_NO_OP_DECISION",
    "VALIDATOR_TAGS",
    "VISIBILITY_HIDDEN_TAG",
    "VISIBILITY_VISIBLE_TAG",
    "LanguageLessonSpec",
    "TargetVisibility",
]
