"""Deterministic validators for the L0-L4 Language Foundations curriculum.

Three compile-time gates:

- ``verify_language_no_target_leakage``: HIDDEN exact answers must not occur
  on any visible surface of their own lesson (field regions, workspace text,
  or metadata).  Unlike C1's sentence-level substring check, matching here is
  token-based (casefolded, punctuation-stripped, whitespace-collapsed): lesson
  answers are words and short phrases, so a plain substring match would report
  false leaks for single-letter payloads.  Sibling cases are not checked
  because episodes are compiled and trained independently; cross-split
  vocabulary exposure is governed by the transfer-disjointness gate instead.
- ``verify_language_visibility_consistency``: the declared visibility must
  match the measured outcome in both directions — HIDDEN means absent,
  VISIBLE_BY_DESIGN means present.  The flag records the verified truth.
- ``verify_language_transfer_disjointness``: within each competency, items
  tagged ``transfer_item:...`` must be pairwise disjoint across the
  train/heldout/regression splits, so heldout gates measure rule transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from runtime.field import LogicalRegion
from runtime.heart import ReasoningDecision

from ..first_form_curriculum import FirstFormCase
from .contracts import (
    TRANSFER_ITEM_PREFIX,
    VISIBILITY_HIDDEN_TAG,
    VISIBILITY_VISIBLE_TAG,
)

_TOKEN_EDGE_PUNCTUATION = " \t.,;:!?\"'`\u201c\u201d\u2018\u2019\u2013\u2014-()[]{}<>*/\\|_"

_SPLITS = ("train", "heldout", "regression")


def _tokens(text: str) -> tuple[str, ...]:
    """Casefolded, whitespace-collapsed, edge-punctuation-stripped tokens."""

    folded = " ".join(text.casefold().split())
    return tuple(
        stripped
        for token in folded.split(" ")
        if (stripped := token.strip(_TOKEN_EDGE_PUNCTUATION))
    )


def _contains_tokens(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    limit = len(haystack) - len(needle) + 1
    return any(haystack[index : index + len(needle)] == needle for index in range(limit))


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


def _lesson_surfaces(case: FirstFormCase) -> tuple[tuple[str, str], ...]:
    """Lesson-specific visible surfaces, excluding the constant Identity region.

    Identity is the constitutional backdrop attended in every episode of every
    curriculum; its background vocabulary (including Axon terms and ordinary
    function words) is not lesson evidence and cannot serve as the answer
    source for a hidden lesson.  Multi-token hidden payloads are still checked
    against Identity by ``find_language_target_leaks`` because a verbatim
    multi-word answer in the constitution would be a genuine leak.
    """

    return tuple(
        (name, text) for name, text in _visible_surfaces(case) if name != "field:identity"
    )


def _supervised_payloads(case: FirstFormCase) -> tuple[str, ...]:
    return tuple(
        target.payload
        for target in case.episode.targets
        if target.decision is ReasoningDecision.DELTA and target.payload
    )


def _visibility_tag(case: FirstFormCase) -> str | None:
    tags = case.episode.mechanism_tags
    if VISIBILITY_HIDDEN_TAG in tags:
        return VISIBILITY_HIDDEN_TAG
    if VISIBILITY_VISIBLE_TAG in tags:
        return VISIBILITY_VISIBLE_TAG
    return None


@dataclass(frozen=True, slots=True)
class LanguageTargetLeak:
    """One observed token-exact occurrence of a hidden target on a visible surface."""

    case_id: str
    surface: str
    detail: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "surface": self.surface, "detail": self.detail}


def find_language_target_leaks(cases: Iterable[FirstFormCase]) -> tuple[LanguageTargetLeak, ...]:
    """Find every hidden target that is token-visible on its own lesson surfaces.

    Lesson evidence surfaces exclude the constant Identity region; the
    Identity text is still checked for multi-token payloads, whose verbatim
    presence there would be a genuine leak rather than background vocabulary.
    """

    leaks: list[LanguageTargetLeak] = []
    for case in cases:
        if _visibility_tag(case) != VISIBILITY_HIDDEN_TAG:
            continue
        surfaces = tuple((name, _tokens(text)) for name, text in _lesson_surfaces(case))
        identity_text = case.episode.snapshot.region(LogicalRegion.IDENTITY).text
        identity_tokens = _tokens(identity_text)
        for payload in _supervised_payloads(case):
            needle = _tokens(payload)
            for surface, tokens in surfaces:
                if _contains_tokens(tokens, needle):
                    leaks.append(
                        LanguageTargetLeak(
                            case_id=case.case_id,
                            surface=surface,
                            detail="hidden target payload is token-visible on a visible surface",
                        )
                    )
            if len(needle) > 1 and _contains_tokens(identity_tokens, needle):
                leaks.append(
                    LanguageTargetLeak(
                        case_id=case.case_id,
                        surface="field:identity",
                        detail="multi-token hidden target appears verbatim in canonical Identity",
                    )
                )
    return tuple(leaks)


def verify_language_no_target_leakage(cases: Iterable[FirstFormCase]) -> None:
    """Raise unless every hidden language target is absent from its own surfaces."""

    items = tuple(cases)
    leaks = find_language_target_leaks(items)
    if leaks:
        rendered = [
            {"case_id": leak.case_id[:16], "surface": leak.surface} for leak in leaks
        ]
        raise ValueError(f"language hidden-target leakage detected: {rendered}")


def verify_language_visibility_consistency(cases: Iterable[FirstFormCase]) -> None:
    """Raise unless declared visibility matches the measured outcome both ways."""

    problems: list[dict[str, str]] = []
    for case in cases:
        tag = _visibility_tag(case)
        if tag is None:
            problems.append({"case_id": case.case_id[:16], "detail": "missing visibility tag"})
            continue
        surfaces = tuple(_tokens(text) for _, text in _lesson_surfaces(case))
        for payload in _supervised_payloads(case):
            needle = _tokens(payload)
            found = any(_contains_tokens(tokens, needle) for tokens in surfaces)
            if tag == VISIBILITY_HIDDEN_TAG and found:
                problems.append(
                    {"case_id": case.case_id[:16], "detail": "hidden target is visible"}
                )
            if tag == VISIBILITY_VISIBLE_TAG and not found:
                problems.append(
                    {
                        "case_id": case.case_id[:16],
                        "detail": "visible-by-design target is absent from visible surfaces",
                    }
                )
    if problems:
        raise ValueError(f"language visibility inconsistency detected: {problems}")


def verify_language_transfer_disjointness(cases: Iterable[FirstFormCase]) -> None:
    """Raise unless transfer items are pairwise split-disjoint per competency."""

    pools: dict[str, dict[str, set[str]]] = {}
    for case in cases:
        items = tuple(
            tag[len(TRANSFER_ITEM_PREFIX) :]
            for tag in case.episode.mechanism_tags
            if tag.startswith(TRANSFER_ITEM_PREFIX)
        )
        if not items:
            continue
        splits = pools.setdefault(case.competency, {split: set() for split in _SPLITS})
        splits[case.episode.split].update(items)
    violations: list[dict[str, str]] = []
    for competency, splits in pools.items():
        for left_index, left in enumerate(_SPLITS):
            for right in _SPLITS[left_index + 1 :]:
                overlap = splits[left] & splits[right]
                if overlap:
                    violations.append(
                        {
                            "competency": competency,
                            "splits": f"{left}/{right}",
                            "items": ",".join(sorted(overlap)[:4]),
                        }
                    )
    if violations:
        raise ValueError(f"language transfer items cross splits: {violations}")


__all__ = [
    "LanguageTargetLeak",
    "find_language_target_leaks",
    "verify_language_no_target_leakage",
    "verify_language_transfer_disjointness",
    "verify_language_visibility_consistency",
]
