"""Evaluation metrics and honest baselines for Axon curriculum episodes.

Every number here comes from comparing a prediction against the episode's
``expected`` block, never from loss. The metrics answer the questions the
curriculum brief asks for directly: exact response accuracy, per-character
accuracy, binding errors (a stored fact, but the wrong one), obsolete-memory
errors (a superseded value), WAIT/COMMIT/END correctness, invalid content
(any character outside the native 95), and WAIT confusion (a COMMIT/END
control where WAIT was required).

The results row shape (:data:`RESULTS_FIELDS`, schema
``axon-curriculum-results-v1``) records the curriculum version and hash, the
split, the difficulty knobs and the architecture/checkpoint identity, so every
evaluated run can be tied back to exact bytes and exact weights.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

from substrate.native import ALPHABET

from .schema import CONTROL_COMMIT, CONTROL_END, CONTROL_WAIT, RESULTS_SCHEMA_ID

__all__ = [
    "RESULTS_FIELDS",
    "RESULTS_SCHEMA_ID",
    "baselines",
    "evaluate_episode",
    "expected_control",
    "expected_text",
]

#: Fields of one ``axon-curriculum-results-v1`` row, in report order.
#: ``architecture_id`` and ``checkpoint_id`` are nullable: results recorded
#: before a checkpoint exists carry null rather than a guess.
RESULTS_FIELDS: tuple[str, ...] = (
    "episode_id",
    "family",
    "split",
    "curriculum_version",
    "curriculum_sha256",
    "difficulty",
    "prediction_text",
    "prediction_control",
    "metrics",
    "seed",
    "architecture_id",
    "checkpoint_id",
)

_NATIVE_CHARACTERS = frozenset(ALPHABET)

#: Family-specific metadata names that may also carry stored facts, tolerated
#: in addition to the schema's own ``facts`` list.
_FACT_KEYS = frozenset({"stored_facts", "associations"})

#: Names under which a correction episode may record superseded values.
_OBSOLETE_KEYS = frozenset(
    {
        "obsolete",
        "obsolete_values",
        "superseded",
        "superseded_values",
        "previous",
        "previous_values",
        "stale",
        "stale_values",
        "old",
        "old_values",
    }
)


def _expected_block(episode: dict) -> dict:
    expected = episode.get("expected")
    if not isinstance(expected, dict):
        raise ValueError(f"episode has no 'expected' object; got {type(expected).__name__}")
    return expected


def expected_text(episode: dict) -> str:
    """The episode's exact expected response text ("" for control-only answers)."""
    value = _expected_block(episode).get("text", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"expected.text must be a str; got {type(value).__name__}")
    return value


def expected_control(episode: dict) -> Optional[int]:
    """The episode's expected WAIT/COMMIT/END label, or None when unlabeled."""
    control = _expected_block(episode).get("control")
    if control is None:
        return None
    if isinstance(control, bool) or not isinstance(control, int):
        raise ValueError(f"expected.control must be an int or None; got {control!r}")
    return control


def _collect_text(value: Any, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_text(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_text(item, out)


def _named_texts(container: Any, names: frozenset[str], depth: int = 0) -> list[str]:
    """Texts stored under any of ``names``, anywhere in the structure (bounded)."""
    found: list[str] = []
    if depth > 4:
        return found
    if isinstance(container, dict):
        for key, value in container.items():
            if isinstance(key, str) and key.lower() in names:
                _collect_text(value, found)
            else:
                found.extend(_named_texts(value, names, depth + 1))
    elif isinstance(container, (list, tuple)):
        for item in container:
            found.extend(_named_texts(item, names, depth + 1))
    return found


def _stored_facts(episode: dict) -> list[str]:
    """Stored-fact candidates for binding errors.

    The schema's ``facts`` list holds either bare values (content families) or
    ``"<key> is <value>"`` lines (association families); the value after the
    first ``" is "`` is added too, because a wrong stored value is exactly the
    binding error the curriculum measures.
    """
    facts: list[str] = []
    declared = episode.get("facts")
    if isinstance(declared, list):
        facts.extend(fact for fact in declared if isinstance(fact, str))
    facts.extend(_named_texts(episode, _FACT_KEYS))
    values: list[str] = []
    for fact in facts:
        if " is " in fact:
            value = fact.split(" is ", 1)[1]
            if value:
                values.append(value)
    ordered: list[str] = []
    for candidate in facts + values:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _correction_superseded(episode: dict) -> list[str]:
    """Superseded values of a correction episode, derived from its own steps.

    Correction episodes store ``"<key> is <old>"`` and later
    ``"replace <key> with <new>"``; the answer is the latest value, so the
    earlier value is the obsolete one. Other families return no derivation.
    """
    if episode.get("family") != "correction":
        return []
    steps = episode.get("steps")
    if not isinstance(steps, list):
        return []
    replaced_key = None
    for step in steps:
        text = step.get("text") if isinstance(step, dict) else None
        if isinstance(text, str) and text.startswith("replace ") and " with " in text:
            replaced_key = text[len("replace ") :].split(" with ", 1)[0]
    if not replaced_key:
        return []
    prefix = replaced_key + " is "
    superseded = []
    for step in steps:
        text = step.get("text") if isinstance(step, dict) else None
        if isinstance(text, str) and text.startswith(prefix) and len(text) > len(prefix):
            superseded.append(text[len(prefix) :])
    return superseded


def _obsolete_values(episode: dict) -> list[str]:
    superseded = _named_texts(episode, _OBSOLETE_KEYS) + _correction_superseded(episode)
    ordered: list[str] = []
    for candidate in superseded:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def evaluate_episode(episode: dict, prediction_text: str, prediction_control: Optional[int]) -> dict:
    """Score one prediction against one episode's expected answer.

    Returns the metrics dict: ``exact_match``, ``per_char_accuracy``,
    ``binding_error``, ``obsolete_error``, ``control_correct``,
    ``invalid_content`` and ``wait_confusion``. Binding/obsolete/control
    metrics are ``None`` when the episode carries nothing to judge them
    against; everything else is a strict bool or float. Predicting outside the
    native 95 sets ``invalid_content`` (and fails closed: it is never matched
    against expected text by conversion). Predicting a superseded value is
    labelled as the more specific ``obsolete_error``, not also as a binding
    error.
    """
    if not isinstance(episode, dict):
        raise TypeError(f"episode must be a dict; got {type(episode).__name__}")
    if not isinstance(prediction_text, str):
        raise TypeError(f"prediction_text must be a str; got {type(prediction_text).__name__}")
    if prediction_control is not None and (
        isinstance(prediction_control, bool) or not isinstance(prediction_control, int)
    ):
        raise TypeError(f"prediction_control must be an int or None; got {prediction_control!r}")

    wanted_text = expected_text(episode)
    wanted_control = expected_control(episode)

    exact_match = prediction_text == wanted_text
    if wanted_text:
        aligned = sum(1 for wanted, got in zip(wanted_text, prediction_text) if wanted == got)
        per_char_accuracy = aligned / len(wanted_text)
    else:
        per_char_accuracy = 0.0
    invalid_content = any(character not in _NATIVE_CHARACTERS for character in prediction_text)

    facts = _stored_facts(episode)
    obsolete = _obsolete_values(episode)

    if obsolete:
        obsolete_error = (
            bool(prediction_text) and prediction_text != wanted_text and prediction_text in set(obsolete)
        )
    else:
        obsolete_error = None

    if facts:
        binding_error = (
            bool(prediction_text)
            and prediction_text != wanted_text
            and prediction_text in set(facts)
            and not obsolete_error
        )
    else:
        binding_error = None

    if wanted_control is None:
        control_correct = None
    else:
        control_correct = prediction_control == wanted_control
    wait_confusion = wanted_control == CONTROL_WAIT and prediction_control in (CONTROL_COMMIT, CONTROL_END)

    return {
        "exact_match": exact_match,
        "per_char_accuracy": per_char_accuracy,
        "binding_error": binding_error,
        "obsolete_error": obsolete_error,
        "control_correct": control_correct,
        "invalid_content": invalid_content,
        "wait_confusion": wait_confusion,
    }


def _most_common(texts: list[str]) -> dict[str, Any]:
    counts = Counter(texts)
    best = max(counts, key=lambda text: (counts[text], text))
    return {"text": best, "accuracy_upper_bound": counts[best] / len(texts)}


def baselines(episodes: Iterable[dict]) -> dict:
    """Simple, honest reference numbers a trained model must beat.

    ``constant`` is the single most common expected text with the accuracy of
    always predicting it; ``memoryless_most_common`` repeats that per family;
    ``control_wait_rate`` is the fraction of labeled episodes where the answer
    is WAIT (the score of always answering WAIT). An empty curriculum returns
    empty/zero numbers rather than raising.
    """
    episode_list = list(episodes)
    if not episode_list:
        return {
            "constant": {"text": "", "accuracy_upper_bound": 0.0},
            "memoryless_most_common": {},
            "control_wait_rate": 0.0,
        }

    texts = [expected_text(episode) for episode in episode_list]
    by_family: dict[str, list[str]] = {}
    for episode, text in zip(episode_list, texts):
        family = episode.get("family")
        by_family.setdefault(family if isinstance(family, str) else "", []).append(text)

    labeled_controls = [control for control in map(expected_control, episode_list) if control is not None]
    wait_rate = (
        sum(1 for control in labeled_controls if control == CONTROL_WAIT) / len(labeled_controls)
        if labeled_controls
        else 0.0
    )

    return {
        "constant": _most_common(texts),
        "memoryless_most_common": {
            family: _most_common(family_texts) for family, family_texts in sorted(by_family.items())
        },
        "control_wait_rate": wait_rate,
    }
