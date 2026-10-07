"""Architecture-independent episode schema for the Axon native-95 curriculum.

Nothing here knows about GRUs, lanes, tensors or any other architecture. An
episode is a plain, JSON-serializable dict: a list of steps the curriculum
decides, one query, and the expected result. Every generator, validator, split
manifest and results report in ``curriculum/`` speaks this shape, so the E0
two-state organism and later architectures can consume the same tasks.

LAW NOTES (non-negotiable):

* Exactly the 95 frozen native characters are admitted. Every text-bearing
  field is checked against ``substrate.native.ALPHABET`` by membership. Outside
  characters fail closed: nothing is converted, escaped, normalized or clamped
  on the way in. (Newline is native; TAB and backtick are not.)
* WAIT/zero-commit is *control state*, never an all-EMPTY content surface. An
  ``"input"`` step must carry non-empty text; ``"tick"``/``"reset"`` steps carry
  no text at all; and ``steps[].control`` is always ``None``. The control triple
  (0 WAIT, 1 COMMIT, 2 END, same semantics as ``core.e0_two_state``) appears only
  in ``expected.control``.
* ``expected`` is the task's answer, never an input. Generators must not copy it
  into the steps or the query.

Two spellings of ``expected.control`` differ on purpose: this validator accepts
only an exact ``int`` (not ``bool``, not ``float``) because controls are hashed
and serialized, while a JSON Schema cannot distinguish ``1`` from ``1.0``.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any

from substrate.native import ALPHABET

__all__ = [
    "CONTROL_COMMIT",
    "CONTROL_END",
    "CONTROL_VALUES",
    "CONTROL_WAIT",
    "CURRICULUM_VERSION",
    "EPISODE_FIELDS",
    "EPISODE_JSON_SCHEMA",
    "EPISODE_SCHEMA_ID",
    "FAMILIES",
    "RESULTS_SCHEMA_ID",
    "SPLIT_SCHEMA_ID",
    "STEP_KINDS",
    "canonical_hash",
    "canonical_json",
    "validate_episode",
]

#: Version of the curriculum contract: bump when the episode shape changes.
CURRICULUM_VERSION = "0.1.0"

EPISODE_SCHEMA_ID = "axon-curriculum-episode-v1"
RESULTS_SCHEMA_ID = "axon-curriculum-results-v1"
SPLIT_SCHEMA_ID = "axon-curriculum-split-v1"

#: Control triple, identical in meaning to ``core.e0_two_state`` (defined here
#: locally so the schema stays architecture-independent: no core import).
CONTROL_WAIT = 0
CONTROL_COMMIT = 1
CONTROL_END = 2
CONTROL_VALUES = (CONTROL_WAIT, CONTROL_COMMIT, CONTROL_END)

STEP_KINDS = ("input", "tick", "reset")

#: The eight progressive task families of brief 1, in teaching order.
FAMILIES = (
    "char_copy",
    "delayed_recall",
    "distracted_recall",
    "key_value",
    "correction",
    "order_binding",
    "control",
    "generalization",
)

#: Every key an episode record must carry.
EPISODE_FIELDS = (
    "episode_id",
    "family",
    "seed",
    "steps",
    "query",
    "expected",
    "facts",
    "delay_ticks",
    "distractor_count",
    "difficulty",
    "split_key",
)

_NATIVE_CHARACTERS = frozenset(ALPHABET)
if len(_NATIVE_CHARACTERS) != 95:  # pragma: no cover - mirrors the substrate guard
    raise RuntimeError("the native alphabet must be exactly 95 distinct characters")

_OUTSIDE_SHOWN = 8

_JSON_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"

# Characters that need an escape to sit literally inside a regex character
# class. The alphabet itself is fixed, but building the class mechanically keeps
# the JSON Schema honest if the alphabet is ever re-derived.
_CLASS_ESCAPES = {
    "\\": "\\\\",
    "]": "\\]",
    "^": "\\^",
    "-": "\\-",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _native_text_pattern() -> str:
    """A regex that matches a string iff every character is native-95."""
    members = "".join(_CLASS_ESCAPES.get(character, character) for character in ALPHABET)
    return f"^[{members}]*$"


_NATIVE_TEXT_PATTERN = _native_text_pattern()


def _outside_characters(text: str) -> list[str]:
    return sorted({character for character in text if character not in _NATIVE_CHARACTERS})


def _is_int(value: Any) -> bool:
    """True for exact integers (numpy integers count); False for bool and float."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def _admit_text(value: Any, path: str, errors: list[str], *, require_non_empty: bool = False) -> None:
    """Record an error unless ``value`` is text whose every character is native."""
    if not isinstance(value, str):
        errors.append(f"{path} must be str; got {type(value).__name__}")
        return
    if require_non_empty and not value:
        errors.append(
            f"{path} must be non-empty: an all-EMPTY surface is WAIT/zero-commit control state, "
            "never episode content"
        )
    outside = _outside_characters(value)
    if outside:
        shown = ", ".join(f"U+{ord(character):04X}" for character in outside[:_OUTSIDE_SHOWN])
        errors.append(
            f"{path} has characters outside the native 95 and fails closed (nothing is converted, "
            f"escaped, normalized or clamped): {shown}"
        )


def _text_schema(**extra: Any) -> dict:
    schema: dict[str, Any] = {"type": "string", "pattern": _NATIVE_TEXT_PATTERN}
    schema.update(extra)
    return schema


def _step_schema() -> dict:
    return {
        "type": "object",
        "required": ["kind", "text", "control"],
        "properties": {
            "kind": {"enum": list(STEP_KINDS)},
            "text": _text_schema(),
            "control": {"type": "null", "description": "always null: control state never rides in a step"},
        },
        "allOf": [
            {
                "if": {"required": ["kind"], "properties": {"kind": {"const": "input"}}},
                "then": {"properties": {"text": {"minLength": 1}}},
                "description": "an input step carries real native content; WAIT is not spelled as empty text",
            },
            {
                "if": {"required": ["kind"], "properties": {"kind": {"enum": ["tick", "reset"]}}},
                "then": {"properties": {"text": {"maxLength": 0}}},
                "description": "tick and reset are control steps and carry no content",
            },
        ],
    }


def _expected_schema() -> dict:
    return {
        "type": "object",
        "required": ["text", "control"],
        "properties": {
            "text": _text_schema(description='the exact expected native text; "" only when the answer is control-only'),
            "control": {"enum": [*CONTROL_VALUES, None]},
        },
    }


def _episode_json_schema() -> dict:
    return {
        "$schema": _JSON_SCHEMA_URI,
        "$id": EPISODE_SCHEMA_ID,
        "title": "Axon curriculum episode",
        "description": (
            f"One curriculum episode, version {CURRICULUM_VERSION}. Extra keys are allowed for "
            "family-specific metadata; every text-bearing field admits only the 95 native characters."
        ),
        "type": "object",
        "required": list(EPISODE_FIELDS),
        "properties": {
            "episode_id": {"type": "string", "minLength": 1, "description": "stable unique id within a split"},
            "family": {"enum": list(FAMILIES)},
            "seed": {"type": "integer", "description": "generator seed that reproduces this episode exactly"},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": _step_schema(),
                "description": "input steps in order, with silent ticks and resets as separate steps",
            },
            "query": _text_schema(description="the retrieval/response prompt; carries no answer text"),
            "expected": _expected_schema(),
            "facts": {"type": "array", "items": _text_schema()},
            "delay_ticks": {"type": ["integer", "null"], "description": "silent ticks between writing and query"},
            "distractor_count": {"type": ["integer", "null"], "description": "visible distractor inputs presented"},
            "difficulty": {"type": "object", "description": "the adjustable knobs this episode was generated with"},
            "split_key": {
                "type": "string",
                "minLength": 1,
                "description": "leakage-aware grouping key (facts/templates/combinations), not a row number",
            },
        },
    }


EPISODE_JSON_SCHEMA: dict = _episode_json_schema()


def validate_episode(episode: dict) -> list[str]:
    """Return a list of plain-language errors; an empty list means the episode is valid.

    Never raises and never mutates: any object, however malformed, comes back as
    errors. Text outside the native 95 is reported, never repaired.
    """
    errors: list[str] = []
    if not isinstance(episode, dict):
        return [f"episode must be a dict; got {type(episode).__name__}"]

    for field in EPISODE_FIELDS:
        if field not in episode:
            errors.append(f"{field} is required")

    if "episode_id" in episode:
        episode_id = episode["episode_id"]
        if not isinstance(episode_id, str) or not episode_id:
            errors.append(f"episode_id must be a non-empty str; got {episode_id!r}")

    if "family" in episode:
        family = episode["family"]
        if family not in FAMILIES:
            errors.append(f"family must be one of {list(FAMILIES)}; got {family!r}")

    if "seed" in episode and not _is_int(episode["seed"]):
        errors.append(f"seed must be an int; got {episode['seed']!r}")

    if "steps" in episode:
        steps = episode["steps"]
        if not isinstance(steps, list):
            errors.append(f"steps must be a list; got {type(steps).__name__}")
        elif not steps:
            errors.append("steps must not be empty: an episode needs at least one step")
        else:
            _check_steps(steps, errors)

    if "query" in episode:
        _admit_text(episode["query"], "query", errors)

    if "expected" in episode:
        _check_expected(episode["expected"], errors)

    if "facts" in episode:
        facts = episode["facts"]
        if not isinstance(facts, list):
            errors.append(f"facts must be a list; got {type(facts).__name__}")
        else:
            for index, fact in enumerate(facts):
                _admit_text(fact, f"facts[{index}]", errors)

    for field in ("delay_ticks", "distractor_count"):
        if field in episode:
            value = episode[field]
            if value is not None and not _is_int(value):
                errors.append(f"{field} must be an int or null; got {value!r}")

    if "difficulty" in episode and not isinstance(episode["difficulty"], dict):
        errors.append(f"difficulty must be an object; got {type(episode['difficulty']).__name__}")

    if "split_key" in episode:
        split_key = episode["split_key"]
        if not isinstance(split_key, str) or not split_key:
            errors.append(f"split_key must be a non-empty str; got {split_key!r}")

    return errors


def _check_steps(steps: list, errors: list[str]) -> None:
    for index, step in enumerate(steps):
        path = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{path} must be an object; got {type(step).__name__}")
            continue

        kind = step.get("kind")
        if "kind" not in step:
            errors.append(f"{path}.kind is required")
        elif kind not in STEP_KINDS:
            errors.append(f"{path}.kind must be one of {list(STEP_KINDS)}; got {kind!r}")

        if "control" not in step:
            errors.append(f"{path}.control is required and must be null")
        elif step["control"] is not None:
            errors.append(
                f"{path}.control must be null; got {step['control']!r}. WAIT/COMMIT/END are control state "
                "and belong in expected.control only, never in a step"
            )

        if "text" not in step:
            errors.append(f"{path}.text is required")
            continue

        text = step["text"]
        _admit_text(text, f"{path}.text", errors)
        if not isinstance(text, str):
            continue
        if kind == "input" and not text:
            errors.append(
                f"{path}.text must be non-empty for kind 'input': a WAIT/zero-commit surface is control "
                "state, not empty content"
            )
        elif kind in ("tick", "reset") and text:
            errors.append(f"{path}.text must be empty for kind {kind!r}: control steps carry no content")


def _check_expected(expected: Any, errors: list[str]) -> None:
    if not isinstance(expected, dict):
        errors.append(f"expected must be an object; got {type(expected).__name__}")
        return

    if "text" not in expected:
        errors.append('expected.text is required (use "" when the expected response is control-only)')
    else:
        _admit_text(expected["text"], "expected.text", errors)

    if "control" not in expected:
        errors.append(f"expected.control is required (one of {list(CONTROL_VALUES)} or null)")
        return
    control = expected["control"]
    if control is None:
        return
    if not _is_int(control) or control not in CONTROL_VALUES:
        errors.append(f"expected.control must be one of {list(CONTROL_VALUES)} or null; got {control!r}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text for hashing: sorted keys, no padding, native text literal.

    ``ensure_ascii=False`` keeps native characters byte-for-byte in the encoded
    string; ``allow_nan=False`` refuses non-JSON floats instead of writing a
    ``NaN`` that no parser could read back.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_hash(obj: Any) -> str:
    """sha256 hex digest of ``canonical_json(obj)`` (utf-8), the repo-wide hash for episodes."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
