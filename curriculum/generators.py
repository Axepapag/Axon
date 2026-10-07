"""Deterministic, architecture-independent generators for the native-95 curriculum.

Eight progressive task families (matching ``curriculum.schema.FAMILIES``), all
built on the same laws:

* One dedicated ``random.Random(seed)`` per episode, draws taken in a fixed
  order, global randomness never touched. The same ``(family, seed,
  difficulty)`` always yields a byte-identical ``canonical_json`` and the same
  ``episode_id``.
* Only the frozen 95 native characters are generated: lowercase letters,
  digits, space and occasionally newline for content, with plain native
  phrasing elsewhere. Nothing is converted, escaped, normalized or clamped;
  the drawing pools are checked against ``substrate.native.ALPHABET`` at
  import time.
* WAIT/zero-commit is control state, never an empty content surface: every
  ``"input"`` step carries non-empty text, ``"tick"`` steps carry none, and
  WAIT/COMMIT/END appear only in ``expected.control``.
* The expected answer is never written into the query; it appears in the
  steps only where the task itself stores it (the fact line to be recalled).
* Every episode is validated with ``curriculum.schema.validate_episode``
  before it is returned; a failure raises ``ValueError`` with the joined
  errors.

Families and their causal rules:

``char_copy``          immediate exact copy of the shown content.
``delayed_recall``     content, then silent ticks, then recall it.
``distracted_recall``  content, silent ticks, then visible distractors that
                       must not be copied; recall the original.
``key_value``          store facts as "key is value" lines; answer the value
                       of one queried key.
``correction``         store a fact, then an overwrite instruction; the answer
                       is the latest value, never the obsolete one.
``order_binding``      store a short ordered list (or key/value pairs) and
                       answer about one position (or key).
``control``            WAIT/COMMIT/END targets: hold, reply with an exact
                       length, or end the episode.
``generalization``     unseen combinations: digit-heavy keys or a distractor
                       pattern no other family defaults to, with
                       ``novel_combination`` set in difficulty.

``split_key`` names the leakage unit of each episode: the template plus the
keys for association families (values excluded, so different values of the
same template share split placement), and the exact memorized content for
content families (so different contents spread across splits while one fact
never straddles two splits).
"""

from __future__ import annotations

import random
from typing import Any, Callable

from curriculum.schema import (
    CONTROL_COMMIT,
    CONTROL_END,
    CONTROL_WAIT,
    FAMILIES,
    canonical_hash,
    validate_episode,
)
from substrate.native import ALPHABET

__all__ = ["FAMILY_DEFAULT_DIFFICULTY", "generate_episode"]

# --- character pools, checked against the frozen alphabet at import time ---

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"
_BASIC_CHARS = _LOWER + _DIGITS
_KEY_LETTERS = "abcdef"
_SPACE = " "
_NEWLINE = "\n"

#: Every character the generators can ever emit (content pools plus the fixed
#: separators and question mark used in ids, split keys and queries).
_GENERATED_CHARACTERS = frozenset(_BASIC_CHARS + _SPACE + _NEWLINE + "-_,?:")
if not _GENERATED_CHARACTERS <= frozenset(ALPHABET):  # pragma: no cover - mirrors the substrate guard
    outside = sorted(_GENERATED_CHARACTERS - frozenset(ALPHABET))
    raise RuntimeError(
        f"curriculum generators would draw characters outside the native 95: {outside}"
    )

_SPACE_CHANCE = 0.20
_NEWLINE_CHANCE = 0.15
_VALUE_LENGTH = 3
_NOVEL_VALUE_LENGTH = 5
_REDRAW_LIMIT = 1000
_EPISODE_ID_HASH_LENGTH = 12

#: Knob defaults per family. ``None`` means "resolved per seed" (see builders).
FAMILY_DEFAULT_DIFFICULTY: dict[str, dict[str, Any]] = {
    "char_copy": {"length": 3},
    "delayed_recall": {"length": 3, "delay": 4},
    "distracted_recall": {
        "length": 3,
        "delay": 4,
        "distractors": 2,
        "distractor_kind": "noise",
    },
    "key_value": {"facts": 3, "delay": 4},
    "correction": {"delay": 4},
    "order_binding": {"items": 3, "delay": 4, "variant": None},
    "control": {"scenario": "commit", "length": 4},
    "generalization": {"novel_combination": True, "mechanic": None},
}

#: Default knobs each generalization mechanic fills in once it is resolved.
_GENERALIZATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "novel_key_value": {"facts": 5, "delay": 8},
    "novel_distracted_recall": {
        "length": 4,
        "delay": 8,
        "distractors": 6,
        "distractor_kind": "alternating",
    },
}

_ALLOWED_KNOBS: dict[str, tuple[str, ...]] = {
    "char_copy": ("length",),
    "delayed_recall": ("length", "delay"),
    "distracted_recall": ("length", "delay", "distractors", "distractor_kind"),
    "key_value": ("facts", "delay"),
    "correction": ("delay",),
    "order_binding": ("items", "delay", "variant"),
    "control": ("scenario", "length"),
    "generalization": (
        "novel_combination",
        "mechanic",
        "facts",
        "delay",
        "length",
        "distractors",
        "distractor_kind",
    ),
}

_DISTRACTOR_KINDS = ("noise", "similar", "repeated")
_GENERALIZATION_MECHANICS = ("novel_key_value", "novel_distracted_recall")
_CONTROL_SCENARIOS = ("wait", "commit", "end")
_ORDER_VARIANTS = ("order", "pairs")
_GENERALIZATION_DISTRACTOR_KIND = "alternating"

if set(FAMILY_DEFAULT_DIFFICULTY) != set(FAMILIES):  # pragma: no cover - mirrors the schema contract
    raise RuntimeError("FAMILY_DEFAULT_DIFFICULTY must cover exactly curriculum.schema.FAMILIES")


# --- steps and drawing helpers ---


def _input_step(text: str) -> dict[str, Any]:
    return {"kind": "input", "text": text, "control": None}


def _tick_step() -> dict[str, Any]:
    return {"kind": "tick", "text": "", "control": None}


def _ticks(count: int) -> list[dict[str, Any]]:
    return [_tick_step() for _ in range(count)]


def _draw_token(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_BASIC_CHARS) for _ in range(length))


def _draw_content(rng: random.Random, length: int) -> str:
    """Lowercase/digit content; occasionally one position becomes space or newline."""
    chars = [rng.choice(_BASIC_CHARS) for _ in range(length)]
    if rng.random() < _SPACE_CHANCE:
        chars[rng.randrange(length)] = _SPACE
    if rng.random() < _NEWLINE_CHANCE:
        chars[rng.randrange(length)] = _NEWLINE
    return "".join(chars)


def _draw_distinct(
    rng: random.Random, count: int, length: int, exclude: tuple[str, ...] = ()
) -> list[str]:
    """``count`` pairwise-distinct tokens, none of them in ``exclude``."""
    seen = set(exclude)
    tokens: list[str] = []
    for _ in range(count):
        for _attempt in range(_REDRAW_LIMIT):
            token = _draw_token(rng, length)
            if token not in seen:
                seen.add(token)
                tokens.append(token)
                break
        else:
            raise ValueError(f"could not draw {count} distinct tokens of length {length}")
    return tokens


def _draw_noise(rng: random.Random, length: int, exclude: set[str]) -> str:
    for _ in range(_REDRAW_LIMIT):
        token = _draw_token(rng, length)
        if token not in exclude:
            return token
    raise ValueError("could not draw a noise distractor distinct from the content")


def _draw_near_match(rng: random.Random, text: str) -> str:
    """A same-length token differing from ``text`` in exactly one position."""
    position = rng.randrange(len(text))
    original = text[position]
    pool = _BASIC_CHARS if original not in _BASIC_CHARS else _BASIC_CHARS.replace(original, "")
    chars = list(text)
    chars[position] = rng.choice(pool)
    return "".join(chars)


def _draw_distractors(rng: random.Random, content: str, count: int, kind: str) -> list[str]:
    if count == 0:
        return []
    if kind == "noise":
        return [_draw_noise(rng, len(content), {content}) for _ in range(count)]
    if kind == "similar":
        return [_draw_near_match(rng, content) for _ in range(count)]
    if kind == "repeated":
        decoy = _draw_noise(rng, len(content), {content})
        return [decoy for _ in range(count)]
    if kind == "alternating":
        first = _draw_noise(rng, len(content), {content})
        second = _draw_noise(rng, len(content), {content, first})
        return [first if index % 2 == 0 else second for index in range(count)]
    raise ValueError(f"unknown distractor kind {kind!r}")


# --- difficulty knobs ---


def _knob_int(difficulty: dict[str, Any], name: str, low: int, high: int) -> int:
    value = difficulty[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"difficulty[{name!r}] must be an int; got {value!r}")
    if not low <= value <= high:
        raise ValueError(
            f"difficulty[{name!r}] must be in {low}..{high}; got {value!r} (never clamped)"
        )
    return value


def _knob_choice(difficulty: dict[str, Any], name: str, choices: tuple[str, ...]) -> str:
    value = difficulty[name]
    if value not in choices:
        raise ValueError(f"difficulty[{name!r}] must be one of {list(choices)}; got {value!r}")
    return value


def _merged(family: str, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = dict(FAMILY_DEFAULT_DIFFICULTY[family])
    merged.update(difficulty)
    return merged


# --- family builders: each returns the task fields of one episode body ---


def _build_char_copy(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("char_copy", difficulty)
    length = _knob_int(merged, "length", 1, 8)
    content = _draw_content(rng, length)
    return {
        "steps": [_input_step(content)],
        "query": "copy the text",
        "expected": {"text": content, "control": None},
        "facts": [content],
        "delay_ticks": 0,
        "distractor_count": None,
        "difficulty": merged,
        "split_key": f"char_copy:copy:{content}",
    }


def _build_delayed_recall(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("delayed_recall", difficulty)
    length = _knob_int(merged, "length", 1, 6)
    delay = _knob_int(merged, "delay", 0, 32)
    content = _draw_content(rng, length)
    steps = [_input_step(content)] + _ticks(delay)
    return {
        "steps": steps,
        "query": "what was the text?",
        "expected": {"text": content, "control": None},
        "facts": [content],
        "delay_ticks": delay,
        "distractor_count": None,
        "difficulty": merged,
        "split_key": f"delayed_recall:recall:{content}",
    }


def _build_distracted_recall(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("distracted_recall", difficulty)
    length = _knob_int(merged, "length", 1, 6)
    delay = _knob_int(merged, "delay", 0, 32)
    count = _knob_int(merged, "distractors", 0, 8)
    kind = _knob_choice(merged, "distractor_kind", _DISTRACTOR_KINDS)
    content = _draw_content(rng, length)
    distractors = _draw_distractors(rng, content, count, kind)
    steps = [_input_step(content)] + _ticks(delay) + [_input_step(item) for item in distractors]
    return {
        "steps": steps,
        "query": "what was the first text?",
        "expected": {"text": content, "control": None},
        "facts": [content],
        "delay_ticks": delay,
        "distractor_count": count,
        "difficulty": merged,
        "split_key": f"distracted_recall:{kind}:{content}",
    }


def _build_key_value(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("key_value", difficulty)
    facts_count = _knob_int(merged, "facts", 1, 6)
    delay = _knob_int(merged, "delay", 0, 32)
    keys = rng.sample(_KEY_LETTERS, facts_count)
    values = _draw_distinct(rng, facts_count, _VALUE_LENGTH)
    queried = rng.randrange(facts_count)
    lines = [f"{key} is {value}" for key, value in zip(keys, values)]
    steps = [_input_step(line) for line in lines] + _ticks(delay)
    return {
        "steps": steps,
        "query": f"what is the value of {keys[queried]}?",
        "expected": {"text": values[queried], "control": None},
        "facts": lines,
        "delay_ticks": delay,
        "distractor_count": None,
        "difficulty": merged,
        "split_key": f"key_value:kv_line:{','.join(sorted(keys))}",
    }


def _build_correction(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("correction", difficulty)
    delay = _knob_int(merged, "delay", 0, 32)
    key = rng.choice(_KEY_LETTERS)
    old_value, new_value = _draw_distinct(rng, 2, _VALUE_LENGTH)
    store_line = f"{key} is {old_value}"
    overwrite_line = f"replace {key} with {new_value}"
    steps = [_input_step(store_line)] + _ticks(delay) + [_input_step(overwrite_line)]
    return {
        "steps": steps,
        "query": f"what is the current value of {key}?",
        "expected": {"text": new_value, "control": None},
        "facts": [store_line, f"{key} is {new_value}"],
        "delay_ticks": delay,
        "distractor_count": None,
        "difficulty": merged,
        "split_key": f"correction:overwrite_line:{key}",
    }


def _build_order_binding(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("order_binding", difficulty)
    items_count = _knob_int(merged, "items", 2, 5)
    delay = _knob_int(merged, "delay", 0, 32)
    variant = merged["variant"]
    if variant is None:
        variant = rng.choice(_ORDER_VARIANTS)
    elif variant not in _ORDER_VARIANTS:
        raise ValueError(
            f"difficulty['variant'] must be one of {list(_ORDER_VARIANTS)}; got {variant!r}"
        )
    merged["variant"] = variant

    if variant == "order":
        items = _draw_distinct(rng, items_count, _VALUE_LENGTH)
        position = rng.randrange(items_count)
        steps = [_input_step("\n".join(items))] + _ticks(delay)
        return {
            "steps": steps,
            "query": f"what is item {position + 1}?",
            "expected": {"text": items[position], "control": None},
            "facts": items,
            "delay_ticks": delay,
            "distractor_count": None,
            "difficulty": merged,
            "split_key": f"order_binding:ordered_list:{','.join(items)}",
        }

    keys = rng.sample(_KEY_LETTERS, items_count)
    values = _draw_distinct(rng, items_count, _VALUE_LENGTH)
    queried = rng.randrange(items_count)
    lines = [f"{key} is {value}" for key, value in zip(keys, values)]
    steps = [_input_step(line) for line in lines] + _ticks(delay)
    return {
        "steps": steps,
        "query": f"what is the value of {keys[queried]}?",
        "expected": {"text": values[queried], "control": None},
        "facts": lines,
        "delay_ticks": delay,
        "distractor_count": None,
        "difficulty": merged,
        "split_key": f"order_binding:pair_line:{','.join(sorted(keys))}",
    }


def _build_control(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("control", difficulty)
    scenario = _knob_choice(merged, "scenario", _CONTROL_SCENARIOS)
    length = _knob_int(merged, "length", 1, 32)
    content = _draw_content(rng, length)
    if scenario == "wait":
        query = "do not respond yet"
        expected = {"text": "", "control": CONTROL_WAIT}
    elif scenario == "commit":
        query = f"respond with exactly {length} characters"
        expected = {"text": content, "control": CONTROL_COMMIT}
    else:
        query = "end the episode"
        expected = {"text": "", "control": CONTROL_END}
    return {
        "steps": [_input_step(content)],
        "query": query,
        "expected": expected,
        "facts": [],
        "delay_ticks": 0,
        "distractor_count": None,
        "difficulty": merged,
        "split_key": f"control:{scenario}:{length}",
    }


def _build_generalization(rng: random.Random, difficulty: dict[str, Any]) -> dict[str, Any]:
    merged = _merged("generalization", difficulty)
    if merged["novel_combination"] is not True:
        raise ValueError(
            "generalization episodes are novel combinations by definition: "
            "difficulty['novel_combination'] must be true"
        )

    explicit = merged["mechanic"]
    if explicit is not None and explicit not in _GENERALIZATION_MECHANICS:
        raise ValueError(
            f"difficulty['mechanic'] must be one of {list(_GENERALIZATION_MECHANICS)}; "
            f"got {explicit!r}"
        )
    wants_facts = "facts" in difficulty
    wants_distracted = any(name in difficulty for name in ("length", "distractors", "distractor_kind"))
    if explicit is None:
        if wants_facts and wants_distracted:
            raise ValueError(
                "difficulty mixes novel_key_value and novel_distracted_recall knobs; "
                "set difficulty['mechanic'] explicitly"
            )
        if wants_facts:
            mechanic = "novel_key_value"
        elif wants_distracted:
            mechanic = "novel_distracted_recall"
        else:
            mechanic = rng.choice(_GENERALIZATION_MECHANICS)
    else:
        mechanic = explicit
        if mechanic == "novel_key_value" and wants_distracted:
            raise ValueError(
                "difficulty['length'/'distractors'/'distractor_kind'] belong to "
                "novel_distracted_recall, not to novel_key_value"
            )
        if mechanic == "novel_distracted_recall" and wants_facts:
            raise ValueError("difficulty['facts'] belongs to novel_key_value")
    merged["mechanic"] = mechanic
    for name, value in _GENERALIZATION_DEFAULTS[mechanic].items():
        merged.setdefault(name, value)

    if mechanic == "novel_key_value":
        facts_count = _knob_int(merged, "facts", 1, 8)
        delay = _knob_int(merged, "delay", 0, 32)
        keys = [str(number) for number in rng.sample(range(10, 100), facts_count)]
        values = _draw_distinct(rng, facts_count, _NOVEL_VALUE_LENGTH)
        queried = rng.randrange(facts_count)
        lines = [f"{key} is {value}" for key, value in zip(keys, values)]
        steps = [_input_step(line) for line in lines] + _ticks(delay)
        return {
            "steps": steps,
            "query": f"what is the value of {keys[queried]}?",
            "expected": {"text": values[queried], "control": None},
            "facts": lines,
            "delay_ticks": delay,
            "distractor_count": None,
            "difficulty": merged,
            "split_key": f"generalization:novel_key_value:{','.join(sorted(keys))}",
        }

    length = _knob_int(merged, "length", 1, 8)
    delay = _knob_int(merged, "delay", 0, 32)
    count = _knob_int(merged, "distractors", 0, 12)
    kind = _knob_choice(merged, "distractor_kind", (_GENERALIZATION_DISTRACTOR_KIND,))
    content = _draw_content(rng, length)
    distractors = _draw_distractors(rng, content, count, kind)
    steps = [_input_step(content)] + _ticks(delay) + [_input_step(item) for item in distractors]
    return {
        "steps": steps,
        "query": "what was the first text?",
        "expected": {"text": content, "control": None},
        "facts": [content],
        "delay_ticks": delay,
        "distractor_count": count,
        "difficulty": merged,
        "split_key": f"generalization:novel_distracted_recall:{kind}:{content}",
    }


_FAMILY_BUILDERS: dict[str, Callable[[random.Random, dict[str, Any]], dict[str, Any]]] = {
    "char_copy": _build_char_copy,
    "delayed_recall": _build_delayed_recall,
    "distracted_recall": _build_distracted_recall,
    "key_value": _build_key_value,
    "correction": _build_correction,
    "order_binding": _build_order_binding,
    "control": _build_control,
    "generalization": _build_generalization,
}


def generate_episode(family: str, seed: int, difficulty: dict | None = None) -> dict:
    """Build one validated curriculum episode from ``(family, seed, difficulty)``.

    ``difficulty`` overrides the family defaults in
    ``FAMILY_DEFAULT_DIFFICULTY`` and is never mutated. The result is fully
    determined by the arguments: the same triple always produces
    byte-identical ``canonical_json`` and the same ``episode_id`` (its short
    hash covers the episode body without the id, which would otherwise be
    circular). Unknown families, unknown difficulty keys, non-integer seeds
    and out-of-range knob values fail closed with ``ValueError``; nothing is
    clamped, converted or silently defaulted.
    """
    if family not in _FAMILY_BUILDERS:
        raise ValueError(f"family must be one of {list(FAMILIES)}; got {family!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int; got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0; got {seed!r}")
    if difficulty is None:
        difficulty = {}
    if not isinstance(difficulty, dict):
        raise ValueError(f"difficulty must be a dict or None; got {type(difficulty).__name__}")
    unknown = sorted(set(difficulty) - set(_ALLOWED_KNOBS[family]))
    if unknown:
        raise ValueError(
            f"unknown difficulty keys for {family}: {unknown}; "
            f"allowed: {list(_ALLOWED_KNOBS[family])}"
        )

    rng = random.Random(seed)
    task = _FAMILY_BUILDERS[family](rng, difficulty)
    body = {
        "family": family,
        "seed": seed,
        "steps": task["steps"],
        "query": task["query"],
        "expected": task["expected"],
        "facts": task["facts"],
        "delay_ticks": task["delay_ticks"],
        "distractor_count": task["distractor_count"],
        "difficulty": task["difficulty"],
        "split_key": task["split_key"],
    }
    short_hash = canonical_hash(body)[:_EPISODE_ID_HASH_LENGTH]
    episode = {"episode_id": f"{family}-{seed:x}-{short_hash}", **body}
    errors = validate_episode(episode)
    if errors:
        raise ValueError("generated episode failed validation: " + "; ".join(errors))
    return episode
