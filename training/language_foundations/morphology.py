"""L3 — morphology: words changing form.

Productive patterns (plurals, past/progressive, third-person agreement,
comparatives, derivation) plus common irregulars.  Regular-rule families hold
out unseen stems so the gate measures rule application; irregulars are
memorization targets with disjoint splits.
"""

from __future__ import annotations

from .contracts import (
    L3_FAMILY,
    VALIDATOR_EXACT_TRANSPORT,
    LanguageLessonSpec,
    TargetVisibility,
)

_BASIS_RULE = (
    "authored fixture; objectively checkable rule transformation of a visible "
    "stem; heldout and regression stems are unseen in training"
)
_BASIS_IRREGULAR = (
    "authored fixture; objectively checkable memorized irregular form hidden "
    "from all visible surfaces; split pools are disjoint"
)

# (stem, plural)
_PLURAL_REGULAR = {
    "train": (
        ("dog", "dogs"),
        ("cat", "cats"),
        ("book", "books"),
        ("tree", "trees"),
        ("car", "cars"),
        ("hand", "hands"),
        ("song", "songs"),
        ("lamp", "lamps"),
    ),
    "heldout": (
        ("rock", "rocks"),
        ("bird", "birds"),
    ),
    "regression": (
        ("cup", "cups"),
        ("road", "roads"),
    ),
}

# (stem, past)
_PAST_TENSE_REGULAR = {
    "train": (
        ("walk", "walked"),
        ("jump", "jumped"),
        ("play", "played"),
        ("call", "called"),
        ("look", "looked"),
        ("clean", "cleaned"),
        ("start", "started"),
        ("plant", "planted"),
        ("watch", "watched"),
        ("smile", "smiled"),
    ),
    "heldout": (
        ("climb", "climbed"),
        ("paint", "painted"),
        ("knock", "knocked"),
    ),
    "regression": (
        ("listen", "listened"),
        ("open", "opened"),
        ("point", "pointed"),
    ),
}

# (stem, progressive)
_PROGRESSIVE = {
    "train": (
        ("play", "playing"),
        ("walk", "walking"),
        ("read", "reading"),
        ("make", "making"),
        ("run", "running"),
        ("sit", "sitting"),
    ),
    "heldout": (
        ("write", "writing"),
        ("swim", "swimming"),
    ),
    "regression": (("sing", "singing"),),
}

# (stem, third-person singular)
_THIRD_PERSON = {
    "train": (
        ("walk", "walks"),
        ("watch", "watches"),
        ("play", "plays"),
        ("carry", "carries"),
        ("fix", "fixes"),
        ("go", "goes"),
    ),
    "heldout": (("push", "pushes"),),
    "regression": (("try", "tries"),),
}

# (word, comparative)
_COMPARATIVE = {
    "train": (
        ("tall", "taller"),
        ("fast", "faster"),
        ("happy", "happier"),
        ("big", "bigger"),
        ("small", "smaller"),
        ("cold", "colder"),
    ),
    "heldout": (("bright", "brighter"),),
    "regression": (("warm", "warmer"),),
}

# (word, derived, instruction)
_DERIVATION = {
    "train": (
        ("teach", "teacher", "add -er to"),
        ("play", "player", "add -er to"),
        ("happy", "unhappy", "add un- to"),
        ("kind", "kindness", "add -ness to"),
        ("dark", "darkness", "add -ness to"),
        ("help", "helpful", "add -ful to"),
    ),
    "heldout": (("sing", "singer", "add -er to"),),
    "regression": (("sad", "sadness", "add -ness to"),),
}

# (word, form_name, irregular form)
_IRREGULAR = {
    "train": (
        ("go", "past tense", "went"),
        ("child", "plural", "children"),
        ("mouse", "plural", "mice"),
        ("is", "past tense", "was"),
        ("eat", "past tense", "ate"),
        ("run", "past tense", "ran"),
        ("man", "plural", "men"),
        ("tooth", "plural", "teeth"),
    ),
    "heldout": (
        ("swim", "past tense", "swam"),
        ("foot", "plural", "feet"),
    ),
    "regression": (
        ("buy", "past tense", "bought"),
        ("catch", "past tense", "caught"),
    ),
}


def _rule_spec(
    *,
    competency: str,
    split: str,
    prompt: str,
    answer: str,
    stem: str,
) -> LanguageLessonSpec:
    return LanguageLessonSpec(
        family=L3_FAMILY,
        competency=competency,
        split=split,
        prompt=prompt,
        answer=answer,
        visibility=TargetVisibility.HIDDEN,
        validator=VALIDATOR_EXACT_TRANSPORT,
        target_basis=_BASIS_RULE,
        transfer_item=stem,
    )


def l3_lesson_specs() -> tuple[LanguageLessonSpec, ...]:
    specs: list[LanguageLessonSpec] = []
    for split, rows in _PLURAL_REGULAR.items():
        for stem, plural in rows:
            specs.append(
                _rule_spec(
                    competency="plural_regular",
                    split=split,
                    prompt=f"Write the plural of: {stem}",
                    answer=plural,
                    stem=stem,
                )
            )
    for split, rows in _PAST_TENSE_REGULAR.items():
        for stem, past in rows:
            specs.append(
                _rule_spec(
                    competency="past_tense_regular",
                    split=split,
                    prompt=f"Write the past tense of: {stem}",
                    answer=past,
                    stem=stem,
                )
            )
    for split, rows in _PROGRESSIVE.items():
        for stem, progressive in rows:
            specs.append(
                _rule_spec(
                    competency="progressive",
                    split=split,
                    prompt=f"Write the -ing form of: {stem}",
                    answer=progressive,
                    stem=stem,
                )
            )
    for split, rows in _THIRD_PERSON.items():
        for stem, form in rows:
            specs.append(
                _rule_spec(
                    competency="third_person",
                    split=split,
                    prompt=f"Write the third-person singular form of: {stem}",
                    answer=form,
                    stem=stem,
                )
            )
    for split, rows in _COMPARATIVE.items():
        for word, comparative in rows:
            specs.append(
                _rule_spec(
                    competency="comparative_superlative",
                    split=split,
                    prompt=f"Write the comparative form of: {word}",
                    answer=comparative,
                    stem=word,
                )
            )
    for split, rows in _DERIVATION.items():
        for word, derived, instruction in rows:
            specs.append(
                _rule_spec(
                    competency="derivation",
                    split=split,
                    prompt=f"Write the word you get when you {instruction}: {word}",
                    answer=derived,
                    stem=word,
                )
            )
    for split, rows in _IRREGULAR.items():
        for word, form_name, irregular in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L3_FAMILY,
                    competency="irregular",
                    split=split,
                    prompt=f"Write the irregular {form_name} of: {word}",
                    answer=irregular,
                    visibility=TargetVisibility.HIDDEN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_IRREGULAR,
                    transfer_item=word,
                )
            )
    return tuple(specs)


__all__ = ["l3_lesson_specs"]
