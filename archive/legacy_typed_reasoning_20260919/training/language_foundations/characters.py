"""L0 — character and transport fluency lessons.

Exact copy, minimal-pair distinction, capitalization pairs, punctuation and
whitespace handling, exact Unicode transport (including multi-unit UTF-8
expansion), EOS/no-op distinction, and insert/replace/delete at exact
addresses.  Copy and selection lessons are VISIBLE_BY_DESIGN; edit payloads
are HIDDEN.
"""

from __future__ import annotations

from runtime.heart import ReasoningOperationKind

from .contracts import (
    L0_FAMILY,
    VALIDATOR_CATEGORICAL_CHOICE,
    VALIDATOR_EXACT_ADDRESSED_EDIT,
    VALIDATOR_EXACT_TRANSPORT,
    VALIDATOR_NO_OP_DECISION,
    LanguageLessonSpec,
    TargetVisibility,
)

_BASIS_COPY = "authored fixture; exact copy of visible text; objectively checkable"
_BASIS_CHOICE = "authored fixture; the answer is one of the visible options"
_BASIS_TRANSFORM = (
    "authored fixture; the target is a case transformation of a visible token"
)
_BASIS_NO_OP = (
    "authored fixture; the response draft is already complete, so the correct "
    "consolidated decision is no_op"
)
_BASIS_EDIT = (
    "authored fixture; exact addressed edit hidden from all visible surfaces; "
    "objectively checkable by applying the operation"
)

_COPY_TRAIN = (
    "apple",
    "hello",
    "cat",
    "axon",
    "field",
    "soul",
    "heart",
    "brother",
    "λ",
    "🧠",
    "naïve",
    "東京",
    "Zürich",
    "café",
)
_COPY_HELDOUT = ("planet", "window", "φίλος", "Москва")
_COPY_REGRESSION = ("table", "green", "∑", "a")

# (word_a, word_b, answer); option order is fixed as authored.
_MINIMAL_PAIRS = {
    "train": (
        ("form", "from", "from"),
        ("cat", "cut", "cat"),
        ("quiet", "quite", "quiet"),
        ("there", "their", "their"),
        ("through", "threw", "through"),
        ("angel", "angle", "angle"),
        ("desert", "dessert", "dessert"),
        ("later", "latter", "later"),
    ),
    "heldout": (
        ("breath", "breathe", "breath"),
        ("loose", "lose", "lose"),
    ),
    "regression": (
        ("plain", "plane", "plane"),
        ("week", "weak", "week"),
    ),
}

# (shown, answer, direction)
_CAPITALIZATION_PAIRS = {
    "train": (
        ("b", "B", "capital"),
        ("d", "D", "capital"),
        ("t", "T", "capital"),
        ("z", "Z", "capital"),
    ),
    "heldout": (("Q", "q", "lowercase"),),
    "regression": (("m", "M", "capital"),),
}

_PUNCTUATION_COPY = {
    "train": (
        "Hello, world.",
        "It's time.",
        "Wait — really?",
        "Yes, exactly.",
    ),
    "heldout": ("Stop. Look. Listen.",),
    "regression": ('She said, "Go home."',),
}

_NO_OP_LESSONS = {
    "train": (
        (
            "The response draft is already complete and correct. Take no action.",
            "All set.",
        ),
        (
            "Nothing needs to change in the response draft. Do not edit anything.",
            "Done.",
        ),
        (
            "The draft already answers the request exactly. Leave it untouched.",
            "Ready.",
        ),
        (
            "No further edit is required; the response draft stands as written.",
            "Complete.",
        ),
    ),
    "heldout": (
        (
            "The response draft is finished. Make no change.",
            "Confirmed.",
        ),
    ),
    "regression": (
        (
            "Everything required is already in the response draft. Take no action.",
            "Noted.",
        ),
    ),
}

# (starter, prompt, edit kind, start, end, payload)
_ADDRESSED_EDITS = {
    "train": (
        (
            "helo",
            "Insert the missing letter at exact position 3.",
            ReasoningOperationKind.INSERT,
            3,
            3,
            "l",
        ),
        (
            "ct",
            "Insert the missing letter at exact position 1.",
            ReasoningOperationKind.INSERT,
            1,
            1,
            "a",
        ),
        (
            "hexo",
            "Replace the letters from exact position 2 up to 4 with the correct letters.",
            ReasoningOperationKind.REPLACE,
            2,
            4,
            "ll",
        ),
        (
            "helllo",
            "Delete the extra letter from exact position 3 up to 4.",
            ReasoningOperationKind.DELETE,
            3,
            4,
            "",
        ),
    ),
    "heldout": (
        (
            "spel",
            "Insert the missing letter at exact position 3.",
            ReasoningOperationKind.INSERT,
            3,
            3,
            "l",
        ),
    ),
    "regression": (
        (
            "catt",
            "Delete the extra letter from exact position 2 up to 3.",
            ReasoningOperationKind.DELETE,
            2,
            3,
            "",
        ),
    ),
}


def l0_lesson_specs() -> tuple[LanguageLessonSpec, ...]:
    specs: list[LanguageLessonSpec] = []
    for split, pool in (
        ("train", _COPY_TRAIN),
        ("heldout", _COPY_HELDOUT),
        ("regression", _COPY_REGRESSION),
    ):
        for text in pool:
            specs.append(
                LanguageLessonSpec(
                    family=L0_FAMILY,
                    competency="exact_copy",
                    split=split,
                    prompt=f"Copy exactly: {text}",
                    answer=text,
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_COPY,
                    transfer_item=text,
                )
            )
    for split, rows in _MINIMAL_PAIRS.items():
        for word_a, word_b, answer in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L0_FAMILY,
                    competency="minimal_pair_selection",
                    split=split,
                    prompt=f"Which spelling matches the shown word: {word_a} or {word_b}?",
                    answer=answer,
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_CATEGORICAL_CHOICE,
                    target_basis=_BASIS_CHOICE,
                    cortex_text=f"[WORD] {answer} [/WORD]",
                    transfer_item=f"{word_a}/{word_b}",
                )
            )
    for split, rows in _CAPITALIZATION_PAIRS.items():
        for shown, answer, direction in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L0_FAMILY,
                    competency="capitalization_pair",
                    split=split,
                    prompt=f"Write the {direction} form of the letter: {shown}",
                    answer=answer,
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_TRANSFORM,
                    transfer_item=shown,
                )
            )
    for split, pool in _PUNCTUATION_COPY.items():
        for text in pool:
            specs.append(
                LanguageLessonSpec(
                    family=L0_FAMILY,
                    competency="punctuation_copy",
                    split=split,
                    prompt=f"Copy exactly, including punctuation: {text}",
                    answer=text,
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_COPY,
                    transfer_item=text,
                )
            )
    for split, rows in _NO_OP_LESSONS.items():
        for prompt, starter in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L0_FAMILY,
                    competency="no_op_distinction",
                    split=split,
                    prompt=prompt,
                    answer="",
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_NO_OP_DECISION,
                    target_basis=_BASIS_NO_OP,
                    draft_starter=starter,
                )
            )
    for split, rows in _ADDRESSED_EDITS.items():
        for starter, prompt, kind, start, end, payload in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L0_FAMILY,
                    competency="addressed_edit",
                    split=split,
                    prompt=prompt,
                    answer=payload,
                    visibility=TargetVisibility.HIDDEN,
                    validator=VALIDATOR_EXACT_ADDRESSED_EDIT,
                    target_basis=_BASIS_EDIT,
                    draft_starter=starter,
                    edit=(kind, start, end),
                    procedural_depth=2,
                )
            )
    return tuple(specs)


__all__ = ["l0_lesson_specs"]
