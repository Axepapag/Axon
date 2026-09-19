"""L1 — spelling, capitalization and punctuation lessons.

Orthographic regularities taught as transformations: misspelling repair,
plural spelling endings, capitalization, terminal punctuation, apostrophes,
and sentence spacing.  Heldout and regression items use unseen words so the
gate measures rule transfer rather than memorization.
"""

from __future__ import annotations

from .contracts import (
    L1_FAMILY,
    VALIDATOR_EXACT_TRANSPORT,
    LanguageLessonSpec,
    TargetVisibility,
)

_BASIS_HIDDEN = (
    "authored fixture; objectively checkable orthographic answer hidden from "
    "all visible surfaces; heldout and regression items are unseen words"
)
_BASIS_TRANSFORM = (
    "authored fixture; the target is an exact orthographic transformation of "
    "the visible prompt text"
)

# (misspelled, correct); pools are disjoint across splits by construction.
_SPELLING_REPAIR = {
    "train": (
        ("helo", "hello"),
        ("poeple", "people"),
        ("recieve", "receive"),
        ("freind", "friend"),
        ("becuase", "because"),
        ("adress", "address"),
        ("wich", "which"),
        ("seperate", "separate"),
        ("tomorow", "tomorrow"),
        ("beleive", "believe"),
        ("goverment", "government"),
        ("arguement", "argument"),
    ),
    "heldout": (
        ("definately", "definitely"),
        ("neccessary", "necessary"),
        ("occurence", "occurrence"),
        ("enviroment", "environment"),
    ),
    "regression": (
        ("calender", "calendar"),
        ("existance", "existence"),
        ("independant", "independent"),
        ("maintainance", "maintenance"),
    ),
}

# (singular, plural)
_PLURAL_SPELLING = {
    "train": (
        ("box", "boxes"),
        ("brush", "brushes"),
        ("church", "churches"),
        ("baby", "babies"),
        ("city", "cities"),
        ("day", "days"),
        ("key", "keys"),
        ("dish", "dishes"),
    ),
    "heldout": (
        ("lady", "ladies"),
        ("watch", "watches"),
    ),
    "regression": (
        ("toy", "toys"),
        ("class", "classes"),
    ),
}

# (shown, corrected)
_CAPITALIZATION = {
    "train": (
        ("the dog barks.", "The dog barks."),
        ("paris is in france.", "Paris is in France."),
        ("maria plays chess.", "Maria plays chess."),
        ("london is large.", "London is large."),
        ("we left early.", "We left early."),
        ("axel reads books.", "Axel reads books."),
    ),
    "heldout": (("berlin is cold.", "Berlin is cold."),),
    "regression": (("sara sings well.", "Sara sings well."),),
}

# (unterminated, terminated)
_TERMINAL_PUNCTUATION = {
    "train": (
        ("What time is it", "What time is it?"),
        ("The train is late", "The train is late."),
        ("Watch out", "Watch out!"),
        ("Where is the station", "Where is the station?"),
        ("The door is open", "The door is open."),
    ),
    "heldout": (("How did that happen", "How did that happen?"),),
    "regression": (("Close the window", "Close the window."),),
}

# (full form, contraction)
_APOSTROPHE_CONTRACTION = {
    "train": (
        ("do not", "don't"),
        ("cannot", "can't"),
        ("she is", "she's"),
        ("it is", "it's"),
        ("we are", "we're"),
    ),
    "heldout": (("they are", "they're"),),
    "regression": (("will not", "won't"),),
}

# (shown, corrected, visibility): whitespace collapses normalize onto the
# visible prompt and are therefore VISIBLE_BY_DESIGN; punctuation-spacing
# repairs keep the exact answer absent and stay HIDDEN.
_SENTENCE_SPACING = {
    "train": (
        ("The  dog  barks.", "The dog barks.", TargetVisibility.VISIBLE_BY_DESIGN),
        ("One  two   three", "One two three", TargetVisibility.VISIBLE_BY_DESIGN),
        ("Hello.World", "Hello. World", TargetVisibility.HIDDEN),
        ("Yes,I agree", "Yes, I agree", TargetVisibility.HIDDEN),
    ),
    "heldout": (("Wait.Hold on", "Wait. Hold on", TargetVisibility.HIDDEN),),
    "regression": (("No,really", "No, really", TargetVisibility.HIDDEN),),
}


def l1_lesson_specs() -> tuple[LanguageLessonSpec, ...]:
    specs: list[LanguageLessonSpec] = []
    for split, rows in _SPELLING_REPAIR.items():
        for wrong, correct in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L1_FAMILY,
                    competency="spelling_repair",
                    split=split,
                    prompt=f"Fix the spelling: {wrong}",
                    answer=correct,
                    visibility=TargetVisibility.HIDDEN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_HIDDEN,
                    transfer_item=correct,
                )
            )
    for split, rows in _PLURAL_SPELLING.items():
        for singular, plural in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L1_FAMILY,
                    competency="plural_spelling",
                    split=split,
                    prompt=f"Write the plural of: {singular}",
                    answer=plural,
                    visibility=TargetVisibility.HIDDEN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_HIDDEN,
                    transfer_item=singular,
                )
            )
    for split, rows in _CAPITALIZATION.items():
        for shown, corrected in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L1_FAMILY,
                    competency="capitalization",
                    split=split,
                    prompt=f"Capitalize correctly: {shown}",
                    answer=corrected,
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_TRANSFORM,
                    transfer_item=shown,
                )
            )
    for split, rows in _TERMINAL_PUNCTUATION.items():
        for shown, terminated in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L1_FAMILY,
                    competency="terminal_punctuation",
                    split=split,
                    prompt=f"Add the correct final punctuation: {shown}",
                    answer=terminated,
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_TRANSFORM,
                    transfer_item=shown,
                )
            )
    for split, rows in _APOSTROPHE_CONTRACTION.items():
        for full, contraction in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L1_FAMILY,
                    competency="apostrophe_contraction",
                    split=split,
                    prompt=f"Write the contraction: {full}",
                    answer=contraction,
                    visibility=TargetVisibility.HIDDEN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_HIDDEN,
                    transfer_item=full,
                )
            )
    for split, rows in _SENTENCE_SPACING.items():
        for shown, corrected, visibility in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L1_FAMILY,
                    competency="sentence_spacing",
                    split=split,
                    prompt=f"Fix the spacing: {shown}",
                    answer=corrected,
                    visibility=visibility,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=(
                        _BASIS_TRANSFORM
                        if visibility is TargetVisibility.VISIBLE_BY_DESIGN
                        else _BASIS_HIDDEN
                    ),
                    transfer_item=shown,
                )
            )
    return tuple(specs)


__all__ = ["l1_lesson_specs"]
