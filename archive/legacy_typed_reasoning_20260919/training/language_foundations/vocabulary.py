"""L2 — vocabulary and lexical relations.

Definition<->word in both directions, synonyms, antonyms, category membership,
selection from context, and Axon-domain terminology.  Free-text definition
answers are exact grounded extractions from the visible Cortex definition
(VISIBLE_BY_DESIGN), never semantic guesses; selection lessons are
categorical choices among visible options.
"""

from __future__ import annotations

from .contracts import (
    L2_FAMILY,
    VALIDATOR_CATEGORICAL_CHOICE,
    VALIDATOR_EXACT_TRANSPORT,
    LanguageLessonSpec,
    TargetVisibility,
)

_BASIS_HIDDEN = (
    "authored fixture; objectively checkable lexical answer hidden from all "
    "visible surfaces; the defining evidence does not contain the target word"
)
_BASIS_GROUNDED = (
    "authored fixture; the answer is an exact extraction from the visible "
    "Cortex definition"
)
_BASIS_CHOICE = "authored fixture; the answer is one of the visible options"

# (definition, word); the definition text never contains the target word, and
# the word pools are globally disjoint from _WORD_TO_DEFINITION so the hidden
# direction is never pre-taught by its inverse.
_DEFINITION_TO_WORD = {
    "train": (
        ("a young dog", "puppy"),
        ("a place where books are kept", "library"),
        ("a tool for hitting nails", "hammer"),
        ("a person who teaches", "teacher"),
        ("frozen water", "ice"),
        ("a vehicle with two wheels", "bicycle"),
        ("the star that lights our day", "sun"),
        ("a doctor for animals", "veterinarian"),
        ("a house made of ice blocks", "igloo"),
        ("a flying insect that makes honey", "bee"),
    ),
    "heldout": (
        ("a tool for digging", "shovel"),
        ("a place where planes land", "airport"),
    ),
    "regression": (
        ("a person who writes books", "author"),
        ("a cover for the head", "hat"),
    ),
}

# (word, definition); the definition stays visible in Cortex.
_WORD_TO_DEFINITION = {
    "train": (
        ("cup", "a container for drinking"),
        ("ship", "a large boat"),
        ("baker", "a person who bakes bread"),
        ("kitten", "a baby cat"),
        ("garden", "a place where plants grow"),
        ("bridge", "a structure that crosses water"),
        ("anchor", "a heavy weight that holds a boat"),
        ("candle", "a wax stick that burns for light"),
        ("mirror", "a surface that shows reflections"),
        ("ladder", "a tool for climbing"),
    ),
    "heldout": (
        ("engine", "a machine that makes power"),
        ("pillow", "a soft cushion for the head"),
    ),
    "regression": (
        ("saddle", "a seat for riding a horse"),
        ("lantern", "a portable light"),
    ),
}

# (word, option_a, option_b, option_c, answer)
_SYNONYMS = {
    "train": (
        ("happy", "glad", "sad", "angry", "glad"),
        ("big", "tiny", "large", "loud", "large"),
        ("fast", "slow", "bright", "quick", "quick"),
        ("begin", "start", "finish", "close", "start"),
        ("hard", "easy", "soft", "difficult", "difficult"),
        ("smart", "dull", "clever", "clumsy", "clever"),
    ),
    "heldout": (
        ("cold", "warm", "chilly", "dry", "chilly"),
        ("end", "begin", "open", "finish", "finish"),
    ),
    "regression": (("small", "huge", "tiny", "wide", "tiny"),),
}

# (word, option_a, option_b, option_c, answer)
_ANTONYMS = {
    "train": (
        ("hot", "cold", "warm", "soft", "cold"),
        ("up", "left", "down", "over", "down"),
        ("light", "bright", "heavy", "dark", "dark"),
        ("rich", "poor", "wealthy", "tall", "poor"),
        ("open", "ajar", "wide", "closed", "closed"),
        ("loud", "noisy", "quiet", "deafening", "quiet"),
    ),
    "heldout": (("fast", "quick", "slow", "rapid", "slow"),),
    "regression": (("strong", "mighty", "weak", "tough", "weak"),),
}

# (word, option_a, option_b, option_c, answer)
_CATEGORY_MEMBERSHIP = {
    "train": (
        ("hammer", "tool", "fruit", "animal", "tool"),
        ("apple", "tool", "fruit", "vehicle", "fruit"),
        ("sparrow", "fish", "reptile", "bird", "bird"),
        ("rose", "flower", "mineral", "insect", "flower"),
        ("car", "building", "vehicle", "garment", "vehicle"),
        ("oak", "cloud", "river", "tree", "tree"),
    ),
    "heldout": (("salmon", "bird", "fish", "stone", "fish"),),
    "regression": (("shirt", "vehicle", "plant", "garment", "garment"),),
}

# (context sentence, question, answer); the answer is one visible option.
_CONTEXT_SELECTION = {
    "train": (
        (
            "The box is enormous.",
            "Does enormous mean very small, very large, or very quiet?",
            "very large",
        ),
        (
            "The water was frigid.",
            "Does frigid mean very hot, very cold, or very sweet?",
            "very cold",
        ),
        (
            "The room was spotless.",
            "Does spotless mean very dirty, very dark, or very clean?",
            "very clean",
        ),
        (
            "The path was narrow.",
            "Does narrow mean very wide, very thin, or very long?",
            "very thin",
        ),
        (
            "The answer was obvious.",
            "Does obvious mean easy to see, hard to see, or impossible?",
            "easy to see",
        ),
        (
            "The task was simple.",
            "Does simple mean difficult, easy, or impossible?",
            "easy",
        ),
    ),
    "heldout": (
        (
            "The soup was bland.",
            "Does bland mean very spicy, very hot, or lacking flavor?",
            "lacking flavor",
        ),
    ),
    "regression": (
        (
            "The climb was steep.",
            "Does steep mean flat, underground, or sharply rising?",
            "sharply rising",
        ),
    ),
}

# Axon-domain vocabulary.  d2w = definition to word (word hidden);
# w2d = word to definition (definition visible in Cortex).
_AXON_DOMAIN = {
    "train": (
        ("d2w", "the single exact shared body of Axon state", "canonical"),
        ("d2w", "the record of where evidence came from", "provenance"),
        ("d2w", "the exact chain of accepted ancestors", "lineage"),
        ("d2w", "a saved restorable training state", "checkpoint"),
        ("w2d", "rollback", "returns training to an earlier accepted state"),
        ("w2d", "consolidator", "the rotating core that emits the final typed delta"),
        ("w2d", "substrate", "the frozen 16D transport cells"),
        ("w2d", "mask", "the boundary between attended and dormant cells"),
    ),
    "heldout": (
        ("d2w", "hidden canonical cells outside the attended field", "dormant"),
        ("w2d", "proposal", "a typed suggested edit from a core"),
    ),
    "regression": (
        ("d2w", "a checked failure kept from silently repeating", "regression"),
        ("w2d", "commit", "the atomic canonical write by the Heart"),
    ),
}


def _choice_spec(
    *,
    competency: str,
    split: str,
    prompt: str,
    answer: str,
    cortex_text: str = "",
    transfer_item: str | None = None,
    procedural_depth: int = 1,
) -> LanguageLessonSpec:
    return LanguageLessonSpec(
        family=L2_FAMILY,
        competency=competency,
        split=split,
        prompt=prompt,
        answer=answer,
        visibility=TargetVisibility.VISIBLE_BY_DESIGN,
        validator=VALIDATOR_CATEGORICAL_CHOICE,
        target_basis=_BASIS_CHOICE,
        cortex_text=cortex_text,
        transfer_item=transfer_item,
        procedural_depth=procedural_depth,
    )


def l2_lesson_specs() -> tuple[LanguageLessonSpec, ...]:
    specs: list[LanguageLessonSpec] = []
    for split, rows in _DEFINITION_TO_WORD.items():
        for definition, word in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L2_FAMILY,
                    competency="definition_to_word",
                    split=split,
                    prompt="What do we call this?",
                    answer=word,
                    visibility=TargetVisibility.HIDDEN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_HIDDEN,
                    cortex_text=f"[DEFINITION] {definition} [/DEFINITION]",
                    transfer_item=word,
                    procedural_depth=2,
                )
            )
    for split, rows in _WORD_TO_DEFINITION.items():
        for word, definition in rows:
            specs.append(
                LanguageLessonSpec(
                    family=L2_FAMILY,
                    competency="word_to_definition",
                    split=split,
                    prompt=f"What does \"{word}\" mean?",
                    answer=definition,
                    visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                    validator=VALIDATOR_EXACT_TRANSPORT,
                    target_basis=_BASIS_GROUNDED,
                    cortex_text=f"[DEFINITION] {word}: {definition} [/DEFINITION]",
                    transfer_item=word,
                )
            )
    for split, rows in _SYNONYMS.items():
        for word, option_a, option_b, option_c, answer in rows:
            specs.append(
                _choice_spec(
                    competency="synonym",
                    split=split,
                    prompt=(
                        f"Which word means nearly the same as {word}: "
                        f"{option_a}, {option_b}, or {option_c}?"
                    ),
                    answer=answer,
                    transfer_item=word,
                )
            )
    for split, rows in _ANTONYMS.items():
        for word, option_a, option_b, option_c, answer in rows:
            specs.append(
                _choice_spec(
                    competency="antonym",
                    split=split,
                    prompt=(
                        f"Which word means the opposite of {word}: "
                        f"{option_a}, {option_b}, or {option_c}?"
                    ),
                    answer=answer,
                    transfer_item=word,
                )
            )
    for split, rows in _CATEGORY_MEMBERSHIP.items():
        for word, option_a, option_b, option_c, answer in rows:
            specs.append(
                _choice_spec(
                    competency="category_membership",
                    split=split,
                    prompt=f"A {word} is a kind of: {option_a}, {option_b}, or {option_c}?",
                    answer=answer,
                    transfer_item=word,
                )
            )
    for split, rows in _CONTEXT_SELECTION.items():
        for context, question, answer in rows:
            specs.append(
                _choice_spec(
                    competency="context_selection",
                    split=split,
                    prompt=question,
                    answer=answer,
                    cortex_text=f"[CONTEXT] {context} [/CONTEXT]",
                    transfer_item=context,
                    procedural_depth=2,
                )
            )
    for split, rows in _AXON_DOMAIN.items():
        for kind, left, right in rows:
            if kind == "d2w":
                definition, word = left, right
                specs.append(
                    LanguageLessonSpec(
                        family=L2_FAMILY,
                        competency="axon_domain_vocabulary",
                        split=split,
                        prompt="Which Axon term names this?",
                        answer=word,
                        visibility=TargetVisibility.HIDDEN,
                        validator=VALIDATOR_EXACT_TRANSPORT,
                        target_basis=_BASIS_HIDDEN,
                        cortex_text=f"[DEFINITION] {definition} [/DEFINITION]",
                        transfer_item=word,
                        procedural_depth=2,
                    )
                )
            else:
                word, definition = left, right
                specs.append(
                    LanguageLessonSpec(
                        family=L2_FAMILY,
                        competency="axon_domain_vocabulary",
                        split=split,
                        prompt=f"What is a {word} in Axon?",
                        answer=definition,
                        visibility=TargetVisibility.VISIBLE_BY_DESIGN,
                        validator=VALIDATOR_EXACT_TRANSPORT,
                        target_basis=_BASIS_GROUNDED,
                        cortex_text=f"[DEFINITION] {definition.capitalize()}. [/DEFINITION]",
                        transfer_item=word,
                    )
                )
    return tuple(specs)


__all__ = ["l2_lesson_specs"]
