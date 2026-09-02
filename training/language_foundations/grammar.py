"""L4 — grammar mechanics.

Local sentence structure with objectively checkable tasks: agreement and
tense choices among visible options, pronoun reference against visible
evidence, articles, and sentence transformations (negation, question
formation, conjunction joins, ordering, grammatical choice, repair).
Choice lessons are categorical; transformations keep the exact answer absent
(HIDDEN) unless it is a case-level copy of visible text.
"""

from __future__ import annotations

from .contracts import (
    L4_FAMILY,
    VALIDATOR_CATEGORICAL_CHOICE,
    VALIDATOR_EXACT_TRANSPORT,
    LanguageLessonSpec,
    TargetVisibility,
)

_BASIS_CHOICE = "authored fixture; the answer is one of the visible options"
_BASIS_TRANSFORM = (
    "authored fixture; objectively checkable sentence transformation whose "
    "exact result is hidden from all visible surfaces"
)
_BASIS_REFERENCE = (
    "authored fixture; the referent is objectively determined by the visible "
    "Cortex evidence and named among the visible options"
)

# (sentence with bracketed options, answer, subject phrase for transfer)
_AGREEMENT = {
    "train": (
        ("The dogs [runs / run] outside.", "run", "the dogs"),
        ("The cat [sit / sits] on the mat.", "sits", "the cat"),
        ("She [walk / walks] to school.", "walks", "she"),
        ("They [is / are] ready.", "are", "they"),
        ("The birds [sing / sings] at dawn.", "sing", "the birds"),
        ("He [play / plays] chess.", "plays", "he"),
        ("The windows [is / are] open.", "are", "the windows"),
        ("My friend [live / lives] nearby.", "lives", "my friend"),
        ("The children [was / were] asleep.", "were", "the children"),
        ("The book [belong / belongs] to Ana.", "belongs", "the book"),
    ),
    "heldout": (
        ("The students [was / were] waiting.", "were", "the students"),
        ("The lamp [shine / shines] brightly.", "shines", "the lamp"),
        ("The teachers [meet / meets] at noon.", "meet", "the teachers"),
    ),
    "regression": (
        ("The clock [tick / ticks] softly.", "ticks", "the clock"),
        ("My shoes [is / are] muddy.", "are", "my shoes"),
    ),
}

# (sentence with bracketed options, answer, template key)
_TENSE_CHOICE = {
    "train": (
        ("Yesterday I [go / went / going] to the store.", "went", "past:i-store"),
        ("Tomorrow we [travel / traveled / will travel] to Berlin.", "will travel", "future:we-berlin"),
        ("Every morning he [runs / ran / running] five kilometers.", "runs", "present:he-runs"),
        ("Last night they [watch / watched / watching] a film.", "watched", "past:they-film"),
        ("Next week she [visits / visited / will visit] her aunt.", "will visit", "future:she-aunt"),
        ("Right now the baby [sleeps / slept / will sleep].", "sleeps", "present:baby-sleeps"),
        ("Last year we [move / moved / will move] to Oslo.", "moved", "past:we-oslo"),
        ("Every winter it [snows / snowed / will snow] here.", "snows", "present:it-snows"),
    ),
    "heldout": (
        ("Yesterday she [calls / called / will call] her brother.", "called", "past:she-brother"),
        ("Tomorrow they [arrive / arrived / will arrive] early.", "will arrive", "future:they-arrive"),
    ),
    "regression": (
        ("Last week I [visit / visited / will visit] the museum.", "visited", "past:i-museum"),
        ("Every day the sun [rises / rose / will rise] in the east.", "rises", "present:sun-rises"),
    ),
}

# (cortex evidence, question, answer)
_PRONOUN_REFERENCE = {
    "train": (
        (
            "Ben has a red ball. Ben gives the ball to Mia.",
            "Who has the ball now: Ben or Mia?",
            "Mia",
        ),
        (
            "Sara lent her pencil to Tom. Tom still has it.",
            "Who has the pencil: Sara or Tom?",
            "Tom",
        ),
        (
            "The dog chased its tail. The cat watched the dog.",
            "Which animal chased its tail: the dog or the cat?",
            "the dog",
        ),
        (
            "Anna called Luis. Luis answered the phone.",
            "Who answered the phone: Anna or Luis?",
            "Luis",
        ),
        (
            "The teacher gave the students their books.",
            "Who received the books: the teacher or the students?",
            "the students",
        ),
        (
            "Mia lost her keys, but Ben found them.",
            "Who found the keys: Mia or Ben?",
            "Ben",
        ),
    ),
    "heldout": (
        (
            "Lena pushed the cart. Omar pulled it back.",
            "Who pulled the cart: Lena or Omar?",
            "Omar",
        ),
    ),
    "regression": (
        (
            "The fox followed the hen into the yard.",
            "Which animal was followed: the fox or the hen?",
            "the hen",
        ),
    ),
}

# (noun, answer)
_ARTICLES = {
    "train": (
        ("apple", "an"),
        ("dog", "a"),
        ("umbrella", "an"),
        ("house", "a"),
        ("hour", "an"),
        ("cat", "a"),
    ),
    "heldout": (("orange", "an"),),
    "regression": (("egg", "an"),),
}

# (statement, negation)
_NEGATION = {
    "train": (
        ("She is ready.", "She is not ready."),
        ("They are home.", "They are not home."),
        ("He can swim.", "He cannot swim."),
        ("The shop is open.", "The shop is not open."),
        ("We are late.", "We are not late."),
    ),
    "heldout": (("The door is locked.", "The door is not locked."),),
    "regression": (("It is raining.", "It is not raining."),),
}

# (statement, question)
_QUESTION_FORMATION = {
    "train": (
        ("The machine is running.", "Is the machine running?"),
        ("She is ready.", "Is she ready?"),
        ("They are home.", "Are they home?"),
        ("The door is open.", "Is the door open?"),
        ("He can swim.", "Can he swim?"),
    ),
    "heldout": (("The train is late.", "Is the train late?"),),
    "regression": (("We are early.", "Are we early?"),),
}

# (conjunction, sentence_a, sentence_b, joined)
_CONJUNCTION_JOIN = {
    "train": (
        (
            "because",
            "The server stopped.",
            "The disk was full.",
            "The server stopped because the disk was full.",
        ),
        ("and", "Ana cooked.", "Luis cleaned.", "Ana cooked and Luis cleaned."),
        ("but", "It rained.", "We stayed dry.", "It rained but we stayed dry."),
        (
            "because",
            "She left early.",
            "She was tired.",
            "She left early because she was tired.",
        ),
        (
            "and",
            "The sun set.",
            "The stars appeared.",
            "The sun set and the stars appeared.",
        ),
    ),
    "heldout": (
        (
            "but",
            "He ran fast.",
            "He missed the bus.",
            "He ran fast but he missed the bus.",
        ),
    ),
    "regression": (
        (
            "because",
            "We stayed home.",
            "The storm came.",
            "We stayed home because the storm came.",
        ),
    ),
}

# (scrambled words, ordered sentence); scrambled order must differ from the
# answer order so the exact target stays hidden under token-level matching.
_SENTENCE_ORDERING = {
    "train": (
        ("dog / the / quickly / ran / home", "The dog quickly ran home."),
        ("is / the / blue / sky", "The sky is blue."),
        ("reads / ana / books / every / night", "Ana reads books every night."),
        ("the / cake / baked / maria / slowly", "Maria baked the cake slowly."),
        ("loudly / the / bell / rang", "The bell rang loudly."),
    ),
    "heldout": (("the / birds / south / flew", "The birds flew south."),),
    "regression": (
        ("children / the / played / outside / happily", "The children played outside happily."),
    ),
}

# (option_a, option_b, answer)
_GRAMMATICAL_CHOICE = {
    "train": (
        ("The dogs run outside.", "The dogs runs outside.", "The dogs run outside."),
        ("She walks to school.", "She walk to school.", "She walks to school."),
        ("They is ready.", "They are ready.", "They are ready."),
        ("He plays chess.", "He play chess.", "He plays chess."),
        ("The cat sit on the mat.", "The cat sits on the mat.", "The cat sits on the mat."),
    ),
    "heldout": (
        ("The birds sings at dawn.", "The birds sing at dawn.", "The birds sing at dawn."),
    ),
    "regression": (
        ("My friend lives nearby.", "My friend live nearby.", "My friend lives nearby."),
    ),
}

# (broken, fixed)
_SENTENCE_REPAIR = {
    "train": (
        ("The dogs runs outside.", "The dogs run outside."),
        ("She walk to school.", "She walks to school."),
        ("They is ready.", "They are ready."),
        ("He play chess.", "He plays chess."),
        ("The cat sit on the mat.", "The cat sits on the mat."),
    ),
    "heldout": (("The birds sings at dawn.", "The birds sing at dawn."),),
    "regression": (("My friend live nearby.", "My friend lives nearby."),),
}


def _choice_spec(
    *,
    competency: str,
    split: str,
    prompt: str,
    answer: str,
    cortex_text: str = "",
    transfer_item: str | None = None,
    target_basis: str = _BASIS_CHOICE,
    procedural_depth: int = 1,
) -> LanguageLessonSpec:
    return LanguageLessonSpec(
        family=L4_FAMILY,
        competency=competency,
        split=split,
        prompt=prompt,
        answer=answer,
        visibility=TargetVisibility.VISIBLE_BY_DESIGN,
        validator=VALIDATOR_CATEGORICAL_CHOICE,
        target_basis=target_basis,
        cortex_text=cortex_text,
        transfer_item=transfer_item,
        procedural_depth=procedural_depth,
    )


def _transform_spec(
    *,
    competency: str,
    split: str,
    prompt: str,
    answer: str,
    transfer_item: str,
    procedural_depth: int = 1,
) -> LanguageLessonSpec:
    return LanguageLessonSpec(
        family=L4_FAMILY,
        competency=competency,
        split=split,
        prompt=prompt,
        answer=answer,
        visibility=TargetVisibility.HIDDEN,
        validator=VALIDATOR_EXACT_TRANSPORT,
        target_basis=_BASIS_TRANSFORM,
        transfer_item=transfer_item,
        procedural_depth=procedural_depth,
    )


def l4_lesson_specs() -> tuple[LanguageLessonSpec, ...]:
    specs: list[LanguageLessonSpec] = []
    for split, rows in _AGREEMENT.items():
        for sentence, answer, subject in rows:
            specs.append(
                _choice_spec(
                    competency="subject_verb_agreement",
                    split=split,
                    prompt=f"Choose the correct form: {sentence}",
                    answer=answer,
                    transfer_item=subject,
                )
            )
    for split, rows in _TENSE_CHOICE.items():
        for sentence, answer, template in rows:
            specs.append(
                _choice_spec(
                    competency="tense_choice",
                    split=split,
                    prompt=f"Choose the correct form: {sentence}",
                    answer=answer,
                    transfer_item=template,
                )
            )
    for split, rows in _PRONOUN_REFERENCE.items():
        for evidence, question, answer in rows:
            specs.append(
                _choice_spec(
                    competency="pronoun_reference",
                    split=split,
                    prompt=question,
                    answer=answer,
                    cortex_text=f"[EVIDENCE] {evidence} [/EVIDENCE]",
                    transfer_item=evidence,
                    target_basis=_BASIS_REFERENCE,
                    procedural_depth=2,
                )
            )
    for split, rows in _ARTICLES.items():
        for noun, answer in rows:
            specs.append(
                _choice_spec(
                    competency="articles",
                    split=split,
                    prompt=f"Choose the correct article: ___ {noun} [a / an]",
                    answer=answer,
                    transfer_item=noun,
                )
            )
    for split, rows in _NEGATION.items():
        for statement, negated in rows:
            specs.append(
                _transform_spec(
                    competency="negation",
                    split=split,
                    prompt=f"Make negative: {statement}",
                    answer=negated,
                    transfer_item=statement,
                )
            )
    for split, rows in _QUESTION_FORMATION.items():
        for statement, question in rows:
            specs.append(
                _transform_spec(
                    competency="question_formation",
                    split=split,
                    prompt=f"Turn into a question: {statement}",
                    answer=question,
                    transfer_item=statement,
                )
            )
    for split, rows in _CONJUNCTION_JOIN.items():
        for conjunction, sentence_a, sentence_b, joined in rows:
            specs.append(
                _transform_spec(
                    competency="conjunction_join",
                    split=split,
                    prompt=f'Join using "{conjunction}": {sentence_a} {sentence_b}',
                    answer=joined,
                    transfer_item=f"{sentence_a}|{sentence_b}",
                    procedural_depth=2,
                )
            )
    for split, rows in _SENTENCE_ORDERING.items():
        for scrambled, ordered in rows:
            specs.append(
                _transform_spec(
                    competency="sentence_ordering",
                    split=split,
                    prompt=f"Arrange into a sentence: {scrambled}",
                    answer=ordered,
                    transfer_item=scrambled,
                    procedural_depth=2,
                )
            )
    for split, rows in _GRAMMATICAL_CHOICE.items():
        for option_a, option_b, answer in rows:
            specs.append(
                _choice_spec(
                    competency="grammatical_choice",
                    split=split,
                    prompt=f'Which is grammatical: "{option_a}" or "{option_b}"?',
                    answer=answer,
                    transfer_item=f"{option_a}|{option_b}",
                )
            )
    for split, rows in _SENTENCE_REPAIR.items():
        for broken, fixed in rows:
            specs.append(
                _transform_spec(
                    competency="sentence_repair",
                    split=split,
                    prompt=f"Fix the sentence: {broken}",
                    answer=fixed,
                    transfer_item=broken,
                )
            )
    return tuple(specs)


__all__ = ["l4_lesson_specs"]
