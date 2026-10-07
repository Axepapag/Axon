"""Tests for the deterministic native-95 curriculum generators.

Covers the required ground: every family validates for several seeds, byte
determinism, outsider-free text, control-family law (no empty input surfaces,
WAIT = control 0), correction returns the latest value, and order binding
returns the bound answer. No huge generation: a few hundred small episodes.
"""
from __future__ import annotations

import copy
import random

import pytest

from curriculum.generators import FAMILY_DEFAULT_DIFFICULTY, generate_episode
from curriculum.schema import (
    CONTROL_COMMIT,
    CONTROL_END,
    CONTROL_WAIT,
    FAMILIES,
    canonical_json,
    validate_episode,
)
from substrate.native import ALPHABET

SEEDS = (0, 1, 2)
_NATIVE = set(ALPHABET)
_HEX_DIGITS = set("0123456789abcdef")


def _walk_strings(value: object, path: str = "episode"):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def _input_texts(episode: dict) -> list[str]:
    return [step["text"] for step in episode["steps"] if step["kind"] == "input"]


def _queried_key(query: str) -> str:
    marker = "what is the value of "
    assert query.startswith(marker) and query.endswith("?")
    return query[len(marker) : -1]


# --- contract shape ---


def test_defaults_cover_every_family():
    assert set(FAMILY_DEFAULT_DIFFICULTY) == set(FAMILIES)
    for family, defaults in FAMILY_DEFAULT_DIFFICULTY.items():
        assert isinstance(defaults, dict), family
        assert defaults, family


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_generates_valid_episodes(family):
    for seed in SEEDS:
        episode = generate_episode(family, seed)
        errors = validate_episode(episode)
        assert errors == [], (family, seed, errors)
        assert episode["family"] == family
        assert episode["seed"] == seed
        assert set(episode) >= {
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
        }


@pytest.mark.parametrize("family", FAMILIES)
def test_episode_id_is_deterministic_and_well_formed(family):
    episode = generate_episode(family, 255)
    prefix = f"{family}-ff-"
    assert episode["episode_id"].startswith(prefix)
    tail = episode["episode_id"][len(prefix) :]
    assert len(tail) == 12
    assert set(tail) <= _HEX_DIGITS
    assert generate_episode(family, 255)["episode_id"] == episode["episode_id"]
    assert generate_episode(family, 256)["episode_id"] != episode["episode_id"]


# --- determinism ---


@pytest.mark.parametrize("family", FAMILIES)
def test_two_calls_are_byte_identical(family):
    for seed in SEEDS:
        first = generate_episode(family, seed)
        second = generate_episode(family, seed)
        assert canonical_json(first) == canonical_json(second)


@pytest.mark.parametrize("family", FAMILIES)
def test_different_seeds_differ(family):
    assert canonical_json(generate_episode(family, 1)) != canonical_json(
        generate_episode(family, 2)
    )


def test_difficulty_dict_order_does_not_change_the_episode():
    first = generate_episode("delayed_recall", 6, {"length": 3, "delay": 2})
    second = generate_episode("delayed_recall", 6, {"delay": 2, "length": 3})
    assert canonical_json(first) == canonical_json(second)


def test_global_randomness_is_never_used():
    random.seed(20261006)
    state_before = random.getstate()
    generate_episode("distracted_recall", 3)
    assert random.getstate() == state_before, "the generator must not touch global randomness"

    random.seed(1)
    first = canonical_json(generate_episode("key_value", 5))
    random.seed(2)
    second = canonical_json(generate_episode("key_value", 5))
    assert first == second, "the global seed must not influence generated episodes"


# --- native-95 law ---


@pytest.mark.parametrize("family", FAMILIES)
def test_every_text_field_is_native95_or_allowed_empty(family):
    for seed in SEEDS:
        episode = generate_episode(family, seed)
        for path, text in _walk_strings(episode):
            outside = sorted({character for character in text if character not in _NATIVE})
            assert not outside, f"{family}/{seed}: {path} has outside characters {outside}"


@pytest.mark.parametrize("family", FAMILIES)
def test_no_empty_content_surfaces_anywhere(family):
    for seed in SEEDS:
        episode = generate_episode(family, seed)
        assert episode["query"] != ""
        assert episode["split_key"] != ""
        assert episode["episode_id"] != ""
        for index, step in enumerate(episode["steps"]):
            assert step["control"] is None, (family, seed, index)
            if step["kind"] == "input":
                assert step["text"] != "", (
                    f"{family}/{seed}: input step {index} is an all-EMPTY surface"
                )
            else:
                assert step["kind"] in ("tick", "reset")
                assert step["text"] == "", (family, seed, index)
        expected = episode["expected"]
        if expected["text"] == "":
            assert expected["control"] in (CONTROL_WAIT, CONTROL_END), (
                f"{family}/{seed}: empty expected text is only allowed for control-only answers"
            )
        if expected["control"] is not None:
            assert type(expected["control"]) is int
        for fact in episode["facts"]:
            assert fact != ""


# --- family semantics ---


def test_char_copy_is_exact_and_bounded():
    for seed in range(20):
        episode = generate_episode("char_copy", seed)
        content = _input_texts(episode)
        assert len(content) == 1
        assert 1 <= len(content[0]) <= 8
        assert episode["expected"] == {"text": content[0], "control": None}
        assert episode["facts"] == [content[0]]
        assert episode["delay_ticks"] == 0
        assert episode["distractor_count"] is None
        assert episode["steps"][0]["kind"] == "input"


def test_delayed_recall_ticks_match_delay_and_keep_content():
    for seed in range(5):
        for delay in (0, 3, 32):
            episode = generate_episode("delayed_recall", seed, {"length": 4, "delay": delay})
            kinds = [step["kind"] for step in episode["steps"]]
            assert kinds == ["input"] + ["tick"] * delay
            content = _input_texts(episode)[0]
            assert len(content) == 4
            assert episode["expected"]["text"] == content
            assert episode["delay_ticks"] == delay


@pytest.mark.parametrize("kind", ["noise", "similar", "repeated"])
def test_distracted_recall_distractors_never_win(kind):
    for seed in range(4):
        episode = generate_episode(
            "distracted_recall",
            seed,
            {"length": 3, "delay": 2, "distractors": 3, "distractor_kind": kind},
        )
        inputs = _input_texts(episode)
        content, distractors = inputs[0], inputs[1:]
        assert episode["expected"]["text"] == content
        assert episode["distractor_count"] == 3
        assert len(distractors) == 3
        for distractor in distractors:
            assert distractor != content, "the answer must not appear among the distractors"
        ticks = [step["kind"] for step in episode["steps"] if step["kind"] == "tick"]
        assert len(ticks) == 2
        if kind == "repeated":
            assert len(set(distractors)) == 1
        if kind == "similar":
            assert all(len(distractor) == len(content) for distractor in distractors)


def test_key_value_query_is_bound_to_one_stored_key():
    for seed in range(6):
        episode = generate_episode("key_value", seed, {"facts": 4, "delay": 2})
        binding = {}
        for fact in episode["facts"]:
            key, value = fact.split(" is ")
            binding[key] = value
        assert len(binding) == 4
        queried = _queried_key(episode["query"])
        assert queried in binding
        assert episode["expected"]["text"] == binding[queried]
        assert episode["delay_ticks"] == 2


def test_key_value_split_key_uses_template_and_keys_not_values():
    episode = generate_episode("key_value", 1, {"facts": 3})
    keys = sorted(fact.split(" is ")[0] for fact in episode["facts"])
    assert episode["split_key"] == f"key_value:kv_line:{','.join(keys)}"


def test_key_value_same_template_and_keys_share_split_placement():
    split_keys = set()
    fact_lists = set()
    for seed in range(5):
        episode = generate_episode("key_value", seed, {"facts": 6, "delay": 0})
        split_keys.add(episode["split_key"])
        fact_lists.add(tuple(episode["facts"]))
    assert split_keys == {"key_value:kv_line:a,b,c,d,e,f"}
    assert len(fact_lists) > 1, "different seeds should carry different values"


def test_correction_expected_is_the_latest_value():
    for seed in range(8):
        for delay in (0, 5):
            episode = generate_episode("correction", seed, {"delay": delay})
            inputs = _input_texts(episode)
            assert len(inputs) == 2
            store, overwrite = inputs
            key_stored, old_value = store.split(" is ")
            assert overwrite.startswith("replace ")
            key_replaced, new_value = overwrite[len("replace ") :].split(" with ")
            assert key_replaced == key_stored
            assert new_value != old_value
            assert episode["expected"]["text"] == new_value
            assert episode["expected"]["text"] != old_value, "the obsolete value must be wrong"
            assert episode["facts"] == [store, f"{key_stored} is {new_value}"]
            assert episode["delay_ticks"] == delay
            assert [step["kind"] for step in episode["steps"]] == ["input"] + ["tick"] * delay + [
                "input"
            ]


def test_order_binding_ordered_list_is_bound_to_position():
    for seed in range(6):
        episode = generate_episode(
            "order_binding", seed, {"items": 4, "delay": 1, "variant": "order"}
        )
        assert episode["difficulty"]["variant"] == "order"
        items = _input_texts(episode)[0].split("\n")
        assert len(items) == 4
        assert episode["facts"] == items
        assert episode["query"].startswith("what is item ") and episode["query"].endswith("?")
        position = int(episode["query"][len("what is item ") : -1])
        assert 1 <= position <= 4
        assert episode["expected"]["text"] == items[position - 1]


def test_order_binding_pairs_are_bound_to_their_key():
    for seed in range(6):
        episode = generate_episode(
            "order_binding", seed, {"items": 3, "delay": 1, "variant": "pairs"}
        )
        assert episode["difficulty"]["variant"] == "pairs"
        binding = {}
        for fact in episode["facts"]:
            key, value = fact.split(" is ")
            binding[key] = value
        assert len(binding) == 3
        queried = _queried_key(episode["query"])
        assert episode["expected"]["text"] == binding[queried]


@pytest.mark.parametrize("scenario", ["wait", "commit", "end"])
def test_control_scenarios_carry_content_and_control_only_answers(scenario):
    expected_control = {"wait": CONTROL_WAIT, "commit": CONTROL_COMMIT, "end": CONTROL_END}[scenario]
    for seed in range(4):
        for length in (1, 9):
            episode = generate_episode("control", seed, {"scenario": scenario, "length": length})
            inputs = _input_texts(episode)
            assert inputs, "control steps must carry content"
            assert all(text != "" for text in inputs), "no all-EMPTY input surface ever"
            assert episode["expected"]["control"] == expected_control
            assert type(episode["expected"]["control"]) is int
            if scenario == "commit":
                assert len(episode["expected"]["text"]) == length
                assert episode["expected"]["text"] == inputs[0]
            else:
                assert episode["expected"]["text"] == ""
            assert episode["facts"] == []
            assert episode["delay_ticks"] == 0
            assert episode["distractor_count"] is None


def test_control_wait_is_never_an_empty_surface():
    for seed in range(4):
        episode = generate_episode("control", seed, {"scenario": "wait", "length": 4})
        assert episode["expected"]["control"] == CONTROL_WAIT
        assert episode["expected"]["text"] == ""
        for step in episode["steps"]:
            assert step["kind"] == "input"
            assert step["text"] != ""


def test_generalization_is_marked_novel_with_digit_heavy_keys():
    for seed in range(5):
        episode = generate_episode("generalization", seed, {"mechanic": "novel_key_value", "facts": 4})
        assert episode["difficulty"]["novel_combination"] is True
        assert episode["difficulty"]["mechanic"] == "novel_key_value"
        assert episode["split_key"].startswith("generalization:novel_key_value:")
        keys = [fact.split(" is ")[0] for fact in episode["facts"]]
        assert len(keys) == 4
        assert all(len(key) == 2 and key.isdigit() for key in keys)
        values = [fact.split(" is ")[1] for fact in episode["facts"]]
        assert all(len(value) == 5 for value in values)
        queried = _queried_key(episode["query"])
        assert queried in keys
        assert episode["expected"]["text"] == values[keys.index(queried)]


def test_generalization_distractors_use_a_pattern_other_families_do_not_default_to():
    for seed in range(4):
        episode = generate_episode(
            "generalization",
            seed,
            {"mechanic": "novel_distracted_recall", "length": 3, "distractors": 4},
        )
        assert episode["difficulty"]["novel_combination"] is True
        assert episode["difficulty"]["distractor_kind"] == "alternating"
        assert episode["difficulty"]["distractor_kind"] not in ("noise", "similar", "repeated")
        inputs = _input_texts(episode)
        content, distractors = inputs[0], inputs[1:]
        assert len(distractors) == 4
        assert episode["expected"]["text"] == content
        assert all(distractor != content for distractor in distractors)
        assert distractors[0] == distractors[2] and distractors[1] == distractors[3]
        assert distractors[0] != distractors[1]


def test_generalization_mechanic_is_resolved_deterministically():
    explicit = generate_episode("generalization", 2, {"facts": 2})
    assert explicit["difficulty"]["mechanic"] == "novel_key_value"
    explicit = generate_episode("generalization", 2, {"distractors": 3})
    assert explicit["difficulty"]["mechanic"] == "novel_distracted_recall"
    default = generate_episode("generalization", 2)
    assert default["difficulty"]["mechanic"] in (
        "novel_key_value",
        "novel_distracted_recall",
    )
    assert default == generate_episode("generalization", 2)


def test_effective_difficulty_reflects_defaults_and_overrides():
    default = generate_episode("delayed_recall", 1)
    assert default["difficulty"] == {"length": 3, "delay": 4}
    overridden = generate_episode("delayed_recall", 1, {"delay": 9})
    assert overridden["difficulty"] == {"length": 3, "delay": 9}


# --- fail-closed behaviour ---


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        generate_episode("telepathy", 1)


@pytest.mark.parametrize("bad_seed", [True, 1.5, -1, "1"])
def test_bad_seeds_are_rejected(bad_seed):
    with pytest.raises(ValueError):
        generate_episode("char_copy", bad_seed)


def test_bad_difficulty_container_is_rejected():
    with pytest.raises(ValueError):
        generate_episode("char_copy", 1, ["length"])


def test_unknown_difficulty_key_is_rejected():
    with pytest.raises(ValueError):
        generate_episode("char_copy", 1, {"lenght": 3})


@pytest.mark.parametrize(
    "family,difficulty",
    [
        ("char_copy", {"length": 0}),
        ("char_copy", {"length": 9}),
        ("delayed_recall", {"delay": 33}),
        ("distracted_recall", {"distractor_kind": "music"}),
        ("distracted_recall", {"distractors": 9}),
        ("key_value", {"facts": 0}),
        ("key_value", {"facts": 7}),
        ("correction", {"delay": 33}),
        ("order_binding", {"items": 1}),
        ("order_binding", {"items": 6}),
        ("control", {"scenario": "sleep"}),
        ("control", {"length": 0}),
        ("control", {"length": 33}),
        ("generalization", {"novel_combination": False}),
        ("generalization", {"mechanic": "novel_echo"}),
        ("generalization", {"mechanic": "novel_key_value", "distractors": 2}),
        ("generalization", {"mechanic": "novel_distracted_recall", "facts": 2}),
        ("generalization", {"facts": 2, "distractors": 2}),
        ("generalization", {"mechanic": "novel_distracted_recall", "distractor_kind": "noise"}),
    ],
)
def test_out_of_range_or_conflicting_difficulty_fails_closed(family, difficulty):
    with pytest.raises(ValueError):
        generate_episode(family, 1, difficulty)


# --- isolation ---


def test_module_defaults_are_never_mutated():
    snapshot = copy.deepcopy(FAMILY_DEFAULT_DIFFICULTY)
    generate_episode("generalization", 1, {"facts": 2})
    generate_episode("order_binding", 1, {"variant": "pairs"})
    generate_episode("control", 1, {"scenario": "wait"})
    assert FAMILY_DEFAULT_DIFFICULTY == snapshot


def test_caller_difficulty_is_never_mutated():
    given = {"length": 2, "delay": 1}
    original = dict(given)
    generate_episode("delayed_recall", 4, given)
    assert given == original
