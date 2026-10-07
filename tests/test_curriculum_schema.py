"""Contract tests for the architecture-independent curriculum episode schema.

Fast and deliberately small: hand-built episodes only, no generation and no data
dumps. The hard Axon rule under test is that WAIT is control state, never an
all-EMPTY content surface, and that characters outside the native 95 fail closed.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator

from curriculum import (
    CONTROL_COMMIT,
    CONTROL_END,
    CONTROL_VALUES,
    CONTROL_WAIT,
    CURRICULUM_VERSION,
    EPISODE_FIELDS,
    EPISODE_JSON_SCHEMA,
    EPISODE_SCHEMA_ID,
    FAMILIES,
    RESULTS_SCHEMA_ID,
    SPLIT_SCHEMA_ID,
    STEP_KINDS,
    canonical_json,
    validate_episode,
)
from curriculum.schema import canonical_hash
from substrate.native import ALPHABET

#: Characters that must never be admitted (TAB and backtick are not native; the
#: accented letter, NBSP, NUL, CR, emoji are outside the frozen 95).
OUTSIDERS = ("\t", "`", "\u00e9", "\u00a0", "\u0000", "\r", "\U0001F600")

TEXT_FIELDS = ("step_text", "query", "expected_text", "facts")

SCHEMA_VALIDATOR = Draft202012Validator(EPISODE_JSON_SCHEMA)


def minimal_episode() -> dict:
    """A valid, hand-built episode: one char copy with a silent tick and a newline target."""
    return {
        "episode_id": "char_copy-0001",
        "family": "char_copy",
        "seed": 11,
        "steps": [
            {"kind": "input", "text": "a", "control": None},
            {"kind": "tick", "text": "", "control": None},
            {"kind": "input", "text": "\n", "control": None},
        ],
        "query": "copy the last input",
        "expected": {"text": "a\n", "control": CONTROL_COMMIT},
        "facts": [],
        "delay_ticks": 1,
        "distractor_count": 0,
        "difficulty": {"sequence_length": 2, "delay_ticks": 1},
        "split_key": "train/char_copy/len2/delay1",
    }


def make_episode(**overrides) -> dict:
    episode = minimal_episode()
    episode.update(overrides)
    return episode


def key_value_episode() -> dict:
    """A valid episode with stored facts, a query and a distractor count."""
    return {
        "episode_id": "key_value-0007",
        "family": "key_value",
        "seed": 23,
        "steps": [
            {"kind": "input", "text": "red=1", "control": None},
            {"kind": "tick", "text": "", "control": None},
            {"kind": "input", "text": "blue=2", "control": None},
            {"kind": "input", "text": "blue?", "control": None},
        ],
        "query": "blue?",
        "expected": {"text": "2", "control": CONTROL_COMMIT},
        "facts": ["red=1", "blue=2"],
        "delay_ticks": 1,
        "distractor_count": 1,
        "difficulty": {"facts": 2, "delay_ticks": 1, "distractor_count": 1},
        "split_key": "train/key_value/colors",
    }


def mutated(name: str) -> dict:
    """Named single-rule violations, used to keep the validator and the JSON Schema in agreement."""
    episode = minimal_episode()
    if name == "outsider_in_step_text":
        episode["steps"][0]["text"] = "a\t"
    elif name == "outsider_in_query":
        episode["query"] = "q\u00e9"
    elif name == "outsider_in_expected_text":
        episode["expected"]["text"] = "\U0001F600"
    elif name == "outsider_in_facts":
        episode["facts"] = ["ok", "\u00a0"]
    elif name == "empty_input_step":
        episode["steps"] = [{"kind": "input", "text": "", "control": None}]
    elif name == "tick_with_content":
        episode["steps"] = [{"kind": "tick", "text": "x", "control": None}]
    elif name == "reset_with_content":
        episode["steps"] = [{"kind": "reset", "text": "x", "control": None}]
    elif name == "step_control_not_null":
        episode["steps"][0]["control"] = CONTROL_WAIT
    elif name == "unknown_step_kind":
        episode["steps"][0]["kind"] = "wait"
    elif name == "unknown_family":
        episode["family"] = "spelling_bee"
    elif name == "missing_split_key":
        del episode["split_key"]
    elif name == "no_steps":
        episode["steps"] = []
    elif name == "control_out_of_range":
        episode["expected"]["control"] = 3
    elif name == "delay_not_int":
        episode["delay_ticks"] = "1"
    elif name == "empty_split_key":
        episode["split_key"] = ""
    else:  # pragma: no cover - guards the test table itself
        raise AssertionError(f"unknown mutation {name!r}")
    return episode


REJECTED_MUTATIONS = (
    "outsider_in_step_text",
    "outsider_in_query",
    "outsider_in_expected_text",
    "outsider_in_facts",
    "empty_input_step",
    "tick_with_content",
    "reset_with_content",
    "step_control_not_null",
    "unknown_step_kind",
    "unknown_family",
    "missing_split_key",
    "no_steps",
    "control_out_of_range",
    "delay_not_int",
    "empty_split_key",
)


def test_curriculum_version_and_schema_ids_are_frozen():
    assert CURRICULUM_VERSION == "0.1.0"
    assert EPISODE_SCHEMA_ID == "axon-curriculum-episode-v1"
    assert RESULTS_SCHEMA_ID == "axon-curriculum-results-v1"
    assert SPLIT_SCHEMA_ID == "axon-curriculum-split-v1"


def test_control_constants_mirror_the_core_triple():
    assert (CONTROL_WAIT, CONTROL_COMMIT, CONTROL_END) == (0, 1, 2)
    assert CONTROL_VALUES == (0, 1, 2)
    assert STEP_KINDS == ("input", "tick", "reset")


def test_families_are_the_eight_step_progression():
    assert FAMILIES == (
        "char_copy",
        "delayed_recall",
        "distracted_recall",
        "key_value",
        "correction",
        "order_binding",
        "control",
        "generalization",
    )
    assert len(set(FAMILIES)) == len(FAMILIES) == 8


def test_episode_record_carries_exactly_the_documented_fields():
    assert set(minimal_episode()) == set(EPISODE_FIELDS)
    assert len(EPISODE_FIELDS) == 11


def test_outsiders_are_really_outside_the_native_alphabet():
    assert len(ALPHABET) == 95
    for character in OUTSIDERS:
        assert character not in ALPHABET, f"U+{ord(character):04X} is native; pick a real outsider"


def test_minimal_episode_is_valid_and_the_validator_does_not_mutate_it():
    episode = minimal_episode()
    before = copy.deepcopy(episode)
    assert validate_episode(episode) == []
    assert episode == before


def test_facts_episode_is_valid():
    assert validate_episode(key_value_episode()) == []


def test_every_native_character_is_admitted():
    for character in ALPHABET:
        episode = make_episode(
            steps=[{"kind": "input", "text": character, "control": None}],
            expected={"text": character, "control": CONTROL_COMMIT},
        )
        assert validate_episode(episode) == [], f"native U+{ord(character):04X} was rejected"
        assert not list(SCHEMA_VALIDATOR.iter_errors(episode)), f"schema rejected native U+{ord(character):04X}"


def test_newline_is_native_content_not_an_empty_surface():
    episode = make_episode(
        steps=[{"kind": "input", "text": "\n", "control": None}],
        query="a\nb",
        expected={"text": "\n\n", "control": CONTROL_COMMIT},
        facts=["a\n"],
    )
    assert validate_episode(episode) == []


@pytest.mark.parametrize("field", TEXT_FIELDS)
@pytest.mark.parametrize("outsider", OUTSIDERS)
def test_outsider_characters_fail_closed_in_every_text_field(field, outsider):
    episode = minimal_episode()
    if field == "step_text":
        episode["steps"][0]["text"] = "a" + outsider
    elif field == "query":
        episode["query"] = outsider + "query"
    elif field == "expected_text":
        episode["expected"]["text"] = outsider
    else:
        episode["facts"] = ["ok", outsider]

    errors = validate_episode(episode)
    assert errors, f"{field} accepted outsider U+{ord(outsider):04X}"
    assert any(f"U+{ord(outsider):04X}" in error for error in errors), errors
    assert list(SCHEMA_VALIDATOR.iter_errors(episode)), "sanity: the JSON Schema must reject this record too"


@pytest.mark.parametrize("outsider", OUTSIDERS)
def test_no_text_field_is_silently_converted(outsider):
    """Fail closed means reported, never repaired: the outsider survives untouched in the episode."""
    episode = minimal_episode()
    episode["query"] = outsider
    assert validate_episode(episode)
    assert episode["query"] == outsider


def test_empty_input_step_is_an_all_empty_surface_and_is_rejected():
    episode = make_episode(steps=[{"kind": "input", "text": "", "control": None}])
    errors = validate_episode(episode)
    assert any("non-empty" in error for error in errors), errors


def test_tick_and_reset_steps_carry_no_content():
    for kind in ("tick", "reset"):
        assert validate_episode(make_episode(steps=[{"kind": kind, "text": "", "control": None}])) == []
        errors = validate_episode(make_episode(steps=[{"kind": kind, "text": "x", "control": None}]))
        assert any("must be empty" in error for error in errors), errors


def test_reset_step_is_a_valid_episode_opener():
    episode = make_episode(steps=[{"kind": "reset", "text": "", "control": None},
                                  {"kind": "input", "text": "z", "control": None}])
    assert validate_episode(episode) == []


@pytest.mark.parametrize("control", [CONTROL_WAIT, CONTROL_COMMIT, CONTROL_END])
def test_a_step_may_never_carry_control_state(control):
    episode = make_episode(steps=[{"kind": "input", "text": "a", "control": control}])
    errors = validate_episode(episode)
    assert any("control must be null" in error for error in errors), errors


def test_step_requires_all_three_keys():
    episode = make_episode(steps=[{"kind": "input", "text": "a"}])
    assert any(".control is required" in error for error in validate_episode(episode))
    episode = make_episode(steps=[{"kind": "input", "control": None}])
    assert any(".text is required" in error for error in validate_episode(episode))


def test_steps_must_be_a_non_empty_list():
    assert any("must not be empty" in error for error in validate_episode(make_episode(steps=[])))
    assert any("must be a list" in error for error in validate_episode(make_episode(steps="input")))


def test_unknown_step_kind_is_rejected():
    errors = validate_episode(make_episode(steps=[{"kind": "wait", "text": "a", "control": None}]))
    assert any(".kind must be one of" in error for error in errors), errors


def test_unknown_family_is_rejected():
    errors = validate_episode(make_episode(family="spelling_bee"))
    assert any(error.startswith("family must be one of") for error in errors), errors


@pytest.mark.parametrize("control", [CONTROL_WAIT, CONTROL_COMMIT, CONTROL_END, None])
def test_expected_control_accepts_the_triple_and_null(control):
    assert validate_episode(make_episode(expected={"text": "", "control": control})) == []


@pytest.mark.parametrize("control", [3, -1, 4, True, False, 2.0, "1", [1]])
def test_expected_control_outside_the_triple_is_rejected(control):
    errors = validate_episode(make_episode(expected={"text": "a", "control": control}))
    assert any(error.startswith("expected.control must be one of") for error in errors), errors


def test_expected_requires_text_and_control():
    assert any("expected.text is required" in error for error in validate_episode(make_episode(expected={"control": 1})))
    assert any("expected.control is required" in error for error in validate_episode(make_episode(expected={"text": "a"})))
    assert any("expected must be an object" in error for error in validate_episode(make_episode(expected=[1])))


@pytest.mark.parametrize("value", [None, 0, 1, 12])
def test_delay_and_distractor_count_accept_int_or_null(value):
    episode = make_episode(delay_ticks=value, distractor_count=value)
    assert validate_episode(episode) == []


@pytest.mark.parametrize("value", ["3", 2.5, True, [3]])
def test_delay_and_distractor_count_reject_everything_else(value):
    errors = validate_episode(make_episode(delay_ticks=value))
    assert any(error.startswith("delay_ticks must be an int or null") for error in errors), errors
    errors = validate_episode(make_episode(distractor_count=value))
    assert any(error.startswith("distractor_count must be an int or null") for error in errors), errors


@pytest.mark.parametrize("seed", [0, 7, 2**40])
def test_seed_accepts_ints(seed):
    assert validate_episode(make_episode(seed=seed)) == []


@pytest.mark.parametrize("seed", ["7", 7.0, True, None])
def test_seed_rejects_non_ints(seed):
    assert any(error.startswith("seed must be an int") for error in validate_episode(make_episode(seed=seed)))


@pytest.mark.parametrize("value", ["", 3, None, ["train"]])
def test_split_key_must_be_a_non_empty_string(value):
    errors = validate_episode(make_episode(split_key=value))
    assert any(error.startswith("split_key must be a non-empty str") for error in errors), errors


def test_facts_must_be_a_list_of_native_strings():
    assert any("facts must be a list" in error for error in validate_episode(make_episode(facts="a")))
    assert any("facts[0] must be str" in error for error in validate_episode(make_episode(facts=[7])))


def test_difficulty_must_be_an_object():
    assert any("difficulty must be an object" in error for error in validate_episode(make_episode(difficulty=3)))


@pytest.mark.parametrize("field", EPISODE_FIELDS)
def test_every_missing_required_field_is_reported(field):
    episode = minimal_episode()
    del episode[field]
    errors = validate_episode(episode)
    assert any(error.startswith(f"{field} is required") for error in errors), errors


def test_an_empty_dict_reports_every_field_once():
    errors = validate_episode({})
    assert len(errors) == len(EPISODE_FIELDS)
    assert all(isinstance(error, str) and error for error in errors)


@pytest.mark.parametrize("garbage", [None, "episode", 42, [], {"steps": "no"}, {"family": ["char_copy"]},
                                     {"expected": {"control": [1]}}, {"steps": [7]}, {"steps": [{}]}])
def test_validator_never_raises_on_garbage(garbage):
    errors = validate_episode(garbage)
    assert isinstance(errors, list) and errors
    assert all(isinstance(error, str) and error for error in errors)


@pytest.mark.parametrize("name", REJECTED_MUTATIONS)
def test_validator_and_json_schema_reject_the_same_records(name):
    episode = mutated(name)
    errors = validate_episode(episode)
    assert errors, f"validator accepted {name}"
    assert list(SCHEMA_VALIDATOR.iter_errors(episode)), f"JSON Schema accepted {name}"
    assert all(isinstance(error, str) and error for error in errors)


@pytest.mark.parametrize("episode", [minimal_episode(), key_value_episode()])
def test_validator_and_json_schema_accept_the_same_records(episode):
    assert validate_episode(episode) == []
    assert list(SCHEMA_VALIDATOR.iter_errors(episode)) == []


def test_episode_json_schema_is_a_compilable_draft_2020_12_schema():
    Draft202012Validator.check_schema(EPISODE_JSON_SCHEMA)
    assert EPISODE_JSON_SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert EPISODE_JSON_SCHEMA["$id"] == EPISODE_SCHEMA_ID
    assert set(EPISODE_JSON_SCHEMA["required"]) == set(EPISODE_FIELDS)
    assert EPISODE_JSON_SCHEMA["properties"]["family"]["enum"] == list(FAMILIES)


def test_canonical_json_sorts_keys_and_drops_padding():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_json({"a": 2, "b": 1}) == canonical_json({"b": 1, "a": 2})
    assert ", " not in canonical_json({"a": 1, "b": [1, 2]})
    assert ": " not in canonical_json({"a": 1, "b": [1, 2]})


def test_canonical_json_keeps_native_text_literal():
    assert canonical_json({"k": "\n"}) == '{"k":"\\n"}'
    assert canonical_json({"k": "\u00e9"}) == '{"k":"\u00e9"}'
    assert "\\u00e9" not in canonical_json({"k": "\u00e9"})


def test_canonical_json_round_trips_and_is_stable():
    episode = key_value_episode()
    text = canonical_json(episode)
    assert json.loads(text) == episode
    assert canonical_json(json.loads(text)) == text


def test_canonical_json_refuses_non_json_floats():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json({"value": value})


def test_canonical_hash_is_the_sha256_of_canonical_json():
    episode = key_value_episode()
    expected = hashlib.sha256(canonical_json(episode).encode("utf-8")).hexdigest()
    assert canonical_hash(episode) == expected
    assert len(expected) == 64
    reordered = {key: episode[key] for key in reversed(list(episode))}
    assert canonical_hash(reordered) == expected
