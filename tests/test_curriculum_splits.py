"""Splits, metrics and preset materialization for the Axon curriculum package."""
from __future__ import annotations

import hashlib
import json

import pytest

from curriculum.schema import (
    CURRICULUM_VERSION,
    SPLIT_SCHEMA_ID,
    canonical_json,
    validate_episode,
)
from curriculum.generators import generate_episode
from curriculum.metrics import RESULTS_FIELDS, baselines, evaluate_episode
from curriculum.presets import PRESETS, materialize
from curriculum.splits import SPLIT_NAMES, assign_split, check_manifest, curriculum_sha256

OUTSIDE = "\t"  # TAB is outside the native 95; newline is native


def _episode(episode_id: str, split_key: str, text: str, control=None, family: str = "char_copy", **extra):
    episode = {
        "episode_id": episode_id,
        "family": family,
        "split_key": split_key,
        "expected": {"text": text, "control": control},
    }
    episode.update(extra)
    return episode


def _reseal(manifest: dict) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def e0_first(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("curriculum-e0-first")
    summary = materialize("e0-first", out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (out_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"dir": out_dir, "summary": summary, "manifest": manifest, "episodes": episodes}


def test_assign_split_is_deterministic_and_three_way():
    episodes = [_episode(f"ep-{index}", f"template-{index}", "x") for index in range(60)]
    buckets = [assign_split(episode) for episode in episodes]
    assert buckets == [assign_split(episode) for episode in episodes]
    assert set(buckets) == {0, 1, 2}
    assert all(assign_split(episode, buckets=2) in (0, 1) for episode in episodes)

    # The same content identity lands in the same bucket even from another row.
    assert assign_split(episodes[7]) == assign_split(_episode("ep-other", "template-7", "different text"))

    with pytest.raises(ValueError):
        assign_split({"episode_id": "no-split-key"})
    with pytest.raises(ValueError):
        assign_split(episodes[0], buckets=4)


def test_freeze_and_check_manifest_are_clean_on_generated_e0_first(e0_first):
    manifest = e0_first["manifest"]
    episodes = e0_first["episodes"]
    assert manifest["schema"] == SPLIT_SCHEMA_ID
    assert manifest["curriculum_version"] == CURRICULUM_VERSION
    assert [bucket["split"] for bucket in manifest["buckets"]] == list(SPLIT_NAMES)
    assert all(validate_episode(episode) == [] for episode in episodes)
    assert check_manifest(manifest) == []
    assert check_manifest(manifest, episodes) == []


def test_manifest_check_catches_leakage_and_duplicate_episode_ids(e0_first):
    leaked = json.loads(json.dumps(e0_first["manifest"]))
    buckets = {bucket["split"]: bucket for bucket in leaked["buckets"]}
    train_id = buckets["train"]["episode_ids"][0]
    twin_id = "episode-leak-twin"
    leaked["split_keys"][twin_id] = leaked["split_keys"][train_id]
    buckets["test"]["episode_ids"].append(twin_id)
    buckets["test"]["episode_sha256"][twin_id] = "0" * 64
    _reseal(leaked)
    errors = check_manifest(leaked)
    assert any("leakage" in error and "split_key" in error for error in errors), errors

    duplicated = json.loads(json.dumps(e0_first["manifest"]))
    dup_buckets = {bucket["split"]: bucket for bucket in duplicated["buckets"]}
    moved_id = dup_buckets["train"]["episode_ids"][0]
    dup_buckets["test"]["episode_ids"].append(moved_id)
    dup_buckets["test"]["episode_sha256"][moved_id] = dup_buckets["train"]["episode_sha256"][moved_id]
    _reseal(duplicated)
    errors = check_manifest(duplicated)
    assert any("duplicate episode_id" in error for error in errors), errors

    out_of_range = json.loads(json.dumps(e0_first["manifest"]))
    out_of_range["buckets"][0]["id"] = 7
    _reseal(out_of_range)
    errors = check_manifest(out_of_range)
    assert any("out of range" in error for error in errors), errors


def test_evaluate_episode_flags_invalid_content():
    episode = _episode("e-invalid", "copy|ab", "ab", control=1)
    clean = evaluate_episode(episode, "ab", 1)
    assert clean["invalid_content"] is False
    assert clean["exact_match"] is True

    dirty = evaluate_episode(episode, "a" + OUTSIDE, 1)
    assert dirty["invalid_content"] is True
    assert dirty["exact_match"] is False
    assert dirty["per_char_accuracy"] == pytest.approx(0.5)

    # Newline is a native character and must never be flagged as invalid.
    assert evaluate_episode(episode, "a\n", 1)["invalid_content"] is False


def test_evaluate_episode_flags_wait_confusion_and_scores_control():
    waiting = _episode("e-wait", "wait|tick", "", control=0, family="control")
    confused = evaluate_episode(waiting, "", 1)
    assert confused["wait_confusion"] is True
    assert confused["control_correct"] is False

    respected = evaluate_episode(waiting, "", 0)
    assert respected["wait_confusion"] is False
    assert respected["control_correct"] is True

    unlabeled = _episode("e-unlabeled", "copy|a", "a", control=None)
    scored = evaluate_episode(unlabeled, "a", None)
    assert scored["control_correct"] is None
    assert scored["wait_confusion"] is False


def test_evaluate_episode_scores_exact_match_and_binding_errors():
    stored = _episode("e-kv", "kv|pet", "cat", control=1, family="key_value")
    stored["facts"] = ["cat", "owl", "fox"]

    right = evaluate_episode(stored, "cat", 1)
    assert right["exact_match"] is True
    assert right["per_char_accuracy"] == 1.0
    assert right["binding_error"] is False

    wrong_fact = evaluate_episode(stored, "owl", 1)
    assert wrong_fact["exact_match"] is False
    assert wrong_fact["per_char_accuracy"] == 0.0
    assert wrong_fact["binding_error"] is True

    unrelated = evaluate_episode(stored, "bee", 1)
    assert unrelated["binding_error"] is False

    corrected = _episode("e-fix", "fix|pet", "new", control=1, family="correction")
    corrected["superseded_values"] = ["old"]
    stale = evaluate_episode(corrected, "old", 1)
    assert stale["obsolete_error"] is True
    fresh = evaluate_episode(corrected, "new", 1)
    assert fresh["obsolete_error"] is False
    assert fresh["exact_match"] is True


def test_evaluate_episode_handles_real_generated_binding_and_correction_episodes():
    key_value = generate_episode("key_value", 4242)
    other_value = next(
        fact.split(" is ", 1)[1]
        for fact in key_value["facts"]
        if fact.split(" is ", 1)[1] != key_value["expected"]["text"]
    )
    right = evaluate_episode(key_value, key_value["expected"]["text"], None)
    assert right["exact_match"] is True
    assert right["binding_error"] is False
    wrong = evaluate_episode(key_value, other_value, None)
    assert wrong["binding_error"] is True

    correction = generate_episode("correction", 4242)
    superseded = correction["steps"][0]["text"].split(" is ", 1)[1]
    stale = evaluate_episode(correction, superseded, None)
    assert stale["obsolete_error"] is True
    assert stale["binding_error"] is False
    fresh = evaluate_episode(correction, correction["expected"]["text"], None)
    assert fresh["obsolete_error"] is False
    assert fresh["exact_match"] is True


def test_baselines_are_honest_reference_numbers():
    episodes = [
        _episode("b-1", "k1", "x", control=0),
        _episode("b-2", "k2", "x", control=1),
        _episode("b-3", "k3", "y", control=0, family="delayed_recall"),
    ]
    reference = baselines(episodes)
    assert reference["constant"]["text"] == "x"
    assert reference["constant"]["accuracy_upper_bound"] == pytest.approx(2 / 3)
    assert reference["control_wait_rate"] == pytest.approx(2 / 3)
    assert set(reference["memoryless_most_common"]) == {"char_copy", "delayed_recall"}
    assert reference["memoryless_most_common"]["delayed_recall"]["text"] == "y"

    empty = baselines([])
    assert empty["constant"]["accuracy_upper_bound"] == 0.0
    assert empty["control_wait_rate"] == 0.0
    assert empty["memoryless_most_common"] == {}


def test_results_fields_cover_the_results_row():
    required = {
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
    }
    assert required <= set(RESULTS_FIELDS)
    assert len(RESULTS_FIELDS) == len(set(RESULTS_FIELDS))


def test_materialize_e0_first_matches_preset_sizes_and_verifies_hashes(e0_first):
    out_dir = e0_first["dir"]
    manifest = e0_first["manifest"]
    episodes = e0_first["episodes"]
    summary = e0_first["summary"]

    report = (out_dir / "REPORT.txt").read_text(encoding="utf-8")
    assert "e0-first" in report and "curriculum_sha256" in report

    buckets = {bucket["split"]: bucket for bucket in manifest["buckets"]}
    by_id = {episode["episode_id"]: episode for episode in episodes}
    preset_families = PRESETS["e0-first"]["families"]

    for family, spec in preset_families.items():
        for split in SPLIT_NAMES:
            in_bucket = [
                episode_id
                for episode_id in buckets[split]["episode_ids"]
                if by_id[episode_id]["family"] == family
            ]
            assert len(in_bucket) == spec[split], (family, split)
    assert len(episodes) == sum(
        spec[split] for spec in preset_families.values() for split in SPLIT_NAMES
    )

    for bucket in manifest["buckets"]:
        for episode_id in bucket["episode_ids"]:
            digest = hashlib.sha256(canonical_json(by_id[episode_id]).encode("utf-8")).hexdigest()
            assert bucket["episode_sha256"][episode_id] == digest
    for episode_id, episode in by_id.items():
        assert manifest["split_keys"][episode_id] == episode["split_key"]

    assert curriculum_sha256(episodes) == summary["curriculum_sha256"]
    assert manifest["manifest_sha256"] == summary["manifest_sha256"]

    with pytest.raises(ValueError):
        materialize("no-such-preset", out_dir)
