from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.field import canonical_json_bytes, canonical_sha256
from training.build_conversational_curriculum import (
    CANONICAL_REGIONS,
    EXAMPLE_SCHEMA,
    MANIFEST_SCHEMA,
)
from training.conversational_no_gold import (
    NoGoldEvaluationCase,
    NoGoldEvaluationError,
    NoGoldPrediction,
    evaluate_no_gold_examples,
    evaluate_no_gold_jsonl,
    load_exportable_conversational_examples,
)


ROOT = Path(__file__).resolve().parent.parent


def _example(
    *,
    user: str,
    target: str | None,
    split: str = "test",
    source_id: str = "public-fixture",
) -> dict:
    active = {region: "" for region in CANONICAL_REGIONS}
    active["user_input"] = user
    base = {
        "schema": EXAMPLE_SCHEMA,
        "example_kind": (
            "structural_noop"
            if target is None
            else "response_draft_replacement"
        ),
        "split": split,
        "source": {
            "source_id": source_id,
            "source_sha256": "1" * 64,
            "conversation_sha256": "2" * 64,
            "assistant_turn_index": 1,
            "derived_example_index": 0,
            "license": {},
            "provenance": {},
            "privacy": {},
        },
        "active_field": active,
        "gold_exclusion_audit": {
            "passed": True,
            "visible_region_names": list(CANONICAL_REGIONS),
            "gold_present_in_visible_regions": [],
        },
        "target_delta": (
            None
            if target is None
            else {
                "op": "replace",
                "region": "response_draft",
                "text": target,
                "char_count": len(target),
                "complete_replacement": True,
            }
        ),
        "structural_noop": (
            {
                "op": "no_op",
                "reason": "No visible update is required.",
                "reason_char_count": 30,
            }
            if target is None
            else None
        ),
    }
    base["example_id"] = canonical_sha256(base)
    return base


def _visible_user(case: NoGoldEvaluationCase) -> str:
    return dict(case.active_field)["user_input"]


def test_predictor_receives_only_visible_immutable_case_then_scores() -> None:
    examples = [
        _example(user="Give the first code.", target="alpha"),
        _example(user="Give the second code.", target="beta"),
        _example(user="Do nothing.", target=None),
    ]
    original = deepcopy(examples)
    observed: list[NoGoldEvaluationCase] = []

    def predictor(case: NoGoldEvaluationCase) -> NoGoldPrediction:
        observed.append(case)
        assert not hasattr(case, "target_delta")
        assert not hasattr(case, "target_text")
        assert not hasattr(case, "example_kind")
        assert not hasattr(case, "example_id")
        user = _visible_user(case)
        if user == "Do nothing.":
            return NoGoldPrediction(case.case_id, "no_op", None)
        return NoGoldPrediction(
            case.case_id,
            "replace_response_draft",
            "alpha" if user == "Give the first code." else "beta",
        )

    report = evaluate_no_gold_examples(examples, predictor)

    assert examples == original
    assert len(observed) == 3
    assert report.metric("case_count") == 3
    assert report.metric("decision_accuracy") == 1.0
    assert report.metric("replacement_exact_text_accuracy") == 1.0
    assert report.metric("replacement_exact_length_accuracy") == 1.0
    assert report.metric("replacement_character_accuracy") == 1.0
    assert report.metric("answer_required_nonempty_rate") == 1.0
    assert report.metric("structural_noop_accuracy") == 1.0
    contract = report.to_dict()["no_gold_contract"]
    assert contract["teacher_forcing_allowed"] is False
    assert "target_delta" not in contract["predictor_input_fields"]
    assert "target_delta" in contract[
        "withheld_until_all_predictions_collected"
    ]


def test_metrics_measure_empty_collapse_length_and_character_errors() -> None:
    examples = [
        _example(user="First.", target="abcd"),
        _example(user="Second.", target="wxyz"),
    ]

    def predictor(case: NoGoldEvaluationCase) -> NoGoldPrediction:
        text = "" if _visible_user(case) == "First." else "wxzzzz"
        return NoGoldPrediction(
            case.case_id,
            "replace_response_draft",
            text,
        )

    report = evaluate_no_gold_examples(examples, predictor)

    assert report.metric("decision_accuracy") == 1.0
    assert report.metric("replacement_exact_text_accuracy") == 0.0
    assert report.metric("replacement_exact_length_accuracy") == 0.0
    assert report.metric("answer_required_nonempty_rate") == 0.5
    assert report.metric("answer_required_empty_rate") == 0.5
    assert report.metric("first_position_empty_rate") == 0.5
    assert report.metric("replacement_character_accuracy") == pytest.approx(
        3 / 10
    )
    assert dict(report.predicted_length_bins) == {
        "empty": 1,
        "1-16": 1,
        "17-32": 0,
        "33-48": 0,
        "49-64": 0,
    }


def test_suite_and_predictions_are_order_independent_and_deterministic() -> None:
    examples = [
        _example(user="One.", target="one"),
        _example(user="Two.", target="two"),
    ]

    def predictor(case: NoGoldEvaluationCase) -> NoGoldPrediction:
        return NoGoldPrediction(
            case.case_id,
            "replace_response_draft",
            "one" if _visible_user(case) == "One." else "two",
        )

    forward = evaluate_no_gold_examples(examples, predictor)
    reverse = evaluate_no_gold_examples(list(reversed(examples)), predictor)

    assert forward == reverse
    assert forward.visible_suite_sha256 == reverse.visible_suite_sha256
    assert forward.gold_suite_sha256 == reverse.gold_suite_sha256
    assert forward.prediction_set_sha256 == reverse.prediction_set_sha256


def test_same_visible_case_with_different_hidden_labels_is_rejected() -> None:
    examples = [
        _example(user="Ambiguous.", target="first", source_id="a"),
        _example(user="Ambiguous.", target="second", source_id="b"),
    ]

    with pytest.raises(NoGoldEvaluationError, match="hidden labels"):
        evaluate_no_gold_examples(
            examples,
            lambda case: NoGoldPrediction(
                case.case_id,
                "replace_response_draft",
                "first",
            ),
        )


def test_gold_visibility_is_recomputed_not_trusted_from_audit() -> None:
    example = _example(user="The answer is leaked.", target="leaked")

    with pytest.raises(NoGoldEvaluationError, match="gold appears"):
        evaluate_no_gold_examples(
            [example],
            lambda case: NoGoldPrediction(
                case.case_id,
                "replace_response_draft",
                "leaked",
            ),
        )


def test_predictor_cannot_return_mapping_wrong_case_or_unsupported_text() -> None:
    example = _example(user="Question.", target="answer")

    with pytest.raises(NoGoldEvaluationError, match="exact NoGoldPrediction"):
        evaluate_no_gold_examples([example], lambda _case: {})  # type: ignore[arg-type]

    with pytest.raises(NoGoldEvaluationError, match="does not match"):
        evaluate_no_gold_examples(
            [example],
            lambda _case: NoGoldPrediction(
                "0" * 64,
                "replace_response_draft",
                "answer",
            ),
        )

    with pytest.raises(ValueError):
        evaluate_no_gold_examples(
            [example],
            lambda case: NoGoldPrediction(
                case.case_id,
                "replace_response_draft",
                "emoji \U0001f642",
            ),
        )


def test_predictor_mutation_of_original_suite_is_rejected() -> None:
    example = _example(user="Question.", target="answer")

    def predictor(case: NoGoldEvaluationCase) -> NoGoldPrediction:
        example["active_field"]["scratch"] = "mutated"
        return NoGoldPrediction(
            case.case_id,
            "replace_response_draft",
            "answer",
        )

    with pytest.raises(NoGoldEvaluationError, match="mutated"):
        evaluate_no_gold_examples([example], predictor)


def test_jsonl_manifest_verification_and_evaluation(tmp_path: Path) -> None:
    examples = [
        _example(user="Question.", target="answer", split="test"),
        _example(user="No operation.", target=None, split="dev"),
    ]
    payload = b"".join(
        canonical_json_bytes(example) + b"\n" for example in examples
    )
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_bytes(payload)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "artifacts": {
            "examples_jsonl": {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )

    loaded = load_exportable_conversational_examples(
        examples_path,
        manifest_path=manifest_path,
    )
    assert len(loaded) == 2

    def predictor(case: NoGoldEvaluationCase) -> NoGoldPrediction:
        if _visible_user(case) == "No operation.":
            return NoGoldPrediction(case.case_id, "no_op", None)
        return NoGoldPrediction(
            case.case_id,
            "replace_response_draft",
            "answer",
        )

    report = evaluate_no_gold_jsonl(
        examples_path,
        predictor,
        manifest_path=manifest_path,
    )
    assert report.metric("decision_accuracy") == 1.0

    manifest["artifacts"]["examples_jsonl"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NoGoldEvaluationError, match="does not match"):
        load_exportable_conversational_examples(
            examples_path,
            manifest_path=manifest_path,
        )


def test_d00_inputs_are_rejected_before_read() -> None:
    for path in (
        r"D:\00\private-conversations.jsonl",
        r"D:\Axon\..\00\private-conversations.jsonl",
    ):
        with pytest.raises(NoGoldEvaluationError, match=r"D:\\00"):
            load_exportable_conversational_examples(path)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")

    with pytest.raises(NoGoldEvaluationError, match="duplicate key"):
        load_exportable_conversational_examples(examples_path)


def test_import_has_no_artifact_writes(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import training.conversational_no_gold",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []
