"""No-gold evaluation harness for the exportable conversational curriculum.

The predictor receives an immutable case containing only the canonical active
field and a visible-payload-derived case ID.  Targets, action labels, source
metadata, structural no-op reasons, and the gold-derived example ID remain in
the evaluator and are consulted only after every prediction has been
collected.

This module does not load a model, start training, or write artifacts.  It is
an isolated contract that a later neural adapter can call.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from runtime.field import canonical_json_bytes, canonical_sha256
from substrate import assert_supported_text

from .build_conversational_curriculum import (
    CANONICAL_REGIONS,
    EXAMPLE_SCHEMA,
    MANIFEST_SCHEMA,
    MAX_TARGET_CHARS,
)


NO_GOLD_CASE_SCHEMA = "axon-conversational-no-gold-case-v1"
NO_GOLD_PREDICTION_SCHEMA = "axon-conversational-no-gold-prediction-v1"
NO_GOLD_REPORT_SCHEMA = "axon-conversational-no-gold-report-v1"
MAX_EXAMPLES_BYTES = 256 * 1024 * 1024
_ACTIONS = frozenset({"replace_response_draft", "no_op"})
_EXAMPLE_KINDS = frozenset(
    {"response_draft_replacement", "structural_noop"}
)
_SPLITS = frozenset({"train", "dev", "test"})
_OUTER_KEYS = frozenset(
    {
        "schema",
        "example_kind",
        "split",
        "source",
        "active_field",
        "gold_exclusion_audit",
        "target_delta",
        "structural_noop",
        "example_id",
    }
)


class NoGoldEvaluationError(ValueError):
    """The suite, predictor result, or no-gold boundary is invalid."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NoGoldEvaluationError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise NoGoldEvaluationError(f"{label} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise NoGoldEvaluationError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        qualifier = "" if allow_empty else " non-empty"
        raise NoGoldEvaluationError(f"{label} must be a{qualifier} string")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise NoGoldEvaluationError(f"{label} must be a 64-character SHA-256")
    return text.upper()


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise NoGoldEvaluationError(f"{label} must be a list or tuple")
    return tuple(value)


def _mean(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _length_bin(length: int) -> str:
    if not 0 <= length <= MAX_TARGET_CHARS:
        raise NoGoldEvaluationError("predicted length is outside decoder width")
    if length == 0:
        return "empty"
    if length <= 16:
        return "1-16"
    if length <= 32:
        return "17-32"
    if length <= 48:
        return "33-48"
    return "49-64"


def _path_mentions_d00(path: Path) -> bool:
    try:
        normalized_path = path.resolve(strict=False)
    except OSError:
        normalized_path = path.absolute()
    normalized = str(normalized_path).replace("/", "\\")
    parts = tuple(part.casefold() for part in PureWindowsPath(normalized).parts)
    return len(parts) >= 2 and parts[0].casefold() == "d:\\" and parts[1] == "00"


def _parse_json_without_duplicate_keys(text: str, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NoGoldEvaluationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise NoGoldEvaluationError(f"{label} is invalid JSON") from exc


@dataclass(frozen=True, slots=True)
class NoGoldEvaluationCase:
    """The complete information made visible to a predictor."""

    case_id: str
    active_field: tuple[tuple[str, str], ...]
    schema: str = field(default=NO_GOLD_CASE_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _sha256(self.case_id, "case_id"))
        if type(self.active_field) is not tuple:
            raise NoGoldEvaluationError("active_field must be a tuple")
        expected_names = tuple(sorted(CANONICAL_REGIONS))
        observed_names: list[str] = []
        normalized: list[tuple[str, str]] = []
        for index, item in enumerate(self.active_field):
            if type(item) is not tuple or len(item) != 2:
                raise NoGoldEvaluationError(
                    f"active_field item {index} must be a pair"
                )
            region = _string(item[0], f"active_field item {index} region")
            text = _string(
                item[1],
                f"active_field item {index} text",
                allow_empty=True,
            )
            assert_supported_text(text)
            observed_names.append(region)
            normalized.append((region, text))
        if tuple(observed_names) != expected_names:
            raise NoGoldEvaluationError(
                "active_field must contain every canonical region in sorted order"
            )
        object.__setattr__(self, "active_field", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "case_id": self.case_id,
            "active_field": dict(self.active_field),
        }


@dataclass(frozen=True, slots=True)
class NoGoldPrediction:
    """One typed predictor result returned without access to gold."""

    case_id: str
    action: Literal["replace_response_draft", "no_op"]
    response_text: str | None
    schema: str = field(default=NO_GOLD_PREDICTION_SCHEMA, init=False)
    prediction_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _sha256(self.case_id, "case_id"))
        action = _string(self.action, "action")
        if action not in _ACTIONS:
            raise NoGoldEvaluationError(f"unsupported action {action!r}")
        if action == "no_op":
            if self.response_text is not None:
                raise NoGoldEvaluationError("no_op cannot carry response_text")
        else:
            text = _string(
                self.response_text,
                "response_text",
                allow_empty=True,
            )
            assert_supported_text(text)
            if len(text) > MAX_TARGET_CHARS:
                raise NoGoldEvaluationError(
                    f"response_text exceeds {MAX_TARGET_CHARS} characters"
                )
            object.__setattr__(self, "response_text", text)
        object.__setattr__(
            self,
            "prediction_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "case_id": self.case_id,
            "action": self.action,
            "response_text": self.response_text,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["prediction_id"] = self.prediction_id
        return value


@dataclass(frozen=True, slots=True)
class NoGoldCaseOutcome:
    """Gold-aware outcome created only after prediction collection finishes."""

    case_id: str
    example_id: str
    split: str
    expected_action: str
    predicted_action: str
    target_length: int
    predicted_length: int
    decision_correct: bool
    exact_text: bool | None
    exact_length: bool | None
    positional_character_matches: int
    positional_character_denominator: int
    prediction_id: str


@dataclass(frozen=True, slots=True)
class NoGoldEvaluationReport:
    """Deterministic suite identity, aggregate metrics, and case outcomes."""

    visible_suite_sha256: str
    gold_suite_sha256: str
    prediction_set_sha256: str
    evaluated_splits: tuple[str, ...]
    metrics: tuple[tuple[str, int | float], ...]
    predicted_length_bins: tuple[tuple[str, int], ...]
    target_length_bins: tuple[tuple[str, int], ...]
    outcomes: tuple[NoGoldCaseOutcome, ...]
    schema: str = field(default=NO_GOLD_REPORT_SCHEMA, init=False)

    def metric(self, name: str) -> int | float:
        try:
            return dict(self.metrics)[name]
        except KeyError as exc:
            raise KeyError(f"unknown no-gold metric {name!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "visible_suite_sha256": self.visible_suite_sha256,
            "gold_suite_sha256": self.gold_suite_sha256,
            "prediction_set_sha256": self.prediction_set_sha256,
            "evaluated_splits": list(self.evaluated_splits),
            "no_gold_contract": {
                "predictor_input_fields": [
                    "schema",
                    "case_id",
                    "active_field",
                ],
                "withheld_until_all_predictions_collected": [
                    "example_kind",
                    "target_delta",
                    "structural_noop",
                    "source",
                    "example_id",
                ],
                "teacher_forcing_allowed": False,
                "gold_visible_in_active_field": False,
            },
            "metrics": dict(self.metrics),
            "predicted_length_bins": dict(self.predicted_length_bins),
            "target_length_bins": dict(self.target_length_bins),
            "outcomes": [
                {
                    "case_id": outcome.case_id,
                    "example_id": outcome.example_id,
                    "split": outcome.split,
                    "expected_action": outcome.expected_action,
                    "predicted_action": outcome.predicted_action,
                    "target_length": outcome.target_length,
                    "predicted_length": outcome.predicted_length,
                    "decision_correct": outcome.decision_correct,
                    "exact_text": outcome.exact_text,
                    "exact_length": outcome.exact_length,
                    "positional_character_matches": (
                        outcome.positional_character_matches
                    ),
                    "positional_character_denominator": (
                        outcome.positional_character_denominator
                    ),
                    "prediction_id": outcome.prediction_id,
                }
                for outcome in self.outcomes
            ],
        }


@dataclass(frozen=True, slots=True)
class _GoldCase:
    example_id: str
    split: str
    case: NoGoldEvaluationCase
    expected_action: str
    target_text: str | None


def _parse_example(value: Any, index: int) -> _GoldCase:
    item = _mapping(value, f"example {index}")
    _exact_keys(item, _OUTER_KEYS, f"example {index}")
    if item["schema"] != EXAMPLE_SCHEMA:
        raise NoGoldEvaluationError(f"example {index} has unknown schema")
    example_id = _sha256(item["example_id"], f"example {index} example_id")
    payload_without_id = dict(item)
    payload_without_id.pop("example_id")
    if canonical_sha256(payload_without_id).upper() != example_id:
        raise NoGoldEvaluationError(f"example {index} example_id mismatch")
    split = _string(item["split"], f"example {index} split")
    if split not in _SPLITS:
        raise NoGoldEvaluationError(f"example {index} has invalid split")

    active = _mapping(item["active_field"], f"example {index} active_field")
    if frozenset(active) != frozenset(CANONICAL_REGIONS):
        raise NoGoldEvaluationError(
            f"example {index} active_field region set mismatch"
        )
    active_pairs = tuple(
        (
            region,
            _string(
                active[region],
                f"example {index} active_field.{region}",
                allow_empty=True,
            ),
        )
        for region in sorted(CANONICAL_REGIONS)
    )
    for _, text in active_pairs:
        assert_supported_text(text)

    audit = _mapping(
        item["gold_exclusion_audit"],
        f"example {index} gold_exclusion_audit",
    )
    _exact_keys(
        audit,
        frozenset(
            {
                "passed",
                "visible_region_names",
                "gold_present_in_visible_regions",
            }
        ),
        f"example {index} gold_exclusion_audit",
    )
    if audit["passed"] is not True:
        raise NoGoldEvaluationError(
            f"example {index} gold exclusion audit did not pass"
        )
    if tuple(audit["visible_region_names"]) != tuple(CANONICAL_REGIONS):
        raise NoGoldEvaluationError(
            f"example {index} gold audit region order mismatch"
        )
    if _sequence(
        audit["gold_present_in_visible_regions"],
        f"example {index} gold_present_in_visible_regions",
    ):
        raise NoGoldEvaluationError(
            f"example {index} claims visible gold"
        )

    kind = _string(item["example_kind"], f"example {index} example_kind")
    if kind not in _EXAMPLE_KINDS:
        raise NoGoldEvaluationError(f"example {index} has invalid kind")
    if kind == "response_draft_replacement":
        target = _mapping(item["target_delta"], f"example {index} target_delta")
        _exact_keys(
            target,
            frozenset(
                {
                    "op",
                    "region",
                    "text",
                    "char_count",
                    "complete_replacement",
                }
            ),
            f"example {index} target_delta",
        )
        if (
            target["op"] != "replace"
            or target["region"] != "response_draft"
            or target["complete_replacement"] is not True
        ):
            raise NoGoldEvaluationError(
                f"example {index} target delta contract mismatch"
            )
        target_text = _string(target["text"], f"example {index} target text")
        assert_supported_text(target_text)
        if not 1 <= len(target_text) <= MAX_TARGET_CHARS:
            raise NoGoldEvaluationError(
                f"example {index} target length is outside [1, 64]"
            )
        if type(target["char_count"]) is not int or target["char_count"] != len(
            target_text
        ):
            raise NoGoldEvaluationError(
                f"example {index} target char_count mismatch"
            )
        if item["structural_noop"] is not None:
            raise NoGoldEvaluationError(
                f"example {index} replacement carries structural_noop"
            )
        expected_action = "replace_response_draft"
    else:
        if item["target_delta"] is not None:
            raise NoGoldEvaluationError(
                f"example {index} structural no-op carries target delta"
            )
        noop = _mapping(
            item["structural_noop"],
            f"example {index} structural_noop",
        )
        _exact_keys(
            noop,
            frozenset({"op", "reason", "reason_char_count"}),
            f"example {index} structural_noop",
        )
        reason = _string(noop["reason"], f"example {index} no-op reason")
        if (
            noop["op"] != "no_op"
            or type(noop["reason_char_count"]) is not int
            or noop["reason_char_count"] != len(reason)
        ):
            raise NoGoldEvaluationError(
                f"example {index} structural no-op contract mismatch"
            )
        target_text = None
        expected_action = "no_op"

    hidden_text = target_text
    if hidden_text is not None:
        leaked_regions = [
            region for region, text in active_pairs if hidden_text in text
        ]
        if leaked_regions:
            raise NoGoldEvaluationError(
                f"example {index} gold appears in visible regions "
                f"{leaked_regions}"
            )

    visible_payload = {
        "schema": NO_GOLD_CASE_SCHEMA,
        "active_field": dict(active_pairs),
    }
    case = NoGoldEvaluationCase(
        case_id=canonical_sha256(visible_payload),
        active_field=active_pairs,
    )
    return _GoldCase(
        example_id=example_id,
        split=split,
        case=case,
        expected_action=expected_action,
        target_text=target_text,
    )


def evaluate_no_gold_examples(
    examples: Sequence[Mapping[str, Any]],
    predictor: Callable[[NoGoldEvaluationCase], NoGoldPrediction],
    *,
    splits: Sequence[str] = ("dev", "test"),
) -> NoGoldEvaluationReport:
    """Collect predictions from visible-only cases, then score against gold."""

    if not callable(predictor):
        raise TypeError("predictor must be callable")
    split_values = tuple(splits)
    if not split_values or any(type(value) is not str for value in split_values):
        raise NoGoldEvaluationError("splits must contain strings")
    if len(split_values) != len(set(split_values)):
        raise NoGoldEvaluationError("splits cannot contain duplicates")
    if any(value not in _SPLITS for value in split_values):
        raise NoGoldEvaluationError("splits contain an unknown split")
    selected_splits = frozenset(split_values)

    raw_examples = tuple(examples)
    if not raw_examples:
        raise NoGoldEvaluationError("example suite must not be empty")
    input_digest_before = canonical_sha256(
        {"examples": [dict(_mapping(item, "example")) for item in raw_examples]}
    )
    gold_cases = [
        _parse_example(item, index)
        for index, item in enumerate(raw_examples)
    ]
    selected = sorted(
        (item for item in gold_cases if item.split in selected_splits),
        key=lambda item: item.case.case_id,
    )
    if not selected:
        raise NoGoldEvaluationError("no examples matched the requested splits")
    case_ids = [item.case.case_id for item in selected]
    if len(case_ids) != len(set(case_ids)):
        raise NoGoldEvaluationError(
            "two examples have the same visible case; hidden labels are ambiguous"
        )

    visible_suite_sha256 = canonical_sha256(
        {"cases": [item.case.to_dict() for item in selected]}
    )
    gold_suite_sha256 = canonical_sha256(
        {
            "examples": [
                {
                    "example_id": item.example_id,
                    "split": item.split,
                    "case_id": item.case.case_id,
                    "expected_action": item.expected_action,
                    "target_text": item.target_text,
                }
                for item in selected
            ]
        }
    )

    # Collect the complete prediction set before consulting a target.
    predictions: dict[str, NoGoldPrediction] = {}
    for item in selected:
        prediction = predictor(item.case)
        if type(prediction) is not NoGoldPrediction:
            raise NoGoldEvaluationError(
                "predictor must return exact NoGoldPrediction objects"
            )
        if prediction.case_id != item.case.case_id:
            raise NoGoldEvaluationError(
                "prediction case_id does not match the visible case"
            )
        if prediction.case_id in predictions:
            raise NoGoldEvaluationError("duplicate prediction case_id")
        predictions[prediction.case_id] = prediction

    input_digest_after = canonical_sha256(
        {"examples": [dict(_mapping(item, "example")) for item in raw_examples]}
    )
    if input_digest_after != input_digest_before:
        raise NoGoldEvaluationError("predictor mutated the example suite")

    outcomes: list[NoGoldCaseOutcome] = []
    decision_correct_count = 0
    replacement_count = 0
    noop_count = 0
    exact_text_count = 0
    exact_length_count = 0
    nonempty_count = 0
    noop_correct_count = 0
    character_matches = 0
    character_denominator = 0
    replacement_predictions: list[str] = []
    predicted_characters: Counter[str] = Counter()
    predicted_bins: Counter[str] = Counter()
    target_bins: Counter[str] = Counter()

    for item in selected:
        prediction = predictions[item.case.case_id]
        decision_correct = prediction.action == item.expected_action
        decision_correct_count += int(decision_correct)
        predicted_text = (
            prediction.response_text
            if prediction.action == "replace_response_draft"
            else ""
        )
        assert predicted_text is not None
        predicted_characters.update(predicted_text)
        predicted_bins[_length_bin(len(predicted_text))] += 1

        if item.expected_action == "replace_response_draft":
            replacement_count += 1
            target_text = item.target_text
            assert target_text is not None
            target_bins[_length_bin(len(target_text))] += 1
            replacement_predictions.append(predicted_text)
            exact_text = (
                prediction.action == "replace_response_draft"
                and predicted_text == target_text
            )
            exact_length = (
                prediction.action == "replace_response_draft"
                and len(predicted_text) == len(target_text)
            )
            exact_text_count += int(exact_text)
            exact_length_count += int(exact_length)
            nonempty_count += int(bool(predicted_text))
            denominator = max(len(predicted_text), len(target_text))
            matches = sum(
                predicted_character == target_character
                for predicted_character, target_character in zip(
                    predicted_text,
                    target_text,
                )
            )
            character_matches += matches
            character_denominator += denominator
        else:
            noop_count += 1
            noop_correct_count += int(prediction.action == "no_op")
            exact_text = None
            exact_length = None
            matches = 0
            denominator = 0

        outcomes.append(
            NoGoldCaseOutcome(
                case_id=item.case.case_id,
                example_id=item.example_id,
                split=item.split,
                expected_action=item.expected_action,
                predicted_action=prediction.action,
                target_length=0 if item.target_text is None else len(item.target_text),
                predicted_length=len(predicted_text),
                decision_correct=decision_correct,
                exact_text=exact_text,
                exact_length=exact_length,
                positional_character_matches=matches,
                positional_character_denominator=denominator,
                prediction_id=prediction.prediction_id,
            )
        )

    total_predicted_characters = sum(predicted_characters.values())
    top_character_fraction = (
        1.0
        if total_predicted_characters == 0
        else max(predicted_characters.values()) / total_predicted_characters
    )
    distinct_response_rate = (
        0.0
        if replacement_count == 0
        else len(set(replacement_predictions)) / replacement_count
    )
    metrics: dict[str, int | float] = {
        "case_count": len(selected),
        "replacement_case_count": replacement_count,
        "structural_noop_case_count": noop_count,
        "decision_accuracy": _mean(decision_correct_count, len(selected)),
        "replacement_exact_text_accuracy": _mean(
            exact_text_count,
            replacement_count,
        ),
        "replacement_exact_length_accuracy": _mean(
            exact_length_count,
            replacement_count,
        ),
        "replacement_character_accuracy": _mean(
            character_matches,
            character_denominator,
        ),
        "answer_required_nonempty_rate": _mean(
            nonempty_count,
            replacement_count,
        ),
        "answer_required_empty_rate": 1.0
        - _mean(nonempty_count, replacement_count),
        "first_position_empty_rate": 1.0
        - _mean(nonempty_count, replacement_count),
        "structural_noop_accuracy": _mean(noop_correct_count, noop_count),
        "predicted_top_character_fraction": top_character_fraction,
        "supported_characters_used": len(predicted_characters),
        "distinct_response_rate": distinct_response_rate,
    }
    ordered_bins = ("empty", "1-16", "17-32", "33-48", "49-64")
    prediction_set_sha256 = canonical_sha256(
        {
            "predictions": [
                predictions[item.case.case_id].to_dict()
                for item in selected
            ]
        }
    )
    return NoGoldEvaluationReport(
        visible_suite_sha256=visible_suite_sha256,
        gold_suite_sha256=gold_suite_sha256,
        prediction_set_sha256=prediction_set_sha256,
        evaluated_splits=tuple(sorted(selected_splits)),
        metrics=tuple(sorted(metrics.items())),
        predicted_length_bins=tuple(
            (name, predicted_bins[name]) for name in ordered_bins
        ),
        target_length_bins=tuple(
            (name, target_bins[name]) for name in ordered_bins
        ),
        outcomes=tuple(outcomes),
    )


def load_exportable_conversational_examples(
    examples_jsonl: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Load one bounded JSONL artifact and optionally verify its manifest hash."""

    path = Path(examples_jsonl)
    if _path_mentions_d00(path):
        raise NoGoldEvaluationError("D:\\00 input is not exportable")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise NoGoldEvaluationError(f"cannot read examples JSONL: {path}") from exc
    if not payload or len(payload) > MAX_EXAMPLES_BYTES:
        raise NoGoldEvaluationError("examples JSONL is empty or exceeds size cap")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NoGoldEvaluationError("examples JSONL must be canonical UTF-8") from exc
    if text.startswith("\ufeff"):
        raise NoGoldEvaluationError("examples JSONL cannot contain a UTF-8 BOM")

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise NoGoldEvaluationError(
                f"examples JSONL line {line_number} is blank"
            )
        value = _parse_json_without_duplicate_keys(
            line,
            f"examples JSONL line {line_number}",
        )
        rows.append(_mapping(value, f"examples JSONL line {line_number}"))
    if not rows:
        raise NoGoldEvaluationError("examples JSONL contains no records")

    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        if _path_mentions_d00(manifest_file):
            raise NoGoldEvaluationError("D:\\00 manifest is not exportable")
        try:
            manifest_text = manifest_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise NoGoldEvaluationError("cannot parse curriculum manifest") from exc
        manifest = _parse_json_without_duplicate_keys(
            manifest_text,
            "curriculum manifest",
        )
        manifest = _mapping(manifest, "manifest")
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise NoGoldEvaluationError("manifest schema mismatch")
        artifacts = _mapping(manifest.get("artifacts"), "manifest artifacts")
        descriptor = _mapping(
            artifacts.get("examples_jsonl"),
            "manifest examples_jsonl",
        )
        if (
            type(descriptor.get("bytes")) is not int
            or descriptor["bytes"] != len(payload)
            or _sha256(
                descriptor.get("sha256"),
                "manifest examples_jsonl sha256",
            )
            != hashlib.sha256(payload).hexdigest().upper()
        ):
            raise NoGoldEvaluationError(
                "examples JSONL does not match curriculum manifest"
            )
    return tuple(rows)


def evaluate_no_gold_jsonl(
    examples_jsonl: str | Path,
    predictor: Callable[[NoGoldEvaluationCase], NoGoldPrediction],
    *,
    manifest_path: str | Path | None = None,
    splits: Sequence[str] = ("dev", "test"),
) -> NoGoldEvaluationReport:
    """Load a verified artifact and run the visible-only evaluation contract."""

    examples = load_exportable_conversational_examples(
        examples_jsonl,
        manifest_path=manifest_path,
    )
    return evaluate_no_gold_examples(examples, predictor, splits=splits)


__all__ = [
    "NO_GOLD_CASE_SCHEMA",
    "NO_GOLD_PREDICTION_SCHEMA",
    "NO_GOLD_REPORT_SCHEMA",
    "MAX_EXAMPLES_BYTES",
    "NoGoldEvaluationError",
    "NoGoldEvaluationCase",
    "NoGoldPrediction",
    "NoGoldCaseOutcome",
    "NoGoldEvaluationReport",
    "evaluate_no_gold_examples",
    "load_exportable_conversational_examples",
    "evaluate_no_gold_jsonl",
]
