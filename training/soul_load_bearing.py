"""Controlled causal evaluation for Axon's persisted soul state.

This module is deliberately additive: it does not change the core, checkpoint,
or mutable :mod:`cores.soul_v2` implementation.  It provides pure
counterfactual state construction, a generic responder adapter, serializable
per-core metrics, and a fail-closed behavioral gate.

The evaluator treats NLL values returned by the adapter as mean
negative-log-likelihood per output character.  That keeps the adapter usable
with the current char-slot core as well as deterministic test responders.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from cores.soul_v2 import SoulState, SoulV2Config


SOUL_LOAD_BEARING_SCHEMA = "axon_soul_load_bearing_eval_v1"
_METADATA_FIELDS = (
    "active",
    "tier",
    "category",
    "salience",
    "dormant_for",
    "tick_born",
)


class SoulCondition(str, Enum):
    CORRECT = "correct"
    ZERO = "zero"
    SWAPPED = "swapped"
    SHUFFLED = "shuffled"


@dataclass(frozen=True)
class SoulConditionStates:
    """Independent state clones for one controlled four-way comparison."""

    correct: SoulState
    zero: SoulState
    swapped: SoulState
    shuffled: SoulState

    def items(self) -> tuple[tuple[SoulCondition, SoulState], ...]:
        return (
            (SoulCondition.CORRECT, self.correct),
            (SoulCondition.ZERO, self.zero),
            (SoulCondition.SWAPPED, self.swapped),
            (SoulCondition.SHUFFLED, self.shuffled),
        )


@dataclass(frozen=True)
class SoulCausalCase:
    """One same-visible-input causal soul case.

    ``owner_target``, ``donor_target``, and ``abstention_target`` must be
    distinct.  The responder is never told which condition is being evaluated;
    it receives only ``visible_input`` and the applicable cloned state.
    """

    case_id: str
    visible_input: Any
    owner_state: SoulState
    donor_state: SoulState
    owner_target: str
    donor_target: str
    abstention_target: str = "unknown"
    shuffle_seed: int = 0
    # Supply this for non-serializable model inputs. JSON-compatible values and
    # tensors are hashed automatically.
    visible_input_sha256: str | None = None


@dataclass(frozen=True)
class SoulResponderOutput:
    """Normalized output from a model-specific evaluator adapter."""

    prediction: str
    target_nll: Mapping[str, float]


class SoulResponder(Protocol):
    def __call__(
        self,
        visible_input: Any,
        soul_state: SoulState,
    ) -> SoulResponderOutput:
        """Evaluate without mutating ``soul_state``."""


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    lower: float
    upper: float
    confidence: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SoulLoadBearingThresholds:
    """Locked thresholds for an individual core.

    Defaults match the proposed promotion-grade gate.  Small development
    suites should pass an explicit lower ``min_cases`` but must not serialize
    those relaxed thresholds as a promotion result.
    """

    min_cases: int = 256
    correct_exact_min: float = 0.90
    correct_char_min: float = 0.98
    zero_nll_gap_mean_min: float = 0.50
    swapped_nll_gap_mean_min: float = 0.30
    shuffled_nll_gap_mean_min: float = 0.30
    # This bound is exclusive: the default requires positive bootstrap
    # evidence rather than allowing a lower bound of exactly zero.
    nll_gap_ci_lower_exclusive_min: float = 0.0
    swapped_donor_follow_min: float = 0.80
    zero_abstention_exact_min: float = 0.90
    shuffled_abstention_exact_min: float = 0.90


@dataclass(frozen=True)
class SoulLoadBearingReport:
    """Checkpoint-serializable result for one core and one fixed suite."""

    core_id: str
    suite_sha256: str
    n_cases: int
    metrics: Mapping[str, Any]
    thresholds: SoulLoadBearingThresholds
    passed: bool
    violations: tuple[str, ...]
    schema: str = SOUL_LOAD_BEARING_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "core_id": self.core_id,
            "suite_sha256": self.suite_sha256,
            "n_cases": self.n_cases,
            "metrics": dict(self.metrics),
            "thresholds": asdict(self.thresholds),
            "passed": self.passed,
            "violations": list(self.violations),
        }


def _tensor_sha256(hasher: Any, label: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    hasher.update(label.encode("utf-8"))
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(json.dumps(list(tensor.shape)).encode("ascii"))
    # Viewing as bytes supports every torch dtype, including bfloat16.
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())


def soul_state_sha256(state: SoulState) -> str:
    """Hash tensor content, every persisted metadata field, tick, and config."""

    validate_soul_state(state)
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            state.cfg.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _tensor_sha256(hasher, "tensor", state.tensor)
    for field_name in _METADATA_FIELDS:
        _tensor_sha256(hasher, field_name, getattr(state, field_name))
    hasher.update(f"current_tick:{int(state.current_tick)}".encode("ascii"))
    return hasher.hexdigest()


def validate_soul_state(state: SoulState) -> None:
    """Reject malformed state before it can enter a causal comparison."""

    if not isinstance(state, SoulState):
        raise TypeError("state must be a SoulState")
    if state.tensor.ndim != 2:
        raise ValueError("soul tensor must have shape (rows, d_model)")
    expected_shape = (state.cfg.total_max_rows, state.cfg.d_model)
    if tuple(state.tensor.shape) != expected_shape:
        raise ValueError(
            f"soul tensor shape {tuple(state.tensor.shape)} does not match config "
            f"{expected_shape}"
        )
    if not torch.isfinite(state.tensor).all().item():
        raise ValueError("soul tensor contains non-finite values")

    rows = expected_shape[0]
    for field_name in _METADATA_FIELDS:
        value = getattr(state, field_name, None)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows,):
            raise ValueError(f"soul metadata {field_name} must have shape ({rows},)")
    if state.active.dtype != torch.bool:
        raise ValueError("soul active metadata must be bool")
    if not torch.isfinite(state.salience).all().item():
        raise ValueError("soul salience contains non-finite values")
    if int(state.current_tick) < 0:
        raise ValueError("soul current_tick must be non-negative")

    n_tiers = len(state.cfg.tiers)
    if n_tiers <= 0:
        raise ValueError("soul config must declare at least one tier")
    if ((state.tier < 0) | (state.tier >= n_tiers)).any().item():
        raise ValueError("soul tier metadata contains an out-of-range index")
    n_categories = len(state.cfg.categories)
    if ((state.category < -1) | (state.category >= n_categories)).any().item():
        raise ValueError("soul category metadata contains an out-of-range index")

    expected_tier = torch.empty_like(state.tier)
    for tier_idx in range(n_tiers):
        expected_tier[state.cfg.tier_slice(tier_idx)] = tier_idx
    if not torch.equal(state.tier.cpu(), expected_tier.cpu()):
        raise ValueError("soul tier metadata does not match config tier slices")


def clone_soul_state(state: SoulState) -> SoulState:
    """Deep-clone all persisted state without using mutable reset helpers."""

    validate_soul_state(state)
    cfg = SoulV2Config.from_dict(state.cfg.to_dict())
    clone = SoulState(cfg, state.tensor.device, state.tensor.dtype)
    clone.tensor = state.tensor.detach().clone()
    for field_name in _METADATA_FIELDS:
        setattr(clone, field_name, getattr(state, field_name).detach().clone())
    clone.current_tick = int(state.current_tick)
    return clone


def assert_swap_compatible(owner: SoulState, donor: SoulState) -> None:
    """Require a donor from the exact same per-core soul layout."""

    validate_soul_state(owner)
    validate_soul_state(donor)
    if tuple(owner.tensor.shape) != tuple(donor.tensor.shape):
        raise ValueError("owner and donor soul tensor shapes are incompatible")
    if owner.cfg.to_dict() != donor.cfg.to_dict():
        raise ValueError("owner and donor soul configs are incompatible")


def _shuffle_active_content_within_groups(
    state: SoulState,
    *,
    seed: int,
) -> SoulState:
    shuffled = clone_soul_state(state)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))

    active_indices = state.active.nonzero(as_tuple=True)[0].cpu()
    groups: dict[tuple[int, int], list[int]] = {}
    for row_idx in active_indices.tolist():
        key = (int(state.tier[row_idx]), int(state.category[row_idx]))
        groups.setdefault(key, []).append(row_idx)

    for key in sorted(groups):
        indices = groups[key]
        if len(indices) < 2:
            continue
        # A non-zero rotation is a deterministic derangement.  Unlike an
        # unconstrained randperm, it cannot accidentally leave content in its
        # original row.
        shift = int(
            torch.randint(
                low=1,
                high=len(indices),
                size=(1,),
                generator=generator,
            ).item()
        )
        source_indices = indices[-shift:] + indices[:-shift]
        destination = torch.tensor(indices, device=state.tensor.device)
        source = torch.tensor(source_indices, device=state.tensor.device)
        shuffled.tensor[destination] = state.tensor[source].detach().clone()

    return shuffled


def build_causal_soul_conditions(
    owner: SoulState,
    donor: SoulState,
    *,
    shuffle_seed: int,
) -> SoulConditionStates:
    """Build four independent conditions without mutating either input.

    ``zero`` changes only tensor content. ``swapped`` is an exact donor clone.
    ``shuffled`` changes only active tensor content and never moves content
    across a ``(tier, category)`` boundary.
    """

    assert_swap_compatible(owner, donor)
    owner_hash = soul_state_sha256(owner)
    donor_hash = soul_state_sha256(donor)

    correct = clone_soul_state(owner)
    zero = clone_soul_state(owner)
    zero.tensor.zero_()
    swapped = clone_soul_state(donor)
    shuffled = _shuffle_active_content_within_groups(owner, seed=shuffle_seed)

    if soul_state_sha256(owner) != owner_hash:
        raise RuntimeError("causal condition construction mutated the owner state")
    if soul_state_sha256(donor) != donor_hash:
        raise RuntimeError("causal condition construction mutated the donor state")
    return SoulConditionStates(correct, zero, swapped, shuffled)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a deterministic percentile bootstrap interval for a mean."""

    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("bootstrap requires at least one value")
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("bootstrap values must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be between zero and one")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    observed_mean = sum(numeric) / len(numeric)
    if len(numeric) == 1:
        return BootstrapInterval(
            observed_mean,
            observed_mean,
            observed_mean,
            confidence,
            samples,
        )

    rng = random.Random(int(seed))
    n = len(numeric)
    bootstrap_means = sorted(
        sum(numeric[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        observed_mean,
        _quantile(bootstrap_means, tail),
        _quantile(bootstrap_means, 1.0 - tail),
        confidence,
        samples,
    )


def _char_accuracy(prediction: str, target: str) -> float:
    width = max(len(prediction), len(target))
    if width == 0:
        return 1.0
    matches = sum(
        int(index < len(prediction) and index < len(target)
            and prediction[index] == target[index])
        for index in range(width)
    )
    return matches / width


def _mean_or_nan(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _condition_metric(
    outputs: Sequence[tuple[SoulResponderOutput, str, float]],
) -> dict[str, float | int]:
    return {
        "n": len(outputs),
        "exact": _mean_or_nan(
            [float(output.prediction == expected) for output, expected, _ in outputs]
        ),
        "char_accuracy": _mean_or_nan(
            [_char_accuracy(output.prediction, expected)
             for output, expected, _ in outputs]
        ),
        "expected_nll": _mean_or_nan(
            [expected_nll for _, _, expected_nll in outputs]
        ),
    }


def _case_seed(global_seed: int, case: SoulCausalCase) -> int:
    payload = f"{int(global_seed)}:{case.case_id}:{int(case.shuffle_seed)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _update_value_hash(hasher: Any, value: Any) -> None:
    """Hash common immutable/model-input containers without pickle."""

    if value is None:
        hasher.update(b"none")
    elif isinstance(value, bool):
        hasher.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        hasher.update(f"int:{value}".encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("visible input contains a non-finite float")
        hasher.update(f"float:{value.hex()}".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(f"str:{len(encoded)}:".encode("ascii"))
        hasher.update(encoded)
    elif isinstance(value, bytes):
        hasher.update(f"bytes:{len(value)}:".encode("ascii"))
        hasher.update(value)
    elif isinstance(value, torch.Tensor):
        _tensor_sha256(hasher, "visible_tensor", value)
    elif isinstance(value, Mapping):
        hasher.update(b"mapping:{")
        if not all(isinstance(key, str) for key in value):
            raise TypeError("visible input mapping keys must be strings")
        for key in sorted(value):
            _update_value_hash(hasher, key)
            _update_value_hash(hasher, value[key])
        hasher.update(b"}")
    elif isinstance(value, (list, tuple)):
        hasher.update(f"{type(value).__name__}:[".encode("ascii"))
        for item in value:
            _update_value_hash(hasher, item)
        hasher.update(b"]")
    else:
        raise TypeError(
            "visible input is not canonically hashable; provide visible_input_sha256"
        )


def _visible_input_sha256(case: SoulCausalCase) -> str:
    if case.visible_input_sha256 is not None:
        if not _valid_sha256(case.visible_input_sha256):
            raise ValueError("visible_input_sha256 must be a hexadecimal SHA-256")
        return case.visible_input_sha256.lower()
    hasher = hashlib.sha256()
    _update_value_hash(hasher, case.visible_input)
    return hasher.hexdigest()


def _suite_sha256(cases: Sequence[SoulCausalCase], global_seed: int) -> str:
    payload: list[dict[str, Any]] = []
    for case in cases:
        payload.append(
            {
                "case_id": case.case_id,
                "owner_target": case.owner_target,
                "donor_target": case.donor_target,
                "abstention_target": case.abstention_target,
                "owner_state_sha256": soul_state_sha256(case.owner_state),
                "donor_state_sha256": soul_state_sha256(case.donor_state),
                "visible_input_sha256": _visible_input_sha256(case),
                "shuffle_seed": _case_seed(global_seed, case),
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _valid_response(
    response: Any,
    *,
    case: SoulCausalCase,
    condition: SoulCondition,
    violations: list[str],
) -> SoulResponderOutput | None:
    prefix = f"{case.case_id}.{condition.value}"
    if not isinstance(response, SoulResponderOutput):
        violations.append(f"{prefix}: responder returned an invalid output")
        return None
    if not isinstance(response.prediction, str):
        violations.append(f"{prefix}: prediction is not a string")
        return None
    if not isinstance(response.target_nll, Mapping):
        violations.append(f"{prefix}: target_nll is missing or invalid")
        return None
    return response


def _read_nll(
    response: SoulResponderOutput,
    target: str,
    *,
    prefix: str,
    violations: list[str],
) -> float:
    if target not in response.target_nll:
        violations.append(f"{prefix}: missing NLL for target {target!r}")
        return math.nan
    try:
        value = float(response.target_nll[target])
    except (TypeError, ValueError):
        violations.append(f"{prefix}: non-numeric NLL for target {target!r}")
        return math.nan
    if not math.isfinite(value) or value < 0.0:
        violations.append(f"{prefix}: invalid NLL for target {target!r}")
        return math.nan
    return value


def _append_minimum_violation(
    violations: list[str],
    *,
    label: str,
    observed: Any,
    minimum: Any,
) -> None:
    try:
        observed_float = float(observed)
        minimum_float = float(minimum)
    except (TypeError, ValueError):
        violations.append(f"{label}: metric or threshold is non-numeric")
        return
    if not math.isfinite(observed_float) or not math.isfinite(minimum_float):
        violations.append(f"{label}: metric or threshold is non-finite")
    elif observed_float < minimum_float:
        violations.append(
            f"{label}={observed_float:.9g} below minimum {minimum_float:.9g}"
        )


def _append_exclusive_minimum_violation(
    violations: list[str],
    *,
    label: str,
    observed: Any,
    minimum: Any,
) -> None:
    try:
        observed_float = float(observed)
        minimum_float = float(minimum)
    except (TypeError, ValueError):
        violations.append(f"{label}: metric or threshold is non-numeric")
        return
    if not math.isfinite(observed_float) or not math.isfinite(minimum_float):
        violations.append(f"{label}: metric or threshold is non-finite")
    elif observed_float <= minimum_float:
        violations.append(
            f"{label}={observed_float:.9g} must be greater than "
            f"{minimum_float:.9g}"
        )


def _apply_thresholds(
    metrics: Mapping[str, Any],
    thresholds: SoulLoadBearingThresholds,
    *,
    n_cases: int,
    violations: list[str],
) -> None:
    if not isinstance(thresholds.min_cases, int) or thresholds.min_cases <= 0:
        violations.append("threshold min_cases must be a positive integer")
    elif n_cases < thresholds.min_cases:
        violations.append(
            f"n_cases={n_cases} below minimum {thresholds.min_cases}"
        )

    conditions = metrics.get("conditions", {})
    correct = conditions.get(SoulCondition.CORRECT.value, {})
    _append_minimum_violation(
        violations,
        label="correct.exact",
        observed=correct.get("exact"),
        minimum=thresholds.correct_exact_min,
    )
    _append_minimum_violation(
        violations,
        label="correct.char_accuracy",
        observed=correct.get("char_accuracy"),
        minimum=thresholds.correct_char_min,
    )
    _append_minimum_violation(
        violations,
        label="swapped.donor_follow",
        observed=metrics.get("swapped_donor_follow"),
        minimum=thresholds.swapped_donor_follow_min,
    )
    abstention = metrics.get("abstention_exact", {})
    _append_minimum_violation(
        violations,
        label="zero.abstention_exact",
        observed=abstention.get("zero"),
        minimum=thresholds.zero_abstention_exact_min,
    )
    _append_minimum_violation(
        violations,
        label="shuffled.abstention_exact",
        observed=abstention.get("shuffled"),
        minimum=thresholds.shuffled_abstention_exact_min,
    )

    gap_minimums = {
        SoulCondition.ZERO.value: thresholds.zero_nll_gap_mean_min,
        SoulCondition.SWAPPED.value: thresholds.swapped_nll_gap_mean_min,
        SoulCondition.SHUFFLED.value: thresholds.shuffled_nll_gap_mean_min,
    }
    for counterfactual, gap_minimum in gap_minimums.items():
        gap = metrics.get("owner_nll_gaps", {}).get(counterfactual, {})
        _append_minimum_violation(
            violations,
            label=f"{counterfactual}.owner_nll_gap.mean",
            observed=gap.get("mean"),
            minimum=gap_minimum,
        )
        _append_exclusive_minimum_violation(
            violations,
            label=f"{counterfactual}.owner_nll_gap.ci_lower",
            observed=gap.get("lower"),
            minimum=thresholds.nll_gap_ci_lower_exclusive_min,
        )


def evaluate_soul_load_bearing(
    *,
    core_id: str,
    cases: Sequence[SoulCausalCase],
    responder: SoulResponder | Callable[[Any, SoulState], SoulResponderOutput],
    thresholds: SoulLoadBearingThresholds | None = None,
    bootstrap_samples: int = 2_000,
    bootstrap_confidence: float = 0.95,
    seed: int = 0,
) -> SoulLoadBearingReport:
    """Evaluate one core against correct/zero/swapped/shuffled soul states.

    State mutation, malformed cases, missing outputs/NLL values, non-finite
    metrics, insufficient cases, or any unmet threshold all fail closed.
    Exceptions from a responder are captured as gate violations so an
    incomplete evaluation cannot be mistaken for a pass.
    """

    locked = thresholds or SoulLoadBearingThresholds()
    case_list = list(cases)
    violations: list[str] = []
    if not isinstance(core_id, str) or not core_id.strip():
        violations.append("core_id is missing or invalid")

    case_ids = [case.case_id for case in case_list]
    if len(set(case_ids)) != len(case_ids):
        violations.append("case_id values must be unique")

    try:
        suite_sha256 = _suite_sha256(case_list, seed)
    except Exception as exc:
        violations.append(f"fixed suite is invalid: {type(exc).__name__}: {exc}")
        suite_sha256 = "0" * 64

    condition_outputs: dict[
        SoulCondition,
        list[tuple[SoulResponderOutput, str, float]],
    ] = {condition: [] for condition in SoulCondition}
    owner_nll_by_condition: dict[SoulCondition, list[float]] = {
        condition: [] for condition in SoulCondition
    }
    immutable = True

    for case in case_list:
        if not isinstance(case.case_id, str) or not case.case_id:
            violations.append("case_id is missing or invalid")
            continue
        targets = (
            case.owner_target,
            case.donor_target,
            case.abstention_target,
        )
        if not all(isinstance(target, str) and target for target in targets):
            violations.append(f"{case.case_id}: targets must be non-empty strings")
            continue
        if len(set(targets)) != 3:
            violations.append(f"{case.case_id}: causal targets must be distinct")
            continue

        try:
            owner_before = soul_state_sha256(case.owner_state)
            donor_before = soul_state_sha256(case.donor_state)
            conditions = build_causal_soul_conditions(
                case.owner_state,
                case.donor_state,
                shuffle_seed=_case_seed(seed, case),
            )
        except Exception as exc:
            violations.append(
                f"{case.case_id}: invalid causal states: {type(exc).__name__}: {exc}"
            )
            continue

        condition_hashes = {
            condition: soul_state_sha256(state)
            for condition, state in conditions.items()
        }
        for counterfactual in (
            SoulCondition.ZERO,
            SoulCondition.SWAPPED,
            SoulCondition.SHUFFLED,
        ):
            if condition_hashes[counterfactual] == condition_hashes[SoulCondition.CORRECT]:
                violations.append(
                    f"{case.case_id}: {counterfactual.value} state is identical "
                    "to correct state"
                )

        expected_target = {
            SoulCondition.CORRECT: case.owner_target,
            SoulCondition.ZERO: case.abstention_target,
            SoulCondition.SWAPPED: case.donor_target,
            SoulCondition.SHUFFLED: case.abstention_target,
        }

        for condition, condition_state in conditions.items():
            before = soul_state_sha256(condition_state)
            try:
                condition_input = copy.deepcopy(case.visible_input)
            except Exception as exc:
                violations.append(
                    f"{case.case_id}.{condition.value}: visible input clone failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            try:
                raw_response = responder(condition_input, condition_state)
            except Exception as exc:
                violations.append(
                    f"{case.case_id}.{condition.value}: responder failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                raw_response = None
            try:
                after = soul_state_sha256(condition_state)
            except Exception as exc:
                after = None
                immutable = False
                violations.append(
                    f"{case.case_id}.{condition.value}: responder left invalid soul "
                    f"state: {type(exc).__name__}: {exc}"
                )
            if after != before:
                immutable = False
                violations.append(
                    f"{case.case_id}.{condition.value}: responder mutated soul state"
                )

            response = _valid_response(
                raw_response,
                case=case,
                condition=condition,
                violations=violations,
            )
            if response is None:
                continue
            prefix = f"{case.case_id}.{condition.value}"
            expected_nll = _read_nll(
                response,
                expected_target[condition],
                prefix=prefix,
                violations=violations,
            )
            owner_nll = _read_nll(
                response,
                case.owner_target,
                prefix=prefix,
                violations=violations,
            )
            condition_outputs[condition].append(
                (response, expected_target[condition], expected_nll)
            )
            owner_nll_by_condition[condition].append(owner_nll)

        try:
            owner_after = soul_state_sha256(case.owner_state)
        except Exception as exc:
            owner_after = None
            violations.append(
                f"{case.case_id}: owner state became invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        if owner_after != owner_before:
            immutable = False
            violations.append(f"{case.case_id}: owner state was mutated")
        try:
            donor_after = soul_state_sha256(case.donor_state)
        except Exception as exc:
            donor_after = None
            violations.append(
                f"{case.case_id}: donor state became invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        if donor_after != donor_before:
            immutable = False
            violations.append(f"{case.case_id}: donor state was mutated")

    conditions_metrics = {
        condition.value: _condition_metric(condition_outputs[condition])
        for condition in SoulCondition
    }
    for condition in SoulCondition:
        conditions_metrics[condition.value]["owner_target_nll"] = _mean_or_nan(
            owner_nll_by_condition[condition]
        )

    owner_nll_gaps: dict[str, dict[str, float | int]] = {}
    correct_nll = owner_nll_by_condition[SoulCondition.CORRECT]
    for index, condition in enumerate(
        (SoulCondition.ZERO, SoulCondition.SWAPPED, SoulCondition.SHUFFLED),
        start=1,
    ):
        counterfactual_nll = owner_nll_by_condition[condition]
        if len(correct_nll) != len(case_list) or len(counterfactual_nll) != len(case_list):
            interval = BootstrapInterval(
                math.nan,
                math.nan,
                math.nan,
                bootstrap_confidence,
                bootstrap_samples,
            )
            violations.append(
                f"{condition.value}: incomplete owner NLL observations"
            )
        else:
            gaps = [
                counterfactual - correct
                for correct, counterfactual in zip(correct_nll, counterfactual_nll)
            ]
            try:
                interval = bootstrap_mean_ci(
                    gaps,
                    confidence=bootstrap_confidence,
                    samples=bootstrap_samples,
                    seed=int(seed) + index,
                )
            except ValueError as exc:
                violations.append(f"{condition.value}: bootstrap failed: {exc}")
                interval = BootstrapInterval(
                    math.nan,
                    math.nan,
                    math.nan,
                    bootstrap_confidence,
                    bootstrap_samples,
                )
        owner_nll_gaps[condition.value] = interval.to_dict()

    swapped_outputs = condition_outputs[SoulCondition.SWAPPED]
    zero_outputs = condition_outputs[SoulCondition.ZERO]
    shuffled_outputs = condition_outputs[SoulCondition.SHUFFLED]
    swapped_donor_follow = _mean_or_nan(
        [
            float(output.prediction == expected)
            for output, expected, _ in swapped_outputs
        ]
    )
    zero_abstention = _mean_or_nan(
        [float(output.prediction == expected) for output, expected, _ in zero_outputs]
    )
    shuffled_abstention = _mean_or_nan(
        [
            float(output.prediction == expected)
            for output, expected, _ in shuffled_outputs
        ]
    )
    all_abstention = list(zero_outputs) + list(shuffled_outputs)
    combined_abstention = _mean_or_nan(
        [
            float(output.prediction == expected)
            for output, expected, _ in all_abstention
        ]
    )

    metrics: dict[str, Any] = {
        "conditions": conditions_metrics,
        "owner_nll_gaps": owner_nll_gaps,
        "swapped_donor_follow": swapped_donor_follow,
        "abstention_exact": {
            "zero": zero_abstention,
            "shuffled": shuffled_abstention,
            "combined": combined_abstention,
        },
        "immutable": immutable,
    }
    if not immutable:
        violations.append("causal evaluation was not immutable")
    _apply_thresholds(
        metrics,
        locked,
        n_cases=len(case_list),
        violations=violations,
    )

    # Preserve stable ordering while avoiding duplicate cascading violations.
    unique_violations = tuple(dict.fromkeys(violations))
    return SoulLoadBearingReport(
        core_id=core_id,
        suite_sha256=suite_sha256,
        n_cases=len(case_list),
        metrics=metrics,
        thresholds=locked,
        passed=not unique_violations,
        violations=unique_violations,
    )


def counterfactual_margin_loss(
    correct_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor | Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    margin: float = 0.20,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Penalize insufficient target-log-prob advantage over counterfactuals.

    The loss is ``relu(margin - (logP_correct - max(logP_counterfactual)))``
    at each non-ignored target position.  The helper changes no model or
    checkpoint shapes and is fully differentiable through the supplied logits.
    """

    if correct_logits.ndim < 2:
        raise ValueError("correct_logits must end with a vocabulary dimension")
    if tuple(targets.shape) != tuple(correct_logits.shape[:-1]):
        raise ValueError("targets shape must equal correct_logits.shape[:-1]")
    if not math.isfinite(float(margin)) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")

    if isinstance(counterfactual_logits, torch.Tensor):
        if tuple(counterfactual_logits.shape) == tuple(correct_logits.shape):
            counterfactuals = [counterfactual_logits]
        elif (
            counterfactual_logits.ndim == correct_logits.ndim + 1
            and tuple(counterfactual_logits.shape[1:])
            == tuple(correct_logits.shape)
        ):
            counterfactuals = list(counterfactual_logits.unbind(0))
        else:
            raise ValueError("counterfactual logits have an incompatible shape")
    else:
        counterfactuals = list(counterfactual_logits)
    if not counterfactuals:
        raise ValueError("at least one counterfactual logits tensor is required")
    if any(tuple(logits.shape) != tuple(correct_logits.shape)
           for logits in counterfactuals):
        raise ValueError("all counterfactual logits must match correct_logits")

    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0).long().unsqueeze(-1)
    correct_log_probability = F.log_softmax(correct_logits, dim=-1).gather(
        -1, safe_targets
    ).squeeze(-1)
    counterfactual_log_probability = torch.stack(
        [
            F.log_softmax(logits, dim=-1).gather(-1, safe_targets).squeeze(-1)
            for logits in counterfactuals
        ],
        dim=0,
    ).amax(dim=0)
    per_position = F.relu(
        float(margin)
        - (correct_log_probability - counterfactual_log_probability)
    )
    per_position = per_position.masked_fill(~valid, 0.0)

    if reduction == "none":
        return per_position
    if reduction == "sum":
        return per_position.sum()
    if valid.any().item():
        return per_position.sum() / valid.sum()
    return correct_logits.sum() * 0.0
