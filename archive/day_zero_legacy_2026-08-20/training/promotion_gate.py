"""Deterministic, fail-closed evaluation of Axon gate contracts.

The legacy :func:`evaluate_promotion_gate` API is intentionally kept small and
backwards compatible.  Persisted continuation pilots use the stricter
:func:`evaluate_pilot_gate` API: it proves that the source and final metrics
were produced by the same fixed suite, evaluates the final absolute
continuation thresholds, and never implies that a pilot is promotable unless
the manifest explicitly authorizes a separate promotion gate.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


EXACT_V3_PILOT_METRICS: dict[str, dict[str, dict[str, float]]] = {
    "copy": {
        "exact_fill": {"min": 0.95},
        "char_acc": {"min": 0.99},
    },
    "partial": {
        "suffix_exact": {"min": 0.20},
    },
    "blank": {
        "char_acc": {"min": 0.27},
        "suffix_char_acc": {"min": 0.20},
        "pred_top_frac": {"max": 0.75},
        "pred_unique": {"min": 30.0},
        "collapse_var": {"min": 1e-6},
    },
}


def exact_v3_pilot_gate(
    *,
    suite_sha256: str,
    source_step: int = 250_000,
    target_step: int = 252_000,
    promotion_authorized: bool = False,
    promotion_metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the canonical exact-v3 250k->252k pilot gate.

    ``promotion_authorized`` defaults to false on purpose.  Passing the pilot
    continuation thresholds only authorizes another reviewed leg.  A caller
    that wants promotion must opt in and persist a separate, non-empty set of
    promotion thresholds.
    """

    if not _valid_sha256(suite_sha256):
        raise ValueError("suite_sha256 must be a 64-character hexadecimal SHA-256")
    if int(source_step) < 0 or int(target_step) <= int(source_step):
        raise ValueError("target_step must be greater than a non-negative source_step")
    if promotion_authorized and (
        not isinstance(promotion_metrics, Mapping) or not promotion_metrics
    ):
        raise ValueError("promotion_authorized requires promotion_metrics")
    if isinstance(promotion_metrics, Mapping) and any(
        not isinstance(mode_metrics, Mapping)
        for mode_metrics in promotion_metrics.values()
    ):
        raise ValueError("promotion_metrics modes must contain metric mappings")

    # Construct fresh nested dictionaries so callers cannot mutate the module
    # constant while serializing or enriching a bundle manifest.
    continuation_metrics = {
        mode: {metric: dict(limits) for metric, limits in metrics.items()}
        for mode, metrics in EXACT_V3_PILOT_METRICS.items()
    }
    return {
        "schema": "axon_charslot_pilot_gate_v1",
        "source_step": int(source_step),
        "target_step": int(target_step),
        "fixed_suite_sha256": suite_sha256.lower(),
        "continuation_metrics": continuation_metrics,
        "promotion_authorized": bool(promotion_authorized),
        "promotion_metrics": (
            {
                str(mode): {
                    str(metric): dict(limits)
                    for metric, limits in mode_metrics.items()
                }
                for mode, mode_metrics in promotion_metrics.items()
            }
            if isinstance(promotion_metrics, Mapping)
            else None
        ),
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def evaluate_promotion_gate(
    metrics_by_mode: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable pass/fail result for a gate manifest.

    Gate conditions are shaped as ``mode -> metric -> {min|max: value}``.
    Unknown modes, metrics, operators, and non-numeric values fail closed.
    """

    if not isinstance(metrics_by_mode, Mapping):
        return {"passed": False, "violations": ["evaluation metrics are missing or invalid"]}
    if not isinstance(gate, Mapping):
        return {"passed": False, "violations": ["promotion gate is missing or invalid"]}

    violations: list[str] = []
    conditions = gate.get("metrics")
    if not isinstance(conditions, Mapping) or not conditions:
        return {"passed": False, "violations": ["promotion gate has no metric conditions"]}

    for mode, mode_conditions in conditions.items():
        observed_mode = metrics_by_mode.get(str(mode))
        if not isinstance(observed_mode, Mapping):
            violations.append(f"missing evaluation mode: {mode}")
            continue
        if not isinstance(mode_conditions, Mapping):
            violations.append(f"invalid conditions for mode: {mode}")
            continue

        for metric, limits in mode_conditions.items():
            if metric not in observed_mode:
                violations.append(f"missing metric: {mode}.{metric}")
                continue
            if not isinstance(limits, Mapping) or not limits:
                violations.append(f"invalid limits: {mode}.{metric}")
                continue
            try:
                observed = float(observed_mode[metric])
            except (TypeError, ValueError):
                violations.append(f"non-numeric metric: {mode}.{metric}")
                continue
            if not math.isfinite(observed):
                violations.append(f"non-finite metric: {mode}.{metric}")
                continue

            for operator, threshold_value in limits.items():
                try:
                    threshold = float(threshold_value)
                except (TypeError, ValueError):
                    violations.append(f"non-numeric threshold: {mode}.{metric}.{operator}")
                    continue
                if not math.isfinite(threshold):
                    violations.append(f"non-finite threshold: {mode}.{metric}.{operator}")
                    continue
                if operator == "min":
                    if observed < threshold:
                        violations.append(
                            f"{mode}.{metric}={observed:.9g} below minimum {threshold:.9g}"
                        )
                elif operator == "max":
                    if observed > threshold:
                        violations.append(
                            f"{mode}.{metric}={observed:.9g} above maximum {threshold:.9g}"
                        )
                else:
                    violations.append(f"unknown gate operator: {mode}.{metric}.{operator}")

    return {"passed": not violations, "violations": violations}


def _validate_fixed_eval(
    evaluation: Mapping[str, Any],
    *,
    label: str,
    expected_step: int,
    expected_suite_sha256: str,
    required_metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Mapping[str, Any]], list[str]]:
    """Validate one fixed-suite evaluation envelope and return its metrics."""

    violations: list[str] = []
    if not isinstance(evaluation, Mapping):
        return {}, [f"{label} evaluation is missing or invalid"]
    if evaluation.get("schema") != "axon_charslot_fixed_eval_v1":
        violations.append(f"{label} evaluation schema is missing or invalid")
    try:
        observed_step = int(evaluation.get("step", -1))
    except (TypeError, ValueError):
        observed_step = -1
    if observed_step != int(expected_step):
        violations.append(f"{label} evaluation step does not equal {expected_step}")

    suite_sha256 = evaluation.get("fixed_suite_sha256")
    if not _valid_sha256(suite_sha256):
        violations.append(f"{label} fixed-suite SHA-256 is missing or invalid")
    elif suite_sha256.lower() != expected_suite_sha256.lower():
        violations.append(f"{label} fixed-suite SHA-256 does not match the gate")

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        violations.append(f"{label} metrics are missing or invalid")
        return {}, violations

    # Source metrics are evidence, not merely a fingerprint.  Require every
    # metric used by either decision and require finite numeric values so a
    # corrupt baseline cannot silently pass through to a report.
    for mode, mode_requirements in required_metrics.items():
        observed_mode = metrics.get(str(mode))
        if not isinstance(observed_mode, Mapping):
            violations.append(f"{label} missing evaluation mode: {mode}")
            continue
        if not isinstance(mode_requirements, Mapping):
            violations.append(f"invalid required metrics for mode: {mode}")
            continue
        for metric in mode_requirements:
            if metric not in observed_mode:
                violations.append(f"{label} missing metric: {mode}.{metric}")
                continue
            try:
                observed = float(observed_mode[metric])
            except (TypeError, ValueError):
                violations.append(f"{label} non-numeric metric: {mode}.{metric}")
                continue
            if not math.isfinite(observed):
                violations.append(f"{label} non-finite metric: {mode}.{metric}")

    return metrics, violations


def evaluate_pilot_gate(
    source_evaluation: Mapping[str, Any],
    final_evaluation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a fixed-suite pilot without conflating continuation/promotion.

    The source and final envelopes must use ``axon_charslot_fixed_eval_v1`` and
    contain ``step``, ``fixed_suite_sha256``, and ``metrics``.  Missing,
    malformed, or non-finite values fail closed.  ``passed`` is retained as an
    alias for ``continuable`` for callers that treat this as a pilot gate; the
    independent ``promotable`` field must be used for checkpoint promotion.
    """

    if not isinstance(gate, Mapping):
        return {
            "passed": False,
            "continuable": False,
            "promotable": False,
            "violations": ["pilot gate is missing or invalid"],
            "contract_violations": ["pilot gate is missing or invalid"],
            "continuation_violations": [],
            "promotion_violations": ["pilot gate does not authorize promotion"],
            "fixed_suite_sha256": None,
            "source_step": -1,
            "target_step": -1,
        }

    contract_violations: list[str] = []
    if gate.get("schema") != "axon_charslot_pilot_gate_v1":
        contract_violations.append("pilot gate schema is missing or invalid")
    try:
        source_step = int(gate.get("source_step", -1))
        target_step = int(gate.get("target_step", -1))
    except (TypeError, ValueError):
        source_step = target_step = -1
    if source_step < 0:
        contract_violations.append("pilot gate source_step is missing or invalid")
    if target_step <= source_step:
        contract_violations.append("pilot gate target_step must be greater than source_step")

    suite_sha256 = gate.get("fixed_suite_sha256")
    if not _valid_sha256(suite_sha256):
        contract_violations.append("pilot gate fixed-suite SHA-256 is missing or invalid")
        suite_sha256 = ""

    continuation_metrics = gate.get("continuation_metrics")
    if not isinstance(continuation_metrics, Mapping) or not continuation_metrics:
        contract_violations.append("pilot gate has no continuation metric conditions")
        continuation_metrics = {}

    promotion_authorized = gate.get("promotion_authorized")
    if not isinstance(promotion_authorized, bool):
        contract_violations.append("pilot gate promotion_authorized must be boolean")
        promotion_authorized = False
    promotion_metrics = gate.get("promotion_metrics")
    if promotion_authorized and (
        not isinstance(promotion_metrics, Mapping) or not promotion_metrics
    ):
        contract_violations.append("authorized promotion has no metric conditions")
        promotion_metrics = {}

    required_metrics: dict[str, dict[str, Any]] = {}
    for condition_set in (continuation_metrics, promotion_metrics or {}):
        if not isinstance(condition_set, Mapping):
            continue
        for mode, mode_metrics in condition_set.items():
            if not isinstance(mode_metrics, Mapping):
                continue
            required_metrics.setdefault(str(mode), {}).update(
                {str(metric): limits for metric, limits in mode_metrics.items()}
            )

    source_metrics, source_violations = _validate_fixed_eval(
        source_evaluation,
        label="source",
        expected_step=source_step,
        expected_suite_sha256=suite_sha256,
        required_metrics=required_metrics,
    )
    final_metrics, final_violations = _validate_fixed_eval(
        final_evaluation,
        label="final",
        expected_step=target_step,
        expected_suite_sha256=suite_sha256,
        required_metrics=required_metrics,
    )
    contract_violations.extend(source_violations)
    contract_violations.extend(final_violations)

    continuation_result = evaluate_promotion_gate(
        final_metrics, {"metrics": continuation_metrics}
    )
    continuation_violations = list(continuation_result["violations"])
    continuable = not contract_violations and continuation_result["passed"] is True

    promotion_violations: list[str] = []
    if not promotion_authorized:
        promotion_violations.append("pilot gate does not authorize promotion")
        promotable = False
    else:
        promotion_result = evaluate_promotion_gate(
            final_metrics, {"metrics": promotion_metrics}
        )
        promotion_violations.extend(promotion_result["violations"])
        promotable = (
            continuable
            and not promotion_violations
            and promotion_result["passed"] is True
        )

    all_violations = contract_violations + continuation_violations
    return {
        "passed": continuable,
        "continuable": continuable,
        "promotable": promotable,
        "violations": all_violations,
        "contract_violations": contract_violations,
        "continuation_violations": continuation_violations,
        "promotion_violations": promotion_violations,
        "fixed_suite_sha256": suite_sha256.lower() if suite_sha256 else None,
        "source_step": source_step,
        "target_step": target_step,
    }
