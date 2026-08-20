"""Fail-closed continuity validation for persisted Axon training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
from typing import Any


def _present(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "numel"):
        try:
            return int(value.numel()) > 0
        except (TypeError, ValueError):
            return False
    if isinstance(value, (str, bytes, Mapping, Sequence)):
        return len(value) > 0
    return True


def _exact_family_keys(value: Any, families: Sequence[str]) -> bool:
    return isinstance(value, Mapping) and set(value.keys()) == set(families)


def _finite_real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _exact_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    normalized = int(value)
    return normalized if normalized >= 0 else None


def _validate_weighted_sampler(
    curriculum: Mapping[str, Any],
    *,
    expected_families: Sequence[str],
    sampler_count: int,
    sampler_position: int,
    continuity_provenance: Any,
) -> tuple[list[str], dict[str, Any]]:
    """Validate the complete deterministic weighted-without-replacement state."""

    violations: list[str] = []
    families = list(expected_families)

    raw_example_count = _exact_nonnegative_int(curriculum.get("example_count"))
    if raw_example_count is None or raw_example_count <= 0:
        violations.append("weighted sampler example_count must be a positive integer")
    raw_global_position = _exact_nonnegative_int(curriculum.get("position"))
    if raw_global_position is None:
        violations.append("weighted sampler global position must be a non-negative integer")

    raw_weights = curriculum.get("family_weights")
    weights: dict[str, float] = {}
    if not _exact_family_keys(raw_weights, families):
        violations.append("weighted sampler family_weights keys do not match expected families")
    else:
        for family in families:
            weight = _finite_real(raw_weights[family])
            if weight is None or weight < 0:
                violations.append(
                    "weighted sampler family_weights must be finite and non-negative"
                )
                weights = {}
                break
            weights[family] = weight
        if weights and not math.isclose(
            sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            violations.append("weighted sampler family_weights are not normalized")
        if weights and not any(weight > 0 for weight in weights.values()):
            violations.append("weighted sampler family_weights have no positive weight")

    raw_orders = curriculum.get("family_orders")
    orders: dict[str, list[int]] = {}
    if not _exact_family_keys(raw_orders, families):
        violations.append("weighted sampler family_orders keys do not match expected families")
    else:
        orders_valid = True
        for family in families:
            family_order = raw_orders[family]
            if not isinstance(family_order, Sequence) or isinstance(
                family_order, (str, bytes)
            ):
                orders_valid = False
                break
            normalized_order: list[int] = []
            for value in family_order:
                index = _exact_nonnegative_int(value)
                if index is None:
                    orders_valid = False
                    break
                normalized_order.append(index)
            if not orders_valid or not normalized_order:
                orders_valid = False
                break
            orders[family] = normalized_order
        if not orders_valid:
            violations.append(
                "weighted sampler family_orders must be non-empty integer sequences"
            )
            orders = {}
        elif sorted(index for order in orders.values() for index in order) != list(
            range(sampler_count)
        ):
            violations.append(
                "weighted sampler family_orders do not partition global sampler indices"
            )

    raw_positions = curriculum.get("family_positions")
    positions: dict[str, int] = {}
    if not _exact_family_keys(raw_positions, families):
        violations.append("weighted sampler family_positions keys do not match expected families")
    else:
        for family in families:
            position = _exact_nonnegative_int(raw_positions[family])
            if position is None:
                violations.append(
                    "weighted sampler family_positions must be non-negative integers"
                )
                positions = {}
                break
            positions[family] = position
        if positions and orders:
            for family in families:
                if positions[family] > len(orders[family]):
                    violations.append(
                        f"weighted sampler family position is out of bounds: {family}"
                    )

    raw_credits = curriculum.get("family_credits")
    credits: dict[str, float] = {}
    if not _exact_family_keys(raw_credits, families):
        violations.append("weighted sampler family_credits keys do not match expected families")
    else:
        for family in families:
            credit = _finite_real(raw_credits[family])
            if credit is None:
                violations.append("weighted sampler family_credits must be finite")
                credits = {}
                break
            credits[family] = credit
        if credits and not math.isclose(
            sum(credits.values()), 0.0, rel_tol=0.0, abs_tol=1e-9
        ):
            violations.append("weighted sampler family_credits are not normalized")

    family_steps = _exact_nonnegative_int(curriculum.get("family_steps"))
    if family_steps is None:
        violations.append("weighted sampler family_steps is missing or invalid")
        family_steps = -1
    elif positions and family_steps != sum(positions.values()):
        violations.append("weighted sampler family_steps does not equal consumed positions")
    if family_steps >= 0 and sampler_position != family_steps:
        violations.append("weighted sampler global position does not equal family_steps")

    exhaustion_policy = curriculum.get("family_exhaustion_policy")
    if exhaustion_policy != "error":
        violations.append("weighted sampler family exhaustion policy is not error")

    # The trainer uses deterministic smooth weighted scheduling.  Replaying at
    # most example_count decisions proves positions and credits came from that
    # policy, instead of only looking numerically plausible.
    if (
        weights
        and orders
        and positions
        and credits
        and family_steps >= 0
        and family_steps <= sampler_count
        and math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        positive_families = [family for family in families if weights[family] > 0]
        expected_positions = {family: 0 for family in families}
        expected_credits = {family: 0.0 for family in families}
        replay_valid = True
        for _ in range(family_steps):
            for family in positive_families:
                expected_credits[family] += weights[family]
            chosen = max(
                positive_families,
                key=lambda family: (
                    expected_credits[family],
                    -families.index(family),
                ),
            )
            if expected_positions[chosen] >= len(orders[chosen]):
                violations.append(
                    "weighted sampler state advances past family exhaustion"
                )
                replay_valid = False
                break
            expected_credits[chosen] -= sum(
                weights[family] for family in positive_families
            )
            expected_positions[chosen] += 1
        if replay_valid and positions != expected_positions:
            violations.append(
                "weighted sampler family_positions do not match deterministic schedule"
            )
        if replay_valid and any(
            not math.isclose(
                credits[family],
                expected_credits[family],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for family in families
        ):
            violations.append(
                "weighted sampler family_credits do not match deterministic schedule"
            )

    provenance_sampler: Mapping[str, Any] | None = None
    if (
        not isinstance(continuity_provenance, Mapping)
        or continuity_provenance.get("schema") != "axon_continuity_provenance_v1"
    ):
        violations.append(
            "weighted sampler continuity provenance schema is missing or invalid"
        )
    else:
        provenance_value = continuity_provenance.get("sampler")
        if not isinstance(provenance_value, Mapping):
            violations.append("weighted sampler continuity provenance is missing or invalid")
        else:
            provenance_sampler = provenance_value
            if provenance_sampler.get("family_sampling_policy") != "weighted":
                violations.append(
                    "weighted sampler provenance family policy does not match training state"
                )
            provenance_weights = provenance_sampler.get("family_weights")
            if not _exact_family_keys(provenance_weights, families):
                violations.append(
                    "weighted sampler provenance family_weights keys do not match expected families"
                )
            elif not _exact_family_keys(raw_weights, families):
                violations.append(
                    "weighted sampler provenance family_weights do not match training state"
                )
            elif weights:
                for family in families:
                    provenance_weight = _finite_real(provenance_weights[family])
                    if provenance_weight is None or not math.isclose(
                        provenance_weight,
                        weights[family],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        violations.append(
                            "weighted sampler provenance family_weights do not match training state"
                        )
                        break
            if list(provenance_sampler.get("families", [])) != families:
                violations.append(
                    "weighted sampler provenance families do not match expected families"
                )
            if provenance_sampler.get("family_exhaustion_policy") != exhaustion_policy:
                violations.append(
                    "weighted sampler provenance exhaustion policy does not match training state"
                )

    return violations, {
        "valid": not violations,
        "family_sampling_policy": "weighted",
        "family_weights": weights,
        "family_positions": positions,
        "family_steps": family_steps,
        "family_exhaustion_policy": exhaustion_policy,
        "provenance_matched": provenance_sampler is not None
        and not any("provenance" in violation for violation in violations),
    }


def validate_checkpoint_continuity(
    payload: Mapping[str, Any],
    *,
    expected_step: int,
    expected_families: Sequence[str],
    require_cuda_rng: bool = True,
    expected_source_step: int | None = None,
    expected_optimizer_policy: str | None = None,
    expected_optimizer_state_restored: bool | None = None,
    expected_effective_lr: float | None = None,
    expected_sampler_policy: str | None = None,
    expected_sampler_reset: bool | None = None,
) -> dict[str, Any]:
    """Validate optimizer, RNG, and exact curriculum continuation state.

    Legacy checkpoints remain valid with the original arguments.  Passing any
    ``expected_*`` provenance argument opts into the stricter pilot contract,
    which requires a persisted ``continuity_provenance`` object with schema
    ``axon_continuity_provenance_v1``.  The effective learning rate is checked
    against both the provenance record and every serialized optimizer group;
    sampler reset/preserve intent is likewise explicit rather than inferred
    from its final position.
    """

    violations: list[str] = []
    if payload.get("checkpoint_schema") != "axon_charslot_checkpoint_v2":
        violations.append("checkpoint schema is not axon_charslot_checkpoint_v2")
    try:
        payload_step = int(payload.get("step", -1))
    except (TypeError, ValueError):
        payload_step = -1
    if payload_step != int(expected_step):
        violations.append(f"checkpoint step does not equal {expected_step}")

    optimizer = payload.get("optimizer_state")
    optimizer_violations: list[str] = []
    if not isinstance(optimizer, Mapping):
        optimizer_violations.append("optimizer_state is missing")
        optimizer_state_count = 0
        optimizer_group_count = 0
    else:
        state = optimizer.get("state")
        groups = optimizer.get("param_groups")
        optimizer_state_count = len(state) if isinstance(state, Mapping) else 0
        optimizer_group_count = (
            len(groups)
            if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes))
            else 0
        )
        if optimizer_state_count <= 0:
            optimizer_violations.append("optimizer state entries are empty")
        if optimizer_group_count <= 0:
            optimizer_violations.append("optimizer parameter groups are empty")
    violations.extend(optimizer_violations)

    training_state = payload.get("training_state")
    rng_violations: list[str] = []
    sampler_violations: list[str] = []
    if not isinstance(training_state, Mapping) or training_state.get("schema") != "axon_training_state_v1":
        rng_violations.append("training_state schema is missing or invalid")
        curriculum = None
    else:
        for key in ("python_random_state", "numpy_random_state", "torch_rng_state"):
            if not _present(training_state.get(key)):
                rng_violations.append(f"{key} is missing or empty")
        if require_cuda_rng and not _present(training_state.get("cuda_rng_state_all")):
            rng_violations.append("cuda_rng_state_all is missing or empty")
        curriculum = training_state.get("curriculum_state")
    violations.extend(rng_violations)

    expected_family_list = list(expected_families)
    sampler_count = 0
    sampler_position = -1
    family_sampling_policy = "global"
    weighted_sampler_result: dict[str, Any] | None = None
    if not isinstance(curriculum, Mapping) or curriculum.get("schema") != "axon_phase0b_sampler_v1":
        sampler_violations.append("curriculum sampler schema is missing or invalid")
    else:
        try:
            sampler_count = int(curriculum.get("example_count", -1))
            sampler_position = int(curriculum.get("position", -1))
        except (TypeError, ValueError):
            sampler_count = -1
            sampler_position = -1
        order = curriculum.get("order")
        if sampler_count <= 0:
            sampler_violations.append("sampler example_count is not positive")
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            sampler_violations.append("sampler order is missing")
        else:
            try:
                normalized_order = [int(value) for value in order]
            except (TypeError, ValueError):
                normalized_order = []
            if len(normalized_order) != sampler_count or sorted(normalized_order) != list(range(sampler_count)):
                sampler_violations.append("sampler order is not a complete permutation")
        if not 0 <= sampler_position <= sampler_count:
            sampler_violations.append("sampler position is out of bounds")
        if not _present(curriculum.get("rng_state")):
            sampler_violations.append("sampler RNG state is missing or empty")
        if list(curriculum.get("families", [])) != expected_family_list:
            sampler_violations.append("sampler families do not match the exact curriculum manifest")
        policy_value = curriculum.get("family_sampling_policy", "global")
        if policy_value not in {"global", "weighted"}:
            sampler_violations.append("sampler family sampling policy is unsupported")
            family_sampling_policy = str(policy_value)
        else:
            family_sampling_policy = str(policy_value)
        if family_sampling_policy == "weighted":
            weighted_violations, weighted_sampler_result = _validate_weighted_sampler(
                curriculum,
                expected_families=expected_family_list,
                sampler_count=sampler_count,
                sampler_position=sampler_position,
                continuity_provenance=payload.get("continuity_provenance"),
            )
            sampler_violations.extend(weighted_violations)
    violations.extend(sampler_violations)

    provenance_requested = any(
        value is not None
        for value in (
            expected_source_step,
            expected_optimizer_policy,
            expected_optimizer_state_restored,
            expected_effective_lr,
            expected_sampler_policy,
            expected_sampler_reset,
        )
    )
    provenance_violations: list[str] = []
    provenance_result: dict[str, Any] | None = None
    if provenance_requested:
        provenance = payload.get("continuity_provenance")
        optimizer_provenance: Mapping[str, Any] | None = None
        sampler_provenance: Mapping[str, Any] | None = None
        provenance_source_step = -1
        optimizer_policy: str | None = None
        optimizer_state_restored: bool | None = None
        effective_lr: float | None = None
        sampler_policy: str | None = None
        sampler_reset: bool | None = None
        sampler_initial_position = -1

        if (
            not isinstance(provenance, Mapping)
            or provenance.get("schema") != "axon_continuity_provenance_v1"
        ):
            provenance_violations.append("continuity provenance schema is missing or invalid")
        else:
            try:
                provenance_source_step = int(provenance.get("source_step", -1))
            except (TypeError, ValueError):
                provenance_source_step = -1
            if provenance_source_step < 0:
                provenance_violations.append("continuity provenance source_step is missing or invalid")
            elif provenance_source_step >= payload_step:
                provenance_violations.append(
                    "continuity provenance source_step is not before the checkpoint step"
                )
            if (
                expected_source_step is not None
                and provenance_source_step != int(expected_source_step)
            ):
                provenance_violations.append(
                    f"continuity provenance source_step does not equal {expected_source_step}"
                )

            optimizer_value = provenance.get("optimizer")
            if not isinstance(optimizer_value, Mapping):
                provenance_violations.append("optimizer provenance is missing or invalid")
            else:
                optimizer_provenance = optimizer_value
                policy_value = optimizer_provenance.get("policy")
                if not isinstance(policy_value, str) or not policy_value:
                    provenance_violations.append("optimizer provenance policy is missing or invalid")
                else:
                    optimizer_policy = policy_value
                    if (
                        expected_optimizer_policy is not None
                        and optimizer_policy != expected_optimizer_policy
                    ):
                        provenance_violations.append(
                            "optimizer provenance policy does not match the expected policy"
                        )

                restored_value = optimizer_provenance.get("state_restored")
                if not isinstance(restored_value, bool):
                    provenance_violations.append(
                        "optimizer provenance state_restored must be boolean"
                    )
                else:
                    optimizer_state_restored = restored_value
                    if (
                        expected_optimizer_state_restored is not None
                        and optimizer_state_restored is not expected_optimizer_state_restored
                    ):
                        provenance_violations.append(
                            "optimizer provenance state_restored does not match the expected policy"
                        )

                try:
                    effective_lr = float(optimizer_provenance.get("effective_lr"))
                except (TypeError, ValueError):
                    effective_lr = None
                if effective_lr is None or not math.isfinite(effective_lr) or effective_lr <= 0:
                    provenance_violations.append(
                        "optimizer provenance effective_lr is missing, non-finite, or non-positive"
                    )
                    effective_lr = None
                elif expected_effective_lr is not None:
                    try:
                        expected_lr = float(expected_effective_lr)
                    except (TypeError, ValueError):
                        expected_lr = math.nan
                    if not math.isfinite(expected_lr) or expected_lr <= 0:
                        provenance_violations.append("expected effective learning rate is invalid")
                    elif not math.isclose(effective_lr, expected_lr, rel_tol=1e-12, abs_tol=0.0):
                        provenance_violations.append(
                            "optimizer provenance effective_lr does not match the expected learning rate"
                        )

                if effective_lr is not None and isinstance(optimizer, Mapping):
                    groups = optimizer.get("param_groups")
                    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
                        for index, group in enumerate(groups):
                            if not isinstance(group, Mapping):
                                provenance_violations.append(
                                    f"optimizer parameter group {index} is invalid"
                                )
                                continue
                            try:
                                group_lr = float(group.get("lr"))
                            except (TypeError, ValueError):
                                group_lr = math.nan
                            if not math.isfinite(group_lr):
                                provenance_violations.append(
                                    f"optimizer parameter group {index} learning rate is missing or non-finite"
                                )
                            elif not math.isclose(
                                group_lr, effective_lr, rel_tol=1e-12, abs_tol=0.0
                            ):
                                provenance_violations.append(
                                    f"optimizer parameter group {index} learning rate does not match effective_lr"
                                )

            sampler_value = provenance.get("sampler")
            if not isinstance(sampler_value, Mapping):
                provenance_violations.append("sampler provenance is missing or invalid")
            else:
                sampler_provenance = sampler_value
                policy_value = sampler_provenance.get("policy")
                if not isinstance(policy_value, str) or not policy_value:
                    provenance_violations.append("sampler provenance policy is missing or invalid")
                else:
                    sampler_policy = policy_value
                    if (
                        expected_sampler_policy is not None
                        and sampler_policy != expected_sampler_policy
                    ):
                        provenance_violations.append(
                            "sampler provenance policy does not match the expected policy"
                        )

                reset_value = sampler_provenance.get("reset")
                if not isinstance(reset_value, bool):
                    provenance_violations.append("sampler provenance reset must be boolean")
                else:
                    sampler_reset = reset_value
                    if expected_sampler_reset is not None and sampler_reset is not expected_sampler_reset:
                        provenance_violations.append(
                            "sampler provenance reset does not match the expected policy"
                        )

                try:
                    sampler_initial_position = int(
                        sampler_provenance.get("initial_position", -1)
                    )
                except (TypeError, ValueError):
                    sampler_initial_position = -1
                if sampler_initial_position < 0:
                    provenance_violations.append(
                        "sampler provenance initial_position is missing or invalid"
                    )
                elif sampler_reset is True and sampler_initial_position != 0:
                    provenance_violations.append(
                        "reset sampler provenance initial_position is not zero"
                    )

                provenance_families = sampler_provenance.get("families")
                if (
                    not isinstance(provenance_families, Sequence)
                    or isinstance(provenance_families, (str, bytes))
                    or list(provenance_families) != expected_family_list
                ):
                    provenance_violations.append(
                        "sampler provenance families do not match the exact curriculum manifest"
                    )

        violations.extend(provenance_violations)
        provenance_result = {
            "valid": not provenance_violations
            and (
                weighted_sampler_result is None
                or weighted_sampler_result.get("provenance_matched") is True
            ),
            "schema": provenance.get("schema") if isinstance(provenance, Mapping) else None,
            "source_step": provenance_source_step,
            "optimizer": {
                "policy": optimizer_policy,
                "state_restored": optimizer_state_restored,
                "effective_lr": effective_lr,
            },
            "sampler": {
                "policy": sampler_policy,
                "reset": sampler_reset,
                "initial_position": sampler_initial_position,
                "families": (
                    list(sampler_provenance.get("families", []))
                    if isinstance(sampler_provenance, Mapping)
                    else []
                ),
            },
        }
        if family_sampling_policy == "weighted" and isinstance(
            sampler_provenance, Mapping
        ):
            provenance_result["sampler"].update(
                {
                    "family_sampling_policy": sampler_provenance.get(
                        "family_sampling_policy"
                    ),
                    "family_weights": dict(
                        sampler_provenance.get("family_weights", {})
                    )
                    if isinstance(sampler_provenance.get("family_weights"), Mapping)
                    else {},
                    "family_exhaustion_policy": sampler_provenance.get(
                        "family_exhaustion_policy"
                    ),
                }
            )

    result = {
        "valid": not violations,
        "violations": violations,
        "step": payload_step,
        "optimizer": {
            "valid": not optimizer_violations,
            "state_entries": optimizer_state_count,
            "param_groups": optimizer_group_count,
        },
        "rng": {
            "valid": not rng_violations,
            "python": isinstance(training_state, Mapping) and _present(training_state.get("python_random_state")),
            "numpy": isinstance(training_state, Mapping) and _present(training_state.get("numpy_random_state")),
            "torch": isinstance(training_state, Mapping) and _present(training_state.get("torch_rng_state")),
            "cuda": isinstance(training_state, Mapping) and _present(training_state.get("cuda_rng_state_all")),
        },
        "sampler": {
            "valid": not sampler_violations,
            "schema": curriculum.get("schema") if isinstance(curriculum, Mapping) else None,
            "example_count": sampler_count,
            "position": sampler_position,
            "order_length": len(curriculum.get("order", [])) if isinstance(curriculum, Mapping) else 0,
            "families": list(curriculum.get("families", [])) if isinstance(curriculum, Mapping) else [],
            "rng_state": isinstance(curriculum, Mapping) and _present(curriculum.get("rng_state")),
            "family_sampling_policy": family_sampling_policy,
        },
    }
    if weighted_sampler_result is not None:
        result["sampler"]["weighted"] = weighted_sampler_result
    if provenance_result is not None:
        result["provenance"] = provenance_result
    return result
