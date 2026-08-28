"""Durable Heart-owned control for independent per-region attention masks.

The canonical regional body never moves.  This store contains only the
Heart's derived-view policy: one explicit policy for every canonical region.
Changing it advances mask-control state and causes a new rail ``view_id`` over
the same canonical ``field_id``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from runtime.field import (
    CANONICAL_REGION_ORDER,
    LogicalRegion,
    RegionMaskPolicy,
    canonical_sha256,
)

from .errors import MaskStateCorruptionError

HEART_REGION_MASK_SCHEMA = "axon-heart-region-masks-v1"


def _normalize_policies(
    policies: Mapping[LogicalRegion | str, RegionMaskPolicy | Mapping[str, Any]] | None,
) -> dict[LogicalRegion, RegionMaskPolicy]:
    normalized = {region: RegionMaskPolicy("all") for region in CANONICAL_REGION_ORDER}
    if policies is None:
        return normalized
    for raw_region, raw_policy in policies.items():
        region = raw_region if isinstance(raw_region, LogicalRegion) else LogicalRegion(raw_region)
        if isinstance(raw_policy, RegionMaskPolicy):
            policy = raw_policy
        elif isinstance(raw_policy, Mapping):
            if set(raw_policy) - {"kind", "limit"}:
                raise MaskStateCorruptionError(f"mask policy for {region.value!r} has unknown fields")
            try:
                policy = RegionMaskPolicy(
                    raw_policy["kind"],
                    raw_policy.get("limit", 0),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MaskStateCorruptionError(f"mask policy for {region.value!r} is invalid") from exc
        else:
            raise TypeError("region mask policies must be RegionMaskPolicy or mappings")
        normalized[region] = policy
    return normalized


@dataclass(frozen=True, slots=True)
class HeartRegionMaskState:
    """One durable revision of the complete ten-region mask-control set."""

    revision: int
    policies: Mapping[LogicalRegion, RegionMaskPolicy]
    state_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("HeartRegionMaskState.revision must be an integer")
        if self.revision < 0:
            raise ValueError("HeartRegionMaskState.revision must be non-negative")
        try:
            supplied_regions = {
                region if isinstance(region, LogicalRegion) else LogicalRegion(region) for region in self.policies
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("mask state contains an unknown canonical region") from exc
        if supplied_regions != set(CANONICAL_REGION_ORDER):
            raise ValueError("mask state must contain every canonical region")
        normalized = _normalize_policies(self.policies)
        object.__setattr__(self, "policies", MappingProxyType(normalized))
        object.__setattr__(self, "state_id", canonical_sha256(self.to_canonical_dict()))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": HEART_REGION_MASK_SCHEMA,
            "revision": self.revision,
            "policies": [
                {
                    "region": region.value,
                    "policy": self.policies[region].to_canonical_dict(),
                }
                for region in CANONICAL_REGION_ORDER
            ],
        }

    def policy_for(self, region: LogicalRegion | str) -> RegionMaskPolicy:
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        return self.policies[logical]


class HeartRegionMaskController:
    """Atomic durable owner of Axon's independent regional mask controls."""

    def __init__(
        self,
        path: Path | str,
        *,
        initial_policies: Mapping[
            LogicalRegion | str,
            RegionMaskPolicy | Mapping[str, Any],
        ]
        | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.exists():
            self._state = self._load()
        else:
            self._state = HeartRegionMaskState(
                revision=0,
                policies=_normalize_policies(initial_policies),
            )
            self._save(self._state)

    @property
    def state(self) -> HeartRegionMaskState:
        return self._state

    def policies(self) -> dict[LogicalRegion, RegionMaskPolicy]:
        """Return a detached complete policy map for one compile/freeze."""

        return dict(self._state.policies)

    def set_policy(
        self,
        region: LogicalRegion | str,
        policy: RegionMaskPolicy,
    ) -> HeartRegionMaskState:
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        if not isinstance(policy, RegionMaskPolicy):
            raise TypeError("set_policy requires a RegionMaskPolicy")
        if self._state.policy_for(logical) == policy:
            return self._state
        policies = self.policies()
        policies[logical] = policy
        updated = HeartRegionMaskState(
            revision=self._state.revision + 1,
            policies=policies,
        )
        self._save(updated)
        self._state = updated
        return updated

    def set_unmasked_percent(
        self,
        region: LogicalRegion | str,
        percent: int,
    ) -> HeartRegionMaskState:
        """Set one region's newest-suffix slider from 0 through 100 percent."""

        return self.set_policy(region, RegionMaskPolicy("tail_percent", percent))

    def recover(self) -> HeartRegionMaskState:
        """Reload durable state and reject revision rollback."""

        loaded = self._load()
        if loaded.revision < self._state.revision:
            raise MaskStateCorruptionError("region mask revision decreased on recovery")
        if loaded.revision == self._state.revision and loaded.state_id != self._state.state_id:
            raise MaskStateCorruptionError("region mask state changed without advancing its revision")
        self._state = loaded
        return loaded

    def _load(self) -> HeartRegionMaskState:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MaskStateCorruptionError(f"cannot read Heart region-mask state: {self.path}") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "revision",
            "policies",
            "state_id",
        }:
            raise MaskStateCorruptionError("serialized region-mask fields are invalid")
        if value.get("schema") != HEART_REGION_MASK_SCHEMA:
            raise MaskStateCorruptionError("unsupported Heart region-mask schema")
        raw_policies = value.get("policies")
        if not isinstance(raw_policies, list):
            raise MaskStateCorruptionError("region-mask policies must be a list")
        parsed: dict[LogicalRegion, RegionMaskPolicy] = {}
        try:
            for item in raw_policies:
                if not isinstance(item, dict) or set(item) != {"region", "policy"}:
                    raise MaskStateCorruptionError("serialized mask policy is invalid")
                region = LogicalRegion(item["region"])
                if region in parsed:
                    raise MaskStateCorruptionError(f"duplicate mask policy for region {region.value!r}")
                policy_value = item["policy"]
                if not isinstance(policy_value, dict) or set(policy_value) != {
                    "kind",
                    "limit",
                }:
                    raise MaskStateCorruptionError("serialized mask policy value is invalid")
                parsed[region] = RegionMaskPolicy(
                    policy_value["kind"],
                    policy_value["limit"],
                )
            state = HeartRegionMaskState(
                revision=value["revision"],
                policies=parsed,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MaskStateCorruptionError("invalid Heart region-mask state") from exc
        if len(parsed) != len(CANONICAL_REGION_ORDER):
            raise MaskStateCorruptionError("region-mask state must contain every canonical region exactly once")
        if state.state_id != value["state_id"]:
            raise MaskStateCorruptionError("Heart region-mask state identity mismatch")
        return state

    def _save(self, state: HeartRegionMaskState) -> None:
        value = {**state.to_canonical_dict(), "state_id": state.state_id}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        data = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


__all__ = [
    "HEART_REGION_MASK_SCHEMA",
    "HeartRegionMaskController",
    "HeartRegionMaskState",
]
