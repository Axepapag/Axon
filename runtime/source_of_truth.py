"""Strict loader for Axon's machine-readable Source-of-Truth capacity law.

The policy contains renewable physical-work and result-selection controls.  It
must never be used as checkpoint capacity, a content-admission ceiling, or a
reason to report partial work as complete.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

CAPACITY_POLICY_SCHEMA = "axon-source-of-truth-capacity-policy-v1"
CAPACITY_POLICY_RELATIVE_PATH = Path("configs/source_of_truth/capacity_policy.json")
EXPECTED_CAPACITY_POLICY_SHA256 = "4a32f18fa296c415ccf34a8c1956d2a4f8afd044265e7f45505782dd53c7b8cd"
_REQUIRED_LAWS = frozenset(
    {
        "no_tissue_content_ceiling",
        "no_silent_truncation",
        "incomplete_never_claimed_complete",
        "resource_controls_are_renewable",
        "resource_controls_do_not_affect_tissue_identity",
        "oversized_exact_items_are_preserved_whole",
    }
)
_CONTROL_KEYS = frozenset(
    {
        "category",
        "value",
        "source_preserved",
        "affects_tissue_identity",
        "continuation",
    }
)
_CATEGORIES = frozenset(
    {
        "physical_processing_unit",
        "compute_budget",
        "optimization_budget",
        "result_count_policy",
    }
)


class SourceOfTruthPolicyError(RuntimeError):
    """The protected policy is absent, stale, malformed, or weakens doctrine."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class GovernedWorkControl:
    name: str
    category: str
    value: int | float
    source_preserved: bool
    affects_tissue_identity: bool
    continuation: str

    def __post_init__(self) -> None:
        if self.category not in _CATEGORIES:
            raise SourceOfTruthPolicyError(f"unknown work-control category for {self.name!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)) or self.value <= 0:
            raise SourceOfTruthPolicyError(f"work control {self.name!r} must have a positive numeric value")
        if not self.source_preserved:
            raise SourceOfTruthPolicyError(f"work control {self.name!r} does not preserve source")
        if self.affects_tissue_identity:
            raise SourceOfTruthPolicyError(f"work control {self.name!r} illegally affects tissue identity")
        if not isinstance(self.continuation, str) or not self.continuation.strip():
            raise SourceOfTruthPolicyError(f"work control {self.name!r} lacks continuation semantics")


@dataclass(frozen=True, slots=True)
class AxonCapacityPolicy:
    path: Path
    authority: str
    laws: Mapping[str, bool]
    controls: Mapping[str, GovernedWorkControl]
    policy_sha256: str

    def control(self, name: str) -> GovernedWorkControl:
        try:
            return self.controls[name]
        except KeyError as exc:
            raise SourceOfTruthPolicyError(f"undeclared finite work control {name!r}") from exc

    def integer(self, name: str) -> int:
        value = self.control(name).value
        if isinstance(value, bool) or not isinstance(value, int):
            raise SourceOfTruthPolicyError(f"work control {name!r} is not an integer")
        return value

    def number(self, name: str) -> float:
        return float(self.control(name).value)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_capacity_policy(
    path: Path | str | None = None,
    *,
    verify_binding: bool = True,
) -> AxonCapacityPolicy:
    policy_path = (
        _default_repo_root() / CAPACITY_POLICY_RELATIVE_PATH if path is None else Path(path).resolve(strict=False)
    )
    try:
        raw = policy_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceOfTruthPolicyError(f"cannot load protected capacity policy: {policy_path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "authority", "laws", "controls"}:
        raise SourceOfTruthPolicyError("capacity policy has an unexpected top-level shape")
    if payload["schema"] != CAPACITY_POLICY_SCHEMA:
        raise SourceOfTruthPolicyError("unsupported capacity-policy schema")
    authority = payload["authority"]
    if not isinstance(authority, str) or not authority.strip():
        raise SourceOfTruthPolicyError("capacity policy requires explicit authority")
    laws = payload["laws"]
    if not isinstance(laws, dict) or set(laws) != _REQUIRED_LAWS or not all(value is True for value in laws.values()):
        raise SourceOfTruthPolicyError("capacity policy must affirm every no-ceilings law")
    raw_controls = payload["controls"]
    if not isinstance(raw_controls, dict) or not raw_controls:
        raise SourceOfTruthPolicyError("capacity policy requires governed work controls")
    controls: dict[str, GovernedWorkControl] = {}
    for name, raw_control in raw_controls.items():
        if not isinstance(name, str) or not name or not isinstance(raw_control, dict):
            raise SourceOfTruthPolicyError("capacity policy contains a malformed control")
        if set(raw_control) != _CONTROL_KEYS:
            raise SourceOfTruthPolicyError(f"work control {name!r} has an unexpected shape")
        controls[name] = GovernedWorkControl(name=name, **raw_control)
    canonical = _canonical_bytes(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    if verify_binding and digest != EXPECTED_CAPACITY_POLICY_SHA256:
        raise SourceOfTruthPolicyError(
            "protected capacity policy hash does not match the code/SOT binding: "
            f"expected {EXPECTED_CAPACITY_POLICY_SHA256}, observed {digest}"
        )
    return AxonCapacityPolicy(
        path=policy_path,
        authority=authority.strip(),
        laws=MappingProxyType(dict(sorted(laws.items()))),
        controls=MappingProxyType(dict(sorted(controls.items()))),
        policy_sha256=digest,
    )


@lru_cache(maxsize=1)
def capacity_policy() -> AxonCapacityPolicy:
    return load_capacity_policy()


__all__ = [
    "CAPACITY_POLICY_RELATIVE_PATH",
    "CAPACITY_POLICY_SCHEMA",
    "EXPECTED_CAPACITY_POLICY_SHA256",
    "AxonCapacityPolicy",
    "GovernedWorkControl",
    "SourceOfTruthPolicyError",
    "capacity_policy",
    "load_capacity_policy",
]
