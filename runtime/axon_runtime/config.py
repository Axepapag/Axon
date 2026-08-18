"""Strict, side-effect-free descriptors for Axon runtime bootstrap.

Importing or parsing this module never loads a checkpoint, starts a model,
creates a directory, opens the runtime database, or enters a tick loop.

``config_id`` values below hash only canonical configuration descriptors.
They are deliberately *not* ``CoreIdentity.core_state_sha256`` values and
must never be substituted for checkpoint tensor-state evidence.  A bootstrap
layer may construct ``CoreIdentity`` only after strict checkpoint inspection
has supplied the real ``ExactV4Checkpoint.core_state_sha256``.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from runtime.field import CANONICAL_REGION_ORDER, LogicalRegion, canonical_sha256

from .contracts import ACTIVE_ROLES
from .projection import PINNED_REGIONS


RUNTIME_CONFIG_SCHEMA = "axon-runtime-bootstrap-config-v1"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_TOP_LEVEL_KEYS = {
    "schema",
    "models",
    "cores",
    "ring",
    "paths",
    "projection",
    "idle",
    "loop",
    "capabilities",
}
_LITERAL_SECRET_KEY_PARTS = (
    "secret",
    "password",
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "credential",
)


class RuntimeConfigError(ValueError):
    """A runtime bootstrap descriptor is malformed or unsafe."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise RuntimeConfigError(f"{label} must be a string-keyed object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeConfigError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeConfigError(f"{label} must be a non-empty string")
    return value


def _identifier(value: Any, label: str) -> str:
    result = _nonempty(value, label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", result):
        raise RuntimeConfigError(f"{label} contains unsafe characters")
    return result


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeConfigError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _number(value: Any, label: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise RuntimeConfigError(
            f"{label} must be finite and >= {minimum}"
        )
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeConfigError(
            f"{label} must be a 64-character hexadecimal SHA-256"
        )
    return value.lower()


def _absolute_path(value: Any, label: str) -> Path:
    raw_value = os.fspath(value) if isinstance(value, os.PathLike) else value
    raw = _nonempty(raw_value, label)
    if "\x00" in raw:
        raise RuntimeConfigError(f"{label} contains NUL")
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeConfigError(f"{label} must be absolute")
    return Path(os.path.abspath(path))


def _strict_string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise RuntimeConfigError(
            f"{label} must be a sequence of non-empty strings"
        )
    result = tuple(value)
    if len(result) != len(set(result)):
        raise RuntimeConfigError(f"{label} cannot contain duplicates")
    return result


def _reject_literal_secret_fields(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _LITERAL_SECRET_KEY_PARTS):
                raise RuntimeConfigError(
                    f"{path}.{key} is a forbidden literal secret field"
                )
            _reject_literal_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_literal_secret_fields(item, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ModelPin:
    model_id: str
    d_model: int
    checkpoint_step: int
    checkpoint_path: Path
    checkpoint_sha256: str
    config_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "d_model",
            _integer(self.d_model, "d_model", minimum=16),
        )
        object.__setattr__(
            self,
            "checkpoint_step",
            _integer(self.checkpoint_step, "checkpoint_step", minimum=1),
        )
        object.__setattr__(
            self,
            "checkpoint_path",
            _absolute_path(self.checkpoint_path, "checkpoint_path"),
        )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "config_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-runtime-model-pin-v1",
            "model_id": self.model_id,
            "d_model": self.d_model,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "ModelPin":
        item = _mapping(value, "model pin")
        _exact_keys(
            item,
            {
                "model_id",
                "d_model",
                "checkpoint_step",
                "checkpoint_path",
                "checkpoint_sha256",
            },
            "model pin",
        )
        return cls(
            model_id=item["model_id"],
            d_model=item["d_model"],
            checkpoint_step=item["checkpoint_step"],
            checkpoint_path=item["checkpoint_path"],
            checkpoint_sha256=item["checkpoint_sha256"],
        )


@dataclass(frozen=True, slots=True)
class LogicalCoreDescriptor:
    core_id: str
    display_name: str
    model_id: str
    soul_id: str
    adapter_namespace: str
    enabled: bool
    lineage: tuple[str, ...]
    role_capabilities: tuple[str, ...]
    config_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("core_id", "model_id", "soul_id", "adapter_namespace"):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "display_name",
            _nonempty(self.display_name, "display_name"),
        )
        if not isinstance(self.enabled, bool):
            raise RuntimeConfigError("enabled must be boolean")
        lineage = _strict_string_sequence(self.lineage, "lineage")
        if not lineage:
            raise RuntimeConfigError("lineage cannot be empty")
        object.__setattr__(self, "lineage", lineage)
        roles = tuple(sorted(_strict_string_sequence(
            self.role_capabilities,
            "role_capabilities",
        )))
        if set(roles) - ACTIVE_ROLES:
            raise RuntimeConfigError("role_capabilities contains unknown roles")
        object.__setattr__(self, "role_capabilities", roles)
        object.__setattr__(
            self,
            "config_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-runtime-logical-core-descriptor-v1",
            "core_id": self.core_id,
            "display_name": self.display_name,
            "model_id": self.model_id,
            "soul_id": self.soul_id,
            "adapter_namespace": self.adapter_namespace,
            "enabled": self.enabled,
            "lineage": list(self.lineage),
            "role_capabilities": list(self.role_capabilities),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "LogicalCoreDescriptor":
        item = _mapping(value, "logical core descriptor")
        _exact_keys(
            item,
            {
                "core_id",
                "display_name",
                "model_id",
                "soul_id",
                "adapter_namespace",
                "enabled",
                "lineage",
                "role_capabilities",
            },
            "logical core descriptor",
        )
        return cls(
            core_id=item["core_id"],
            display_name=item["display_name"],
            model_id=item["model_id"],
            soul_id=item["soul_id"],
            adapter_namespace=item["adapter_namespace"],
            enabled=item["enabled"],
            lineage=_strict_string_sequence(item["lineage"], "lineage"),
            role_capabilities=_strict_string_sequence(
                item["role_capabilities"],
                "role_capabilities",
            ),
        )


@dataclass(frozen=True, slots=True)
class RotationRing:
    mode: str
    core_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.mode != "round_robin":
            raise RuntimeConfigError("ring mode must be 'round_robin'")
        core_ids = tuple(
            _identifier(value, "ring core_id") for value in self.core_ids
        )
        if len(core_ids) < 3 or len(core_ids) != len(set(core_ids)):
            raise RuntimeConfigError(
                "ring requires at least three unique core IDs"
            )
        object.__setattr__(self, "core_ids", core_ids)

    @classmethod
    def from_mapping(cls, value: Any) -> "RotationRing":
        item = _mapping(value, "ring")
        _exact_keys(item, {"mode", "core_ids"}, "ring")
        return cls(
            mode=item["mode"],
            core_ids=_strict_string_sequence(item["core_ids"], "ring.core_ids"),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "core_ids": list(self.core_ids)}


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    state_root: Path
    runtime_db: Path
    soul_store: Path
    private_state_store: Path
    dormant_db: Path
    stop_file: Path

    def __post_init__(self) -> None:
        for name in (
            "state_root",
            "runtime_db",
            "soul_store",
            "private_state_store",
            "dormant_db",
            "stop_file",
        ):
            object.__setattr__(
                self,
                name,
                _absolute_path(getattr(self, name), name),
            )
        root = os.path.normcase(str(self.state_root))
        for name in (
            "runtime_db",
            "soul_store",
            "private_state_store",
            "dormant_db",
            "stop_file",
        ):
            candidate = os.path.normcase(str(getattr(self, name)))
            try:
                common = os.path.commonpath((root, candidate))
            except ValueError as exc:
                raise RuntimeConfigError(
                    f"{name} is on a different volume than state_root"
                ) from exc
            if common != root or candidate == root:
                raise RuntimeConfigError(
                    f"{name} must be a child of state_root"
                )
        values = (
            self.runtime_db,
            self.soul_store,
            self.private_state_store,
            self.dormant_db,
            self.stop_file,
        )
        if len({os.path.normcase(str(value)) for value in values}) != len(values):
            raise RuntimeConfigError("runtime paths must be unique")

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimePaths":
        item = _mapping(value, "paths")
        expected = {
            "state_root",
            "runtime_db",
            "soul_store",
            "private_state_store",
            "dormant_db",
            "stop_file",
        }
        _exact_keys(item, expected, "paths")
        return cls(**{name: item[name] for name in expected})

    def to_canonical_dict(self) -> dict[str, str]:
        return {
            "state_root": str(self.state_root),
            "runtime_db": str(self.runtime_db),
            "soul_store": str(self.soul_store),
            "private_state_store": str(self.private_state_store),
            "dormant_db": str(self.dormant_db),
            "stop_file": str(self.stop_file),
        }


@dataclass(frozen=True, slots=True)
class ProjectionSettings:
    global_char_budget: int
    max_selected_spans: int
    max_read_pages_per_action: int
    region_char_budgets: tuple[tuple[str, int], ...]
    pinned_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_char_budget",
            _integer(
                self.global_char_budget,
                "projection.global_char_budget",
                minimum=384,
            ),
        )
        object.__setattr__(
            self,
            "max_selected_spans",
            _integer(
                self.max_selected_spans,
                "projection.max_selected_spans",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "max_read_pages_per_action",
            _integer(
                self.max_read_pages_per_action,
                "projection.max_read_pages_per_action",
                minimum=1,
            ),
        )
        raw_budgets = dict(self.region_char_budgets)
        expected = {region.value for region in CANONICAL_REGION_ORDER}
        if set(raw_budgets) != expected or len(raw_budgets) != len(
            self.region_char_budgets
        ):
            raise RuntimeConfigError(
                "projection region budgets must exactly cover logical regions"
            )
        budgets = tuple(
            (
                region.value,
                _integer(
                    raw_budgets[region.value],
                    f"projection budget {region.value}",
                ),
            )
            for region in CANONICAL_REGION_ORDER
        )
        if sum(budget for _, budget in budgets) < self.global_char_budget:
            raise RuntimeConfigError(
                "projection region budgets cannot total below global budget"
            )
        object.__setattr__(self, "region_char_budgets", budgets)
        pinned = tuple(sorted(_strict_string_sequence(
            self.pinned_regions,
            "projection.pinned_regions",
        )))
        try:
            pinned_values = {LogicalRegion(value) for value in pinned}
        except ValueError as exc:
            raise RuntimeConfigError(
                "projection.pinned_regions contains an unknown region"
            ) from exc
        if not PINNED_REGIONS <= pinned_values:
            raise RuntimeConfigError("projection is missing required pinned regions")
        object.__setattr__(self, "pinned_regions", pinned)

    @classmethod
    def from_mapping(cls, value: Any) -> "ProjectionSettings":
        item = _mapping(value, "projection")
        expected = {
            "global_char_budget",
            "max_selected_spans",
            "max_read_pages_per_action",
            "region_char_budgets",
            "pinned_regions",
        }
        _exact_keys(item, expected, "projection")
        budgets = _mapping(
            item["region_char_budgets"],
            "projection.region_char_budgets",
        )
        return cls(
            global_char_budget=item["global_char_budget"],
            max_selected_spans=item["max_selected_spans"],
            max_read_pages_per_action=item["max_read_pages_per_action"],
            region_char_budgets=tuple(budgets.items()),
            pinned_regions=_strict_string_sequence(
                item["pinned_regions"],
                "projection.pinned_regions",
            ),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "global_char_budget": self.global_char_budget,
            "max_selected_spans": self.max_selected_spans,
            "max_read_pages_per_action": self.max_read_pages_per_action,
            "region_char_budgets": dict(self.region_char_budgets),
            "pinned_regions": list(self.pinned_regions),
        }


@dataclass(frozen=True, slots=True)
class IdleSettings:
    capture_limit_per_quantum: int
    job_limit_per_quantum: int
    lease_ticks: int
    max_quanta_per_tick: int

    def __post_init__(self) -> None:
        for name in (
            "capture_limit_per_quantum",
            "job_limit_per_quantum",
            "lease_ticks",
            "max_quanta_per_tick",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), f"idle.{name}", minimum=1),
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "IdleSettings":
        item = _mapping(value, "idle")
        expected = {
            "capture_limit_per_quantum",
            "job_limit_per_quantum",
            "lease_ticks",
            "max_quanta_per_tick",
        }
        _exact_keys(item, expected, "idle")
        return cls(**{name: item[name] for name in expected})

    def to_canonical_dict(self) -> dict[str, int]:
        return {
            "capture_limit_per_quantum": self.capture_limit_per_quantum,
            "job_limit_per_quantum": self.job_limit_per_quantum,
            "lease_ticks": self.lease_ticks,
            "max_quanta_per_tick": self.max_quanta_per_tick,
        }


@dataclass(frozen=True, slots=True)
class LoopSettings:
    continuous: bool
    tick_interval_ms: int
    idle_sleep_ms: int
    failure_backoff_initial_ms: int
    failure_backoff_max_ms: int
    failure_backoff_multiplier: float
    stop_file_poll_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.continuous, bool) or not self.continuous:
            raise RuntimeConfigError("loop.continuous must be true")
        for name, minimum in (
            ("tick_interval_ms", 0),
            ("idle_sleep_ms", 1),
            ("failure_backoff_initial_ms", 1),
            ("failure_backoff_max_ms", 1),
            ("stop_file_poll_ms", 1),
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), f"loop.{name}", minimum=minimum),
            )
        if self.failure_backoff_initial_ms > self.failure_backoff_max_ms:
            raise RuntimeConfigError(
                "initial failure backoff cannot exceed maximum"
            )
        object.__setattr__(
            self,
            "failure_backoff_multiplier",
            _number(
                self.failure_backoff_multiplier,
                "loop.failure_backoff_multiplier",
                minimum=1.0,
            ),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "LoopSettings":
        item = _mapping(value, "loop")
        expected = {
            "continuous",
            "tick_interval_ms",
            "idle_sleep_ms",
            "failure_backoff_initial_ms",
            "failure_backoff_max_ms",
            "failure_backoff_multiplier",
            "stop_file_poll_ms",
        }
        _exact_keys(item, expected, "loop")
        return cls(**{name: item[name] for name in expected})

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "continuous": self.continuous,
            "tick_interval_ms": self.tick_interval_ms,
            "idle_sleep_ms": self.idle_sleep_ms,
            "failure_backoff_initial_ms": self.failure_backoff_initial_ms,
            "failure_backoff_max_ms": self.failure_backoff_max_ms,
            "failure_backoff_multiplier": self.failure_backoff_multiplier,
            "stop_file_poll_ms": self.stop_file_poll_ms,
        }


@dataclass(frozen=True, slots=True)
class CapabilitySettings:
    typed_tools_enabled: bool
    advisors_enabled: bool

    def __post_init__(self) -> None:
        if self.typed_tools_enabled is not False:
            raise RuntimeConfigError("typed tools must be globally disabled")
        if self.advisors_enabled is not False:
            raise RuntimeConfigError("advisors must be globally disabled")

    @classmethod
    def from_mapping(cls, value: Any) -> "CapabilitySettings":
        item = _mapping(value, "capabilities")
        expected = {"typed_tools_enabled", "advisors_enabled"}
        _exact_keys(item, expected, "capabilities")
        return cls(
            typed_tools_enabled=item["typed_tools_enabled"],
            advisors_enabled=item["advisors_enabled"],
        )

    def to_canonical_dict(self) -> dict[str, bool]:
        return {
            "typed_tools_enabled": self.typed_tools_enabled,
            "advisors_enabled": self.advisors_enabled,
        }


@dataclass(frozen=True, slots=True)
class RuntimeBootstrapConfig:
    models: tuple[ModelPin, ...]
    cores: tuple[LogicalCoreDescriptor, ...]
    ring: RotationRing
    paths: RuntimePaths
    projection: ProjectionSettings
    idle: IdleSettings
    loop: LoopSettings
    capabilities: CapabilitySettings
    config_id: str = field(init=False)

    def __post_init__(self) -> None:
        models = tuple(self.models)
        cores = tuple(self.cores)
        if not models or not all(isinstance(item, ModelPin) for item in models):
            raise RuntimeConfigError("models must contain ModelPin values")
        if not cores or not all(
            isinstance(item, LogicalCoreDescriptor) for item in cores
        ):
            raise RuntimeConfigError(
                "cores must contain LogicalCoreDescriptor values"
            )
        model_ids = [model.model_id for model in models]
        if len(model_ids) != len(set(model_ids)):
            raise RuntimeConfigError("model IDs must be unique")
        for attribute in ("core_id", "soul_id", "adapter_namespace"):
            values = [getattr(core, attribute) for core in cores]
            if len(values) != len(set(values)):
                raise RuntimeConfigError(f"{attribute} values must be unique")
        unknown_models = {core.model_id for core in cores} - set(model_ids)
        if unknown_models:
            raise RuntimeConfigError(
                f"cores reference unknown model IDs: {sorted(unknown_models)}"
            )
        enabled = tuple(core for core in cores if core.enabled)
        if len(enabled) < 3:
            raise RuntimeConfigError(
                "at least three enabled unique logical cores are required"
            )
        for core in enabled:
            if set(core.role_capabilities) != ACTIVE_ROLES:
                raise RuntimeConfigError(
                    "every enabled rotating core must support all active roles"
                )
        enabled_ids = {core.core_id for core in enabled}
        if set(self.ring.core_ids) != enabled_ids or len(
            self.ring.core_ids
        ) != len(enabled_ids):
            raise RuntimeConfigError(
                "round-robin ring must exactly cover enabled cores"
            )
        counts = Counter(core.model_id for core in enabled)
        if any(counts[model_id] < 2 for model_id in model_ids):
            raise RuntimeConfigError(
                "each pinned model requires at least two enabled logical clones"
            )
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "cores", cores)
        object.__setattr__(
            self,
            "config_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_CONFIG_SCHEMA,
            "models": [
                {
                    key: value
                    for key, value in model.to_canonical_dict().items()
                    if key != "schema"
                }
                for model in self.models
            ],
            "cores": [
                {
                    key: value
                    for key, value in core.to_canonical_dict().items()
                    if key != "schema"
                }
                for core in self.cores
            ],
            "ring": self.ring.to_canonical_dict(),
            "paths": self.paths.to_canonical_dict(),
            "projection": self.projection.to_canonical_dict(),
            "idle": self.idle.to_canonical_dict(),
            "loop": self.loop.to_canonical_dict(),
            "capabilities": self.capabilities.to_canonical_dict(),
        }

    def model_pin(self, model_id: str) -> ModelPin:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(model_id)

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeBootstrapConfig":
        item = _mapping(value, "runtime config")
        _reject_literal_secret_fields(item)
        _exact_keys(item, _TOP_LEVEL_KEYS, "runtime config")
        if item["schema"] != RUNTIME_CONFIG_SCHEMA:
            raise RuntimeConfigError("unknown runtime config schema")
        if not isinstance(item["models"], list):
            raise RuntimeConfigError("models must be a list")
        if not isinstance(item["cores"], list):
            raise RuntimeConfigError("cores must be a list")
        return cls(
            models=tuple(ModelPin.from_mapping(model) for model in item["models"]),
            cores=tuple(
                LogicalCoreDescriptor.from_mapping(core)
                for core in item["cores"]
            ),
            ring=RotationRing.from_mapping(item["ring"]),
            paths=RuntimePaths.from_mapping(item["paths"]),
            projection=ProjectionSettings.from_mapping(item["projection"]),
            idle=IdleSettings.from_mapping(item["idle"]),
            loop=LoopSettings.from_mapping(item["loop"]),
            capabilities=CapabilitySettings.from_mapping(
                item["capabilities"]
            ),
        )


def parse_runtime_config(text: str | bytes) -> RuntimeBootstrapConfig:
    """Parse strict JSON, rejecting duplicate keys and non-finite numbers."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeConfigError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeConfigError(f"non-finite JSON constant {token}")
            ),
        )
    except RuntimeConfigError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError("invalid runtime config JSON") from exc
    return RuntimeBootstrapConfig.from_mapping(value)


def load_runtime_config(path: str | Path) -> RuntimeBootstrapConfig:
    """Read one descriptor file; perform no other filesystem or runtime work."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return parse_runtime_config(source.read_bytes())


__all__ = [
    "RUNTIME_CONFIG_SCHEMA",
    "RuntimeConfigError",
    "ModelPin",
    "LogicalCoreDescriptor",
    "RotationRing",
    "RuntimePaths",
    "ProjectionSettings",
    "IdleSettings",
    "LoopSettings",
    "CapabilitySettings",
    "RuntimeBootstrapConfig",
    "parse_runtime_config",
    "load_runtime_config",
]
