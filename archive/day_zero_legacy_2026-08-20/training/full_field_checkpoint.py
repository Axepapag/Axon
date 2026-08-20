"""Fail-closed, tensor-preserving inheritance of trusted char-slot cores.

This migration changes only the external field-view/curriculum contract.  It
does not widen, project, rename, or otherwise transform a learned core tensor.
Source optimizer state is evidence only: the read-bearing phase requires a
fresh optimizer over the returned parameters unless a future, separately
audited named-state migration is supplied.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from cores.core import AxonCore, CoreConfig


FULL_FIELD_INHERITANCE_SCHEMA = "axon_full_field_checkpoint_inheritance_v1"
_TRUSTED_CHECKPOINT_SCHEMA = "axon_charslot_checkpoint_v2"
_WRITER_EXACT_NAMES = {
    "soul_reflect_gate",
    "soul_compartment_gate",
}
_WRITER_PREFIXES = (
    "soul_reflect.",
    "action_norm.",
    "soul_action_norm.",
    "action_to_soul.",
    "soul_router.",
    "soul_router_norm.",
)
_SOUL_METADATA_FIELDS = (
    "active",
    "tier",
    "category",
    "salience",
    "dormant_for",
    "tick_born",
)


class FullFieldCheckpointError(ValueError):
    """The source cannot prove byte-identical view-contract inheritance."""


@dataclass(frozen=True)
class FullFieldCheckpointInheritance:
    """Loaded read-bearing core plus its JSON-safe inheritance proof."""

    core: AxonCore
    manifest: Mapping[str, Any]

    def trainable_named_parameters(
        self,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Yield the exact parameter set for the required fresh optimizer."""

        for name, parameter in self.core.named_parameters():
            if parameter.requires_grad:
                yield name, parameter

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        for _, parameter in self.trainable_named_parameters():
            yield parameter


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    # reshape handles scalar parameters such as ``char_temp`` before viewing
    # their storage as bytes.
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_entry(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.layout != torch.strided:
        raise FullFieldCheckpointError(
            f"core_state tensor {name!r} must use strided layout"
        )
    if tensor.is_floating_point() or tensor.is_complex():
        if not torch.isfinite(tensor).all().item():
            raise FullFieldCheckpointError(
                f"core_state tensor {name!r} contains non-finite values"
            )
    raw = _tensor_bytes(tensor)
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def core_state_tensor_manifest(
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Return a deterministic name/dtype/shape/byte hash for every tensor."""

    if not isinstance(state, Mapping) or not state:
        raise FullFieldCheckpointError("core_state is missing or empty")
    if not all(isinstance(name, str) and name for name in state):
        raise FullFieldCheckpointError("core_state keys must be non-empty strings")

    entries: list[dict[str, Any]] = []
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise FullFieldCheckpointError(
                f"core_state value {name!r} is not a tensor"
            )
        entries.append(_tensor_entry(name, tensor))

    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "tensor_count": len(entries),
        "total_numel": sum(entry["numel"] for entry in entries),
        "total_bytes": sum(entry["nbytes"] for entry in entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "tensors": entries,
    }


def _canonical_core_config(raw: Any) -> CoreConfig:
    if not isinstance(raw, Mapping):
        raise FullFieldCheckpointError("checkpoint cfg is missing or invalid")
    expected_keys = set(CoreConfig().to_dict())
    observed_keys = set(raw)
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    if missing:
        raise FullFieldCheckpointError(
            f"checkpoint cfg is missing fields: {', '.join(missing)}"
        )
    if unexpected:
        raise FullFieldCheckpointError(
            f"checkpoint cfg has unknown fields: {', '.join(unexpected)}"
        )

    d_model = raw.get("d_model")
    if isinstance(d_model, bool) or not isinstance(d_model, Integral):
        raise FullFieldCheckpointError(
            "checkpoint cfg must declare an integer d_model"
        )
    if int(d_model) <= 0:
        raise FullFieldCheckpointError("checkpoint d_model must be positive")

    try:
        cfg = CoreConfig.from_dict(dict(raw))
    except Exception as exc:
        raise FullFieldCheckpointError(
            f"checkpoint cfg cannot construct CoreConfig: {type(exc).__name__}: {exc}"
        ) from exc
    if dict(raw) != cfg.to_dict():
        raise FullFieldCheckpointError(
            "checkpoint cfg is not in canonical CoreConfig form"
        )
    if cfg.char_slot_mode is not True:
        raise FullFieldCheckpointError("checkpoint char_slot_mode must be true")
    if int(cfg.char_slot_max_slots) != 384:
        raise FullFieldCheckpointError(
            "checkpoint char_slot_max_slots must equal 384"
        )
    if int(cfg.char_n_regions) != 3:
        raise FullFieldCheckpointError(
            "checkpoint char_n_regions must equal 3"
        )
    if cfg.soul_mode != "act_reflect_v2":
        raise FullFieldCheckpointError(
            "checkpoint soul_mode must equal act_reflect_v2"
        )
    return cfg


def _source_float_dtype(
    state: Mapping[str, torch.Tensor],
) -> torch.dtype:
    dtypes = {
        tensor.dtype
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    }
    if len(dtypes) != 1:
        rendered = ", ".join(sorted(str(dtype) for dtype in dtypes)) or "none"
        raise FullFieldCheckpointError(
            f"core_state must use one floating dtype, found: {rendered}"
        )
    dtype = next(iter(dtypes))
    if dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise FullFieldCheckpointError(
            f"core_state floating dtype is unsupported: {dtype}"
        )
    return dtype


def _assert_exact_state_layout(
    source: Mapping[str, torch.Tensor],
    fresh: Mapping[str, torch.Tensor],
) -> None:
    source_keys = set(source)
    fresh_keys = set(fresh)
    missing = sorted(fresh_keys - source_keys)
    unexpected = sorted(source_keys - fresh_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise FullFieldCheckpointError(
            "core_state keys do not match fresh AxonCore: " + "; ".join(details)
        )
    for name in sorted(fresh):
        source_tensor = source[name]
        fresh_tensor = fresh[name]
        if not isinstance(source_tensor, torch.Tensor):
            raise FullFieldCheckpointError(
                f"core_state value {name!r} is not a tensor"
            )
        if tuple(source_tensor.shape) != tuple(fresh_tensor.shape):
            raise FullFieldCheckpointError(
                f"core_state shape mismatch for {name}: "
                f"source={tuple(source_tensor.shape)} "
                f"fresh={tuple(fresh_tensor.shape)}"
            )
        if source_tensor.dtype != fresh_tensor.dtype:
            raise FullFieldCheckpointError(
                f"core_state dtype mismatch for {name}: "
                f"source={source_tensor.dtype} fresh={fresh_tensor.dtype}"
            )


def _is_writer_parameter(name: str) -> bool:
    return name in _WRITER_EXACT_NAMES or name.startswith(_WRITER_PREFIXES)


def freeze_unused_core_writer_parameters(core: AxonCore) -> tuple[str, ...]:
    """Freeze the known discarded writer branch without changing tensor bytes."""

    frozen: list[str] = []
    for name, parameter in core.named_parameters():
        if _is_writer_parameter(name):
            parameter.requires_grad_(False)
            frozen.append(name)
    if not frozen:
        raise FullFieldCheckpointError(
            "fresh AxonCore exposes no known writer parameters to freeze"
        )
    remaining = [
        name
        for name, parameter in core.named_parameters()
        if _is_writer_parameter(name) and parameter.requires_grad
    ]
    if remaining:
        raise FullFieldCheckpointError(
            f"writer parameters remain trainable: {remaining}"
        )
    return tuple(sorted(frozen))


def _exact_nonnegative_step(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FullFieldCheckpointError(
            "checkpoint step must be a non-negative integer"
        )
    step = int(value)
    if step < 0:
        raise FullFieldCheckpointError(
            "checkpoint step must be a non-negative integer"
        )
    return step


def _tier_rows_from_cfg(raw: Any, *, label: str, d_model: int) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise FullFieldCheckpointError(f"{label} is present but invalid")
    raw_d_model = raw.get("d_model")
    if (
        isinstance(raw_d_model, bool)
        or not isinstance(raw_d_model, Integral)
        or int(raw_d_model) != d_model
    ):
        raise FullFieldCheckpointError(
            f"{label} d_model does not match core d_model"
        )
    tiers = raw.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise FullFieldCheckpointError(f"{label} tiers are missing or invalid")
    total = 0
    for index, tier in enumerate(tiers):
        if not isinstance(tier, Mapping):
            raise FullFieldCheckpointError(
                f"{label} tier {index} is invalid"
            )
        rows = tier.get("max_rows")
        if isinstance(rows, bool) or not isinstance(rows, Integral) or int(rows) < 0:
            raise FullFieldCheckpointError(
                f"{label} tier {index} max_rows is invalid"
            )
        total += int(rows)
    return total


def _soul_facts(
    payload: Mapping[str, Any],
    *,
    cfg: CoreConfig,
) -> dict[str, Any]:
    state_present = payload.get("soul_state") is not None
    manager_present = payload.get("soul_mgr_state") is not None
    payload_cfg = payload.get("soul_cfg")
    payload_cfg_rows = _tier_rows_from_cfg(
        payload_cfg,
        label="soul_cfg",
        d_model=cfg.d_model,
    )

    external_rows: int | None = None
    external_width: int | None = None
    state_cfg_rows: int | None = None
    if state_present:
        state = payload.get("soul_state")
        if not isinstance(state, Mapping):
            raise FullFieldCheckpointError("soul_state is present but invalid")
        tensor = state.get("tensor")
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise FullFieldCheckpointError(
                "soul_state tensor must have shape (rows, d_model)"
            )
        if not torch.isfinite(tensor).all().item():
            raise FullFieldCheckpointError(
                "soul_state tensor contains non-finite values"
            )
        external_rows, external_width = map(int, tensor.shape)
        if external_width != cfg.d_model:
            raise FullFieldCheckpointError(
                "soul_state tensor width does not match core d_model"
            )
        for field_name in _SOUL_METADATA_FIELDS:
            metadata = state.get(field_name)
            if (
                not isinstance(metadata, torch.Tensor)
                or tuple(metadata.shape) != (external_rows,)
            ):
                raise FullFieldCheckpointError(
                    f"soul_state metadata {field_name} must have shape "
                    f"({external_rows},)"
                )
        state_cfg_rows = _tier_rows_from_cfg(
            state.get("cfg"),
            label="soul_state cfg",
            d_model=cfg.d_model,
        )
        if state_cfg_rows != external_rows:
            raise FullFieldCheckpointError(
                "soul_state cfg row count does not match its tensor"
            )
        if payload_cfg_rows is not None and payload_cfg_rows != external_rows:
            raise FullFieldCheckpointError(
                "soul_cfg row count does not match soul_state tensor"
            )

    core_base_rows = int(cfg.soul_rows)
    core_hot_rows = int(cfg.soul_hot_rows)
    core_total_rows = core_base_rows + core_hot_rows
    external_contract_rows = (
        external_rows if external_rows is not None else payload_cfg_rows
    )
    return {
        "state_present": state_present,
        "manager_state_present": manager_present,
        "payload_config_present": payload_cfg is not None,
        "core_base_rows": core_base_rows,
        "core_hot_rows": core_hot_rows,
        "core_declared_rows": core_total_rows,
        "external_state_rows": external_rows,
        "external_state_d_model": external_width,
        "external_state_config_rows": state_cfg_rows,
        "external_payload_config_rows": payload_cfg_rows,
        "external_contract_rows": external_contract_rows,
        "row_delta_external_minus_core": (
            external_contract_rows - core_total_rows
            if external_contract_rows is not None
            else None
        ),
        "canonical_128_vs_168": {
            "core_rows": 128,
            "external_rows": 168,
            "delta": 40,
            "source_matches": (
                core_total_rows == 128 and external_contract_rows == 168
            ),
        },
        # The read path accepts an external row count. The discarded core
        # writer must remain disabled because its configured 128-row layout
        # does not describe a 168-row temperature-tiered state.
        "external_rows_are_read_context_only": True,
    }


def _optimizer_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("optimizer_state")
    present = value is not None
    state_entries: int | None = None
    parameter_groups: int | None = None
    if present:
        if not isinstance(value, Mapping):
            raise FullFieldCheckpointError(
                "optimizer_state is present but invalid"
            )
        state = value.get("state")
        groups = value.get("param_groups")
        if not isinstance(state, Mapping) or not isinstance(groups, list):
            raise FullFieldCheckpointError(
                "optimizer_state state/param_groups are invalid"
            )
        state_entries = len(state)
        parameter_groups = len(groups)
    return {
        "source_state_present": present,
        "source_state_entries": state_entries,
        "source_parameter_groups": parameter_groups,
        "source_state_inherited": False,
        "policy": "fresh_required",
        "fresh_optimizer_required": True,
        "named_state_migration": None,
        "later_named_state_migration_may_override": True,
    }


def load_full_field_checkpoint(
    source_path: str | Path,
    *,
    expected_d_model: int | None = None,
) -> FullFieldCheckpointInheritance:
    """Strictly inherit one core with zero learned-tensor changes.

    The function is read-only. It does not save a migrated checkpoint or
    manifest; call :func:`save_full_field_inheritance_manifest` explicitly to
    persist the returned JSON-safe proof.
    """

    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FullFieldCheckpointError(
            f"source checkpoint is not a file: {path}"
        )
    source_size = path.stat().st_size
    source_file_sha256 = _file_sha256(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise FullFieldCheckpointError(
            f"source checkpoint cannot be loaded: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FullFieldCheckpointError(
            "source checkpoint payload must be a mapping"
        )

    schema = payload.get("checkpoint_schema")
    if schema not in (None, _TRUSTED_CHECKPOINT_SCHEMA):
        raise FullFieldCheckpointError(
            f"checkpoint schema is not trusted: {schema!r}"
        )
    cfg = _canonical_core_config(payload.get("cfg"))
    if expected_d_model is not None:
        if (
            isinstance(expected_d_model, bool)
            or not isinstance(expected_d_model, Integral)
            or int(expected_d_model) <= 0
        ):
            raise FullFieldCheckpointError(
                "expected_d_model must be a positive integer"
            )
        if cfg.d_model != int(expected_d_model):
            raise FullFieldCheckpointError(
                f"width conversion is forbidden: source d_model={cfg.d_model}, "
                f"requested d_model={int(expected_d_model)}"
            )
    step = _exact_nonnegative_step(payload.get("step"))

    source_state = payload.get("core_state")
    if not isinstance(source_state, Mapping):
        raise FullFieldCheckpointError("core_state is missing or invalid")
    source_state_manifest = core_state_tensor_manifest(source_state)
    source_state_hash_before = source_state_manifest["manifest_sha256"]
    source_dtype = _source_float_dtype(source_state)

    try:
        core = AxonCore(cfg).to(device=torch.device("cpu"), dtype=source_dtype)
    except Exception as exc:
        raise FullFieldCheckpointError(
            f"fresh AxonCore construction failed: {type(exc).__name__}: {exc}"
        ) from exc
    _assert_exact_state_layout(source_state, core.state_dict())
    try:
        incompatible = core.load_state_dict(source_state, strict=True)
    except Exception as exc:
        raise FullFieldCheckpointError(
            f"strict core_state load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FullFieldCheckpointError(
            "strict core_state load reported missing or unexpected keys"
        )

    inherited_state_manifest = core_state_tensor_manifest(core.state_dict())
    source_state_after_load = core_state_tensor_manifest(source_state)
    if (
        source_state_after_load["manifest_sha256"]
        != source_state_hash_before
    ):
        raise FullFieldCheckpointError(
            "strict load mutated source core_state tensors"
        )
    if inherited_state_manifest != source_state_manifest:
        raise FullFieldCheckpointError(
            "inherited core tensors are not byte-identical to source"
        )

    frozen_writer_parameters = freeze_unused_core_writer_parameters(core)
    core.soul_readonly = True
    after_freeze_manifest = core_state_tensor_manifest(core.state_dict())
    if after_freeze_manifest != inherited_state_manifest:
        raise FullFieldCheckpointError(
            "writer freeze changed inherited tensor bytes"
        )
    trainable_parameter_names = sorted(
        name
        for name, parameter in core.named_parameters()
        if parameter.requires_grad
    )

    optimizer = _optimizer_facts(payload)
    soul = _soul_facts(payload, cfg=cfg)
    source_file_sha256_after = _file_sha256(path)
    if (
        source_file_sha256_after != source_file_sha256
        or path.stat().st_size != source_size
    ):
        raise FullFieldCheckpointError(
            "source checkpoint changed during inheritance audit"
        )

    manifest: dict[str, Any] = {
        "schema": FULL_FIELD_INHERITANCE_SCHEMA,
        "passed": True,
        "source": {
            "path": str(path),
            "file_size": source_size,
            "file_sha256": source_file_sha256,
            "checkpoint_schema": (
                schema if schema is not None else "legacy_unversioned"
            ),
            "step": step,
        },
        "migration": {
            "scope": "view_contract_only",
            "curriculum_contract_changes_allowed": True,
            "learned_tensor_changes": False,
            "tensor_renames": False,
            "tensor_projection": False,
            "width_conversion": False,
            "source_d_model": cfg.d_model,
            "inherited_d_model": cfg.d_model,
        },
        "core_config": cfg.to_dict(),
        "core_state": source_state_manifest,
        "inheritance_proof": {
            "fresh_axon_core_instantiated": True,
            "strict_state_load": True,
            "source_state_manifest_sha256": source_state_hash_before,
            "inherited_state_manifest_sha256": inherited_state_manifest[
                "manifest_sha256"
            ],
            "all_tensor_bytes_identical": True,
            "source_payload_unchanged": True,
            "source_file_unchanged": True,
        },
        "read_bearing_phase": {
            "core_soul_readonly": True,
            "writer_policy": "freeze_and_exclude",
            "writer_policy_version": "known_unused_core_writer_v1",
            "frozen_writer_parameter_names": list(frozen_writer_parameters),
            "frozen_writer_parameter_count": len(frozen_writer_parameters),
            "trainable_parameter_names": trainable_parameter_names,
            "trainable_parameter_count": len(trainable_parameter_names),
        },
        "optimizer": optimizer,
        "soul": soul,
        "promotion": {
            "authorized": False,
            "promotable": False,
            "reason": "view-contract inheritance is not behavioral promotion",
        },
    }
    # Prove now that the public manifest has no tensors, Paths, NaNs, or other
    # checkpoint-only values before returning it to callers.
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    return FullFieldCheckpointInheritance(core=core, manifest=manifest)


def save_full_field_inheritance_manifest(
    manifest: Mapping[str, Any],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Explicitly persist a validated inheritance manifest as canonical JSON."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    if manifest.get("schema") != FULL_FIELD_INHERITANCE_SCHEMA:
        raise FullFieldCheckpointError(
            "manifest schema is missing or invalid"
        )
    if manifest.get("passed") is not True:
        raise FullFieldCheckpointError(
            "only a passed inheritance manifest may be saved"
        )
    encoded = json.dumps(
        dict(manifest),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return path
