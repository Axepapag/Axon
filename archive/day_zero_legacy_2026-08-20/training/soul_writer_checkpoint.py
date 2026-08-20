"""Full-state contract for the quarantined differentiable soul-writer pilot.

The contract is intentionally separate from promotable Axon checkpoints.  It
allows only the external writer and the exact soul-read allowlist to train.
Every other inherited core tensor remains byte-identical to its audited source.
Every resume-critical state is explicit: source/current tensor manifests,
per-tensor source-to-current delta hashes, named optimizer coverage, the
authoritative 168-row soul, process RNGs, and curriculum/sampler/tick
continuity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulState, SoulV2Config
from training.differentiable_soul_writer import (
    DifferentiableSoulWriter,
    SoulWriterConfig,
    configure_writer_pilot_parameters,
    writer_pilot_reader_parameter_names,
)
from training.full_field_checkpoint import (
    FULL_FIELD_INHERITANCE_SCHEMA,
    core_state_tensor_manifest,
)
from training.soul_load_bearing import soul_state_sha256, validate_soul_state


SOUL_WRITER_CHECKPOINT_SCHEMA = "axon_soul_writer_pilot_checkpoint_v2"
SOUL_WRITER_CHECKPOINT_VERSION = 2
SOUL_WRITER_TICK_SCHEMA = "axon_soul_writer_tick_state_v1"
SOUL_WRITER_FREEZE_SCHEMA = "axon_soul_writer_freeze_manifest_v2"
SOUL_WRITER_CORE_DELTA_SCHEMA = "axon_soul_writer_core_delta_manifest_v1"
SOUL_WRITER_QUARANTINE_SCHEMA = "axon_soul_writer_quarantine_v1"
_EXPECTED_TIER_LAYOUT = (("hot", 128), ("warm", 32), ("cold", 8))
_OPTIMIZER_CLASS = "torch.optim.AdamW"


class SoulWriterCheckpointError(ValueError):
    """A quarantined writer checkpoint is incomplete or inconsistent."""


@dataclass(frozen=True)
class LoadedSoulWriterCheckpoint:
    core: AxonCore
    writer: DifferentiableSoulWriter
    optimizer: torch.optim.AdamW
    soul_state: SoulState
    curriculum_state: Mapping[str, Any]
    sampler_state: Mapping[str, Any]
    tick_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    source_inheritance_manifest: Mapping[str, Any]
    freeze_manifest: Mapping[str, Any]
    quarantine_provenance: Mapping[str, Any]
    completed: bool
    file_sha256: str

    def restore_rng(self) -> None:
        """Explicitly restore the captured process RNG state."""

        restore_rng_state(self.rng_state)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _update_object_hash(hasher: Any, value: Any) -> None:
    if value is None:
        hasher.update(b"N")
    elif isinstance(value, bool):
        hasher.update(b"B1" if value else b"B0")
    elif isinstance(value, Integral):
        hasher.update(f"I{int(value)};".encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise SoulWriterCheckpointError("state contains a non-finite float")
        hasher.update(f"F{value.hex()};".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(f"S{len(encoded)}:".encode("ascii"))
        hasher.update(encoded)
    elif isinstance(value, bytes):
        hasher.update(f"Y{len(value)}:".encode("ascii"))
        hasher.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise SoulWriterCheckpointError("state tensor contains non-finite values")
        hasher.update(b"T")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(json.dumps(list(tensor.shape)).encode("ascii"))
        hasher.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise SoulWriterCheckpointError("state array contains non-finite values")
        hasher.update(b"A")
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(json.dumps(list(array.shape)).encode("ascii"))
        hasher.update(array.tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b"M{")
        keys = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        for key in keys:
            _update_object_hash(hasher, key)
            _update_object_hash(hasher, value[key])
        hasher.update(b"}")
    elif isinstance(value, tuple):
        hasher.update(b"(")
        for item in value:
            _update_object_hash(hasher, item)
        hasher.update(b")")
    elif isinstance(value, list):
        hasher.update(b"[")
        for item in value:
            _update_object_hash(hasher, item)
        hasher.update(b"]")
    else:
        raise SoulWriterCheckpointError(
            f"state contains unsupported value type {type(value).__name__}"
        )


def deterministic_state_sha256(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_object_hash(hasher, value)
    return hasher.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SoulWriterCheckpointError(
            f"inheritance manifest is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_inheritance(
    manifest: Any,
) -> tuple[dict[str, Any], str]:
    if not isinstance(manifest, Mapping):
        raise SoulWriterCheckpointError(
            "source inheritance manifest is missing or invalid"
        )
    value = copy.deepcopy(dict(manifest))
    if value.get("schema") != FULL_FIELD_INHERITANCE_SCHEMA:
        raise SoulWriterCheckpointError(
            "source inheritance manifest schema is invalid"
        )
    if value.get("passed") is not True:
        raise SoulWriterCheckpointError(
            "source inheritance manifest did not pass"
        )
    migration = value.get("migration")
    proof = value.get("inheritance_proof")
    promotion = value.get("promotion")
    core_state = value.get("core_state")
    source = value.get("source")
    if (
        not isinstance(migration, Mapping)
        or migration.get("scope") != "view_contract_only"
        or migration.get("learned_tensor_changes") is not False
        or migration.get("width_conversion") is not False
    ):
        raise SoulWriterCheckpointError(
            "source inheritance is not a no-tensor-change view contract"
        )
    if (
        not isinstance(proof, Mapping)
        or proof.get("strict_state_load") is not True
        or proof.get("all_tensor_bytes_identical") is not True
        or proof.get("source_file_unchanged") is not True
    ):
        raise SoulWriterCheckpointError(
            "source inheritance byte-identity proof is incomplete"
        )
    if (
        not isinstance(promotion, Mapping)
        or promotion.get("authorized") is not False
        or promotion.get("promotable") is not False
    ):
        raise SoulWriterCheckpointError(
            "source inheritance promotion quarantine is invalid"
        )
    if (
        not isinstance(core_state, Mapping)
        or not _valid_sha256(core_state.get("manifest_sha256"))
    ):
        raise SoulWriterCheckpointError(
            "source inheritance core-state hash is missing"
        )
    if (
        not isinstance(source, Mapping)
        or not _valid_sha256(source.get("file_sha256"))
    ):
        raise SoulWriterCheckpointError(
            "source inheritance file hash is missing"
        )
    core_hash = core_state["manifest_sha256"]
    if (
        proof.get("source_state_manifest_sha256") != core_hash
        or proof.get("inherited_state_manifest_sha256") != core_hash
    ):
        raise SoulWriterCheckpointError(
            "source inheritance tensor proof does not match core-state hash"
        )
    return value, _canonical_json_sha256(value)


def _module_state_manifest(
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> dict[str, Any]:
    try:
        return core_state_tensor_manifest(state)
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"{label} state manifest failed: {type(exc).__name__}: {exc}"
        ) from exc


def _single_float_dtype(
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> torch.dtype:
    dtypes = {
        tensor.dtype
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
    }
    if len(dtypes) != 1:
        rendered = ", ".join(sorted(str(dtype) for dtype in dtypes)) or "none"
        raise SoulWriterCheckpointError(
            f"{label} state must use one floating dtype, found {rendered}"
        )
    return next(iter(dtypes))


def _core_config_from_payload(value: Any) -> CoreConfig:
    if not isinstance(value, Mapping):
        raise SoulWriterCheckpointError("core config is missing or invalid")
    try:
        cfg = CoreConfig.from_dict(dict(value))
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"core config is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    if dict(value) != cfg.to_dict():
        raise SoulWriterCheckpointError("core config is not canonical")
    if (
        not cfg.char_slot_mode
        or cfg.char_slot_max_slots != 384
        or cfg.char_n_regions != 3
        or cfg.soul_mode != "act_reflect_v2"
    ):
        raise SoulWriterCheckpointError(
            "core config is not the inherited 384-slot act_reflect_v2 contract"
        )
    return cfg


def _writer_config_from_payload(value: Any, *, d_model: int) -> SoulWriterConfig:
    if not isinstance(value, Mapping):
        raise SoulWriterCheckpointError("writer config is missing or invalid")
    expected = {
        "d_model",
        "hot_rows",
        "warm_rows",
        "cold_rows",
        "gate_init",
        "total_rows",
    }
    if set(value) != expected:
        raise SoulWriterCheckpointError(
            "writer config fields do not match the pilot schema"
        )
    if value.get("total_rows") != 168:
        raise SoulWriterCheckpointError("writer config must declare 168 rows")
    try:
        cfg = SoulWriterConfig(
            d_model=int(value["d_model"]),
            hot_rows=int(value["hot_rows"]),
            warm_rows=int(value["warm_rows"]),
            cold_rows=int(value["cold_rows"]),
            gate_init=float(value["gate_init"]),
        )
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"writer config is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    if cfg.d_model != d_model or cfg.to_dict() != dict(value):
        raise SoulWriterCheckpointError(
            "writer config does not match inherited core d_model"
        )
    return cfg


def _assert_state_layout(
    source: Mapping[str, torch.Tensor],
    fresh: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(source) != set(fresh):
        missing = sorted(set(fresh) - set(source))
        unexpected = sorted(set(source) - set(fresh))
        raise SoulWriterCheckpointError(
            f"{label} state keys mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for name in sorted(fresh):
        tensor = source[name]
        if not isinstance(tensor, torch.Tensor):
            raise SoulWriterCheckpointError(
                f"{label} state {name!r} is not a tensor"
            )
        if tuple(tensor.shape) != tuple(fresh[name].shape):
            raise SoulWriterCheckpointError(
                f"{label} state shape mismatch for {name}"
            )
        if tensor.dtype != fresh[name].dtype:
            raise SoulWriterCheckpointError(
                f"{label} state dtype mismatch for {name}"
            )


def _manifest_entries_by_name(
    manifest: Any,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        raise SoulWriterCheckpointError(f"{label} tensor manifest is missing")
    entries = manifest.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise SoulWriterCheckpointError(f"{label} tensor manifest entries are invalid")
    by_name: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise SoulWriterCheckpointError(
                f"{label} tensor manifest entry is invalid"
            )
        entry = dict(raw_entry)
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise SoulWriterCheckpointError(
                f"{label} tensor manifest names are invalid"
            )
        if not _valid_sha256(entry.get("sha256")):
            raise SoulWriterCheckpointError(
                f"{label} tensor manifest hash is invalid for {name}"
            )
        by_name[name] = entry
    if (
        manifest.get("tensor_count") != len(by_name)
        or not _valid_sha256(manifest.get("manifest_sha256"))
    ):
        raise SoulWriterCheckpointError(f"{label} tensor manifest summary is invalid")
    if [entry["name"] for entry in entries] != sorted(by_name):
        raise SoulWriterCheckpointError(
            f"{label} tensor manifest entries are not sorted"
        )
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest["manifest_sha256"]:
        raise SoulWriterCheckpointError(
            f"{label} tensor manifest hash does not match its entries"
        )
    return by_name


def _core_delta_manifest(
    *,
    source_manifest: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
    reader_names: Sequence[str],
) -> dict[str, Any]:
    source = _manifest_entries_by_name(source_manifest, label="source core")
    current = _manifest_entries_by_name(current_manifest, label="current core")
    if set(source) != set(current):
        missing = sorted(set(source) - set(current))
        unexpected = sorted(set(current) - set(source))
        raise SoulWriterCheckpointError(
            "current core tensor names differ from source inheritance: "
            f"missing={missing}, unexpected={unexpected}"
        )

    reader_state_names: set[str] = set()
    for full_name in reader_names:
        if (
            not isinstance(full_name, str)
            or not full_name.startswith("core.")
            or len(full_name) <= len("core.")
        ):
            raise SoulWriterCheckpointError(
                "reader allowlist contains an invalid core parameter name"
            )
        reader_state_names.add(full_name[len("core.") :])
    if len(reader_state_names) != len(reader_names) or not reader_state_names:
        raise SoulWriterCheckpointError("reader allowlist is empty or duplicated")
    unknown_reader = sorted(reader_state_names - set(current))
    if unknown_reader:
        raise SoulWriterCheckpointError(
            f"reader allowlist names are absent from core state: {unknown_reader}"
        )

    entries: list[dict[str, Any]] = []
    changed_reader_names: list[str] = []
    byte_locked_names: list[str] = []
    for name in sorted(current):
        source_entry = source[name]
        current_entry = current[name]
        for field in ("dtype", "shape", "numel", "nbytes"):
            if source_entry.get(field) != current_entry.get(field):
                raise SoulWriterCheckpointError(
                    f"current core tensor layout changed from source for {name}"
                )
        source_sha256 = source_entry["sha256"]
        current_sha256 = current_entry["sha256"]
        changed = source_sha256 != current_sha256
        allowlisted_reader = name in reader_state_names
        if changed and not allowlisted_reader:
            raise SoulWriterCheckpointError(
                "non-allowlisted core tensor changed from source inheritance: "
                f"{name}"
            )
        if allowlisted_reader:
            if changed:
                changed_reader_names.append(name)
            classification = "allowlisted_reader"
        else:
            byte_locked_names.append(name)
            classification = "byte_locked"
        delta_sha256 = hashlib.sha256(
            (
                f"{name}\0{source_sha256.lower()}\0"
                f"{current_sha256.lower()}"
            ).encode("ascii")
        ).hexdigest()
        entries.append(
            {
                "name": name,
                "classification": classification,
                "source_sha256": source_sha256,
                "current_sha256": current_sha256,
                "delta_sha256": delta_sha256,
                "changed": changed,
            }
        )

    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": SOUL_WRITER_CORE_DELTA_SCHEMA,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "current_manifest_sha256": current_manifest["manifest_sha256"],
        "reader_state_names": sorted(reader_state_names),
        "changed_reader_state_names": changed_reader_names,
        "byte_locked_state_names": byte_locked_names,
        "reader_tensor_count": len(reader_state_names),
        "changed_reader_tensor_count": len(changed_reader_names),
        "byte_locked_tensor_count": len(byte_locked_names),
        "all_non_reader_tensors_byte_identical": True,
        "delta_hash_contract": (
            "sha256(name\\0source_tensor_sha256\\0current_tensor_sha256)"
        ),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "tensors": entries,
    }


def _derive_freeze_manifest(
    core: AxonCore,
    writer: DifferentiableSoulWriter,
) -> dict[str, Any]:
    core_named = list(core.named_parameters())
    writer_named = list(writer.named_parameters())
    expected_reader_names = set(writer_pilot_reader_parameter_names(core))
    observed_reader_names = {
        f"core.{name}" for name, parameter in core_named if parameter.requires_grad
    }
    missing_reader = sorted(expected_reader_names - observed_reader_names)
    unexpected_reader = sorted(observed_reader_names - expected_reader_names)
    if missing_reader or unexpected_reader:
        raise SoulWriterCheckpointError(
            "core trainable parameters differ from the exact reader allowlist: "
            f"missing={missing_reader}, unexpected={unexpected_reader}"
        )
    writer_frozen = sorted(
        f"writer.{name}" for name, parameter in writer_named
        if not parameter.requires_grad
    )
    if writer_frozen:
        raise SoulWriterCheckpointError(
            "reader-writer pilot has frozen external writer parameters: "
            f"{writer_frozen}"
        )
    all_core_names = {f"core.{name}" for name, _ in core_named}
    writer_names = sorted(f"writer.{name}" for name, _ in writer_named)
    reader_names = sorted(expected_reader_names)
    frozen_names = sorted(all_core_names - expected_reader_names)
    trainable_names = sorted(expected_reader_names | set(writer_names))
    if not frozen_names or not trainable_names:
        raise SoulWriterCheckpointError(
            "freeze manifest has an empty core or writer parameter set"
        )
    return {
        "schema": SOUL_WRITER_FREEZE_SCHEMA,
        "policy": "allowlisted_soul_reader_plus_external_writer",
        "core_soul_readonly": bool(core.soul_readonly),
        "core_state_lock_scope": "all_non_reader_tensors",
        "reader_names": reader_names,
        "writer_names": writer_names,
        "frozen_names": frozen_names,
        "trainable_names": trainable_names,
        "reader_count": len(reader_names),
        "writer_count": len(writer_names),
        "frozen_count": len(frozen_names),
        "trainable_count": len(trainable_names),
    }


def _apply_allowlisted_reader_freeze(
    core: AxonCore,
    writer: DifferentiableSoulWriter,
) -> dict[str, Any]:
    policy = configure_writer_pilot_parameters(
        core,
        writer,
        train_read_path=True,
    )
    manifest = _derive_freeze_manifest(core, writer)
    if tuple(manifest["trainable_names"]) != policy.trainable_names:
        raise SoulWriterCheckpointError(
            "configured trainable policy differs from freeze manifest"
        )
    return manifest


def _named_parameters(
    core: AxonCore,
    writer: DifferentiableSoulWriter,
) -> dict[str, torch.nn.Parameter]:
    return {
        **{
            f"core.{name}": parameter
            for name, parameter in core.named_parameters()
        },
        **{
            f"writer.{name}": parameter
            for name, parameter in writer.named_parameters()
        },
    }


def _optimizer_envelope(
    optimizer: torch.optim.Optimizer,
    *,
    named_parameters: Mapping[str, torch.nn.Parameter],
    expected_trainable_names: Sequence[str],
) -> dict[str, Any]:
    if type(optimizer) is not torch.optim.AdamW:
        raise SoulWriterCheckpointError(
            "writer pilot optimizer must be exactly torch.optim.AdamW"
        )
    expected = list(expected_trainable_names)
    if len(set(expected)) != len(expected) or not expected:
        raise SoulWriterCheckpointError(
            "expected trainable parameter names are invalid"
        )
    param_name_by_id = {
        id(parameter): name
        for name, parameter in named_parameters.items()
    }
    state_dict = optimizer.state_dict()
    saved_groups = state_dict.get("param_groups")
    saved_state = state_dict.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, Mapping):
        raise SoulWriterCheckpointError("optimizer state_dict is invalid")
    if len(saved_groups) != len(optimizer.param_groups) or not saved_groups:
        raise SoulWriterCheckpointError("optimizer parameter groups are invalid")

    entries: list[dict[str, Any]] = []
    seen_names: list[str] = []
    seen_state_ids: set[Any] = set()
    for group_index, (live_group, saved_group) in enumerate(
        zip(optimizer.param_groups, saved_groups, strict=True)
    ):
        live_params = live_group.get("params")
        state_ids = saved_group.get("params")
        if (
            not isinstance(live_params, list)
            or not isinstance(state_ids, list)
            or len(live_params) != len(state_ids)
        ):
            raise SoulWriterCheckpointError(
                f"optimizer group {group_index} parameter mapping is invalid"
            )
        for group_position, (parameter, state_id) in enumerate(
            zip(live_params, state_ids, strict=True)
        ):
            name = param_name_by_id.get(id(parameter))
            if name is None:
                raise SoulWriterCheckpointError(
                    "optimizer contains an undeclared parameter"
                )
            if not parameter.requires_grad:
                raise SoulWriterCheckpointError(
                    f"optimizer contains frozen parameter {name}"
                )
            if name in seen_names or state_id in seen_state_ids:
                raise SoulWriterCheckpointError(
                    "optimizer named mapping contains duplicates"
                )
            state_value = saved_state.get(state_id)
            if not isinstance(state_value, Mapping) or not state_value:
                raise SoulWriterCheckpointError(
                    f"optimizer state is missing for trainable parameter {name}"
                )
            required_state_names = {"step", "exp_avg", "exp_avg_sq"}
            if bool(live_group.get("amsgrad", False)):
                required_state_names.add("max_exp_avg_sq")
            if set(state_value) != required_state_names:
                raise SoulWriterCheckpointError(
                    f"optimizer AdamW state fields are incomplete for {name}"
                )
            for state_name, state_tensor in state_value.items():
                if isinstance(state_tensor, torch.Tensor):
                    if state_tensor.is_floating_point() and not torch.isfinite(
                        state_tensor
                    ).all().item():
                        raise SoulWriterCheckpointError(
                            f"optimizer state {name}.{state_name} is non-finite"
                        )
                    if state_tensor.numel() != 1 and tuple(state_tensor.shape) != tuple(
                        parameter.shape
                    ):
                        raise SoulWriterCheckpointError(
                            f"optimizer state shape mismatch for {name}.{state_name}"
                        )
            seen_names.append(name)
            seen_state_ids.add(state_id)
            entries.append(
                {
                    "name": name,
                    "group_index": group_index,
                    "group_position": group_position,
                    "state_id": state_id,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "state_sha256": deterministic_state_sha256(state_value),
                }
            )

    if sorted(seen_names) != sorted(expected):
        missing = sorted(set(expected) - set(seen_names))
        unexpected = sorted(set(seen_names) - set(expected))
        raise SoulWriterCheckpointError(
            f"optimizer named coverage mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if set(saved_state) != seen_state_ids:
        raise SoulWriterCheckpointError(
            "optimizer state contains unnamed or stale entries"
        )
    state_copy = copy.deepcopy(state_dict)
    return {
        "class": _OPTIMIZER_CLASS,
        "state_dict": state_copy,
        "state_dict_sha256": deterministic_state_sha256(state_copy),
        "named_parameter_map": entries,
        "coverage": {
            "declared_trainable": len(expected),
            "mapped": len(entries),
            "state_present": len(seen_state_ids),
            "fraction": 1.0,
            "complete": True,
        },
    }


def _validate_optimizer_envelope(
    envelope: Any,
    *,
    core: AxonCore,
    writer: DifferentiableSoulWriter,
    freeze_manifest: Mapping[str, Any],
) -> torch.optim.AdamW:
    if not isinstance(envelope, Mapping):
        raise SoulWriterCheckpointError("optimizer envelope is missing")
    if envelope.get("class") != _OPTIMIZER_CLASS:
        raise SoulWriterCheckpointError("optimizer class is invalid")
    state_dict = envelope.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise SoulWriterCheckpointError("optimizer state_dict is missing")
    if envelope.get("state_dict_sha256") != deterministic_state_sha256(state_dict):
        raise SoulWriterCheckpointError("optimizer state_dict hash mismatch")
    entries = envelope.get("named_parameter_map")
    coverage = envelope.get("coverage")
    if not isinstance(entries, list) or not isinstance(coverage, Mapping):
        raise SoulWriterCheckpointError(
            "optimizer named mapping or coverage is missing"
        )
    expected_names = list(freeze_manifest["trainable_names"])
    if (
        coverage.get("complete") is not True
        or coverage.get("fraction") != 1.0
        or coverage.get("declared_trainable") != len(expected_names)
        or coverage.get("mapped") != len(expected_names)
        or coverage.get("state_present") != len(expected_names)
    ):
        raise SoulWriterCheckpointError("optimizer coverage is incomplete")

    named = _named_parameters(core, writer)
    seen_names: set[str] = set()
    grouped_entries: dict[int, list[Mapping[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SoulWriterCheckpointError("optimizer named entry is invalid")
        name = entry.get("name")
        if not isinstance(name, str) or name not in named or name in seen_names:
            raise SoulWriterCheckpointError(
                "optimizer named entry has an invalid parameter name"
            )
        parameter = named[name]
        if not parameter.requires_grad:
            raise SoulWriterCheckpointError(
                f"optimizer named entry references frozen parameter {name}"
            )
        if entry.get("shape") != list(parameter.shape):
            raise SoulWriterCheckpointError(
                f"optimizer parameter shape mismatch for {name}"
            )
        if entry.get("dtype") != str(parameter.dtype):
            raise SoulWriterCheckpointError(
                f"optimizer parameter dtype mismatch for {name}"
            )
        group_index = entry.get("group_index")
        group_position = entry.get("group_position")
        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, Integral)
            or isinstance(group_position, bool)
            or not isinstance(group_position, Integral)
            or int(group_index) < 0
            or int(group_position) < 0
        ):
            raise SoulWriterCheckpointError(
                "optimizer group coordinates are invalid"
            )
        grouped_entries.setdefault(int(group_index), []).append(entry)
        seen_names.add(name)
    if seen_names != set(expected_names):
        raise SoulWriterCheckpointError(
            "optimizer named mapping does not cover the freeze manifest"
        )

    saved_groups = state_dict.get("param_groups")
    saved_state = state_dict.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, Mapping):
        raise SoulWriterCheckpointError("optimizer serialized groups are invalid")
    if set(grouped_entries) != set(range(len(saved_groups))):
        raise SoulWriterCheckpointError("optimizer group coverage is invalid")

    live_groups: list[dict[str, Any]] = []
    for group_index, saved_group in enumerate(saved_groups):
        if not isinstance(saved_group, Mapping):
            raise SoulWriterCheckpointError(
                f"optimizer group {group_index} is invalid"
            )
        ordered = sorted(
            grouped_entries[group_index],
            key=lambda entry: int(entry["group_position"]),
        )
        if [int(entry["group_position"]) for entry in ordered] != list(
            range(len(ordered))
        ):
            raise SoulWriterCheckpointError(
                f"optimizer group {group_index} positions are incomplete"
            )
        saved_ids = saved_group.get("params")
        entry_ids = [entry["state_id"] for entry in ordered]
        if not isinstance(saved_ids, list) or entry_ids != saved_ids:
            raise SoulWriterCheckpointError(
                f"optimizer group {group_index} state IDs do not match"
            )
        for entry in ordered:
            state_value = saved_state.get(entry["state_id"])
            if not isinstance(state_value, Mapping) or not state_value:
                raise SoulWriterCheckpointError(
                    f"optimizer state is missing for {entry['name']}"
                )
            if entry.get("state_sha256") != deterministic_state_sha256(state_value):
                raise SoulWriterCheckpointError(
                    f"optimizer named state hash mismatch for {entry['name']}"
                )
        options = {
            key: copy.deepcopy(value)
            for key, value in saved_group.items()
            if key != "params"
        }
        live_groups.append(
            {
                **options,
                "params": [named[entry["name"]] for entry in ordered],
            }
        )
    try:
        optimizer = torch.optim.AdamW(live_groups)
        optimizer.load_state_dict(copy.deepcopy(dict(state_dict)))
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"optimizer strict restore failed: {type(exc).__name__}: {exc}"
        ) from exc

    rebuilt = _optimizer_envelope(
        optimizer,
        named_parameters=named,
        expected_trainable_names=expected_names,
    )
    if (
        rebuilt["class"] != envelope.get("class")
        or rebuilt["state_dict_sha256"] != envelope.get("state_dict_sha256")
        or rebuilt["named_parameter_map"] != entries
        or rebuilt["coverage"] != dict(coverage)
    ):
        raise SoulWriterCheckpointError(
            "optimizer named state changed during strict restore"
        )
    return optimizer


def _authoritative_soul_state(state: SoulState, *, d_model: int) -> None:
    try:
        validate_soul_state(state)
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"soul state is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    layout = tuple(
        (tier.name, int(tier.max_rows))
        for tier in state.cfg.tiers
    )
    if layout != _EXPECTED_TIER_LAYOUT:
        raise SoulWriterCheckpointError(
            "soul state must use exact 128-hot/32-warm/8-cold tiers"
        )
    if state.cfg.d_model != d_model or tuple(state.tensor.shape) != (168, d_model):
        raise SoulWriterCheckpointError(
            "soul state must have exact shape (168, d_model)"
        )
    if not state.active[:128].any().item():
        raise SoulWriterCheckpointError(
            "soul state must expose at least one active hot row"
        )


def _soul_payload(state: SoulState) -> dict[str, Any]:
    return {
        "tensor": state.tensor.detach().cpu().clone(),
        "active": state.active.detach().cpu().clone(),
        "tier": state.tier.detach().cpu().clone(),
        "category": state.category.detach().cpu().clone(),
        "salience": state.salience.detach().cpu().clone(),
        "dormant_for": state.dormant_for.detach().cpu().clone(),
        "tick_born": state.tick_born.detach().cpu().clone(),
        "current_tick": int(state.current_tick),
        "cfg": copy.deepcopy(state.cfg.to_dict()),
    }


def _soul_envelope(state: SoulState, *, d_model: int) -> dict[str, Any]:
    _authoritative_soul_state(state, d_model=d_model)
    before = soul_state_sha256(state)
    payload = _soul_payload(state)
    clone = SoulState.from_saveable(
        copy.deepcopy(payload),
        torch.device("cpu"),
        payload["tensor"].dtype,
    )
    after = soul_state_sha256(state)
    cloned_hash = soul_state_sha256(clone)
    if before != after or cloned_hash != before:
        raise SoulWriterCheckpointError(
            "soul state snapshot was not immutable and byte-identical"
        )
    return {
        "layout": {
            "hot_rows": 128,
            "warm_rows": 32,
            "cold_rows": 8,
            "total_rows": 168,
            "d_model": d_model,
        },
        "immutable": True,
        "sha256": before,
        "payload": payload,
    }


def _load_soul_envelope(
    envelope: Any,
    *,
    d_model: int,
) -> SoulState:
    if not isinstance(envelope, Mapping):
        raise SoulWriterCheckpointError("soul envelope is missing")
    expected_layout = {
        "hot_rows": 128,
        "warm_rows": 32,
        "cold_rows": 8,
        "total_rows": 168,
        "d_model": d_model,
    }
    if envelope.get("layout") != expected_layout:
        raise SoulWriterCheckpointError("soul envelope layout is invalid")
    if envelope.get("immutable") is not True:
        raise SoulWriterCheckpointError("soul envelope is not immutable")
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise SoulWriterCheckpointError("soul payload is missing")
    try:
        tensor = payload.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor is missing")
        state = SoulState.from_saveable(
            copy.deepcopy(dict(payload)),
            torch.device("cpu"),
            tensor.dtype,
        )
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"soul payload cannot be restored: {type(exc).__name__}: {exc}"
        ) from exc
    _authoritative_soul_state(state, d_model=d_model)
    observed_hash = soul_state_sha256(state)
    if not _valid_sha256(envelope.get("sha256")) or observed_hash != envelope.get(
        "sha256"
    ):
        raise SoulWriterCheckpointError("soul state hash mismatch")
    return state


def capture_rng_state() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    state = {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": (
            [value.clone().cpu() for value in torch.cuda.get_rng_state_all()]
            if cuda_available
            else []
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_count,
    }
    _validate_rng_state(state)
    return {
        "state": state,
        "sha256": deterministic_state_sha256(state),
    }


def _validate_rng_state(envelope: Any) -> Mapping[str, Any]:
    if not isinstance(envelope, Mapping):
        raise SoulWriterCheckpointError("RNG envelope is missing")
    state = envelope.get("state") if "state" in envelope else envelope
    if not isinstance(state, Mapping):
        raise SoulWriterCheckpointError("RNG state is missing")
    required = {
        "python",
        "numpy",
        "torch",
        "cuda",
        "cuda_available",
        "cuda_device_count",
    }
    if set(state) != required:
        raise SoulWriterCheckpointError("RNG state fields are incomplete")
    try:
        random.Random().setstate(copy.deepcopy(state["python"]))
        np.random.RandomState().set_state(copy.deepcopy(state["numpy"]))
        generator = torch.Generator(device="cpu")
        generator.set_state(state["torch"].detach().cpu().clone())
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"CPU RNG state is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        not isinstance(state["torch"], torch.Tensor)
        or state["torch"].dtype != torch.uint8
        or state["torch"].numel() == 0
    ):
        raise SoulWriterCheckpointError("Torch RNG state is invalid")
    cuda_available = state["cuda_available"]
    cuda_count = state["cuda_device_count"]
    cuda_states = state["cuda"]
    if not isinstance(cuda_available, bool):
        raise SoulWriterCheckpointError("CUDA RNG availability flag is invalid")
    if (
        isinstance(cuda_count, bool)
        or not isinstance(cuda_count, Integral)
        or int(cuda_count) < 0
        or not isinstance(cuda_states, list)
    ):
        raise SoulWriterCheckpointError("CUDA RNG metadata is invalid")
    if cuda_available:
        if int(cuda_count) <= 0 or len(cuda_states) != int(cuda_count):
            raise SoulWriterCheckpointError("CUDA RNG state count is invalid")
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.numel() == 0
            for value in cuda_states
        ):
            raise SoulWriterCheckpointError("CUDA RNG state tensor is invalid")
    elif int(cuda_count) != 0 or cuda_states:
        raise SoulWriterCheckpointError(
            "CUDA RNG state must be explicitly empty when unavailable"
        )
    if "sha256" in envelope:
        if envelope.get("sha256") != deterministic_state_sha256(state):
            raise SoulWriterCheckpointError("RNG state hash mismatch")
    return state


def restore_rng_state(envelope: Mapping[str, Any]) -> None:
    state = _validate_rng_state(envelope)
    if state["cuda_available"] and (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != int(state["cuda_device_count"])
    ):
        raise SoulWriterCheckpointError(
            "local CUDA topology cannot restore checkpoint RNG state"
        )
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch"].detach().cpu().clone())
    if state["cuda_available"]:
        torch.cuda.set_rng_state_all(
            [value.detach().cpu().clone() for value in state["cuda"]]
        )


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise SoulWriterCheckpointError(f"{label} must be a non-negative integer")
    return int(value)


def _validate_continuity_states(
    curriculum: Any,
    sampler: Any,
    tick: Any,
    *,
    soul_tick: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(curriculum, Mapping):
        raise SoulWriterCheckpointError("curriculum state is missing")
    curriculum_copy = copy.deepcopy(dict(curriculum))
    if not isinstance(curriculum_copy.get("schema"), str) or not curriculum_copy[
        "schema"
    ]:
        raise SoulWriterCheckpointError("curriculum schema is missing")
    if not _valid_sha256(curriculum_copy.get("manifest_sha256")):
        raise SoulWriterCheckpointError(
            "curriculum manifest_sha256 is missing or invalid"
        )
    example_count = _exact_nonnegative_int(
        curriculum_copy.get("example_count"),
        label="curriculum example_count",
    )
    if example_count <= 0:
        raise SoulWriterCheckpointError(
            "curriculum example_count must be positive"
        )
    families = curriculum_copy.get("families")
    if (
        not isinstance(families, list)
        or not families
        or not all(isinstance(family, str) and family for family in families)
        or len(set(families)) != len(families)
    ):
        raise SoulWriterCheckpointError("curriculum families are invalid")

    if not isinstance(sampler, Mapping):
        raise SoulWriterCheckpointError("sampler state is missing")
    sampler_copy = copy.deepcopy(dict(sampler))
    if not isinstance(sampler_copy.get("schema"), str) or not sampler_copy["schema"]:
        raise SoulWriterCheckpointError("sampler schema is missing")
    position = _exact_nonnegative_int(
        sampler_copy.get("position"),
        label="sampler position",
    )
    order = sampler_copy.get("order")
    if (
        not isinstance(order, list)
        or any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in order
        )
        or sorted(int(index) for index in order) != list(range(example_count))
    ):
        raise SoulWriterCheckpointError(
            "sampler order is not a complete curriculum permutation"
        )
    if position > example_count:
        raise SoulWriterCheckpointError("sampler position is out of bounds")
    if sampler_copy.get("rng_state") is None:
        raise SoulWriterCheckpointError("sampler rng_state is missing")
    if "families" in sampler_copy and sampler_copy["families"] != families:
        raise SoulWriterCheckpointError(
            "sampler families do not match curriculum families"
        )

    if not isinstance(tick, Mapping):
        raise SoulWriterCheckpointError("tick state is missing")
    tick_copy = copy.deepcopy(dict(tick))
    if tick_copy.get("schema") != SOUL_WRITER_TICK_SCHEMA:
        raise SoulWriterCheckpointError("tick state schema is invalid")
    for key in ("global_step", "soul_tick", "write_events", "recall_events"):
        _exact_nonnegative_int(tick_copy.get(key), label=f"tick {key}")
    if int(tick_copy["soul_tick"]) != int(soul_tick):
        raise SoulWriterCheckpointError(
            "tick soul_tick does not match immutable SoulState"
        )
    if tick_copy.get("phase") not in {
        "write",
        "delay",
        "recall",
        "boundary",
    }:
        raise SoulWriterCheckpointError("tick phase is invalid")
    return curriculum_copy, sampler_copy, tick_copy


def _continuity_envelope(
    curriculum: Mapping[str, Any],
    sampler: Mapping[str, Any],
    tick: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "curriculum": {
            "state": curriculum,
            "sha256": deterministic_state_sha256(curriculum),
        },
        "sampler": {
            "state": sampler,
            "sha256": deterministic_state_sha256(sampler),
        },
        "tick": {
            "state": tick,
            "sha256": deterministic_state_sha256(tick),
        },
    }


def _load_continuity_envelope(
    envelope: Any,
    *,
    soul_tick: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "curriculum",
        "sampler",
        "tick",
    }:
        raise SoulWriterCheckpointError("continuity envelope is incomplete")
    states: list[Any] = []
    for label in ("curriculum", "sampler", "tick"):
        item = envelope[label]
        if (
            not isinstance(item, Mapping)
            or "state" not in item
            or item.get("sha256") != deterministic_state_sha256(item["state"])
        ):
            raise SoulWriterCheckpointError(
                f"{label} continuity state hash mismatch"
            )
        states.append(item["state"])
    return _validate_continuity_states(
        states[0],
        states[1],
        states[2],
        soul_tick=soul_tick,
    )


def _quarantine_provenance(
    *,
    pilot_id: str,
    reason: str,
    inheritance_sha256: str,
) -> dict[str, Any]:
    if not isinstance(pilot_id, str) or not pilot_id.strip():
        raise SoulWriterCheckpointError("pilot_id must be non-empty")
    if not isinstance(reason, str) or not reason.strip():
        raise SoulWriterCheckpointError("quarantine reason must be non-empty")
    return {
        "schema": SOUL_WRITER_QUARANTINE_SCHEMA,
        "pilot_id": pilot_id,
        "reason": reason,
        "quarantined": True,
        "promotable": False,
        "promotion_authorized": False,
        "source_inheritance_manifest_sha256": inheritance_sha256,
    }


def _validate_quarantine(
    value: Any,
    *,
    inheritance_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SoulWriterCheckpointError("quarantine provenance is missing")
    provenance = copy.deepcopy(dict(value))
    if (
        provenance.get("schema") != SOUL_WRITER_QUARANTINE_SCHEMA
        or not isinstance(provenance.get("pilot_id"), str)
        or not provenance["pilot_id"]
        or not isinstance(provenance.get("reason"), str)
        or not provenance["reason"]
        or provenance.get("quarantined") is not True
        or provenance.get("promotable") is not False
        or provenance.get("promotion_authorized") is not False
        or provenance.get("source_inheritance_manifest_sha256")
        != inheritance_sha256
    ):
        raise SoulWriterCheckpointError(
            "quarantine provenance is invalid or promotable"
        )
    return provenance


def build_soul_writer_checkpoint(
    *,
    core: AxonCore,
    writer: DifferentiableSoulWriter,
    optimizer: torch.optim.Optimizer,
    soul_state: SoulState,
    source_inheritance_manifest: Mapping[str, Any],
    curriculum_state: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    tick_state: Mapping[str, Any],
    completed: bool,
    pilot_id: str,
    quarantine_reason: str,
) -> dict[str, Any]:
    """Build and self-validate one full-state quarantined checkpoint payload."""

    if not isinstance(completed, bool):
        raise SoulWriterCheckpointError("completed must be a bool")
    inheritance, inheritance_sha256 = _validate_source_inheritance(
        source_inheritance_manifest
    )
    if core.cfg.to_dict() != inheritance.get("core_config"):
        raise SoulWriterCheckpointError(
            "live core config does not match source inheritance"
        )
    if writer.cfg.d_model != core.cfg.d_model or writer.cfg.total_rows != 168:
        raise SoulWriterCheckpointError(
            "writer config does not match inherited core and 168-row layout"
        )

    core_manifest = _module_state_manifest(
        core.state_dict(),
        label="core",
    )
    writer_manifest = _module_state_manifest(
        writer.state_dict(),
        label="writer",
    )
    freeze_manifest = _derive_freeze_manifest(core, writer)
    if freeze_manifest["core_soul_readonly"] is not True:
        raise SoulWriterCheckpointError(
            "reader-writer pilot requires core.soul_readonly=True"
        )
    core_delta_manifest = _core_delta_manifest(
        source_manifest=inheritance["core_state"],
        current_manifest=core_manifest,
        reader_names=freeze_manifest["reader_names"],
    )
    named = _named_parameters(core, writer)
    optimizer_envelope = _optimizer_envelope(
        optimizer,
        named_parameters=named,
        expected_trainable_names=freeze_manifest["trainable_names"],
    )
    soul_before = soul_state_sha256(soul_state)
    soul_envelope = _soul_envelope(soul_state, d_model=core.cfg.d_model)
    curriculum, sampler, tick = _validate_continuity_states(
        curriculum_state,
        sampler_state,
        tick_state,
        soul_tick=soul_state.current_tick,
    )
    rng_envelope = capture_rng_state()
    quarantine = _quarantine_provenance(
        pilot_id=pilot_id,
        reason=quarantine_reason,
        inheritance_sha256=inheritance_sha256,
    )

    payload: dict[str, Any] = {
        "checkpoint_schema": SOUL_WRITER_CHECKPOINT_SCHEMA,
        "schema_version": SOUL_WRITER_CHECKPOINT_VERSION,
        "checkpoint_kind": "quarantined_differentiable_soul_writer_pilot",
        "step": int(tick["global_step"]),
        "completed": completed,
        "promotable": False,
        "promotion_authorized": False,
        "quarantine": quarantine,
        "source_inheritance": {
            "manifest": inheritance,
            "manifest_sha256": inheritance_sha256,
            "source_file_sha256": inheritance["source"]["file_sha256"],
            "source_core_state_manifest_sha256": inheritance["core_state"][
                "manifest_sha256"
            ],
        },
        "core": {
            "config": copy.deepcopy(core.cfg.to_dict()),
            "state": copy.deepcopy(core.state_dict()),
            "source_state_manifest": copy.deepcopy(inheritance["core_state"]),
            "state_manifest": core_manifest,
            "delta_manifest": core_delta_manifest,
            "non_reader_unchanged_from_source_inheritance": True,
        },
        "writer": {
            "config": copy.deepcopy(writer.cfg.to_dict()),
            "state": copy.deepcopy(writer.state_dict()),
            "state_manifest": writer_manifest,
        },
        "freeze_manifest": freeze_manifest,
        "optimizer": optimizer_envelope,
        "soul": soul_envelope,
        "rng": rng_envelope,
        "continuity": _continuity_envelope(curriculum, sampler, tick),
    }
    if soul_state_sha256(soul_state) != soul_before:
        raise SoulWriterCheckpointError(
            "checkpoint construction mutated live SoulState"
        )
    # Validate the complete in-memory payload without letting fresh module
    # construction advance the caller's process RNG streams.
    validation_rng = capture_rng_state()
    try:
        _load_payload(payload, file_sha256="in_memory")
    finally:
        restore_rng_state(validation_rng)
    return payload


def _load_payload(
    payload: Any,
    *,
    file_sha256: str,
    expected_source_inheritance_sha256: str | None = None,
) -> LoadedSoulWriterCheckpoint:
    if not isinstance(payload, Mapping):
        raise SoulWriterCheckpointError("checkpoint payload must be a mapping")
    required_keys = {
        "checkpoint_schema",
        "schema_version",
        "checkpoint_kind",
        "step",
        "completed",
        "promotable",
        "promotion_authorized",
        "quarantine",
        "source_inheritance",
        "core",
        "writer",
        "freeze_manifest",
        "optimizer",
        "soul",
        "rng",
        "continuity",
    }
    if set(payload) != required_keys:
        missing = sorted(required_keys - set(payload))
        unexpected = sorted(set(payload) - required_keys)
        raise SoulWriterCheckpointError(
            f"checkpoint fields mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if (
        payload.get("checkpoint_schema") != SOUL_WRITER_CHECKPOINT_SCHEMA
        or payload.get("schema_version") != SOUL_WRITER_CHECKPOINT_VERSION
        or payload.get("checkpoint_kind")
        != "quarantined_differentiable_soul_writer_pilot"
    ):
        raise SoulWriterCheckpointError("checkpoint schema/version is invalid")
    if not isinstance(payload.get("completed"), bool):
        raise SoulWriterCheckpointError("checkpoint completed flag is invalid")
    if (
        payload.get("promotable") is not False
        or payload.get("promotion_authorized") is not False
    ):
        raise SoulWriterCheckpointError(
            "writer pilot checkpoint must remain non-promotable"
        )
    _exact_nonnegative_int(payload.get("step"), label="checkpoint step")

    source = payload.get("source_inheritance")
    if not isinstance(source, Mapping):
        raise SoulWriterCheckpointError("source inheritance envelope is missing")
    inheritance, inheritance_sha256 = _validate_source_inheritance(
        source.get("manifest")
    )
    if (
        expected_source_inheritance_sha256 is not None
        and inheritance_sha256.lower()
        != expected_source_inheritance_sha256.lower()
    ):
        raise SoulWriterCheckpointError(
            "source inheritance manifest does not match expected hash"
        )
    if (
        source.get("manifest_sha256") != inheritance_sha256
        or source.get("source_file_sha256")
        != inheritance["source"]["file_sha256"]
        or source.get("source_core_state_manifest_sha256")
        != inheritance["core_state"]["manifest_sha256"]
    ):
        raise SoulWriterCheckpointError(
            "source inheritance proof/hash mismatch"
        )
    quarantine = _validate_quarantine(
        payload.get("quarantine"),
        inheritance_sha256=inheritance_sha256,
    )

    core_envelope = payload.get("core")
    if not isinstance(core_envelope, Mapping):
        raise SoulWriterCheckpointError("core envelope is missing")
    if (
        core_envelope.get("non_reader_unchanged_from_source_inheritance")
        is not True
    ):
        raise SoulWriterCheckpointError(
            "core does not declare the non-reader inheritance byte lock"
        )
    cfg = _core_config_from_payload(core_envelope.get("config"))
    if cfg.to_dict() != inheritance.get("core_config"):
        raise SoulWriterCheckpointError(
            "checkpoint core config differs from source inheritance"
        )
    core_state = core_envelope.get("state")
    if not isinstance(core_state, Mapping):
        raise SoulWriterCheckpointError("checkpoint core state is missing")
    core_manifest = _module_state_manifest(core_state, label="core")
    if core_envelope.get("state_manifest") != core_manifest:
        raise SoulWriterCheckpointError(
            "checkpoint current core state/hash mismatch"
        )
    if core_envelope.get("source_state_manifest") != inheritance.get(
        "core_state"
    ):
        raise SoulWriterCheckpointError(
            "checkpoint source core manifest differs from inheritance"
        )
    core_dtype = _single_float_dtype(core_state, label="core")
    try:
        core = AxonCore(cfg).to(torch.device("cpu"), dtype=core_dtype)
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"fresh core construction failed: {type(exc).__name__}: {exc}"
        ) from exc
    expected_reader_names = writer_pilot_reader_parameter_names(core)
    expected_delta_manifest = _core_delta_manifest(
        source_manifest=inheritance["core_state"],
        current_manifest=core_manifest,
        reader_names=expected_reader_names,
    )
    if core_envelope.get("delta_manifest") != expected_delta_manifest:
        raise SoulWriterCheckpointError("checkpoint core delta manifest mismatch")
    _assert_state_layout(core_state, core.state_dict(), label="core")
    try:
        core.load_state_dict(core_state, strict=True)
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"strict core load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if _module_state_manifest(core.state_dict(), label="loaded core") != core_manifest:
        raise SoulWriterCheckpointError(
            "loaded core is not byte-identical to checkpoint"
        )

    writer_envelope = payload.get("writer")
    if not isinstance(writer_envelope, Mapping):
        raise SoulWriterCheckpointError("writer envelope is missing")
    writer_cfg = _writer_config_from_payload(
        writer_envelope.get("config"),
        d_model=cfg.d_model,
    )
    writer_state = writer_envelope.get("state")
    if not isinstance(writer_state, Mapping):
        raise SoulWriterCheckpointError("writer state is missing")
    writer_manifest = _module_state_manifest(writer_state, label="writer")
    if writer_envelope.get("state_manifest") != writer_manifest:
        raise SoulWriterCheckpointError("writer state hash mismatch")
    writer_dtype = _single_float_dtype(writer_state, label="writer")
    if writer_dtype != core_dtype:
        raise SoulWriterCheckpointError(
            "writer dtype does not match inherited core dtype"
        )
    writer = DifferentiableSoulWriter(writer_cfg).to(
        torch.device("cpu"),
        dtype=writer_dtype,
    )
    _assert_state_layout(writer_state, writer.state_dict(), label="writer")
    try:
        writer.load_state_dict(writer_state, strict=True)
    except Exception as exc:
        raise SoulWriterCheckpointError(
            f"strict writer load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if _module_state_manifest(
        writer.state_dict(),
        label="loaded writer",
    ) != writer_manifest:
        raise SoulWriterCheckpointError(
            "loaded writer is not byte-identical to checkpoint"
        )

    observed_freeze = _apply_allowlisted_reader_freeze(core, writer)
    freeze_manifest = payload.get("freeze_manifest")
    if not isinstance(freeze_manifest, Mapping) or dict(
        freeze_manifest
    ) != observed_freeze:
        raise SoulWriterCheckpointError("freeze manifest mismatch")
    if _module_state_manifest(core.state_dict(), label="frozen core") != core_manifest:
        raise SoulWriterCheckpointError("freeze policy changed core tensor bytes")

    optimizer = _validate_optimizer_envelope(
        payload.get("optimizer"),
        core=core,
        writer=writer,
        freeze_manifest=observed_freeze,
    )
    soul = _load_soul_envelope(payload.get("soul"), d_model=cfg.d_model)
    curriculum, sampler, tick = _load_continuity_envelope(
        payload.get("continuity"),
        soul_tick=soul.current_tick,
    )
    if int(tick["global_step"]) != int(payload["step"]):
        raise SoulWriterCheckpointError(
            "checkpoint step does not match tick global_step"
        )
    rng = payload.get("rng")
    _validate_rng_state(rng)
    return LoadedSoulWriterCheckpoint(
        core=core,
        writer=writer,
        optimizer=optimizer,
        soul_state=soul,
        curriculum_state=curriculum,
        sampler_state=sampler,
        tick_state=tick,
        rng_state=copy.deepcopy(dict(rng)),
        source_inheritance_manifest=inheritance,
        freeze_manifest=observed_freeze,
        quarantine_provenance=quarantine,
        completed=bool(payload["completed"]),
        file_sha256=file_sha256,
    )


def save_soul_writer_checkpoint(
    destination: str | Path,
    *,
    core: AxonCore,
    writer: DifferentiableSoulWriter,
    optimizer: torch.optim.Optimizer,
    soul_state: SoulState,
    source_inheritance_manifest: Mapping[str, Any],
    curriculum_state: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    tick_state: Mapping[str, Any],
    completed: bool,
    pilot_id: str,
    quarantine_reason: str,
    overwrite: bool = False,
) -> Path:
    """Build, validate, and atomically save one quarantined pilot checkpoint."""

    payload = build_soul_writer_checkpoint(
        core=core,
        writer=writer,
        optimizer=optimizer,
        soul_state=soul_state,
        source_inheritance_manifest=source_inheritance_manifest,
        curriculum_state=curriculum_state,
        sampler_state=sampler_state,
        tick_state=tick_state,
        completed=completed,
        pilot_id=pilot_id,
        quarantine_reason=quarantine_reason,
    )
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        del payload
        # Re-load the exact temporary artifact before it can replace the
        # destination. A truncated or serialization-corrupt file never lands.
        load_soul_writer_checkpoint(temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_soul_writer_checkpoint(
    source: str | Path,
    *,
    restore_rng: bool = False,
    expected_source_inheritance_sha256: str | None = None,
) -> LoadedSoulWriterCheckpoint:
    """Load and strictly validate a quarantined full-state writer checkpoint."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise SoulWriterCheckpointError(
            f"writer checkpoint is not a file: {path}"
        )
    before = _file_sha256(path)
    if (
        expected_source_inheritance_sha256 is not None
        and not _valid_sha256(expected_source_inheritance_sha256)
    ):
        raise SoulWriterCheckpointError(
            "expected source inheritance hash is invalid"
        )
    caller_rng = capture_rng_state()
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise SoulWriterCheckpointError(
                f"writer checkpoint cannot be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        loaded = _load_payload(
            payload,
            file_sha256=before,
            expected_source_inheritance_sha256=(
                expected_source_inheritance_sha256
            ),
        )
        if _file_sha256(path) != before:
            raise SoulWriterCheckpointError(
                "writer checkpoint changed during strict load"
            )
    finally:
        restore_rng_state(caller_rng)
    if restore_rng:
        loaded.restore_rng()
    return loaded


__all__ = [
    "SOUL_WRITER_CHECKPOINT_SCHEMA",
    "SOUL_WRITER_CHECKPOINT_VERSION",
    "SOUL_WRITER_TICK_SCHEMA",
    "SOUL_WRITER_FREEZE_SCHEMA",
    "SOUL_WRITER_CORE_DELTA_SCHEMA",
    "SOUL_WRITER_QUARANTINE_SCHEMA",
    "SoulWriterCheckpointError",
    "LoadedSoulWriterCheckpoint",
    "deterministic_state_sha256",
    "capture_rng_state",
    "restore_rng_state",
    "build_soul_writer_checkpoint",
    "save_soul_writer_checkpoint",
    "load_soul_writer_checkpoint",
]
