"""Pinned, fail-closed loading for exact-v4 inference checkpoints.

Training checkpoints are pickle-backed PyTorch artifacts.  The runtime only
opens one after the caller pins the SHA-256 of the *whole file*.  Optimizer and
RNG payloads are acknowledged as source evidence, but are neither restored nor
retained by the returned object.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from cores.core import AxonCore, CoreConfig
from training.soul_compressor import SoulCompressor


CHECKPOINT_SCHEMA = "axon_exact_v4_checkpoint_v1"
HOT_ONLY = "HOT_ONLY"
HOT_WARM = "HOT_WARM"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED_PAYLOAD_KEYS = {
    "checkpoint_schema",
    "step",
    "cfg",
    "core_state",
    "optimizer_state",
    "rng_state",
    "soul_state",
}
_OPTIONAL_PAYLOAD_KEYS = {"soul_compressor_state"}
_CFG_KEYS = {
    "d_model",
    "n_heads",
    "n_layers",
    "ffn_dim",
    "dropout",
    "n_ticks",
    "grad_ticks",
    "soul_rows",
    "soul_mode",
    "soul_gate_init",
    "soul_hot_rows",
    "soul_write_mode",
    "n_soul_compartments",
    "char_slot_mode",
    "char_slot_max_slots",
    "char_n_regions",
}
_RNG_REQUIRED_KEYS = {
    "python_random_state",
    "numpy_random_state",
    "torch_rng_state",
}
_RNG_OPTIONAL_KEYS = {"cuda_rng_state_all"}
_CORE_STATE_HASH_SCHEMA = "axon-runtime-frozen-core-state-sha256-v1"


class CheckpointContractError(RuntimeError):
    """An exact-v4 checkpoint failed its pinned runtime contract."""


class CheckpointHashError(CheckpointContractError):
    """The artifact bytes do not match the caller's pinned SHA-256."""


@dataclass(frozen=True, slots=True)
class CheckpointSourceEvidence:
    """Non-restorable acknowledgement of training-only source material."""

    optimizer_state_present: bool
    rng_state_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExactV4Checkpoint:
    """Fully validated inference material, detached from the source payload.

    ``core_state_sha256`` hashes the post-load, frozen ``AxonCore`` tensor
    state.  It excludes optimizer, RNG, soul, compressor, and serialization
    metadata, so it is suitable for ``CoreIdentity.core_state_sha256`` while
    ``checkpoint_sha256`` continues to identify the complete source artifact.
    """

    path: Path
    checkpoint_sha256: str
    core_state_sha256: str
    byte_length: int
    step: int
    cfg: CoreConfig
    core: AxonCore
    initial_soul: torch.Tensor
    initial_soul_mask: torch.Tensor
    soul_compressor: SoulCompressor | None
    soul_tier_status: str
    mmap_used: bool
    source_evidence: CheckpointSourceEvidence


def file_sha256(path: str | Path) -> str:
    """Hash every byte in *path* without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inference_core_state_sha256(
    state: Mapping[str, torch.Tensor],
) -> str:
    """Hash one frozen inference state dict in a deterministic tensor format.

    Tensor entries are ordered by name.  Each entry binds its name, dtype,
    shape, element count, byte count, and the SHA-256 of its contiguous CPU
    bytes.  The returned digest hashes a canonical JSON manifest containing
    those entries and a versioned schema.  Mapping insertion order therefore
    cannot affect the result.

    Callers constructing identity evidence should pass the validated
    post-``load_state_dict`` module state, never the training payload mapping.
    """

    if not isinstance(state, Mapping) or not state:
        raise CheckpointContractError(
            "inference core state must be a non-empty mapping"
        )
    if not all(isinstance(name, str) and name for name in state):
        raise CheckpointContractError(
            "inference core state keys must be non-empty strings"
        )
    entries: list[dict[str, Any]] = []
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise CheckpointContractError(
                f"inference core state value {name!r} is not a tensor"
            )
        if tensor.layout is not torch.strided:
            raise CheckpointContractError(
                f"inference core state tensor {name!r} must use strided layout"
            )
        value = tensor.detach().cpu().contiguous()
        if (
            value.is_floating_point() or value.is_complex()
        ) and not bool(torch.isfinite(value).all()):
            raise CheckpointContractError(
                f"inference core state tensor {name!r} contains non-finite "
                "values"
            )
        byte_view = value.reshape(-1).view(torch.uint8).numpy()
        raw_sha256 = hashlib.sha256(memoryview(byte_view)).hexdigest()
        entries.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "numel": int(value.numel()),
                "nbytes": int(value.numel() * value.element_size()),
                "sha256": raw_sha256,
            }
        )
    canonical = json.dumps(
        {
            "schema": _CORE_STATE_HASH_SCHEMA,
            "tensors": entries,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _pinned_hash(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CheckpointHashError(
            "expected_sha256 must be a 64-character hexadecimal SHA-256"
        )
    return value.lower()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise CheckpointContractError(f"{label} must be a string-keyed mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise CheckpointContractError(
            f"{label} keys mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _exact_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointContractError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointContractError(f"{label} must be finite")
    return result


def _validate_cfg(value: Any) -> CoreConfig:
    raw = _mapping(value, "cfg")
    _exact_keys(raw, _CFG_KEYS, "cfg")
    ints = {
        "d_model": _exact_int(raw["d_model"], "cfg.d_model", minimum=16),
        "n_heads": _exact_int(raw["n_heads"], "cfg.n_heads", minimum=1),
        "n_layers": _exact_int(raw["n_layers"], "cfg.n_layers", minimum=1),
        "ffn_dim": _exact_int(raw["ffn_dim"], "cfg.ffn_dim", minimum=1),
        "n_ticks": _exact_int(raw["n_ticks"], "cfg.n_ticks", minimum=1),
        "grad_ticks": _exact_int(raw["grad_ticks"], "cfg.grad_ticks", minimum=1),
        "soul_rows": _exact_int(raw["soul_rows"], "cfg.soul_rows"),
        "soul_hot_rows": _exact_int(
            raw["soul_hot_rows"],
            "cfg.soul_hot_rows",
        ),
        "n_soul_compartments": _exact_int(
            raw["n_soul_compartments"],
            "cfg.n_soul_compartments",
            minimum=1,
        ),
        "char_slot_max_slots": _exact_int(
            raw["char_slot_max_slots"],
            "cfg.char_slot_max_slots",
            minimum=1,
        ),
        "char_n_regions": _exact_int(
            raw["char_n_regions"],
            "cfg.char_n_regions",
            minimum=1,
        ),
    }
    if ints["d_model"] % ints["n_heads"]:
        raise CheckpointContractError(
            "cfg.d_model must be divisible by cfg.n_heads"
        )
    if ints["grad_ticks"] > ints["n_ticks"]:
        raise CheckpointContractError("cfg.grad_ticks cannot exceed cfg.n_ticks")
    if ints["soul_rows"] + ints["soul_hot_rows"] <= 0:
        raise CheckpointContractError("checkpoint soul layout has no rows")
    dropout = _finite_float(raw["dropout"], "cfg.dropout")
    if not 0.0 <= dropout < 1.0:
        raise CheckpointContractError("cfg.dropout must be in [0, 1)")
    soul_gate_init = _finite_float(
        raw["soul_gate_init"],
        "cfg.soul_gate_init",
    )
    if raw["soul_mode"] != "act_reflect_v2":
        raise CheckpointContractError(
            "exact-v4 runtime requires cfg.soul_mode='act_reflect_v2'"
        )
    if raw["soul_write_mode"] not in {"direct", "compartments"}:
        raise CheckpointContractError("unknown cfg.soul_write_mode")
    if not isinstance(raw["char_slot_mode"], bool) or not raw["char_slot_mode"]:
        raise CheckpointContractError(
            "exact-v4 runtime requires cfg.char_slot_mode=true"
        )
    if ints["char_slot_max_slots"] != 384:
        raise CheckpointContractError(
            "exact-v4 runtime requires exactly 384 char slots"
        )
    if ints["char_n_regions"] != 3:
        raise CheckpointContractError(
            "exact-v4 runtime requires exactly three physical roles"
        )
    normalized = {
        **ints,
        "dropout": dropout,
        "soul_mode": raw["soul_mode"],
        "soul_gate_init": soul_gate_init,
        "soul_write_mode": raw["soul_write_mode"],
        "char_slot_mode": True,
    }
    return CoreConfig.from_dict(normalized)


def _fresh_core(cfg: CoreConfig) -> AxonCore:
    # Constructors initialize parameters.  Isolate those temporary draws from
    # the process-global torch RNG before strict checkpoint replacement.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return AxonCore(cfg)


def _fresh_compressor(cfg: CoreConfig) -> SoulCompressor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return SoulCompressor(
            d_model=cfg.d_model,
            hot_rows=cfg.soul_hot_rows,
            warm_rows=cfg.soul_rows,
            num_heads=cfg.n_heads,
        )


def _validate_tensor(
    tensor: Any,
    *,
    label: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise CheckpointContractError(f"{label} must be a tensor")
    if tensor.layout is not torch.strided:
        raise CheckpointContractError(f"{label} must use strided layout")
    if tuple(tensor.shape) != shape:
        raise CheckpointContractError(
            f"{label} shape mismatch: {tuple(tensor.shape)} != {shape}"
        )
    if tensor.dtype is not dtype:
        raise CheckpointContractError(
            f"{label} dtype mismatch: {tensor.dtype} != {dtype}"
        )
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise CheckpointContractError(f"{label} contains non-finite values")
    return tensor


def _validate_module_state(
    source_value: Any,
    target: torch.nn.Module,
    *,
    label: str,
) -> Mapping[str, torch.Tensor]:
    source = _mapping(source_value, label)
    target_state = target.state_dict()
    _exact_keys(source, set(target_state), label)
    for name, expected in target_state.items():
        _validate_tensor(
            source[name],
            label=f"{label}.{name}",
            shape=tuple(expected.shape),
            dtype=expected.dtype,
        )
    return source  # type: ignore[return-value]


def _load_payload(path: Path) -> tuple[Mapping[str, Any], bool]:
    kwargs = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        return torch.load(path, mmap=True, **kwargs), True
    except TypeError as exc:
        if "mmap" not in str(exc):
            raise
    except RuntimeError as exc:
        message = str(exc).lower()
        if "mmap" not in message or "old serialization" not in message:
            raise
    return torch.load(path, **kwargs), False


def inspect_exact_v4_checkpoint(
    path: str | Path,
    expected_sha256: str,
    *,
    device: str | torch.device = "cpu",
) -> ExactV4Checkpoint:
    """Hash, inspect, strictly load, and detach one exact-v4 checkpoint."""

    source_path = Path(path).resolve(strict=True)
    if not source_path.is_file():
        raise CheckpointContractError("checkpoint path must be a regular file")
    pinned = _pinned_hash(expected_sha256)
    before = source_path.stat()
    actual = file_sha256(source_path)
    after_hash = source_path.stat()
    if actual != pinned:
        raise CheckpointHashError(
            f"checkpoint SHA-256 mismatch: expected {pinned}, got {actual}"
        )
    if (
        before.st_size != after_hash.st_size
        or before.st_mtime_ns != after_hash.st_mtime_ns
    ):
        raise CheckpointHashError("checkpoint changed while it was hashed")

    try:
        payload_raw, mmap_used = _load_payload(source_path)
    except Exception as exc:
        if isinstance(exc, CheckpointContractError):
            raise
        raise CheckpointContractError(
            f"could not load pinned checkpoint: {type(exc).__name__}: {exc}"
        ) from exc
    payload = _mapping(payload_raw, "checkpoint payload")
    expected_keys = _REQUIRED_PAYLOAD_KEYS | (
        {"soul_compressor_state"}
        if "soul_compressor_state" in payload
        else set()
    )
    _exact_keys(payload, expected_keys, "checkpoint payload")
    if payload["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise CheckpointContractError(
            f"checkpoint_schema must be {CHECKPOINT_SCHEMA!r}"
        )
    step = _exact_int(payload["step"], "checkpoint step")
    cfg = _validate_cfg(payload["cfg"])

    optimizer = _mapping(payload["optimizer_state"], "optimizer_state")
    _exact_keys(optimizer, {"state", "param_groups"}, "optimizer_state")
    rng = _mapping(payload["rng_state"], "rng_state")
    rng_keys = set(rng)
    if (
        not _RNG_REQUIRED_KEYS <= rng_keys
        or rng_keys - (_RNG_REQUIRED_KEYS | _RNG_OPTIONAL_KEYS)
    ):
        raise CheckpointContractError(
            "rng_state must contain the recognized Python, NumPy, and Torch "
            "states, with optional CUDA state"
        )
    evidence = CheckpointSourceEvidence(
        optimizer_state_present=True,
        rng_state_keys=tuple(sorted(rng_keys)),
    )

    core = _fresh_core(cfg)
    core_state = _validate_module_state(
        payload["core_state"],
        core,
        label="core_state",
    )
    try:
        incompatible = core.load_state_dict(core_state, strict=True)
    except Exception as exc:
        raise CheckpointContractError(
            f"strict core_state load failed: {type(exc).__name__}: {exc}"
        ) from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointContractError("strict core_state load was incomplete")

    soul_state = _mapping(payload["soul_state"], "soul_state")
    _exact_keys(soul_state, {"soul", "soul_mask"}, "soul_state")
    rows = cfg.total_soul_rows()
    soul_source = _validate_tensor(
        soul_state["soul"],
        label="soul_state.soul",
        shape=(1, rows, cfg.d_model),
        dtype=torch.float32,
    )
    mask_source = _validate_tensor(
        soul_state["soul_mask"],
        label="soul_state.soul_mask",
        shape=(1, rows),
        dtype=torch.bool,
    )
    # Runtime private state is unbatched and owns independent CPU storage.
    initial_soul = soul_source[0].detach().cpu().contiguous().clone()
    initial_mask = mask_source[0].detach().cpu().contiguous().clone()

    compressor: SoulCompressor | None = None
    status = HOT_ONLY
    if "soul_compressor_state" in payload:
        if cfg.soul_hot_rows <= 0 or cfg.soul_rows <= 0:
            raise CheckpointContractError(
                "soul compressor requires non-empty hot and warm tiers"
            )
        compressor = _fresh_compressor(cfg)
        compressor_state = _validate_module_state(
            payload["soul_compressor_state"],
            compressor,
            label="soul_compressor_state",
        )
        try:
            incompatible = compressor.load_state_dict(
                compressor_state,
                strict=True,
            )
        except Exception as exc:
            raise CheckpointContractError(
                "strict soul compressor load failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise CheckpointContractError(
                "strict soul compressor load was incomplete"
            )
        for parameter in compressor.parameters():
            parameter.requires_grad_(False)
        compressor.eval()
        status = HOT_WARM

    for parameter in core.parameters():
        parameter.requires_grad_(False)
    core.use_checkpoint = False
    core.eval()
    # Hash the actual strict-loaded module after it has entered immutable
    # inference mode.  Do not derive identity from the source payload mapping.
    core_state_hash = inference_core_state_sha256(core.state_dict())
    target_device = torch.device(device)
    core.to(target_device)
    if compressor is not None:
        compressor.to(target_device)

    after_load = source_path.stat()
    if (
        after_hash.st_size != after_load.st_size
        or after_hash.st_mtime_ns != after_load.st_mtime_ns
    ):
        raise CheckpointHashError("checkpoint changed while it was inspected")

    # Do not expose payload/core_state/optimizer/RNG objects.  Only copied
    # inference modules, copied soul bytes, and source-evidence names survive.
    return ExactV4Checkpoint(
        path=source_path,
        checkpoint_sha256=actual,
        core_state_sha256=core_state_hash,
        byte_length=after_load.st_size,
        step=step,
        cfg=cfg,
        core=core,
        initial_soul=initial_soul,
        initial_soul_mask=initial_mask,
        soul_compressor=compressor,
        soul_tier_status=status,
        mmap_used=mmap_used,
        source_evidence=evidence,
    )


load_exact_v4_checkpoint = inspect_exact_v4_checkpoint


__all__ = [
    "CHECKPOINT_SCHEMA",
    "HOT_ONLY",
    "HOT_WARM",
    "CheckpointContractError",
    "CheckpointHashError",
    "CheckpointSourceEvidence",
    "ExactV4Checkpoint",
    "file_sha256",
    "inference_core_state_sha256",
    "inspect_exact_v4_checkpoint",
    "load_exact_v4_checkpoint",
]
