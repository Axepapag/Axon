from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from cores.core import AxonCore, CoreConfig
from runtime.axon_runtime.checkpoint import (
    HOT_ONLY,
    HOT_WARM,
    CheckpointContractError,
    CheckpointHashError,
    file_sha256,
    inference_core_state_sha256,
    inspect_exact_v4_checkpoint,
)
from runtime.axon_runtime.soul_store import (
    SoulBlobContractError,
    SoulBlobCorruptionError,
    SoulBlobStore,
    decode_soul_blob,
    encode_soul_blob,
)
from training.soul_compressor import SoulCompressor


def tiny_cfg() -> dict[str, object]:
    return CoreConfig(
        d_model=16,
        n_heads=1,
        n_layers=1,
        ffn_dim=32,
        dropout=0.0,
        n_ticks=2,
        grad_ticks=1,
        soul_rows=2,
        soul_mode="act_reflect_v2",
        soul_gate_init=0.05,
        soul_hot_rows=2,
        soul_write_mode="compartments",
        n_soul_compartments=2,
        char_slot_mode=True,
        char_slot_max_slots=384,
        char_n_regions=3,
    ).to_dict()


def checkpoint_payload(*, compressor: bool = False) -> dict[str, object]:
    cfg = CoreConfig.from_dict(tiny_cfg())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        core = AxonCore(cfg)
        soul = torch.randn(1, cfg.total_soul_rows(), cfg.d_model)
        compressor_module = SoulCompressor(
            d_model=cfg.d_model,
            hot_rows=cfg.soul_hot_rows,
            warm_rows=cfg.soul_rows,
            num_heads=cfg.n_heads,
        )
    payload: dict[str, object] = {
        "checkpoint_schema": "axon_exact_v4_checkpoint_v1",
        "step": 12,
        "cfg": cfg.to_dict(),
        "core_state": deepcopy(core.state_dict()),
        "optimizer_state": {"state": {}, "param_groups": []},
        "rng_state": {
            "python_random_state": ("source-evidence",),
            "numpy_random_state": ("source-evidence",),
            "torch_rng_state": torch.arange(8, dtype=torch.uint8),
        },
        "soul_state": {
            "soul": soul,
            "soul_mask": torch.tensor([[True, True, True, False]]),
        },
    }
    if compressor:
        payload["soul_compressor_state"] = deepcopy(
            compressor_module.state_dict()
        )
    return payload


def save_payload(tmp_path: Path, payload: dict[str, object], name: str) -> tuple[Path, str]:
    path = tmp_path / name
    torch.save(payload, path)
    return path, file_sha256(path)


def test_pinned_exact_v4_load_is_strict_detached_and_inference_only(
    tmp_path: Path,
) -> None:
    payload = checkpoint_payload()
    source_soul = payload["soul_state"]["soul"]  # type: ignore[index]
    path, digest = save_payload(tmp_path, payload, "tiny.pt")
    loaded = inspect_exact_v4_checkpoint(path, digest)

    assert loaded.checkpoint_sha256 == digest
    assert len(loaded.core_state_sha256) == 64
    assert loaded.core_state_sha256 == inference_core_state_sha256(
        loaded.core.state_dict()
    )
    assert loaded.byte_length == path.stat().st_size
    assert loaded.step == 12
    assert loaded.core.training is False
    assert all(not parameter.requires_grad for parameter in loaded.core.parameters())
    assert loaded.initial_soul.shape == (4, 16)
    assert loaded.initial_soul_mask.shape == (4,)
    assert loaded.initial_soul.data_ptr() != source_soul.data_ptr()
    assert loaded.soul_tier_status == HOT_ONLY
    assert loaded.soul_compressor is None
    assert loaded.source_evidence.optimizer_state_present is True
    assert loaded.source_evidence.rng_state_keys == (
        "numpy_random_state",
        "python_random_state",
        "torch_rng_state",
    )
    assert not hasattr(loaded, "optimizer_state")
    assert not hasattr(loaded, "rng_state")
    assert isinstance(loaded.mmap_used, bool)


def test_core_state_hash_is_order_independent_and_excludes_training_state(
    tmp_path: Path,
) -> None:
    base_payload = checkpoint_payload()
    base_path, base_file_hash = save_payload(
        tmp_path,
        base_payload,
        "hash-base.pt",
    )
    base = inspect_exact_v4_checkpoint(base_path, base_file_hash)

    evidence_variant = checkpoint_payload()
    state = evidence_variant["core_state"]
    assert isinstance(state, dict)
    evidence_variant["core_state"] = dict(reversed(tuple(state.items())))
    evidence_variant["optimizer_state"] = {
        "state": {"training-only": torch.tensor([91])},
        "param_groups": [{"training-only": True}],
    }
    rng = evidence_variant["rng_state"]
    assert isinstance(rng, dict)
    rng["torch_rng_state"] = torch.full((8,), 255, dtype=torch.uint8)
    soul = evidence_variant["soul_state"]
    assert isinstance(soul, dict)
    soul["soul"] = soul["soul"] + 3.0
    variant_path, variant_file_hash = save_payload(
        tmp_path,
        evidence_variant,
        "hash-evidence-variant.pt",
    )
    variant = inspect_exact_v4_checkpoint(variant_path, variant_file_hash)

    assert variant.checkpoint_sha256 != base.checkpoint_sha256
    assert variant.core_state_sha256 == base.core_state_sha256
    assert variant.core_state_sha256 == inference_core_state_sha256(
        dict(reversed(tuple(variant.core.state_dict().items())))
    )

    changed_payload = checkpoint_payload()
    changed_state = changed_payload["core_state"]
    assert isinstance(changed_state, dict)
    name = next(
        key
        for key, tensor in changed_state.items()
        if tensor.is_floating_point() and tensor.numel()
    )
    changed_state[name] = changed_state[name].clone()
    changed_state[name].reshape(-1)[0] += 0.25
    changed_path, changed_file_hash = save_payload(
        tmp_path,
        changed_payload,
        "hash-core-changed.pt",
    )
    changed = inspect_exact_v4_checkpoint(changed_path, changed_file_hash)

    assert changed.core_state_sha256 != base.core_state_sha256


def test_wrong_hash_schema_and_payload_key_fail_closed(tmp_path: Path) -> None:
    path, digest = save_payload(tmp_path, checkpoint_payload(), "base.pt")
    with pytest.raises(CheckpointHashError):
        inspect_exact_v4_checkpoint(path, "0" * 64)

    wrong_schema = checkpoint_payload()
    wrong_schema["checkpoint_schema"] = "legacy"
    schema_path, schema_hash = save_payload(
        tmp_path,
        wrong_schema,
        "schema.pt",
    )
    with pytest.raises(CheckpointContractError, match="checkpoint_schema"):
        inspect_exact_v4_checkpoint(schema_path, schema_hash)

    extra = checkpoint_payload()
    extra["untrusted"] = True
    extra_path, extra_hash = save_payload(tmp_path, extra, "extra.pt")
    with pytest.raises(CheckpointContractError, match="keys mismatch"):
        inspect_exact_v4_checkpoint(extra_path, extra_hash)

    with pytest.raises(CheckpointHashError):
        inspect_exact_v4_checkpoint(path, digest[:-1])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("core_shape", "shape mismatch"),
        ("core_nan", "non-finite"),
        ("soul_shape", "shape mismatch"),
        ("soul_nan", "non-finite"),
        ("mask_dtype", "dtype mismatch"),
        ("cfg_extra", "cfg keys mismatch"),
    ],
)
def test_shape_dtype_finite_and_cfg_contracts(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    payload = checkpoint_payload()
    core_state = payload["core_state"]  # type: ignore[assignment]
    soul_state = payload["soul_state"]  # type: ignore[assignment]
    if mutation == "core_shape":
        key = next(iter(core_state))
        core_state[key] = core_state[key].reshape(-1)[:-1]
    elif mutation == "core_nan":
        key = next(
            name for name, tensor in core_state.items() if tensor.numel() > 0
        )
        core_state[key] = core_state[key].clone()
        core_state[key].reshape(-1)[0] = float("nan")
    elif mutation == "soul_shape":
        soul_state["soul"] = soul_state["soul"][:, :-1, :]
    elif mutation == "soul_nan":
        soul_state["soul"] = soul_state["soul"].clone()
        soul_state["soul"][0, 0, 0] = float("nan")
    elif mutation == "mask_dtype":
        soul_state["soul_mask"] = soul_state["soul_mask"].to(torch.uint8)
    elif mutation == "cfg_extra":
        payload["cfg"]["unknown"] = 1  # type: ignore[index]
    path, digest = save_payload(tmp_path, payload, f"{mutation}.pt")
    with pytest.raises(CheckpointContractError, match=match):
        inspect_exact_v4_checkpoint(path, digest)


def test_compressor_is_strict_optional_and_never_fresh_initialized(
    tmp_path: Path,
) -> None:
    with_compressor = checkpoint_payload(compressor=True)
    path, digest = save_payload(tmp_path, with_compressor, "warm.pt")
    warm = inspect_exact_v4_checkpoint(path, digest)
    assert warm.soul_tier_status == HOT_WARM
    assert isinstance(warm.soul_compressor, SoulCompressor)
    assert warm.soul_compressor.training is False
    assert all(
        not parameter.requires_grad
        for parameter in warm.soul_compressor.parameters()
    )

    no_compressor = checkpoint_payload(compressor=False)
    path, digest = save_payload(tmp_path, no_compressor, "hot-only.pt")
    hot_only = inspect_exact_v4_checkpoint(path, digest)
    assert hot_only.soul_tier_status == HOT_ONLY
    assert hot_only.soul_compressor is None

    corrupt = checkpoint_payload(compressor=True)
    state = corrupt["soul_compressor_state"]  # type: ignore[assignment]
    state.pop(next(iter(state)))
    path, digest = save_payload(tmp_path, corrupt, "bad-compressor.pt")
    with pytest.raises(CheckpointContractError, match="keys mismatch"):
        inspect_exact_v4_checkpoint(path, digest)


def test_soul_blob_round_trip_has_owned_storage_and_detects_corruption(
    tmp_path: Path,
) -> None:
    soul = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = torch.tensor([True, False, True])
    blob = encode_soul_blob(
        soul,
        mask,
        soul_id="soul-a",
        generation=4,
        base_field_id="field-a",
    )
    soul.fill_(-10)
    mask.fill_(False)
    decoded = decode_soul_blob(blob.data, blob.sha256)
    assert decoded.soul[0, 0].item() == 0.0
    assert decoded.mask.tolist() == [True, False, True]
    assert decoded.soul.data_ptr() != soul.data_ptr()

    store = SoulBlobStore(tmp_path / "souls")
    stored = store.persist(blob)
    first = store.load(blob.sha256)
    second = store.load(blob.sha256)
    assert stored.path.name == f"{blob.sha256}.soul"
    assert first.soul.data_ptr() != second.soul.data_ptr()
    assert torch.equal(first.soul, second.soul)

    corrupt = bytearray(blob.data)
    corrupt[-1] ^= 1
    with pytest.raises(SoulBlobCorruptionError):
        decode_soul_blob(corrupt, blob.sha256)
    stored.path.write_bytes(bytes(corrupt))
    with pytest.raises(SoulBlobCorruptionError):
        store.load(blob.sha256)

    with pytest.raises(SoulBlobContractError):
        encode_soul_blob(
            torch.full((2, 2), float("nan")),
            torch.ones(2, dtype=torch.bool),
            soul_id="s",
            generation=0,
            base_field_id="f",
        )
