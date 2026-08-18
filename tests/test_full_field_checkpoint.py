from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulState, SoulV2Config, TierSpec
from training.full_field_checkpoint import (
    FULL_FIELD_INHERITANCE_SCHEMA,
    FullFieldCheckpointError,
    core_state_tensor_manifest,
    load_full_field_checkpoint,
    save_full_field_inheritance_manifest,
)


def _cfg(*, d_model: int = 16) -> CoreConfig:
    return CoreConfig(
        d_model=d_model,
        n_heads=1,
        n_layers=1,
        ffn_dim=32,
        dropout=0.0,
        n_ticks=1,
        grad_ticks=1,
        soul_rows=64,
        soul_mode="act_reflect_v2",
        soul_gate_init=0.05,
        soul_hot_rows=64,
        soul_write_mode="compartments",
        n_soul_compartments=4,
        char_slot_mode=True,
        char_slot_max_slots=384,
        char_n_regions=3,
    )


def _external_soul(d_model: int) -> tuple[dict, dict]:
    cfg = SoulV2Config(
        d_model=d_model,
        tiers=[
            TierSpec(name="hot", max_rows=128),
            TierSpec(name="warm", max_rows=32),
            TierSpec(name="cold", max_rows=8),
        ],
        categories=["episodic", "lessons"],
    )
    state = SoulState(cfg, torch.device("cpu"), torch.float32)
    state.tensor[:3] = torch.arange(3 * d_model).reshape(3, d_model)
    state.active[:3] = True
    state.category[:2] = torch.tensor([0, 1])
    state.category[2] = -1
    state.salience[:3] = torch.tensor([0.9, 0.8, 0.7])
    state.tick_born[:3] = torch.tensor([1, 2, 3])
    state.current_tick = 5
    return cfg.to_dict(), state.to_saveable()


def _payload(
    *,
    d_model: int = 16,
    optimizer: bool = True,
    soul: bool = True,
) -> dict:
    torch.manual_seed(44)
    cfg = _cfg(d_model=d_model)
    core = AxonCore(cfg)
    payload = {
        "checkpoint_schema": "axon_charslot_checkpoint_v2",
        "step": 252_000,
        "cfg": cfg.to_dict(),
        "core_state": {
            name: tensor.detach().clone()
            for name, tensor in core.state_dict().items()
        },
    }
    if optimizer:
        opt = torch.optim.AdamW(core.parameters(), lr=1e-3)
        payload["optimizer_state"] = opt.state_dict()
    if soul:
        soul_cfg, soul_state = _external_soul(d_model)
        payload["soul_cfg"] = soul_cfg
        payload["soul_state"] = soul_state
        payload["soul_mgr_state"] = {"evidence": torch.tensor([1.0])}
    return payload


def _save(tmp_path: Path, payload: dict, name: str = "source.pt") -> Path:
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_inheritance_is_byte_identical_and_json_serializable(
    tmp_path: Path,
) -> None:
    payload = _payload()
    source_manifest = core_state_tensor_manifest(payload["core_state"])
    path = _save(tmp_path, payload)
    file_before = _sha256(path)
    output_path = tmp_path / "inheritance.json"

    inherited = load_full_field_checkpoint(path, expected_d_model=16)
    manifest = inherited.manifest

    assert not output_path.exists()
    assert _sha256(path) == file_before
    assert manifest["schema"] == FULL_FIELD_INHERITANCE_SCHEMA
    assert manifest["passed"] is True
    assert manifest["source"]["file_sha256"] == file_before
    assert manifest["core_state"] == source_manifest
    assert manifest["inheritance_proof"] == {
        "fresh_axon_core_instantiated": True,
        "strict_state_load": True,
        "source_state_manifest_sha256": source_manifest["manifest_sha256"],
        "inherited_state_manifest_sha256": source_manifest["manifest_sha256"],
        "all_tensor_bytes_identical": True,
        "source_payload_unchanged": True,
        "source_file_unchanged": True,
    }
    assert manifest["migration"]["scope"] == "view_contract_only"
    assert manifest["migration"]["learned_tensor_changes"] is False
    assert manifest["migration"]["width_conversion"] is False
    assert manifest["promotion"]["authorized"] is False
    assert manifest["promotion"]["promotable"] is False
    json.dumps(manifest, sort_keys=True, allow_nan=False)

    inherited_state = inherited.core.state_dict()
    for name, source_tensor in payload["core_state"].items():
        assert torch.equal(inherited_state[name], source_tensor)
        assert inherited_state[name].dtype == source_tensor.dtype

    saved = save_full_field_inheritance_manifest(manifest, output_path)
    assert saved == output_path.resolve()
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        save_full_field_inheritance_manifest(manifest, output_path)


@pytest.mark.parametrize(
    ("optimizer", "soul"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_presence_facts_and_fresh_optimizer_policy(
    tmp_path: Path,
    optimizer: bool,
    soul: bool,
) -> None:
    path = _save(
        tmp_path,
        _payload(optimizer=optimizer, soul=soul),
        f"source-{optimizer}-{soul}.pt",
    )

    manifest = load_full_field_checkpoint(
        path,
        expected_d_model=16,
    ).manifest

    assert manifest["optimizer"]["source_state_present"] is optimizer
    assert manifest["optimizer"]["source_state_inherited"] is False
    assert manifest["optimizer"]["policy"] == "fresh_required"
    assert manifest["optimizer"]["fresh_optimizer_required"] is True
    assert manifest["optimizer"]["named_state_migration"] is None
    assert manifest["soul"]["state_present"] is soul
    assert manifest["soul"]["manager_state_present"] is soul
    if soul:
        assert manifest["soul"]["core_declared_rows"] == 128
        assert manifest["soul"]["external_contract_rows"] == 168
        assert manifest["soul"]["row_delta_external_minus_core"] == 40
        assert manifest["soul"]["canonical_128_vs_168"] == {
            "core_rows": 128,
            "external_rows": 168,
            "delta": 40,
            "source_matches": True,
        }
    else:
        assert manifest["soul"]["external_contract_rows"] is None
        assert manifest["soul"]["canonical_128_vs_168"]["source_matches"] is False


def test_known_writer_parameters_are_frozen_and_excluded(
    tmp_path: Path,
) -> None:
    inherited = load_full_field_checkpoint(
        _save(tmp_path, _payload()),
        expected_d_model=16,
    )
    named = dict(inherited.core.named_parameters())
    frozen = inherited.manifest["read_bearing_phase"][
        "frozen_writer_parameter_names"
    ]
    trainable = dict(inherited.trainable_named_parameters())

    assert inherited.core.soul_readonly is True
    assert frozen
    assert any(name.startswith("soul_reflect.") for name in frozen)
    assert any(name.startswith("action_to_soul.") for name in frozen)
    assert any(name.startswith("soul_router.") for name in frozen)
    assert all(named[name].requires_grad is False for name in frozen)
    assert not set(frozen) & set(trainable)
    assert "soul_ingest.attn.qkv.weight" in trainable
    assert any(name.endswith("soul_cross_gate") for name in trainable)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("char_slot_mode", False, "char_slot_mode must be true"),
        ("char_slot_max_slots", 383, "must equal 384"),
        ("char_n_regions", 4, "must equal 3"),
        ("soul_mode", "concat", "must equal act_reflect_v2"),
    ],
)
def test_rejects_non_charslot_source_configs(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    payload = _payload()
    payload["cfg"][field] = value
    with pytest.raises(FullFieldCheckpointError, match=message):
        load_full_field_checkpoint(_save(tmp_path, payload))


def test_rejects_missing_or_requested_different_width(tmp_path: Path) -> None:
    missing = _payload()
    del missing["cfg"]["d_model"]
    with pytest.raises(FullFieldCheckpointError, match="missing fields: d_model"):
        load_full_field_checkpoint(_save(tmp_path, missing, "missing.pt"))

    with pytest.raises(FullFieldCheckpointError, match="width conversion is forbidden"):
        load_full_field_checkpoint(
            _save(tmp_path, _payload(), "width.pt"),
            expected_d_model=64,
        )


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "nan"])
def test_rejects_corrupt_core_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    payload = _payload()
    state = payload["core_state"]
    name = next(iter(state))
    if corruption == "missing":
        del state[name]
        message = "keys do not match fresh AxonCore"
    elif corruption == "unexpected":
        state["not.a.real.tensor"] = torch.zeros(1)
        message = "keys do not match fresh AxonCore"
    elif corruption == "shape":
        state[name] = state[name].reshape(-1)[:1]
        message = "shape mismatch"
    else:
        state[name] = state[name].clone()
        state[name].reshape(-1)[0] = torch.nan
        message = "non-finite"

    with pytest.raises(FullFieldCheckpointError, match=message):
        load_full_field_checkpoint(_save(tmp_path, payload))


def test_rejects_corrupt_optional_state_and_untrusted_schema(
    tmp_path: Path,
) -> None:
    invalid_optimizer = _payload()
    invalid_optimizer["optimizer_state"] = {"state": "wrong", "param_groups": []}
    with pytest.raises(FullFieldCheckpointError, match="optimizer_state"):
        load_full_field_checkpoint(
            _save(tmp_path, invalid_optimizer, "optimizer.pt")
        )

    invalid_soul = _payload()
    invalid_soul["soul_state"]["tensor"] = torch.zeros(168, 8)
    with pytest.raises(FullFieldCheckpointError, match="width"):
        load_full_field_checkpoint(_save(tmp_path, invalid_soul, "soul.pt"))

    invalid_schema = _payload()
    invalid_schema["checkpoint_schema"] = "unknown"
    with pytest.raises(FullFieldCheckpointError, match="not trusted"):
        load_full_field_checkpoint(
            _save(tmp_path, invalid_schema, "schema.pt")
        )


def test_legacy_unversioned_charslot_payload_is_explicitly_recorded(
    tmp_path: Path,
) -> None:
    payload = _payload(optimizer=False)
    del payload["checkpoint_schema"]
    manifest = load_full_field_checkpoint(
        _save(tmp_path, payload),
        expected_d_model=16,
    ).manifest
    assert manifest["source"]["checkpoint_schema"] == "legacy_unversioned"


def test_audit_does_not_mutate_in_memory_source_payload(tmp_path: Path) -> None:
    payload = _payload()
    tensor_hash_before = core_state_tensor_manifest(payload["core_state"])
    config_before = copy.deepcopy(payload["cfg"])
    path = _save(tmp_path, payload)

    load_full_field_checkpoint(path, expected_d_model=16)

    assert payload["cfg"] == config_before
    assert core_state_tensor_manifest(payload["core_state"]) == tensor_hash_before
