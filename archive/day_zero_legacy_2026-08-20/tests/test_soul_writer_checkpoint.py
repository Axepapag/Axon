from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulState, SoulV2Config, TierSpec
from training.differentiable_soul_writer import (
    DifferentiableSoulWriter,
    SoulWriterConfig,
    configure_writer_pilot_parameters,
    writer_pilot_reader_parameter_names,
)
from training.full_field_checkpoint import (
    core_state_tensor_manifest,
    load_full_field_checkpoint,
)
from training.soul_load_bearing import soul_state_sha256
from training.soul_writer_checkpoint import (
    SOUL_WRITER_CHECKPOINT_SCHEMA,
    SOUL_WRITER_CHECKPOINT_VERSION,
    SOUL_WRITER_CORE_DELTA_SCHEMA,
    SOUL_WRITER_TICK_SCHEMA,
    SoulWriterCheckpointError,
    capture_rng_state,
    deterministic_state_sha256,
    load_soul_writer_checkpoint,
    save_soul_writer_checkpoint,
)


def _core_cfg() -> CoreConfig:
    return CoreConfig(
        d_model=16,
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


def _inheritance(tmp_path: Path):
    torch.manual_seed(81)
    core = AxonCore(_core_cfg())
    source = tmp_path / "trusted-source.pt"
    torch.save(
        {
            "checkpoint_schema": "axon_charslot_checkpoint_v2",
            "step": 252_000,
            "cfg": core.cfg.to_dict(),
            "core_state": copy.deepcopy(core.state_dict()),
        },
        source,
    )
    return load_full_field_checkpoint(source, expected_d_model=16)


def _soul() -> SoulState:
    cfg = SoulV2Config(
        d_model=16,
        tiers=[
            TierSpec(name="hot", max_rows=128),
            TierSpec(name="warm", max_rows=32),
            TierSpec(name="cold", max_rows=8),
        ],
        categories=["episodic", "lessons"],
    )
    state = SoulState(cfg, torch.device("cpu"), torch.float32)
    generator = torch.Generator().manual_seed(82)
    state.tensor = torch.randn(168, 16, generator=generator)
    state.active[:130] = True
    state.category[:128] = torch.arange(128) % 2
    state.salience[:130] = torch.linspace(0.1, 1.0, 130)
    state.dormant_for[:130] = torch.arange(130)
    state.tick_born[:130] = torch.arange(130) + 10
    state.current_tick = 321
    return state


def _continuity(soul_tick: int = 321):
    curriculum = {
        "schema": "axon_soul_writer_curriculum_v1",
        "manifest_sha256": "a" * 64,
        "example_count": 4,
        "families": ["write_delay_recall_v1"],
    }
    sampler = {
        "schema": "axon_soul_writer_sampler_v1",
        "position": 2,
        "order": [2, 0, 3, 1],
        "rng_state": random.Random(9).getstate(),
        "families": ["write_delay_recall_v1"],
    }
    tick = {
        "schema": SOUL_WRITER_TICK_SCHEMA,
        "global_step": 17,
        "soul_tick": soul_tick,
        "write_events": 17,
        "recall_events": 17,
        "phase": "boundary",
    }
    return curriculum, sampler, tick


def _live_state(
    tmp_path: Path,
    *,
    initialize_optimizer: bool = True,
    train_read_path: bool = True,
):
    inherited = _inheritance(tmp_path)
    core = inherited.core
    torch.manual_seed(83)
    writer = DifferentiableSoulWriter(SoulWriterConfig(d_model=16))
    policy = configure_writer_pilot_parameters(
        core,
        writer,
        train_read_path=train_read_path,
    )
    expected_reader_names = (
        set(writer_pilot_reader_parameter_names(core))
        if train_read_path
        else set()
    )
    assert {
        name for name in policy.trainable_names if name.startswith("core.")
    } == expected_reader_names
    all_named = {
        **{
            f"core.{name}": parameter
            for name, parameter in core.named_parameters()
        },
        **{
            f"writer.{name}": parameter
            for name, parameter in writer.named_parameters()
        },
    }
    trainable_parameters = [
        all_named[name] for name in policy.trainable_names
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=2e-4,
    )
    if initialize_optimizer:
        # Every allowlisted reader and writer parameter receives a finite,
        # nonzero gradient so AdamW materializes complete named state.
        loss = sum(parameter.sum() for parameter in trainable_parameters)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    soul = _soul()
    curriculum, sampler, tick = _continuity(soul.current_tick)
    return {
        "core": core,
        "writer": writer,
        "optimizer": optimizer,
        "soul_state": soul,
        "source_inheritance_manifest": inherited.manifest,
        "curriculum_state": curriculum,
        "sampler_state": sampler,
        "tick_state": tick,
        "completed": False,
        "pilot_id": "writer-pilot-test",
        "quarantine_reason": "causal writer proof is not a promotion gate",
    }


def _save_valid(tmp_path: Path, name: str = "writer.pt"):
    kwargs = _live_state(tmp_path)
    path = save_soul_writer_checkpoint(tmp_path / name, **kwargs)
    return path, kwargs


def _assert_state_dict_equal(left, right) -> None:
    assert set(left) == set(right)
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert torch.equal(left[name], right[name]), name


def test_atomic_full_state_round_trip_is_strict_and_quarantined(
    tmp_path: Path,
) -> None:
    kwargs = _live_state(tmp_path)
    core_before = core_state_tensor_manifest(kwargs["core"].state_dict())
    writer_before = core_state_tensor_manifest(kwargs["writer"].state_dict())
    soul_before = soul_state_sha256(kwargs["soul_state"])
    rng_before = capture_rng_state()["sha256"]
    path = tmp_path / "writer-pilot.pt"

    saved = save_soul_writer_checkpoint(path, **kwargs)
    assert saved == path.resolve()
    assert rng_before == capture_rng_state()["sha256"]
    assert core_state_tensor_manifest(kwargs["core"].state_dict()) == core_before
    assert core_state_tensor_manifest(kwargs["writer"].state_dict()) == writer_before
    assert soul_state_sha256(kwargs["soul_state"]) == soul_before
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))

    loaded = load_soul_writer_checkpoint(path)
    assert rng_before == capture_rng_state()["sha256"]
    assert loaded.completed is False
    assert loaded.file_sha256
    assert loaded.quarantine_provenance["quarantined"] is True
    assert loaded.quarantine_provenance["promotable"] is False
    assert loaded.quarantine_provenance["promotion_authorized"] is False
    assert loaded.source_inheritance_manifest == kwargs[
        "source_inheritance_manifest"
    ]
    assert (
        loaded.freeze_manifest["policy"]
        == "allowlisted_soul_reader_plus_external_writer"
    )
    expected_reader_names = set(
        writer_pilot_reader_parameter_names(loaded.core)
    )
    observed_reader_names = {
        f"core.{name}"
        for name, parameter in loaded.core.named_parameters()
        if parameter.requires_grad
    }
    assert observed_reader_names == expected_reader_names
    assert set(loaded.freeze_manifest["reader_names"]) == expected_reader_names
    assert all(
        parameter.requires_grad for parameter in loaded.writer.parameters()
    )
    assert loaded.core.soul_readonly is True
    _assert_state_dict_equal(
        loaded.core.state_dict(),
        kwargs["core"].state_dict(),
    )
    _assert_state_dict_equal(
        loaded.writer.state_dict(),
        kwargs["writer"].state_dict(),
    )
    assert soul_state_sha256(loaded.soul_state) == soul_before
    assert loaded.curriculum_state == kwargs["curriculum_state"]
    assert loaded.tick_state == kwargs["tick_state"]
    assert deterministic_state_sha256(loaded.sampler_state) == (
        deterministic_state_sha256(kwargs["sampler_state"])
    )
    assert deterministic_state_sha256(loaded.optimizer.state_dict()) == (
        deterministic_state_sha256(kwargs["optimizer"].state_dict())
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_schema"] == SOUL_WRITER_CHECKPOINT_SCHEMA
    assert payload["schema_version"] == SOUL_WRITER_CHECKPOINT_VERSION
    assert payload["promotable"] is False
    assert payload["promotion_authorized"] is False
    assert payload["completed"] is False
    assert payload["soul"]["layout"] == {
        "hot_rows": 128,
        "warm_rows": 32,
        "cold_rows": 8,
        "total_rows": 168,
        "d_model": 16,
    }
    assert payload["optimizer"]["coverage"]["complete"] is True
    assert payload["optimizer"]["coverage"]["fraction"] == 1.0
    assert (
        payload["optimizer"]["coverage"]["mapped"]
        == len(payload["freeze_manifest"]["trainable_names"])
    )
    assert (
        payload["core"]["delta_manifest"]["schema"]
        == SOUL_WRITER_CORE_DELTA_SCHEMA
    )
    assert payload["core"]["source_state_manifest"] == kwargs[
        "source_inheritance_manifest"
    ]["core_state"]
    assert (
        payload["core"]["delta_manifest"][
            "all_non_reader_tensors_byte_identical"
        ]
        is True
    )
    assert payload["core"]["delta_manifest"]["changed_reader_tensor_count"] > 0
    assert payload["core"]["state_manifest"] != payload["core"][
        "source_state_manifest"
    ]
    assert all(
        entry["delta_sha256"]
        for entry in payload["core"]["delta_manifest"]["tensors"]
    )
    assert set(payload["rng"]["state"]) == {
        "python",
        "numpy",
        "torch",
        "cuda",
        "cuda_available",
        "cuda_device_count",
    }


def test_explicit_rng_restore_reproduces_checkpoint_state(tmp_path: Path) -> None:
    original = capture_rng_state()
    try:
        path, _ = _save_valid(tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = payload["rng"]["sha256"]
        random.random()
        np.random.random()
        torch.rand(4)
        assert capture_rng_state()["sha256"] != expected

        loaded = load_soul_writer_checkpoint(path, restore_rng=True)
        assert loaded.rng_state["sha256"] == expected
        assert capture_rng_state()["sha256"] == expected
    finally:
        # Keep this test isolated from the rest of the process.
        from training.soul_writer_checkpoint import restore_rng_state

        restore_rng_state(original)


def test_completed_is_persisted_but_never_promotable(tmp_path: Path) -> None:
    kwargs = _live_state(tmp_path)
    kwargs["completed"] = True
    path = save_soul_writer_checkpoint(tmp_path / "completed.pt", **kwargs)
    loaded = load_soul_writer_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert loaded.completed is True
    assert payload["completed"] is True
    assert payload["promotable"] is False
    assert payload["quarantine"]["promotable"] is False


def test_load_can_pin_the_external_source_inheritance_hash(
    tmp_path: Path,
) -> None:
    path, _ = _save_valid(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = payload["source_inheritance"]["manifest_sha256"]

    loaded = load_soul_writer_checkpoint(
        path,
        expected_source_inheritance_sha256=expected,
    )
    assert loaded.source_inheritance_manifest["passed"] is True
    with pytest.raises(
        SoulWriterCheckpointError,
        match="does not match expected hash",
    ):
        load_soul_writer_checkpoint(
            path,
            expected_source_inheritance_sha256="0" * 64,
        )


def test_save_requires_complete_named_optimizer_coverage(
    tmp_path: Path,
) -> None:
    kwargs = _live_state(tmp_path, initialize_optimizer=False)
    with pytest.raises(
        SoulWriterCheckpointError,
        match="optimizer state is missing",
    ):
        save_soul_writer_checkpoint(tmp_path / "missing-opt.pt", **kwargs)
    assert not (tmp_path / "missing-opt.pt").exists()


def test_save_rejects_optimizer_that_omits_reader_parameters(
    tmp_path: Path,
) -> None:
    kwargs = _live_state(tmp_path)
    writer_only = torch.optim.AdamW(kwargs["writer"].parameters(), lr=2e-4)
    loss = sum(parameter.sum() for parameter in kwargs["writer"].parameters())
    loss.backward()
    writer_only.step()
    writer_only.zero_grad(set_to_none=True)
    kwargs["optimizer"] = writer_only

    with pytest.raises(
        SoulWriterCheckpointError,
        match="optimizer named coverage mismatch",
    ):
        save_soul_writer_checkpoint(
            tmp_path / "missing-reader-opt.pt",
            **kwargs,
        )


def test_save_rejects_optimizer_that_contains_frozen_core_parameter(
    tmp_path: Path,
) -> None:
    kwargs = _live_state(tmp_path)
    trainable = [
        parameter
        for parameter in (
            *kwargs["core"].parameters(),
            *kwargs["writer"].parameters(),
        )
        if parameter.requires_grad
    ]
    frozen = dict(kwargs["core"].named_parameters())[
        "action_to_soul.q.weight"
    ]
    kwargs["optimizer"] = torch.optim.AdamW(
        [*trainable, frozen],
        lr=2e-4,
    )
    loss = sum(parameter.sum() for parameter in trainable)
    loss.backward()
    kwargs["optimizer"].step()
    kwargs["optimizer"].zero_grad(set_to_none=True)

    with pytest.raises(
        SoulWriterCheckpointError,
        match="optimizer contains frozen parameter",
    ):
        save_soul_writer_checkpoint(
            tmp_path / "frozen-core-opt.pt",
            **kwargs,
        )


def test_save_rejects_changed_core_and_wrong_freeze(tmp_path: Path) -> None:
    changed = _live_state(tmp_path)
    with torch.no_grad():
        dict(changed["core"].named_parameters())[
            "action_to_soul.q.weight"
        ].add_(1.0)
    with pytest.raises(
        SoulWriterCheckpointError,
        match="non-allowlisted core tensor changed",
    ):
        save_soul_writer_checkpoint(tmp_path / "changed-core.pt", **changed)

    wrong_freeze_dir = tmp_path / "wrong-freeze"
    wrong_freeze_dir.mkdir()
    wrong_freeze = _live_state(wrong_freeze_dir)
    dict(wrong_freeze["core"].named_parameters())[
        "layers.0.attn.qkv.weight"
    ].requires_grad_(True)
    with pytest.raises(
        SoulWriterCheckpointError,
        match="exact reader allowlist",
    ):
        save_soul_writer_checkpoint(
            tmp_path / "wrong-freeze.pt",
            **wrong_freeze,
        )


def test_save_rejects_writer_only_policy_without_reader_allowlist(
    tmp_path: Path,
) -> None:
    kwargs = _live_state(tmp_path, train_read_path=False)
    with pytest.raises(
        SoulWriterCheckpointError,
        match="exact reader allowlist",
    ):
        save_soul_writer_checkpoint(
            tmp_path / "writer-only-policy.pt",
            **kwargs,
        )


def test_save_rejects_non_authoritative_soul_layout(tmp_path: Path) -> None:
    kwargs = _live_state(tmp_path)
    wrong_cfg = SoulV2Config(
        d_model=16,
        tiers=[
            TierSpec(name="hot", max_rows=128),
            TierSpec(name="warm", max_rows=32),
            TierSpec(name="cold", max_rows=7),
        ],
        categories=["episodic"],
    )
    wrong = SoulState(wrong_cfg, torch.device("cpu"), torch.float32)
    wrong.active[:128] = True
    kwargs["soul_state"] = wrong
    kwargs["tick_state"]["soul_tick"] = wrong.current_tick
    with pytest.raises(
        SoulWriterCheckpointError,
        match="128-hot/32-warm/8-cold",
    ):
        save_soul_writer_checkpoint(tmp_path / "wrong-soul.pt", **kwargs)


def _write_corrupt(
    valid_path: Path,
    destination: Path,
    mutation,
) -> Path:
    payload = torch.load(valid_path, map_location="cpu", weights_only=False)
    mutation(payload)
    torch.save(payload, destination)
    return destination


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    [
        (
            "missing-rng",
            lambda payload: payload.pop("rng"),
            "checkpoint fields mismatch",
        ),
        (
            "promotable",
            lambda payload: payload.__setitem__("promotable", True),
            "must remain non-promotable",
        ),
        (
            "quarantine",
            lambda payload: payload["quarantine"].__setitem__(
                "quarantined", False
            ),
            "quarantine provenance",
        ),
        (
            "completed",
            lambda payload: payload.__setitem__("completed", 1),
            "completed flag",
        ),
        (
            "inheritance-hash",
            lambda payload: payload["source_inheritance"].__setitem__(
                "manifest_sha256", "0" * 64
            ),
            "inheritance proof/hash",
        ),
        (
            "core-byte",
            lambda payload: payload["core"]["state"][
                next(iter(payload["core"]["state"]))
            ].reshape(-1).__setitem__(0, 99.0),
            "core state/hash",
        ),
        (
            "writer-shape",
            lambda payload: payload["writer"]["state"].__setitem__(
                "gate", torch.zeros(2)
            ),
            "writer state",
        ),
        (
            "optimizer-map",
            lambda payload: payload["optimizer"]["named_parameter_map"].pop(),
            "optimizer",
        ),
        (
            "optimizer-state",
            lambda payload: payload["optimizer"]["state_dict"][
                "state"
            ].pop(next(iter(payload["optimizer"]["state_dict"]["state"]))),
            "optimizer",
        ),
        (
            "soul-row",
            lambda payload: payload["soul"]["payload"].__setitem__(
                "tensor", payload["soul"]["payload"]["tensor"][:-1]
            ),
            "soul",
        ),
        (
            "rng-state",
            lambda payload: payload["rng"]["state"].pop("torch"),
            "RNG",
        ),
        (
            "curriculum",
            lambda payload: payload["continuity"].pop("curriculum"),
            "continuity envelope",
        ),
        (
            "sampler",
            lambda payload: payload["continuity"]["sampler"][
                "state"
            ].pop("order"),
            "sampler",
        ),
        (
            "tick",
            lambda payload: payload["continuity"]["tick"]["state"].__setitem__(
                "soul_tick", 999
            ),
            "tick",
        ),
        (
            "freeze",
            lambda payload: payload["freeze_manifest"][
                "trainable_names"
            ].pop(),
            "freeze manifest",
        ),
    ],
)
def test_load_fails_closed_on_omissions_and_corruption(
    tmp_path: Path,
    name: str,
    mutation,
    message: str,
) -> None:
    valid, _ = _save_valid(tmp_path, "valid.pt")
    corrupt = _write_corrupt(
        valid,
        tmp_path / f"{name}.pt",
        mutation,
    )
    with pytest.raises(SoulWriterCheckpointError, match=message):
        load_soul_writer_checkpoint(corrupt)


def test_load_rejects_rehashed_non_allowlisted_core_mutation(
    tmp_path: Path,
) -> None:
    valid, _ = _save_valid(tmp_path, "valid-rehashed.pt")
    corrupt = tmp_path / "rehashed-non-reader.pt"
    payload = torch.load(valid, map_location="cpu", weights_only=False)
    with torch.no_grad():
        payload["core"]["state"]["action_to_soul.q.weight"].add_(1.0)
    payload["core"]["state_manifest"] = core_state_tensor_manifest(
        payload["core"]["state"]
    )
    torch.save(payload, corrupt)

    with pytest.raises(
        SoulWriterCheckpointError,
        match="non-allowlisted core tensor changed",
    ):
        load_soul_writer_checkpoint(corrupt)


def test_atomic_failure_leaves_no_destination_or_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _live_state(tmp_path)
    destination = tmp_path / "atomic-failure.pt"

    def fail_save(payload, path):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "training.soul_writer_checkpoint.torch.save",
        fail_save,
    )
    with pytest.raises(OSError, match="simulated"):
        save_soul_writer_checkpoint(destination, **kwargs)

    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_atomic_save_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path, kwargs = _save_valid(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_soul_writer_checkpoint(path, **kwargs)
    assert path.read_bytes() == original
