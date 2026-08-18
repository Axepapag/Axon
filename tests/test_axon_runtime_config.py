from __future__ import annotations

from dataclasses import FrozenInstanceError
import copy
import json
from pathlib import Path

import pytest

from runtime.axon_runtime.config import (
    RuntimeBootstrapConfig,
    RuntimeConfigError,
    load_runtime_config,
    parse_runtime_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CPU_SMOKE_CONFIG = REPO_ROOT / "ops" / "axon_runtime.cpu-smoke.json"
EXPECTED_MODEL_PINS = {
    "core64d-exact-v4-416k": {
        "d_model": 64,
        "checkpoint_step": 416_000,
        "checkpoint_sha256": (
            "8fb8f4e8042d9b4687451c2b530a85dc4100e0488b4a67b04c30860afff1c5d1"
        ),
        "checkpoint_name": "ckpt_416000.pt",
    },
    "core128d-exact-v4-252k": {
        "d_model": 128,
        "checkpoint_step": 252_000,
        "checkpoint_sha256": (
            "276fab962a485a52ef169082b83c24017d0f43f385ab2daf2e84a9b9923923a8"
        ),
        "checkpoint_name": "ckpt_252000.pt",
    },
}


def _mapping() -> dict[str, object]:
    return json.loads(CPU_SMOKE_CONFIG.read_text(encoding="utf-8"))


def _parse_mapping(value: dict[str, object]) -> RuntimeBootstrapConfig:
    return parse_runtime_config(json.dumps(value))


def test_cpu_smoke_config_pins_clones_ring_and_safe_bounds() -> None:
    config = load_runtime_config(CPU_SMOKE_CONFIG)

    assert {model.model_id for model in config.models} == set(
        EXPECTED_MODEL_PINS
    )
    for model in config.models:
        expected = EXPECTED_MODEL_PINS[model.model_id]
        assert model.d_model == expected["d_model"]
        assert model.checkpoint_step == expected["checkpoint_step"]
        assert model.checkpoint_sha256 == expected["checkpoint_sha256"]
        assert model.checkpoint_path.name == expected["checkpoint_name"]
        assert model.checkpoint_path.is_absolute()
        assert len(model.config_id) == 64
        assert not hasattr(model, "core_state_sha256")

    assert len(config.cores) == 4
    assert len({core.core_id for core in config.cores}) == 4
    assert len({core.soul_id for core in config.cores}) == 4
    assert len({core.adapter_namespace for core in config.cores}) == 4
    assert all(core.enabled for core in config.cores)
    assert all(
        set(core.role_capabilities) == {"proposer", "consolidator", "sleeper"}
        for core in config.cores
    )
    assert all(not hasattr(core, "core_state_sha256") for core in config.cores)
    assert {
        model_id: sum(core.model_id == model_id for core in config.cores)
        for model_id in EXPECTED_MODEL_PINS
    } == {
        "core64d-exact-v4-416k": 2,
        "core128d-exact-v4-252k": 2,
    }
    assert config.ring.core_ids == (
        "axon64-a",
        "axon128-a",
        "axon64-b",
        "axon128-b",
    )

    state_root = config.paths.state_root
    assert state_root.is_absolute()
    for path in (
        config.paths.runtime_db,
        config.paths.soul_store,
        config.paths.private_state_store,
        config.paths.dormant_db,
        config.paths.stop_file,
    ):
        assert path.is_absolute()
        assert path.is_relative_to(state_root)

    assert config.projection.global_char_budget == 4096
    assert config.projection.max_selected_spans == 256
    assert config.projection.max_read_pages_per_action == 32
    assert config.idle.max_quanta_per_tick == 1
    assert config.loop.continuous is True
    assert config.loop.failure_backoff_initial_ms <= (
        config.loop.failure_backoff_max_ms
    )
    assert config.capabilities.typed_tools_enabled is False
    assert config.capabilities.advisors_enabled is False


def test_config_is_frozen_and_canonical_round_trip_is_stable() -> None:
    config = load_runtime_config(CPU_SMOKE_CONFIG)

    with pytest.raises(FrozenInstanceError):
        config.models[0].d_model = 128  # type: ignore[misc]

    round_tripped = RuntimeBootstrapConfig.from_mapping(
        config.to_canonical_dict()
    )
    assert round_tripped == config
    assert round_tripped.config_id == config.config_id


def test_parse_does_not_touch_models_or_create_runtime_state(
    tmp_path: Path,
) -> None:
    value = _mapping()
    future_root = tmp_path / "not-created" / "axon_runtime"
    paths = value["paths"]
    assert isinstance(paths, dict)
    paths.update(
        {
            "state_root": str(future_root),
            "runtime_db": str(future_root / "runtime.sqlite3"),
            "soul_store": str(future_root / "souls"),
            "private_state_store": str(future_root / "private_states"),
            "dormant_db": str(future_root / "dormant.sqlite3"),
            "stop_file": str(future_root / "STOP"),
        }
    )

    config = _parse_mapping(value)

    assert config.paths.state_root == future_root.resolve()
    assert not future_root.exists()


def test_duplicate_json_keys_and_non_finite_numbers_are_rejected() -> None:
    text = CPU_SMOKE_CONFIG.read_text(encoding="utf-8")
    duplicate = text.replace(
        '"schema": "axon-runtime-bootstrap-config-v1",',
        (
            '"schema": "axon-runtime-bootstrap-config-v1",\n'
            '  "schema": "axon-runtime-bootstrap-config-v1",'
        ),
        1,
    )
    with pytest.raises(RuntimeConfigError, match="duplicate JSON key"):
        parse_runtime_config(duplicate)

    value = _mapping()
    loop = value["loop"]
    assert isinstance(loop, dict)
    loop["failure_backoff_multiplier"] = float("nan")
    with pytest.raises(RuntimeConfigError, match="non-finite JSON constant"):
        _parse_mapping(value)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.update({"unexpected": True}),
            "keys mismatch",
        ),
        (
            lambda value: value["models"][0].update(  # type: ignore[index,union-attr]
                {"checkpoint_sha256": "bad"}
            ),
            "64-character hexadecimal",
        ),
        (
            lambda value: value["cores"][0].update(  # type: ignore[index,union-attr]
                {"model_id": "missing-model"}
            ),
            "unknown model IDs",
        ),
        (
            lambda value: value["cores"][1].update(  # type: ignore[index,union-attr]
                {"soul_id": value["cores"][0]["soul_id"]}  # type: ignore[index]
            ),
            "soul_id values must be unique",
        ),
        (
            lambda value: value["ring"].update(  # type: ignore[union-attr]
                {"core_ids": value["ring"]["core_ids"][:-1]}  # type: ignore[index]
            ),
            "exactly cover enabled cores",
        ),
        (
            lambda value: value["capabilities"].update(  # type: ignore[union-attr]
                {"typed_tools_enabled": True}
            ),
            "typed tools must be globally disabled",
        ),
        (
            lambda value: value["loop"].update(  # type: ignore[union-attr]
                {"failure_backoff_initial_ms": 60_000}
            ),
            "initial failure backoff cannot exceed maximum",
        ),
    ],
)
def test_strict_cross_descriptor_contracts(
    mutate: object,
    match: str,
) -> None:
    value = copy.deepcopy(_mapping())
    mutate(value)  # type: ignore[operator]
    with pytest.raises(RuntimeConfigError, match=match):
        _parse_mapping(value)


def test_relative_outside_and_literal_secret_paths_are_rejected(
    tmp_path: Path,
) -> None:
    relative = _mapping()
    relative_paths = relative["paths"]
    assert isinstance(relative_paths, dict)
    relative_paths["runtime_db"] = "runtime.sqlite3"
    with pytest.raises(RuntimeConfigError, match="must be absolute"):
        _parse_mapping(relative)

    outside = _mapping()
    outside_paths = outside["paths"]
    assert isinstance(outside_paths, dict)
    outside_paths["runtime_db"] = str(tmp_path / "outside.sqlite3")
    with pytest.raises(
        RuntimeConfigError,
        match="child of state_root|different volume",
    ):
        _parse_mapping(outside)

    literal_secret = _mapping()
    literal_secret["api_key"] = "must-not-be-here"
    with pytest.raises(RuntimeConfigError, match="forbidden literal secret"):
        _parse_mapping(literal_secret)
