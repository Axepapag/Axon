from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cores.core import AxonCore, CoreConfig
from runtime.axon_runtime.bootstrap import (
    BootstrapDependencies,
    RuntimeAlreadyRunningError,
    RuntimeConfigMismatchError,
    RuntimeIdentityMismatchError,
    RuntimeModelPinError,
    RuntimeRunnerLock,
    UnsupportedRuntimeConfigError,
    materialize_runtime,
)
from runtime.axon_runtime.checkpoint import (
    CheckpointSourceEvidence,
    ExactV4Checkpoint,
    HOT_ONLY,
    inference_core_state_sha256,
)
from runtime.axon_runtime.config import parse_runtime_config
from runtime.axon_runtime.core_backend import CoreBackend
from runtime.axon_runtime.ingress import IngressEvent
from runtime.field import CANONICAL_REGION_ORDER, canonical_sha256


def _tiny_checkpoint(
    path: Path,
    *,
    d_model: int,
    step: int,
    seed: int,
) -> ExactV4Checkpoint:
    cfg = CoreConfig(
        d_model=d_model,
        n_heads=1,
        n_layers=1,
        ffn_dim=d_model * 2,
        dropout=0.0,
        n_ticks=1,
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
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        core = AxonCore(cfg)
        soul = torch.randn(cfg.total_soul_rows(), cfg.d_model)
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    core.eval()
    return ExactV4Checkpoint(
        path=path,
        checkpoint_sha256=canonical_sha256(
            ["synthetic-checkpoint", d_model, step, seed]
        ),
        core_state_sha256=inference_core_state_sha256(core.state_dict()),
        byte_length=0,
        step=step,
        cfg=cfg,
        core=core,
        initial_soul=soul,
        initial_soul_mask=torch.ones(
            cfg.total_soul_rows(),
            dtype=torch.bool,
        ),
        soul_compressor=None,
        soul_tier_status=HOT_ONLY,
        mmap_used=False,
        source_evidence=CheckpointSourceEvidence(
            optimizer_state_present=True,
            rng_state_keys=(
                "numpy_random_state",
                "python_random_state",
                "torch_rng_state",
            ),
        ),
    )


def _config_mapping(
    tmp_path: Path,
    checkpoints: dict[str, ExactV4Checkpoint],
) -> dict:
    state = tmp_path / "state"
    model_16 = checkpoints["model-16"]
    model_32 = checkpoints["model-32"]
    budgets = {
        region.value: 512 for region in CANONICAL_REGION_ORDER
    }
    return {
        "schema": "axon-runtime-bootstrap-config-v1",
        "models": [
            {
                "model_id": "model-16",
                "d_model": 16,
                "checkpoint_step": model_16.step,
                "checkpoint_path": str(model_16.path),
                "checkpoint_sha256": model_16.checkpoint_sha256,
            },
            {
                "model_id": "model-32",
                "d_model": 32,
                "checkpoint_step": model_32.step,
                "checkpoint_path": str(model_32.path),
                "checkpoint_sha256": model_32.checkpoint_sha256,
            },
        ],
        "cores": [
            {
                "core_id": "core-16-a",
                "display_name": "Core 16 A",
                "model_id": "model-16",
                "soul_id": "soul-16-a",
                "adapter_namespace": "adapter.core-16-a",
                "enabled": True,
                "lineage": ["exact-v4", "clone-a"],
                "role_capabilities": [
                    "proposer",
                    "consolidator",
                    "sleeper",
                ],
            },
            {
                "core_id": "core-32-a",
                "display_name": "Core 32 A",
                "model_id": "model-32",
                "soul_id": "soul-32-a",
                "adapter_namespace": "adapter.core-32-a",
                "enabled": True,
                "lineage": ["exact-v4", "clone-a"],
                "role_capabilities": [
                    "proposer",
                    "consolidator",
                    "sleeper",
                ],
            },
            {
                "core_id": "core-16-b",
                "display_name": "Core 16 B",
                "model_id": "model-16",
                "soul_id": "soul-16-b",
                "adapter_namespace": "adapter.core-16-b",
                "enabled": True,
                "lineage": ["exact-v4", "clone-b"],
                "role_capabilities": [
                    "proposer",
                    "consolidator",
                    "sleeper",
                ],
            },
            {
                "core_id": "core-32-b",
                "display_name": "Core 32 B",
                "model_id": "model-32",
                "soul_id": "soul-32-b",
                "adapter_namespace": "adapter.core-32-b",
                "enabled": True,
                "lineage": ["exact-v4", "clone-b"],
                "role_capabilities": [
                    "proposer",
                    "consolidator",
                    "sleeper",
                ],
            },
        ],
        "ring": {
            "mode": "round_robin",
            "core_ids": [
                "core-16-a",
                "core-32-a",
                "core-16-b",
                "core-32-b",
            ],
        },
        "paths": {
            "state_root": str(state),
            "runtime_db": str(state / "runtime.sqlite3"),
            "soul_store": str(state / "souls"),
            "private_state_store": str(state / "private"),
            "dormant_db": str(state / "dormant.sqlite3"),
            "stop_file": str(state / "STOP"),
        },
        "projection": {
            "global_char_budget": 4096,
            "max_selected_spans": 256,
            "max_read_pages_per_action": 32,
            "region_char_budgets": budgets,
            "pinned_regions": [
                "user_input",
                "response_draft",
                "scratch",
                "tool_results",
            ],
        },
        "idle": {
            "capture_limit_per_quantum": 4,
            "job_limit_per_quantum": 2,
            "lease_ticks": 2,
            "max_quanta_per_tick": 1,
        },
        "loop": {
            "continuous": True,
            "tick_interval_ms": 0,
            "idle_sleep_ms": 10,
            "failure_backoff_initial_ms": 10,
            "failure_backoff_max_ms": 100,
            "failure_backoff_multiplier": 2.0,
            "stop_file_poll_ms": 10,
        },
        "capabilities": {
            "typed_tools_enabled": False,
            "advisors_enabled": False,
        },
    }


def _foundation(tmp_path: Path):
    checkpoints = {
        "model-16": _tiny_checkpoint(
            tmp_path / "model-16.pt",
            d_model=16,
            step=11,
            seed=16,
        ),
        "model-32": _tiny_checkpoint(
            tmp_path / "model-32.pt",
            d_model=32,
            step=22,
            seed=32,
        ),
    }
    mapping = _config_mapping(tmp_path, checkpoints)
    config = parse_runtime_config(json.dumps(mapping))
    calls: list[tuple[str, str]] = []

    def load(
        backend: CoreBackend,
        pin,
        device: str,
    ):
        calls.append((pin.model_id, device))
        return backend.register_loaded_model(
            pin.model_id,
            checkpoints[pin.model_id],
        )

    dependencies = BootstrapDependencies(model_loader=load)
    return checkpoints, mapping, config, dependencies, calls


def test_production_materialization_fails_closed_on_legacy_exact_v4(
    tmp_path: Path,
) -> None:
    _, _, config, _, _ = _foundation(tmp_path)
    with pytest.raises(
        UnsupportedRuntimeConfigError,
        match="not canonical Axon anatomy",
    ):
        materialize_runtime(config)
    assert not config.paths.state_root.exists()


def test_materialize_new_tick_close_and_recover_without_real_checkpoints(
    tmp_path: Path,
) -> None:
    checkpoints, _, config, dependencies, calls = _foundation(tmp_path)
    assert not config.paths.state_root.exists()

    runtime = materialize_runtime(config, dependencies=dependencies)
    try:
        assert runtime.initialized_new_store is True
        assert calls == [("model-16", "cpu"), ("model-32", "cpu")]
        assert runtime.initial_head.tick_seq == 0
        assert runtime.initial_head.snapshot.tick_id == 0
        assert runtime.initial_head.snapshot.parent_field_id is None
        assert runtime.initial_head.snapshot.source_manifest_ids == (
            config.config_id,
        )
        assert len(runtime.initial_head.snapshot.regions) == len(
            CANONICAL_REGION_ORDER
        )
        assert all(
            not region.text for region in runtime.initial_head.snapshot.regions
        )
        assert runtime.store.persisted_identities() == runtime.identities
        assert tuple(item.core_id for item in runtime.identities) == (
            "core-16-a",
            "core-32-a",
            "core-16-b",
            "core-32-b",
        )
        assert {
            item.core_state_sha256 for item in runtime.identities
        } == {
            checkpoint.core_state_sha256
            for checkpoint in checkpoints.values()
        }
        first = runtime.backend.logical_core("core-16-a")
        second = runtime.backend.logical_core("core-16-b")
        assert first.shared_model is second.shared_model
        assert first.soul.data_ptr() != second.soul.data_ptr()
        assert first.adapter_manifest["namespace"] == "adapter.core-16-a"
        assert second.adapter_manifest["namespace"] == "adapter.core-16-b"
        assert first.rng_manifest["stream_id"] == "core-16-a"
        assert second.rng_manifest["stream_id"] == "core-16-b"

        runtime.ingress_queue.enqueue(
            IngressEvent(
                idempotency_key="bootstrap-user-0",
                source="test-user",
                source_sequence=0,
                event_kind="user_input",
                exact_text="hello",
                provenance="test:bootstrap",
            )
        )
        result = runtime.engine.tick_once()
        assert result.commit.tick_seq == 0
        assert runtime.store.recover().tick_seq == 1
        assert len(runtime.ingress_queue.receipts()) == 1

        with pytest.raises(RuntimeAlreadyRunningError):
            materialize_runtime(config, dependencies=dependencies)
    finally:
        runtime.close()

    restarted = materialize_runtime(config, dependencies=dependencies)
    try:
        assert restarted.initialized_new_store is False
        assert restarted.initial_head.tick_seq == 1
        assert restarted.store.persisted_identities() == restarted.identities
        for manifest in restarted.initial_head.core_state_manifests:
            logical = restarted.backend.logical_core(manifest.core_id)
            assert logical.committed_field_id == manifest.committed_field_id
            assert (
                restarted.private_state_store.load(
                    manifest.cursor_state_sha256
                ).soul_blob_sha256
                == manifest.soul_state_sha256
            )
    finally:
        restarted.close()


def test_existing_store_rejects_config_identity_drift_fail_closed(
    tmp_path: Path,
) -> None:
    _, mapping, config, dependencies, _ = _foundation(tmp_path)
    with materialize_runtime(config, dependencies=dependencies):
        pass

    drifted = json.loads(json.dumps(mapping))
    drifted["cores"][0]["display_name"] = "Drifted Core Name"
    drifted_config = parse_runtime_config(json.dumps(drifted))
    with pytest.raises(RuntimeIdentityMismatchError):
        materialize_runtime(drifted_config, dependencies=dependencies)

    # Failure released the process lock and did not advance canonical truth.
    with materialize_runtime(config, dependencies=dependencies) as recovered:
        assert recovered.initial_head.tick_seq == 0
        assert recovered.store.persisted_identities() == recovered.identities


def test_existing_store_rejects_nonidentity_config_drift(
    tmp_path: Path,
) -> None:
    _, mapping, config, dependencies, _ = _foundation(tmp_path)
    with materialize_runtime(config, dependencies=dependencies):
        pass

    drifted = json.loads(json.dumps(mapping))
    drifted["loop"]["tick_interval_ms"] = 17
    drifted_config = parse_runtime_config(json.dumps(drifted))
    with pytest.raises(RuntimeConfigMismatchError):
        materialize_runtime(drifted_config, dependencies=dependencies)

    with materialize_runtime(config, dependencies=dependencies) as recovered:
        assert recovered.initial_head.tick_seq == 0


@pytest.mark.parametrize("field,value", [("checkpoint_step", 12), ("d_model", 17)])
def test_checkpoint_step_and_width_pins_fail_before_store_initialization(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    _, mapping, _, dependencies, _ = _foundation(tmp_path)
    mapping["models"][0][field] = value
    config = parse_runtime_config(json.dumps(mapping))

    with pytest.raises(RuntimeModelPinError):
        materialize_runtime(config, dependencies=dependencies)
    assert not config.paths.runtime_db.exists()

    # The failed attempt also released the runner lock.
    lock = RuntimeRunnerLock(
        config.paths.state_root / "axon-runtime.runner.lock"
    )
    lock.acquire(config_id=config.config_id)
    lock.release()


def test_runner_lock_blocks_a_second_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "runner.lock"
    lock = RuntimeRunnerLock(lock_path)
    lock.acquire(config_id="a" * 64)
    try:
        code = """
import sys
from runtime.axon_runtime.bootstrap import (
    RuntimeAlreadyRunningError,
    RuntimeRunnerLock,
)
lock = RuntimeRunnerLock(sys.argv[1])
try:
    lock.acquire(config_id="b" * 64)
except RuntimeAlreadyRunningError:
    print("blocked")
else:
    print("acquired")
    lock.release()
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(lock_path)],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.stdout.strip() == "blocked"
    finally:
        lock.release()
