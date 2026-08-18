from __future__ import annotations

import json
from pathlib import Path

from runtime.axon_runtime.cli import (
    enqueue_event,
    runtime_status,
    write_stop_marker,
)
from runtime.axon_runtime.config import parse_runtime_config
from runtime.axon_runtime.contracts import CoreIdentity, CoreStateManifest
from runtime.axon_runtime.store import RuntimeStore
from runtime.field import CANONICAL_REGION_ORDER, SharedFieldSnapshot, canonical_sha256


def _config(tmp_path: Path):
    state = tmp_path / "state"
    cores = []
    ring = ("core-16-a", "core-32-a", "core-16-b", "core-32-b")
    for core_id, model_id in zip(
        ring,
        ("model-16", "model-32", "model-16", "model-32"),
        strict=True,
    ):
        cores.append(
            {
                "core_id": core_id,
                "display_name": core_id,
                "model_id": model_id,
                "soul_id": f"soul-{core_id}",
                "adapter_namespace": f"adapter.{core_id}",
                "enabled": True,
                "lineage": ["exact-v4", core_id],
                "role_capabilities": [
                    "proposer",
                    "consolidator",
                    "sleeper",
                ],
            }
        )
    mapping = {
        "schema": "axon-runtime-bootstrap-config-v1",
        "models": [
            {
                "model_id": "model-16",
                "d_model": 16,
                "checkpoint_step": 1,
                "checkpoint_path": str(tmp_path / "model-16.pt"),
                "checkpoint_sha256": canonical_sha256("checkpoint-16"),
            },
            {
                "model_id": "model-32",
                "d_model": 32,
                "checkpoint_step": 1,
                "checkpoint_path": str(tmp_path / "model-32.pt"),
                "checkpoint_sha256": canonical_sha256("checkpoint-32"),
            },
        ],
        "cores": cores,
        "ring": {"mode": "round_robin", "core_ids": list(ring)},
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
            "region_char_budgets": {
                region.value: 512 for region in CANONICAL_REGION_ORDER
            },
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
    return parse_runtime_config(json.dumps(mapping))


def _initialize(config) -> None:
    config.paths.state_root.mkdir(parents=True)
    identities = []
    for descriptor in config.cores:
        pin = config.model_pin(descriptor.model_id)
        identities.append(
            CoreIdentity(
                core_id=descriptor.core_id,
                display_name=descriptor.display_name,
                base_checkpoint_path=str(pin.checkpoint_path),
                base_checkpoint_sha256=pin.checkpoint_sha256,
                model_id=descriptor.model_id,
                core_state_sha256=canonical_sha256(
                    ["core-state", descriptor.model_id]
                ),
                lineage=descriptor.lineage,
                soul_id=descriptor.soul_id,
                adapter_namespace=descriptor.adapter_namespace,
                role_capabilities=descriptor.role_capabilities,
            )
        )
    genesis = SharedFieldSnapshot.empty(
        source_manifest_ids=(config.config_id,)
    )
    states = tuple(
        CoreStateManifest(
            core_id=identity.core_id,
            soul_id=identity.soul_id,
            model_id=identity.model_id,
            core_state_sha256=identity.core_state_sha256,
            generation=0,
            tick_seq=0,
            committed_field_id=genesis.field_id,
            soul_state_sha256=canonical_sha256(
                [identity.core_id, "soul"]
            ),
            cursor_state_sha256=canonical_sha256(
                [identity.core_id, "private"]
            ),
            adapter_set_sha256=canonical_sha256(
                [identity.core_id, "adapter"]
            ),
            rng_state_sha256=canonical_sha256(
                [identity.core_id, "rng"]
            ),
        )
        for identity in identities
    )
    with RuntimeStore(config.paths.runtime_db) as store:
        store.initialize(genesis, tuple(identities), states)


def test_status_on_missing_runtime_is_read_only(tmp_path: Path) -> None:
    config = _config(tmp_path)

    status = runtime_status(config)

    assert status["initialized"] is False
    assert not config.paths.state_root.exists()


def test_stop_marker_is_atomic_and_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first = write_stop_marker(config)
    second = write_stop_marker(config)

    assert first["created"] is True
    assert second["created"] is False
    payload = json.loads(config.paths.stop_file.read_text(encoding="ascii"))
    assert payload["config_id"] == config.config_id


def test_enqueue_allocates_sequences_and_status_reports_pending(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _initialize(config)

    first = enqueue_event(
        config,
        event_kind="user_input",
        exact_text="hello",
        idempotency_key="cli-one",
    )
    retry = enqueue_event(
        config,
        event_kind="user_input",
        exact_text="hello",
        idempotency_key="cli-one",
    )
    second = enqueue_event(
        config,
        event_kind="diary",
        exact_text="remember this",
    )
    status = runtime_status(config)

    assert first == retry
    assert first.source_sequence == 0
    assert second.source_sequence == 1
    assert status["initialized"] is True
    assert status["pending_ingress"] == 2
    assert status["ring_matches_config"] is True
    assert status["next_roles"] == {
        "proposer": "core-16-a",
        "consolidator": "core-32-b",
        "sleeper": "core-16-b",
        "standby": ["core-32-a"],
    }

