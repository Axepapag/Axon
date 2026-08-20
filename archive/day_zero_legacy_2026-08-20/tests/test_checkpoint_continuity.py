from __future__ import annotations

import json
import random

import torch

from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulManagerV2, SoulV2Config, TierSpec
from training.trainer_slot import CheckpointManager, Phase0bCurriculum


def _write_curriculum(tmp_path) -> None:
    rows = []
    for i in range(12):
        rows.append(
            {
                "family": "runtime_response_delta_v1",
                "source": f"source-{i}",
                "active_field": {"user_input": f"question {i}", "response_draft": ""},
                "target_delta": {"region": "response_draft", "op": "replace", "text": f"answer {i}"},
            }
        )
    path = tmp_path / "runtime_response_delta_v1.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _curriculum(tmp_path) -> Phase0bCurriculum:
    return Phase0bCurriculum(
        str(tmp_path),
        "runtime_response_delta_v1",
        max_chars=64,
        rng=random.Random(17),
    )


def test_phase0b_sampler_state_resumes_exact_next_examples(tmp_path) -> None:
    _write_curriculum(tmp_path)
    first = _curriculum(tmp_path)
    for _ in range(17):
        first.next()
    state = first.state_dict()
    expected = [first.next()["source"] for _ in range(20)]

    resumed = _curriculum(tmp_path)
    resumed.load_state_dict(state)
    actual = [resumed.next()["source"] for _ in range(20)]
    assert actual == expected


def test_final_checkpoint_contains_optimizer_and_training_state(tmp_path) -> None:
    cfg = CoreConfig(
        d_model=16,
        n_layers=1,
        n_heads=1,
        ffn_dim=32,
        soul_mode="act_reflect_v2",
        soul_rows=2,
        soul_hot_rows=2,
        char_slot_mode=True,
        char_slot_max_slots=4,
    )
    core = AxonCore(cfg)
    soul_cfg = SoulV2Config(
        d_model=16,
        tiers=[TierSpec("hot", 2), TierSpec("warm", 1), TierSpec("cold", 1)],
        categories=["episodic"],
    )
    soul_mgr = SoulManagerV2(soul_cfg, torch.device("cpu"), torch.float32, n_heads=1)
    optimizer = torch.optim.AdamW(list(core.parameters()) + list(soul_mgr.parameters()), lr=1e-3)
    curriculum = type("Curriculum", (), {"state_dict": lambda self: {"schema": "test_sampler"}})()
    training_provenance = {"schema": "axon_training_provenance_v1", "source_checkpoint_step": 200000}
    continuity_provenance = {
        "schema": "axon_continuity_provenance_v1",
        "source_step": 200000,
        "optimizer": {
            "policy": "restore_state_override_lr",
            "state_restored": True,
            "effective_lr": 1e-4,
        },
        "sampler": {
            "policy": "reset_exact_v3",
            "reset": True,
            "initial_position": 0,
            "families": ["runtime_response_delta_chain_v3"],
        },
    }

    manager = CheckpointManager(tmp_path, keep=1)
    path = manager.save(
        core,
        soul_mgr,
        cfg,
        soul_cfg,
        250000,
        optimizer=optimizer,
        curriculum=curriculum,
        include_optimizer=True,
        training_provenance=training_provenance,
        continuity_provenance=continuity_provenance,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    done = json.loads((tmp_path / "checkpoint_done.json").read_text(encoding="utf-8"))

    assert payload["checkpoint_schema"] == "axon_charslot_checkpoint_v2"
    assert "optimizer_state" in payload
    assert payload["training_state"]["curriculum_state"] == {"schema": "test_sampler"}
    assert payload["training_provenance"] == training_provenance
    assert payload["continuity_provenance"] == continuity_provenance
    assert done == {"step": 250000, "time": done["time"], "optimizer_state": True}
