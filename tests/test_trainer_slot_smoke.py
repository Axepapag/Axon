"""Smoke tests for the cleaned 16D charfield trainer."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from training.trainer_slot import train


def _save_rng_state() -> tuple[Any, Any, Any]:
    return torch.get_rng_state(), np.random.get_state(), random.getstate()


def _restore_rng_state(state: tuple[Any, Any, Any]) -> None:
    torch.set_rng_state(state[0])
    np.random.set_state(state[1])
    random.setstate(state[2])


def _smoke_namespace(**overrides) -> argparse.Namespace:
    defaults = {
        "mode": "phase0",
        "threshold": "charslot",
        "history_chars": 64,
        "user_chars": 32,
        "charslot_mode_weights": "0.25,0.50,0.25",
        "charslot_rung_preset": "manual",
        "charslot_partial_frac": 0.5,
        "phase0_target_words": 0,
        "text_corpus": "",
        "text_corpus_weight": 0.7,
        "text_corpus_max": 1000,
        "core_cfg": "64,1,1,256",
        "device": "cpu",
        "fp32": True,
        "fp16": False,
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "grad_checkpoint": False,
        "dropout": 0.0,
        "steps": 30,
        "smoke": True,
        "smoke_steps": 30,
        "draft_teacher_steps": 30,
        "draft_mask_ramp_steps": 60,
        "history_turns": 10,
        "curriculum_dir": None,
        "containers_path": None,
        "run_dir": tempfile.mkdtemp(prefix="charslot_smoke_"),
        "max_response_chars": 16,
        "soul_rows": 8,
        "soul_hot_rows": 8,
        "n_soul_compartments": 2,
        "soul_gate_init": 0.05,
        "hot_rows": 16,
        "warm_rows": 4,
        "cold_rows": 2,
        "router_threshold": 0.5,
        "write_gate_init": 0.1,
        "eval_every": 30,
        "eval_n": 8,
        "eval_samples": 0,
        "checkpoint_every": 10 ** 9,
        "log_every": 10,
        "collapse_threshold": 1e-6,
        "resume": "",
        "seed": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_train_smoke_phase0() -> None:
    """Fast CPU smoke: the exact 16D charfield loop runs end to end."""
    args = _smoke_namespace(mode="phase0", steps=30)
    rng = _save_rng_state()
    try:
        train(args)
    finally:
        _restore_rng_state(rng)
    shutil.rmtree(args.run_dir, ignore_errors=True)


def _write_phase0b_fixture(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "family": "runtime_response_delta_v1",
            "active_field": {
                "conversation_history": "user: inspect",
                "structured_knowledge": "fact: focused tests are useful",
                "user_input": "inspect repo",
                "response_draft": "",
            },
            "target_delta": {"region": "response_draft", "op": "replace", "text": "run tests"},
        },
        {
            "family": "scratchpad_delta_v1",
            "active_field": {
                "conversation_history": "",
                "structured_knowledge": "procedure: debug\nstep: read traceback",
                "scratch": "",
                "user_input": "route failed",
                "response_draft": "",
            },
            "target_delta": {"region": "scratch", "op": "replace", "text": "next step: run focused test"},
        },
    ]
    for family in ("runtime_response_delta_v1", "scratchpad_delta_v1"):
        with (path / f"{family}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                if row["family"] == family:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def test_train_smoke_phase0b() -> None:
    phase0b_dir = Path(tempfile.mkdtemp(prefix="phase0b_fixture_"))
    _write_phase0b_fixture(phase0b_dir)
    args = _smoke_namespace(
        mode="phase0b",
        steps=20,
        smoke_steps=20,
        phase0b_dir=str(phase0b_dir),
        phase0b_families="runtime_response_delta_v1,scratchpad_delta_v1",
        phase0b_max_examples=10,
        max_response_chars=32,
        eval_every=20,
    )
    rng = _save_rng_state()
    try:
        train(args)
    finally:
        _restore_rng_state(rng)
        shutil.rmtree(args.run_dir, ignore_errors=True)
        shutil.rmtree(phase0b_dir, ignore_errors=True)


def test_resume_rejects_core_shape_mismatch() -> None:
    args = _smoke_namespace(
        mode="phase0",
        steps=1,
        smoke=False,
        eval_every=10 ** 9,
        checkpoint_every=10 ** 9,
        eval_samples=0,
    )
    rng = _save_rng_state()
    try:
        train(args)
        ckpt = Path(args.run_dir) / "ckpt_0.pt"
        bad_resume = _smoke_namespace(
            mode="phase0",
            steps=2,
            smoke=True,
            core_cfg="32,1,1,128",
            resume=str(ckpt),
            eval_samples=0,
        )
        with pytest.raises(ValueError, match="core config mismatch"):
            train(bad_resume)
    finally:
        _restore_rng_state(rng)
        shutil.rmtree(args.run_dir, ignore_errors=True)
        if "bad_resume" in locals():
            shutil.rmtree(bad_resume.run_dir, ignore_errors=True)


def test_train_rejects_removed_modes() -> None:
    args = _smoke_namespace(mode="recall", steps=1)
    try:
        with pytest.raises(ValueError, match="phase0, phase0b, or phase1a only"):
            train(args)
    finally:
        shutil.rmtree(args.run_dir, ignore_errors=True)
