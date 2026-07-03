"""Smoke tests for training/trainer_slot.py.

These tests verify that the train-as-you-live loop runs on CPU, loss falls,
and the smoke gate passes with a tiny config.  They deliberately do NOT
launch GPU runs or long curricula.
"""
from __future__ import annotations

import argparse
import random
import shutil
import tempfile
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
        "core_cfg": "64,1,1,256",
        "device": "cpu",
        "fp32": True,
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "dropout": 0.0,
        "steps": 30,
        "smoke": True,
        "smoke_steps": 30,
        "curriculum_dir": None,
        "containers_path": None,
        "recall_curriculum": "",
        "recall_synthetic": 200,
        "run_dir": tempfile.mkdtemp(prefix="slot_smoke_"),
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
        "cf_probe_every": 30,
        "cf_probe_n": 8,
        "checkpoint_every": 10 ** 9,
        "log_every": 10,
        "collapse_threshold": 1e-6,
        "resume": "",
        "seed": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_train_smoke_phase0() -> None:
    """Fast CPU smoke: loss falls and smoke gate passes."""
    args = _smoke_namespace(mode="phase0", steps=30)
    rng = _save_rng_state()
    try:
        train(args)
    finally:
        _restore_rng_state(rng)
    shutil.rmtree(args.run_dir, ignore_errors=True)


def test_train_smoke_recall_runs() -> None:
    """Recall mode runs through the cf_probe harness without crashing."""
    args = _smoke_namespace(mode="recall", steps=30, eval_every=30, cf_probe_every=30)
    rng = _save_rng_state()
    try:
        train(args)
    finally:
        _restore_rng_state(rng)
    shutil.rmtree(args.run_dir, ignore_errors=True)


def test_train_smoke_both_modes() -> None:
    args = _smoke_namespace(mode="both", steps=30)
    rng = _save_rng_state()
    try:
        train(args)
    finally:
        _restore_rng_state(rng)
    shutil.rmtree(args.run_dir, ignore_errors=True)
