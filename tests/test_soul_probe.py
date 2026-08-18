from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from training.soul_probe import summarize_checkpoint


def _fake_checkpoint(path: Path) -> None:
    tensor = torch.zeros((6, 4), dtype=torch.float32)
    tensor[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tensor[1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    tensor[3] = torch.tensor([0.0, 0.0, 3.0, 0.0])
    torch.save(
        {
            "step": 123,
            "cfg": {"d_model": 4, "n_layers": 1, "n_heads": 1, "ffn_dim": 16},
            "soul_state": {
                "tensor": tensor,
                "active": torch.tensor([True, True, False, True, False, False]),
                "tier": torch.tensor([0, 0, 0, 1, 1, 2]),
                "category": torch.tensor([0, 1, -1, -1, -1, -1]),
                "salience": torch.tensor([0.4, 0.2, 0.0, 0.9, 0.0, 0.0]),
                "dormant_for": torch.tensor([0, 2, 0, 0, 0, 0]),
                "tick_born": torch.tensor([10, 11, 0, 12, 0, 0]),
                "current_tick": 50,
                "cfg": {
                    "d_model": 4,
                    "tiers": [
                        {"name": "hot", "max_rows": 3},
                        {"name": "warm", "max_rows": 2},
                        {"name": "cold", "max_rows": 1},
                    ],
                    "categories": ["episodic", "lessons"],
                },
            },
        },
        path,
    )


def test_soul_probe_summarizes_tiers_without_vectors(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    _fake_checkpoint(ckpt)

    summary = summarize_checkpoint(ckpt, top_rows=2)

    assert summary["checkpoint_step"] == 123
    assert summary["soul"]["tensor_shape"] == [6, 4]
    assert summary["soul"]["active_rows"] == 3
    assert summary["tiers"][0]["name"] == "hot"
    assert summary["tiers"][0]["active"] == 2
    assert summary["tiers"][1]["active"] == 1
    assert summary["categories"][0]["name"] == "episodic"
    assert summary["categories"][0]["active"] == 1
    assert "vector" in " ".join(summary["notes"]).lower()
    assert "tensor" not in summary["top_active_rows"][0]


def test_soul_probe_cli_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    _fake_checkpoint(ckpt)

    proc = subprocess.run(
        [
            sys.executable,
            "training/soul_probe.py",
            "--checkpoint",
            str(ckpt),
            "--top-rows",
            "1",
            "--json",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(proc.stdout)
    assert data["kind"] == "axon_soul_probe"
    assert len(data["top_active_rows"]) == 1
