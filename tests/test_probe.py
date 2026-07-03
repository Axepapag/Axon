"""Tests for the read-fidelity probe over frozen adapter summaries."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from heads.probe import (
    ReadFidelityProbe,
    ReadFidelityProbeConfig,
    build_probe_targets,
    probe_metrics,
    train_probe,
)
from slots.slot_spec import KIND_TEXT
from substrate import default_alphabet

ALPHABET = default_alphabet()
ALPHABET_SIZE = len(ALPHABET)


@pytest.fixture
def cfg() -> ReadFidelityProbeConfig:
    return ReadFidelityProbeConfig(d_model=64, alphabet_size=ALPHABET_SIZE)


@pytest.fixture
def probe(cfg: ReadFidelityProbeConfig) -> ReadFidelityProbe:
    return ReadFidelityProbe(cfg)


class TestProbeShapes:
    def test_forward_batch(self, probe: ReadFidelityProbe, cfg: ReadFidelityProbeConfig):
        x = torch.randn(4, cfg.d_model)
        out = probe(x)
        assert out["kind_logits"].shape == (4, cfg.num_kind_classes)
        assert out["char_logits"].shape == (4, probe.max_n, cfg.alphabet_size)

    def test_forward_single(self, probe: ReadFidelityProbe, cfg: ReadFidelityProbeConfig):
        x = torch.randn(cfg.d_model)
        out = probe(x)
        assert out["kind_logits"].shape == (1, cfg.num_kind_classes)
        assert out["char_logits"].shape == (1, probe.max_n, cfg.alphabet_size)

    def test_targets_shape(self):
        texts = ["hello", "worldwide"]
        kinds = [KIND_TEXT, KIND_TEXT]
        kind_t, char_t = build_probe_targets(texts, kinds, (8, 16, 32))
        assert kind_t.shape == (2,)
        assert char_t.shape == (2, 32)
        assert (char_t[0, 5:] == -100).all()
        assert (char_t[1, 9:] == -100).all()


class TestProbeMetrics:
    def test_random_probe_kind_accuracy_near_chance(self, probe: ReadFidelityProbe):
        summaries = torch.randn(50, probe.cfg.d_model)
        texts = ["hello"] * 50
        kinds = [KIND_TEXT] * 50
        metrics = probe_metrics(probe, summaries, texts, kinds)
        assert "kind_top1_accuracy" in metrics
        # Random 5-class classifier should be around 0.2; allow generous margin.
        assert metrics["kind_top1_accuracy"] < 0.6

    def test_perfect_probe_first_n_exact(self, cfg: ReadFidelityProbeConfig):
        probe = ReadFidelityProbe(cfg)
        # Make the probe output perfect logits for first-8 prefix.
        text = "abcdefgh"
        summaries = torch.randn(1, cfg.d_model)
        with torch.no_grad():
            for i, char in enumerate(text):
                idx = ALPHABET.index(char)
                probe.char_heads[i].weight.fill_(-100.0)
                probe.char_heads[i].weight[idx, :] = 100.0
            probe.kind_head.weight.fill_(-100.0)
            probe.kind_head.weight[0, :] = 100.0
        metrics = probe_metrics(probe, summaries, [text], [KIND_TEXT])
        assert metrics["kind_top1_accuracy"] == pytest.approx(1.0)
        assert metrics["first_8_exact_prefix"] == pytest.approx(1.0)


class TestProbeSaveLoad:
    def test_round_trip(self, probe: ReadFidelityProbe, cfg: ReadFidelityProbeConfig):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.pt"
            probe.save(path)
            loaded = ReadFidelityProbe.load(path)
        assert loaded.cfg.d_model == cfg.d_model
        assert loaded.cfg.alphabet_size == cfg.alphabet_size
        for k, v in probe.state_dict().items():
            assert torch.allclose(v, loaded.state_dict()[k])


class TestProbeTrain:
    def test_probe_train_reduces_loss(self):
        torch.manual_seed(0)
        cfg = ReadFidelityProbeConfig(d_model=32, alphabet_size=ALPHABET_SIZE, hidden_dim=32)
        probe = ReadFidelityProbe(cfg)
        n = 40
        summaries = torch.randn(n, cfg.d_model)
        texts = [f"slot{i:02d}" for i in range(n)]
        kinds = [KIND_TEXT] * n
        history = train_probe(
            probe, summaries, texts, kinds,
            epochs=20, lr=1e-2, batch_size=8, device="cpu",
        )
        assert history["train_loss"][-1] < history["train_loss"][0]
