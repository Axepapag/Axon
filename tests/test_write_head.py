"""Tests for the shared text write head and its runtime diff helper."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from heads.write_head import (
    WriteHead,
    WriteHeadConfig,
    commit_diff,
    compute_loss,
    encode_decoded_slot,
)
from slots.slot_spec import MAX_TEXT_CHARS, KIND_TEXT, pack_slot, unpack_slot
from substrate import default_alphabet

ALPHABET = default_alphabet()
ALPHABET_SIZE = len(ALPHABET)


@pytest.fixture
def cfg() -> WriteHeadConfig:
    return WriteHeadConfig(d_model=64, alphabet_size=ALPHABET_SIZE)


@pytest.fixture
def head(cfg: WriteHeadConfig) -> WriteHead:
    return WriteHead(cfg)


class TestWriteHeadShapes:
    def test_forward_batch(self, head: WriteHead, cfg: WriteHeadConfig):
        x = torch.randn(4, cfg.d_model)
        out = head(x)
        assert out["text_logits"].shape == (4, MAX_TEXT_CHARS, cfg.alphabet_size)
        assert out["length_logits"].shape == (4, cfg.num_length_classes)

    def test_forward_single(self, head: WriteHead, cfg: WriteHeadConfig):
        x = torch.randn(cfg.d_model)
        out = head(x)
        assert out["text_logits"].shape == (1, MAX_TEXT_CHARS, cfg.alphabet_size)
        assert out["length_logits"].shape == (1, cfg.num_length_classes)

    def test_decode_shape(self, head: WriteHead, cfg: WriteHeadConfig):
        x = torch.randn(3, cfg.d_model)
        texts, lengths = head.decode(x)
        assert len(texts) == 3
        assert len(lengths) == 3
        assert all(isinstance(t, str) for t in texts)
        assert all(0 <= L <= MAX_TEXT_CHARS for L in lengths)

    def test_config_from_substrate(self):
        cfg = WriteHeadConfig.from_substrate(128)
        assert cfg.d_model == 128
        assert cfg.alphabet_size == ALPHABET_SIZE
        assert cfg.max_text_chars == MAX_TEXT_CHARS
        assert cfg.num_length_classes == MAX_TEXT_CHARS + 1


class TestWriteHeadLossAndMasking:
    def test_loss_masked_past_length(self, cfg: WriteHeadConfig):
        """Positions beyond target length must be ignored in text loss."""
        head = WriteHead(cfg)
        B = 2
        target_lengths = torch.tensor([5, 3], dtype=torch.long)
        # Build target chars where active positions are index 0, padded with -100.
        target_chars = torch.full((B, MAX_TEXT_CHARS), -100, dtype=torch.long)
        target_chars[0, :5] = 0
        target_chars[1, :3] = 0

        # Make logits that are perfectly correct for active positions.
        text_logits = torch.zeros(B, MAX_TEXT_CHARS, cfg.alphabet_size)
        for b in range(B):
            L = target_lengths[b].item()
            text_logits[b, :L, 0] = 100.0
            # Past-length positions are deliberately wrong (favor index 1).
            text_logits[b, L:, 1] = 100.0
        # Length is correct.
        length_logits = torch.zeros(B, cfg.num_length_classes)
        length_logits[range(B), target_lengths] = 100.0

        out = {"text_logits": text_logits, "length_logits": length_logits}
        losses = compute_loss(out, target_chars, target_lengths)
        assert losses["text"].item() == pytest.approx(0.0, abs=1e-4)
        assert losses["length"].item() == pytest.approx(0.0, abs=1e-4)

    def test_loss_nonzero_when_wrong(self, cfg: WriteHeadConfig):
        head = WriteHead(cfg)
        B = 2
        target_lengths = torch.tensor([4, 4], dtype=torch.long)
        target_chars = torch.full((B, MAX_TEXT_CHARS), -100, dtype=torch.long)
        target_chars[:, :4] = torch.tensor([0, 1, 2, 3])
        out = head(torch.randn(B, cfg.d_model))
        losses = compute_loss(out, target_chars, target_lengths)
        assert losses["total"].item() > 0.0


class TestCommitDiff:
    def test_no_changes(self):
        assert commit_diff("hello", "hello", 5) == []

    def test_single_changed_span(self):
        deltas = commit_diff("hello world", "hello there", 11)
        assert len(deltas) == 1
        assert deltas[0] == {"update_span": {"start": 6, "end": 11, "text": "world"}}

    def test_multiple_changed_spans(self):
        # The delta text is the decoded (active) text, not the current text.
        deltas = commit_diff("abc def ghi", "abc xyz ghi", 11)
        assert len(deltas) == 1  # contiguous run
        assert deltas[0]["update_span"]["text"] == "def"

    def test_only_changed_positions_no_padding(self):
        deltas = commit_diff("hello", "hello world", 5)
        assert len(deltas) == 1
        assert deltas[0] == {"update_span": {"start": 5, "end": 11, "text": ""}}

    def test_truncated_length_clears_tail(self):
        # Decoded length shorter than current; tail must be cleared explicitly.
        deltas = commit_diff("hello", "hello world", 5)
        assert deltas == [{"update_span": {"start": 5, "end": 11, "text": ""}}]

    def test_decoded_text_longer_than_length_truncated(self):
        # Active text is truncated to length; only differing spans are emitted.
        deltas = commit_diff("hello world extra", "hi world", 8)
        assert deltas == [
            {"update_span": {"start": 1, "end": 4, "text": "ell"}},
            {"update_span": {"start": 5, "end": 8, "text": " wo"}},
        ]

    def test_empty_active_clears_all(self):
        deltas = commit_diff("", "hello", 0)
        assert deltas == [{"update_span": {"start": 0, "end": 5, "text": ""}}]


class TestEncodeDecodedSlot:
    def test_roundtrip_through_substrate(self):
        text = "The dog is good"
        length = len(text)
        vec = encode_decoded_slot(text, length, kind=KIND_TEXT)
        assert vec.shape == (8192,)
        slot = pack_slot(text=text, kind=KIND_TEXT)
        # The proposal should be substrate-exact.
        assert torch.allclose(vec, torch.from_numpy(slot.vector), atol=1e-6)


class TestSaveLoad:
    def test_round_trip(self, head: WriteHead, cfg: WriteHeadConfig):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.pt"
            head.save(path)
            loaded = WriteHead.load(path)
        assert loaded.cfg.d_model == cfg.d_model
        assert loaded.cfg.alphabet_size == cfg.alphabet_size
        for k, v in head.state_dict().items():
            assert torch.allclose(v, loaded.state_dict()[k])

    def test_payload_includes_config(self, head: WriteHead):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.pt"
            head.save(path)
            payload = torch.load(path, weights_only=False)
        assert payload["version"] is not None
        assert payload["config"]["d_model"] == head.cfg.d_model


class TestSmokeGateLogic:
    """The smoke gate requires loss to fall AND exact-fill to climb above floor."""

    def test_gate_passes_when_both_improve(self):
        from training.trainer_write_head import smoke_gate_passed
        floor = 0.05
        initial = {"loss": 5.0, "positive_exact_fill": 0.0}
        final = {"loss": 1.0, "positive_exact_fill": 0.10}
        passed, reason = smoke_gate_passed(initial, final, floor)
        assert passed
        assert "floor" in reason

    def test_gate_fails_when_loss_flat(self):
        from training.trainer_write_head import smoke_gate_passed
        floor = 0.05
        initial = {"loss": 2.0, "positive_exact_fill": 0.0}
        final = {"loss": 1.95, "positive_exact_fill": 0.10}
        passed, _reason = smoke_gate_passed(initial, final, floor)
        assert not passed

    def test_gate_fails_when_exact_fill_below_floor(self):
        from training.trainer_write_head import smoke_gate_passed
        floor = 0.05
        initial = {"loss": 5.0, "positive_exact_fill": 0.0}
        final = {"loss": 1.0, "positive_exact_fill": 0.01}
        passed, _reason = smoke_gate_passed(initial, final, floor)
        assert not passed
