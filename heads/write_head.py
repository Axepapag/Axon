"""Axon shared text write head.

Per `RESOLUTION_writehead-0703` and `SOURCE_OF_TRUTH.md` Layer 5, the write
head is a shared decode organ (one per ``d_model`` size). It consumes a single
core/adapter ``d_model`` slot vector and emits:

  - per-position character logits for the 256 text positions of an 8192D slot;
  - a 257-class length logit vector (classes 0..256).

Training uses discrete per-position cross-entropy masked past the target length,
plus length cross-entropy. At runtime the decoded slot is diffed against the
current field slot and only changed spans are committed as typed deltas.

This module also provides helpers for turning decoded text back into a
substrate-exact 8192D slot proposal and for computing the typed delta set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from substrate import default_alphabet
from slots.slot_spec import (
    SLOT_WIDTH,
    MAX_TEXT_CHARS,
    KIND_TEXT,
    pack_slot,
)

WRITE_HEAD_VERSION = "axon_write_head_v1"


@dataclass
class WriteHeadConfig:
    """Checkpoint payload for the write head (Layer 16 provenance rules)."""

    d_model: int
    alphabet_size: int
    max_text_chars: int = MAX_TEXT_CHARS
    num_length_classes: int = MAX_TEXT_CHARS + 1  # 0..256 inclusive
    version: str = WRITE_HEAD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WriteHeadConfig":
        return cls(
            d_model=int(d["d_model"]),
            alphabet_size=int(d["alphabet_size"]),
            max_text_chars=int(d.get("max_text_chars", MAX_TEXT_CHARS)),
            num_length_classes=int(d.get("num_length_classes", MAX_TEXT_CHARS + 1)),
            version=str(d.get("version", WRITE_HEAD_VERSION)),
        )

    @classmethod
    def from_substrate(cls, d_model: int) -> "WriteHeadConfig":
        """Build a config using the live substrate alphabet size."""
        return cls(d_model=d_model, alphabet_size=len(default_alphabet()))


class WriteHead(nn.Module):
    """Shared decode organ: d_model -> (text logits, length logits)."""

    def __init__(self, cfg: WriteHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        # One linear projection to all 256 * alphabet character logits.
        self.text_head = nn.Linear(
            cfg.d_model, cfg.max_text_chars * cfg.alphabet_size, bias=False
        )
        # Small dedicated length classifier.
        self.length_head = nn.Linear(cfg.d_model, cfg.num_length_classes, bias=False)
        # Start small; the smoke gate must show learning, not a lucky init.
        nn.init.normal_(self.text_head.weight, std=0.02)
        nn.init.normal_(self.length_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward one or more d_model slot summaries.

        Args:
            x: (d_model,) or (B, d_model).

        Returns:
            dict with ``text_logits`` (B, max_text_chars, alphabet_size) and
            ``length_logits`` (B, num_length_classes).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = self.norm(x)
        text_logits = self.text_head(h).view(
            -1, self.cfg.max_text_chars, self.cfg.alphabet_size
        )
        length_logits = self.length_head(h)
        return {"text_logits": text_logits, "length_logits": length_logits}

    @torch.no_grad()
    def decode(
        self, x: torch.Tensor
    ) -> tuple[list[str], list[int]]:
        """Decode slot summaries into active text strings and lengths.

        Args:
            x: (d_model,) or (B, d_model).

        Returns:
            (texts, lengths) where ``texts[i]`` is the active prefix of length
            ``lengths[i]``.
        """
        self.eval()
        out = self.forward(x)
        text_ids = out["text_logits"].argmax(dim=-1)  # (B, 256)
        length_ids = out["length_logits"].argmax(dim=-1)  # (B,)
        alphabet = default_alphabet()
        n_alphabet = len(alphabet)
        texts: list[str] = []
        lengths: list[int] = []
        for i in range(text_ids.shape[0]):
            length = min(int(length_ids[i].item()), self.cfg.max_text_chars)
            chars: list[str] = []
            for j in range(length):
                idx = int(text_ids[i, j].item())
                chars.append(alphabet[idx] if 0 <= idx < n_alphabet else "")
            texts.append("".join(chars))
            lengths.append(length)
        return texts, lengths

    def save(self, path: str | Path) -> None:
        """Persist the head with its full config in the payload."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": WRITE_HEAD_VERSION,
            "config": self.cfg.to_dict(),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "WriteHead":
        """Load a write head from a payload that includes its config."""
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("version") != WRITE_HEAD_VERSION:
            raise ValueError(
                f"write head version mismatch: expected {WRITE_HEAD_VERSION!r}, "
                f"got {payload.get('version')!r}"
            )
        cfg = WriteHeadConfig.from_dict(payload["config"])
        head = cls(cfg)
        head.load_state_dict(payload["state_dict"])
        return head


def compute_loss(
    outputs: dict[str, torch.Tensor],
    target_chars: torch.Tensor,
    target_lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Discrete per-position CE + length CE, with positions past length masked.

    Args:
        outputs: forward() dict.
        target_chars: (B, max_text_chars) long, alphabet indices; padded
            positions must be ``-100`` so cross-entropy ignores them.
        target_lengths: (B,) long in [0, max_text_chars].

    Returns:
        dict with ``total``, ``text``, and ``length`` losses.
    """
    text_logits = outputs["text_logits"]  # (B, 256, A)
    length_logits = outputs["length_logits"]  # (B, 257)
    text_loss = F.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        target_chars.reshape(-1),
        ignore_index=-100,
    )
    length_loss = F.cross_entropy(length_logits, target_lengths)
    total_loss = text_loss + length_loss
    return {
        "total": total_loss,
        "text": text_loss,
        "length": length_loss,
    }


def encode_decoded_slot(
    decoded_text: str,
    length: int,
    kind: str = KIND_TEXT,
) -> torch.Tensor:
    """Turn decoded active text into an 8192D slot proposal.

    The proposal is already substrate-exact because ``pack_slot`` uses the
    frozen ``char_to_slot`` encoder. The adapter's ``codebook_snap`` is still
    required at commitment time, but on a valid pack it is a no-op.
    """
    active_text = decoded_text[:length]
    slot = pack_slot(text=active_text, kind=kind)
    return torch.from_numpy(slot.vector)


def commit_diff(
    decoded_slot_text: str,
    current_slot_text: str,
    length: int,
) -> list[dict[str, Any]]:
    """Pure-function diff: emit typed delta ops that touch only changed spans.

    The active prefix is ``decoded_slot_text[:length]``. Positions at or past
    ``length`` are treated as padding/cleared. The resulting ``update_span``
    records have ``start``, ``end``, and ``text``. An empty ``text`` means a
    deletion/clear of that span in the current slot.

    No silent truncation: every position that differs between the decoded
    active text and the current slot is represented in a delta.
    """
    active = (decoded_slot_text + " " * length)[:length]
    max_len = max(length, len(current_slot_text))
    deltas: list[dict[str, Any]] = []
    i = 0
    while i < max_len:
        a = active[i] if i < length else ""
        b = current_slot_text[i] if i < len(current_slot_text) else ""
        if a == b:
            i += 1
            continue
        start = i
        while i < max_len:
            a2 = active[i] if i < length else ""
            b2 = current_slot_text[i] if i < len(current_slot_text) else ""
            if a2 == b2:
                break
            i += 1
        end = i
        deltas.append({"update_span": {"start": start, "end": end, "text": active[start:end]}})
    return deltas
