"""Read-fidelity probe over frozen adapter summaries.

Per the slot-era read-fidelity resolution, this is a tiny trainable probe that reports
how much slot information survives the lossy adapter projection. It is not the
binding response-delta gate (that is the Q6 counterfactual exact-fill task), but it
is a necessary diagnostic: if the probe cannot recover slot kind or first-N
characters from a frozen summary, the core cannot use what is not there.

The probe is deliberately small and CPU-runnable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from substrate import default_alphabet
from slots.slot_spec import (
    MAX_TEXT_CHARS,
    KIND_TEXT,
    KIND_RESPONSE_DRAFT,
    KIND_EDGE_CONT,
    KIND_CONTROL,
    KIND_EMPTY,
)

READ_FIDELITY_PROBE_VERSION = "axon_read_fidelity_probe_v1"

# Locked first-N recovery positions from Q5.
DEFAULT_N_POSITIONS = (8, 16, 32)

_KIND_CODES = [KIND_TEXT, KIND_RESPONSE_DRAFT, KIND_EDGE_CONT, KIND_CONTROL, KIND_EMPTY]


def _kind_to_index(kind: str) -> int:
    try:
        return _KIND_CODES.index(kind)
    except ValueError:
        return _KIND_CODES.index(KIND_EMPTY)


def _char_to_index(char: str) -> int | None:
    alphabet = default_alphabet()
    try:
        return alphabet.index(char)
    except ValueError:
        return None


@dataclass
class ReadFidelityProbeConfig:
    """Checkpoint payload for the read-fidelity probe."""

    d_model: int
    alphabet_size: int
    n_positions: tuple[int, ...] = DEFAULT_N_POSITIONS
    hidden_dim: int = 128
    num_kind_classes: int = len(_KIND_CODES)
    version: str = READ_FIDELITY_PROBE_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_positions"] = list(d["n_positions"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReadFidelityProbeConfig":
        return cls(
            d_model=int(d["d_model"]),
            alphabet_size=int(d["alphabet_size"]),
            n_positions=tuple(int(x) for x in d.get("n_positions", DEFAULT_N_POSITIONS)),
            hidden_dim=int(d.get("hidden_dim", 128)),
            num_kind_classes=int(d.get("num_kind_classes", len(_KIND_CODES))),
            version=str(d.get("version", READ_FIDELITY_PROBE_VERSION)),
        )

    @classmethod
    def from_substrate(cls, d_model: int, hidden_dim: int = 128) -> "ReadFidelityProbeConfig":
        return cls(
            d_model=d_model,
            alphabet_size=len(default_alphabet()),
            hidden_dim=hidden_dim,
        )


class ReadFidelityProbe(nn.Module):
    """Tiny MLP probe: d_model summary -> kind logits + first-N char logits."""

    def __init__(self, cfg: ReadFidelityProbeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.shared = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden_dim, bias=True),
            nn.GELU(),
        )
        self.kind_head = nn.Linear(cfg.hidden_dim, cfg.num_kind_classes, bias=False)
        # One small head for every position up to the largest reporting N.
        self.max_n = max(cfg.n_positions)
        self.char_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.alphabet_size, bias=False) for _ in range(self.max_n)]
        )
        # Initialise conservatively.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward one or more d_model summaries.

        Args:
            x: (d_model,) or (B, d_model).

        Returns:
            dict with ``kind_logits`` (B, num_kind_classes) and
            ``char_logits`` (B, max_n, alphabet_size).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = self.shared(self.norm(x))
        kind_logits = self.kind_head(h)
        char_logits = torch.stack([head(h) for head in self.char_heads], dim=1)
        return {"kind_logits": kind_logits, "char_logits": char_logits}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": READ_FIDELITY_PROBE_VERSION,
            "config": self.cfg.to_dict(),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "ReadFidelityProbe":
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("version") != READ_FIDELITY_PROBE_VERSION:
            raise ValueError(
                f"probe version mismatch: expected {READ_FIDELITY_PROBE_VERSION!r}, "
                f"got {payload.get('version')!r}"
            )
        cfg = ReadFidelityProbeConfig.from_dict(payload["config"])
        probe = cls(cfg)
        probe.load_state_dict(payload["state_dict"])
        return probe


def build_probe_targets(
    texts: Sequence[str],
    kinds: Sequence[str],
    n_positions: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build target tensors for the probe.

    Args:
        texts: decoded active slot texts.
        kinds: slot kind characters.
        n_positions: first-N positions to supervise.

    Returns:
        kind_targets: (B,) long.
        char_targets: (B, max(n_positions)) long, padded with -100.
    """
    max_n = max(n_positions)
    kind_targets = torch.tensor([_kind_to_index(k) for k in kinds], dtype=torch.long)
    char_targets = torch.full((len(texts), max_n), -100, dtype=torch.long)
    for i, text in enumerate(texts):
        for j in range(min(max_n, len(text))):
            idx = _char_to_index(text[j])
            if idx is not None:
                char_targets[i, j] = idx
    return kind_targets, char_targets


def probe_metrics(
    probe: ReadFidelityProbe,
    summaries: torch.Tensor,
    texts: Sequence[str],
    kinds: Sequence[str],
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate the probe on a batch of frozen adapter summaries.

    Reports:
      - kind_top1_accuracy
      - first_N_position_accuracy for each N in the probe config
      - first_N_exact_prefix for each N (full first-N string correct, only
        counted for slots whose active length is at least N)
    """
    if device is None:
        device = summaries.device
    probe.eval()
    summaries = summaries.to(device)
    kind_targets, char_targets = build_probe_targets(texts, kinds, probe.cfg.n_positions)
    kind_targets = kind_targets.to(device)
    char_targets = char_targets.to(device)

    with torch.no_grad():
        out = probe(summaries)
        kind_preds = out["kind_logits"].argmax(dim=-1)
        char_preds = out["char_logits"].argmax(dim=-1)

    kind_acc = float((kind_preds == kind_targets).float().mean().item())
    metrics: dict[str, float] = {"kind_top1_accuracy": kind_acc}

    for n in probe.cfg.n_positions:
        pred_n = char_preds[:, :n]
        tgt_n = char_targets[:, :n]
        valid = (tgt_n != -100)
        if valid.any():
            pos_acc = float((pred_n[valid] == tgt_n[valid]).float().mean().item())
        else:
            pos_acc = float("nan")
        metrics[f"first_{n}_position_accuracy"] = pos_acc

        # Full prefix exact: slot must be long enough and all N positions correct.
        long_enough = torch.tensor(
            [len(t) >= n for t in texts], dtype=torch.bool, device=device
        )
        if long_enough.any():
            prefix_correct = ((pred_n == tgt_n) | (~valid)).all(dim=-1)
            exact = float(prefix_correct[long_enough].float().mean().item())
        else:
            exact = float("nan")
        metrics[f"first_{n}_exact_prefix"] = exact

    return metrics


def train_probe(
    probe: ReadFidelityProbe,
    train_summaries: torch.Tensor,
    train_texts: Sequence[str],
    train_kinds: Sequence[str],
    val_summaries: torch.Tensor | None = None,
    val_texts: Sequence[str] | None = None,
    val_kinds: Sequence[str] | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> dict[str, list[float]]:
    """Simple supervised trainer for the read-fidelity probe."""
    device = torch.device(device)
    probe.to(device)
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    n_positions = probe.cfg.n_positions
    kind_targets, char_targets = build_probe_targets(train_texts, train_kinds, n_positions)
    dataset = torch.utils.data.TensorDataset(
        train_summaries, kind_targets, char_targets
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for x, kt, ct in loader:
            x, kt, ct = x.to(device), kt.to(device), ct.to(device)
            out = probe(x)
            kind_loss = F.cross_entropy(out["kind_logits"], kt)
            char_loss = F.cross_entropy(
                out["char_logits"].reshape(-1, out["char_logits"].size(-1)),
                ct.reshape(-1),
                ignore_index=-100,
            )
            loss = kind_loss + char_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        history["train_loss"].append(float(np.mean(epoch_losses)))

        if val_summaries is not None and val_texts is not None and val_kinds is not None:
            probe.eval()
            with torch.no_grad():
                v_kt, v_ct = build_probe_targets(val_texts, val_kinds, n_positions)
                v_kt, v_ct = v_kt.to(device), v_ct.to(device)
                out = probe(val_summaries.to(device))
                kind_loss = F.cross_entropy(out["kind_logits"], v_kt)
                char_loss = F.cross_entropy(
                    out["char_logits"].reshape(-1, out["char_logits"].size(-1)),
                    v_ct.reshape(-1),
                    ignore_index=-100,
                )
                history["val_loss"].append(float((kind_loss + char_loss).item()))
            probe.train()

    return history
