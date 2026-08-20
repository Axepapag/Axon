"""Phase-A soul-tier compressor.

At episode boundaries the ``SoulCompressor`` takes the volatile hot rows,
cross-attends the warm (base) rows into them, and writes a gated update back
into the warm rows.  A tiny reconstruction loss encourages the compressed
warm state to retain information that was present in the hot rows.

The core itself keeps the flat ``[1, total_rows, d_model]`` soul tensor; the
compressor is an external module that is only exercised at episode boundaries.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoulCompressor(nn.Module):
    """Compress hot-row state into warm rows with cross-attention + gate."""

    def __init__(
        self,
        d_model: int,
        hot_rows: int,
        warm_rows: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by num_heads {num_heads}"
            )
        self.d_model = d_model
        self.hot_rows = hot_rows
        self.warm_rows = warm_rows

        # Warm rows attend to hot rows; the resulting update is gated and
        # added back to the warm rows.
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True,
        )
        self.update_norm = nn.LayerNorm(d_model)
        self.update_proj = nn.Linear(d_model, d_model, bias=False)

        # Initialize the learned scalar gate so sigmoid(gate) == 0.05.
        # This keeps the warm-row update very small at the start of training.
        gate_logit = math.log(0.05 / (1.0 - 0.05))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit))

        # Tiny reconstructor MLP: warm-state summary -> reconstructed hot rows.
        self.reconstructor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 2, hot_rows * d_model, bias=False),
        )

    def forward(
        self,
        hot: torch.Tensor,
        warm: torch.Tensor,
    ) -> torch.Tensor:
        """Return updated warm rows.

        Args:
            hot:  ``(B, hot_rows, d_model)`` volatile hot-tier rows.
            warm: ``(B, warm_rows, d_model)`` aggregated warm-tier rows.

        Returns:
            ``(B, warm_rows, d_model)`` warm rows after compression.
        """
        if tuple(hot.shape[-2:]) != (self.hot_rows, self.d_model):
            raise ValueError(
                f"hot shape {tuple(hot.shape)} does not match "
                f"(B, {self.hot_rows}, {self.d_model})"
            )
        if tuple(warm.shape[-2:]) != (self.warm_rows, self.d_model):
            raise ValueError(
                f"warm shape {tuple(warm.shape)} does not match "
                f"(B, {self.warm_rows}, {self.d_model})"
            )

        update, _ = self.cross_attn(
            warm,
            hot,
            hot,
            need_weights=False,
        )
        update = self.update_proj(self.update_norm(update))
        gate = torch.sigmoid(self.gate_logit)
        return warm + gate * update

    def compression_loss(
        self,
        hot: torch.Tensor,
        new_warm: torch.Tensor,
    ) -> torch.Tensor:
        """MSE reconstruction loss: new_warm must still be able to recover hot.

        The reconstruction target is detached so the loss only supervises the
        compressor and the warm-row update, not the hot rows themselves.
        """
        if tuple(hot.shape[-2:]) != (self.hot_rows, self.d_model):
            raise ValueError(
                f"hot shape {tuple(hot.shape)} does not match "
                f"(B, {self.hot_rows}, {self.d_model})"
            )
        if tuple(new_warm.shape[-2:]) != (self.warm_rows, self.d_model):
            raise ValueError(
                f"new_warm shape {tuple(new_warm.shape)} does not match "
                f"(B, {self.warm_rows}, {self.d_model})"
            )

        pooled = new_warm.mean(dim=1)  # (B, d_model)
        reconstructed = self.reconstructor(pooled).view(
            hot.shape[0],
            self.hot_rows,
            self.d_model,
        )
        return F.mse_loss(reconstructed, hot.detach())
