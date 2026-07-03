"""Axon v7 core — the transformer. Same architecture as v6 (RoPE attention,
SwiGLU FFN, pre-norm), used by BOTH the trainer and the runtime. No
generation head, no vocabulary — it refines fields in d_model space; the
ProjectionBank moves between 16D substrate and d_model.

v7 config addition: `soul_rows` — the per-core private hidden state
(Contract 5). Each soul row lives at full d_model width (128 in dev
cores, 1024 at scale) — pre-language thought, never letter-encoded. The
rows themselves are managed by the caller; the config records how many
this core was built to carry.

v7.1 soul mode: `act_reflect` — the soul is not concatenated as public
content. Each cycle first ingests the private soul, uses it to modulate
the shared-state layers, then updates the soul from the action it just
produced. The old `concat` mode remains loadable for prior checkpoints.

v7.2 soul mode: `act_reflect_v2` — the soul stays private but remains
row-addressable. Shared-state rows cross-attend into individual soul rows
during the act pass, then soul rows cross-attend back over the produced
shared action during reflection. No mean-pooled soul summary is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class CoreConfig:
    d_model: int = 128
    n_heads: int = 1
    n_layers: int = 2
    ffn_dim: int = 16384
    dropout: float = 0.0
    # Recursive refinement: total forward ticks per example, and how many
    # of the last ticks carry gradients (deep supervision).
    n_ticks: int = 3
    grad_ticks: int = 2
    # v7: the soul — per-core private hidden rows carried tick to tick.
    soul_rows: int = 64
    # "concat" keeps original v7 behavior. "act_reflect" implements the
    # soul-first / act / reflect cycle with pooled modulation.
    # "act_reflect_v2" keeps the soul row-addressable with private
    # bidirectional cross-attention.
    soul_mode: str = "concat"
    # v7.3: initial value for soul_cross_gate and soul_reflect_gate.
    # Keep the global default conservative; smoke configs can explicitly
    # start half-open with 0.5 when we want to prove the soul channel.
    # Existing checkpoints keep their learned gate tensors; this only
    # affects new cores.
    soul_gate_init: float = 0.05
    # v7.4/v7.6: writable soul tiers. hot_rows=0 is the legacy read-only
    # alphabet soul (byte-identical). hot_rows>0 adds M writable rows above the
    # frozen alphabet base.
    #
    # v7.6: soul_write_mode controls how updates reach the hot rows:
    #   "direct"       — raw update gated by soul_reflect_gate only.
    #   "compartments" — pill-organizer write: K labeled compartments, a router
    #                    head on the field output decides which compartment(s)
    #                    receive the update. Anything that doesn't route to a
    #                    compartment passes through unchanged.
    soul_hot_rows: int = 0
    soul_write_mode: str = "compartments"
    n_soul_compartments: int = 8

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CoreConfig":
        d_model = int(d.get("d_model", 128))
        n_heads = int(d.get("n_heads", d.get("heads", 1)))
        n_layers = int(d.get("n_layers", d.get("layers", 2)))
        if "ffn_dim" in d:
            ffn_dim = int(d["ffn_dim"])
        elif "ffn_multiplier" in d:
            ffn_dim = d_model * int(d["ffn_multiplier"])
        else:
            ffn_dim = 16384
        return cls(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ffn_dim=ffn_dim, dropout=float(d.get("dropout", 0.0)),
            n_ticks=int(d.get("n_ticks", 3)),
            grad_ticks=int(d.get("grad_ticks", 2)),
            soul_rows=int(d.get("soul_rows", d.get("memory_rows", 64))),
            soul_mode=str(d.get("soul_mode", "concat")),
            soul_gate_init=float(d.get("soul_gate_init", 0.05)),
            soul_hot_rows=int(d.get("soul_hot_rows", 0)),
            soul_write_mode=str(d.get("soul_write_mode", "compartments")),
            n_soul_compartments=int(d.get("n_soul_compartments", 8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model, "n_heads": self.n_heads,
            "n_layers": self.n_layers, "ffn_dim": self.ffn_dim,
            "dropout": self.dropout, "n_ticks": self.n_ticks,
            "grad_ticks": self.grad_ticks, "soul_rows": self.soul_rows,
            "soul_mode": self.soul_mode,
            "soul_gate_init": self.soul_gate_init,
            "soul_hot_rows": self.soul_hot_rows,
            "soul_write_mode": self.soul_write_mode,
            "n_soul_compartments": self.n_soul_compartments,
        }

    def total_soul_rows(self) -> int:
        return int(self.soul_rows) + int(self.soul_hot_rows)


class MultiHeadAttention(nn.Module):
    """Attention with RoPE. Same parameter names/shapes as v6 (qkv, out)."""

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self._rope(q)
        k = self._rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, hd = x.shape
        half = hd // 2
        pos = torch.arange(N, device=x.device, dtype=torch.float32)[:, None]
        freqs = torch.arange(half, device=x.device, dtype=torch.float32)
        angles = pos * (1.0 / (10000.0 ** (freqs / max(half, 1))))
        cos_a = torch.cos(angles)[None, None]
        sin_a = torch.sin(angles)[None, None]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        r0 = x0 * cos_a - x1 * sin_a
        r1 = x0 * sin_a + x1 * cos_a
        out = torch.empty_like(x)
        out[..., 0::2] = r0
        out[..., 1::2] = r1
        return out


class CrossAttention(nn.Module):
    """Attention from one private/public sequence into another sequence."""

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = query.shape
        M = context.shape[1]
        H, hd = self.n_heads, self.head_dim
        q = self.q(query).reshape(B, N, H, hd).transpose(1, 2)
        kv = self.kv(context).reshape(B, M, 2, H, hd).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self._rope(q)
        k = self._rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if context_mask is not None:
            scores = scores.masked_fill(~context_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, hd = x.shape
        half = hd // 2
        pos = torch.arange(N, device=x.device, dtype=torch.float32)[:, None]
        freqs = torch.arange(half, device=x.device, dtype=torch.float32)
        angles = pos * (1.0 / (10000.0 ** (freqs / max(half, 1))))
        cos_a = torch.cos(angles)[None, None]
        sin_a = torch.sin(angles)[None, None]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        r0 = x0 * cos_a - x1 * sin_a
        r1 = x0 * sin_a + x1 * cos_a
        out = torch.empty_like(x)
        out[..., 0::2] = r0
        out[..., 1::2] = r1
        return out


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.w = nn.Linear(dim_in, dim_out * 2, bias=False)
        self.out = nn.Linear(dim_out, dim_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * val)


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SoulModulatedTransformerLayer(TransformerLayer):
    """Transformer layer where the private soul acts as a processing lens.

    The soul is not part of the public attention sequence. A pooled soul
    context scale/shifts the attention and FFN inputs, so the same shared
    state can be transformed differently depending on the core's private
    carried state.
    """

    def __init__(self, d_model: int, ffn_dim: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__(d_model, ffn_dim, n_heads, dropout)
        self.soul_mod = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4, bias=False),
        )
        nn.init.zeros_(self.soul_mod[1].weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        soul_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if soul_context is None:
            return super().forward(x, mask)
        a_scale, a_shift, f_scale, f_shift = self.soul_mod(soul_context).chunk(4, dim=-1)
        a = self.norm1(x)
        a = a * (1.0 + a_scale[:, None, :]) + a_shift[:, None, :]
        x = x + self.dropout(self.attn(a, mask))
        f = self.norm2(x)
        f = f * (1.0 + f_scale[:, None, :]) + f_shift[:, None, :]
        x = x + self.dropout(self.ffn(f))
        return x


class SoulCrossAttentionTransformerLayer(TransformerLayer):
    """Transformer layer whose shared rows can address private soul rows."""

    def __init__(self, d_model: int, ffn_dim: int, n_heads: int = 1, dropout: float = 0.0,
                 soul_gate_init: float = 0.05):
        super().__init__(d_model, ffn_dim, n_heads, dropout)
        self.cross_q_norm = nn.LayerNorm(d_model)
        self.cross_ctx_norm = nn.LayerNorm(d_model)
        self.soul_cross = CrossAttention(d_model, n_heads, dropout)
        # v7.3: configurable gate init. Smoke configs may set this higher;
        # existing checkpoints keep their learned gate tensors via load_state_dict.
        self.soul_cross_gate = nn.Parameter(torch.tensor(soul_gate_init))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        soul_rows: torch.Tensor | None = None,
        soul_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        if soul_rows is not None and soul_rows.shape[1] > 0:
            cross = self.soul_cross(
                self.cross_q_norm(x),
                self.cross_ctx_norm(soul_rows),
                context_mask=soul_mask,
            )
            x = x + self.dropout(self.soul_cross_gate * cross)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class AxonCore(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.soul_mode == "act_reflect":
            layer_cls = SoulModulatedTransformerLayer
        elif cfg.soul_mode == "act_reflect_v2":
            layer_cls = SoulCrossAttentionTransformerLayer
        else:
            layer_cls = TransformerLayer
        self.layers = nn.ModuleList([
            layer_cls(cfg.d_model, cfg.ffn_dim, cfg.n_heads, cfg.dropout,
                      soul_gate_init=cfg.soul_gate_init)
            if layer_cls is SoulCrossAttentionTransformerLayer else
            layer_cls(cfg.d_model, cfg.ffn_dim, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        if cfg.soul_mode in ("act_reflect", "act_reflect_v2"):
            soul_ffn_dim = max(cfg.d_model * 4, min(cfg.ffn_dim, cfg.d_model * 8))
            self.soul_ingest = TransformerLayer(
                cfg.d_model, soul_ffn_dim, cfg.n_heads, cfg.dropout)
            self.soul_reflect = TransformerLayer(
                cfg.d_model, soul_ffn_dim, cfg.n_heads, cfg.dropout)
        if cfg.soul_mode == "act_reflect":
            self.soul_norm = nn.LayerNorm(cfg.d_model)
            self.action_norm = nn.LayerNorm(cfg.d_model)
            self.action_to_soul = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        elif cfg.soul_mode == "act_reflect_v2":
            self.action_norm = nn.LayerNorm(cfg.d_model)
            self.soul_action_norm = nn.LayerNorm(cfg.d_model)
            self.action_to_soul = CrossAttention(
                cfg.d_model, cfg.n_heads, cfg.dropout)
            self.soul_reflect_gate = nn.Parameter(torch.tensor(cfg.soul_gate_init))
        # v7.6: compartment router — pill-organizer write.
        # K compartments, each gets rows_per_comp = hot_rows // K rows.
        # Router: mean-pool field output → K logits → softmax → per-compartment gate.
        if cfg.soul_hot_rows > 0 and cfg.soul_write_mode == "compartments":
            assert cfg.soul_mode == "act_reflect_v2", (
                "soul_hot_rows > 0 requires act_reflect_v2"
            )
            K = cfg.n_soul_compartments
            self.soul_router = nn.Linear(cfg.d_model, K, bias=False)
            nn.init.zeros_(self.soul_router.weight)  # start neutral
            self.soul_router_norm = nn.LayerNorm(cfg.d_model)
            self.soul_compartment_gate = nn.Parameter(torch.tensor(cfg.soul_gate_init))
            # rows per compartment (last compartment absorbs remainder)
            self._rows_per_comp = cfg.soul_hot_rows // K
        # Gradient checkpointing: when True (set by the trainer during
        # training), each layer's activations are recomputed in the
        # backward pass instead of being stored. Trades a little compute
        # for a large drop in memory — the fat FFN intermediate (rows x
        # ffn_dim) is no longer held. Mathematically identical (PyTorch
        # preserves RNG state, so dropout matches on recompute). The
        # runtime leaves this off (inference runs under no_grad anyway).
        self.use_checkpoint = False
        # v7.x read-only soul: when True the body STILL reads/attends the soul
        # rows, but the carried soul is never written (no reflect/update) —
        # the soul stays bit-constant across all ticks/episodes. Set by the
        # trainer when a fixed soul (e.g. the alphabet codebook) is seeded.
        self.soul_readonly = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = x
        for layer in self.layers:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, mask, use_reentrant=False)
            else:
                h = layer(h, mask)
        return {"hidden": self.norm(h)}

    def forward_with_soul(
        self,
        field: torch.Tensor,
        soul: torch.Tensor,
        mask: torch.Tensor | None = None,
        soul_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One refinement cycle over shared field + private soul.

        concat mode preserves the old v7 behavior. act_reflect mode:
          1. ingest the soul by itself;
          2. process shared state through soul-modulated layers;
          3. update the soul from the action just produced.

        act_reflect_v2 keeps the same cycle but uses private cross-attention
        instead of mean-pooling the soul into one modulation vector.

        soul_mask: (B, total_rows) bool — True for active soul rows. Inactive
        rows are masked out of cross-attention. This enables the dynamic
        temperature-tiered soul where rows inflate and deflate.
        """
        if self.cfg.soul_mode not in ("act_reflect", "act_reflect_v2"):
            n = field.shape[1]
            out = self.forward(torch.cat([field, soul], dim=1), mask=mask)["hidden"]
            soul_out = soul if self.soul_readonly else out[:, n:]
            return {"field": out[:, :n], "soul": soul_out, "soul_kl": None}

        if soul.shape[1] == 0:
            ingested_soul = soul
        else:
            ingested_soul = self.soul_ingest(soul)

        if self.cfg.soul_mode == "act_reflect_v2":
            h = field
            for layer in self.layers:
                if self.use_checkpoint and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(
                        layer, h, mask, ingested_soul, soul_mask,
                        use_reentrant=False)
                else:
                    h = layer(h, mask, ingested_soul, soul_mask)
            field_out = self.norm(h)

            soul_kl = None
            if soul.shape[1] == 0 or self.soul_readonly:
                # Read-only path: the soul is bit-constant.  With hot_rows=0 this
                # is byte-identical to v7.3.
                soul_out = soul
            else:
                soul_update = self.action_to_soul(
                    self.soul_action_norm(ingested_soul),
                    self.action_norm(field_out),
                    mask,
                )
                soul_kl = None
                hot_rows = int(self.cfg.soul_hot_rows)
                if hot_rows > 0:
                    base_rows = int(self.cfg.soul_rows)
                    write_mode = self.cfg.soul_write_mode

                    if write_mode == "direct":
                        # v7.6 direct write — raw soul_update gated by reflect gate.
                        hot_in = (
                            ingested_soul[:, base_rows:, :]
                            + self.soul_reflect_gate * soul_update[:, base_rows:, :]
                        )
                        hot_out = self.soul_reflect(hot_in)
                        base_out = soul[:, :base_rows, :].detach()
                        soul_out = torch.cat([base_out, hot_out], dim=1)

                    elif write_mode == "compartments":
                        # v7.6 pill-organizer write.
                        # Router: mean-pool the field output → K compartment logits.
                        # Each compartment gets a slice of hot rows.
                        # Only routed compartments receive updates; the rest
                        # pass through unchanged.
                        K = self.cfg.n_soul_compartments
                        rpc = self._rows_per_comp
                        field_pooled = self.soul_router_norm(field_out).mean(dim=1)  # (B, d)
                        router_logits = self.soul_router(field_pooled)  # (B, K)
                        router_weights = F.softmax(router_logits, dim=-1)  # (B, K)

                        hot_old = ingested_soul[:, base_rows:, :]  # (B, M, d)
                        hot_update = soul_update[:, base_rows:, :]  # (B, M, d)

                        # Build per-row gate from router weights
                        row_gates = torch.zeros(hot_old.shape[0], hot_old.shape[1], 1,
                                                device=hot_old.device, dtype=hot_old.dtype)
                        for comp in range(K):
                            start = comp * rpc
                            end = start + rpc if comp < K - 1 else hot_old.shape[1]
                            row_gates[:, start:end, :] = router_weights[:, comp:comp+1, None]

                        # Gated update: only routed compartments get written
                        hot_in = hot_old + self.soul_compartment_gate * row_gates * hot_update
                        hot_out = self.soul_reflect(hot_in)
                        base_out = soul[:, :base_rows, :].detach()
                        soul_out = torch.cat([base_out, hot_out], dim=1)

                    else:
                        raise ValueError(f"unknown soul_write_mode: {write_mode}")
                else:
                    reflect_in = ingested_soul + self.soul_reflect_gate * soul_update
                    soul_out = self.soul_reflect(reflect_in)
            return {"field": field_out, "soul": soul_out, "soul_kl": soul_kl}

        if soul.shape[1] == 0:
            soul_context = field.new_zeros((field.shape[0], field.shape[2]))
        else:
            soul_context = self.soul_norm(ingested_soul).mean(dim=1)

        h = field
        for layer in self.layers:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, mask, soul_context, use_reentrant=False)
            else:
                h = layer(h, mask, soul_context)
        field_out = self.norm(h)

        if soul.shape[1] == 0 or self.soul_readonly:
            soul_out = soul
        else:
            action_context = self.action_norm(field_out).mean(dim=1)
            reflect_in = ingested_soul + self.action_to_soul(action_context)[:, None, :]
            soul_out = self.soul_reflect(reflect_in)
        return {"field": field_out, "soul": soul_out, "soul_kl": None}


if __name__ == "__main__":
    cfg = CoreConfig(d_model=32, ffn_dim=64, n_layers=1, soul_rows=8)
    core = AxonCore(cfg)
    x = torch.randn(1, 20, 32)
    out = core(x)["hidden"]
    assert out.shape == (1, 20, 32)
    cfg2 = CoreConfig(d_model=32, ffn_dim=64, n_layers=1, soul_rows=8,
                      soul_mode="act_reflect")
    core2 = AxonCore(cfg2)
    field = torch.randn(1, 20, 32)
    soul = torch.randn(1, 8, 32)
    out2 = core2.forward_with_soul(field, soul)
    assert out2["field"].shape == (1, 20, 32)
    assert out2["soul"].shape == (1, 8, 32)
    cfg3 = CoreConfig(d_model=32, ffn_dim=64, n_layers=1, soul_rows=8,
                      soul_mode="act_reflect_v2")
    core3 = AxonCore(cfg3)
    out3 = core3.forward_with_soul(field, soul)
    assert out3["field"].shape == (1, 20, 32)
    assert out3["soul"].shape == (1, 8, 32)
    print(f"core OK: {sum(p.numel() for p in core.parameters()):,} params, cfg={cfg.to_dict()}")
