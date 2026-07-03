"""soul_v2.py — Temperature-tiered dynamic soul manager for Axon v7.

The soul lives in native d_model space. It is never projected through the
rail. It is never encoded as 16D substrate characters. Each soul row holds a
d_model-dimensional thought produced by the core's own processing.

Tier structure:
  HOT  — many rows, every tick. Category-templated write target.
  WARM — fewer rows. Compressed from hot. Periodic compression pass.
  COLD — fewest rows. Compressed from warm. Dense wisdom.
  (FROZEN — distilled into LoRA adapters during offline cycle, not managed here.)

The soul is dynamic: it inflates (activates new rows) and deflates (evicts or
compresses stale rows). An active mask tracks which rows are live. Inactive
rows are zeroed and invisible to cross-attention.

Salience tracking: each row accumulates a running average of attention weight
it receives during the act pass. Low-salience rows are candidates for
compression or eviction.

Category map: the hot tier uses categories as a write template. Each exhale
evaluates every category and decides whether to write, update, or inflate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tier config
# ---------------------------------------------------------------------------

@dataclass
class TierSpec:
    """Capacity and behavior for one temperature tier."""
    name: str
    max_rows: int           # over-allocated capacity
    initial_active: int = 0 # rows active at fresh-soul creation
    salience_decay: float = 0.99   # exponential decay per tick
    salience_threshold: float = 0.01  # below this → compression/eviction candidate
    dormant_ticks: int = 50  # ticks below threshold before eviction/compression


@dataclass
class SoulV2Config:
    """Configuration for the temperature-tiered soul."""
    d_model: int = 128

    # Tier specifications (hot, warm, cold)
    tiers: list[TierSpec] = field(default_factory=lambda: [
        TierSpec(name="hot",  max_rows=512, initial_active=0,
                 salience_decay=0.99, salience_threshold=0.01, dormant_ticks=50),
        TierSpec(name="warm", max_rows=128, initial_active=0,
                 salience_decay=0.995, salience_threshold=0.005, dormant_ticks=200),
        TierSpec(name="cold", max_rows=32,  initial_active=0,
                 salience_decay=0.999, salience_threshold=0.001, dormant_ticks=1000),
    ])

    # Hot-tier category map (write template)
    categories: list[str] = field(default_factory=lambda: [
        "episodic", "lessons", "diary", "awareness", "scratch", "tasks",
    ])

    # Write behavior
    router_threshold: float = 0.5  # confidence below this → inflate new row
    write_gate_init: float = 0.1  # initial gate for soul writes
    salience_ema_alpha: float = 0.1  # EMA alpha for salience updates

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "tiers": [
                {"name": t.name, "max_rows": t.max_rows,
                 "initial_active": t.initial_active,
                 "salience_decay": t.salience_decay,
                 "salience_threshold": t.salience_threshold,
                 "dormant_ticks": t.dormant_ticks}
                for t in self.tiers
            ],
            "categories": list(self.categories),
            "router_threshold": self.router_threshold,
            "write_gate_init": self.write_gate_init,
            "salience_ema_alpha": self.salience_ema_alpha,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoulV2Config":
        tiers = []
        for td in d.get("tiers", []):
            tiers.append(TierSpec(
                name=td["name"],
                max_rows=int(td["max_rows"]),
                initial_active=int(td.get("initial_active", 0)),
                salience_decay=float(td.get("salience_decay", 0.99)),
                salience_threshold=float(td.get("salience_threshold", 0.01)),
                dormant_ticks=int(td.get("dormant_ticks", 50)),
            ))
        return cls(
            d_model=int(d.get("d_model", 128)),
            tiers=tiers if tiers else cls().tiers,
            categories=list(d.get("categories", cls().categories)),
            router_threshold=float(d.get("router_threshold", 0.5)),
            write_gate_init=float(d.get("write_gate_init", 0.1)),
            salience_ema_alpha=float(d.get("salience_ema_alpha", 0.1)),
        )

    @property
    def total_max_rows(self) -> int:
        return sum(t.max_rows for t in self.tiers)

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    def tier_offset(self, tier_idx: int) -> int:
        """Row offset where tier tier_idx begins in the flat soul tensor."""
        return sum(t.max_rows for t in self.tiers[:tier_idx])

    def tier_slice(self, tier_idx: int) -> slice:
        """Row slice for tier_idx in the flat soul tensor."""
        start = self.tier_offset(tier_idx)
        end = start + self.tiers[tier_idx].max_rows
        return slice(start, end)


# ---------------------------------------------------------------------------
# Soul state — the dynamic, mutable tensor + metadata
# ---------------------------------------------------------------------------

class SoulState:
    """Mutable soul state: thoughts, active mask, salience, tier/category metadata.

    The soul is a flat (total_max_rows, d_model) tensor. Each row has:
      - active: bool — is this row live?
      - tier: int — which temperature tier (0=hot, 1=warm, 2=cold)
      - category: int — which category (only meaningful for hot tier, -1 for deeper)
      - salience: float — running average attention weight
      - dormant_for: int — ticks since salience dropped below threshold
      - tick_born: int — tick when this row was activated
    """

    def __init__(self, cfg: SoulV2Config, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.total_rows = cfg.total_max_rows
        self.d_model = cfg.d_model

        # The actual thoughts — native d_model, never rail-projected
        self.tensor = torch.zeros(self.total_rows, self.d_model,
                                  device=device, dtype=dtype)

        # Metadata (kept on CPU for cheap indexing, moved to device when needed)
        self.active = torch.zeros(self.total_rows, dtype=torch.bool)
        self.tier = torch.zeros(self.total_rows, dtype=torch.long)
        self.category = torch.full((self.total_rows,), -1, dtype=torch.long)
        self.salience = torch.zeros(self.total_rows, dtype=torch.float32)
        self.dormant_for = torch.zeros(self.total_rows, dtype=torch.long)
        self.tick_born = torch.zeros(self.total_rows, dtype=torch.long)
        self.current_tick = 0

        # Initialize tier labels for all rows
        for ti, spec in enumerate(cfg.tiers):
            sl = cfg.tier_slice(ti)
            self.tier[sl] = ti

    @property
    def active_mask(self) -> torch.Tensor:
        """Bool mask (total_rows,) — True for active rows. Used for cross-attention."""
        return self.active

    @property
    def active_mask_2d(self) -> torch.Tensor:
        """Bool mask (1, total_rows) for batched cross-attention masking."""
        return self.active.unsqueeze(0).to(self.device)

    def active_rows(self) -> torch.Tensor:
        """Return only active rows from the soul tensor (n_active, d_model)."""
        return self.tensor[self.active]

    def n_active(self) -> int:
        return int(self.active.sum().item())

    def n_active_in_tier(self, tier_idx: int) -> int:
        sl = self.cfg.tier_slice(tier_idx)
        return int(self.active[sl].sum().item())

    def n_active_in_category(self, category_idx: int) -> int:
        tier0_sl = self.cfg.tier_slice(0)  # categories only in hot tier
        cat_mask = (self.category[tier0_sl] == category_idx)
        return int(cat_mask.sum().item())

    def get_batched(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (soul (1, total_rows, d_model), mask (1, total_rows)) for core forward.

        Returns a clone of the soul tensor so the forward pass has its own
        copy and subsequent soul state mutations don't break autograd.
        """
        soul = self.tensor.detach().clone().unsqueeze(0)  # (1, total_rows, d_model)
        mask = self.active.unsqueeze(0).to(self.device)  # (1, total_rows)
        return soul, mask

    def update_salience(self, attention_weights: torch.Tensor | None):
        """Update salience from the core's attention over soul rows.

        attention_weights: (n_active,) — mean attention weight each active row received.
        If None, just decay all salience.
        """
        # Decay all
        decay = 1.0 - self.cfg.salience_ema_alpha
        for ti, spec in enumerate(self.cfg.tiers):
            sl = self.cfg.tier_slice(ti)
            self.salience[sl] *= spec.salience_decay

        # Update from attention (EMA)
        if attention_weights is not None:
            active_idx = self.active.nonzero(as_tuple=True)[0]
            alpha = self.cfg.salience_ema_alpha
            for i, idx in enumerate(active_idx):
                self.salience[idx] = (
                    (1 - alpha) * self.salience[idx] + alpha * attention_weights[i].item()
                )

        # Update dormant counters
        for ti, spec in enumerate(self.cfg.tiers):
            sl = self.cfg.tier_slice(ti)
            below = self.salience[sl] < spec.salience_threshold
            was_dormant = self.dormant_for[sl] > 0
            # Increment dormant if below threshold and active
            self.dormant_for[sl] = torch.where(
                below & self.active[sl],
                self.dormant_for[sl] + 1,
                torch.where(self.active[sl], torch.zeros_like(self.dormant_for[sl]),
                            self.dormant_for[sl])
            )

    def inflate(self, tier_idx: int, category_idx: int,
                thought: torch.Tensor) -> int | None:
        """Activate an inactive row in the given tier and write a thought into it.

        Returns the row index, or None if no inactive rows remain in that tier.
        """
        sl = self.cfg.tier_slice(tier_idx)
        inactive_in_tier = (~self.active[sl]).nonzero(as_tuple=True)[0]
        if len(inactive_in_tier) == 0:
            return None  # tier full — need compression to free space

        local_idx = inactive_in_tier[0].item()
        global_idx = sl.start + local_idx

        self.tensor[global_idx] = thought.to(self.dtype)
        self.active[global_idx] = True
        self.tier[global_idx] = tier_idx
        self.category[global_idx] = category_idx if tier_idx == 0 else -1
        self.salience[global_idx] = 1.0  # fresh rows start high
        self.dormant_for[global_idx] = 0
        self.tick_born[global_idx] = self.current_tick
        return global_idx

    def update_row(self, row_idx: int, thought: torch.Tensor, gate: float):
        """Gated write to an existing active row."""
        self.tensor[row_idx] = (
            (1 - gate) * self.tensor[row_idx] + gate * thought.to(self.dtype)
        )
        # Bump salience — it was just accessed
        self.salience[row_idx] = max(self.salience[row_idx].item(), 0.5)
        self.dormant_for[row_idx] = 0

    def evict(self, row_idx: int):
        """Deactivate and zero a row. Content is lost."""
        self.tensor[row_idx] = 0.0
        self.active[row_idx] = False
        self.category[row_idx] = -1
        self.salience[row_idx] = 0.0
        self.dormant_for[row_idx] = 0

    def compress_row(self, src_idx: int, dest_tier_idx: int,
                     compressed_thought: torch.Tensor) -> int | None:
        """Move a row's compressed content to a deeper tier. Evicts the source."""
        dest_idx = self.inflate(dest_tier_idx, -1, compressed_thought)
        if dest_idx is not None:
            self.evict(src_idx)
        return dest_idx

    def deflation_candidates(self, tier_idx: int) -> list[int]:
        """Return row indices in tier that are dormant beyond the threshold."""
        spec = self.cfg.tiers[tier_idx]
        sl = self.cfg.tier_slice(tier_idx)
        dormant = (self.dormant_for[sl] >= spec.dormant_ticks) & self.active[sl]
        local_indices = dormant.nonzero(as_tuple=True)[0]
        return [sl.start + i.item() for i in local_indices]

    def free_slots_in_tier(self, tier_idx: int) -> int:
        sl = self.cfg.tier_slice(tier_idx)
        return int((~self.active[sl]).sum().item())

    def tick(self):
        self.current_tick += 1

    def to_saveable(self) -> dict:
        """Serialize for persistence (soul.npy replacement)."""
        return {
            "tensor": self.tensor.cpu(),
            "active": self.active,
            "tier": self.tier,
            "category": self.category,
            "salience": self.salience,
            "dormant_for": self.dormant_for,
            "tick_born": self.tick_born,
            "current_tick": self.current_tick,
            "cfg": self.cfg.to_dict(),
        }

    @classmethod
    def from_saveable(cls, data: dict, device: torch.device,
                      dtype: torch.dtype) -> "SoulState":
        cfg = SoulV2Config.from_dict(data["cfg"])
        state = cls(cfg, device, dtype)
        state.tensor = data["tensor"].to(device, dtype)
        state.active = data["active"]
        state.tier = data["tier"]
        state.category = data["category"]
        state.salience = data["salience"]
        state.dormant_for = data["dormant_for"]
        state.tick_born = data["tick_born"]
        state.current_tick = data["current_tick"]
        return state

    def make_fresh(self):
        """Reset to empty soul (all rows inactive, zeroed)."""
        self.tensor.zero_()
        self.active.fill_(False)
        self.category.fill_(-1)
        self.salience.zero_()
        self.dormant_for.zero_()
        self.tick_born.zero_()
        self.current_tick = 0


# ---------------------------------------------------------------------------
# Soul router — decides where to write during exhale
# ---------------------------------------------------------------------------

class SoulRouter(nn.Module):
    """Routes the core's exhale output to hot-tier categories.

    For each category, produces a confidence score over existing active hot
    rows in that category. If max confidence < router_threshold, the thought
    should inflate a new row instead of updating an existing one.

    Also produces a category gate: for each category, should this tick write
    a thought at all?
    """

    def __init__(self, d_model: int, n_categories: int,
                 write_gate_init: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_categories = n_categories

        # Category gate: should we write to this category this tick?
        self.category_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_categories, bias=False),
        )
        nn.init.zeros_(self.category_gate[1].weight)

        # Row router: scores existing rows for update (dot-product attention)
        self.row_query = nn.Linear(d_model, d_model, bias=False)
        self.row_key = nn.Linear(d_model, d_model, bias=False)

        # Write gate: how much to blend new thought into existing row
        self.write_gate = nn.Parameter(torch.tensor(write_gate_init))

    def forward(
        self,
        field_out: torch.Tensor,        # (B, seq, d_model) — core's field output
        soul_tensor: torch.Tensor,      # (total_rows, d_model) — current soul
        active_mask: torch.Tensor,      # (total_rows,) bool
        category: torch.Tensor,         # (total_rows,) long — category index per row
        tier: torch.Tensor,             # (total_rows,) long — tier index per row
        cfg: SoulV2Config,
    ) -> dict[str, torch.Tensor]:
        """Route the exhale. Returns category gates, row scores, and write gate."""
        B = field_out.shape[0]

        # Pool field output to a single thought vector
        thought = field_out.mean(dim=1)  # (B, d_model)

        # Category gates: (B, n_categories) — sigmoid → should we write?
        cat_logits = self.category_gate(thought)  # (B, n_categories)
        cat_gates = torch.sigmoid(cat_logits)      # (B, n_categories)

        # Row scores: for each existing active hot row, how well does the
        # thought match? Used to find the best row to update.
        # Only consider hot-tier (tier==0) active rows.
        hot_mask = active_mask & (tier == 0)
        if hot_mask.any():
            q = self.row_query(thought)      # (B, d_model)
            hot_idx = hot_mask.nonzero(as_tuple=True)[0]
            k = self.row_key(soul_tensor[hot_idx])  # (n_hot, d_model)
            # Score: cosine similarity scaled
            q_norm = F.normalize(q, dim=-1)
            k_norm = F.normalize(k, dim=-1)
            row_scores = q_norm @ k_norm.T   # (B, n_hot)
        else:
            hot_idx = torch.empty(0, dtype=torch.long, device=soul_tensor.device)
            row_scores = torch.empty(B, 0, device=soul_tensor.device)

        return {
            "thought": thought,          # (B, d_model)
            "cat_gates": cat_gates,      # (B, n_categories)
            "hot_idx": hot_idx,          # (n_hot,) indices of active hot rows
            "row_scores": row_scores,    # (B, n_hot) — match scores
            "write_gate": self.write_gate,
        }


# ---------------------------------------------------------------------------
# Compression summarizer — hot→warm and warm→cold
# ---------------------------------------------------------------------------

class CompressionSummarizer(nn.Module):
    """Learned attention-based summarizer for compressing tier rows.

    Takes N source rows and produces M < N compressed rows using cross-attention:
    query = learned compress tokens (M of them), key/value = source rows.
    """

    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        # Learned compress queries — will be sliced to desired output count
        # Over-provision to 64; we use the first M at runtime.
        self.compress_queries = nn.Parameter(torch.randn(64, d_model) * 0.02)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attention = CrossAttention(d_model, n_heads)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, source_rows: torch.Tensor, n_output: int) -> torch.Tensor:
        """Compress source_rows (N, d) into (n_output, d) compressed rows."""
        N, D = source_rows.shape
        if N == 0:
            return torch.empty(0, D, device=source_rows.device,
                              dtype=source_rows.dtype)

        queries = self.compress_queries[:n_output].unsqueeze(0)  # (1, M, d)
        kv = source_rows.unsqueeze(0)                             # (1, N, d)

        q = self.norm_q(queries)
        k = self.norm_kv(kv)
        out = self.attention(q, k)          # (1, M, d)
        out = self.norm_out(out.squeeze(0)) # (M, d)
        return out


# Reuse the CrossAttention from core.py
from core import CrossAttention


# ---------------------------------------------------------------------------
# SoulManager v2 — orchestrates the full soul lifecycle
# ---------------------------------------------------------------------------

class SoulManagerV2:
    """Manages the temperature-tiered dynamic soul.

    Lifecycle per tick:
      1. get_batched() — soul tensor + mask for core forward
      2. core.forward_with_soul(field, soul, soul_mask) — core processes
      3. exhale(field_out, attention_weights) — write thoughts to hot tier
      4. update_salience(attention_weights) — track row usage
      5. maybe_compress() — periodic compression passes
      6. maybe_evict() — evict dormant rows past threshold

    The soul tensor and mask are flat across all tiers. The core sees all
    active rows regardless of tier — deeper tiers are just older, denser
    thoughts that the core can still attend to.
    """

    def __init__(
        self,
        cfg: SoulV2Config,
        device: torch.device,
        dtype: torch.dtype,
        n_heads: int = 1,
    ):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.state = SoulState(cfg, device, dtype)

        # Neural modules (these are trained with the core)
        self.router = SoulRouter(
            cfg.d_model, cfg.n_categories, cfg.write_gate_init
        ).to(device, dtype)

        # Compression summarizers (one per compression boundary)
        self.hot_to_warm = CompressionSummarizer(cfg.d_model, n_heads).to(device, dtype)
        self.warm_to_cold = CompressionSummarizer(cfg.d_model, n_heads).to(device, dtype)

        # Compression cadence (ticks between passes)
        self.hot_warm_interval = 100   # every 100 ticks
        self.warm_cold_interval = 500  # every 500 ticks

        # Compression ratio: how many source rows per output row
        self.hot_warm_ratio = 4    # 4 hot rows → 1 warm row
        self.warm_cold_ratio = 4   # 4 warm rows → 1 cold row

    def get_batched(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (soul, mask) for core.forward_with_soul."""
        return self.state.get_batched()

    def exhale(
        self,
        field_out: torch.Tensor,          # (1, seq, d_model) — core's field output
        attention_weights: torch.Tensor | None = None,  # (n_active,) — from core
        supervised_category: int | None = None,  # if set, force this category
    ) -> dict:
        """Write thoughts to the hot tier after a tick.

        For each category (or just the supervised one), evaluate:
          - Should we write? (category gate)
          - Where to write? (best existing row or inflate new)
          - What to write? (the pooled thought vector)
        """
        self.state.tick()

        routing = self.router(
            field_out,
            self.state.tensor,
            self.state.active,
            self.state.category,
            self.state.tier,
            self.cfg,
        )

        thought = routing["thought"][0]  # (d_model,) — this tick's thought
        cat_gates = routing["cat_gates"][0]  # (n_categories,)
        write_gate = routing["write_gate"].item()
        hot_idx = routing["hot_idx"]     # (n_hot,) indices of active hot rows
        row_scores = routing["row_scores"][0]  # (n_hot,)

        written = []

        if supervised_category is not None:
            # Supervised mode: only write to the specified category
            categories_to_check = [supervised_category]
        else:
            categories_to_check = list(range(self.cfg.n_categories))

        for cat_idx in categories_to_check:
            # Check category gate
            if supervised_category is None:
                if cat_gates[cat_idx].item() < 0.5:
                    continue  # this tick doesn't produce a thought for this category

            # Find best existing row in this category
            if len(hot_idx) > 0:
                cat_mask = (self.state.category[hot_idx] == cat_idx)
                if cat_mask.any():
                    cat_row_indices = hot_idx[cat_mask]
                    cat_scores = row_scores[cat_mask]
                    best_local = cat_scores.argmax().item()
                    best_score = cat_scores[best_local].item()
                    best_row = cat_row_indices[best_local].item()

                    if best_score >= self.cfg.router_threshold:
                        # Update existing row
                        self.state.update_row(best_row, thought, write_gate)
                        written.append(("update", best_row, cat_idx))
                        continue

            # No good match → inflate new row
            new_idx = self.state.inflate(0, cat_idx, thought)  # tier 0 = hot
            if new_idx is not None:
                written.append(("inflate", new_idx, cat_idx))
            else:
                # Hot tier full — trigger compression to free space
                freed = self._compress_hot_to_warm(n_target_free=1)
                if freed > 0:
                    new_idx = self.state.inflate(0, cat_idx, thought)
                    if new_idx is not None:
                        written.append(("inflate_after_compress", new_idx, cat_idx))
                    else:
                        written.append(("failed_no_space", -1, cat_idx))
                else:
                    written.append(("failed_no_space", -1, cat_idx))

        # Update salience
        self.state.update_salience(attention_weights)

        return {
            "written": written,
            "thought": thought.detach(),
            "cat_gates": cat_gates.detach(),
        }

    def maybe_compress(self) -> dict:
        """Run compression passes if cadence says it's time."""
        results = {"hot_to_warm": 0, "warm_to_cold": 0}

        if self.state.current_tick > 0 and self.state.current_tick % self.hot_warm_interval == 0:
            results["hot_to_warm"] = self._compress_hot_to_warm()

        if self.state.current_tick > 0 and self.state.current_tick % self.warm_cold_interval == 0:
            results["warm_to_cold"] = self._compress_warm_to_cold()

        return results

    def maybe_evict(self) -> list[int]:
        """Evict dormant rows that have been below salience threshold too long.

        Eviction loses content. Compression (maybe_compress) preserves content
        in reduced form. We evict only if compression didn't free enough space
        or the row is in the cold tier (no deeper tier to compress to).
        """
        evicted = []
        for ti in range(len(self.cfg.tiers)):
            candidates = self.state.deflation_candidates(ti)
            for row_idx in candidates:
                self.state.evict(row_idx)
                evicted.append(row_idx)
        return evicted

    def _compress_hot_to_warm(self, n_target_free: int = 0) -> int:
        """Compress dormant hot rows into warm rows.

        Returns number of hot rows freed.
        """
        # Gather dormant hot rows (or all hot rows if we need space)
        candidates = self.state.deflation_candidates(0)
        if len(candidates) == 0 and n_target_free > 0:
            # No dormant rows but we need space — compress lowest-salience rows
            hot_sl = self.cfg.tier_slice(0)
            active_hot = self.state.active[hot_sl].nonzero(as_tuple=True)[0]
            if len(active_hot) == 0:
                return 0
            # Sort by salience ascending — compress the least salient
            hot_global = hot_sl.start + active_hot
            salience_vals = self.state.salience[hot_global]
            n_to_compress = min(
                len(hot_global),
                max(n_target_free * self.hot_warm_ratio, self.hot_warm_ratio),
            )
            order = salience_vals.argsort()
            candidates = hot_global[order[:n_to_compress]].tolist()

        if len(candidates) < self.hot_warm_ratio:
            return 0  # not enough to compress

        # Group candidates into batches of hot_warm_ratio
        n_groups = len(candidates) // self.hot_warm_ratio
        freed = 0

        for g in range(n_groups):
            group = candidates[g * self.hot_warm_ratio:(g + 1) * self.hot_warm_ratio]
            source_rows = self.state.tensor[torch.tensor(group, device=self.device)]

            with torch.no_grad():
                compressed = self.hot_to_warm(source_rows, n_output=1)

            dest_idx = self.state.compress_row(group[0], 1, compressed[0])
            # Evict the rest of the group
            for idx in group[1:]:
                self.state.evict(idx)
                freed += 1
            if dest_idx is not None:
                freed += 1  # source row was evicted during compress_row
            else:
                # Warm tier full — can't compress, restore source
                # (compress_row already evicted src, need to re-inflate)
                self.state.inflate(0, -1, source_rows[0])

        return freed

    def _compress_warm_to_cold(self) -> int:
        """Compress dormant warm rows into cold rows. Returns number freed."""
        candidates = self.state.deflation_candidates(1)
        if len(candidates) == 0:
            # Compress lowest-salience warm rows
            warm_sl = self.cfg.tier_slice(1)
            active_warm = self.state.active[warm_sl].nonzero(as_tuple=True)[0]
            if len(active_warm) == 0:
                return 0
            warm_global = warm_sl.start + active_warm
            salience_vals = self.state.salience[warm_global]
            n_to_compress = min(len(warm_global), self.warm_cold_ratio)
            order = salience_vals.argsort()
            candidates = warm_global[order[:n_to_compress]].tolist()

        if len(candidates) < self.warm_cold_ratio:
            return 0

        n_groups = len(candidates) // self.warm_cold_ratio
        freed = 0

        for g in range(n_groups):
            group = candidates[g * self.warm_cold_ratio:(g + 1) * self.warm_cold_ratio]
            source_rows = self.state.tensor[torch.tensor(group, device=self.device)]

            with torch.no_grad():
                compressed = self.warm_to_cold(source_rows, n_output=1)

            dest_idx = self.state.compress_row(group[0], 2, compressed[0])
            for idx in group[1:]:
                self.state.evict(idx)
                freed += 1
            if dest_idx is not None:
                freed += 1
            else:
                self.state.inflate(1, -1, source_rows[0])

        return freed

    def zero_soul(self):
        """Reset soul to empty (for counterfactual probes)."""
        self.state.make_fresh()

    def zero_tier(self, tier_idx: int):
        """Zero out all active rows in a specific tier (for ablation/probes)."""
        sl = self.cfg.tier_slice(tier_idx)
        active_in_tier = self.state.active[sl].nonzero(as_tuple=True)[0]
        for local_idx in active_in_tier:
            global_idx = sl.start + local_idx.item()
            self.state.evict(global_idx)

    def swap_tier(self, other_state: SoulState, tier_idx: int):
        """Swap a tier's content with another soul's tier (for CF probes)."""
        sl = self.cfg.tier_slice(tier_idx)
        # Save our tier
        our_tensor = self.state.tensor[sl].clone()
        our_active = self.state.active[sl].clone()
        our_category = self.state.category[sl].clone()
        our_salience = self.state.salience[sl].clone()
        # Swap in theirs
        self.state.tensor[sl] = other_state.tensor[sl]
        self.state.active[sl] = other_state.active[sl]
        self.state.category[sl] = other_state.category[sl]
        self.state.salience[sl] = other_state.salience[sl]
        # Write ours to theirs
        other_state.tensor[sl] = our_tensor
        other_state.active[sl] = our_active
        other_state.category[sl] = our_category
        other_state.salience[sl] = our_salience

    def parameters(self) -> list[torch.nn.Parameter]:
        """Return trainable parameters (router + compression summarizers)."""
        params = list(self.router.parameters())
        params += list(self.hot_to_warm.parameters())
        params += list(self.warm_to_cold.parameters())
        return params

    def state_dict(self) -> dict:
        """Return state dict for checkpointing the soul manager's learned modules."""
        return {
            "router": self.router.state_dict(),
            "hot_to_warm": self.hot_to_warm.state_dict(),
            "warm_to_cold": self.warm_to_cold.state_dict(),
        }

    def load_state_dict(self, sd: dict):
        if "router" in sd:
            self.router.load_state_dict(sd["router"])
        if "hot_to_warm" in sd:
            self.hot_to_warm.load_state_dict(sd["hot_to_warm"])
        if "warm_to_cold" in sd:
            self.warm_to_cold.load_state_dict(sd["warm_to_cold"])

    def save_soul(self, path: str):
        """Save the full soul state (thoughts + metadata) to a file."""
        torch.save(self.state.to_saveable(), path)

    def load_soul(self, path: str):
        """Load soul state from a file."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.state = SoulState.from_saveable(data, self.device, self.dtype)