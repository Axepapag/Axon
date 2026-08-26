"""First permanent neural Heart translation/conduction core.

This module is model tissue, not Heart authority.  It consumes exact frozen 16D
character substrate cells, learns source/destination dialect-conditioned semantic
transport, and emits a destination character distribution plus explicit semantic
and grounding heads.  Canonical Shared Field mutation remains outside this model
and behind the deterministic Heart transaction boundary.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import nn

from substrate import get_letter_bank

HEART_TRANSLATION_ARCHITECTURE_V3 = "heart-translation-64d-real-d64-paged-complete-field-v3"
HEART_TRANSLATION_ARCHITECTURE_V4 = (
    "heart-translation-64d-real-d64-paged-complete-field-positional-copy-v4"
)
HEART_TRANSLATION_ARCHITECTURE_V5 = (
    "heart-translation-64d-real-d64-paged-complete-field-governed-positional-copy-v5"
)
# Compatibility name for the accepted permanent v3 tissue. New anatomy must use
# heart_translation_architecture_id(cfg) rather than silently relabeling v3.
HEART_TRANSLATION_ARCHITECTURE = HEART_TRANSLATION_ARCHITECTURE_V3

HEART_SEMANTIC_LABELS: Mapping[str, tuple[str, ...]] = {
    "polarity_negation": ("positive", "negative"),
    "modality": ("asserted", "capability", "obligation", "possibility"),
    "quantification": ("singular", "generic", "universal", "existential", "zero"),
    "temporal_relation": ("atemporal", "past", "present", "future"),
    "causal_relation": ("none", "cause_before_effect", "effect_because_cause"),
    "speech_act": ("assertion", "question", "request", "warning"),
}


@dataclass(frozen=True, slots=True)
class HeartTranslationCoreConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 4096
    dropout: float = 0.05
    source_page_chars: int = 256
    max_dialects: int = 16
    lift_seed: int = 41
    positional_copy: bool = False
    positional_copy_requires_opt_in: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int) or self.d_model < 64:
            raise ValueError("Heart translation d_model must be an integer >= 64")
        if isinstance(self.n_heads, bool) or not isinstance(self.n_heads, int) or self.n_heads < 1:
            raise ValueError("n_heads must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if isinstance(self.n_layers, bool) or not isinstance(self.n_layers, int) or self.n_layers < 1:
            raise ValueError("n_layers must be a positive integer")
        if isinstance(self.ffn_dim, bool) or not isinstance(self.ffn_dim, int) or self.ffn_dim < self.d_model:
            raise ValueError("ffn_dim must be an integer >= d_model")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if (
            isinstance(self.source_page_chars, bool)
            or not isinstance(self.source_page_chars, int)
            or self.source_page_chars < 1
        ):
            raise ValueError("source_page_chars must be a positive processing-unit size")
        if self.max_dialects < 2:
            raise ValueError("max_dialects must be at least two")
        if not isinstance(self.positional_copy, bool):
            raise ValueError("positional_copy must be boolean")
        if not isinstance(self.positional_copy_requires_opt_in, bool):
            raise ValueError("positional_copy_requires_opt_in must be boolean")
        if self.positional_copy_requires_opt_in and not self.positional_copy:
            raise ValueError("positional-copy opt-in requires positional-copy anatomy")

    def to_canonical_dict(self) -> dict[str, int | float]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": float(self.dropout),
            "source_page_chars": self.source_page_chars,
            "max_dialects": self.max_dialects,
            "lift_seed": self.lift_seed,
            "positional_copy": self.positional_copy,
            "positional_copy_requires_opt_in": self.positional_copy_requires_opt_in,
        }


def heart_translation_architecture_id(cfg: HeartTranslationCoreConfig) -> str:
    default = HeartTranslationCoreConfig()
    if cfg == default:
        return HEART_TRANSLATION_ARCHITECTURE_V3
    if (
        cfg.positional_copy
        and cfg.positional_copy_requires_opt_in
        and replace(
            cfg,
            positional_copy=False,
            positional_copy_requires_opt_in=False,
        )
        == default
    ):
        return HEART_TRANSLATION_ARCHITECTURE_V5
    if (
        cfg.positional_copy
        and not cfg.positional_copy_requires_opt_in
        and replace(cfg, positional_copy=False) == default
    ):
        return HEART_TRANSLATION_ARCHITECTURE_V4
    return (
        f"heart-translation-{cfg.d_model}d-{cfg.n_layers}l-"
        f"{cfg.n_heads}h-{cfg.ffn_dim}ffn-paged-v3"
        + ("-positional-copy" if cfg.positional_copy else "")
        + ("-governed-opt-in" if cfg.positional_copy_requires_opt_in else "")
    )


def _frozen_orthogonal_lift(d_model: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + d_model)
    matrix = torch.randn(d_model, 16, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.T.contiguous().to(torch.float32)


@dataclass(slots=True)
class HeartDecoderTrace:
    """Inspectable decoder evidence; the gate is generation probability."""

    target_log_probs: torch.Tensor
    memory_attention: torch.Tensor
    generation_gate: torch.Tensor
    positional_copy_gate: torch.Tensor | None
    positional_copy_available: torch.Tensor | None


@dataclass(slots=True)
class HeartTranslationOutput:
    target_log_probs: torch.Tensor
    decoder_trace: HeartDecoderTrace
    semantic_logits: dict[str, torch.Tensor]
    referent_start_logits: torch.Tensor
    referent_end_logits: torch.Tensor
    grounding_start_logits: torch.Tensor
    grounding_end_logits: torch.Tensor
    source_coverage: "HeartSourceCoverage"


@dataclass(frozen=True, slots=True)
class HeartGeneratedTranslation:
    text: str
    terminated: bool
    generated_characters: int


@dataclass(frozen=True, slots=True)
class HeartSourceCoverage:
    """Auditable proof that both ordered source sweeps visited every character."""

    page_chars: int
    sweeps: int
    source_characters: tuple[int, ...]
    source_index_sha256: tuple[str, ...]
    visited_characters_per_sweep: tuple[tuple[int, ...], ...]
    page_spans: tuple[tuple[tuple[int, int], ...], ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class _HeartSourceSweepTrace:
    visited_characters: tuple[int, ...]
    page_spans: tuple[tuple[tuple[int, int], ...], ...]


def _sinusoidal_positions(
    positions: torch.Tensor,
    d_model: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Length-unbounded deterministic positions; no learned table can run out."""

    if positions.ndim != 2:
        raise ValueError("positions must be rank-2")
    work = positions.to(dtype=torch.float32).unsqueeze(-1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    angles = work * frequencies
    encoded = torch.zeros(*positions.shape, d_model, device=positions.device, dtype=torch.float32)
    encoded[..., 0::2] = torch.sin(angles)
    encoded[..., 1::2] = torch.cos(angles[..., : encoded[..., 1::2].shape[-1]])
    return encoded.to(dtype=dtype)


class HeartTranslationCore(nn.Module):
    """A shallow, FFN-heavy cardiac translator grounded directly in 16D cells.

    The core has explicit learned query lanes for each critical semantic family,
    referent identity, and grounding.  These are observable auxiliary heads rather
    than hidden claims that surface text reconstruction alone preserved meaning.
    """

    def __init__(self, cfg: HeartTranslationCoreConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or HeartTranslationCoreConfig()
        self.cfg = cfg
        bank = get_letter_bank()
        self.characters = tuple(bank.chars[:-1])
        self.char_to_index = {char: index for index, char in enumerate(self.characters)}
        self.vocab_size = len(self.characters)
        self.source_pad_index = self.vocab_size
        self.eos_index = self.vocab_size
        self.bos_index = self.vocab_size + 1
        self.decoder_pad_index = self.vocab_size + 2

        # The canonical D64 rail carries literal frozen substrate cells, not
        # cosine-normalized renderer helpers.  V3 therefore learns from the
        # exact raw 16D cells produced by char_to_slot.
        self.register_buffer("bank16", torch.from_numpy(bank.vecs[:-1].copy()).float())
        self.register_buffer("char_lift", _frozen_orthogonal_lift(cfg.d_model, cfg.lift_seed))

        self.source_dialect_embedding = nn.Embedding(cfg.max_dialects, cfg.d_model)
        self.destination_dialect_embedding = nn.Embedding(cfg.max_dialects, cfg.d_model)

        self.query_names = (
            "global",
            "referent_identity",
            "grounding_provenance",
            *tuple(HEART_SEMANTIC_LABELS.keys()),
        )
        self.query_to_index = {name: index for index, name in enumerate(self.query_names)}
        self.query_tokens = nn.Parameter(torch.randn(len(self.query_names), cfg.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.encoder_norm = nn.LayerNorm(cfg.d_model)

        self.semantic_heads = nn.ModuleDict(
            {
                name: nn.Linear(cfg.d_model, len(labels))
                for name, labels in HEART_SEMANTIC_LABELS.items()
            }
        )
        self.referent_start_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.referent_end_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.grounding_start_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.grounding_end_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.decoder_embedding = nn.Embedding(self.vocab_size + 3, cfg.d_model)
        self.decoder_init = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.decoder = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.decoder_memory_attention = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.decoder_output = nn.Linear(cfg.d_model, self.vocab_size + 1)
        self.copy_gate = nn.Linear(cfg.d_model * 2, 1)
        self.positional_copy_gate = (
            nn.Linear(cfg.d_model * 2, 1) if cfg.positional_copy else None
        )

        nn.init.normal_(self.source_dialect_embedding.weight, std=0.02)
        nn.init.normal_(self.destination_dialect_embedding.weight, std=0.02)
        nn.init.normal_(self.decoder_embedding.weight, std=0.02)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, 0.5)
        if self.positional_copy_gate is not None:
            # Exact-zero preserves every v3 output at the migration boundary.
            # clamp() retains a useful gradient at zero for the identity route.
            nn.init.zeros_(self.positional_copy_gate.weight)
            nn.init.zeros_(self.positional_copy_gate.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _validate_dialects(self, source_dialect_ids: torch.Tensor, destination_dialect_ids: torch.Tensor) -> None:
        for name, ids in (
            ("source_dialect_ids", source_dialect_ids),
            ("destination_dialect_ids", destination_dialect_ids),
        ):
            if ids.ndim != 1:
                raise ValueError(f"{name} must be rank-1")
            if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.cfg.max_dialects):
                raise ValueError(f"{name} contains an out-of-range dialect id")

    def _source_lift(self, source_indices: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        safe = source_indices.clamp(min=0, max=max(0, self.vocab_size - 1))
        cells = self.bank16[safe]
        cells = cells * source_mask.unsqueeze(-1).to(cells.dtype)
        return cells @ self.char_lift

    def _source_lift_cells(self, source_cells16: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        if source_cells16.ndim != 3 or source_cells16.shape[:2] != source_mask.shape:
            raise ValueError("source_cells16 must have shape [batch, characters, 16]")
        if source_cells16.shape[2] != 16:
            raise ValueError("source_cells16 last dimension must be the exact 16D substrate")
        if not bool(torch.isfinite(source_cells16).all()):
            raise ValueError("source_cells16 must be finite")
        cells = source_cells16.to(dtype=self.char_lift.dtype)
        cells = cells * source_mask.unsqueeze(-1).to(cells.dtype)
        return cells @ self.char_lift

    def _source_coverage(
        self,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        traces: tuple[_HeartSourceSweepTrace, ...],
    ) -> HeartSourceCoverage:
        lengths = source_mask.sum(dim=1).tolist()
        page_spans = tuple(
            tuple(
                (start, min(start + self.cfg.source_page_chars, int(length)))
                for start in range(0, int(length), self.cfg.source_page_chars)
            )
            for length in lengths
        )
        expected = tuple(int(length) for length in lengths)
        source_hashes = tuple(
            hashlib.sha256(
                bytes(
                    int(value)
                    for value in source_indices[row, : int(length)].detach().cpu().tolist()
                )
            ).hexdigest()
            for row, length in enumerate(lengths)
        )
        return HeartSourceCoverage(
            page_chars=self.cfg.source_page_chars,
            sweeps=len(traces),
            source_characters=expected,
            source_index_sha256=source_hashes,
            visited_characters_per_sweep=tuple(trace.visited_characters for trace in traces),
            page_spans=page_spans,
            complete=bool(
                len(traces) == 2
                and all(trace.visited_characters == expected for trace in traces)
                and all(trace.page_spans == page_spans for trace in traces)
            ),
        )

    def _source_sweep(
        self,
        *,
        query_states: torch.Tensor,
        lifted_chars: torch.Tensor,
        source_mask: torch.Tensor,
        source_dialect: torch.Tensor,
        destination_dialect: torch.Tensor,
        source_positions: torch.Tensor,
        collect_memory: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, _HeartSourceSweepTrace]:
        """Visit all source pages in order while carrying learned query state."""

        batch, source_length, _ = lifted_chars.shape
        memory_pages: list[torch.Tensor] = []
        visited_characters = [0] * batch
        visited_spans: list[list[tuple[int, int]]] = [[] for _ in range(batch)]
        prefix_mask = torch.ones(
            batch,
            len(self.query_names),
            dtype=torch.bool,
            device=source_mask.device,
        )
        for start in range(0, source_length, self.cfg.source_page_chars):
            end = min(start + self.cfg.source_page_chars, source_length)
            page_mask = source_mask[:, start:end]
            page_counts = page_mask.sum(dim=1).tolist()
            for row, count in enumerate(page_counts):
                count = int(count)
                if count:
                    visited_characters[row] += count
                    visited_spans[row].append((start, start + count))
            positions = source_positions[:, start:end]
            page_chars = (
                lifted_chars[:, start:end]
                + _sinusoidal_positions(positions, self.cfg.d_model, dtype=lifted_chars.dtype)
                + source_dialect
                + destination_dialect
            )
            valid = torch.cat((prefix_mask, page_mask), dim=1)
            encoded = self.encoder(
                torch.cat((query_states, page_chars), dim=1),
                src_key_padding_mask=~valid,
            )
            encoded = self.encoder_norm(encoded)
            updated_queries = encoded[:, : len(self.query_names)]
            active_rows = page_mask.any(dim=1).view(batch, 1, 1)
            query_states = torch.where(active_rows, updated_queries, query_states)
            if collect_memory:
                memory_pages.append(encoded[:, len(self.query_names) :])
        memory = torch.cat(memory_pages, dim=1) if collect_memory else None
        return (
            query_states,
            memory,
            _HeartSourceSweepTrace(
                visited_characters=tuple(visited_characters),
                page_spans=tuple(tuple(row) for row in visited_spans),
            ),
        )

    def _encode(
        self,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        source_dialect_ids: torch.Tensor,
        destination_dialect_ids: torch.Tensor,
        source_cells16: torch.Tensor | None = None,
        source_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, HeartSourceCoverage]:
        if source_indices.ndim != 2 or source_mask.shape != source_indices.shape:
            raise ValueError("source_indices/source_mask must be matching rank-2 tensors")
        if source_indices.shape[0] != source_dialect_ids.shape[0] or source_indices.shape[0] != destination_dialect_ids.shape[0]:
            raise ValueError("dialect ids must match source batch size")
        self._validate_dialects(source_dialect_ids, destination_dialect_ids)
        if not bool(source_mask.any(dim=1).all()):
            raise ValueError("every Heart translation source must contain at least one exact character")

        lengths = source_mask.sum(dim=1)
        expected_mask = (
            torch.arange(source_indices.shape[1], device=source_mask.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(source_mask, expected_mask):
            raise ValueError("Heart source masks must be contiguous exact-character prefixes")

        if source_positions is None:
            source_positions = torch.arange(
                source_indices.shape[1], device=source_indices.device, dtype=torch.long
            ).unsqueeze(0).expand(source_indices.shape[0], -1)
        if source_positions.shape != source_indices.shape:
            raise ValueError("source_positions must match source_indices")
        if bool((source_positions[source_mask] < 0).any()):
            raise ValueError("valid source_positions must be non-negative")
        for row, length in enumerate(lengths.tolist()):
            valid_positions = source_positions[row, : int(length)]
            if valid_positions.numel() > 1 and not bool(
                (valid_positions[1:] > valid_positions[:-1]).all()
            ):
                raise ValueError("valid source_positions must be strictly increasing")

        batch = source_indices.shape[0]
        source_dialect = self.source_dialect_embedding(source_dialect_ids).unsqueeze(1)
        destination_dialect = self.destination_dialect_embedding(destination_dialect_ids).unsqueeze(1)
        lifted_chars = (
            self._source_lift(source_indices, source_mask)
            if source_cells16 is None
            else self._source_lift_cells(source_cells16, source_mask)
        )
        query_states = (
            self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
            + source_dialect
            + destination_dialect
        )
        # Sweep one builds a complete-field recurrent state. Sweep two starts
        # from that state so every addressable character representation is
        # conditioned on a summary that has already visited the complete source.
        query_states, _, first_trace = self._source_sweep(
            query_states=query_states,
            lifted_chars=lifted_chars,
            source_mask=source_mask,
            source_dialect=source_dialect,
            destination_dialect=destination_dialect,
            source_positions=source_positions,
            collect_memory=False,
        )
        query_states, memory, second_trace = self._source_sweep(
            query_states=query_states,
            lifted_chars=lifted_chars,
            source_mask=source_mask,
            source_dialect=source_dialect,
            destination_dialect=destination_dialect,
            source_positions=source_positions,
            collect_memory=True,
        )
        if memory is None:
            raise RuntimeError("complete Heart source sweep produced no addressable memory")
        coverage = self._source_coverage(
            source_indices,
            source_mask,
            (first_trace, second_trace),
        )
        if not coverage.complete:
            raise RuntimeError("Heart source sweep failed exact complete-field coverage")
        return query_states, memory, coverage

    def _pointer_logits(
        self,
        query_state: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        query = projection(query_state).unsqueeze(1)
        logits = (memory * query).sum(dim=-1) / math.sqrt(self.cfg.d_model)
        return logits.masked_fill(~source_mask, torch.finfo(logits.dtype).min)

    def _decode(
        self,
        query_states: torch.Tensor,
        memory: torch.Tensor,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        destination_dialect_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        allow_positional_copy_route: bool = False,
    ) -> HeartDecoderTrace:
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must be rank-2")
        batch, target_length = decoder_input_ids.shape
        if batch != source_indices.shape[0]:
            raise ValueError("decoder batch must match source batch")
        if not isinstance(allow_positional_copy_route, bool):
            raise ValueError("allow_positional_copy_route must be boolean")
        positions = torch.arange(target_length, device=decoder_input_ids.device).unsqueeze(0).expand(batch, target_length)
        destination_dialect = self.destination_dialect_embedding(destination_dialect_ids)
        decoder_inputs = (
            self.decoder_embedding(decoder_input_ids)
            + _sinusoidal_positions(
                positions,
                self.cfg.d_model,
                dtype=self.decoder_embedding.weight.dtype,
            )
            + destination_dialect.unsqueeze(1)
        )
        global_state = query_states[:, self.query_to_index["global"]]
        hidden0 = torch.tanh(self.decoder_init(torch.cat((global_state, destination_dialect), dim=-1))).unsqueeze(0)
        # Deep-copied Trainer candidates can lose cuDNN's contiguous GRU
        # packing. Re-flattening is value-preserving and avoids repeated
        # internal repacks during long autoregressive diagnostics.
        self.decoder.flatten_parameters()
        recurrent, _ = self.decoder(decoder_inputs, hidden0)
        attended, attention = self.decoder_memory_attention(
            recurrent,
            memory,
            memory,
            key_padding_mask=~source_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        hidden = self.decoder_norm(recurrent + attended)
        generation = torch.softmax(self.decoder_output(hidden), dim=-1)

        copy = torch.zeros_like(generation)
        source_ids = source_indices.clamp(min=0, max=max(0, self.vocab_size - 1))
        copy_indices = source_ids.unsqueeze(1).expand(batch, target_length, source_ids.shape[1])
        copy.scatter_add_(2, copy_indices, attention * source_mask.unsqueeze(1).to(attention.dtype))
        gate = torch.sigmoid(self.copy_gate(torch.cat((hidden, attended), dim=-1)))
        probs = gate * generation + (1.0 - gate) * copy
        positional_gate: torch.Tensor | None = None
        positional_available: torch.Tensor | None = None
        route_exposed = self.positional_copy_gate is not None and (
            not self.cfg.positional_copy_requires_opt_in or allow_positional_copy_route
        )
        if self.positional_copy_gate is not None:
            source_lengths = source_mask.sum(dim=1)
            target_positions = torch.arange(target_length, device=source_indices.device).view(1, -1)
            character_available = target_positions < source_lengths.view(-1, 1)
            eos_available = target_positions.eq(source_lengths.view(-1, 1))
            positional_available = (character_available | eos_available) & route_exposed
            positional = torch.zeros_like(generation)
            safe_positions = target_positions.clamp(max=max(0, source_indices.shape[1] - 1)).expand(
                batch, -1
            )
            same_address_ids = source_indices.gather(1, safe_positions).clamp(
                min=0, max=max(0, self.vocab_size - 1)
            )
            positional.scatter_(2, same_address_ids.unsqueeze(-1), character_available.unsqueeze(-1).to(positional.dtype))
            positional[:, :, self.eos_index] += eos_available.to(positional.dtype)
            raw_positional_gate = self.positional_copy_gate(torch.cat((hidden, attended), dim=-1))
            positional_gate = raw_positional_gate.clamp(min=0.0, max=1.0)
            positional_gate = positional_gate * positional_available.unsqueeze(-1).to(positional_gate.dtype)
            # At the initialized exact-zero gate this is bit-preserving v3
            # behavior. Training may add same-address conduction without
            # removing the learned generation or content-attention routes.
            probs = probs + positional_gate * (positional - probs)
        return HeartDecoderTrace(
            target_log_probs=probs.clamp_min(torch.finfo(probs.dtype).tiny).log(),
            memory_attention=attention,
            generation_gate=gate,
            positional_copy_gate=positional_gate,
            positional_copy_available=positional_available,
        )

    def forward(
        self,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        source_dialect_ids: torch.Tensor,
        destination_dialect_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        source_cells16: torch.Tensor | None = None,
        source_positions: torch.Tensor | None = None,
        allow_positional_copy_route: bool = False,
    ) -> HeartTranslationOutput:
        query_states, memory, coverage = self._encode(
            source_indices,
            source_mask,
            source_dialect_ids,
            destination_dialect_ids,
            source_cells16,
            source_positions,
        )
        semantic_logits = {
            name: self.semantic_heads[name](query_states[:, self.query_to_index[name]])
            for name in HEART_SEMANTIC_LABELS
        }
        referent = query_states[:, self.query_to_index["referent_identity"]]
        grounding = query_states[:, self.query_to_index["grounding_provenance"]]
        decoder_trace = self._decode(
            query_states,
            memory,
            source_indices,
            source_mask,
            destination_dialect_ids,
            decoder_input_ids,
            allow_positional_copy_route=allow_positional_copy_route,
        )
        return HeartTranslationOutput(
            target_log_probs=decoder_trace.target_log_probs,
            decoder_trace=decoder_trace,
            semantic_logits=semantic_logits,
            referent_start_logits=self._pointer_logits(
                referent, memory, source_mask, self.referent_start_query
            ),
            referent_end_logits=self._pointer_logits(
                referent, memory, source_mask, self.referent_end_query
            ),
            grounding_start_logits=self._pointer_logits(
                grounding, memory, source_mask, self.grounding_start_query
            ),
            grounding_end_logits=self._pointer_logits(
                grounding, memory, source_mask, self.grounding_end_query
            ),
            source_coverage=coverage,
        )

    @torch.no_grad()
    def greedy_translate(
        self,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        source_dialect_ids: torch.Tensor,
        destination_dialect_ids: torch.Tensor,
        *,
        max_chars: int,
        source_cells16: torch.Tensor | None = None,
        source_positions: torch.Tensor | None = None,
        allow_positional_copy_route: bool = False,
    ) -> tuple[HeartGeneratedTranslation, ...]:
        limit = int(max_chars)
        if limit < 1:
            raise ValueError("max_chars must be positive")
        query_states, memory, _ = self._encode(
            source_indices,
            source_mask,
            source_dialect_ids,
            destination_dialect_ids,
            source_cells16,
            source_positions,
        )
        batch = source_indices.shape[0]
        generated = torch.full((batch, 1), self.bos_index, dtype=torch.long, device=source_indices.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=source_indices.device)
        output_indices: list[list[int]] = [[] for _ in range(batch)]
        for _ in range(limit):
            decoder_trace = self._decode(
                query_states,
                memory,
                source_indices,
                source_mask,
                destination_dialect_ids,
                generated,
                allow_positional_copy_route=allow_positional_copy_route,
            )
            next_ids = decoder_trace.target_log_probs[:, -1].argmax(dim=-1)
            for row, value in enumerate(next_ids.tolist()):
                if finished[row]:
                    continue
                if value == self.eos_index:
                    finished[row] = True
                else:
                    output_indices[row].append(value)
            if bool(finished.all()):
                break
            decoder_next = next_ids.clone()
            decoder_next[finished] = self.decoder_pad_index
            generated = torch.cat((generated, decoder_next.unsqueeze(1)), dim=1)
        return tuple(
            HeartGeneratedTranslation(
                text="".join(self.characters[index] for index in row),
                terminated=bool(finished[index]),
                generated_characters=len(row),
            )
            for index, row in enumerate(output_indices)
        )


def migrate_heart_translation_v3_to_v4(
    source_state_dict: Mapping[str, torch.Tensor],
    source_config: HeartTranslationCoreConfig,
) -> tuple[HeartTranslationCoreConfig, OrderedDict[str, torch.Tensor]]:
    """Add exact-zero positional-copy tissue without altering v3 parameters."""

    if source_config.positional_copy:
        raise ValueError("source config is already positional-copy anatomy")
    source_model = HeartTranslationCore(source_config)
    source_model.load_state_dict(source_state_dict, strict=True)
    destination_config = replace(source_config, positional_copy=True)
    destination_model = HeartTranslationCore(destination_config)
    destination_state = destination_model.state_dict()
    source_keys = set(source_model.state_dict())
    destination_keys = set(destination_state)
    new_keys = destination_keys - source_keys
    expected_new_keys = {
        "positional_copy_gate.weight",
        "positional_copy_gate.bias",
    }
    if new_keys != expected_new_keys or source_keys - destination_keys:
        raise RuntimeError("unexpected v3-to-v4 Heart state topology")
    migrated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, destination_value in destination_state.items():
        if name in source_state_dict:
            source_value = source_state_dict[name]
            if source_value.shape != destination_value.shape or source_value.dtype != destination_value.dtype:
                raise RuntimeError(f"incompatible v3 Heart tensor {name}")
            migrated[name] = source_value.detach().clone()
        else:
            migrated[name] = torch.zeros_like(destination_value)
    return destination_config, migrated


def migrate_heart_translation_v3_to_v5(
    source_state_dict: Mapping[str, torch.Tensor],
    source_config: HeartTranslationCoreConfig,
) -> tuple[HeartTranslationCoreConfig, OrderedDict[str, torch.Tensor]]:
    """Add exact-zero same-address tissue behind a governed opt-in boundary."""

    destination_config, migrated = migrate_heart_translation_v3_to_v4(
        source_state_dict,
        source_config,
    )
    return replace(destination_config, positional_copy_requires_opt_in=True), migrated


__all__ = [
    "HEART_SEMANTIC_LABELS",
    "HEART_TRANSLATION_ARCHITECTURE",
    "HEART_TRANSLATION_ARCHITECTURE_V3",
    "HEART_TRANSLATION_ARCHITECTURE_V4",
    "HEART_TRANSLATION_ARCHITECTURE_V5",
    "HeartDecoderTrace",
    "HeartGeneratedTranslation",
    "HeartSourceCoverage",
    "HeartTranslationCore",
    "HeartTranslationCoreConfig",
    "HeartTranslationOutput",
    "heart_translation_architecture_id",
    "migrate_heart_translation_v3_to_v4",
    "migrate_heart_translation_v3_to_v5",
]
