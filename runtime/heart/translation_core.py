"""First permanent neural Heart translation/conduction core.

This module is model tissue, not Heart authority.  It consumes exact frozen 16D
character substrate cells, learns source/destination dialect-conditioned semantic
transport, and emits a destination character distribution plus explicit semantic
and grounding heads.  Canonical Shared Field mutation remains outside this model
and behind the deterministic Heart transaction boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from substrate import get_letter_bank


HEART_TRANSLATION_ARCHITECTURE = "heart-translation-64d-shallow-ffn-v1"

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
    max_source_chars: int = 192
    max_target_chars: int = 192
    max_dialects: int = 16
    lift_seed: int = 41

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
        if self.max_source_chars < 1 or self.max_target_chars < 1:
            raise ValueError("source/target character limits must be positive")
        if self.max_dialects < 2:
            raise ValueError("max_dialects must be at least two")

    def to_canonical_dict(self) -> dict[str, int | float]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": float(self.dropout),
            "max_source_chars": self.max_source_chars,
            "max_target_chars": self.max_target_chars,
            "max_dialects": self.max_dialects,
            "lift_seed": self.lift_seed,
        }


def heart_translation_architecture_id(cfg: HeartTranslationCoreConfig) -> str:
    default = HeartTranslationCoreConfig()
    if cfg == default:
        return HEART_TRANSLATION_ARCHITECTURE
    return (
        f"heart-translation-{cfg.d_model}d-{cfg.n_layers}l-"
        f"{cfg.n_heads}h-{cfg.ffn_dim}ffn-v1"
    )


def _frozen_orthogonal_lift(d_model: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + d_model)
    matrix = torch.randn(d_model, 16, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.T.contiguous().to(torch.float32)


@dataclass(slots=True)
class HeartTranslationOutput:
    target_log_probs: torch.Tensor
    semantic_logits: dict[str, torch.Tensor]
    referent_start_logits: torch.Tensor
    referent_end_logits: torch.Tensor
    grounding_start_logits: torch.Tensor
    grounding_end_logits: torch.Tensor


class HeartTranslationCore(nn.Module):
    """A shallow, FFN-heavy cardiac translator grounded directly in 16D cells.

    The core has explicit learned query lanes for each critical semantic family,
    referent identity, and grounding.  These are observable auxiliary heads rather
    than hidden claims that surface text reconstruction alone preserved meaning.
    """

    def __init__(self, cfg: HeartTranslationCoreConfig = HeartTranslationCoreConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        bank = get_letter_bank()
        self.characters = tuple(bank.chars[:-1])
        self.char_to_index = {char: index for index, char in enumerate(self.characters)}
        self.vocab_size = len(self.characters)
        self.source_pad_index = self.vocab_size
        self.eos_index = self.vocab_size
        self.bos_index = self.vocab_size + 1
        self.decoder_pad_index = self.vocab_size + 2

        self.register_buffer("bank16", torch.from_numpy(bank.vecs_unit[:-1].copy()).float())
        self.register_buffer("char_lift", _frozen_orthogonal_lift(cfg.d_model, cfg.lift_seed))

        self.source_dialect_embedding = nn.Embedding(cfg.max_dialects, cfg.d_model)
        self.destination_dialect_embedding = nn.Embedding(cfg.max_dialects, cfg.d_model)
        self.source_position_embedding = nn.Embedding(cfg.max_source_chars, cfg.d_model)

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
        self.decoder_position_embedding = nn.Embedding(cfg.max_target_chars + 1, cfg.d_model)
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

        nn.init.normal_(self.source_dialect_embedding.weight, std=0.02)
        nn.init.normal_(self.destination_dialect_embedding.weight, std=0.02)
        nn.init.normal_(self.source_position_embedding.weight, std=0.02)
        nn.init.normal_(self.decoder_embedding.weight, std=0.02)
        nn.init.normal_(self.decoder_position_embedding.weight, std=0.02)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, 0.5)

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

    def _encode(
        self,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        source_dialect_ids: torch.Tensor,
        destination_dialect_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_indices.ndim != 2 or source_mask.shape != source_indices.shape:
            raise ValueError("source_indices/source_mask must be matching rank-2 tensors")
        if source_indices.shape[1] > self.cfg.max_source_chars:
            raise ValueError("source exceeds configured Heart source character limit")
        if source_indices.shape[0] != source_dialect_ids.shape[0] or source_indices.shape[0] != destination_dialect_ids.shape[0]:
            raise ValueError("dialect ids must match source batch size")
        self._validate_dialects(source_dialect_ids, destination_dialect_ids)
        if not bool(source_mask.any(dim=1).all()):
            raise ValueError("every Heart translation source must contain at least one exact character")

        batch, source_length = source_indices.shape
        positions = torch.arange(source_length, device=source_indices.device).unsqueeze(0).expand(batch, source_length)
        source_dialect = self.source_dialect_embedding(source_dialect_ids).unsqueeze(1)
        destination_dialect = self.destination_dialect_embedding(destination_dialect_ids).unsqueeze(1)
        chars = (
            self._source_lift(source_indices, source_mask)
            + self.source_position_embedding(positions)
            + source_dialect
            + destination_dialect
        )
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1) + source_dialect + destination_dialect
        tokens = torch.cat((queries, chars), dim=1)
        prefix_mask = torch.ones(batch, len(self.query_names), dtype=torch.bool, device=source_mask.device)
        valid = torch.cat((prefix_mask, source_mask), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        encoded = self.encoder_norm(encoded)
        query_states = encoded[:, : len(self.query_names)]
        memory = encoded[:, len(self.query_names) :]
        return query_states, memory

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
    ) -> torch.Tensor:
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must be rank-2")
        if decoder_input_ids.shape[1] > self.cfg.max_target_chars + 1:
            raise ValueError("decoder input exceeds configured Heart target character limit")
        batch, target_length = decoder_input_ids.shape
        if batch != source_indices.shape[0]:
            raise ValueError("decoder batch must match source batch")
        positions = torch.arange(target_length, device=decoder_input_ids.device).unsqueeze(0).expand(batch, target_length)
        destination_dialect = self.destination_dialect_embedding(destination_dialect_ids)
        decoder_inputs = (
            self.decoder_embedding(decoder_input_ids)
            + self.decoder_position_embedding(positions)
            + destination_dialect.unsqueeze(1)
        )
        global_state = query_states[:, self.query_to_index["global"]]
        hidden0 = torch.tanh(self.decoder_init(torch.cat((global_state, destination_dialect), dim=-1))).unsqueeze(0)
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
        return probs.clamp_min(torch.finfo(probs.dtype).tiny).log()

    def forward(
        self,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        source_dialect_ids: torch.Tensor,
        destination_dialect_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> HeartTranslationOutput:
        query_states, memory = self._encode(
            source_indices,
            source_mask,
            source_dialect_ids,
            destination_dialect_ids,
        )
        semantic_logits = {
            name: self.semantic_heads[name](query_states[:, self.query_to_index[name]])
            for name in HEART_SEMANTIC_LABELS
        }
        referent = query_states[:, self.query_to_index["referent_identity"]]
        grounding = query_states[:, self.query_to_index["grounding_provenance"]]
        return HeartTranslationOutput(
            target_log_probs=self._decode(
                query_states,
                memory,
                source_indices,
                source_mask,
                destination_dialect_ids,
                decoder_input_ids,
            ),
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
        )

    @torch.no_grad()
    def greedy_translate(
        self,
        source_indices: torch.Tensor,
        source_mask: torch.Tensor,
        source_dialect_ids: torch.Tensor,
        destination_dialect_ids: torch.Tensor,
        *,
        max_chars: int | None = None,
    ) -> tuple[str, ...]:
        limit = self.cfg.max_target_chars if max_chars is None else min(int(max_chars), self.cfg.max_target_chars)
        if limit < 1:
            raise ValueError("max_chars must be positive")
        query_states, memory = self._encode(
            source_indices,
            source_mask,
            source_dialect_ids,
            destination_dialect_ids,
        )
        batch = source_indices.shape[0]
        generated = torch.full((batch, 1), self.bos_index, dtype=torch.long, device=source_indices.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=source_indices.device)
        output_indices: list[list[int]] = [[] for _ in range(batch)]
        for _ in range(limit):
            log_probs = self._decode(
                query_states,
                memory,
                source_indices,
                source_mask,
                destination_dialect_ids,
                generated,
            )
            next_ids = log_probs[:, -1].argmax(dim=-1)
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
        return tuple("".join(self.characters[index] for index in row) for row in output_indices)


__all__ = [
    "HEART_TRANSLATION_ARCHITECTURE",
    "HEART_SEMANTIC_LABELS",
    "HeartTranslationCoreConfig",
    "HeartTranslationOutput",
    "HeartTranslationCore",
]
