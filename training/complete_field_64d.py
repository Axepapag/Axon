"""Complete-field 64D reader and two-tick scratch/response writer.

This module is deliberately independent from the archived fixed-window
trainers.  Exact text remains in the canonical field.  A physical page is
only a bounded processing unit; every active character in every canonical
region is visited before a decoder may run.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from substrate import assert_supported_text, get_letter_bank


REGION_ORDER: tuple[str, ...] = (
    "conversation_history",
    "user_input",
    "structured_knowledge",
    "situation_awareness",
    "tool_results",
    "advisor_input",
    "task_state",
    "scratch",
    "response_draft",
    "diary",
)
REGION_TO_ID = {name: index for index, name in enumerate(REGION_ORDER)}
WRITABLE_REGIONS = frozenset({"scratch", "response_draft"})


@dataclass(frozen=True)
class ReaderConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 1
    ffn_dim: int = 192
    state_tokens: int = 4
    page_size: int = 256
    max_pages: int = 4096
    dropout: float = 0.05
    max_output_chars: int = 512
    lift_seed: int = 7

    def __post_init__(self) -> None:
        if self.d_model != 64:
            raise ValueError("R0 is intentionally locked to one 64D core")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.page_size < 1 or self.state_tokens < 1:
            raise ValueError("page_size and state_tokens must be positive")
        if self.max_output_chars < 65:
            raise ValueError("R0 output must not reproduce the obsolete 64-character cap")


@dataclass(frozen=True)
class FieldPage:
    region: str
    region_id: int
    region_page_index: int
    logical_page_index: int
    region_start: int
    region_end: int
    global_start: int
    global_end: int
    text: str
    sha256: str


@dataclass(frozen=True)
class CoverageManifest:
    page_size: int
    expected_characters: int
    observed_characters: int
    expected_regions: tuple[str, ...]
    visited_regions: tuple[str, ...]
    page_count: int
    zero_gaps: bool
    zero_duplicates: bool
    complete: bool
    field_sha256: str
    page_sha256: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_field(field: Mapping[str, str]) -> dict[str, str]:
    unknown = sorted(set(field) - set(REGION_ORDER))
    if unknown:
        raise ValueError(f"unknown field regions: {unknown}")
    result: dict[str, str] = {}
    for region in REGION_ORDER:
        text = field.get(region, "")
        if not isinstance(text, str):
            raise TypeError(f"region {region} must be text")
        assert_supported_text(text)
        result[region] = text
    return result


class CompleteFieldPager:
    """Create auditable, region-preserving pages without silent truncation."""

    def __init__(self, page_size: int):
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self.page_size = int(page_size)

    def paginate(self, field: Mapping[str, str]) -> tuple[list[FieldPage], CoverageManifest]:
        exact = canonical_field(field)
        pages: list[FieldPage] = []
        global_offset = 0
        logical_page = 0
        intervals: dict[str, list[tuple[int, int]]] = {name: [] for name in REGION_ORDER}
        for region in REGION_ORDER:
            text = exact[region]
            # Empty regions still receive a zero-character marker page so the
            # learned reader attends the region identity on every sweep.
            starts = range(0, len(text), self.page_size) if text else (0,)
            for region_page, start in enumerate(starts):
                chunk = text[start : start + self.page_size]
                end = start + len(chunk)
                pages.append(
                    FieldPage(
                        region=region,
                        region_id=REGION_TO_ID[region],
                        region_page_index=region_page,
                        logical_page_index=logical_page,
                        region_start=start,
                        region_end=end,
                        global_start=global_offset + start,
                        global_end=global_offset + end,
                        text=chunk,
                        sha256=_sha(chunk),
                    )
                )
                logical_page += 1
                if chunk:
                    intervals[region].append((start, end))
            global_offset += len(text)

        gaps = False
        duplicates = False
        observed = 0
        for region in REGION_ORDER:
            cursor = 0
            for start, end in intervals[region]:
                if start > cursor:
                    gaps = True
                if start < cursor:
                    duplicates = True
                cursor = max(cursor, end)
                observed += max(0, end - start)
            if cursor != len(exact[region]):
                gaps = True
        expected = sum(len(exact[name]) for name in REGION_ORDER)
        field_payload = "".join(f"{name}:{len(exact[name])}:{exact[name]}" for name in REGION_ORDER)
        manifest = CoverageManifest(
            page_size=self.page_size,
            expected_characters=expected,
            observed_characters=observed,
            expected_regions=REGION_ORDER,
            visited_regions=tuple(dict.fromkeys(page.region for page in pages)),
            page_count=len(pages),
            zero_gaps=not gaps,
            zero_duplicates=not duplicates,
            complete=(
                not gaps
                and not duplicates
                and observed == expected
                and tuple(dict.fromkeys(page.region for page in pages)) == REGION_ORDER
            ),
            field_sha256=_sha(field_payload),
            page_sha256=tuple(page.sha256 for page in pages),
        )
        return pages, manifest


def frozen_orthogonal_lift(d_model: int = 64, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + d_model)
    matrix = torch.randn(d_model, 16, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.T.contiguous().to(torch.float32)


class CompleteField64D(nn.Module):
    """One shallow core that gains logical depth through complete-field ticks."""

    def __init__(self, cfg: ReaderConfig = ReaderConfig()):
        super().__init__()
        self.cfg = cfg
        bank = get_letter_bank()
        self.characters = tuple(bank.chars[:-1])
        self.char_to_index = {char: index for index, char in enumerate(self.characters)}
        self.vocab_size = len(self.characters)
        self.eos_index = self.vocab_size
        self.bos_index = self.vocab_size + 1
        self.register_buffer("bank16", torch.from_numpy(bank.vecs_unit[:-1].copy()).float())
        self.register_buffer("char_lift", frozen_orthogonal_lift(cfg.d_model, cfg.lift_seed))

        self.region_embedding = nn.Embedding(len(REGION_ORDER), cfg.d_model)
        self.local_position = nn.Embedding(cfg.page_size + 1, cfg.d_model)
        self.page_position = nn.Embedding(cfg.max_pages, cfg.d_model)
        self.global_position = nn.Linear(2, cfg.d_model, bias=False)
        self.empty_region_marker = nn.Parameter(torch.zeros(cfg.d_model))
        self.initial_state = nn.Parameter(torch.randn(cfg.state_tokens, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.page_encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.state_norm = nn.LayerNorm(cfg.d_model)

        self.decoder_embedding = nn.Embedding(self.vocab_size + 2, cfg.d_model)
        self.decoder_head_embedding = nn.Embedding(2, cfg.d_model)
        self.decoder_init = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.decoder = nn.GRU(cfg.d_model, cfg.d_model, batch_first=True)
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.decoder_output = nn.Linear(cfg.d_model, self.vocab_size + 1)

        nn.init.normal_(self.region_embedding.weight, std=0.02)
        nn.init.normal_(self.local_position.weight, std=0.02)
        nn.init.normal_(self.page_position.weight, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _page_tensor(self, page: FieldPage) -> torch.Tensor:
        if page.text:
            indices = torch.tensor(
                [self.char_to_index[char] for char in page.text],
                dtype=torch.long,
                device=self.device,
            )
            chars = self.bank16.index_select(0, indices) @ self.char_lift
        else:
            chars = self.empty_region_marker.unsqueeze(0)
        length = chars.shape[0]
        local = torch.arange(length, device=self.device)
        region = self.region_embedding(
            torch.full((length,), page.region_id, dtype=torch.long, device=self.device)
        )
        page_pos = self.page_position(
            torch.full(
                (length,),
                page.logical_page_index % self.cfg.max_pages,
                dtype=torch.long,
                device=self.device,
            )
        )
        denom = max(1.0, float(page.global_end + 1))
        start = math.log1p(page.global_start) / math.log1p(denom)
        end = math.log1p(page.global_end) / math.log1p(denom)
        global_features = torch.tensor([start, end], device=self.device).expand(length, 2)
        return chars + region + self.local_position(local) + page_pos + self.global_position(global_features)

    def read_field(self, field: Mapping[str, str]) -> tuple[torch.Tensor, CoverageManifest]:
        pages, manifest = CompleteFieldPager(self.cfg.page_size).paginate(field)
        if not manifest.complete:
            raise RuntimeError("coverage manifest is incomplete; decoder finalization denied")
        state = self.initial_state.unsqueeze(0)
        for page in pages:
            tokens = self._page_tensor(page).unsqueeze(0)
            encoded = self.page_encoder(torch.cat((state, tokens), dim=1))
            state = encoded[:, : self.cfg.state_tokens]
        return self.state_norm(state), manifest

    def _target_indices(self, text: str) -> torch.Tensor:
        assert_supported_text(text)
        if len(text) > self.cfg.max_output_chars:
            raise ValueError(
                f"target has {len(text)} characters; configured maximum is "
                f"{self.cfg.max_output_chars}; refusing silent truncation"
            )
        return torch.tensor(
            [self.char_to_index[char] for char in text] + [self.eos_index],
            dtype=torch.long,
            device=self.device,
        )

    def decode_teacher(self, reader_state: torch.Tensor, target: str, head: int) -> tuple[torch.Tensor, torch.Tensor]:
        targets = self._target_indices(target).unsqueeze(0)
        bos = torch.full((1, 1), self.bos_index, dtype=torch.long, device=self.device)
        decoder_input = torch.cat((bos, targets[:, :-1]), dim=1)
        summary = reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([head], device=self.device))
        hidden = torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
        output, _ = self.decoder(self.decoder_embedding(decoder_input), hidden)
        return self.decoder_output(self.decoder_norm(output)), targets

    @torch.no_grad()
    def decode_greedy(self, reader_state: torch.Tensor, head: int, max_chars: int | None = None) -> tuple[str, bool]:
        limit = self.cfg.max_output_chars if max_chars is None else min(max_chars, self.cfg.max_output_chars)
        summary = reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([head], device=self.device))
        hidden = torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
        token = torch.full((1, 1), self.bos_index, dtype=torch.long, device=self.device)
        chars: list[str] = []
        for _ in range(limit + 1):
            output, hidden = self.decoder(self.decoder_embedding(token), hidden)
            logits = self.decoder_output(self.decoder_norm(output[:, -1]))
            index = int(logits.argmax(dim=-1).item())
            if index == self.eos_index:
                return "".join(chars), True
            if index >= self.vocab_size:
                return "".join(chars), False
            chars.append(self.characters[index])
            token = torch.tensor([[index]], dtype=torch.long, device=self.device)
            if len(chars) >= limit:
                break
        return "".join(chars), False

    def forward_transaction(
        self,
        field: Mapping[str, str],
        scratch_target: str,
        response_target: str,
    ) -> dict[str, Any]:
        exact = canonical_field(field)
        state1, coverage1 = self.read_field(exact)
        scratch_logits, scratch_indices = self.decode_teacher(state1, scratch_target, head=0)
        second_field = dict(exact)
        second_field["scratch"] = scratch_target
        state2, coverage2 = self.read_field(second_field)
        response_logits, response_indices = self.decode_teacher(state2, response_target, head=1)
        return {
            "scratch_logits": scratch_logits,
            "scratch_targets": scratch_indices,
            "response_logits": response_logits,
            "response_targets": response_indices,
            "coverage_tick1": coverage1,
            "coverage_tick2": coverage2,
            "response_state": state2,
        }

    @torch.no_grad()
    def run_transaction(self, field: Mapping[str, str]) -> dict[str, Any]:
        exact = canonical_field(field)
        state1, coverage1 = self.read_field(exact)
        scratch, scratch_terminated = self.decode_greedy(state1, head=0)
        second_field = dict(exact)
        second_field["scratch"] = scratch
        state2, coverage2 = self.read_field(second_field)
        response, response_terminated = self.decode_greedy(state2, head=1)
        if not coverage1.complete or not coverage2.complete:
            raise RuntimeError("complete coverage required before field delta")
        return {
            "scratch": scratch,
            "response_draft": response,
            "scratch_terminated": scratch_terminated,
            "response_terminated": response_terminated,
            "coverage_tick1": coverage1.to_dict(),
            "coverage_tick2": coverage2.to_dict(),
            "typed_delta": self.typed_delta(exact, scratch, response),
        }

    @staticmethod
    def typed_delta(base_field: Mapping[str, str], scratch: str, response: str) -> dict[str, Any]:
        # There is intentionally no diary operation in R0.
        operations = []
        for region, text in (("scratch", scratch), ("response_draft", response)):
            if region not in WRITABLE_REGIONS:
                raise RuntimeError(f"R0 attempted sealed write to {region}")
            old = base_field.get(region, "")
            operations.append(
                {"op": "replace", "region": region, "start": 0, "end": len(old), "text": text}
            )
        return {"schema": "axon-r0-scratch-response-delta-v1", "operations": operations}


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eos_weight: float = 4.0,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    eos_index = logits.shape[-1] - 1
    weights = torch.ones_like(per_token)
    weights = torch.where(targets == eos_index, weights * eos_weight, weights)
    return (per_token * weights).sum() / weights.sum()


def teacher_char_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    prediction = logits.argmax(dim=-1)
    return float((prediction == targets).float().mean().item())


__all__ = [
    "REGION_ORDER",
    "WRITABLE_REGIONS",
    "ReaderConfig",
    "FieldPage",
    "CoverageManifest",
    "CompleteFieldPager",
    "CompleteField64D",
    "canonical_field",
    "frozen_orthogonal_lift",
    "sequence_cross_entropy",
    "teacher_char_accuracy",
]
