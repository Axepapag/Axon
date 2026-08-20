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


@dataclass(frozen=True)
class AddressableMemory:
    """Encoded field tokens paired with their immutable source identities."""

    states: torch.Tensor
    char_indices: torch.Tensor
    region_ids: torch.Tensor
    region_positions: torch.Tensor


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
        self.memory_norm = nn.LayerNorm(cfg.d_model)

        self.decoder_embedding = nn.Embedding(self.vocab_size + 2, cfg.d_model)
        self.decoder_head_embedding = nn.Embedding(2, cfg.d_model)
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
        # V6 separates semantic cross-attention from the exact copy pointer.
        # The pointer is one explicit head over source positions so training can
        # supervise the exact occurrence rather than rewarding every matching
        # character in the field.
        self.position_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.position_key = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.copy_gate = nn.Linear(cfg.d_model * 2, 1)

        nn.init.normal_(self.region_embedding.weight, std=0.02)
        nn.init.normal_(self.local_position.weight, std=0.02)
        nn.init.normal_(self.page_position.weight, std=0.02)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, 1.5)

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

    def read_field_with_memory(
        self,
        field: Mapping[str, str],
    ) -> tuple[torch.Tensor, AddressableMemory, CoverageManifest]:
        """Sweep every page and retain encoded tokens plus exact source identities."""
        pages, manifest = CompleteFieldPager(self.cfg.page_size).paginate(field)
        if not manifest.complete:
            raise RuntimeError("coverage manifest is incomplete; decoder finalization denied")
        state = self.initial_state.unsqueeze(0)
        memory_states: list[torch.Tensor] = []
        memory_char_indices: list[torch.Tensor] = []
        memory_region_ids: list[torch.Tensor] = []
        memory_region_positions: list[torch.Tensor] = []
        for page in pages:
            tokens = self._page_tensor(page).unsqueeze(0)
            encoded = self.page_encoder(torch.cat((state, tokens), dim=1))
            state = encoded[:, : self.cfg.state_tokens]
            page_memory = encoded[:, self.cfg.state_tokens :]
            memory_states.append(page_memory)
            if page.text:
                page_chars = torch.tensor(
                    [self.char_to_index[char] for char in page.text],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                # Empty-region markers remain addressable context but are never
                # eligible copy sources.
                page_chars = torch.full((1,), -1, dtype=torch.long, device=self.device)
            memory_char_indices.append(page_chars.unsqueeze(0))
            memory_region_ids.append(
                torch.full(
                    (1, page_memory.shape[1]),
                    page.region_id,
                    dtype=torch.long,
                    device=self.device,
                )
            )
            if page.text:
                page_positions = torch.arange(
                    page.region_start,
                    page.region_end,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                page_positions = torch.full((1,), -1, dtype=torch.long, device=self.device)
            memory_region_positions.append(page_positions.unsqueeze(0))
        addressable_memory = AddressableMemory(
            states=self.memory_norm(torch.cat(memory_states, dim=1)),
            char_indices=torch.cat(memory_char_indices, dim=1),
            region_ids=torch.cat(memory_region_ids, dim=1),
            region_positions=torch.cat(memory_region_positions, dim=1),
        )
        return self.state_norm(state), addressable_memory, manifest

    def read_field(self, field: Mapping[str, str]) -> tuple[torch.Tensor, CoverageManifest]:
        state, _, manifest = self.read_field_with_memory(field)
        return state, manifest

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

    def _decoder_logits(
        self,
        output: torch.Tensor,
        memory: AddressableMemory | None,
        *,
        return_alignment: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Mix generation with a V6 exact-position copy distribution.

        Semantic context still comes from ordinary multi-head cross-attention,
        but copy probabilities come from a separate one-head position pointer.
        This makes the exact source occurrence observable and directly
        supervisable instead of crediting every matching character.
        """
        if memory is None:
            logits = self.decoder_output(self.decoder_norm(output))
            if return_alignment:
                empty = torch.empty(
                    output.shape[0], output.shape[1], 0, device=output.device, dtype=output.dtype
                )
                return logits, {
                    "position_logits": empty,
                    "generate_gate_logits": torch.full(
                        (output.shape[0], output.shape[1]),
                        30.0,
                        device=output.device,
                        dtype=output.dtype,
                    ),
                }
            return logits

        context, _ = self.decoder_memory_attention(
            output,
            memory.states,
            memory.states,
            need_weights=False,
        )
        fused = self.decoder_norm(output + context)
        generated = F.softmax(self.decoder_output(fused), dim=-1)

        query = self.position_query(fused)
        key = self.position_key(memory.states)
        position_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.cfg.d_model)
        valid_sources = memory.char_indices.ge(0).unsqueeze(1)
        masked_position_logits = position_logits.masked_fill(
            ~valid_sources,
            torch.finfo(position_logits.dtype).min,
        )
        pointer_attention = F.softmax(masked_position_logits, dim=-1)
        pointer_attention = pointer_attention * valid_sources.to(pointer_attention.dtype)
        copy_mass = pointer_attention.sum(dim=-1, keepdim=True)
        pointer_attention = pointer_attention / copy_mass.clamp_min(
            torch.finfo(pointer_attention.dtype).tiny
        )
        safe_indices = memory.char_indices.clamp_min(0).unsqueeze(1).expand(
            -1,
            output.shape[1],
            -1,
        )
        copied = torch.zeros_like(generated).scatter_add(
            dim=-1,
            index=safe_indices,
            src=pointer_attention,
        )

        generate_gate_logits = self.copy_gate(torch.cat((output, context), dim=-1)).squeeze(-1)
        generate_gate = torch.sigmoid(generate_gate_logits).unsqueeze(-1)
        has_copy_source = copy_mass.gt(0).to(generate_gate.dtype)
        generate_gate = generate_gate * has_copy_source + (1.0 - has_copy_source)
        probabilities = generate_gate * generated + (1.0 - generate_gate) * copied
        log_probabilities = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
        if return_alignment:
            return log_probabilities, {
                "position_logits": masked_position_logits,
                "generate_gate_logits": generate_gate_logits,
            }
        return log_probabilities

    def decode_teacher(
        self,
        reader_state: torch.Tensor,
        target: str,
        head: int,
        memory: AddressableMemory | None = None,
        *,
        return_alignment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        targets = self._target_indices(target).unsqueeze(0)
        bos = torch.full((1, 1), self.bos_index, dtype=torch.long, device=self.device)
        decoder_input = torch.cat((bos, targets[:, :-1]), dim=1)
        summary = reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([head], device=self.device))
        hidden = torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
        output, _ = self.decoder(self.decoder_embedding(decoder_input), hidden)
        decoded = self._decoder_logits(output, memory, return_alignment=return_alignment)
        if return_alignment:
            logits, alignment = decoded
            return logits, targets, alignment
        return decoded, targets

    def decode_scheduled(
        self,
        reader_state: torch.Tensor,
        target: str,
        head: int,
        teacher_forcing_ratio: float,
        memory: AddressableMemory | None = None,
        *,
        return_alignment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        """Train on model-generated prefixes so greedy behavior cannot hide behind teacher forcing."""
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between zero and one")
        if teacher_forcing_ratio >= 1.0:
            return self.decode_teacher(
                reader_state,
                target,
                head,
                memory=memory,
                return_alignment=return_alignment,
            )
        targets = self._target_indices(target).unsqueeze(0)
        summary = reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([head], device=self.device))
        hidden = torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
        token = torch.full((1, 1), self.bos_index, dtype=torch.long, device=self.device)
        teacher_choices = (
            torch.rand(max(0, targets.shape[1] - 1), device=self.device)
            < teacher_forcing_ratio
        ).tolist()
        logits: list[torch.Tensor] = []
        pointer_logits: list[torch.Tensor] = []
        gate_logits: list[torch.Tensor] = []
        for position in range(targets.shape[1]):
            output, hidden = self.decoder(self.decoder_embedding(token), hidden)
            decoded = self._decoder_logits(
                output[:, -1:],
                memory,
                return_alignment=return_alignment,
            )
            if return_alignment:
                step_logits, step_alignment = decoded
                pointer_logits.append(step_alignment["position_logits"])
                gate_logits.append(step_alignment["generate_gate_logits"])
            else:
                step_logits = decoded
            logits.append(step_logits)
            if position + 1 >= targets.shape[1]:
                continue
            use_teacher = bool(teacher_choices[position])
            token = (
                targets[:, position : position + 1]
                if use_teacher
                else step_logits.argmax(dim=-1).detach()
            )
        joined_logits = torch.cat(logits, dim=1)
        if return_alignment:
            return joined_logits, targets, {
                "position_logits": torch.cat(pointer_logits, dim=1),
                "generate_gate_logits": torch.cat(gate_logits, dim=1),
            }
        return joined_logits, targets

    def alignment_supervision(
        self,
        *,
        target_text: str,
        memory: AddressableMemory,
        decoder_alignment: Mapping[str, torch.Tensor],
        specification: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Score exact source-position and copy/generate labels for one decoder head."""
        if specification.get("schema") != "axon-r0-target-alignment-v1":
            raise ValueError("unsupported R0 target-alignment schema")
        segments = specification.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError("alignment segments must be a list")
        position_logits = decoder_alignment["position_logits"]
        gate_logits = decoder_alignment["generate_gate_logits"]
        if position_logits.ndim != 3 or gate_logits.ndim != 2:
            raise ValueError("decoder alignment tensors have invalid rank")
        if position_logits.shape[:2] != gate_logits.shape:
            raise ValueError("decoder alignment tensor lengths disagree")
        if gate_logits.shape[1] != len(target_text) + 1:
            raise ValueError("decoder alignment length does not match target plus EOS")

        zero = gate_logits.sum() * 0.0
        position_losses: list[torch.Tensor] = []
        gate_losses: list[torch.Tensor] = []
        position_correct = 0
        gate_correct = 0
        supervised_copy_positions = 0

        for segment in segments:
            required = {
                "target_start",
                "target_end",
                "source_region",
                "source_start",
                "source_end",
                "text_sha256",
                "authority",
            }
            if not isinstance(segment, Mapping) or set(segment) != required:
                raise ValueError("alignment segment fields are invalid")
            target_start = int(segment["target_start"])
            target_end = int(segment["target_end"])
            source_start = int(segment["source_start"])
            source_end = int(segment["source_end"])
            source_region = str(segment["source_region"])
            if source_region not in REGION_TO_ID:
                raise ValueError(f"unknown alignment source region {source_region}")
            if (
                target_start < 0
                or target_end > len(target_text)
                or target_end <= target_start
                or source_start < 0
                or source_end <= source_start
                or target_end - target_start != source_end - source_start
            ):
                raise ValueError("alignment segment bounds are invalid or unequal")
            target_fragment = target_text[target_start:target_end]
            if _sha(target_fragment) != segment["text_sha256"]:
                raise ValueError("alignment target fragment hash mismatch")

            source_chars: list[str] = []
            source_indices: list[int] = []
            region_id = REGION_TO_ID[source_region]
            for offset, source_position in enumerate(range(source_start, source_end)):
                matches = (
                    (memory.region_ids[0] == region_id)
                    & (memory.region_positions[0] == source_position)
                    & memory.char_indices[0].ge(0)
                ).nonzero(as_tuple=False).flatten()
                if matches.numel() != 1:
                    raise ValueError(
                        f"alignment source {source_region}[{source_position}] does not resolve uniquely"
                    )
                memory_index = int(matches.item())
                char_index = int(memory.char_indices[0, memory_index].item())
                source_chars.append(self.characters[char_index])
                source_indices.append(memory_index)

                target_position = target_start + offset
                expected_source = torch.tensor(
                    [memory_index], dtype=torch.long, device=self.device
                )
                position_loss = F.cross_entropy(
                    position_logits[:, target_position, :], expected_source
                )
                position_losses.append(position_loss)
                predicted_source = int(
                    position_logits[0, target_position].argmax(dim=-1).item()
                )
                position_correct += int(predicted_source == memory_index)

                copy_target = torch.zeros((1,), device=self.device, dtype=gate_logits.dtype)
                copy_gate_loss = F.binary_cross_entropy_with_logits(
                    gate_logits[:, target_position], copy_target
                )
                gate_losses.append(copy_gate_loss)
                gate_correct += int(float(gate_logits[0, target_position].item()) < 0.0)
                supervised_copy_positions += 1

            source_fragment = "".join(source_chars)
            if source_fragment != target_fragment or _sha(source_fragment) != segment["text_sha256"]:
                raise ValueError("alignment source text does not equal supervised target text")

        eos_supervised = bool(specification.get("supervise_eos_generate", True))
        if eos_supervised:
            eos_target = torch.ones((1,), device=self.device, dtype=gate_logits.dtype)
            eos_position = len(target_text)
            gate_losses.append(
                F.binary_cross_entropy_with_logits(gate_logits[:, eos_position], eos_target)
            )
            gate_correct += int(float(gate_logits[0, eos_position].item()) >= 0.0)

        position_loss = torch.stack(position_losses).mean() if position_losses else zero
        gate_loss = torch.stack(gate_losses).mean() if gate_losses else zero
        gate_count = supervised_copy_positions + int(eos_supervised)
        return {
            "position_loss": position_loss,
            "gate_loss": gate_loss,
            "copy_positions": supervised_copy_positions,
            "position_correct": position_correct,
            "gate_supervised_positions": gate_count,
            "gate_correct": gate_correct,
            "position_accuracy": (
                position_correct / supervised_copy_positions
                if supervised_copy_positions
                else 1.0
            ),
            "gate_accuracy": gate_correct / gate_count if gate_count else 1.0,
        }

    @torch.no_grad()
    def decode_greedy(
        self,
        reader_state: torch.Tensor,
        head: int,
        max_chars: int | None = None,
        memory: AddressableMemory | None = None,
    ) -> tuple[str, bool]:
        limit = self.cfg.max_output_chars if max_chars is None else min(max_chars, self.cfg.max_output_chars)
        summary = reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([head], device=self.device))
        hidden = torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
        token = torch.full((1, 1), self.bos_index, dtype=torch.long, device=self.device)
        chars: list[str] = []
        for _ in range(limit + 1):
            output, hidden = self.decoder(self.decoder_embedding(token), hidden)
            logits = self._decoder_logits(output[:, -1:], memory).squeeze(1)
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
        teacher_forcing_ratio: float = 1.0,
        alignment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        exact = canonical_field(field)
        if alignment is not None:
            if alignment.get("schema") != "axon-r0-source-alignment-v1":
                raise ValueError("unsupported R0 source-alignment schema")
            if set(alignment) != {"schema", "scratch", "response_draft"}:
                raise ValueError("R0 source alignment must define scratch and response_draft")

        state1, memory1, coverage1 = self.read_field_with_memory(exact)
        scratch_decoded = self.decode_scheduled(
            state1,
            scratch_target,
            head=0,
            teacher_forcing_ratio=teacher_forcing_ratio,
            memory=memory1,
            return_alignment=alignment is not None,
        )
        scratch_supervision: dict[str, Any] | None = None
        if alignment is not None:
            scratch_logits, scratch_indices, scratch_decoder_alignment = scratch_decoded
            scratch_supervision = self.alignment_supervision(
                target_text=scratch_target,
                memory=memory1,
                decoder_alignment=scratch_decoder_alignment,
                specification=alignment["scratch"],
            )
        else:
            scratch_logits, scratch_indices = scratch_decoded

        second_field = dict(exact)
        second_field["scratch"] = scratch_target
        state2, memory2, coverage2 = self.read_field_with_memory(second_field)
        response_decoded = self.decode_scheduled(
            state2,
            response_target,
            head=1,
            teacher_forcing_ratio=teacher_forcing_ratio,
            memory=memory2,
            return_alignment=alignment is not None,
        )
        response_supervision: dict[str, Any] | None = None
        if alignment is not None:
            response_logits, response_indices, response_decoder_alignment = response_decoded
            response_supervision = self.alignment_supervision(
                target_text=response_target,
                memory=memory2,
                decoder_alignment=response_decoder_alignment,
                specification=alignment["response_draft"],
            )
        else:
            response_logits, response_indices = response_decoded

        result: dict[str, Any] = {
            "scratch_logits": scratch_logits,
            "scratch_targets": scratch_indices,
            "response_logits": response_logits,
            "response_targets": response_indices,
            "coverage_tick1": coverage1,
            "coverage_tick2": coverage2,
            "response_state": state2,
        }
        if scratch_supervision is not None and response_supervision is not None:
            result["alignment"] = {
                "scratch": scratch_supervision,
                "response_draft": response_supervision,
                "position_loss": (
                    scratch_supervision["position_loss"]
                    + response_supervision["position_loss"]
                ),
                "gate_loss": (
                    scratch_supervision["gate_loss"]
                    + response_supervision["gate_loss"]
                ),
                "copy_positions": (
                    scratch_supervision["copy_positions"]
                    + response_supervision["copy_positions"]
                ),
                "position_correct": (
                    scratch_supervision["position_correct"]
                    + response_supervision["position_correct"]
                ),
                "gate_supervised_positions": (
                    scratch_supervision["gate_supervised_positions"]
                    + response_supervision["gate_supervised_positions"]
                ),
                "gate_correct": (
                    scratch_supervision["gate_correct"]
                    + response_supervision["gate_correct"]
                ),
            }
        return result

    @torch.no_grad()
    def run_transaction(self, field: Mapping[str, str]) -> dict[str, Any]:
        exact = canonical_field(field)
        state1, memory1, coverage1 = self.read_field_with_memory(exact)
        scratch, scratch_terminated = self.decode_greedy(
            state1,
            head=0,
            memory=memory1,
        )
        second_field = dict(exact)
        second_field["scratch"] = scratch
        state2, memory2, coverage2 = self.read_field_with_memory(second_field)
        response, response_terminated = self.decode_greedy(
            state2,
            head=1,
            memory=memory2,
        )
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
