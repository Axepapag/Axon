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
from runtime.field import (
    CANONICAL_REGION_ORDER,
    CompiledD64Field,
    D64CharacterPage,
    D64FieldCompiler,
    FieldDelta,
    LogicalRegion,
    SharedFieldSnapshot,
    apply_compiled_delta,
    replacement_delta,
)


REGION_ORDER: tuple[str, ...] = (
    "conversation_history",
    "user_input",
    "cortex",
    "situation_awareness",
    "tool_results",
    "advisor_input",
    "task_state",
    "scratch",
    "response_draft",
    "diary",
)
REGION_TO_ID = {name: index for index, name in enumerate(REGION_ORDER)}


@dataclass(frozen=True)
class ReaderConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 1
    ffn_dim: int = 192
    state_tokens: int = 4
    page_size: int = 256
    dropout: float = 0.05
    inference_budget_chars: int = 512
    lift_seed: int = 7

    def __post_init__(self) -> None:
        if self.d_model != 64:
            raise ValueError("R0 is intentionally locked to one 64D core")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.page_size < 1 or self.state_tokens < 1:
            raise ValueError("page_size and state_tokens must be positive")
        if self.inference_budget_chars < 1:
            raise ValueError("inference_budget_chars must be positive")




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


def coverage_manifest_from_compiled(
    compiled: CompiledD64Field,
    page_size: int,
) -> CoverageManifest:
    """Derive the trainer's observable coverage record from the canonical D64 rail."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if not isinstance(compiled, CompiledD64Field):
        raise TypeError("compiled must be CompiledD64Field")
    if not compiled.coverage.complete:
        raise RuntimeError("incomplete canonical D64 rail cannot produce coverage")

    pages = list(compiled.iter_character_pages(page_size))
    active = compiled.active_texts()
    expected_regions = tuple(region.value for region in CANONICAL_REGION_ORDER)
    visited_regions = tuple(dict.fromkeys(page.region.value for page in pages))
    observed = 0
    zero_gaps = True
    zero_duplicates = True
    cursor_by_region = {region: 0 for region in expected_regions}
    for page in pages:
        region = page.region.value
        cursor = cursor_by_region[region]
        if page.region_start > cursor:
            zero_gaps = False
        if page.region_start < cursor:
            zero_duplicates = False
        if page.region_end < page.region_start:
            zero_gaps = False
        if page.text != active[region][page.region_start:page.region_end]:
            zero_gaps = False
        cursor_by_region[region] = max(cursor, page.region_end)
        observed += len(page.text)
    for region in expected_regions:
        if cursor_by_region[region] != len(active[region]):
            zero_gaps = False

    expected = compiled.coverage.expected_active_characters
    complete = (
        compiled.coverage.complete
        and zero_gaps
        and zero_duplicates
        and observed == expected
        and visited_regions == expected_regions
    )
    return CoverageManifest(
        page_size=page_size,
        expected_characters=expected,
        observed_characters=observed,
        expected_regions=expected_regions,
        visited_regions=visited_regions,
        page_count=len(pages),
        zero_gaps=zero_gaps,
        zero_duplicates=zero_duplicates,
        complete=complete,
        field_sha256=compiled.source_field_id,
        page_sha256=tuple(_sha(page.text) for page in pages),
    )


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




def frozen_orthogonal_lift(d_model: int = 64, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + d_model)
    matrix = torch.randn(d_model, 16, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    return q.T.contiguous().to(torch.float32)


def sinusoidal_positions(
    positions: torch.Tensor,
    d_model: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic positions with no learned page or sequence ceiling."""

    if positions.ndim != 1:
        raise ValueError("positions must be rank-1")
    work = positions.to(dtype=torch.float32).unsqueeze(-1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    angles = work * frequencies
    encoded = torch.zeros(positions.shape[0], d_model, device=positions.device, dtype=torch.float32)
    encoded[:, 0::2] = torch.sin(angles)
    encoded[:, 1::2] = torch.cos(angles[:, : encoded[:, 1::2].shape[-1]])
    return encoded.to(dtype=dtype)


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
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, 1.5)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


    def _compiled_page_tensor(self, page: D64CharacterPage) -> torch.Tensor:
        """Lift exact 16D cells unpacked from the canonical D64 rail.

        The rail stores the raw frozen substrate cells.  V6 historically used
        unit-normalized bank cells before its frozen orthogonal lift, so this
        adapter normalizes the exact rail lanes deterministically before the
        existing lift.  The compiler remains the character authority while
        current V6 weights see the same input scale they were trained on.
        """
        if page.text:
            raw = torch.from_numpy(np.array(page.cells16, copy=True)).to(
                device=self.device, dtype=self.bank16.dtype
            )
            norms = raw.norm(dim=-1, keepdim=True).clamp_min(
                torch.finfo(raw.dtype).tiny
            )
            chars = (raw / norms) @ self.char_lift
        else:
            chars = self.empty_region_marker.unsqueeze(0)
        length = chars.shape[0]
        local = torch.arange(length, device=self.device)
        region = self.region_embedding(
            torch.full((length,), page.region_id, dtype=torch.long, device=self.device)
        )
        local_pos = sinusoidal_positions(
            local,
            self.cfg.d_model,
            dtype=chars.dtype,
        )
        page_pos = sinusoidal_positions(
            torch.full(
                (length,),
                page.logical_page_index,
                dtype=torch.long,
                device=self.device,
            ),
            self.cfg.d_model,
            dtype=chars.dtype,
        )
        denom = max(1.0, float(page.global_end + 1))
        start = math.log1p(page.global_start) / math.log1p(denom)
        end = math.log1p(page.global_end) / math.log1p(denom)
        global_features = torch.tensor(
            [start, end], device=self.device
        ).expand(length, 2)
        return (
            chars
            + region
            + local_pos
            + page_pos
            + self.global_position(global_features)
        )

    def read_compiled_with_memory(
        self,
        compiled: CompiledD64Field,
    ) -> tuple[torch.Tensor, AddressableMemory, CoverageManifest]:
        """Read every exact character from a complete canonical D64 rail."""
        if not isinstance(compiled, CompiledD64Field):
            raise TypeError("compiled must be CompiledD64Field")
        if not compiled.coverage.complete:
            raise RuntimeError("incomplete canonical D64 rail cannot be read")
        pages = list(compiled.iter_character_pages(self.cfg.page_size))
        manifest = coverage_manifest_from_compiled(compiled, self.cfg.page_size)
        if not manifest.complete:
            raise RuntimeError("coverage manifest is incomplete; decoder finalization denied")

        state = self.initial_state.unsqueeze(0)
        memory_states: list[torch.Tensor] = []
        memory_char_indices: list[torch.Tensor] = []
        memory_region_ids: list[torch.Tensor] = []
        memory_region_positions: list[torch.Tensor] = []
        for page in pages:
            tokens = self._compiled_page_tensor(page).unsqueeze(0)
            encoded = self.page_encoder(torch.cat((state, tokens), dim=1))
            state = encoded[:, : self.cfg.state_tokens]
            page_memory = encoded[:, self.cfg.state_tokens :]
            memory_states.append(page_memory)
            if page.text:
                page_chars = torch.tensor(
                    [self.char_to_index[address.character] for address in page.addresses],
                    dtype=torch.long,
                    device=self.device,
                )
                page_positions = torch.tensor(
                    [address.region_position for address in page.addresses],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                page_chars = torch.full(
                    (1,), -1, dtype=torch.long, device=self.device
                )
                page_positions = torch.full(
                    (1,), -1, dtype=torch.long, device=self.device
                )
            memory_char_indices.append(page_chars.unsqueeze(0))
            memory_region_ids.append(
                torch.full(
                    (1, page_memory.shape[1]),
                    page.region_id,
                    dtype=torch.long,
                    device=self.device,
                )
            )
            memory_region_positions.append(page_positions.unsqueeze(0))

        addressable_memory = AddressableMemory(
            states=self.memory_norm(torch.cat(memory_states, dim=1)),
            char_indices=torch.cat(memory_char_indices, dim=1),
            region_ids=torch.cat(memory_region_ids, dim=1),
            region_positions=torch.cat(memory_region_positions, dim=1),
        )
        return self.state_norm(state), addressable_memory, manifest

    def read_snapshot_with_memory(
        self,
        snapshot: SharedFieldSnapshot,
        *,
        compiler: D64FieldCompiler | None = None,
    ) -> tuple[torch.Tensor, AddressableMemory, CoverageManifest, CompiledD64Field]:
        """Canonical runtime/training entry point for one complete D64 read."""
        active_compiler = D64FieldCompiler() if compiler is None else compiler
        compiled = active_compiler.compile(snapshot)
        compiled.verify_roundtrip(snapshot)
        state, memory, manifest = self.read_compiled_with_memory(compiled)
        return state, memory, manifest, compiled



    def _target_indices(self, text: str) -> torch.Tensor:
        assert_supported_text(text)
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
        limit = self.cfg.inference_budget_chars if max_chars is None else int(max_chars)
        if limit < 1:
            raise ValueError("max_chars must be positive")
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

    def forward_canonical_transaction(
        self,
        snapshot: SharedFieldSnapshot,
        scratch_target: str,
        response_target: str,
        teacher_forcing_ratio: float = 1.0,
        alignment: Mapping[str, Any] | None = None,
        *,
        compiler: D64FieldCompiler | None = None,
        author_core_id: str = "teacher",
    ) -> dict[str, Any]:
        """Train through canonical snapshot -> compiler -> typed delta anatomy."""
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        active_compiler = D64FieldCompiler() if compiler is None else compiler
        if alignment is not None:
            if alignment.get("schema") != "axon-r0-source-alignment-v1":
                raise ValueError("unsupported R0 source-alignment schema")
            if set(alignment) != {"schema", "scratch", "response_draft"}:
                raise ValueError(
                    "canonical R0 source alignment must define scratch and response_draft"
                )

        compiled1 = active_compiler.compile(snapshot)
        state1, memory1, coverage1 = self.read_compiled_with_memory(compiled1)
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

        if snapshot.region(LogicalRegion.SCRATCH).text == scratch_target:
            scratch_delta = None
            second_snapshot = snapshot
        else:
            scratch_delta = replacement_delta(
                snapshot,
                compiled1,
                region=LogicalRegion.SCRATCH,
                text=scratch_target,
                author_core_id=author_core_id,
                pass_id="scratch",
                provenance="canonical_d64_training_teacher_scratch",
            )
            second_snapshot = apply_compiled_delta(snapshot, compiled1, scratch_delta)
        compiled2 = active_compiler.compile(second_snapshot)
        state2, memory2, coverage2 = self.read_compiled_with_memory(compiled2)
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

        if second_snapshot.region(LogicalRegion.RESPONSE_DRAFT).text == response_target:
            response_delta = None
            final_snapshot = second_snapshot
        else:
            response_delta = replacement_delta(
                second_snapshot,
                compiled2,
                region=LogicalRegion.RESPONSE_DRAFT,
                text=response_target,
                author_core_id=author_core_id,
                pass_id="response_draft",
                provenance="canonical_d64_training_teacher_response",
            )
            final_snapshot = apply_compiled_delta(
                second_snapshot, compiled2, response_delta
            )

        result: dict[str, Any] = {
            "scratch_logits": scratch_logits,
            "scratch_targets": scratch_indices,
            "response_logits": response_logits,
            "response_targets": response_indices,
            "coverage_tick1": coverage1,
            "coverage_tick2": coverage2,
            "response_state": state2,
            "base_snapshot": snapshot,
            "scratch_snapshot": second_snapshot,
            "final_snapshot": final_snapshot,
            "compiled_tick1": compiled1,
            "compiled_tick2": compiled2,
            "scratch_delta": scratch_delta,
            "response_delta": response_delta,
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
    def run_canonical_transaction(
        self,
        snapshot: SharedFieldSnapshot,
        *,
        compiler: D64FieldCompiler | None = None,
        author_core_id: str = "core64d-r0",
    ) -> dict[str, Any]:
        """Greedy canonical inference path using the same compiler/delta anatomy."""
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        active_compiler = D64FieldCompiler() if compiler is None else compiler
        compiled1 = active_compiler.compile(snapshot)
        state1, memory1, coverage1 = self.read_compiled_with_memory(compiled1)
        scratch, scratch_terminated = self.decode_greedy(
            state1, head=0, memory=memory1
        )
        if (
            not scratch_terminated
            or snapshot.region(LogicalRegion.SCRATCH).text == scratch
        ):
            scratch_delta = None
            second_snapshot = snapshot
        else:
            scratch_delta = replacement_delta(
                snapshot,
                compiled1,
                region=LogicalRegion.SCRATCH,
                text=scratch,
                author_core_id=author_core_id,
                pass_id="scratch",
                provenance="canonical_d64_runtime_scratch",
            )
            second_snapshot = apply_compiled_delta(snapshot, compiled1, scratch_delta)
        compiled2 = active_compiler.compile(second_snapshot)
        state2, memory2, coverage2 = self.read_compiled_with_memory(compiled2)
        response, response_terminated = self.decode_greedy(
            state2, head=1, memory=memory2
        )
        if (
            not response_terminated
            or second_snapshot.region(LogicalRegion.RESPONSE_DRAFT).text == response
        ):
            response_delta = None
            final_snapshot = second_snapshot
        else:
            response_delta = replacement_delta(
                second_snapshot,
                compiled2,
                region=LogicalRegion.RESPONSE_DRAFT,
                text=response,
                author_core_id=author_core_id,
                pass_id="response_draft",
                provenance="canonical_d64_runtime_response",
            )
            final_snapshot = apply_compiled_delta(
                second_snapshot, compiled2, response_delta
            )
        return {
            "scratch": scratch,
            "response_draft": response,
            "scratch_terminated": scratch_terminated,
            "response_terminated": response_terminated,
            "coverage_tick1": coverage1.to_dict(),
            "coverage_tick2": coverage2.to_dict(),
            "typed_delta": {
                "schema": "axon-canonical-d64-transaction-v1",
                "scratch": None if scratch_delta is None else {**scratch_delta.to_canonical_dict(), "delta_id": scratch_delta.delta_id},
                "response_draft": None if response_delta is None else {**response_delta.to_canonical_dict(), "delta_id": response_delta.delta_id},
            },
            "scratch_delta": scratch_delta,
            "response_delta": response_delta,
            "base_snapshot": snapshot,
            "scratch_snapshot": second_snapshot,
            "final_snapshot": final_snapshot,
            "compiled_tick1": compiled1,
            "compiled_tick2": compiled2,
        }





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
    "ReaderConfig",
    "CoverageManifest",
    "coverage_manifest_from_compiled",
    "CompleteField64D",
    "canonical_field",
    "frozen_orthogonal_lift",
    "sequence_cross_entropy",
    "teacher_char_accuracy",
]
