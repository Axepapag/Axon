"""Standalone English-native D64 reasoning tissue.

The active Core reads the frozen Shared Field, proposal workspaces, and its
private Soul, then emits variable-length Unicode. FIRST and REFINED are free
English proposals. CONSOLIDATED is compact tagged-region English; Heart alone
turns that desired-state text into internal typed FieldDelta mutations.

The pre-2026-09-19 typed decision/operation/address implementation remains in
``legacy_typed_reasoning_d64`` only as historical checkpoint evidence. This
module neither imports nor subclasses that executable implementation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from runtime.field import (
    IDENTITY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    CompiledD64Field,
    D64FieldCompiler,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_region_order,
    canonical_sha256,
)
from runtime.heart import (
    EnglishProposal,
    ReasoningPassRequest,
    ReasoningPassResult,
    TechnicalFinalVerdict,
)
from runtime.soul import (
    SOUL_TEMPERATURE_ORDER,
    SoulLayer,
    SoulSnapshot,
    SoulTemperature,
    SoulTransition,
    apply_soul_transition,
)
from runtime.source_of_truth import capacity_policy
from substrate import TRANSPORT_VOCAB_SIZE, decode_unicode_tokens, encode_unicode_text

from .complete_field_64d import (
    AddressableMemory,
    CompleteField64D,
    CoverageManifest,
    ReaderConfig,
)

LIVING_REASONING_ARCHITECTURE_SCHEMA = "axon-living-reasoning-english-architecture-v1"
ENGLISH_REASONING_OUTPUT_CONTRACT = "english-proposal-tagged-final-v1"
ENGLISH_REASONING_TERMINATION_CONTRACT = "generated-eos-independent-of-content-gate-v1"
D64_ENGLISH_MIGRATION_SCHEMA = "axon-d64-english-reasoning-migration-v1"
D64_SOUL_CODEC_VERSION = "axon-d64-recurrent-soul-codec-v1"
D64_SOUL_MEDIA_TYPE = "application/x-axon-d64-recurrent-state"
_SOUL_MAGIC = b"AXSLD641"
_SOUL_HEADER = struct.Struct("<8sII")
_PHASE_TO_ID = {"first": 0, "refined": 1, "consolidated": 2}

# Historical donor-only names. They are not active modules or routing concepts;
# this allowlist merely proves which extra tensors may be discarded when an old
# typed checkpoint is copied into the English topology.
LEGACY_TYPED_OUTPUT_TENSOR_PREFIXES = (
    "decision_head.",
    "operation_head.",
    "region_head.",
    "start_query.",
    "end_query.",
    "boundary_seed",
)


@dataclass(frozen=True, slots=True)
class LivingReasoningCoreConfig:
    """Current D64 anatomy; no retired typed or termination-route switches."""

    d_model: int = 64
    n_heads: int = 1
    n_layers: int = 2
    ffn_dim: int = 131072
    state_tokens: int = 4
    page_size: int = 32
    dropout: float = 0.05
    soul_codec_version: str = D64_SOUL_CODEC_VERSION
    lift_seed: int = 7
    generate_gate_bias: float = 1.5
    field_schema_version: str = SCHEMA_VERSION
    reasoning_output_contract: str = ENGLISH_REASONING_OUTPUT_CONTRACT
    architecture_id: str = field(init=False)

    def __post_init__(self) -> None:
        canonical_region_order(self.field_schema_version)
        for name in ("d_model", "n_heads", "n_layers", "ffn_dim", "state_tokens", "page_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model != 64:
            raise ValueError("living D64 reasoning is locked to d_model=64")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise TypeError("dropout must be numeric")
        dropout = float(self.dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite in [0, 1)")
        if not isinstance(self.generate_gate_bias, (int, float)) or isinstance(self.generate_gate_bias, bool):
            raise TypeError("generate_gate_bias must be numeric")
        generate_gate_bias = float(self.generate_gate_bias)
        if not math.isfinite(generate_gate_bias):
            raise ValueError("generate_gate_bias must be finite")
        if not isinstance(self.soul_codec_version, str) or not self.soul_codec_version:
            raise ValueError("soul_codec_version must be non-empty")
        if self.reasoning_output_contract != ENGLISH_REASONING_OUTPUT_CONTRACT:
            raise ValueError(
                "LivingReasoningCoreConfig supports only the English proposal/tagged FINAL contract"
            )
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "generate_gate_bias", generate_gate_bias)
        object.__setattr__(
            self,
            "architecture_id",
            "living-d64-english-"
            + canonical_sha256(self.to_canonical_dict(include_id=False))[:24],
        )

    def reader_config(self) -> ReaderConfig:
        return ReaderConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ffn_dim=self.ffn_dim,
            state_tokens=self.state_tokens,
            page_size=self.page_size,
            dropout=self.dropout,
            lift_seed=self.lift_seed,
            generate_gate_bias=self.generate_gate_bias,
            field_schema_version=self.field_schema_version,
        )

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        # Keep the already-published English architecture identity stable across
        # this implementation refactor. generate_gate_bias is initialization,
        # not topology, and historically was not part of architecture identity.
        value: dict[str, Any] = {
            "schema": LIVING_REASONING_ARCHITECTURE_SCHEMA,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "state_tokens": self.state_tokens,
            "page_size": self.page_size,
            "dropout": self.dropout,
            "soul_codec_version": self.soul_codec_version,
            "lift_seed": self.lift_seed,
        }
        if self.field_schema_version not in {SCHEMA_VERSION, IDENTITY_SCHEMA_VERSION}:
            value["field_schema_version"] = self.field_schema_version
            value["canonical_region_order"] = [
                region.value for region in canonical_region_order(self.field_schema_version)
            ]
        value["reasoning_output_contract"] = self.reasoning_output_contract
        value["termination_contract"] = ENGLISH_REASONING_TERMINATION_CONTRACT
        if include_id:
            value["architecture_id"] = self.architecture_id
        return value


class D64SoulCodec:
    """Exact architecture-owned codec for opaque recurrent Soul tensors."""

    def __init__(self, config: LivingReasoningCoreConfig) -> None:
        self.config = config
        self.tensor_layout = (
            f"{config.soul_codec_version}:{config.architecture_id}:"
            f"f32le[{config.state_tokens},{config.d_model}]"
        )

    def encode(self, state: torch.Tensor, temperature: SoulTemperature) -> SoulLayer:
        expected = (1, self.config.state_tokens, self.config.d_model)
        if tuple(state.shape) != expected:
            raise ValueError(f"Soul exhale state must have shape {expected}")
        array = state.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        payload = _SOUL_HEADER.pack(
            _SOUL_MAGIC,
            self.config.state_tokens,
            self.config.d_model,
        ) + array.astype("<f4", copy=False).tobytes(order="C")
        return SoulLayer(
            temperature=temperature,
            payload=payload,
            media_type=D64_SOUL_MEDIA_TYPE,
            tensor_layout=self.tensor_layout,
        )

    def decode(
        self,
        layer: SoulLayer,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not layer.payload:
            return None
        if layer.media_type != D64_SOUL_MEDIA_TYPE or layer.tensor_layout != self.tensor_layout:
            raise ValueError("opaque Soul layer does not use this architecture's tensor dialect")
        expected_bytes = (
            _SOUL_HEADER.size
            + self.config.state_tokens * self.config.d_model * torch.float32.itemsize
        )
        if len(layer.payload) != expected_bytes:
            raise ValueError("opaque Soul recurrent payload has the wrong byte length")
        magic, state_tokens, d_model = _SOUL_HEADER.unpack(layer.payload[: _SOUL_HEADER.size])
        if (
            magic != _SOUL_MAGIC
            or state_tokens != self.config.state_tokens
            or d_model != self.config.d_model
        ):
            raise ValueError("opaque Soul recurrent payload header is incompatible")
        values = torch.frombuffer(
            bytearray(layer.payload[_SOUL_HEADER.size :]),
            dtype=torch.float32,
        ).clone()
        state = values.reshape(1, self.config.state_tokens, self.config.d_model)
        if not torch.isfinite(state).all():
            raise ValueError("opaque Soul recurrent payload contains non-finite values")
        return state.to(device=device, dtype=dtype)


@dataclass(slots=True)
class LivingReasoningForward:
    """One English-native reasoning pass before text decoding."""

    reader_state: torch.Tensor
    canonical_memory: AddressableMemory
    complete_memory: AddressableMemory
    inhaled_state: torch.Tensor
    exhaled_state: torch.Tensor
    canonical_coverage: CoverageManifest
    proposal_coverages: tuple[CoverageManifest, ...]
    soul_telemetry: Mapping[str, float]
    core_id: str
    parameter_generation: str
    phase: str
    source_field_id: str
    source_tick_id: int
    view_id: str
    rail_id: str
    surface_id: str


@dataclass(frozen=True, slots=True)
class CausalLivingUnroll:
    """Exact FIRST -> REFINED -> CONSOLIDATED training-time Soul chain."""

    outputs: tuple[LivingReasoningForward, ...]
    souls: tuple[SoulSnapshot, ...]
    transitions: tuple[SoulTransition, ...]

    def __post_init__(self) -> None:
        if len(self.outputs) != 3 or len(self.souls) != 4 or len(self.transitions) != 3:
            raise ValueError("causal living unroll requires three phases and four Soul snapshots")
        for index, transition in enumerate(self.transitions):
            if transition.before_soul_id != self.souls[index].soul_id:
                raise ValueError("training unroll transition does not inhale its causal predecessor")
            if self.souls[index + 1].parent_soul_id != self.souls[index].soul_id:
                raise ValueError("training unroll Soul lineage is discontinuous")


@dataclass(frozen=True, slots=True)
class TensorCopyReceipt:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source_sha256: str
    target_sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.source_sha256 or not self.target_sha256:
            raise ValueError("tensor copy receipt fields must be non-empty")
        if self.source_sha256 != self.target_sha256:
            raise ValueError("tensor copy receipt is not byte exact")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "source_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
        }


@dataclass(frozen=True, slots=True)
class D64EnglishMigrationReceipt:
    """Exact proof that a typed donor became an English-native candidate."""

    source_architecture_id: str
    target_architecture_id: str
    source_parameter_generation: str
    target_parameter_generation: str
    copied_tensors: tuple[TensorCopyReceipt, ...]
    retired_tensors: tuple[str, ...]
    initialization: str = "no new learned tensors; exact donor subset copy"
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_architecture_id",
            "target_architecture_id",
            "source_parameter_generation",
            "target_parameter_generation",
            "initialization",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"D64EnglishMigrationReceipt.{name} must be non-empty")
        if self.source_architecture_id == self.target_architecture_id:
            raise ValueError("English migration must create a new architecture identity")
        if self.source_parameter_generation == self.target_parameter_generation:
            raise ValueError("English migration must create a new parameter generation")
        copied = tuple(self.copied_tensors)
        copied_names = tuple(item.name for item in copied)
        if not copied or copied_names != tuple(sorted(copied_names)):
            raise ValueError("copied tensor receipts must be complete and sorted")
        retired = tuple(sorted(set(self.retired_tensors)))
        if retired != tuple(self.retired_tensors) or not retired:
            raise ValueError("retired tensor names must be non-empty, unique, and sorted")
        if set(copied_names) & set(retired):
            raise ValueError("a tensor cannot be both copied and retired")
        object.__setattr__(self, "copied_tensors", copied)
        object.__setattr__(self, "retired_tensors", retired)
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": D64_ENGLISH_MIGRATION_SCHEMA,
            "source_architecture_id": self.source_architecture_id,
            "target_architecture_id": self.target_architecture_id,
            "source_parameter_generation": self.source_parameter_generation,
            "target_parameter_generation": self.target_parameter_generation,
            "copied_tensors": [item.to_canonical_dict() for item in self.copied_tensors],
            "retired_tensors": list(self.retired_tensors),
            "initialization": self.initialization,
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


def _tensor_sha256(tensor: torch.Tensor) -> str:
    packed = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(
        packed.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    ).hexdigest()


def _is_retired_typed_tensor(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix)
        for prefix in LEGACY_TYPED_OUTPUT_TENSOR_PREFIXES
    )


def _join_memory(items: Iterable[AddressableMemory]) -> AddressableMemory:
    memories = tuple(items)
    if not memories:
        raise ValueError("at least one addressable memory is required")
    return AddressableMemory(
        states=torch.cat([item.states for item in memories], dim=1),
        char_indices=torch.cat([item.char_indices for item in memories], dim=1),
        region_ids=torch.cat([item.region_ids for item in memories], dim=1),
        region_positions=torch.cat([item.region_positions for item in memories], dim=1),
        receipts=tuple(receipt for item in memories for receipt in item.receipts),
        segments=tuple(segment for item in memories for segment in item.segments),
    )


class LivingReasoningCoreD64(CompleteField64D):
    """Standalone D64 Core whose learned public surface is Unicode text."""

    def __init__(self, config: LivingReasoningCoreConfig | None = None) -> None:
        active_config = config or LivingReasoningCoreConfig()
        if not isinstance(active_config, LivingReasoningCoreConfig):
            raise TypeError(
                "English-native LivingReasoningCoreD64 requires LivingReasoningCoreConfig"
            )
        self.living_config = active_config
        super().__init__(active_config.reader_config())
        cfg = active_config
        self.soul_codec = D64SoulCodec(cfg)

        # Replace the base native-only text surface with the permanent exact
        # 351-category Unicode transport plus EMPTY/EOS; BOS is input-only.
        self.vocab_size = TRANSPORT_VOCAB_SIZE
        self.empty_index = TRANSPORT_VOCAB_SIZE
        self.eos_index = TRANSPORT_VOCAB_SIZE + 1
        self.bos_index = TRANSPORT_VOCAB_SIZE + 2
        self.decoder_embedding = nn.Embedding(self.bos_index + 1, cfg.d_model)
        self.decoder_output = nn.Linear(cfg.d_model, self.eos_index + 1)

        self.phase_embedding = nn.Embedding(len(_PHASE_TO_ID), cfg.d_model)
        self.soul_projection = nn.ModuleDict(
            {
                temperature.value: nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                for temperature in SOUL_TEMPERATURE_ORDER
            }
        )
        self.soul_gate_logits = nn.Parameter(torch.zeros(len(SOUL_TEMPERATURE_ORDER)))

    @property
    def architecture_id(self) -> str:
        return self.living_config.architecture_id

    def _copy_token_id(self, transport_token_id: int) -> int:
        return transport_token_id

    def _target_indices(self, text: str) -> torch.Tensor:
        return torch.tensor(
            [*encode_unicode_text(text), self.eos_index],
            dtype=torch.long,
            device=self.device,
        )

    def _initial_decoder_hidden(self, reader_state: torch.Tensor, head: int) -> torch.Tensor:
        if head not in (0, 1):
            raise ValueError("decoder head must be zero or one")
        summary = reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([head], device=self.device))
        return torch.tanh(
            self.decoder_init(torch.cat((summary, head_vec), dim=-1))
        ).unsqueeze(0)

    def _decoder_logits(
        self,
        output: torch.Tensor,
        memory: AddressableMemory | None,
        *,
        return_alignment: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Use ordinary generated EOS while the copy gate routes content only."""

        mixed_logits, alignment = CompleteField64D._decoder_logits(
            self,
            output,
            memory,
            return_alignment=True,
        )
        generated_log_probabilities = F.log_softmax(
            alignment["generated_logits"], dim=-1
        )
        if memory is None:
            mixed_logits = generated_log_probabilities
        else:
            tiny = torch.finfo(mixed_logits.dtype).tiny
            epsilon = torch.finfo(mixed_logits.dtype).eps
            generated_eos = generated_log_probabilities[..., self.eos_index].exp()
            mixed_eos = mixed_logits[..., self.eos_index].exp()
            generated_eos = generated_eos.clamp(min=tiny, max=1.0 - epsilon)
            mixed_eos = mixed_eos.clamp(min=tiny, max=1.0 - epsilon)
            content_scale = torch.log1p(-generated_eos) - torch.log1p(-mixed_eos)
            mixed_logits = mixed_logits.clone()
            mixed_logits[..., : self.eos_index] = (
                mixed_logits[..., : self.eos_index] + content_scale.unsqueeze(-1)
            )
            mixed_logits[..., self.eos_index] = generated_log_probabilities[
                ..., self.eos_index
            ]
        if return_alignment:
            return mixed_logits, alignment
        return mixed_logits

    def alignment_supervision(
        self,
        *,
        target_text: str,
        memory: AddressableMemory,
        decoder_alignment: Mapping[str, torch.Tensor],
        specification: Mapping[str, Any],
        position_reduction: str = "mean",
    ) -> dict[str, Any]:
        """Supervise every exact Unicode transport cell and ordinary EOS routing."""

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

        scalar_offsets: list[int] = []
        transport_count = 0
        for character in target_text:
            scalar_offsets.append(transport_count)
            transport_count += len(encode_unicode_text(character))
        if gate_logits.shape[1] != transport_count + 1:
            raise ValueError(
                "decoder alignment length does not match Unicode transport plus EOS"
            )

        zero = gate_logits.sum() * 0.0
        position_losses: list[torch.Tensor] = []
        copy_gate_losses: list[torch.Tensor] = []
        eos_gate_losses: list[torch.Tensor] = []
        position_correct = 0
        copy_gate_correct = 0
        supervised_copy_positions = 0
        source_memory_indices: list[int] = []
        learned_decision_mask = torch.ones(
            (1, transport_count + 1),
            dtype=torch.bool,
            device=self.device,
        )
        region_to_id = {
            region.value: index for index, region in enumerate(self.region_order)
        }
        required = {
            "target_start",
            "target_end",
            "source_region",
            "source_start",
            "source_end",
            "text_sha256",
            "authority",
        }
        for segment in segments:
            if not isinstance(segment, Mapping) or set(segment) != required:
                raise ValueError("alignment segment fields are invalid")
            target_start = int(segment["target_start"])
            target_end = int(segment["target_end"])
            source_start = int(segment["source_start"])
            source_end = int(segment["source_end"])
            source_region = str(segment["source_region"])
            if source_region not in region_to_id:
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
            if (
                hashlib.sha256(target_fragment.encode("utf-8")).hexdigest()
                != segment["text_sha256"]
            ):
                raise ValueError("alignment target fragment hash mismatch")

            for scalar_offset, source_position in enumerate(
                range(source_start, source_end)
            ):
                target_scalar_position = target_start + scalar_offset
                expected_tokens = tuple(
                    encode_unicode_text(target_text[target_scalar_position])
                )
                matches = (
                    (
                        (memory.region_ids[0] == region_to_id[source_region])
                        & (memory.region_positions[0] == source_position)
                        & memory.char_indices[0].ge(0)
                    )
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                if matches.numel() != len(expected_tokens):
                    raise ValueError(
                        "alignment source scalar does not resolve to its exact transport cells"
                    )
                observed_tokens = tuple(
                    int(memory.char_indices[0, int(memory_index)].item())
                    for memory_index in matches
                )
                if observed_tokens != expected_tokens:
                    raise ValueError(
                        "alignment source transport does not equal supervised target"
                    )
                decoder_start = scalar_offsets[target_scalar_position]
                for token_offset, memory_index in enumerate(matches):
                    target_position = decoder_start + token_offset
                    expected_source = torch.tensor(
                        [int(memory_index.item())],
                        dtype=torch.long,
                        device=self.device,
                    )
                    source_logits = position_logits[:, target_position, :]
                    position_losses.append(
                        F.cross_entropy(source_logits, expected_source)
                    )
                    predicted_source = int(source_logits[0].argmax(dim=-1).item())
                    source_memory_indices.append(int(memory_index.item()))
                    position_correct += int(
                        predicted_source == int(memory_index.item())
                    )
                    copy_target = torch.zeros(
                        (1,), device=self.device, dtype=gate_logits.dtype
                    )
                    copy_gate_losses.append(
                        F.binary_cross_entropy_with_logits(
                            gate_logits[:, target_position], copy_target
                        )
                    )
                    copy_gate_correct += int(
                        float(gate_logits[0, target_position].item()) < 0.0
                    )
                    supervised_copy_positions += 1

        eos_supervised = bool(specification.get("supervise_eos_generate", True))
        eos_gate_correct = 0
        if eos_supervised:
            eos_target = torch.ones((1,), device=self.device, dtype=gate_logits.dtype)
            eos_gate_losses.append(
                F.binary_cross_entropy_with_logits(
                    gate_logits[:, transport_count], eos_target
                )
            )
            eos_gate_correct = int(
                float(gate_logits[0, transport_count].item()) >= 0.0
            )

        if not position_losses:
            position_loss = zero
        elif position_reduction == "mean":
            position_loss = torch.stack(position_losses).mean()
        elif position_reduction == "sum":
            position_loss = torch.stack(position_losses).sum()
        else:
            raise ValueError(
                f"unsupported alignment position reduction {position_reduction!r}"
            )
        copy_gate_loss = (
            torch.stack(copy_gate_losses).mean() if copy_gate_losses else zero
        )
        eos_gate_loss = (
            torch.stack(eos_gate_losses).mean() if eos_gate_losses else zero
        )
        gate_losses = copy_gate_losses + eos_gate_losses
        gate_loss = torch.stack(gate_losses).mean() if gate_losses else zero
        eos_gate_supervised = int(eos_supervised)
        gate_count = supervised_copy_positions + eos_gate_supervised
        gate_correct = copy_gate_correct + eos_gate_correct
        return {
            "position_loss": position_loss,
            "gate_loss": gate_loss,
            "copy_gate_loss": copy_gate_loss,
            "eos_gate_loss": eos_gate_loss,
            "copy_positions": supervised_copy_positions,
            "source_memory_indices": tuple(source_memory_indices),
            "deterministic_continuation_positions": 0,
            "learned_decision_mask": learned_decision_mask,
            "position_correct": position_correct,
            "gate_supervised_positions": gate_count,
            "gate_correct": gate_correct,
            "copy_gate_correct": copy_gate_correct,
            "eos_gate_supervised_positions": eos_gate_supervised,
            "eos_gate_correct": eos_gate_correct,
            "position_accuracy": (
                position_correct / supervised_copy_positions
                if supervised_copy_positions
                else 1.0
            ),
            "copy_gate_accuracy": (
                copy_gate_correct / supervised_copy_positions
                if supervised_copy_positions
                else 1.0
            ),
            "eos_gate_accuracy": (
                eos_gate_correct / eos_gate_supervised
                if eos_gate_supervised
                else 1.0
            ),
            "gate_accuracy": gate_correct / gate_count if gate_count else 1.0,
        }

    def inhale(
        self,
        soul: SoulSnapshot,
        *,
        expected_core_id: str,
        parameter_generation: str,
        phase: str,
        ablate_temperatures: Iterable[SoulTemperature | str] = (),
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        if soul.core_id != expected_core_id:
            raise ValueError("a living core cannot inhale another core's private Soul")
        if soul.architecture_id != self.architecture_id:
            raise ValueError("private Soul architecture does not match the living core")
        if soul.parameter_generation != parameter_generation:
            raise ValueError(
                "private Soul parameter generation does not match the living core"
            )
        if phase not in _PHASE_TO_ID:
            raise ValueError(f"unsupported living reasoning phase {phase!r}")
        ablated = {
            item if isinstance(item, SoulTemperature) else SoulTemperature(item)
            for item in ablate_temperatures
        }
        baseline = self.initial_state.unsqueeze(0)
        state = baseline
        decoded_count = 0
        contribution_l2 = 0.0
        gates = torch.sigmoid(self.soul_gate_logits)
        for index, temperature in enumerate(SOUL_TEMPERATURE_ORDER):
            if temperature in ablated:
                continue
            decoded = self.soul_codec.decode(
                soul.layer(temperature),
                device=self.device,
                dtype=baseline.dtype,
            )
            if decoded is None:
                continue
            contribution = (
                self.soul_projection[temperature.value](decoded) * gates[index]
            )
            state = state + contribution
            contribution_l2 += float(contribution.detach().norm().item())
            decoded_count += 1
        phase_vector = self.phase_embedding(
            torch.tensor(_PHASE_TO_ID[phase], device=self.device)
        ).reshape(1, 1, -1)
        state = state + phase_vector
        return state, {
            "decoded_soul_layers": float(decoded_count),
            "soul_contribution_l2": contribution_l2,
            "inhaled_state_l2": float(state.detach().norm().item()),
        }

    def _proposal_compiled(self, text: str, index: int) -> CompiledD64Field:
        snapshot = SharedFieldSnapshot.from_texts(
            {LogicalRegion.ADVISOR_INPUT: text},
            source_manifest_ids=(f"derived-proposal-rail:{index}",),
        )
        return D64FieldCompiler().compile(snapshot)

    def forward_surfaces(
        self,
        *,
        soul: SoulSnapshot,
        expected_core_id: str,
        parameter_generation: str,
        phase: str,
        canonical: CompiledD64Field,
        proposal_texts: Iterable[str] = (),
        ablate_temperatures: Iterable[SoulTemperature | str] = (),
        view_id: str | None = None,
    ) -> LivingReasoningForward:
        state, telemetry = self.inhale(
            soul,
            expected_core_id=expected_core_id,
            parameter_generation=parameter_generation,
            phase=phase,
            ablate_temperatures=ablate_temperatures,
        )
        inhaled_state = state
        state, canonical_memory, canonical_coverage = self.read_compiled_with_memory(
            canonical,
            initial_state=state,
            memory_segment_kind="canonical",
            memory_segment_label=f"canonical:{canonical.rail_id}",
        )
        memories = [canonical_memory]
        proposal_coverages: list[CoverageManifest] = []
        for index, text in enumerate(proposal_texts):
            compiled = self._proposal_compiled(text, index)
            state, memory, coverage = self.read_compiled_with_memory(
                compiled,
                initial_state=state,
                memory_segment_kind="proposal",
                memory_segment_label=f"proposal:{index}:{compiled.rail_id}",
            )
            memories.append(memory)
            proposal_coverages.append(coverage)
        telemetry = {
            **telemetry,
            "exhaled_state_l2": float(state.detach().norm().item()),
            "state_change_l2": float(
                (state.detach() - self.initial_state.unsqueeze(0)).norm().item()
            ),
            "canonical_pages": float(canonical_coverage.page_count),
            "proposal_pages": float(
                sum(item.page_count for item in proposal_coverages)
            ),
        }
        complete_memory = _join_memory(memories)
        resolved_view_id = canonical.rail_id if view_id is None else view_id
        if not isinstance(resolved_view_id, str) or not resolved_view_id:
            raise ValueError("living reasoning view_id must be non-empty")
        surface_id = canonical_sha256(
            {
                "schema": "axon-living-d64-english-decoder-surface-v1",
                "field_id": canonical.source_field_id,
                "tick_id": canonical.source_tick_id,
                "view_id": resolved_view_id,
                "rail_id": canonical.rail_id,
                "complete_memory_id": complete_memory.memory_id,
                "phase": phase,
                "reasoning_output_contract": ENGLISH_REASONING_OUTPUT_CONTRACT,
            }
        )
        return LivingReasoningForward(
            reader_state=state,
            canonical_memory=canonical_memory,
            complete_memory=complete_memory,
            inhaled_state=inhaled_state,
            exhaled_state=state,
            canonical_coverage=canonical_coverage,
            proposal_coverages=tuple(proposal_coverages),
            soul_telemetry=telemetry,
            core_id=expected_core_id,
            parameter_generation=parameter_generation,
            phase=phase,
            source_field_id=canonical.source_field_id,
            source_tick_id=canonical.source_tick_id,
            view_id=resolved_view_id,
            rail_id=canonical.rail_id,
            surface_id=surface_id,
        )

    def exhale_transition(
        self,
        *,
        before: SoulSnapshot,
        exhaled_state: torch.Tensor,
        tick_uid: str,
        request_id: str,
        phase: str,
    ) -> SoulTransition:
        return SoulTransition(
            core_id=before.core_id,
            architecture_id=before.architecture_id,
            parameter_generation=before.parameter_generation,
            before_soul_id=before.soul_id,
            before_generation=before.generation,
            tick_uid=tick_uid,
            request_id=request_id,
            phase=phase,
            updates=(
                self.soul_codec.encode(exhaled_state, SoulTemperature.HOT),
            ),
        )

    def unroll_runtime_phases(
        self,
        *,
        initial_soul: SoulSnapshot,
        expected_core_id: str,
        parameter_generation: str,
        tick_uid: str,
        canonical: CompiledD64Field,
        first_workspace_text: str,
        refined_workspace_text: str,
        ablate_temperatures: Iterable[SoulTemperature | str] = (),
    ) -> CausalLivingUnroll:
        """Run the service phase order with persisted Soul at each boundary."""

        phase_inputs = (
            ("first", ()),
            ("refined", (first_workspace_text,)),
            ("consolidated", (first_workspace_text, refined_workspace_text)),
        )
        soul = initial_soul
        souls = [soul]
        outputs: list[LivingReasoningForward] = []
        transitions: list[SoulTransition] = []
        for phase, proposal_texts in phase_inputs:
            output = self.forward_surfaces(
                soul=soul,
                expected_core_id=expected_core_id,
                parameter_generation=parameter_generation,
                phase=phase,
                canonical=canonical,
                proposal_texts=proposal_texts,
                ablate_temperatures=ablate_temperatures,
            )
            request_id = canonical_sha256(
                {
                    "schema": "axon-training-runtime-request-v1",
                    "architecture_id": self.architecture_id,
                    "core_id": expected_core_id,
                    "parameter_generation": parameter_generation,
                    "tick_uid": tick_uid,
                    "phase": phase,
                    "field_id": canonical.source_field_id,
                    "before_soul_id": soul.soul_id,
                    "proposal_text_sha256": [
                        canonical_sha256({"text": text}) for text in proposal_texts
                    ],
                }
            )
            transition = self.exhale_transition(
                before=soul,
                exhaled_state=output.exhaled_state,
                tick_uid=tick_uid,
                request_id=request_id,
                phase=phase,
            )
            soul = apply_soul_transition(soul, transition)
            outputs.append(output)
            transitions.append(transition)
            souls.append(soul)
        return CausalLivingUnroll(
            outputs=tuple(outputs),
            souls=tuple(souls),
            transitions=tuple(transitions),
        )

    @torch.no_grad()
    def decode_transport_greedy(
        self,
        output: LivingReasoningForward,
        *,
        work_units: int | None = None,
    ) -> tuple[str, bool]:
        return next(self.iter_decode_transport(output, work_units=work_units))

    def iter_decode_transport(
        self,
        output: LivingReasoningForward,
        *,
        work_units: int | None = None,
    ) -> Iterator[tuple[str, bool]]:
        """Yield at renewable work slices; EOS alone terminates generated text."""

        work_slice = (
            capacity_policy().integer(
                "reasoning.emission_work_slice_transport_units"
            )
            if work_units is None
            else int(work_units)
        )
        if work_slice < 1:
            raise ValueError("work_units must be positive")
        summary = output.reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([1], device=self.device))
        hidden = torch.tanh(
            self.decoder_init(torch.cat((summary, head_vec), dim=-1))
        ).unsqueeze(0)
        token = torch.full(
            (1, 1), self.bos_index, dtype=torch.long, device=self.device
        )
        transport: list[int] = []
        with torch.no_grad():
            while True:
                for _ in range(work_slice):
                    decoded, hidden = self.decoder(
                        self.decoder_embedding(token), hidden
                    )
                    logits = self._decoder_logits(
                        decoded[:, -1:], output.complete_memory
                    ).squeeze(1)
                    category = int(logits.argmax(dim=-1).item())
                    if category == self.eos_index:
                        try:
                            yield decode_unicode_tokens(transport), True
                        except ValueError:
                            yield "", False
                        return
                    if category >= TRANSPORT_VOCAB_SIZE:
                        yield "", False
                        return
                    transport.append(category)
                    token = torch.tensor(
                        [[category]], dtype=torch.long, device=self.device
                    )
                yield "", False

    @torch.no_grad()
    def decode_transport_diagnostic(
        self,
        output: LivingReasoningForward,
        *,
        work_units: int,
        max_decisions: int,
    ) -> dict[str, Any]:
        """Record bounded free-run EOS and Unicode evidence without authority."""

        if work_units < 1 or max_decisions < 1:
            raise ValueError("diagnostic work_units and max_decisions must be positive")
        summary = output.reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([1], device=self.device))
        hidden = torch.tanh(
            self.decoder_init(torch.cat((summary, head_vec), dim=-1))
        ).unsqueeze(0)
        token = torch.full(
            (1, 1), self.bos_index, dtype=torch.long, device=self.device
        )
        transport: list[int] = []
        eos_probabilities: list[float] = []
        eos_ranks: list[int] = []
        generated_eos_probabilities: list[float] = []
        generated_eos_logits: list[float] = []
        generated_eos_ranks: list[int] = []
        generate_route_probabilities: list[float] = []
        invalid_category: int | None = None
        terminated = False
        unicode_valid = True
        emitted_text = ""
        with torch.no_grad():
            for _decision in range(max_decisions):
                decoded, hidden = self.decoder(
                    self.decoder_embedding(token), hidden
                )
                mixed, alignment = self._decoder_logits(
                    decoded[:, -1:], output.complete_memory, return_alignment=True
                )
                mixed_probabilities = mixed.exp().squeeze(0).squeeze(0)
                generated_logits = alignment["generated_logits"].squeeze(0).squeeze(0)
                generated_probabilities = F.softmax(generated_logits, dim=-1)
                eos_probability = float(mixed_probabilities[self.eos_index].item())
                generated_eos_probability = float(
                    generated_probabilities[self.eos_index].item()
                )
                eos_probabilities.append(eos_probability)
                generated_eos_probabilities.append(generated_eos_probability)
                generated_eos_logits.append(
                    float(generated_logits[self.eos_index].item())
                )
                eos_ranks.append(
                    int(
                        (mixed_probabilities > mixed_probabilities[self.eos_index])
                        .sum()
                        .item()
                    )
                    + 1
                )
                generated_eos_ranks.append(
                    int(
                        (
                            generated_probabilities
                            > generated_probabilities[self.eos_index]
                        )
                        .sum()
                        .item()
                    )
                    + 1
                )
                route_logits = alignment.get("generate_gate_logits")
                if route_logits is None:
                    generate_route_probabilities.append(1.0)
                else:
                    generate_route_probabilities.append(
                        float(torch.sigmoid(route_logits[0, 0]).item())
                    )
                category = int(mixed.argmax(dim=-1).item())
                if category == self.eos_index:
                    terminated = True
                    try:
                        emitted_text = decode_unicode_tokens(transport)
                    except ValueError:
                        unicode_valid = False
                    break
                if category >= TRANSPORT_VOCAB_SIZE:
                    invalid_category = category
                    unicode_valid = False
                    break
                transport.append(category)
                token = torch.tensor(
                    [[category]], dtype=torch.long, device=self.device
                )
        if not transport:
            unicode_valid = unicode_valid and terminated
        else:
            try:
                decode_unicode_tokens(transport)
            except ValueError:
                unicode_valid = False
        decision_count = len(eos_probabilities)
        return {
            "work_units": int(work_units),
            "max_decisions": int(max_decisions),
            "decision_count": decision_count,
            "transport_categories": list(transport),
            "transport_unit_count": len(transport),
            "terminated": terminated,
            "first_slice_terminated": bool(terminated and decision_count <= work_units),
            "work_slice_exhausted": bool(
                not terminated and decision_count >= work_units
            ),
            "max_decisions_exhausted": bool(
                not terminated and decision_count >= max_decisions
            ),
            "invalid_transport_category": invalid_category,
            "emitted_text": emitted_text if terminated and unicode_valid else "",
            "unicode_valid": unicode_valid,
            "eos_probability_first": eos_probabilities[0] if eos_probabilities else None,
            "eos_probability_last": eos_probabilities[-1] if eos_probabilities else None,
            "eos_probability_max": max(eos_probabilities, default=None),
            "eos_rank_first": eos_ranks[0] if eos_ranks else None,
            "eos_rank_last": eos_ranks[-1] if eos_ranks else None,
            "generated_eos_probability_first": (
                generated_eos_probabilities[0] if generated_eos_probabilities else None
            ),
            "generated_eos_probability_last": (
                generated_eos_probabilities[-1] if generated_eos_probabilities else None
            ),
            "generated_eos_probability_max": max(
                generated_eos_probabilities, default=None
            ),
            "generated_eos_logit_first": (
                generated_eos_logits[0] if generated_eos_logits else None
            ),
            "generated_eos_logit_last": (
                generated_eos_logits[-1] if generated_eos_logits else None
            ),
            "generated_eos_rank_first": (
                generated_eos_ranks[0] if generated_eos_ranks else None
            ),
            "generated_eos_rank_last": (
                generated_eos_ranks[-1] if generated_eos_ranks else None
            ),
            "generate_route_probability_first": (
                generate_route_probabilities[0] if generate_route_probabilities else None
            ),
            "generate_route_probability_last": (
                generate_route_probabilities[-1] if generate_route_probabilities else None
            ),
        }

    def forward_request(
        self,
        request: ReasoningPassRequest,
        *,
        ablate_temperatures: Iterable[SoulTemperature | str] = (),
    ) -> LivingReasoningForward:
        if request.descriptor.d_model != 64:
            raise ValueError("LivingReasoningCoreD64 requires a D64 home rail")
        return self.forward_surfaces(
            soul=request.soul,
            expected_core_id=request.descriptor.core_id,
            parameter_generation=request.descriptor.parameter_generation,
            phase=request.phase,
            canonical=request.rail.exact_surface,
            proposal_texts=(item.text for item in request.proposal_rails),
            ablate_temperatures=ablate_temperatures,
            view_id=request.image.view_id,
        )

    @torch.no_grad()
    def emit(self, request: ReasoningPassRequest) -> ReasoningPassResult:
        """Emit one English proposal or tagged FINAL through the shared decoder."""

        output = self.forward_request(request)
        transition = self.exhale_transition(
            before=request.soul,
            exhaled_state=output.exhaled_state,
            tick_uid=request.image.identity.tick_uid,
            request_id=request.request_id,
            phase=request.phase,
        )
        text, terminated = self.decode_transport_greedy(output)
        if not terminated:
            raise RuntimeError(
                "English reasoning decoder did not terminate within its renewable work slice"
            )
        if not text.strip():
            raise RuntimeError("English reasoning decoder returned empty text")
        common = {
            "base_field_id": request.image.identity.base_field_id,
            "base_tick_id": request.image.identity.base_tick_id,
            "author_core_id": request.descriptor.core_id,
            "rail_d_model": request.descriptor.d_model,
            "text": text,
        }
        if request.phase in {"first", "refined"}:
            public_output = EnglishProposal(pass_id=request.phase, **common)
        elif request.phase == "consolidated":
            public_output = TechnicalFinalVerdict(**common)
        else:
            raise RuntimeError(f"unsupported English reasoning phase {request.phase!r}")
        return ReasoningPassResult(
            output=public_output,
            soul_transition=transition,
        )

    def architecture_report(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            **self.living_config.to_canonical_dict(),
            "parameter_count": parameters,
            "trainable_parameter_count": trainable,
            "parameter_bytes_fp32": parameters * 4,
            "parameter_bytes_fp16": parameters * 2,
            "reasoning_output_contract": ENGLISH_REASONING_OUTPUT_CONTRACT,
            "termination_contract": ENGLISH_REASONING_TERMINATION_CONTRACT,
            "retired_learned_heads": [
                "decision",
                "operation",
                "region",
                "start_address",
                "end_address",
            ],
            "public_output": "variable_length_unicode_text",
            "first_refined_surface": "EnglishProposal",
            "consolidated_surface": "TechnicalFinalVerdict",
        }


def migrate_typed_checkpoint_state_to_english_variant(
    source_state: Mapping[str, torch.Tensor],
    target: LivingReasoningCoreD64,
    *,
    source_architecture_id: str,
    source_parameter_generation: str,
    target_parameter_generation: str,
) -> D64EnglishMigrationReceipt:
    """Strict state_dict-only transplant from an old typed checkpoint."""

    if not isinstance(target, LivingReasoningCoreD64):
        raise TypeError("target must be an English-native LivingReasoningCoreD64")
    if (
        not source_architecture_id
        or not source_parameter_generation
        or not target_parameter_generation
    ):
        raise ValueError("migration identities and generations must be non-empty")
    target_state = target.state_dict()
    source_names = set(source_state)
    target_names = set(target_state)
    missing = sorted(target_names - source_names)
    if missing:
        raise ValueError(
            "typed donor is missing target tensors: " + ", ".join(missing)
        )
    extras = sorted(source_names - target_names)
    invalid_extras = [
        name for name in extras if not _is_retired_typed_tensor(name)
    ]
    if invalid_extras:
        raise ValueError(
            "typed donor contains unexplained tensors outside the retired interface: "
            + ", ".join(invalid_extras)
        )
    if not extras:
        raise ValueError("typed donor does not contain the retired output tensors")

    copied_state: dict[str, torch.Tensor] = {}
    receipts: list[TensorCopyReceipt] = []
    for name in sorted(target_names):
        source_tensor = source_state[name]
        target_tensor = target_state[name]
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError(f"typed donor state {name!r} is not a tensor")
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(f"typed donor tensor shape differs for {name}")
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(f"typed donor tensor dtype differs for {name}")
        copied_state[name] = source_tensor.detach().clone()

    target.load_state_dict(copied_state, strict=True)
    migrated_state = target.state_dict()
    for name in sorted(target_names):
        source_digest = _tensor_sha256(source_state[name])
        target_digest = _tensor_sha256(migrated_state[name])
        receipts.append(
            TensorCopyReceipt(
                name=name,
                shape=tuple(source_state[name].shape),
                dtype=str(source_state[name].dtype),
                source_sha256=source_digest,
                target_sha256=target_digest,
            )
        )
    return D64EnglishMigrationReceipt(
        source_architecture_id=source_architecture_id,
        target_architecture_id=target.architecture_id,
        source_parameter_generation=source_parameter_generation,
        target_parameter_generation=target_parameter_generation,
        copied_tensors=tuple(receipts),
        retired_tensors=tuple(extras),
    )


def candidate_a_config(**overrides: Any) -> LivingReasoningCoreConfig:
    values = asdict(LivingReasoningCoreConfig())
    values.pop("architecture_id", None)
    values.update(overrides)
    return LivingReasoningCoreConfig(**values)


__all__ = [
    "D64_ENGLISH_MIGRATION_SCHEMA",
    "D64_SOUL_CODEC_VERSION",
    "D64_SOUL_MEDIA_TYPE",
    "ENGLISH_REASONING_OUTPUT_CONTRACT",
    "ENGLISH_REASONING_TERMINATION_CONTRACT",
    "LEGACY_TYPED_OUTPUT_TENSOR_PREFIXES",
    "LIVING_REASONING_ARCHITECTURE_SCHEMA",
    "CausalLivingUnroll",
    "D64EnglishMigrationReceipt",
    "D64SoulCodec",
    "LivingReasoningCoreConfig",
    "LivingReasoningCoreD64",
    "LivingReasoningForward",
    "TensorCopyReceipt",
    "candidate_a_config",
    "migrate_typed_checkpoint_state_to_english_variant",
]
