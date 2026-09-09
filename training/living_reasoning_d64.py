"""Soul-conditioned, complete-field D64 reasoning tissue.

This module is the first neural implementation of the permanent runtime
contract.  It inhales one core's opaque private Soul before perception, carries
that recurrent state across every eligible canonical and proposal page, emits
typed-output logits, and exhales the resulting recurrent state as an opaque HOT
Soul successor.  Pages are bounded compute units, never a context limit.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from runtime.field import (
    CANONICAL_REGION_ORDER,
    CompiledD64Field,
    D64FieldCompiler,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)
from runtime.heart import (
    CategoricalTextFrame,
    ReasoningDecision,
    ReasoningEmission,
    ReasoningOperationEmission,
    ReasoningOperationKind,
    ReasoningPassRequest,
    ReasoningPassResult,
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
from substrate import (
    TRANSPORT_VOCAB_SIZE,
    decode_unicode_tokens,
    encode_unicode_text,
)

from .complete_field_64d import (
    AddressableMemory,
    CompleteField64D,
    CoverageManifest,
    MemoryCellReceipt,
    ReaderConfig,
)

LIVING_REASONING_ARCHITECTURE_SCHEMA = "axon-living-reasoning-architecture-v1"
LIVING_REASONING_RECEIPT_ARCHITECTURE_SCHEMA = "axon-living-reasoning-architecture-v2"
D64_DECODER_EXECUTION_STATE_SCHEMA = "axon-d64-decoder-execution-state-v1"
D64_DECODER_TRACE_SCHEMA = "axon-d64-decoder-emission-trace-v1"
D64_RECEIPT_MIGRATION_SCHEMA = "axon-d64-receipt-architecture-migration-v1"
D64_SOUL_CODEC_VERSION = "axon-d64-recurrent-soul-codec-v1"
D64_SOUL_MEDIA_TYPE = "application/x-axon-d64-recurrent-state"
_SOUL_MAGIC = b"AXSLD641"
_SOUL_HEADER = struct.Struct("<8sII")
_PHASE_TO_ID = {"first": 0, "refined": 1, "consolidated": 2}
_RETIRED_V1_OUTPUT_IDENTITY_MARKER = 512


@dataclass(frozen=True, slots=True)
class LivingReasoningCoreConfig:
    d_model: int = 64
    n_heads: int = 1
    n_layers: int = 2
    ffn_dim: int = 131_072
    state_tokens: int = 4
    page_size: int = 32
    dropout: float = 0.05
    soul_codec_version: str = D64_SOUL_CODEC_VERSION
    lift_seed: int = 7
    generate_gate_bias: float = 1.5
    receipt_continuation: bool = False
    architecture_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "n_heads",
            "n_layers",
            "ffn_dim",
            "state_tokens",
            "page_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model != 64:
            raise ValueError("this first living-core implementation consumes the physical D64 rail")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.soul_codec_version:
            raise ValueError("soul_codec_version must be non-empty")
        if not math.isfinite(float(self.generate_gate_bias)):
            raise ValueError("generate_gate_bias must be a finite float")
        if not isinstance(self.receipt_continuation, bool):
            raise TypeError("receipt_continuation must be boolean")
        object.__setattr__(self, "generate_gate_bias", float(self.generate_gate_bias))
        if self.receipt_continuation:
            identity = self.to_canonical_dict(False)
            prefix = "living-d64-receipt-"
        else:
            identity = self._v1_identity_projection()
            prefix = "living-d64-"
        object.__setattr__(
            self,
            "architecture_id",
            prefix + canonical_sha256(identity)[:24],
        )

    def _v1_identity_projection(self) -> dict[str, Any]:
        """Preserve every trained v1 tensor/Soul identity without retaining its ceiling.

        The original schema incorrectly mixed a decoder work allowance into
        tissue identity.  It never changed tensor topology.  This frozen
        projection reproduces existing architecture IDs while live decoding
        takes its renewable work slice from the protected SOT policy.
        """

        value = self.to_canonical_dict(False)
        value["inference_budget_transport_units"] = _RETIRED_V1_OUTPUT_IDENTITY_MARKER
        return value

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": (
                LIVING_REASONING_RECEIPT_ARCHITECTURE_SCHEMA
                if self.receipt_continuation
                else LIVING_REASONING_ARCHITECTURE_SCHEMA
            ),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "state_tokens": self.state_tokens,
            "page_size": self.page_size,
            "dropout": float(self.dropout),
            "soul_codec_version": self.soul_codec_version,
            "lift_seed": self.lift_seed,
        }
        if self.receipt_continuation:
            value.update(
                {
                    "receipt_continuation": True,
                    "decoder_execution_state_schema": D64_DECODER_EXECUTION_STATE_SCHEMA,
                    "emission_routes": [
                        "learned_generate",
                        "learned_copy_anchor",
                        "deterministic_receipt_continuation",
                    ],
                }
            )
        if include_id:
            value["architecture_id"] = self.architecture_id
        return value

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
        )


class D64SoulCodec:
    """Exact architecture-owned codec for opaque recurrent Soul tensors."""

    def __init__(self, config: LivingReasoningCoreConfig) -> None:
        self.config = config
        self.tensor_layout = (
            f"{config.soul_codec_version}:{config.architecture_id}:f32le[{config.state_tokens},{config.d_model}]"
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
        expected_bytes = _SOUL_HEADER.size + self.config.state_tokens * self.config.d_model * torch.float32.itemsize
        if len(layer.payload) != expected_bytes:
            raise ValueError("opaque Soul recurrent payload has the wrong byte length")
        magic, state_tokens, d_model = _SOUL_HEADER.unpack(layer.payload[: _SOUL_HEADER.size])
        if magic != _SOUL_MAGIC or state_tokens != self.config.state_tokens or d_model != self.config.d_model:
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
    reader_state: torch.Tensor
    canonical_memory: AddressableMemory
    complete_memory: AddressableMemory
    decision_logits: torch.Tensor
    operation_logits: torch.Tensor
    region_logits: torch.Tensor
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


class DecoderEmissionRoute(str, Enum):
    LEARNED_GENERATE = "learned_generate"
    LEARNED_COPY_ANCHOR = "learned_copy_anchor"
    DETERMINISTIC_RECEIPT_CONTINUATION = "deterministic_receipt_continuation"


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
class D64ReceiptMigrationReceipt:
    source_architecture_id: str
    target_architecture_id: str
    source_parameter_generation: str
    target_parameter_generation: str
    copied_tensors: tuple[TensorCopyReceipt, ...]
    new_state_fields: tuple[str, ...]
    initialization: str
    disabled_route_equivalence: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_architecture_id",
            "target_architecture_id",
            "source_parameter_generation",
            "target_parameter_generation",
            "initialization",
            "disabled_route_equivalence",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"D64ReceiptMigrationReceipt.{name} must be non-empty")
        if self.source_architecture_id == self.target_architecture_id:
            raise ValueError("receipt migration must create a new architecture identity")
        if self.source_parameter_generation == self.target_parameter_generation:
            raise ValueError("receipt migration must create a new parameter generation")
        names = tuple(item.name for item in self.copied_tensors)
        if not names or names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("copied tensor receipts must be complete, unique, and sorted")
        if len(set(self.new_state_fields)) != len(self.new_state_fields):
            raise ValueError("new decoder state fields must be unique")
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": D64_RECEIPT_MIGRATION_SCHEMA,
            "source_architecture_id": self.source_architecture_id,
            "target_architecture_id": self.target_architecture_id,
            "source_parameter_generation": self.source_parameter_generation,
            "target_parameter_generation": self.target_parameter_generation,
            "copied_tensors": [item.to_canonical_dict() for item in self.copied_tensors],
            "new_state_fields": list(self.new_state_fields),
            "initialization": self.initialization,
            "disabled_route_equivalence": self.disabled_route_equivalence,
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


@dataclass(frozen=True, slots=True)
class DecoderTraceEvent:
    step_index: int
    route: DecoderEmissionRoute
    category: int
    memory_index: int | None
    receipt_id: str | None
    anchor_receipt_id: str | None
    transport_unit_index: int | None
    transport_unit_count: int | None
    trace_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("decoder trace step_index must be a non-negative integer")
        if isinstance(self.category, bool) or not isinstance(self.category, int) or self.category < 0:
            raise ValueError("decoder trace category must be a non-negative integer")
        has_memory = self.memory_index is not None
        if self.route is DecoderEmissionRoute.LEARNED_GENERATE:
            if self.category == TRANSPORT_VOCAB_SIZE or self.category > TRANSPORT_VOCAB_SIZE + 1:
                raise ValueError("learned generation trace contains EMPTY, BOS, or invalid category")
            if has_memory or any(
                item is not None
                for item in (
                    self.receipt_id,
                    self.anchor_receipt_id,
                    self.transport_unit_index,
                    self.transport_unit_count,
                )
            ):
                raise ValueError("learned generation cannot claim a memory receipt")
        else:
            if not 0 <= self.category < TRANSPORT_VOCAB_SIZE:
                raise ValueError("copy/continuation trace category must be exact transport")
            if (
                self.memory_index is None
                or isinstance(self.memory_index, bool)
                or self.memory_index < 0
                or not self.receipt_id
                or not self.anchor_receipt_id
                or self.transport_unit_index is None
                or self.transport_unit_count is None
            ):
                raise ValueError("copy/continuation trace requires exact receipt identity")
            if (
                isinstance(self.transport_unit_index, bool)
                or isinstance(self.transport_unit_count, bool)
                or not isinstance(self.transport_unit_index, int)
                or not isinstance(self.transport_unit_count, int)
                or self.transport_unit_count < 1
                or not 0 <= self.transport_unit_index < self.transport_unit_count
            ):
                raise ValueError("copy/continuation trace transport-unit identity is invalid")
        object.__setattr__(self, "trace_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": D64_DECODER_TRACE_SCHEMA,
            "step_index": self.step_index,
            "route": self.route.value,
            "category": self.category,
            "memory_index": self.memory_index,
            "receipt_id": self.receipt_id,
            "anchor_receipt_id": self.anchor_receipt_id,
            "transport_unit_index": self.transport_unit_index,
            "transport_unit_count": self.transport_unit_count,
        }
        if include_id:
            value["trace_id"] = self.trace_id
        return value

    @classmethod
    def from_canonical_dict(cls, value: Mapping[str, Any]) -> "DecoderTraceEvent":
        expected = {
            "schema",
            "step_index",
            "route",
            "category",
            "memory_index",
            "receipt_id",
            "anchor_receipt_id",
            "transport_unit_index",
            "transport_unit_count",
            "trace_id",
        }
        if set(value) != expected or value.get("schema") != D64_DECODER_TRACE_SCHEMA:
            raise ValueError("decoder trace schema or fields are invalid")
        event = cls(
            step_index=int(value["step_index"]),
            route=DecoderEmissionRoute(str(value["route"])),
            category=int(value["category"]),
            memory_index=(None if value["memory_index"] is None else int(value["memory_index"])),
            receipt_id=(None if value["receipt_id"] is None else str(value["receipt_id"])),
            anchor_receipt_id=(
                None if value["anchor_receipt_id"] is None else str(value["anchor_receipt_id"])
            ),
            transport_unit_index=(
                None
                if value["transport_unit_index"] is None
                else int(value["transport_unit_index"])
            ),
            transport_unit_count=(
                None
                if value["transport_unit_count"] is None
                else int(value["transport_unit_count"])
            ),
        )
        if event.trace_id != value["trace_id"]:
            raise ValueError("decoder trace identity mismatch")
        return event


@dataclass(frozen=True, slots=True)
class DecoderExecutionState:
    """Portable full causal decoder state for exact renewable-slice resume."""

    architecture_id: str
    parameter_generation: str
    core_id: str
    field_id: str
    tick_id: int
    view_id: str
    rail_id: str
    surface_id: str
    memory_id: str
    memory_segment_ids: tuple[str, ...]
    pass_id: str
    head: int
    hidden_shape: tuple[int, ...]
    hidden_f32le_b64: str
    previous_category: int
    pending_anchor_memory_index: int | None
    pending_anchor_receipt_id: str | None
    pending_next_unit_index: int | None
    transport_categories: tuple[int, ...]
    trace: tuple[DecoderTraceEvent, ...]
    work_units_completed: int
    terminated: bool = False
    malformed_reason: str | None = None
    state_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "architecture_id",
            "parameter_generation",
            "core_id",
            "field_id",
            "view_id",
            "rail_id",
            "surface_id",
            "memory_id",
            "pass_id",
            "hidden_f32le_b64",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"DecoderExecutionState.{name} must be non-empty")
        if len(set(self.memory_segment_ids)) != len(self.memory_segment_ids) or not all(
            isinstance(item, str) and item for item in self.memory_segment_ids
        ):
            raise ValueError("decoder state memory segment identities are invalid")
        if self.hidden_shape != (1, 1, 64):
            raise ValueError("D64 decoder hidden shape must be exactly [1, 1, 64]")
        try:
            hidden_bytes = base64.b64decode(self.hidden_f32le_b64.encode("ascii"), validate=True)
        except Exception as exc:
            raise ValueError("decoder hidden state is not canonical base64") from exc
        if len(hidden_bytes) != math.prod(self.hidden_shape) * torch.float32.itemsize:
            raise ValueError("decoder hidden state byte length is invalid")
        pending = (
            self.pending_anchor_memory_index,
            self.pending_anchor_receipt_id,
            self.pending_next_unit_index,
        )
        if any(item is None for item in pending) != all(item is None for item in pending):
            raise ValueError("decoder pending continuation fields must be all present or all absent")
        if self.pending_anchor_memory_index is not None and (
            self.pending_anchor_memory_index < 0 or self.pending_next_unit_index < 1
        ):
            raise ValueError("decoder pending continuation values are invalid")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < TRANSPORT_VOCAB_SIZE
            for item in self.transport_categories
        ):
            raise ValueError("decoder state contains an invalid transport category")
        if (
            isinstance(self.previous_category, bool)
            or not isinstance(self.previous_category, int)
            or not 0 <= self.previous_category <= TRANSPORT_VOCAB_SIZE + 2
        ):
            raise ValueError("decoder previous category is invalid")
        if self.work_units_completed != len(self.trace) or self.work_units_completed < 0:
            raise ValueError("decoder work count must equal its complete trace")
        if tuple(item.step_index for item in self.trace) != tuple(range(len(self.trace))):
            raise ValueError("decoder trace positions are discontinuous")
        eos_index = TRANSPORT_VOCAB_SIZE + 1
        traced_transport = tuple(item.category for item in self.trace if item.category != eos_index)
        if traced_transport != self.transport_categories:
            raise ValueError("decoder trace and exact transport stream disagree")
        eos_positions = [index for index, item in enumerate(self.trace) if item.category == eos_index]
        if eos_positions and eos_positions != [len(self.trace) - 1]:
            raise ValueError("decoder trace EOS must be unique and final")
        if self.terminated != bool(eos_positions) and (
            self.malformed_reason is None or not eos_positions
        ):
            raise ValueError("decoder termination state disagrees with EOS trace")
        expected_previous = (
            self.trace[-1].category if self.trace else TRANSPORT_VOCAB_SIZE + 2
        )
        if self.previous_category != expected_previous:
            raise ValueError("decoder previous category disagrees with its trace")
        if self.terminated and self.malformed_reason is not None:
            raise ValueError("decoder state cannot be terminated and malformed")
        if (self.terminated or self.malformed_reason is not None) and any(item is not None for item in pending):
            raise ValueError("finished decoder state cannot retain a pending continuation")
        object.__setattr__(self, "state_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": D64_DECODER_EXECUTION_STATE_SCHEMA,
            "architecture_id": self.architecture_id,
            "parameter_generation": self.parameter_generation,
            "core_id": self.core_id,
            "field_id": self.field_id,
            "tick_id": self.tick_id,
            "view_id": self.view_id,
            "rail_id": self.rail_id,
            "surface_id": self.surface_id,
            "memory_id": self.memory_id,
            "memory_segment_ids": list(self.memory_segment_ids),
            "pass_id": self.pass_id,
            "head": self.head,
            "hidden_shape": list(self.hidden_shape),
            "hidden_f32le_b64": self.hidden_f32le_b64,
            "previous_category": self.previous_category,
            "pending_anchor_memory_index": self.pending_anchor_memory_index,
            "pending_anchor_receipt_id": self.pending_anchor_receipt_id,
            "pending_next_unit_index": self.pending_next_unit_index,
            "transport_categories": list(self.transport_categories),
            "trace": [item.to_canonical_dict() for item in self.trace],
            "work_units_completed": self.work_units_completed,
            "terminated": self.terminated,
            "malformed_reason": self.malformed_reason,
        }
        if include_id:
            value["state_id"] = self.state_id
        return value

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict()) + b"\n"

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "DecoderExecutionState":
        try:
            value = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("decoder execution state is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("decoder execution state must be a JSON object")
        expected = {
            "schema",
            "architecture_id",
            "parameter_generation",
            "core_id",
            "field_id",
            "tick_id",
            "view_id",
            "rail_id",
            "surface_id",
            "memory_id",
            "memory_segment_ids",
            "pass_id",
            "head",
            "hidden_shape",
            "hidden_f32le_b64",
            "previous_category",
            "pending_anchor_memory_index",
            "pending_anchor_receipt_id",
            "pending_next_unit_index",
            "transport_categories",
            "trace",
            "work_units_completed",
            "terminated",
            "malformed_reason",
            "state_id",
        }
        if set(value) != expected or value.get("schema") != D64_DECODER_EXECUTION_STATE_SCHEMA:
            raise ValueError("decoder execution state schema or fields are invalid")
        state = cls(
            architecture_id=str(value["architecture_id"]),
            parameter_generation=str(value["parameter_generation"]),
            core_id=str(value["core_id"]),
            field_id=str(value["field_id"]),
            tick_id=int(value["tick_id"]),
            view_id=str(value["view_id"]),
            rail_id=str(value["rail_id"]),
            surface_id=str(value["surface_id"]),
            memory_id=str(value["memory_id"]),
            memory_segment_ids=tuple(str(item) for item in value["memory_segment_ids"]),
            pass_id=str(value["pass_id"]),
            head=int(value["head"]),
            hidden_shape=tuple(int(item) for item in value["hidden_shape"]),
            hidden_f32le_b64=str(value["hidden_f32le_b64"]),
            previous_category=int(value["previous_category"]),
            pending_anchor_memory_index=(
                None
                if value["pending_anchor_memory_index"] is None
                else int(value["pending_anchor_memory_index"])
            ),
            pending_anchor_receipt_id=(
                None
                if value["pending_anchor_receipt_id"] is None
                else str(value["pending_anchor_receipt_id"])
            ),
            pending_next_unit_index=(
                None
                if value["pending_next_unit_index"] is None
                else int(value["pending_next_unit_index"])
            ),
            transport_categories=tuple(int(item) for item in value["transport_categories"]),
            trace=tuple(DecoderTraceEvent.from_canonical_dict(item) for item in value["trace"]),
            work_units_completed=int(value["work_units_completed"]),
            terminated=bool(value["terminated"]),
            malformed_reason=(
                None if value["malformed_reason"] is None else str(value["malformed_reason"])
            ),
        )
        if state.state_id != value["state_id"]:
            raise ValueError("decoder execution state identity mismatch")
        return state

    def hidden_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        payload = base64.b64decode(self.hidden_f32le_b64.encode("ascii"), validate=True)
        hidden = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone().reshape(self.hidden_shape)
        if not torch.isfinite(hidden).all():
            raise ValueError("decoder hidden state contains non-finite values")
        return hidden.to(device=device, dtype=dtype)


@dataclass(frozen=True, slots=True)
class DecoderSliceResult:
    state: DecoderExecutionState
    text: str
    complete: bool
    malformed_reason: str | None


@dataclass(slots=True)
class CausalDecoderStep:
    hidden: torch.Tensor
    mixed_logits: torch.Tensor
    generated_logits: torch.Tensor
    position_logits: torch.Tensor
    generate_gate_logits: torch.Tensor


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
    """Configurable D64 core with load-bearing private Soul and typed heads."""

    def __init__(self, config: LivingReasoningCoreConfig | None = None) -> None:
        self.living_config = config or LivingReasoningCoreConfig()
        super().__init__(self.living_config.reader_config())
        cfg = self.living_config
        self.soul_codec = D64SoulCodec(cfg)

        # Replace the historical native-only decoder with the permanent
        # 351 transport categories + EMPTY + EOS.  BOS is input-only.
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
        self.decision_head = nn.Linear(cfg.d_model, len(ReasoningDecision))
        self.operation_head = nn.Linear(cfg.d_model, len(ReasoningOperationKind))
        self.region_head = nn.Linear(cfg.d_model, len(CANONICAL_REGION_ORDER))
        self.start_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.end_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.boundary_seed = nn.Parameter(torch.randn(cfg.d_model) * 0.02)

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

    def _initial_decoder_hidden(
        self,
        reader_state: torch.Tensor,
        head: int,
    ) -> torch.Tensor:
        if head not in (0, 1):
            raise ValueError("decoder head must be zero or one")
        summary = reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([head], device=self.device))
        return torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)

    def causal_decoder_step(
        self,
        *,
        previous_category: int,
        hidden: torch.Tensor,
        memory: AddressableMemory | None,
    ) -> CausalDecoderStep:
        """One shared recurrent step used by every receipt-enabled decode mode."""

        if (
            isinstance(previous_category, bool)
            or not isinstance(previous_category, int)
            or not 0 <= previous_category <= self.bos_index
        ):
            raise ValueError("previous decoder category is invalid")
        expected_hidden = (1, 1, self.living_config.d_model)
        if tuple(hidden.shape) != expected_hidden:
            raise ValueError(f"decoder hidden state must have shape {expected_hidden}")
        token = torch.tensor(
            [[previous_category]],
            dtype=torch.long,
            device=self.device,
        )
        decoded, next_hidden = self.decoder(self.decoder_embedding(token), hidden)
        mixed_logits, alignment = self._decoder_logits(
            decoded[:, -1:],
            memory,
            return_alignment=True,
        )
        return CausalDecoderStep(
            hidden=next_hidden,
            mixed_logits=mixed_logits,
            generated_logits=alignment["generated_logits"],
            position_logits=alignment["position_logits"],
            generate_gate_logits=alignment["generate_gate_logits"],
        )

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
        if not self.living_config.receipt_continuation:
            return super().decode_teacher(
                reader_state,
                target,
                head,
                memory=memory,
                return_alignment=return_alignment,
            )
        targets = self._target_indices(target).unsqueeze(0)
        hidden = self._initial_decoder_hidden(reader_state, head)
        previous = self.bos_index
        logits: list[torch.Tensor] = []
        generated_logits: list[torch.Tensor] = []
        position_logits: list[torch.Tensor] = []
        gate_logits: list[torch.Tensor] = []
        for position in range(targets.shape[1]):
            step = self.causal_decoder_step(
                previous_category=previous,
                hidden=hidden,
                memory=memory,
            )
            hidden = step.hidden
            logits.append(step.mixed_logits)
            generated_logits.append(step.generated_logits)
            position_logits.append(step.position_logits)
            gate_logits.append(step.generate_gate_logits)
            previous = int(targets[0, position].item())
        joined = torch.cat(logits, dim=1)
        if not return_alignment:
            return joined, targets
        return joined, targets, {
            "generated_logits": torch.cat(generated_logits, dim=1),
            "position_logits": torch.cat(position_logits, dim=1),
            "generate_gate_logits": torch.cat(gate_logits, dim=1),
        }

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
        if not self.living_config.receipt_continuation:
            return super().decode_scheduled(
                reader_state,
                target,
                head,
                teacher_forcing_ratio,
                memory=memory,
                return_alignment=return_alignment,
            )
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between zero and one")
        targets = self._target_indices(target).unsqueeze(0)
        hidden = self._initial_decoder_hidden(reader_state, head)
        previous = self.bos_index
        teacher_choices = (
            torch.rand(max(0, targets.shape[1] - 1), device=self.device)
            < teacher_forcing_ratio
        ).tolist()
        logits: list[torch.Tensor] = []
        generated_logits: list[torch.Tensor] = []
        position_logits: list[torch.Tensor] = []
        gate_logits: list[torch.Tensor] = []
        for position in range(targets.shape[1]):
            step = self.causal_decoder_step(
                previous_category=previous,
                hidden=hidden,
                memory=memory,
            )
            hidden = step.hidden
            logits.append(step.mixed_logits)
            generated_logits.append(step.generated_logits)
            position_logits.append(step.position_logits)
            gate_logits.append(step.generate_gate_logits)
            if position + 1 < targets.shape[1]:
                previous = (
                    int(targets[0, position].item())
                    if bool(teacher_choices[position])
                    else int(step.mixed_logits[0, 0].argmax(dim=-1).detach().item())
                )
        joined = torch.cat(logits, dim=1)
        if not return_alignment:
            return joined, targets
        return joined, targets, {
            "generated_logits": torch.cat(generated_logits, dim=1),
            "position_logits": torch.cat(position_logits, dim=1),
            "generate_gate_logits": torch.cat(gate_logits, dim=1),
        }

    def alignment_supervision(
        self,
        *,
        target_text: str,
        memory: AddressableMemory,
        decoder_alignment: Mapping[str, torch.Tensor],
        specification: Mapping[str, Any],
        position_reduction: str = "mean",
    ) -> dict[str, Any]:
        """Supervise exact source positions across Unicode transport expansion.

        The inherited V6 helper assumes one decoder category per character.
        Living tissue uses one category per native scalar or strict UTF-8 byte,
        so a single canonical scalar may occupy one to four exact rail cells.
        This override preserves scalar-addressed curriculum contracts while
        supervising every expanded transport position in order.
        """

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
            raise ValueError("decoder alignment length does not match Unicode transport plus EOS")

        zero = gate_logits.sum() * 0.0
        position_losses: list[torch.Tensor] = []
        copy_gate_losses: list[torch.Tensor] = []
        eos_gate_losses: list[torch.Tensor] = []
        position_correct = copy_gate_correct = supervised_copy_positions = 0
        deterministic_continuation_positions = 0
        learned_decision_mask = torch.ones(
            (1, transport_count + 1),
            dtype=torch.bool,
            device=self.device,
        )
        region_to_id = {
            region.value: index for index, region in enumerate(CANONICAL_REGION_ORDER)
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
            if hashlib.sha256(target_fragment.encode("utf-8")).hexdigest() != segment["text_sha256"]:
                raise ValueError("alignment target fragment hash mismatch")

            for scalar_offset, source_position in enumerate(range(source_start, source_end)):
                target_scalar_position = target_start + scalar_offset
                expected_tokens = tuple(encode_unicode_text(target_text[target_scalar_position]))
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
                    raise ValueError("alignment source transport does not equal supervised target")
                decoder_start = scalar_offsets[target_scalar_position]
                for token_offset, memory_index in enumerate(matches):
                    target_position = decoder_start + token_offset
                    if self.living_config.receipt_continuation and token_offset > 0:
                        learned_decision_mask[:, target_position] = False
                        deterministic_continuation_positions += 1
                        continue
                    expected_source = torch.tensor(
                        [int(memory_index.item())], dtype=torch.long, device=self.device
                    )
                    source_logits = position_logits[:, target_position, :]
                    position_losses.append(F.cross_entropy(source_logits, expected_source))
                    predicted_source = int(source_logits[0].argmax(dim=-1).item())
                    position_correct += int(predicted_source == int(memory_index.item()))
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
                F.binary_cross_entropy_with_logits(gate_logits[:, transport_count], eos_target)
            )
            eos_gate_correct += int(
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
        eos_gate_loss = torch.stack(eos_gate_losses).mean() if eos_gate_losses else zero
        gate_losses = copy_gate_losses + eos_gate_losses
        gate_loss = torch.stack(gate_losses).mean() if gate_losses else zero
        gate_count = supervised_copy_positions + int(eos_supervised)
        gate_correct = copy_gate_correct + eos_gate_correct
        return {
            "position_loss": position_loss,
            "gate_loss": gate_loss,
            "copy_gate_loss": copy_gate_loss,
            "eos_gate_loss": eos_gate_loss,
            "copy_positions": supervised_copy_positions,
            "deterministic_continuation_positions": deterministic_continuation_positions,
            "learned_decision_mask": learned_decision_mask,
            "position_correct": position_correct,
            "gate_supervised_positions": gate_count,
            "gate_correct": gate_correct,
            "copy_gate_correct": copy_gate_correct,
            "eos_gate_supervised_positions": int(eos_supervised),
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
                eos_gate_correct / int(eos_supervised) if eos_supervised else 1.0
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
            raise ValueError("private Soul parameter generation does not match the living core")
        if phase not in _PHASE_TO_ID:
            raise ValueError(f"unsupported living reasoning phase {phase!r}")
        ablated = {item if isinstance(item, SoulTemperature) else SoulTemperature(item) for item in ablate_temperatures}
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
            contribution = self.soul_projection[temperature.value](decoded) * gates[index]
            state = state + contribution
            contribution_l2 += float(contribution.detach().norm().item())
            decoded_count += 1
        phase_vector = self.phase_embedding(torch.tensor(_PHASE_TO_ID[phase], device=self.device)).reshape(1, 1, -1)
        state = state + phase_vector
        return state, {
            "decoded_soul_layers": float(decoded_count),
            "soul_contribution_l2": contribution_l2,
            "inhaled_state_l2": float(state.detach().norm().item()),
        }

    def _proposal_compiled(self, text: str, index: int) -> CompiledD64Field:
        snapshot = SharedFieldSnapshot.from_texts(
            {
                LogicalRegion.ADVISOR_INPUT: text,
            },
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
        summary = state.mean(dim=1)
        telemetry = {
            **telemetry,
            "exhaled_state_l2": float(state.detach().norm().item()),
            "state_change_l2": float((state.detach() - self.initial_state.unsqueeze(0)).norm().item()),
            "canonical_pages": float(canonical_coverage.page_count),
            "proposal_pages": float(sum(item.page_count for item in proposal_coverages)),
        }
        complete_memory = _join_memory(memories)
        resolved_view_id = canonical.rail_id if view_id is None else view_id
        if not isinstance(resolved_view_id, str) or not resolved_view_id:
            raise ValueError("living reasoning view_id must be non-empty")
        surface_id = canonical_sha256(
            {
                "schema": "axon-living-d64-decoder-surface-v1",
                "field_id": canonical.source_field_id,
                "tick_id": canonical.source_tick_id,
                "view_id": resolved_view_id,
                "rail_id": canonical.rail_id,
                "complete_memory_id": complete_memory.memory_id,
                "phase": phase,
            }
        )
        return LivingReasoningForward(
            reader_state=state,
            canonical_memory=canonical_memory,
            complete_memory=complete_memory,
            decision_logits=self.decision_head(summary),
            operation_logits=self.operation_head(summary),
            region_logits=self.region_head(summary),
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

    def boundary_logits(
        self,
        output: LivingReasoningForward,
        region: LogicalRegion,
    ) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
        memory = output.canonical_memory
        region_id = CANONICAL_REGION_ORDER.index(region)
        mask = (memory.region_ids[0] == region_id) & memory.region_positions[0].ge(0)
        positions = tuple(sorted(set(int(item) for item in memory.region_positions[0, mask].tolist())))
        candidates = tuple(
            sorted({0, *(position for position in positions), *(position + 1 for position in positions)})
        )
        representations: list[torch.Tensor] = []
        for candidate in candidates:
            adjacent = mask & (
                (memory.region_positions[0] == candidate) | (memory.region_positions[0] == candidate - 1)
            )
            if bool(adjacent.any()):
                representation = memory.states[0, adjacent].mean(dim=0)
            else:
                representation = self.boundary_seed + self.region_embedding.weight[region_id]
            representations.append(representation)
        keys = torch.stack(representations, dim=0)
        summary = output.reader_state.mean(dim=1)
        scale = math.sqrt(self.living_config.d_model)
        start = torch.matmul(self.start_query(summary), keys.T) / scale
        end = torch.matmul(self.end_query(summary), keys.T) / scale
        return candidates, start, end

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
            updates=(self.soul_codec.encode(exhaled_state, SoulTemperature.HOT),),
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
        """Run the service phase order without a training-only Soul shortcut.

        Persisted bytes form each next-phase inhale, intentionally making the
        phase boundary a truncated-gradient boundary exactly like service.
        """

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
                    "proposal_text_sha256": [canonical_sha256({"text": text}) for text in proposal_texts],
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

    @staticmethod
    def _hidden_f32le_b64(hidden: torch.Tensor) -> str:
        array = hidden.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        return base64.b64encode(array.astype("<f4", copy=False).tobytes(order="C")).decode(
            "ascii"
        )

    @staticmethod
    def _memory_segment_ids(memory: AddressableMemory) -> tuple[str, ...]:
        return tuple(segment.segment_id for segment in memory.segments)

    def initial_decoder_execution_state(
        self,
        output: LivingReasoningForward,
        *,
        head: int = 1,
    ) -> DecoderExecutionState:
        if not self.living_config.receipt_continuation:
            raise ValueError("decoder execution state requires the opt-in receipt architecture")
        hidden = self._initial_decoder_hidden(output.reader_state, head)
        return DecoderExecutionState(
            architecture_id=self.architecture_id,
            parameter_generation=output.parameter_generation,
            core_id=output.core_id,
            field_id=output.source_field_id,
            tick_id=output.source_tick_id,
            view_id=output.view_id,
            rail_id=output.rail_id,
            surface_id=output.surface_id,
            memory_id=output.complete_memory.memory_id,
            memory_segment_ids=self._memory_segment_ids(output.complete_memory),
            pass_id=output.phase,
            head=head,
            hidden_shape=tuple(hidden.shape),
            hidden_f32le_b64=self._hidden_f32le_b64(hidden),
            previous_category=self.bos_index,
            pending_anchor_memory_index=None,
            pending_anchor_receipt_id=None,
            pending_next_unit_index=None,
            transport_categories=(),
            trace=(),
            work_units_completed=0,
        )

    def _validate_decoder_execution_bindings(
        self,
        output: LivingReasoningForward,
        state: DecoderExecutionState,
    ) -> None:
        expected = (
            self.architecture_id,
            output.parameter_generation,
            output.core_id,
            output.source_field_id,
            output.source_tick_id,
            output.view_id,
            output.rail_id,
            output.surface_id,
            output.complete_memory.memory_id,
            self._memory_segment_ids(output.complete_memory),
            output.phase,
        )
        observed = (
            state.architecture_id,
            state.parameter_generation,
            state.core_id,
            state.field_id,
            state.tick_id,
            state.view_id,
            state.rail_id,
            state.surface_id,
            state.memory_id,
            state.memory_segment_ids,
            state.pass_id,
        )
        if observed != expected:
            raise ValueError("decoder execution state is stale or bound to another surface")
        if state.head not in (0, 1):
            raise ValueError("decoder execution state head is invalid")
        memory = output.complete_memory
        for event in state.trace:
            if event.memory_index is None:
                continue
            receipt = memory.receipt(event.memory_index)
            if receipt.receipt_id != event.receipt_id:
                raise ValueError("decoder trace receipt is stale or substituted")
        if state.pending_anchor_memory_index is not None:
            anchor = memory.receipt(state.pending_anchor_memory_index)
            if (
                anchor.receipt_id != state.pending_anchor_receipt_id
                or anchor.address.transport_unit_index != 0
                or anchor.address.transport_unit_count <= state.pending_next_unit_index
            ):
                raise ValueError("decoder pending continuation receipt is stale or malformed")

    def _execution_state_from_hidden(
        self,
        prior: DecoderExecutionState,
        *,
        hidden: torch.Tensor,
        previous_category: int,
        pending_anchor_memory_index: int | None,
        pending_anchor_receipt_id: str | None,
        pending_next_unit_index: int | None,
        transport_categories: tuple[int, ...],
        trace: tuple[DecoderTraceEvent, ...],
        terminated: bool = False,
        malformed_reason: str | None = None,
    ) -> DecoderExecutionState:
        return DecoderExecutionState(
            architecture_id=prior.architecture_id,
            parameter_generation=prior.parameter_generation,
            core_id=prior.core_id,
            field_id=prior.field_id,
            tick_id=prior.tick_id,
            view_id=prior.view_id,
            rail_id=prior.rail_id,
            surface_id=prior.surface_id,
            memory_id=prior.memory_id,
            memory_segment_ids=prior.memory_segment_ids,
            pass_id=prior.pass_id,
            head=prior.head,
            hidden_shape=tuple(hidden.shape),
            hidden_f32le_b64=self._hidden_f32le_b64(hidden),
            previous_category=previous_category,
            pending_anchor_memory_index=pending_anchor_memory_index,
            pending_anchor_receipt_id=pending_anchor_receipt_id,
            pending_next_unit_index=pending_next_unit_index,
            transport_categories=transport_categories,
            trace=trace,
            work_units_completed=len(trace),
            terminated=terminated,
            malformed_reason=malformed_reason,
        )

    def _select_learned_emission(
        self,
        step: CausalDecoderStep,
        memory: AddressableMemory,
    ) -> tuple[DecoderEmissionRoute, int, int | None, MemoryCellReceipt | None]:
        gate = float(step.generate_gate_logits[0, 0].item())
        if gate >= 0.0:
            category = int(step.generated_logits[0, 0].argmax(dim=-1).item())
            return DecoderEmissionRoute.LEARNED_GENERATE, category, None, None
        if step.position_logits.shape[-1] != memory.states.shape[1]:
            raise ValueError("learned pointer surface does not match addressable memory")
        memory_index = int(step.position_logits[0, 0].argmax(dim=-1).item())
        receipt = memory.receipt(memory_index)
        category = int(memory.char_indices[0, memory_index].item())
        if category != receipt.address.transport_token_id:
            raise ValueError("learned copy category disagrees with exact compiler receipt")
        return DecoderEmissionRoute.LEARNED_COPY_ANCHOR, category, memory_index, receipt

    @torch.no_grad()
    def advance_decoder_execution(
        self,
        output: LivingReasoningForward,
        state: DecoderExecutionState,
        *,
        work_units: int,
    ) -> DecoderSliceResult:
        """Advance one portable receipt-governed decoder state by a renewable slice."""

        if not self.living_config.receipt_continuation:
            raise ValueError("receipt-governed execution requires the opt-in architecture")
        if isinstance(work_units, bool) or not isinstance(work_units, int) or work_units < 1:
            raise ValueError("work_units must be a positive integer")
        self._validate_decoder_execution_bindings(output, state)
        if state.terminated or state.malformed_reason is not None:
            raise ValueError("finished decoder execution cannot be advanced")

        memory = output.complete_memory
        hidden = state.hidden_tensor(device=self.device, dtype=self.initial_state.dtype)
        previous = state.previous_category
        pending_anchor_index = state.pending_anchor_memory_index
        pending_anchor_id = state.pending_anchor_receipt_id
        pending_next_unit = state.pending_next_unit_index
        transport = state.transport_categories
        trace = state.trace

        for _ in range(work_units):
            step = self.causal_decoder_step(
                previous_category=previous,
                hidden=hidden,
                memory=memory,
            )
            try:
                if pending_anchor_index is not None:
                    anchor = memory.receipt(pending_anchor_index)
                    if (
                        anchor.receipt_id != pending_anchor_id
                        or anchor.address.transport_unit_index != 0
                        or pending_next_unit is None
                        or pending_next_unit >= anchor.address.transport_unit_count
                    ):
                        raise ValueError("pending scalar anchor is stale or malformed")
                    memory_index = memory.continuation_index(anchor, pending_next_unit)
                    receipt = memory.receipt(memory_index)
                    if receipt.segment_id != anchor.segment_id:
                        raise ValueError("receipt continuation crossed a memory segment")
                    category = int(receipt.address.transport_token_id)
                    route = DecoderEmissionRoute.DETERMINISTIC_RECEIPT_CONTINUATION
                    next_unit = pending_next_unit + 1
                    if next_unit >= anchor.address.transport_unit_count:
                        next_anchor_index = None
                        next_anchor_id = None
                        next_unit = None
                    else:
                        next_anchor_index = pending_anchor_index
                        next_anchor_id = pending_anchor_id
                    anchor_receipt_id = anchor.receipt_id
                else:
                    route, category, memory_index, receipt = self._select_learned_emission(
                        step,
                        memory,
                    )
                    if route is DecoderEmissionRoute.LEARNED_GENERATE:
                        next_anchor_index = None
                        next_anchor_id = None
                        next_unit = None
                        anchor_receipt_id = None
                    else:
                        if receipt is None or memory_index is None:
                            raise ValueError("learned copy lacks its exact compiler receipt")
                        address = receipt.address
                        if address.transport_unit_index != 0:
                            raise ValueError("learned copy selected a mid-scalar transport unit")
                        next_anchor_index = (
                            memory_index if address.transport_unit_count > 1 else None
                        )
                        next_anchor_id = (
                            receipt.receipt_id if address.transport_unit_count > 1 else None
                        )
                        next_unit = 1 if address.transport_unit_count > 1 else None
                        anchor_receipt_id = receipt.receipt_id
            except (IndexError, ValueError) as exc:
                malformed = self._execution_state_from_hidden(
                    state,
                    hidden=hidden,
                    previous_category=previous,
                    pending_anchor_memory_index=None,
                    pending_anchor_receipt_id=None,
                    pending_next_unit_index=None,
                    transport_categories=transport,
                    trace=trace,
                    malformed_reason=str(exc),
                )
                return DecoderSliceResult(
                    state=malformed,
                    text="",
                    complete=False,
                    malformed_reason=str(exc),
                )

            if category == self.eos_index:
                if route is not DecoderEmissionRoute.LEARNED_GENERATE:
                    reason = "only learned generation may emit EOS"
                    malformed = self._execution_state_from_hidden(
                        state,
                        hidden=hidden,
                        previous_category=previous,
                        pending_anchor_memory_index=None,
                        pending_anchor_receipt_id=None,
                        pending_next_unit_index=None,
                        transport_categories=transport,
                        trace=trace,
                        malformed_reason=reason,
                    )
                    return DecoderSliceResult(malformed, "", False, reason)
                event = DecoderTraceEvent(
                    step_index=len(trace),
                    route=route,
                    category=category,
                    memory_index=None,
                    receipt_id=None,
                    anchor_receipt_id=None,
                    transport_unit_index=None,
                    transport_unit_count=None,
                )
                trace = (*trace, event)
                finished = self._execution_state_from_hidden(
                    state,
                    hidden=step.hidden,
                    previous_category=category,
                    pending_anchor_memory_index=None,
                    pending_anchor_receipt_id=None,
                    pending_next_unit_index=None,
                    transport_categories=transport,
                    trace=trace,
                    terminated=True,
                )
                try:
                    text = decode_unicode_tokens(transport)
                except ValueError:
                    reason = "decoder terminated with malformed Unicode transport"
                    malformed = self._execution_state_from_hidden(
                        state,
                        hidden=step.hidden,
                        previous_category=category,
                        pending_anchor_memory_index=None,
                        pending_anchor_receipt_id=None,
                        pending_next_unit_index=None,
                        transport_categories=transport,
                        trace=trace,
                        malformed_reason=reason,
                    )
                    return DecoderSliceResult(malformed, "", False, reason)
                return DecoderSliceResult(finished, text, True, None)

            if not 0 <= category < TRANSPORT_VOCAB_SIZE:
                reason = "decoder emitted EMPTY or an invalid transport category"
                malformed = self._execution_state_from_hidden(
                    state,
                    hidden=hidden,
                    previous_category=previous,
                    pending_anchor_memory_index=None,
                    pending_anchor_receipt_id=None,
                    pending_next_unit_index=None,
                    transport_categories=transport,
                    trace=trace,
                    malformed_reason=reason,
                )
                return DecoderSliceResult(malformed, "", False, reason)

            if route is DecoderEmissionRoute.LEARNED_GENERATE:
                receipt_id = None
                unit_index = None
                unit_count = None
            else:
                assert memory_index is not None and receipt is not None
                receipt_id = receipt.receipt_id
                unit_index = receipt.address.transport_unit_index
                unit_count = receipt.address.transport_unit_count
            event = DecoderTraceEvent(
                step_index=len(trace),
                route=route,
                category=category,
                memory_index=memory_index,
                receipt_id=receipt_id,
                anchor_receipt_id=anchor_receipt_id,
                transport_unit_index=unit_index,
                transport_unit_count=unit_count,
            )
            trace = (*trace, event)
            transport = (*transport, category)
            hidden = step.hidden
            previous = category
            pending_anchor_index = next_anchor_index
            pending_anchor_id = next_anchor_id
            pending_next_unit = next_unit

        incomplete = self._execution_state_from_hidden(
            state,
            hidden=hidden,
            previous_category=previous,
            pending_anchor_memory_index=pending_anchor_index,
            pending_anchor_receipt_id=pending_anchor_id,
            pending_next_unit_index=pending_next_unit,
            transport_categories=transport,
            trace=trace,
        )
        return DecoderSliceResult(incomplete, "", False, None)

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
        state: DecoderExecutionState | None = None,
    ) -> Iterator[tuple[str, bool]]:
        """Yield control at renewable slice boundaries and preserve decoder state.

        Every nonterminal yield is explicitly incomplete. A caller may retain
        this iterator and resume it without a total output bound. Durable
        cross-process continuation is a Heart-host responsibility and remains
        a serving gate; the core never converts slice exhaustion into success.
        """

        work_slice = (
            capacity_policy().integer("reasoning.emission_work_slice_transport_units")
            if work_units is None
            else int(work_units)
        )
        if work_slice < 1:
            raise ValueError("work_units must be positive")
        if self.living_config.receipt_continuation:
            current = (
                self.initial_decoder_execution_state(output)
                if state is None
                else state
            )
            while True:
                result = self.advance_decoder_execution(
                    output,
                    current,
                    work_units=work_slice,
                )
                yield result.text, result.complete
                if result.complete or result.malformed_reason is not None:
                    return
                current = result.state
        elif state is not None:
            raise ValueError("legacy decoder cannot consume receipt execution state")
        else:
            yield from self._iter_decode_transport_legacy(output, work_slice=work_slice)

    def _iter_decode_transport_legacy(
        self,
        output: LivingReasoningForward,
        *,
        work_slice: int,
    ) -> Iterator[tuple[str, bool]]:
        """Bit-compatible historical decoder used only when the conduit is disabled."""

        summary = output.reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([1], device=self.device))
        hidden = torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
        token = torch.full((1, 1), self.bos_index, dtype=torch.long, device=self.device)
        transport: list[int] = []
        with torch.no_grad():
            while True:
                for _ in range(work_slice):
                    decoded, hidden = self.decoder(self.decoder_embedding(token), hidden)
                    logits = self._decoder_logits(
                        decoded[:, -1:],
                        output.complete_memory,
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
                        [[category]],
                        dtype=torch.long,
                        device=self.device,
                    )
                yield "", False

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
        """Conservative runtime adapter; untrained candidates are never registered live."""

        output = self.forward_request(request)
        transition = self.exhale_transition(
            before=request.soul,
            exhaled_state=output.exhaled_state,
            tick_uid=request.image.identity.tick_uid,
            request_id=request.request_id,
            phase=request.phase,
        )
        decision = tuple(ReasoningDecision)[int(output.decision_logits.argmax(dim=-1).item())]
        if decision is not ReasoningDecision.DELTA:
            detail = CategoricalTextFrame.from_text(
                "model no-op" if decision is ReasoningDecision.NO_OP else "model abstained",
                d_model=64,
            )
            emission = ReasoningEmission(
                base_field_id=request.image.identity.base_field_id,
                base_tick_id=request.image.identity.base_tick_id,
                author_core_id=request.descriptor.core_id,
                pass_id=request.phase,
                rail_d_model=64,
                decision=decision,
                detail=detail,
            )
            return ReasoningPassResult(emission, transition)

        allowed = request.descriptor.authority_grant().governed_regions
        region_logits = output.region_logits.clone()
        for index, region in enumerate(CANONICAL_REGION_ORDER):
            if region not in allowed:
                region_logits[:, index] = torch.finfo(region_logits.dtype).min
        region = CANONICAL_REGION_ORDER[int(region_logits.argmax(dim=-1).item())]
        operation = tuple(ReasoningOperationKind)[int(output.operation_logits.argmax(dim=-1).item())]
        candidates, start_logits, end_logits = self.boundary_logits(output, region)
        start = candidates[int(start_logits.argmax(dim=-1).item())]
        end = candidates[int(end_logits.argmax(dim=-1).item())]
        if operation is ReasoningOperationKind.INSERT:
            end = start
        text, terminated = self.decode_transport_greedy(output)
        invalid = (
            (operation is ReasoningOperationKind.DELETE and end <= start)
            or (operation is ReasoningOperationKind.REPLACE and (end < start or not text))
            or (operation is ReasoningOperationKind.INSERT and not text)
            or (operation is not ReasoningOperationKind.DELETE and not terminated)
        )
        if invalid:
            emission = ReasoningEmission(
                base_field_id=request.image.identity.base_field_id,
                base_tick_id=request.image.identity.base_tick_id,
                author_core_id=request.descriptor.core_id,
                pass_id=request.phase,
                rail_d_model=64,
                decision=ReasoningDecision.ABSTAIN,
                detail=CategoricalTextFrame.from_text(
                    "malformed neural delta rejected",
                    d_model=64,
                ),
            )
            return ReasoningPassResult(emission, transition)
        payload = CategoricalTextFrame.from_text(
            "" if operation is ReasoningOperationKind.DELETE else text,
            d_model=64,
        )
        emission = ReasoningEmission(
            base_field_id=request.image.identity.base_field_id,
            base_tick_id=request.image.identity.base_tick_id,
            author_core_id=request.descriptor.core_id,
            pass_id=request.phase,
            rail_d_model=64,
            decision=ReasoningDecision.DELTA,
            operations=(
                ReasoningOperationEmission(
                    kind=operation,
                    region=region,
                    start=start,
                    end=end,
                    payload=payload,
                    provenance=f"living-core:{self.architecture_id}",
                ),
            ),
        )
        return ReasoningPassResult(emission, transition)

    def architecture_report(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            **self.living_config.to_canonical_dict(),
            "parameter_count": parameters,
            "trainable_parameter_count": trainable,
            "parameter_bytes_fp32": parameters * 4,
            "parameter_bytes_fp16": parameters * 2,
        }


def migrate_legacy_weights_to_receipt_variant(
    source: LivingReasoningCoreD64,
    target: LivingReasoningCoreD64,
    *,
    source_parameter_generation: str,
    target_parameter_generation: str,
) -> D64ReceiptMigrationReceipt:
    """Strictly copy v1 tensors into a fresh v2 candidate and return exact lineage."""

    if source.living_config.receipt_continuation:
        raise ValueError("migration source must be the legacy receipt-disabled architecture")
    if not target.living_config.receipt_continuation:
        raise ValueError("migration target must be the opt-in receipt architecture")
    if not source_parameter_generation or not target_parameter_generation:
        raise ValueError("migration parameter generations must be non-empty")
    source_state = source.state_dict()
    target_state = target.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError("migration tensor names differ; forgiving load is forbidden")
    for name in source_state:
        if source_state[name].shape != target_state[name].shape:
            raise ValueError(f"migration tensor shape differs for {name}")
        if source_state[name].dtype != target_state[name].dtype:
            raise ValueError(f"migration tensor dtype differs for {name}")
    target.load_state_dict(source_state, strict=True)

    copied: list[TensorCopyReceipt] = []
    for name in sorted(source_state):
        source_tensor = source_state[name].detach().to(device="cpu").contiguous()
        target_tensor = target.state_dict()[name].detach().to(device="cpu").contiguous()
        source_digest = hashlib.sha256(
            source_tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
        ).hexdigest()
        target_digest = hashlib.sha256(
            target_tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
        ).hexdigest()
        copied.append(
            TensorCopyReceipt(
                name=name,
                shape=tuple(source_tensor.shape),
                dtype=str(source_tensor.dtype),
                source_sha256=source_digest,
                target_sha256=target_digest,
            )
        )
    return D64ReceiptMigrationReceipt(
        source_architecture_id=source.architecture_id,
        target_architecture_id=target.architecture_id,
        source_parameter_generation=source_parameter_generation,
        target_parameter_generation=target_parameter_generation,
        copied_tensors=tuple(copied),
        new_state_fields=tuple(DecoderExecutionState.__dataclass_fields__),
        initialization="strict_exact_tensor_copy; no new learned parameters",
        disabled_route_equivalence=(
            "verified: receipt_continuation=false dispatches the unchanged v1 teacher, "
            "scheduled, and greedy decoder paths"
        ),
    )


def candidate_a_config(**overrides: Any) -> LivingReasoningCoreConfig:
    values = asdict(LivingReasoningCoreConfig())
    values.pop("architecture_id", None)
    values.update(overrides)
    return LivingReasoningCoreConfig(**values)


__all__ = [
    "D64_DECODER_EXECUTION_STATE_SCHEMA",
    "D64_DECODER_TRACE_SCHEMA",
    "D64_RECEIPT_MIGRATION_SCHEMA",
    "D64_SOUL_CODEC_VERSION",
    "D64_SOUL_MEDIA_TYPE",
    "LIVING_REASONING_ARCHITECTURE_SCHEMA",
    "LIVING_REASONING_RECEIPT_ARCHITECTURE_SCHEMA",
    "CausalDecoderStep",
    "CausalLivingUnroll",
    "D64ReceiptMigrationReceipt",
    "D64SoulCodec",
    "DecoderEmissionRoute",
    "DecoderExecutionState",
    "DecoderSliceResult",
    "LivingReasoningCoreConfig",
    "LivingReasoningCoreD64",
    "LivingReasoningForward",
    "TensorCopyReceipt",
    "candidate_a_config",
    "migrate_legacy_weights_to_receipt_variant",
]
