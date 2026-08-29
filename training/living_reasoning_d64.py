"""Soul-conditioned, complete-field D64 reasoning tissue.

This module is the first neural implementation of the permanent runtime
contract.  It inhales one core's opaque private Soul before perception, carries
that recurrent state across every eligible canonical and proposal page, emits
typed-output logits, and exhales the resulting recurrent state as an opaque HOT
Soul successor.  Pages are bounded compute units, never a context limit.
"""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from runtime.field import (
    CANONICAL_REGION_ORDER,
    CompiledD64Field,
    D64FieldCompiler,
    LogicalRegion,
    SharedFieldSnapshot,
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
from substrate import (
    TRANSPORT_VOCAB_SIZE,
    decode_unicode_tokens,
    encode_unicode_text,
)

from .complete_field_64d import AddressableMemory, CompleteField64D, CoverageManifest, ReaderConfig

LIVING_REASONING_ARCHITECTURE_SCHEMA = "axon-living-reasoning-architecture-v1"
D64_SOUL_CODEC_VERSION = "axon-d64-recurrent-soul-codec-v1"
D64_SOUL_MEDIA_TYPE = "application/x-axon-d64-recurrent-state"
_SOUL_MAGIC = b"AXSLD641"
_SOUL_HEADER = struct.Struct("<8sII")
_PHASE_TO_ID = {"first": 0, "refined": 1, "consolidated": 2}


@dataclass(frozen=True, slots=True)
class LivingReasoningCoreConfig:
    d_model: int = 64
    n_heads: int = 1
    n_layers: int = 2
    ffn_dim: int = 131_072
    state_tokens: int = 4
    page_size: int = 32
    dropout: float = 0.05
    inference_budget_transport_units: int = 512
    soul_codec_version: str = D64_SOUL_CODEC_VERSION
    lift_seed: int = 7
    architecture_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "n_heads",
            "n_layers",
            "ffn_dim",
            "state_tokens",
            "page_size",
            "inference_budget_transport_units",
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
        object.__setattr__(
            self,
            "architecture_id",
            "living-d64-" + canonical_sha256(self.to_canonical_dict(False))[:24],
        )

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": LIVING_REASONING_ARCHITECTURE_SCHEMA,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ffn_dim": self.ffn_dim,
            "state_tokens": self.state_tokens,
            "page_size": self.page_size,
            "dropout": float(self.dropout),
            "inference_budget_transport_units": self.inference_budget_transport_units,
            "soul_codec_version": self.soul_codec_version,
            "lift_seed": self.lift_seed,
        }
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
            inference_budget_chars=self.inference_budget_transport_units,
            lift_seed=self.lift_seed,
        )


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


def _join_memory(items: Iterable[AddressableMemory]) -> AddressableMemory:
    memories = tuple(items)
    if not memories:
        raise ValueError("at least one addressable memory is required")
    return AddressableMemory(
        states=torch.cat([item.states for item in memories], dim=1),
        char_indices=torch.cat([item.char_indices for item in memories], dim=1),
        region_ids=torch.cat([item.region_ids for item in memories], dim=1),
        region_positions=torch.cat([item.region_positions for item in memories], dim=1),
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
            contribution = self.soul_projection[temperature.value](decoded) * gates[index]
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
        )
        memories = [canonical_memory]
        proposal_coverages: list[CoverageManifest] = []
        for index, text in enumerate(proposal_texts):
            compiled = self._proposal_compiled(text, index)
            state, memory, coverage = self.read_compiled_with_memory(
                compiled,
                initial_state=state,
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
        return LivingReasoningForward(
            reader_state=state,
            canonical_memory=canonical_memory,
            complete_memory=_join_memory(memories),
            decision_logits=self.decision_head(summary),
            operation_logits=self.operation_head(summary),
            region_logits=self.region_head(summary),
            inhaled_state=inhaled_state,
            exhaled_state=state,
            canonical_coverage=canonical_coverage,
            proposal_coverages=tuple(proposal_coverages),
            soul_telemetry=telemetry,
        )

    def boundary_logits(
        self,
        output: LivingReasoningForward,
        region: LogicalRegion,
    ) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
        memory = output.canonical_memory
        region_id = CANONICAL_REGION_ORDER.index(region)
        mask = (memory.region_ids[0] == region_id) & memory.region_positions[0].ge(0)
        positions = tuple(
            sorted(set(int(item) for item in memory.region_positions[0, mask].tolist()))
        )
        candidates = tuple(sorted({0, *(position for position in positions), *(position + 1 for position in positions)}))
        representations: list[torch.Tensor] = []
        for candidate in candidates:
            adjacent = mask & (
                (memory.region_positions[0] == candidate)
                | (memory.region_positions[0] == candidate - 1)
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
    ) -> tuple[str, bool]:
        limit = self.living_config.inference_budget_transport_units
        summary = output.reader_state.mean(dim=1)
        head_vec = self.decoder_head_embedding(torch.tensor([1], device=self.device))
        hidden = torch.tanh(self.decoder_init(torch.cat((summary, head_vec), dim=-1))).unsqueeze(0)
        token = torch.full((1, 1), self.bos_index, dtype=torch.long, device=self.device)
        transport: list[int] = []
        for _ in range(limit + 1):
            decoded, hidden = self.decoder(self.decoder_embedding(token), hidden)
            logits = self._decoder_logits(decoded[:, -1:], output.complete_memory).squeeze(1)
            category = int(logits.argmax(dim=-1).item())
            if category == self.eos_index:
                try:
                    return decode_unicode_tokens(transport), True
                except ValueError:
                    return "", False
            if category >= TRANSPORT_VOCAB_SIZE:
                return "", False
            transport.append(category)
            token = torch.tensor([[category]], dtype=torch.long, device=self.device)
        return "", False

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
        operation = tuple(ReasoningOperationKind)[
            int(output.operation_logits.argmax(dim=-1).item())
        ]
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


def candidate_a_config(**overrides: Any) -> LivingReasoningCoreConfig:
    values = asdict(LivingReasoningCoreConfig())
    values.pop("architecture_id", None)
    values.update(overrides)
    return LivingReasoningCoreConfig(**values)


__all__ = [
    "D64_SOUL_CODEC_VERSION",
    "D64_SOUL_MEDIA_TYPE",
    "LIVING_REASONING_ARCHITECTURE_SCHEMA",
    "CausalLivingUnroll",
    "D64SoulCodec",
    "LivingReasoningCoreConfig",
    "LivingReasoningCoreD64",
    "LivingReasoningForward",
    "candidate_a_config",
]
