"""English-native D64 reasoning tissue.

The active learned surface is deliberately small: the Core reads the frozen
Shared Field, proposal boards, and its private Soul, then emits variable-length
Unicode text with the already-proven transport decoder.  FIRST and REFINED are
free English proposals.  CONSOLIDATED is compact tagged-region English.  Heart
alone turns the latter into an internal typed FieldDelta.

The pre-2026-09-19 typed decision/operation/address anatomy is retained only in
``legacy_typed_reasoning_d64`` so immutable checkpoints can be inspected and
used as governed donors.  Active instances contain none of those learned heads.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from runtime.field import CompiledD64Field, canonical_sha256
from runtime.heart import (
    EnglishProposal,
    ReasoningPassRequest,
    ReasoningPassResult,
    TechnicalFinalVerdict,
)
from runtime.soul import SoulSnapshot, SoulTemperature

from .complete_field_64d import AddressableMemory, CompleteField64D, CoverageManifest
from .legacy_typed_reasoning_d64 import (
    D64_DECODER_EXECUTION_STATE_SCHEMA,
    D64_DECODER_TRACE_SCHEMA,
    D64_RECEIPT_MIGRATION_SCHEMA,
    D64_SOUL_CODEC_VERSION,
    D64_SOUL_MEDIA_TYPE,
    CausalDecoderStep,
    CausalLivingUnroll,
    D64ReceiptMigrationReceipt,
    D64SoulCodec,
    DecoderEmissionRoute,
    DecoderExecutionState,
    DecoderSliceResult,
    TensorCopyReceipt,
    _join_memory,
    migrate_legacy_weights_to_receipt_variant,
)
from .legacy_typed_reasoning_d64 import (
    LivingReasoningCoreConfig as LegacyTypedLivingReasoningCoreConfig,
)
from .legacy_typed_reasoning_d64 import (
    LivingReasoningCoreD64 as LegacyTypedLivingReasoningCoreD64,
)

LIVING_REASONING_ARCHITECTURE_SCHEMA = "axon-living-reasoning-english-architecture-v1"
LIVING_REASONING_RECEIPT_ARCHITECTURE_SCHEMA = (
    "axon-living-reasoning-english-receipt-architecture-v1"
)
ENGLISH_REASONING_OUTPUT_CONTRACT = "english-proposal-tagged-final-v1"
ENGLISH_REASONING_TERMINATION_CONTRACT = "generated-eos-independent-of-content-gate-v1"
D64_ENGLISH_MIGRATION_SCHEMA = "axon-d64-english-reasoning-migration-v1"

# The only learned tensors that may exist in a pre-amendment donor but may not
# exist in an English-native target.  Everything else must match exactly.
LEGACY_TYPED_OUTPUT_TENSOR_PREFIXES = (
    "decision_head.",
    "operation_head.",
    "region_head.",
    "start_query.",
    "end_query.",
    "boundary_seed",
)


@dataclass(frozen=True, slots=True)
class LivingReasoningCoreConfig(LegacyTypedLivingReasoningCoreConfig):
    """Active D64 config with an identity-distinct English output contract."""

    reasoning_output_contract: str = ENGLISH_REASONING_OUTPUT_CONTRACT

    def __post_init__(self) -> None:
        # Reuse every reader/decoder/Soul invariant from the proven D64 body,
        # then deliberately sever architecture identity from the retired typed
        # output topology.
        LegacyTypedLivingReasoningCoreConfig.__post_init__(self)
        if self.receipt_continuation or self.eos_generate_head_route or self.termination_head_route:
            raise ValueError(
                "English-native reasoning retired receipt/termination-head routes; "
                "EOS is always the generated text terminator"
            )
        if self.reasoning_output_contract != ENGLISH_REASONING_OUTPUT_CONTRACT:
            raise ValueError(
                "LivingReasoningCoreConfig supports only the English proposal/tagged FINAL contract"
            )
        prefix = "living-d64-english-"
        object.__setattr__(
            self,
            "architecture_id",
            prefix + canonical_sha256(self.to_canonical_dict(include_id=False))[:24],
        )

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = LegacyTypedLivingReasoningCoreConfig.to_canonical_dict(self, include_id=False)
        value["schema"] = LIVING_REASONING_ARCHITECTURE_SCHEMA
        value["reasoning_output_contract"] = self.reasoning_output_contract
        value["termination_contract"] = ENGLISH_REASONING_TERMINATION_CONTRACT
        if include_id:
            value["architecture_id"] = self.architecture_id
        return value


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
class D64EnglishMigrationReceipt:
    """Exact proof that a typed donor became an English-native candidate.

    The migration has no forgiving load and initializes no new learned tensors:
    the target is the donor body/decoder/Soul tissue with only the superseded
    typed decision/operation/address parameters removed.
    """

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
    return any(name == prefix or name.startswith(prefix) for prefix in LEGACY_TYPED_OUTPUT_TENSOR_PREFIXES)


def migrate_typed_checkpoint_state_to_english_variant(
    source_state: Mapping[str, torch.Tensor],
    target: "LivingReasoningCoreD64",
    *,
    source_architecture_id: str,
    source_parameter_generation: str,
    target_parameter_generation: str,
) -> D64EnglishMigrationReceipt:
    """Strictly transplant an old typed checkpoint into the English topology.

    Every target tensor must exist in the donor with identical shape and dtype.
    The donor may contain extra tensors only when their names are one of the
    explicitly retired typed-output families.  This is therefore a governed
    donor transition, never a checkpoint resume.
    """

    if not isinstance(target, LivingReasoningCoreD64):
        raise TypeError("target must be an English-native LivingReasoningCoreD64")
    if not source_architecture_id or not source_parameter_generation or not target_parameter_generation:
        raise ValueError("migration identities and generations must be non-empty")
    target_state = target.state_dict()
    source_names = set(source_state)
    target_names = set(target_state)
    missing = sorted(target_names - source_names)
    if missing:
        raise ValueError("typed donor is missing target tensors: " + ", ".join(missing))
    extras = sorted(source_names - target_names)
    invalid_extras = [name for name in extras if not _is_retired_typed_tensor(name)]
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


class LivingReasoningCoreD64(LegacyTypedLivingReasoningCoreD64):
    """D64 Core whose only learned public output is variable-length text."""

    def __init__(self, config: LivingReasoningCoreConfig | None = None) -> None:
        active_config = config or LivingReasoningCoreConfig()
        if not isinstance(active_config, LivingReasoningCoreConfig):
            raise TypeError("English-native LivingReasoningCoreD64 requires LivingReasoningCoreConfig")
        super().__init__(active_config)

        # The parent is the frozen pre-amendment implementation used solely to
        # reuse the proven reader/decoder/Soul mechanics.  Delete every learned
        # typed-output parameter immediately; active state_dicts therefore have
        # no decision/operation/region/address head to train or accidentally use.
        for name in (
            "decision_head",
            "operation_head",
            "region_head",
            "start_query",
            "end_query",
            "boundary_seed",
        ):
            delattr(self, name)


    def _decoder_logits(
        self,
        output: torch.Tensor,
        memory: AddressableMemory | None,
        *,
        return_alignment: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Use ordinary generated EOS while the copy gate routes content only.

        The retired motor trainer coupled termination to the copy/generate gate.
        English-native reasoning does not: EOS probability comes directly from
        the generated text distribution, while the copy pointer may still route
        non-EOS substrate content.  Training and free-running decoding consume
        this same normalized distribution.
        """

        mixed_logits, alignment = CompleteField64D._decoder_logits(
            self, output, memory, return_alignment=True
        )
        generated_log_probabilities = F.log_softmax(alignment["generated_logits"], dim=-1)
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
            mixed_logits[..., self.eos_index] = generated_log_probabilities[..., self.eos_index]
        if return_alignment:
            return mixed_logits, alignment
        return mixed_logits

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
            "proposal_pages": float(sum(item.page_count for item in proposal_coverages)),
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

    def boundary_logits(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "typed region/address heads were retired; English-native reasoning has no boundary logits"
        )

    @torch.no_grad()
    def emit(self, request: ReasoningPassRequest) -> ReasoningPassResult:
        """Emit one English proposal or tagged FINAL using the shared text decoder."""

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
            raise RuntimeError("English reasoning decoder did not terminate within its renewable work slice")
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
        else:  # ReasoningPassRequest already rejects this; retain fail-closed locality.
            raise RuntimeError(f"unsupported English reasoning phase {request.phase!r}")
        return ReasoningPassResult(output=public_output, soul_transition=transition)

    def architecture_report(self) -> dict[str, Any]:
        report = super().architecture_report()
        report.pop("eos_generate_head_route", None)
        report.pop("termination_head_route", None)
        report["reasoning_output_contract"] = ENGLISH_REASONING_OUTPUT_CONTRACT
        report["termination_contract"] = ENGLISH_REASONING_TERMINATION_CONTRACT
        report["retired_learned_heads"] = [
            "decision",
            "operation",
            "region",
            "start_address",
            "end_address",
        ]
        report["public_output"] = "variable_length_unicode_text"
        report["first_refined_surface"] = "EnglishProposal"
        report["consolidated_surface"] = "TechnicalFinalVerdict"
        return report


def candidate_a_config(**overrides: Any) -> LivingReasoningCoreConfig:
    values = asdict(LivingReasoningCoreConfig())
    values.pop("architecture_id", None)
    values.update(overrides)
    return LivingReasoningCoreConfig(**values)


__all__ = [
    "D64_DECODER_EXECUTION_STATE_SCHEMA",
    "D64_DECODER_TRACE_SCHEMA",
    "D64_ENGLISH_MIGRATION_SCHEMA",
    "D64_RECEIPT_MIGRATION_SCHEMA",
    "D64_SOUL_CODEC_VERSION",
    "D64_SOUL_MEDIA_TYPE",
    "ENGLISH_REASONING_OUTPUT_CONTRACT",
    "ENGLISH_REASONING_TERMINATION_CONTRACT",
    "LEGACY_TYPED_OUTPUT_TENSOR_PREFIXES",
    "LIVING_REASONING_ARCHITECTURE_SCHEMA",
    "LIVING_REASONING_RECEIPT_ARCHITECTURE_SCHEMA",
    "CausalDecoderStep",
    "CausalLivingUnroll",
    "D64EnglishMigrationReceipt",
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
    "migrate_typed_checkpoint_state_to_english_variant",
]
