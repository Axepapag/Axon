"""Additive multi-tick training over exact-v4 shared-field episodes.

This module is intentionally separate from ``trainer_slot.py``.  It validates
the exact-v4 builder contract, materializes the canonical shared field for each
tick, deeply supervises the complete 64-position proposal, and teacher-commits
one 64-character segment only after the model-visible forward has completed.

The supplied soul, when enabled, is a read-only conditioning tensor.  No soul
state is returned, persisted, updated, or claimed by this trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cores.core import AxonCore, CHAR_SLOT_DIM
from cores.soul_v2 import SoulState
from runtime.core_proposer import AxonCoreRegionProposer
from runtime.field import (
    CANONICAL_REGION_ORDER,
    CONTEXT_END,
    CONTEXT_START,
    CORE_WRITABLE_REGIONS,
    PROPOSAL_END,
    PROPOSAL_START,
    FieldDelta,
    FieldSpan,
    FieldView,
    FieldViewCursor,
    LogicalRegion,
    PagedFieldView,
    READ_PAGING_PROTOCOL,
    RegionState,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    canonical_json_bytes,
    compile_next_read_page,
    proposal_payload_capacity,
    proposal_tail_offset,
)
from runtime.multi_tick_refiner import MultiTickResult, TickRecord
from substrate import assert_supported_text, default_alphabet, get_letter_bank
from training.soul_compressor import SoulCompressor

from training.build_multitick_curriculum import (
    BUILDER_VERSION,
    CANONICAL_REGIONS as BUILDER_CANONICAL_REGIONS,
    REGION_TARGET_BUDGET,
    SCHEMA as EXACT_V4_SCHEMA,
    SEGMENT_COUNT,
    TYPED_SEGMENT_DELTA_SCHEMA,
    apply_segment_delta,
    canonical_json_bytes as builder_canonical_json_bytes,
    field_sha256,
    sha256_text,
    transition_phase_identity,
)


PROPOSAL_WIDTH = PROPOSAL_END - PROPOSAL_START
if PROPOSAL_WIDTH != 64:
    raise RuntimeError("multi-tick trainer requires the inherited 64-slot proposal")
if len(default_alphabet()) != 95:
    raise RuntimeError("multi-tick trainer requires the frozen 95-character alphabet")


class MultiTickTrainingContractError(RuntimeError):
    """An exact-v4 episode or training state failed closed."""


def _debug_view_diff(stored_hash: str, view: FieldView) -> None:
    """Print a deep diff when a deterministic view hash drifts cross-platform."""

    print(f"DEBUG read_view_hash stored={stored_hash}")
    print(f"DEBUG read_view_hash computed={view.view_hash}")
    print(
        f"DEBUG proposal decode: "
        f"{view.decode(PROPOSAL_START, PROPOSAL_END, blanks_as='')!r}"
    )
    print(
        f"DEBUG context tail decode: "
        f"{view.decode(max(CONTEXT_START, CONTEXT_END - 64), CONTEXT_END, blanks_as='')!r}"
    )

    components: dict[str, str] = {
        "field16": hashlib.sha256(view.field16.tobytes(order="C")).hexdigest(),
        "role_ids": hashlib.sha256(view.role_ids.tobytes(order="C")).hexdigest(),
        "region_ids": hashlib.sha256(
            view.logical_region_ids.tobytes(order="C")
        ).hexdigest(),
        "attention": hashlib.sha256(
            view.attention_mask.tobytes(order="C")
        ).hexdigest(),
        "write": hashlib.sha256(view.write_mask.tobytes(order="C")).hexdigest(),
        "slot_refs_json": hashlib.sha256(
            canonical_json_bytes(
                [ref.to_canonical_dict() for ref in view.slot_refs]
            )
        ).hexdigest(),
        "omissions_json": hashlib.sha256(
            canonical_json_bytes(
                [om.to_canonical_dict() for om in view.omissions]
            )
        ).hexdigest(),
    }
    print(f"DEBUG computed component hashes: {components}")

    for decimals in (4, 5, 6, 7, 8, 9):
        rounded = np.round(view.field16, decimals)
        rounded_hash = hashlib.sha256(rounded.tobytes(order="C")).hexdigest()
        tag = "MATCH" if rounded_hash == stored_hash else "no match"
        print(f"DEBUG field16 rounded to {decimals} decimals: {rounded_hash} ({tag})")

    slot_summary: dict[str, int] = {}
    for ref in view.slot_refs:
        key = f"{ref.kind.value}:{ref.logical_region.value if ref.logical_region else None}"
        slot_summary[key] = slot_summary.get(key, 0) + 1
    print(f"DEBUG slot_ref summary: {slot_summary}")
    print(f"DEBUG omissions count: {len(view.omissions)}")
    sys.stdout.flush()


_WRITER_PARAMETER_ROOTS = (
    "soul_reflect_gate",
    "soul_reflect.",
    "action_norm.",
    "soul_action_norm.",
    "action_to_soul.",
    "soul_router.",
    "soul_router_norm.",
    "soul_compartment_gate",
)

_SOUL_READER_MARKERS = (
    "soul_ingest.",
    ".soul_cross_gate",
    ".cross_q_norm.",
    ".cross_ctx_norm.",
    ".soul_cross.",
)


def _is_writer_parameter(name: str) -> bool:
    return any(
        name == root or name.startswith(root)
        for root in _WRITER_PARAMETER_ROOTS
    )


def _is_soul_reader_parameter(name: str) -> bool:
    return any(marker in name for marker in _SOUL_READER_MARKERS)


def freeze_unused_writer_parameters(core: AxonCore) -> tuple[str, ...]:
    """Freeze parameters bypassed by ``soul_readonly=True``."""

    frozen: list[str] = []
    for name, parameter in core.named_parameters():
        if _is_writer_parameter(name):
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen.append(name)
    return tuple(frozen)


def freeze_inactive_soul_reader_parameters(core: AxonCore) -> tuple[str, ...]:
    """Freeze private-read modules when training is explicitly soul-free."""

    frozen: list[str] = []
    for name, parameter in core.named_parameters():
        if _is_soul_reader_parameter(name):
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen.append(name)
    return tuple(frozen)


def declared_trainable_parameter_names(
    core: AxonCore,
    *,
    soul_read_bearing: bool,
    train_soul: bool = False,
) -> tuple[str, ...]:
    """List parameters genuinely exercised by the selected forward path."""

    cfg = core.cfg
    compartments_mode = (
        train_soul
        and cfg.soul_hot_rows > 0
        and cfg.soul_write_mode == "compartments"
    )
    names: list[str] = []
    for name, parameter in core.named_parameters():
        if not parameter.requires_grad:
            continue
        if not train_soul and _is_writer_parameter(name):
            continue
        if not soul_read_bearing and _is_soul_reader_parameter(name):
            continue
        # In compartments mode the raw reflect gate is bypassed in favor of
        # the per-compartment gate; declaring it trainable creates a false
        # contract failure.  Conversely, in direct mode there are no
        # compartments, so the compartment gate is unused.
        if compartments_mode and name == "soul_reflect_gate":
            continue
        if train_soul and not compartments_mode and name == "soul_compartment_gate":
            continue
        names.append(name)
    return tuple(names)


def declared_trainable_parameters(
    core: AxonCore,
    *,
    soul_read_bearing: bool,
    train_soul: bool = False,
) -> tuple[torch.nn.Parameter, ...]:
    names = set(
        declared_trainable_parameter_names(
            core,
            soul_read_bearing=soul_read_bearing,
            train_soul=train_soul,
        )
    )
    return tuple(
        parameter
        for name, parameter in core.named_parameters()
        if name in names
    )


@dataclass(frozen=True, slots=True)
class GradientCoverage:
    declared_names: tuple[str, ...]
    covered_names: tuple[str, ...]
    missing_names: tuple[str, ...]
    nonfinite_names: tuple[str, ...]
    zero_names: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_names
            or self.nonfinite_names
            or self.zero_names
        )


def verify_finite_nonzero_gradient_coverage(
    core: AxonCore,
    declared_names: Sequence[str],
    *,
    raise_on_error: bool = True,
    extra_modules: Mapping[str, torch.nn.Module] | None = None,
) -> GradientCoverage:
    """Verify every declared parameter has a finite, nonzero gradient."""

    declared = tuple(declared_names)
    if len(declared) != len(set(declared)):
        raise ValueError("declared parameter names contain duplicates")
    named = dict(core.named_parameters())
    if extra_modules:
        for prefix, module in extra_modules.items():
            for name, parameter in module.named_parameters():
                key = f"{prefix}.{name}" if prefix else name
                named[key] = parameter
    unknown = tuple(name for name in declared if name not in named)
    if unknown:
        raise ValueError(f"unknown declared parameters: {unknown}")

    covered: list[str] = []
    missing: list[str] = []
    nonfinite: list[str] = []
    zero: list[str] = []
    for name in declared:
        gradient = named[name].grad
        if gradient is None:
            missing.append(name)
            continue
        if not torch.isfinite(gradient).all().item():
            nonfinite.append(name)
            continue
        if not bool(torch.count_nonzero(gradient).item()):
            zero.append(name)
            continue
        covered.append(name)

    report = GradientCoverage(
        declared_names=declared,
        covered_names=tuple(covered),
        missing_names=tuple(missing),
        nonfinite_names=tuple(nonfinite),
        zero_names=tuple(zero),
    )
    if raise_on_error and not report.valid:
        raise MultiTickTrainingContractError(
            "declared gradient coverage failed: "
            f"missing={report.missing_names}, "
            f"nonfinite={report.nonfinite_names}, "
            f"zero={report.zero_names}"
        )
    return report


def optimizer_parameter_names(
    optimizer: torch.optim.Optimizer,
    core: AxonCore,
    extra_modules: Mapping[str, torch.nn.Module] | None = None,
) -> tuple[str, ...]:
    by_id: dict[int, str] = {
        id(parameter): name for name, parameter in core.named_parameters()
    }
    if extra_modules:
        for prefix, module in extra_modules.items():
            for name, parameter in module.named_parameters():
                key = f"{prefix}.{name}" if prefix else name
                by_id[id(parameter)] = key
    names: list[str] = []
    unknown = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = by_id.get(id(parameter))
            if name is None:
                unknown += 1
            else:
                names.append(name)
    if unknown:
        raise MultiTickTrainingContractError(
            f"optimizer contains {unknown} parameter(s) outside this core "
            "or the supplied extra modules"
        )
    if len(names) != len(set(names)):
        raise MultiTickTrainingContractError(
            "optimizer contains duplicate parameters"
        )
    return tuple(names)


def verify_optimizer_coverage(
    optimizer: torch.optim.Optimizer,
    core: AxonCore,
    declared_names: Sequence[str],
    extra_modules: Mapping[str, torch.nn.Module] | None = None,
) -> tuple[str, ...]:
    actual = optimizer_parameter_names(optimizer, core, extra_modules=extra_modules)
    expected = tuple(declared_names)
    missing = tuple(sorted(set(expected) - set(actual)))
    unexpected = tuple(sorted(set(actual) - set(expected)))
    if missing or unexpected:
        raise MultiTickTrainingContractError(
            "optimizer coverage does not match declared trainables: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return actual


@dataclass(frozen=True, slots=True)
class ValidatedTick:
    tick_index: int
    transition_index: int
    phase: str
    phase_identity: str
    segment_index: int
    segment_count: int
    segment_phase: str
    target_region: LogicalRegion
    target_text: str
    committed_region_text: str
    delta_start: int
    delta_end: int
    delta_replacement: str
    proposal_view_offset: int
    read_cursor_before: FieldViewCursor
    read_cursor_after: FieldViewCursor
    read_cycle_complete: bool
    read_view_hash: str
    evidence_refs: tuple[str, ...]
    input_builder_field_sha256: str
    committed_builder_field_sha256: str
    base_region_sha256: str
    committed_region_sha256: str
    typed_delta_sha256: str
    field_before_texts: tuple[tuple[str, str], ...]
    field_after_texts: tuple[tuple[str, str], ...]

    def before_text(self, region: LogicalRegion | str) -> str:
        name = region.value if isinstance(region, LogicalRegion) else str(region)
        return dict(self.field_before_texts)[name]

    def after_text(self, region: LogicalRegion | str) -> str:
        name = region.value if isinstance(region, LogicalRegion) else str(region)
        return dict(self.field_after_texts)[name]


@dataclass(frozen=True, slots=True)
class ValidatedEpisode:
    episode_id: str
    family: str
    initial_snapshot: SharedFieldSnapshot
    initial_builder_field_sha256: str
    final_builder_field_sha256: str
    ticks: tuple[ValidatedTick, ...]
    launch_gate_passed: bool


def _contract_error(path: str, message: str) -> MultiTickTrainingContractError:
    return MultiTickTrainingContractError(f"{path}: {message}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _contract_error(path, "must be an object")
    return value


def _require_hash(value: Any, expected: str, path: str) -> None:
    if not isinstance(value, str) or value.upper() != expected.upper():
        raise _contract_error(
            path,
            f"hash mismatch (expected {expected}, got {value!r})",
        )


def _validate_builder_field(
    raw_field: Any,
    path: str,
) -> tuple[Mapping[str, Any], tuple[tuple[str, str], ...]]:
    field = _require_mapping(raw_field, path)
    expected_regions = set(BUILDER_CANONICAL_REGIONS)
    actual_regions = set(field)
    if actual_regions != expected_regions or len(field) != 10:
        raise _contract_error(
            path,
            "must contain exactly the ten canonical regions; "
            f"missing={sorted(expected_regions - actual_regions)}, "
            f"unexpected={sorted(actual_regions - expected_regions)}",
        )

    texts: list[tuple[str, str]] = []
    for region in BUILDER_CANONICAL_REGIONS:
        cell = _require_mapping(field[region], f"{path}.{region}")
        text = cell.get("text")
        if not isinstance(text, str):
            raise _contract_error(f"{path}.{region}.text", "must be a string")
        try:
            assert_supported_text(text)
        except ValueError as exc:
            raise _contract_error(
                f"{path}.{region}.text",
                f"is not exact-v4 substrate text: {exc}",
            ) from exc
        spans = cell.get("spans", [])
        if not isinstance(spans, list):
            raise _contract_error(f"{path}.{region}.spans", "must be a list")
        for span_index, raw_span in enumerate(spans):
            span = _require_mapping(
                raw_span,
                f"{path}.{region}.spans[{span_index}]",
            )
            start = span.get("start")
            end = span.get("end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start <= end <= len(text)
            ):
                raise _contract_error(
                    f"{path}.{region}.spans[{span_index}]",
                    "contains invalid character bounds",
                )
        texts.append((region, text))
    try:
        field_sha256(field)  # type: ignore[arg-type]
    except Exception as exc:
        raise _contract_error(path, f"builder field is not hashable: {exc}") from exc
    return field, tuple(texts)


def _snapshot_from_builder_field(
    *,
    field: Mapping[str, Any],
    episode_id: str,
    builder_field_hash: str,
    provenance: Mapping[str, Any],
) -> SharedFieldSnapshot:
    source = str(provenance.get("record_pointer", ""))
    evidence = provenance.get("evidence_refs", [])
    evidence_refs = (
        tuple(str(item) for item in evidence)
        if isinstance(evidence, list)
        else ()
    )
    source_manifest_ids = [builder_field_hash]
    source_sha = provenance.get("source_sha256")
    if isinstance(source_sha, str) and source_sha:
        source_manifest_ids.append(source_sha)

    regions: list[RegionState] = []
    for logical_region in CANONICAL_REGION_ORDER:
        cell = _require_mapping(
            field[logical_region.value],
            f"initial_field.{logical_region.value}",
        )
        text = str(cell["text"])
        spans: tuple[FieldSpan, ...] = ()
        if text:
            builder_spans = cell.get("spans", [])
            span_audit = json.dumps(
                builder_spans,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            spans = (
                FieldSpan(
                    span_id=(
                        f"{episode_id}:{logical_region.value}:"
                        f"{sha256_text(text)[:16].lower()}"
                    ),
                    text=text,
                    kind="exact_v4_builder_cell",
                    source=source,
                    provenance=span_audit,
                    container_refs=evidence_refs,
                ),
            )
        regions.append(RegionState(name=logical_region, spans=spans))
    return SharedFieldSnapshot(
        tick_id=0,
        regions=tuple(regions),
        source_manifest_ids=tuple(source_manifest_ids),
    )


def _texts_dict(items: tuple[tuple[str, str], ...]) -> dict[str, str]:
    return dict(items)


def _snapshot_texts(snapshot: SharedFieldSnapshot) -> dict[str, str]:
    return {
        region.value: snapshot.region(region).text
        for region in CANONICAL_REGION_ORDER
    }


def validate_exact_v4_episode(episode: Mapping[str, Any]) -> ValidatedEpisode:
    """Validate builder identity, hashes, diffs, and teacher continuity."""

    if not isinstance(episode, Mapping):
        raise TypeError("episode must be a mapping")
    if episode.get("schema") != EXACT_V4_SCHEMA:
        raise _contract_error(
            "schema",
            f"expected {EXACT_V4_SCHEMA!r}",
        )
    if str(episode.get("builder_version")) != str(BUILDER_VERSION):
        raise _contract_error(
            "builder_version",
            f"expected {BUILDER_VERSION!r}",
        )
    episode_id = episode.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise _contract_error("episode_id", "must be a non-empty string")
    family = episode.get("family")
    if not isinstance(family, str) or not family:
        raise _contract_error("family", "must be a non-empty string")
    logical_regions = episode.get("logical_regions")
    if logical_regions != list(BUILDER_CANONICAL_REGIONS):
        raise _contract_error(
            "logical_regions",
            "must exactly match the exact-v4 builder order",
        )

    profile = _require_mapping(episode.get("field_profile"), "field_profile")
    if profile.get("total_slots") != 384:
        raise _contract_error("field_profile.total_slots", "must equal 384")
    if profile.get("learned_physical_role_count") != 3:
        raise _contract_error(
            "field_profile.learned_physical_role_count",
            "must equal 3",
        )
    read_paging = _require_mapping(
        profile.get("read_paging"),
        "field_profile.read_paging",
    )
    expected_read_paging = {
        "protocol": READ_PAGING_PROTOCOL,
        "cursor_scope": "episode",
        "proposal_switch_resets_cursor": False,
        "reset_condition": "complete_sealed_context_and_user_cycle",
        "proposal_window": "tag_aware_current_region_tail",
    }
    if dict(read_paging) != expected_read_paging:
        raise _contract_error(
            "field_profile.read_paging",
            "does not match the episode-global runtime paging contract",
        )
    segment_protocol = _require_mapping(
        episode.get("segment_protocol"),
        "segment_protocol",
    )
    expected_protocol = {
        "name": "fixed_four_by_sixty_four_v1",
        "segment_count": SEGMENT_COUNT,
        "segment_width": PROPOSAL_WIDTH,
        "region_target_budget": REGION_TARGET_BUDGET,
        "segment_zero": "replace_entire_current_region",
        "later_segments": "append_at_current_region_end",
        "empty_segments": "supervised_no_op",
    }
    if dict(segment_protocol) != expected_protocol:
        raise _contract_error(
            "segment_protocol",
            "does not match the fixed four-by-sixty-four contract",
        )
    read_coverage_contract = _require_mapping(
        episode.get("read_coverage_contract"),
        "read_coverage_contract",
    )
    transaction_target_coverage = _require_mapping(
        read_coverage_contract.get("transaction_target_coverage"),
        "read_coverage_contract.transaction_target_coverage",
    )
    expected_read_coverage_keys = {
        "audited_cursor_scope",
        "sealed_read_cycle_completed_within_episode",
        "writable_region_post_commit_readback_guaranteed",
        "whole_field_read_coverage_proven",
        "blocking_reason",
        "transaction_target_coverage",
        "launch_gate_passed",
    }
    if set(read_coverage_contract) != expected_read_coverage_keys:
        raise _contract_error(
            "read_coverage_contract",
            "contains missing or unexpected fields",
        )
    if (
        read_coverage_contract.get("audited_cursor_scope")
        != "sealed_context_regions_plus_user_input"
        or read_coverage_contract.get(
            "writable_region_post_commit_readback_guaranteed"
        )
        is not False
        or read_coverage_contract.get("whole_field_read_coverage_proven")
        is not False
        or read_coverage_contract.get("launch_gate_passed") is not False
    ):
        raise _contract_error(
            "read_coverage_contract",
            "must fail closed until writable post-commit readback is proven",
        )

    initial_field, initial_texts = _validate_builder_field(
        episode.get("initial_field"),
        "initial_field",
    )
    initial_hash = field_sha256(initial_field)  # type: ignore[arg-type]
    _require_hash(
        episode.get("initial_field_sha256"),
        initial_hash,
        "initial_field_sha256",
    )

    provenance = _require_mapping(episode.get("provenance"), "provenance")
    initial_snapshot = _snapshot_from_builder_field(
        field=initial_field,
        episode_id=episode_id,
        builder_field_hash=initial_hash,
        provenance=provenance,
    )
    runtime_snapshot = initial_snapshot
    read_cursor = FieldViewCursor()

    raw_ticks = episode.get("ticks")
    if not isinstance(raw_ticks, list) or not raw_ticks:
        raise _contract_error("ticks", "must be a non-empty list")
    if len(raw_ticks) % SEGMENT_COUNT:
        raise _contract_error(
            "ticks",
            f"must contain exactly {SEGMENT_COUNT} ticks per transition",
        )
    validated_ticks: list[ValidatedTick] = []
    previous_after: Mapping[str, Any] = initial_field
    transition_region: LogicalRegion | None = None
    transition_phase = ""
    transition_identity = ""
    transition_chars = 0
    transition_hash = ""

    for expected_index, raw_tick in enumerate(raw_ticks):
        path = f"ticks[{expected_index}]"
        tick = _require_mapping(raw_tick, path)
        if tick.get("tick_index") != expected_index:
            raise _contract_error(
                f"{path}.tick_index",
                f"must equal {expected_index}",
            )
        expected_transition_index = expected_index // SEGMENT_COUNT
        expected_segment_index = expected_index % SEGMENT_COUNT
        if tick.get("transition_index") != expected_transition_index:
            raise _contract_error(
                f"{path}.transition_index",
                f"must equal {expected_transition_index}",
            )
        if tick.get("segment_index") != expected_segment_index:
            raise _contract_error(
                f"{path}.segment_index",
                f"must equal {expected_segment_index}",
            )
        if tick.get("segment_count") != SEGMENT_COUNT:
            raise _contract_error(
                f"{path}.segment_count",
                f"must equal {SEGMENT_COUNT}",
            )
        phase = tick.get("phase")
        if not isinstance(phase, str) or not phase:
            raise _contract_error(f"{path}.phase", "must be a non-empty string")
        phase_identity = tick.get("phase_identity")
        expected_phase_identity = transition_phase_identity(
            phase,
            expected_transition_index,
        )
        if phase_identity != expected_phase_identity:
            raise _contract_error(
                f"{path}.phase_identity",
                f"must equal {expected_phase_identity!r}",
            )
        expected_segment_phase = (
            "replace_current_region"
            if expected_segment_index == 0
            else "append_at_current_end"
        )
        if tick.get("segment_phase") != expected_segment_phase:
            raise _contract_error(
                f"{path}.segment_phase",
                f"must equal {expected_segment_phase!r}",
            )
        raw_target = tick.get("target_region")
        try:
            target = (
                raw_target
                if isinstance(raw_target, LogicalRegion)
                else LogicalRegion(raw_target)
            )
        except (TypeError, ValueError) as exc:
            raise _contract_error(
                f"{path}.target_region",
                f"unknown logical region {raw_target!r}",
            ) from exc
        if target not in CORE_WRITABLE_REGIONS:
            raise _contract_error(
                f"{path}.target_region",
                f"{target.value!r} is sealed by the runtime field contract",
            )
        if expected_segment_index == 0:
            transition_region = target
            transition_phase = phase
            transition_identity = expected_phase_identity
            raw_transition_chars = tick.get("transition_target_chars")
            if (
                isinstance(raw_transition_chars, bool)
                or not isinstance(raw_transition_chars, int)
                or not 0 <= raw_transition_chars <= REGION_TARGET_BUDGET
            ):
                raise _contract_error(
                    f"{path}.transition_target_chars",
                    f"must be an integer in [0, {REGION_TARGET_BUDGET}]",
                )
            raw_transition_hash = tick.get("transition_target_sha256")
            if (
                not isinstance(raw_transition_hash, str)
                or len(raw_transition_hash) != 64
            ):
                raise _contract_error(
                    f"{path}.transition_target_sha256",
                    "must be a SHA-256 string",
                )
            transition_chars = raw_transition_chars
            transition_hash = raw_transition_hash.upper()
        else:
            if (
                target is not transition_region
                or phase != transition_phase
                or phase_identity != transition_identity
            ):
                raise _contract_error(
                    path,
                    "transition region and phase identity must remain fixed for four ticks",
                )
            if tick.get("transition_target_chars") != transition_chars:
                raise _contract_error(
                    f"{path}.transition_target_chars",
                    "must remain fixed across the transition",
                )
            _require_hash(
                tick.get("transition_target_sha256"),
                transition_hash,
                f"{path}.transition_target_sha256",
            )

        field_before, before_texts = _validate_builder_field(
            tick.get("field_before"),
            f"{path}.field_before",
        )
        if field_before != previous_after:
            raise _contract_error(
                f"{path}.field_before",
                "does not equal the prior teacher-committed field",
            )
        before_hash = field_sha256(field_before)  # type: ignore[arg-type]
        _require_hash(
            tick.get("input_field_sha256"),
            before_hash,
            f"{path}.input_field_sha256",
        )

        target_text = tick.get("segment_target_text")
        if not isinstance(target_text, str):
            raise _contract_error(
                f"{path}.segment_target_text",
                "must be a string",
            )
        try:
            assert_supported_text(target_text)
        except ValueError as exc:
            raise _contract_error(
                f"{path}.segment_target_text",
                str(exc),
            ) from exc
        if len(target_text) > PROPOSAL_WIDTH:
            raise _contract_error(
                f"{path}.segment_target_text",
                f"length {len(target_text)} exceeds {PROPOSAL_WIDTH} slots",
            )
        expected_segment_chars = min(
            PROPOSAL_WIDTH,
            max(0, transition_chars - expected_segment_index * PROPOSAL_WIDTH),
        )
        if len(target_text) != expected_segment_chars:
            raise _contract_error(
                f"{path}.segment_target_text",
                f"length must equal scheduled slice width {expected_segment_chars}",
            )
        committed_text = tick.get("complete_proposed_region_text")
        if not isinstance(committed_text, str):
            raise _contract_error(
                f"{path}.complete_proposed_region_text",
                "must be a string",
            )
        try:
            assert_supported_text(committed_text)
        except ValueError as exc:
            raise _contract_error(
                f"{path}.complete_proposed_region_text",
                str(exc),
            ) from exc
        if len(committed_text) > REGION_TARGET_BUDGET:
            raise _contract_error(
                f"{path}.complete_proposed_region_text",
                f"length {len(committed_text)} exceeds {REGION_TARGET_BUDGET}",
            )

        base_text = field_before[target.value]["text"]
        _require_hash(
            tick.get("base_region_sha256"),
            sha256_text(base_text),
            f"{path}.base_region_sha256",
        )
        _require_hash(
            tick.get("committed_region_sha256"),
            sha256_text(committed_text),
            f"{path}.committed_region_sha256",
        )
        active_view = _require_mapping(
            tick.get("active_view"),
            f"{path}.active_view",
        )
        proposal_projection = _require_mapping(
            active_view.get("proposal_projection"),
            f"{path}.active_view.proposal_projection",
        )
        payload_capacity = proposal_payload_capacity(target.value)
        proposal_view_offset = proposal_tail_offset(target.value, base_text)
        visible_text = base_text[
            proposal_view_offset : proposal_view_offset + payload_capacity
        ]
        expected_projection = {
            "dynamic_target_region": target.value,
            "current_chars": len(base_text),
            "current_text_sha256": sha256_text(base_text),
            "payload_capacity": payload_capacity,
            "proposal_view_offset": proposal_view_offset,
            "visible_start": proposal_view_offset,
            "visible_end": proposal_view_offset + len(visible_text),
            "visible_chars": len(visible_text),
            "visible_text_sha256": sha256_text(visible_text),
            "shows_current_tail": True,
        }
        if dict(proposal_projection) != expected_projection:
            raise _contract_error(
                f"{path}.active_view.proposal_projection",
                "does not describe the current proposal-region tail",
            )
        if _snapshot_texts(runtime_snapshot) != _texts_dict(before_texts):
            raise _contract_error(
                f"{path}.field_before",
                "does not match the runtime snapshot used for read paging",
            )
        read_page = compile_next_read_page(
            runtime_snapshot,
            proposal_region=target,
            cursor=read_cursor,
            proposal_offset=proposal_view_offset,
        )
        raw_read_page = _require_mapping(
            tick.get("read_page"),
            f"{path}.read_page",
        )
        expected_read_page = read_page.to_audit_dict()
        read_page_mismatch = dict(raw_read_page) != expected_read_page
        if read_page_mismatch:
            # Debug: print the first differing keys to isolate Linux/Windows drift.
            print(f"DEBUG {path}.read_page mismatch")
            for key in sorted(set(raw_read_page.keys()) | set(expected_read_page.keys())):
                a = raw_read_page.get(key)
                b = expected_read_page.get(key)
                if a != b:
                    print(f"  {key}: stored={a!r}")
                    print(f"  {key}: computed={b!r}")
            _debug_view_diff(
                str(raw_read_page.get("read_view_hash", "")),
                read_page.view,
            )
            if os.environ.get("AXON_RELAX_READ_VIEW_HASH") != "1":
                raise _contract_error(
                    f"{path}.read_page",
                    "does not match the deterministic runtime read page",
                )
            print(
                f"WARNING {path}.read_page hash mismatch ignored due to "
                "AXON_RELAX_READ_VIEW_HASH=1"
            )
        context_projection = _require_mapping(
            active_view.get("context_projection"),
            f"{path}.active_view.context_projection",
        )
        expected_context_page = {
            "read_page_index": read_page.cursor.page_index,
            "read_view_hash": read_page.view.view_hash,
            "cursor_before": read_page.cursor.to_canonical_dict(),
            "cursor_after": read_page.next_cursor.to_canonical_dict(),
            "read_cycle_complete": read_page.read_cycle_complete,
        }
        if os.environ.get("AXON_RELAX_READ_VIEW_HASH") == "1":
            expected_context_page.pop("read_view_hash", None)
        for key, value in expected_context_page.items():
            if context_projection.get(key) != value:
                raise _contract_error(
                    f"{path}.active_view.context_projection.{key}",
                    "does not match the deterministic runtime read page",
                )

        expected_start = 0 if expected_segment_index == 0 else len(base_text)
        expected_end = len(base_text)
        expected_committed_text = (
            base_text[:expected_start] + target_text + base_text[expected_end:]
        )
        if committed_text != expected_committed_text:
            raise _contract_error(
                f"{path}.complete_proposed_region_text",
                "does not match the structural segment commit",
            )
        typed_delta = _require_mapping(
            tick.get("typed_delta"),
            f"{path}.typed_delta",
        )
        record_pointer = provenance.get("record_pointer")
        if not isinstance(record_pointer, str) or not record_pointer:
            raise _contract_error(
                "provenance.record_pointer",
                "must be a non-empty string",
            )
        expected_delta_prefix = {
            "schema": TYPED_SEGMENT_DELTA_SCHEMA,
            "op": "replace_text",
            "region": target.value,
            "start": expected_start,
            "end": expected_end,
            "replacement": target_text,
            "evidence_ref": record_pointer,
            "base_region_sha256": sha256_text(base_text),
            "result_region_sha256": sha256_text(committed_text),
            "base_field_sha256": before_hash,
        }
        for key, value in expected_delta_prefix.items():
            if typed_delta.get(key) != value:
                raise _contract_error(
                    f"{path}.typed_delta.{key}",
                    "does not match the exact structural segment delta",
                )
        if set(typed_delta) != set(expected_delta_prefix) | {"result_field_sha256"}:
            raise _contract_error(
                f"{path}.typed_delta",
                "contains missing or unexpected structural delta fields",
            )
        replayed = (
            base_text[: int(typed_delta["start"])]
            + str(typed_delta["replacement"])
            + base_text[int(typed_delta["end"]) :]
        )
        if replayed != committed_text:
            raise _contract_error(
                f"{path}.typed_delta",
                "replay does not produce the committed full region",
            )

        field_after, after_texts = _validate_builder_field(
            tick.get("field_after_teacher_commit"),
            f"{path}.field_after_teacher_commit",
        )
        try:
            replayed_field = apply_segment_delta(
                field_before,  # type: ignore[arg-type]
                typed_delta,
            )
        except Exception as exc:
            raise _contract_error(
                f"{path}.typed_delta",
                f"full-field replay failed: {exc}",
            ) from exc
        if replayed_field != field_after:
            raise _contract_error(
                f"{path}.field_after_teacher_commit",
                "does not equal the exact full-field delta replay",
            )
        for region in BUILDER_CANONICAL_REGIONS:
            expected_text = (
                committed_text
                if region == target.value
                else field_before[region]["text"]
            )
            if field_after[region]["text"] != expected_text:
                raise _contract_error(
                    f"{path}.field_after_teacher_commit.{region}.text",
                    "does not match the one-region teacher commit",
                )
        after_hash = field_sha256(field_after)  # type: ignore[arg-type]
        _require_hash(
            typed_delta.get("result_field_sha256"),
            after_hash,
            f"{path}.typed_delta.result_field_sha256",
        )
        supervision = _require_mapping(
            tick.get("model_supervision"),
            f"{path}.model_supervision",
        )
        if dict(supervision) != {
            "target_is_segment_only": True,
            "target_slots": PROPOSAL_WIDTH,
            "target_chars": len(target_text),
        }:
            raise _contract_error(
                f"{path}.model_supervision",
                "does not match segment-only supervision",
            )
        teacher = _require_mapping(
            tick.get("teacher_forced"),
            f"{path}.teacher_forced",
        )
        if set(teacher) != {
            "commit_segment_delta",
            "committed_field_sha256",
        }:
            raise _contract_error(
                f"{path}.teacher_forced",
                "contains missing, unexpected, or legacy fields",
            )
        if teacher.get("commit_segment_delta") is not True:
            raise _contract_error(
                f"{path}.teacher_forced.commit_segment_delta",
                "must be true",
            )
        _require_hash(
            teacher.get("committed_field_sha256"),
            after_hash,
            f"{path}.teacher_forced.committed_field_sha256",
        )
        free_running = _require_mapping(
            tick.get("free_running_evaluation"),
            f"{path}.free_running_evaluation",
        )
        if set(free_running) != {
            "commit_model_segment_structurally",
            "compare_region_sha256",
            "compare_field_sha256",
            "teacher_forcing_allowed",
        }:
            raise _contract_error(
                f"{path}.free_running_evaluation",
                "contains missing, unexpected, or legacy fields",
            )
        if free_running.get("teacher_forcing_allowed") is not False:
            raise _contract_error(
                f"{path}.free_running_evaluation.teacher_forcing_allowed",
                "must be false",
            )
        _require_hash(
            free_running.get("compare_region_sha256"),
            sha256_text(committed_text),
            f"{path}.free_running_evaluation.compare_region_sha256",
        )
        _require_hash(
            free_running.get("compare_field_sha256"),
            after_hash,
            f"{path}.free_running_evaluation.compare_field_sha256",
        )
        if free_running.get("commit_model_segment_structurally") is not True:
            raise _contract_error(
                f"{path}.free_running_evaluation.commit_model_segment_structurally",
                "must be true",
            )

        evidence = tick.get("evidence_refs", [])
        if not isinstance(evidence, list):
            raise _contract_error(f"{path}.evidence_refs", "must be a list")
        if expected_segment_index == SEGMENT_COUNT - 1:
            if len(committed_text) != transition_chars:
                raise _contract_error(
                    f"{path}.complete_proposed_region_text",
                    "final segment length does not match the transition target",
                )
            _require_hash(
                transition_hash,
                sha256_text(committed_text),
                f"{path}.transition_target_sha256",
            )
        validated_tick = ValidatedTick(
            tick_index=expected_index,
            transition_index=expected_transition_index,
            phase=phase,
            phase_identity=expected_phase_identity,
            segment_index=expected_segment_index,
            segment_count=SEGMENT_COUNT,
            segment_phase=expected_segment_phase,
            target_region=target,
            target_text=target_text,
            committed_region_text=committed_text,
            delta_start=expected_start,
            delta_end=expected_end,
            delta_replacement=target_text,
            proposal_view_offset=proposal_view_offset,
            read_cursor_before=read_page.cursor,
            read_cursor_after=read_page.next_cursor,
            read_cycle_complete=read_page.read_cycle_complete,
            read_view_hash=read_page.view.view_hash,
            evidence_refs=tuple(
                sorted(set(str(item) for item in evidence))
            ),
            input_builder_field_sha256=before_hash,
            committed_builder_field_sha256=after_hash,
            base_region_sha256=sha256_text(base_text),
            committed_region_sha256=sha256_text(committed_text),
            typed_delta_sha256=hashlib.sha256(
                builder_canonical_json_bytes(typed_delta)
            ).hexdigest().upper(),
            field_before_texts=before_texts,
            field_after_texts=after_texts,
        )
        validated_ticks.append(validated_tick)
        runtime_snapshot = _teacher_commit(runtime_snapshot, validated_tick)
        if _snapshot_texts(runtime_snapshot) != _texts_dict(after_texts):
            raise _contract_error(
                f"{path}.field_after_teacher_commit",
                "does not match the runtime teacher transaction",
            )
        read_cursor = read_page.next_cursor
        previous_after = field_after

    expected_blocking_reason = (
        "four output ticks do not guarantee post-commit readback of "
        "every character in a 256-character writable region"
    )
    if read_coverage_contract.get("blocking_reason") != expected_blocking_reason:
        raise _contract_error(
            "read_coverage_contract.blocking_reason",
            "does not state the exact unresolved readback blocker",
        )
    expected_target_characters = sum(
        len(tick.target_text) for tick in validated_ticks
    )
    expected_transaction_coverage = {
        "transition_count": len(validated_ticks) // SEGMENT_COUNT,
        "target_characters": expected_target_characters,
        "supervised_segment_characters": expected_target_characters,
        "each_target_character_supervised_exactly_once": True,
        "segment_count_per_transition": SEGMENT_COUNT,
        "segment_width": PROPOSAL_WIDTH,
    }
    if dict(transaction_target_coverage) != expected_transaction_coverage:
        raise _contract_error(
            "read_coverage_contract.transaction_target_coverage",
            "does not prove lossless four-by-sixty-four target supervision",
        )
    if read_coverage_contract.get(
        "sealed_read_cycle_completed_within_episode"
    ) is not any(tick.read_cycle_complete for tick in validated_ticks):
        raise _contract_error(
            "read_coverage_contract.sealed_read_cycle_completed_within_episode",
            "does not match the audited per-tick cursor schedule",
        )

    final_hash = field_sha256(previous_after)  # type: ignore[arg-type]
    _require_hash(
        episode.get("final_teacher_forced_field_sha256"),
        final_hash,
        "final_teacher_forced_field_sha256",
    )
    return ValidatedEpisode(
        episode_id=episode_id,
        family=family,
        initial_snapshot=initial_snapshot,
        initial_builder_field_sha256=initial_hash,
        final_builder_field_sha256=final_hash,
        ticks=tuple(validated_ticks),
        launch_gate_passed=False,
    )


@dataclass(frozen=True, slots=True)
class TickTrainingMetrics:
    tick_index: int
    phase: str
    target_region: LogicalRegion
    target_text: str
    predicted_text: str
    target_length: int
    proposal_slots: int
    loss: float
    padded_accuracy: float
    target_character_accuracy: float
    exact_padded_match: bool
    input_snapshot: SharedFieldSnapshot
    input_view_hash: str
    read_page_index: int
    read_cursor_before: FieldViewCursor
    read_cursor_after: FieldViewCursor
    read_cycle_complete: bool
    input_builder_field_sha256: str
    output_snapshot_id: str
    predicted_indices: tuple[int, ...] = ()


TRANSITION_LENGTH_BINS: tuple[tuple[int, str], ...] = (
    (64, "000-064"),
    (128, "065-128"),
    (192, "129-192"),
    (REGION_TARGET_BUDGET, "193-256"),
)


def transition_length_bin(target_length: int) -> str:
    """Return the audited exact-v4 full-transition length bin."""

    if (
        isinstance(target_length, bool)
        or not isinstance(target_length, int)
        or not 0 <= target_length <= REGION_TARGET_BUDGET
    ):
        raise ValueError(
            f"target_length must be an integer in [0, {REGION_TARGET_BUDGET}]"
        )
    for upper_bound, label in TRANSITION_LENGTH_BINS:
        if target_length <= upper_bound:
            return label
    raise AssertionError("transition length bin table is incomplete")


@dataclass(frozen=True, slots=True)
class TransitionReconstructionMetrics:
    evaluation_mode: str
    transition_index: int
    phase: str
    phase_identity: str
    target_region: LogicalRegion
    target_length_bin: str
    segment_predictions: tuple[str, ...]
    predicted_text: str
    target_text: str
    predicted_length: int
    target_length: int
    length_delta: int
    length_match: bool
    predicted_sha256: str
    target_sha256: str
    hash_match: bool
    character_matches: int
    comparison_characters: int
    character_accuracy: float
    exact_text_match: bool
    reconstruction_gate_passed: bool


def reconstruct_transition_metrics(
    ticks: Sequence[ValidatedTick],
    predicted_segments: Sequence[str],
    *,
    evaluation_mode: str,
) -> tuple[TransitionReconstructionMetrics, ...]:
    """Reconstruct and gate complete targets only after four segment outputs."""

    if evaluation_mode not in {"teacher_forced", "free_running"}:
        raise ValueError("evaluation_mode must be teacher_forced or free_running")
    tick_values = tuple(ticks)
    predictions = tuple(predicted_segments)
    if len(tick_values) != len(predictions):
        raise MultiTickTrainingContractError(
            "transition reconstruction requires one prediction per tick"
        )
    if not tick_values or len(tick_values) % SEGMENT_COUNT:
        raise MultiTickTrainingContractError(
            "transition reconstruction requires complete four-tick groups"
        )
    if any(not isinstance(value, str) for value in predictions):
        raise TypeError("predicted segment values must be strings")
    for index, value in enumerate(predictions):
        if len(value) > PROPOSAL_WIDTH:
            raise MultiTickTrainingContractError(
                f"predicted segment {index} exceeds {PROPOSAL_WIDTH} characters"
            )
        try:
            assert_supported_text(value)
        except ValueError as exc:
            raise MultiTickTrainingContractError(
                f"predicted segment {index} is not substrate text: {exc}"
            ) from exc

    reconstructed: list[TransitionReconstructionMetrics] = []
    for transition_index in range(len(tick_values) // SEGMENT_COUNT):
        start = transition_index * SEGMENT_COUNT
        group = tick_values[start : start + SEGMENT_COUNT]
        group_predictions = predictions[start : start + SEGMENT_COUNT]
        if (
            tuple(tick.transition_index for tick in group)
            != (transition_index,) * SEGMENT_COUNT
            or tuple(tick.segment_index for tick in group)
            != tuple(range(SEGMENT_COUNT))
        ):
            raise MultiTickTrainingContractError(
                f"transition {transition_index} is not a canonical four-segment group"
            )
        target_text = group[-1].committed_region_text
        if "".join(tick.target_text for tick in group) != target_text:
            raise MultiTickTrainingContractError(
                f"transition {transition_index} target segments do not reconstruct"
            )
        predicted_text = "".join(group_predictions)
        predicted_length = len(predicted_text)
        target_length = len(target_text)
        comparison_characters = max(predicted_length, target_length)
        character_matches = sum(
            predicted_text[index] == target_text[index]
            for index in range(min(predicted_length, target_length))
        )
        character_accuracy = (
            1.0
            if comparison_characters == 0
            else character_matches / comparison_characters
        )
        predicted_hash = sha256_text(predicted_text)
        target_hash = sha256_text(target_text)
        length_match = predicted_length == target_length
        exact_text_match = predicted_text == target_text
        hash_match = predicted_hash == target_hash
        gate_passed = (
            exact_text_match
            and hash_match
            and length_match
            and character_accuracy == 1.0
        )
        reconstructed.append(
            TransitionReconstructionMetrics(
                evaluation_mode=evaluation_mode,
                transition_index=transition_index,
                phase=group[0].phase,
                phase_identity=group[0].phase_identity,
                target_region=group[0].target_region,
                target_length_bin=transition_length_bin(target_length),
                segment_predictions=group_predictions,
                predicted_text=predicted_text,
                target_text=target_text,
                predicted_length=predicted_length,
                target_length=target_length,
                length_delta=predicted_length - target_length,
                length_match=length_match,
                predicted_sha256=predicted_hash,
                target_sha256=target_hash,
                hash_match=hash_match,
                character_matches=character_matches,
                comparison_characters=comparison_characters,
                character_accuracy=character_accuracy,
                exact_text_match=exact_text_match,
                reconstruction_gate_passed=gate_passed,
            )
        )
    return tuple(reconstructed)


def _transition_bin_gates(
    transitions: Sequence[TransitionReconstructionMetrics],
) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (
            label,
            all(
                transition.reconstruction_gate_passed
                for transition in transitions
                if transition.target_length_bin == label
            ),
        )
        for _, label in TRANSITION_LENGTH_BINS
        if any(
            transition.target_length_bin == label
            for transition in transitions
        )
    )


@dataclass(frozen=True, slots=True)
class MultiTickForwardResult:
    episode_id: str
    total_loss: torch.Tensor
    tick_loss_tensors: tuple[torch.Tensor, ...]
    ticks: tuple[TickTrainingMetrics, ...]
    transitions: tuple[TransitionReconstructionMetrics, ...]
    initial_snapshot: SharedFieldSnapshot
    final_snapshot: SharedFieldSnapshot
    declared_trainable_names: tuple[str, ...]
    soul_read_bearing: bool
    train_soul: bool = False
    soul_continuity_loss: torch.Tensor | None = None
    final_soul: torch.Tensor | None = None
    final_soul_mask: torch.Tensor | None = None

    @property
    def transition_reconstruction_gate_passed(self) -> bool:
        return bool(self.transitions) and all(
            transition.reconstruction_gate_passed
            for transition in self.transitions
        )

    @property
    def transition_bin_gates(self) -> tuple[tuple[str, bool], ...]:
        return _transition_bin_gates(self.transitions)


@dataclass(frozen=True, slots=True)
class FreeRunningEpisodeResult:
    rollout: MultiTickResult
    transitions: tuple[TransitionReconstructionMetrics, ...]
    read_pages: tuple[PagedFieldView, ...]

    @property
    def initial_snapshot(self) -> SharedFieldSnapshot:
        return self.rollout.initial_snapshot

    @property
    def final_snapshot(self) -> SharedFieldSnapshot:
        return self.rollout.final_snapshot

    @property
    def records(self) -> tuple[TickRecord, ...]:
        return self.rollout.records

    @property
    def deltas(self) -> tuple[FieldDelta, ...]:
        return self.rollout.deltas

    @property
    def free_running(self) -> bool:
        return self.rollout.free_running

    @property
    def stopped_stable(self) -> bool:
        return self.rollout.stopped_stable

    @property
    def run_hash(self) -> str:
        return self.rollout.run_hash

    def replay(self) -> SharedFieldSnapshot:
        return self.rollout.replay()

    @property
    def transition_reconstruction_gate_passed(self) -> bool:
        return bool(self.transitions) and all(
            transition.reconstruction_gate_passed
            for transition in self.transitions
        )

    @property
    def transition_bin_gates(self) -> tuple[tuple[str, bool], ...]:
        return _transition_bin_gates(self.transitions)


@dataclass(frozen=True, slots=True)
class MultiTickTrainStep:
    forward: MultiTickForwardResult
    gradient_coverage: GradientCoverage
    optimizer_names: tuple[str, ...]
    updated_names: tuple[str, ...]


def _core_device_dtype(core: AxonCore) -> tuple[torch.device, torch.dtype]:
    reference = next(core.parameters(), None)
    if reference is None:
        raise MultiTickTrainingContractError("AxonCore has no parameters")
    return reference.device, reference.dtype


def _validate_core(core: AxonCore) -> None:
    if not isinstance(core, AxonCore):
        raise TypeError("core must be a real AxonCore")
    if not core.cfg.char_slot_mode:
        raise MultiTickTrainingContractError(
            "core must have char_slot_mode=True"
        )
    if core.cfg.soul_mode != "act_reflect_v2":
        raise MultiTickTrainingContractError(
            "multi-tick trainer requires soul_mode='act_reflect_v2'"
        )
    if core.cfg.char_n_regions != 3:
        raise MultiTickTrainingContractError(
            "core must retain exactly three learned physical roles"
        )
    if core.cfg.char_slot_max_slots < PROPOSAL_END:
        raise MultiTickTrainingContractError(
            "core char_slot_max_slots is smaller than 384"
        )
    if tuple(core.char_lift.shape) != (CHAR_SLOT_DIM, core.cfg.d_model):
        raise MultiTickTrainingContractError("core char_lift has invalid shape")


def _prepare_soul(
    core: AxonCore,
    *,
    soul_read_bearing: bool,
    train_soul: bool = False,
    soul: torch.Tensor | None,
    soul_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = _core_device_dtype(core)
    if soul is None and soul_mask is None:
        if train_soul:
            # Trainable souls need a non-empty default state (base rows + hot
            # rows).  All rows start active so the writer has something to
            # read while it learns to update the hot rows.
            total_rows = int(core.cfg.soul_rows) + int(core.cfg.soul_hot_rows)
            if total_rows <= 0:
                raise MultiTickTrainingContractError(
                    "train_soul=True requires core.cfg.soul_rows + soul_hot_rows > 0"
                )
            soul_value = torch.zeros(
                (1, total_rows, core.cfg.d_model),
                device=device,
                dtype=dtype,
            )
            mask_value = torch.ones(
                (1, total_rows),
                device=device,
                dtype=torch.bool,
            )
            return soul_value, mask_value
        if soul_read_bearing:
            raise MultiTickTrainingContractError(
                "read-bearing training requires soul and soul_mask"
            )
        return (
            torch.zeros((1, 0, core.cfg.d_model), device=device, dtype=dtype),
            torch.zeros((1, 0), device=device, dtype=torch.bool),
        )
    if soul is None or soul_mask is None:
        raise MultiTickTrainingContractError(
            "soul and soul_mask must be supplied together"
        )
    if not soul_read_bearing:
        raise MultiTickTrainingContractError(
            "supplied soul requires soul_read_bearing=True"
        )
    if soul.ndim == 2:
        soul_value = soul.unsqueeze(0)
    elif soul.ndim == 3 and soul.shape[0] == 1:
        soul_value = soul
    else:
        raise MultiTickTrainingContractError(
            "soul must have shape (rows, d_model) or (1, rows, d_model)"
        )
    if soul_value.shape[1] <= 0 or soul_value.shape[2] != core.cfg.d_model:
        raise MultiTickTrainingContractError(
            "soul must contain rows at the core d_model width"
        )
    if not soul_value.dtype.is_floating_point:
        raise MultiTickTrainingContractError("soul must be floating point")
    if not torch.isfinite(soul_value).all().item():
        raise MultiTickTrainingContractError("soul contains non-finite values")

    if soul_mask.ndim == 1:
        mask_value = soul_mask.unsqueeze(0)
    elif soul_mask.ndim == 2 and soul_mask.shape[0] == 1:
        mask_value = soul_mask
    else:
        raise MultiTickTrainingContractError(
            "soul_mask must have shape (rows,) or (1, rows)"
        )
    if tuple(mask_value.shape) != tuple(soul_value.shape[:2]):
        raise MultiTickTrainingContractError(
            "soul_mask shape does not match soul rows"
        )
    if mask_value.dtype != torch.bool:
        raise MultiTickTrainingContractError("soul_mask must use bool dtype")
    if not mask_value.any().item():
        raise MultiTickTrainingContractError(
            "read-bearing soul_mask must expose at least one row"
        )
    return (
        soul_value.detach().clone().to(device=device, dtype=dtype),
        mask_value.detach().clone().to(device=device, dtype=torch.bool),
    )


def _target_indices(
    target: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    bank = get_letter_bank()
    char_to_index = {
        character: index
        for index, character in enumerate(bank.chars)
        if index != bank.empty_index
    }
    indices = torch.full(
        (1, PROPOSAL_WIDTH),
        int(bank.empty_index),
        device=device,
        dtype=torch.long,
    )
    for position, character in enumerate(target):
        try:
            indices[0, position] = char_to_index[character]
        except KeyError as exc:
            raise MultiTickTrainingContractError(
                f"target contains unsupported character {character!r}"
            ) from exc
    return indices


def _decode_prediction(indices: torch.Tensor) -> str:
    bank = get_letter_bank()
    chars: list[str] = []
    for raw_index in indices.detach().cpu().tolist():
        index = int(raw_index)
        if index == bank.empty_index:
            break
        if not 0 <= index < len(bank.chars):
            raise MultiTickTrainingContractError(
                f"core emitted invalid alphabet index {index}"
            )
        chars.append(bank.chars[index])
    return "".join(chars)


def _teacher_commit(
    snapshot: SharedFieldSnapshot,
    tick: ValidatedTick,
) -> SharedFieldSnapshot:
    prior = snapshot.region(tick.target_region).text
    if prior == tick.committed_region_text:
        return snapshot
    delta = FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=snapshot.tick_id,
        author_core_id="exact-v4-teacher",
        pass_id=f"teacher-tick-{tick.tick_index:04d}",
        operations=(
            ReplaceText(
                region=tick.target_region,
                start=tick.delta_start,
                end=tick.delta_end,
                text=tick.delta_replacement,
                provenance=(
                    f"exact-v4:{tick.input_builder_field_sha256}:"
                    f"{tick.typed_delta_sha256}"
                ),
                edge_refs=tick.evidence_refs,
            ),
        ),
        evidence=tick.evidence_refs,
    )
    return apply_delta(snapshot, delta)


class MultiTickTrainer:
    """Deep-supervised trainer over immutable exact-v4 field snapshots."""

    def __init__(
        self,
        core: AxonCore,
        *,
        soul_read_bearing: bool = False,
        train_soul: bool = False,
        soul_continuity_weight: float = 0.01,
        soul_l2_weight: float = 1e-5,
        soul_compressor: SoulCompressor | None = None,
        soul_compression_weight: float = 0.1,
    ) -> None:
        _validate_core(core)
        self.core = core
        self.soul_read_bearing = bool(soul_read_bearing)
        self.train_soul = bool(train_soul)
        self.soul_continuity_weight = float(soul_continuity_weight)
        self.soul_l2_weight = float(soul_l2_weight)
        self.soul_compressor = soul_compressor
        self.soul_compression_weight = float(soul_compression_weight)
        if self.train_soul and not self.soul_read_bearing:
            raise MultiTickTrainingContractError(
                "train_soul=True requires soul_read_bearing=True"
            )
        if self.soul_compressor is not None:
            if not self.train_soul:
                raise MultiTickTrainingContractError(
                    "soul_compressor requires train_soul=True"
                )
            if core.cfg.soul_hot_rows <= 0:
                raise MultiTickTrainingContractError(
                    "soul_compressor requires core.cfg.soul_hot_rows > 0"
                )
            expected_warm = int(core.cfg.soul_rows)
            expected_hot = int(core.cfg.soul_hot_rows)
            if self.soul_compressor.warm_rows != expected_warm:
                raise MultiTickTrainingContractError(
                    f"soul_compressor warm_rows {self.soul_compressor.warm_rows} "
                    f"!= core soul_rows {expected_warm}"
                )
            if self.soul_compressor.hot_rows != expected_hot:
                raise MultiTickTrainingContractError(
                    f"soul_compressor hot_rows {self.soul_compressor.hot_rows} "
                    f"!= core soul_hot_rows {expected_hot}"
                )
            if self.soul_compressor.d_model != core.cfg.d_model:
                raise MultiTickTrainingContractError(
                    f"soul_compressor d_model {self.soul_compressor.d_model} "
                    f"!= core d_model {core.cfg.d_model}"
                )
        if self.train_soul:
            # Writer parameters must be trainable.  Explicitly thaw any that
            # were frozen by a previous trainer/checkpoint.
            self.frozen_writer_names = ()
            for name, parameter in core.named_parameters():
                if _is_writer_parameter(name):
                    parameter.requires_grad_(True)
                    parameter.grad = None
            # The compartment router is zero-initialized so it starts
            # neutral, but a perfectly zero matrix gives the router-norm
            # branch no gradient during the first trainable-soul step.
            # Re-initialize it to small random values once train_soul begins.
            if (
                core.cfg.soul_hot_rows > 0
                and core.cfg.soul_write_mode == "compartments"
                and hasattr(core, "soul_router")
            ):
                nn.init.normal_(core.soul_router.weight, std=0.02)
        else:
            self.frozen_writer_names = freeze_unused_writer_parameters(core)
        self.frozen_inactive_soul_reader_names: tuple[str, ...] = ()
        if not self.soul_read_bearing:
            self.frozen_inactive_soul_reader_names = (
                freeze_inactive_soul_reader_parameters(core)
            )
        self.core_declared_trainable_names = declared_trainable_parameter_names(
            core,
            soul_read_bearing=self.soul_read_bearing,
            train_soul=self.train_soul,
        )
        declared_names = list(self.core_declared_trainable_names)
        if self.soul_compressor is not None:
            for name, _ in self.soul_compressor.named_parameters():
                declared_names.append(f"soul_compressor.{name}")
        self.declared_trainable_names = tuple(declared_names)
        if not self.declared_trainable_names:
            raise MultiTickTrainingContractError(
                "no declared trainable parameters remain"
            )
        bank = get_letter_bank()
        if len(bank.chars) != 96 or bank.empty_index != 95:
            raise MultiTickTrainingContractError(
                "letter bank must contain 95 characters plus one empty class"
            )

    def optimizer_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        named: dict[str, torch.nn.Parameter] = dict(self.core.named_parameters())
        if self.soul_compressor is not None:
            for name, parameter in self.soul_compressor.named_parameters():
                named[f"soul_compressor.{name}"] = parameter
        return tuple(named[name] for name in self.declared_trainable_names)

    def build_optimizer(
        self,
        *,
        lr: float,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("optimizer lr must be finite and positive")
        return optimizer_cls(self.optimizer_parameters(), lr=lr, **kwargs)

    def forward_episode(
        self,
        episode: Mapping[str, Any],
        *,
        soul: torch.Tensor | None = None,
        soul_mask: torch.Tensor | None = None,
        gradient_scale: float = 1.0,
    ) -> MultiTickForwardResult:
        validated = validate_exact_v4_episode(episode)
        soul_value, soul_mask_value = _prepare_soul(
            self.core,
            soul_read_bearing=self.soul_read_bearing,
            train_soul=self.train_soul,
            soul=soul,
            soul_mask=soul_mask,
        )
        soul_before = soul_value.detach().clone()
        soul_mask_before = soul_mask_value.detach().clone()
        device, dtype = _core_device_dtype(self.core)
        bank = get_letter_bank()
        bank_unit = torch.from_numpy(bank.vecs_unit.copy()).to(
            device=device,
            dtype=dtype,
        )

        current = validated.initial_snapshot
        read_cursor = FieldViewCursor()
        tick_losses: list[torch.Tensor] = []
        soul_continuity_losses: list[torch.Tensor] = []
        metrics: list[TickTrainingMetrics] = []
        prior_soul_readonly = bool(self.core.soul_readonly)
        self.core.soul_readonly = not self.train_soul
        # When training the soul we need back-propagation through time so the
        # cross-entropy loss at tick t+1 supervises the writer at tick t.
        # Gradient checkpointing trades recomputation for memory so we can hold
        # the full episode graph without the deterministic segfault caused by
        # accumulating eight uncheckpointed forward graphs.
        prior_use_checkpoint = bool(self.core.use_checkpoint)
        use_bptt = self.train_soul and torch.is_grad_enabled()
        if use_bptt:
            self.core.use_checkpoint = True
        accumulated_loss: torch.Tensor | None = None
        try:
            for tick in validated.ticks:
                expected_before = _texts_dict(tick.field_before_texts)
                if _snapshot_texts(current) != expected_before:
                    raise MultiTickTrainingContractError(
                        f"tick {tick.tick_index} runtime snapshot does not match "
                        "the builder field_before text"
                    )
                input_snapshot = current
                read_page = compile_next_read_page(
                    input_snapshot,
                    proposal_region=tick.target_region,
                    cursor=read_cursor,
                    proposal_offset=tick.proposal_view_offset,
                )
                hash_ok = (
                    read_page.view.view_hash == tick.read_view_hash
                    or os.environ.get("AXON_RELAX_READ_VIEW_HASH") == "1"
                )
                if (
                    read_page.cursor != tick.read_cursor_before
                    or read_page.next_cursor != tick.read_cursor_after
                    or read_page.read_cycle_complete
                    is not tick.read_cycle_complete
                    or not hash_ok
                ):
                    raise MultiTickTrainingContractError(
                        f"tick {tick.tick_index} runtime read page does not "
                        "match the validated builder schedule"
                    )
                if os.environ.get("AXON_RELAX_READ_VIEW_HASH") == "1":
                    print(
                        f"WARNING tick {tick.tick_index} read_view_hash mismatch "
                        "ignored due to AXON_RELAX_READ_VIEW_HASH=1"
                    )
                view = read_page.view
                field16 = torch.from_numpy(
                    np.array(view.field16, copy=True)
                ).unsqueeze(0).to(device=device, dtype=dtype)
                physical_roles = torch.from_numpy(
                    np.array(view.role_ids, copy=True)
                ).unsqueeze(0).to(device=device, dtype=torch.long)
                attention_mask = torch.from_numpy(
                    np.array(view.attention_mask, copy=True)
                ).unsqueeze(0).to(device=device, dtype=torch.bool)

                # soul_prev must stay in the graph for BPTT when train_soul is
                # true; detach it only in the read-only / eval path.
                soul_prev = soul_value if use_bptt else soul_value.detach().clone()
                output = self.core.forward_charslot(
                    field16=field16,
                    region_ids=physical_roles,
                    soul=soul_value,
                    mask=attention_mask,
                    soul_mask=soul_mask_value,
                    response_slice=slice(PROPOSAL_START, PROPOSAL_END),
                )
                returned_soul = output.get("soul")
                if not isinstance(returned_soul, torch.Tensor):
                    raise MultiTickTrainingContractError(
                        "AxonCore did not return a soul tensor"
                    )
                if not self.train_soul:
                    if not torch.equal(returned_soul, soul_value):
                        raise MultiTickTrainingContractError(
                            "AxonCore violated the read-only soul contract"
                        )
                delta = output.get("response_delta_16")
                if not isinstance(delta, torch.Tensor):
                    raise MultiTickTrainingContractError(
                        "AxonCore did not return response_delta_16"
                    )
                if tuple(delta.shape) != (1, PROPOSAL_WIDTH, CHAR_SLOT_DIM):
                    raise MultiTickTrainingContractError(
                        f"proposal delta has shape {tuple(delta.shape)}, expected "
                        f"(1, {PROPOSAL_WIDTH}, {CHAR_SLOT_DIM})"
                    )
                logits = self.core.charslot_logits(delta, bank_unit)
                if tuple(logits.shape) != (1, PROPOSAL_WIDTH, 96):
                    raise MultiTickTrainingContractError(
                        f"proposal logits have invalid shape {tuple(logits.shape)}"
                    )
                if not torch.isfinite(logits).all().item():
                    raise MultiTickTrainingContractError(
                        "proposal logits contain non-finite values"
                    )
                target_indices = _target_indices(
                    tick.target_text,
                    device=device,
                )
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    target_indices.reshape(-1),
                    reduction="mean",
                )
                if not torch.isfinite(loss).item():
                    raise MultiTickTrainingContractError(
                        f"tick {tick.tick_index} loss is non-finite"
                    )
                if self.train_soul:
                    # Continuity regularizer: penalize large, unmasked changes in
                    # the active soul rows between consecutive ticks.  The main
                    # cross-entropy loss still drives useful changes.
                    if soul_mask_value.any().item():
                        continuity = F.mse_loss(
                            returned_soul * soul_mask_value.unsqueeze(-1),
                            soul_prev * soul_mask_value.unsqueeze(-1),
                        )
                        # Small L2 penalty on the active soul rows keeps the
                        # memory state from drifting to arbitrarily large values
                        # as it accumulates across episodes.
                        l2 = (
                            (returned_soul * soul_mask_value.unsqueeze(-1))
                            .pow(2)
                            .mean()
                        )
                    else:
                        continuity = torch.tensor(
                            0.0, device=device, dtype=dtype
                        )
                        l2 = torch.tensor(0.0, device=device, dtype=dtype)
                    soul_continuity_losses.append(continuity.detach())
                    if use_bptt:
                        loss = (
                            loss
                            + self.soul_continuity_weight * continuity
                            + self.soul_l2_weight * l2
                        )
                        # Carry the soul through time without detaching so BPTT
                        # can train the writer from future cross-entropy losses.
                        soul_value = returned_soul
                    else:
                        soul_value = returned_soul.detach()
                else:
                    soul_value = returned_soul.detach()

                # In trainable-soul training mode we accumulate the full episode
                # loss and back-propagate once at the end (with gradient
                # checkpointing).  In read-only or eval mode we back-propagate
                # each tick immediately, if gradients are enabled.
                if use_bptt:
                    scaled = loss * float(gradient_scale)
                    if accumulated_loss is None:
                        accumulated_loss = scaled
                    else:
                        accumulated_loss = accumulated_loss + scaled
                else:
                    if loss.requires_grad:
                        scaled_loss = loss * float(gradient_scale)
                        scaled_loss.backward()
                tick_losses.append(loss.detach())

                predicted_indices = logits.argmax(dim=-1)
                padded_accuracy = float(
                    (predicted_indices == target_indices)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
                predicted_text = _decode_prediction(predicted_indices[0])
                if tick.target_text:
                    target_character_accuracy = float(
                        (
                            predicted_indices[:, : len(tick.target_text)]
                            == target_indices[:, : len(tick.target_text)]
                        )
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                else:
                    # An empty segment still has a real 64-slot all-empty
                    # target.  Score those slots instead of granting 1.0.
                    target_character_accuracy = padded_accuracy

                current = _teacher_commit(input_snapshot, tick)
                read_cursor = read_page.next_cursor
                expected_after = _texts_dict(tick.field_after_texts)
                if _snapshot_texts(current) != expected_after:
                    raise MultiTickTrainingContractError(
                        f"tick {tick.tick_index} teacher commit does not match "
                        "the builder field_after text"
                    )
                metrics.append(
                    TickTrainingMetrics(
                        tick_index=tick.tick_index,
                        phase=tick.phase,
                        target_region=tick.target_region,
                        target_text=tick.target_text,
                        predicted_text=predicted_text,
                        target_length=len(tick.target_text),
                        proposal_slots=PROPOSAL_WIDTH,
                        loss=float(loss.detach().cpu()),
                        padded_accuracy=padded_accuracy,
                        target_character_accuracy=target_character_accuracy,
                        exact_padded_match=bool(
                            (predicted_indices == target_indices).all().item()
                        ),
                        input_snapshot=input_snapshot,
                        input_view_hash=view.view_hash,
                        read_page_index=read_page.cursor.page_index,
                        read_cursor_before=read_page.cursor,
                        read_cursor_after=read_page.next_cursor,
                        read_cycle_complete=read_page.read_cycle_complete,
                        input_builder_field_sha256=(
                            tick.input_builder_field_sha256
                        ),
                        output_snapshot_id=current.field_id,
                        predicted_indices=tuple(
                            predicted_indices[0].detach().cpu().tolist()
                        ),
                    )
                )
            # Phase-A soul compression: at episode boundaries, distill the
            # volatile hot rows back into the warm (base) rows.
            compression_loss: torch.Tensor | None = None
            if (
                self.soul_compressor is not None
                and self.train_soul
                and int(self.core.cfg.soul_hot_rows) > 0
            ):
                base_rows = int(self.core.cfg.soul_rows)
                hot_rows = int(self.core.cfg.soul_hot_rows)
                warm = soul_value[:, :base_rows, :]
                hot = soul_value[:, base_rows : base_rows + hot_rows, :]
                new_warm = self.soul_compressor(hot, warm)
                compression_loss = self.soul_compressor.compression_loss(
                    hot, new_warm
                )
                soul_value = torch.cat([new_warm, hot], dim=1)
                if use_bptt and accumulated_loss is not None:
                    accumulated_loss = (
                        accumulated_loss
                        + self.soul_compression_weight * compression_loss
                    )

            if use_bptt and accumulated_loss is not None:
                accumulated_loss.backward()
        finally:
            self.core.soul_readonly = prior_soul_readonly
            self.core.use_checkpoint = prior_use_checkpoint

        if not self.train_soul:
            if not torch.equal(soul_value, soul_before) or not torch.equal(
                soul_mask_value,
                soul_mask_before,
            ):
                raise MultiTickTrainingContractError(
                    "read-only soul input changed during multi-tick forward"
                )
        if len(tick_losses) != len(validated.ticks):
            raise MultiTickTrainingContractError(
                "deep supervision did not produce one loss per tick"
            )
        total_loss = torch.stack(tick_losses).mean()
        if compression_loss is not None:
            total_loss = (
                total_loss
                + self.soul_compression_weight * compression_loss.detach()
            )
        continuity_loss: torch.Tensor | None = None
        if self.train_soul:
            if len(soul_continuity_losses) != len(validated.ticks):
                raise MultiTickTrainingContractError(
                    "trainable soul path did not produce one continuity loss per tick"
                )
            continuity_loss = (
                torch.stack(soul_continuity_losses).mean()
                * self.soul_continuity_weight
            )
        if not torch.isfinite(total_loss).item():
            raise MultiTickTrainingContractError(
                "aggregate multi-tick loss is non-finite"
            )
        transitions = reconstruct_transition_metrics(
            validated.ticks,
            tuple(metric.predicted_text for metric in metrics),
            evaluation_mode="teacher_forced",
        )
        return MultiTickForwardResult(
            episode_id=validated.episode_id,
            total_loss=total_loss,
            tick_loss_tensors=tuple(tick_losses),
            ticks=tuple(metrics),
            transitions=transitions,
            initial_snapshot=validated.initial_snapshot,
            final_snapshot=current,
            declared_trainable_names=self.declared_trainable_names,
            soul_read_bearing=self.soul_read_bearing,
            train_soul=self.train_soul,
            soul_continuity_loss=(
                continuity_loss.detach().clone()
                if self.train_soul else None
            ),
            final_soul=(
                soul_value.detach().clone()
                if self.soul_read_bearing else None
            ),
            final_soul_mask=(
                soul_mask_value.detach().clone()
                if self.soul_read_bearing else None
            ),
        )

    def train_episode(
        self,
        episode: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        *,
        soul: torch.Tensor | None = None,
        soul_mask: torch.Tensor | None = None,
        verify_gradients: bool = True,
    ) -> MultiTickTrainStep:
        extra_modules: dict[str, torch.nn.Module] | None = None
        if self.soul_compressor is not None:
            extra_modules = {"soul_compressor": self.soul_compressor}
        optimizer_names = verify_optimizer_coverage(
            optimizer,
            self.core,
            self.declared_trainable_names,
            extra_modules=extra_modules,
        )
        named: dict[str, torch.nn.Parameter] = dict(self.core.named_parameters())
        if self.soul_compressor is not None:
            for name, parameter in self.soul_compressor.named_parameters():
                named[f"soul_compressor.{name}"] = parameter
        before = {
            name: parameter.detach().clone()
            for name, parameter in named.items()
            if name in self.declared_trainable_names
        }
        self.core.train()
        optimizer.zero_grad(set_to_none=True)
        forward = self.forward_episode(
            episode,
            soul=soul,
            soul_mask=soul_mask,
            gradient_scale=1.0,
        )
        # forward_episode already back-propagated each tick's loss (and the
        # compression loss when a compressor is attached), so gradients are
        # already accumulated in the trainable parameters.
        coverage = verify_finite_nonzero_gradient_coverage(
            self.core,
            self.declared_trainable_names,
            raise_on_error=verify_gradients,
            extra_modules=extra_modules,
        )
        optimizer.step()
        updated = tuple(
            name
            for name in self.declared_trainable_names
            if not torch.equal(before[name], named[name].detach())
        )
        return MultiTickTrainStep(
            forward=forward,
            gradient_coverage=coverage,
            optimizer_names=optimizer_names,
            updated_names=updated,
        )


def forward_multitick_episode(
    core: AxonCore,
    episode: Mapping[str, Any],
    *,
    soul: torch.Tensor | None = None,
    soul_mask: torch.Tensor | None = None,
    train_soul: bool = False,
    gradient_scale: float = 1.0,
) -> MultiTickForwardResult:
    trainer = MultiTickTrainer(
        core,
        soul_read_bearing=(soul is not None) or train_soul,
        train_soul=train_soul,
    )
    return trainer.forward_episode(
        episode,
        soul=soul,
        soul_mask=soul_mask,
        gradient_scale=gradient_scale,
    )


def require_exact_v4_launch_gate(
    episode: Mapping[str, Any],
) -> ValidatedEpisode:
    """Fail closed while whole-field writable readback remains unproven."""

    validated = validate_exact_v4_episode(episode)
    if not validated.launch_gate_passed:
        raise MultiTickTrainingContractError(
            "exact-v4 launch gate failed: four output ticks do not guarantee "
            "post-commit readback of every character in a 256-character "
            "writable region"
        )
    return validated


def evaluate_free_running_episode(
    core: AxonCore,
    episode: Mapping[str, Any],
    *,
    soul_state: SoulState | None = None,
    max_output_chars: int = PROPOSAL_WIDTH,
    author_core_id: str = "axon-core-free-running",
) -> FreeRunningEpisodeResult:
    """Evaluate an episode without passing any gold draft to the rollout.

    Gold targets are used only by :func:`validate_exact_v4_episode` before the
    rollout.  The live schedule retains only region, phase identity, and segment
    position.  Segment zero replaces the model's own current region; later
    segments append to the model's own prior output.
    """

    _validate_core(core)
    validated = validate_exact_v4_episode(episode)
    schedule = tuple(
        (
            tick.target_region,
            tick.phase_identity,
            tick.segment_index,
            tick.segment_count,
        )
        for tick in validated.ticks
    )
    proposer = AxonCoreRegionProposer(
        core,
        soul_state=soul_state,
        max_output_chars=max_output_chars,
    )
    prior_soul_readonly = bool(core.soul_readonly)
    core.soul_readonly = True
    try:
        current = validated.initial_snapshot
        read_cursor = FieldViewCursor()
        read_pages: list[PagedFieldView] = []
        records: list[TickRecord] = []
        deltas: list[FieldDelta] = []
        for tick_index, (
            target_region,
            phase_identity,
            segment_index,
            segment_count,
        ) in enumerate(schedule):
            if segment_count != SEGMENT_COUNT:
                raise MultiTickTrainingContractError(
                    "free-running schedule changed segment count"
                )
            prior = current.region(target_region).text
            proposal_offset = proposal_tail_offset(target_region.value, prior)
            read_page = compile_next_read_page(
                current,
                proposal_region=target_region,
                cursor=read_cursor,
                proposal_offset=proposal_offset,
            )
            view = read_page.view
            proposal = proposer(
                snapshot=current,
                view=view,
                tick_index=tick_index,
                target_region=target_region,
            )
            if proposal.target_region is not target_region:
                raise MultiTickTrainingContractError(
                    f"tick {tick_index} proposer changed scheduled region"
                )
            if segment_index == 0:
                start = 0
                end = len(prior)
            else:
                start = len(prior)
                end = len(prior)
            committed_text = prior[:start] + proposal.text + prior[end:]
            pass_id = (
                f"free-tick-{tick_index:04d}:"
                f"{phase_identity}:segment-{segment_index + 1}-of-{segment_count}"
            )
            input_field_id = current.field_id
            no_op = committed_text == prior
            delta: FieldDelta | None = None
            if no_op:
                next_snapshot = current
            else:
                delta = FieldDelta(
                    base_field_id=current.field_id,
                    base_tick_id=current.tick_id,
                    author_core_id=author_core_id,
                    pass_id=pass_id,
                    operations=(
                        ReplaceText(
                            region=target_region,
                            start=start,
                            end=end,
                            text=proposal.text,
                            provenance=(
                                proposal.provenance
                                or f"exact-v4-free-running:{phase_identity}"
                            ),
                            edge_refs=proposal.evidence_refs,
                        ),
                    ),
                    evidence=proposal.evidence_refs,
                )
                next_snapshot = apply_delta(current, delta)
                deltas.append(delta)
            records.append(
                TickRecord(
                    tick_index=tick_index,
                    pass_id=pass_id,
                    target_region=target_region,
                    input_field_id=input_field_id,
                    input_view_hash=view.view_hash,
                    prior_text=prior,
                    proposed_text=proposal.text,
                    committed_text=committed_text,
                    teacher_forced=False,
                    no_op=no_op,
                    delta_id=None if delta is None else delta.delta_id,
                    output_field_id=next_snapshot.field_id,
                    evidence_refs=proposal.evidence_refs,
                    reference_edit_distance=None,
                )
            )
            current = next_snapshot
            read_cursor = read_page.next_cursor
            read_pages.append(read_page)
        rollout = MultiTickResult(
            initial_snapshot=validated.initial_snapshot,
            final_snapshot=current,
            records=tuple(records),
            deltas=tuple(deltas),
            free_running=True,
            stopped_stable=False,
        )
        # Gold stays entirely outside the rollout.  Only after every model
        # proposal has been committed do we compare grouped outputs with each
        # held-out final transition reference.
        transitions = reconstruct_transition_metrics(
            validated.ticks,
            tuple(record.proposed_text for record in rollout.records),
            evaluation_mode="free_running",
        )
        return FreeRunningEpisodeResult(
            rollout=rollout,
            transitions=transitions,
            read_pages=tuple(read_pages),
        )
    finally:
        core.soul_readonly = prior_soul_readonly


__all__ = [
    "PROPOSAL_WIDTH",
    "SoulCompressor",
    "MultiTickTrainingContractError",
    "GradientCoverage",
    "ValidatedTick",
    "ValidatedEpisode",
    "TickTrainingMetrics",
    "TRANSITION_LENGTH_BINS",
    "transition_length_bin",
    "TransitionReconstructionMetrics",
    "reconstruct_transition_metrics",
    "MultiTickForwardResult",
    "FreeRunningEpisodeResult",
    "MultiTickTrainStep",
    "freeze_unused_writer_parameters",
    "freeze_inactive_soul_reader_parameters",
    "declared_trainable_parameter_names",
    "declared_trainable_parameters",
    "verify_finite_nonzero_gradient_coverage",
    "optimizer_parameter_names",
    "verify_optimizer_coverage",
    "validate_exact_v4_episode",
    "require_exact_v4_launch_gate",
    "MultiTickTrainer",
    "forward_multitick_episode",
    "evaluate_free_running_episode",
]
