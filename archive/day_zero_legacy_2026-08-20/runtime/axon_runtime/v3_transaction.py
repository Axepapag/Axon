"""Neural-free v3 transaction composition, validation, and replay.

This module builds the second isolated protocol slice: a complete synthetic
three-pass cognitive tick over an immutable working field ``W``.  Every
artifact is content-addressed, every page is replayable evidence, and every
reference is verified independently of manifest assertions.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from runtime.field import (
    FieldDelta,
    FieldSpan,
    InsertText,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)

from .field_transaction import (
    AppendSpans,
    FieldTransactionAudit,
    SystemFieldUpdate,
    apply_system_update,
    compose_tick_transaction,
    replay_field_transaction,
)
from .v3_contracts import (
    MAX_PROTOCOL_INT,
    ConsolidationRecord,
    CorePassRecord,
    ProposalBoardManifest,
    ReadCycleManifest,
    SoulTransitionDispositionRecord,
    SoulTransitionRecord,
    TickCommitRecordV3,
    TickPhasePlan,
    V3ContractError,
    validate_tick_artifact_graph,
)
from .v3_serde import canonical_json_text


class V3TransactionError(V3ContractError):
    """A v3 synthetic transaction failed validation or replay."""


class V3TransactionReplayError(V3TransactionError):
    """A v3 transaction did not replay to its claimed audit identity."""


class V3TransactionPopulationError(V3TransactionError):
    """The supplied private-state population does not match the tick plan."""


class V3TransactionArtifactError(V3TransactionError):
    """A content-addressed artifact does not match its claimed references."""


class V3TransactionPageError(V3TransactionError):
    """Read-page coverage or cursor linkage is invalid."""


class V3TransactionDeltaError(V3TransactionError):
    """Field-delta authorship or reference binding is invalid."""


class V3TransactionInvocationError(V3TransactionError):
    """Invocation authority was claimed outside the FINAL consolidator output."""


_INPUT_PHASES = frozenset({"GENESIS", "INPUT"})
_PASS_PHASES = ("INITIAL", "REFINE", "FINAL")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V3TransactionError(f"{label} must be a non-empty string")
    return value


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V3TransactionError(f"{label} must be a non-negative integer")
    if value > MAX_PROTOCOL_INT:
        raise V3TransactionError(f"{label} exceeds signed 64-bit range")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise V3TransactionError(f"{label} must be a 64-character SHA-256")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise V3TransactionError(f"{label} must be a 64-character SHA-256")
    return lowered


def _optional_sha256(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, label)


def _optional_nonempty(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, label)


def _derived_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{canonical_sha256(dict(value))}"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_text_hash(*parts: str) -> str:
    return _content_hash("\x00".join(parts))


def _validate_shared_field_snapshot(value: Any, label: str) -> SharedFieldSnapshot:
    if not isinstance(value, SharedFieldSnapshot):
        raise V3TransactionError(f"{label} must be SharedFieldSnapshot")
    return value


def _validate_system_update(value: Any, label: str) -> SystemFieldUpdate:
    if not isinstance(value, SystemFieldUpdate):
        raise V3TransactionError(f"{label} must be SystemFieldUpdate")
    return value


def _validate_field_transaction_audit(value: Any, label: str) -> FieldTransactionAudit:
    if not isinstance(value, FieldTransactionAudit):
        raise V3TransactionError(f"{label} must be FieldTransactionAudit")
    return value


def _exact_type(value: Any, expected: type[Any], label: str) -> None:
    if type(value) is not expected:
        raise V3TransactionError(f"{label} must be exactly {expected.__name__}")


def _exact_tuple_of(value: Any, expected: type[Any], label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise V3TransactionError(f"{label} must be exactly a tuple")
    for index, item in enumerate(value):
        if type(item) is not expected:
            raise V3TransactionError(
                f"{label} item {index} must be exactly {expected.__name__}"
            )
    return value


@dataclass(frozen=True, slots=True)
class PrivateStateArtifactV3:
    """Immutable neural-free private-state leaf used as input or candidate."""

    core_id: str
    tick_seq: int
    phase: str
    substep: int
    soul_sha256: str
    cursor_state_sha256: str
    rng_state_sha256: str | None
    model_binding_epoch_id: str
    parent_state_leaf_id: str | None
    working_field_id: str | None
    board_id: str | None
    delta_id: str | None
    state_leaf_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _nonempty(self.core_id, "core_id"))
        object.__setattr__(
            self, "tick_seq", _exact_nonnegative_int(self.tick_seq, "tick_seq")
        )
        object.__setattr__(self, "phase", _nonempty(self.phase, "phase"))
        object.__setattr__(
            self, "substep", _exact_nonnegative_int(self.substep, "substep")
        )
        object.__setattr__(self, "soul_sha256", _sha256(self.soul_sha256, "soul_sha256"))
        object.__setattr__(
            self, "cursor_state_sha256", _sha256(self.cursor_state_sha256, "cursor_state_sha256")
        )
        object.__setattr__(
            self, "rng_state_sha256", _optional_sha256(self.rng_state_sha256, "rng_state_sha256")
        )
        object.__setattr__(
            self, "model_binding_epoch_id",
            _nonempty(self.model_binding_epoch_id, "model_binding_epoch_id"),
        )
        object.__setattr__(
            self, "parent_state_leaf_id",
            _optional_nonempty(self.parent_state_leaf_id, "parent_state_leaf_id"),
        )
        object.__setattr__(
            self, "working_field_id", _optional_sha256(self.working_field_id, "working_field_id")
        )
        object.__setattr__(
            self, "board_id", _optional_nonempty(self.board_id, "board_id")
        )
        object.__setattr__(
            self, "delta_id", _optional_nonempty(self.delta_id, "delta_id")
        )
        object.__setattr__(
            self, "state_leaf_id", _derived_id("private-state", self.to_canonical_dict())
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-private-state-artifact-v3",
            "core_id": self.core_id,
            "tick_seq": self.tick_seq,
            "phase": self.phase,
            "substep": self.substep,
            "soul_sha256": self.soul_sha256,
            "cursor_state_sha256": self.cursor_state_sha256,
            "rng_state_sha256": self.rng_state_sha256,
            "model_binding_epoch_id": self.model_binding_epoch_id,
            "parent_state_leaf_id": self.parent_state_leaf_id,
            "working_field_id": self.working_field_id,
            "board_id": self.board_id,
            "delta_id": self.delta_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["state_leaf_id"] = self.state_leaf_id
        return value


@dataclass(frozen=True, slots=True)
class ActiveProjectionArtifactV3:
    """Exact ordered projection payload reconstructed from working field ``W``."""

    source_working_field_id: str
    selected_spans: tuple[tuple[str, str, str, str], ...]
    selected_char_count: int
    source_char_count: int
    projection_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_working_field_id",
            _sha256(self.source_working_field_id, "source_working_field_id"),
        )
        spans = tuple(self.selected_spans)
        for index, item in enumerate(spans):
            if not (
                isinstance(item, tuple)
                and len(item) == 4
                and all(isinstance(part, str) for part in item)
            ):
                raise V3TransactionError(
                    f"selected_spans item {index} must be a (region, span_id, text, hash) tuple"
                )
        object.__setattr__(self, "selected_spans", spans)
        object.__setattr__(
            self, "selected_char_count",
            _exact_nonnegative_int(self.selected_char_count, "selected_char_count"),
        )
        object.__setattr__(
            self, "source_char_count",
            _exact_nonnegative_int(self.source_char_count, "source_char_count"),
        )
        if self.selected_char_count > self.source_char_count:
            raise V3TransactionError(
                "selected characters cannot exceed source characters"
            )
        object.__setattr__(
            self, "projection_id", canonical_sha256(self.to_canonical_dict())
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-active-projection-artifact-v3",
            "source_working_field_id": self.source_working_field_id,
            "selected_spans": [
                {
                    "region": region,
                    "span_id": span_id,
                    "text": text,
                    "canonical_hash": span_hash,
                }
                for region, span_id, text, span_hash in self.selected_spans
            ],
            "selected_char_count": self.selected_char_count,
            "source_char_count": self.source_char_count,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["projection_id"] = self.projection_id
        return value

    @property
    def projection_text(self) -> str:
        return "".join(text for _, _, text, _ in self.selected_spans)


@dataclass(frozen=True, slots=True)
class ReadPageArtifactV3:
    """One replayable page of an effective view with cursor linkage."""

    core_id: str
    tick_seq: int
    phase: str
    working_field_id: str
    projection_id: str
    board_id: str | None
    page_index: int
    start_offset: int
    end_offset: int
    characters: str
    cursor_before_hash: str
    cursor_after_hash: str
    total_effective_character_count: int
    page_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _nonempty(self.core_id, "core_id"))
        object.__setattr__(
            self, "tick_seq", _exact_nonnegative_int(self.tick_seq, "tick_seq")
        )
        object.__setattr__(self, "phase", _nonempty(self.phase, "phase"))
        object.__setattr__(
            self, "working_field_id", _sha256(self.working_field_id, "working_field_id")
        )
        object.__setattr__(
            self, "projection_id", _sha256(self.projection_id, "projection_id")
        )
        object.__setattr__(
            self, "board_id", _optional_nonempty(self.board_id, "board_id")
        )
        object.__setattr__(
            self, "page_index", _exact_nonnegative_int(self.page_index, "page_index")
        )
        object.__setattr__(
            self, "start_offset", _exact_nonnegative_int(self.start_offset, "start_offset")
        )
        object.__setattr__(
            self, "end_offset", _exact_nonnegative_int(self.end_offset, "end_offset")
        )
        if not isinstance(self.characters, str):
            raise V3TransactionError("characters must be a string")
        object.__setattr__(self, "characters", self.characters)
        object.__setattr__(
            self, "cursor_before_hash", _sha256(self.cursor_before_hash, "cursor_before_hash")
        )
        object.__setattr__(
            self, "cursor_after_hash", _sha256(self.cursor_after_hash, "cursor_after_hash")
        )
        object.__setattr__(
            self, "total_effective_character_count",
            _exact_nonnegative_int(
                self.total_effective_character_count,
                "total_effective_character_count",
            ),
        )
        if self.end_offset < self.start_offset:
            raise V3TransactionError("end_offset must be >= start_offset")
        if self.end_offset - self.start_offset != len(self.characters):
            raise V3TransactionError("page character length does not match offsets")
        if self.end_offset > self.total_effective_character_count:
            raise V3TransactionError("page extends past the effective view")
        object.__setattr__(
            self, "page_id", canonical_sha256(self.to_canonical_dict())
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-read-page-artifact-v3",
            "core_id": self.core_id,
            "tick_seq": self.tick_seq,
            "phase": self.phase,
            "working_field_id": self.working_field_id,
            "projection_id": self.projection_id,
            "board_id": self.board_id,
            "page_index": self.page_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "characters": self.characters,
            "cursor_before_hash": self.cursor_before_hash,
            "cursor_after_hash": self.cursor_after_hash,
            "total_effective_character_count": self.total_effective_character_count,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["page_id"] = self.page_id
        return value


@dataclass(frozen=True, slots=True)
class SyntheticTickSpec:
    online_core_ids: tuple[str, ...]
    offline_core_id: str
    consolidator_core_id: str
    accepted: bool
    seed: int = 0
    completion_order: tuple[str, ...] = ()
    model_binding_epoch_id: str = "synthetic-binding-v1"


@dataclass(frozen=True, slots=True)
class TickTransactionV3:
    system_update: SystemFieldUpdate
    projection: ActiveProjectionArtifactV3
    plan: TickPhasePlan
    input_private_states: tuple[PrivateStateArtifactV3, ...]
    candidate_private_states: tuple[PrivateStateArtifactV3, ...]
    read_pages: tuple[ReadPageArtifactV3, ...]
    read_cycles: tuple[ReadCycleManifest, ...]
    deltas: tuple[Any, ...]
    soul_transitions: tuple[SoulTransitionRecord, ...]
    initial_passes: tuple[CorePassRecord, ...]
    initial_board: ProposalBoardManifest
    refine_passes: tuple[CorePassRecord, ...]
    refine_board: ProposalBoardManifest
    consolidation: ConsolidationRecord
    dispositions: tuple[SoulTransitionDispositionRecord, ...]
    field_audit: FieldTransactionAudit
    commit: TickCommitRecordV3


@dataclass(frozen=True, slots=True)
class TickReplayV3:
    final_snapshot: SharedFieldSnapshot
    final_core_state_leaf_ids: tuple[tuple[str, str], ...]
    final_soul_sha256_by_core: tuple[tuple[str, str], ...]
    field_transaction_audit_id: str
    commit_id: str


def _validate_spec(spec: SyntheticTickSpec) -> tuple[tuple[str, ...], set[str]]:
    if not isinstance(spec, SyntheticTickSpec):
        raise V3TransactionError("spec must be SyntheticTickSpec")
    online = tuple(_nonempty(core_id, "online_core_id") for core_id in spec.online_core_ids)
    if len(online) != len(set(online)):
        raise V3TransactionPopulationError("online_core_ids cannot contain duplicates")
    if len(online) != 3:
        raise V3TransactionPopulationError(
            "synthetic tick slice requires exactly three online cores"
        )
    offline = _nonempty(spec.offline_core_id, "offline_core_id")
    consolidator = _nonempty(spec.consolidator_core_id, "consolidator_core_id")
    if offline in online:
        raise V3TransactionPopulationError("offline core cannot be online")
    if consolidator not in online:
        raise V3TransactionPopulationError("consolidator must be online")
    if spec.completion_order:
        completion = tuple(spec.completion_order)
        if sorted(completion) != sorted(online):
            raise V3TransactionPopulationError(
                "completion_order must be a permutation of online_core_ids"
            )
    return online, {offline}


def _validate_private_states(
    all_private_states: Sequence[PrivateStateArtifactV3],
    online: tuple[str, ...],
    offline: str,
    model_binding_epoch_id: str,
) -> dict[str, PrivateStateArtifactV3]:
    states = tuple(all_private_states)
    if not states:
        raise V3TransactionPopulationError("all_private_states cannot be empty")
    by_id: dict[str, PrivateStateArtifactV3] = {}
    by_core: dict[str, PrivateStateArtifactV3] = {}
    for index, state in enumerate(states):
        if not isinstance(state, PrivateStateArtifactV3):
            raise V3TransactionPopulationError(
                f"all_private_states item {index} must be PrivateStateArtifactV3"
            )
        if state.state_leaf_id in by_id:
            raise V3TransactionPopulationError(
                "duplicate private-state leaf ID in all_private_states"
            )
        if state.core_id in by_core:
            raise V3TransactionPopulationError(
                "duplicate core_id in all_private_states"
            )
        if state.model_binding_epoch_id != model_binding_epoch_id:
            raise V3TransactionPopulationError(
                f"private state for {state.core_id} has wrong model binding"
            )
        by_id[state.state_leaf_id] = state
        by_core[state.core_id] = state
    expected_population = set(online) | {offline}
    if set(by_core) != expected_population:
        raise V3TransactionPopulationError(
            "all_private_states population must be online cores plus offline core"
        )
    return by_core


def _build_active_projection(working: SharedFieldSnapshot) -> ActiveProjectionArtifactV3:
    selected: list[tuple[str, str, str, str]] = []
    source_chars = 0
    for region in working.regions:
        for span in region.spans:
            source_chars += len(span.text)
            selected.append(
                (
                    region.name.value,
                    span.span_id,
                    span.text,
                    span.canonical_hash,
                )
            )
    selected_spans = tuple(selected)
    selected_chars = sum(len(text) for _, _, text, _ in selected_spans)
    return ActiveProjectionArtifactV3(
        source_working_field_id=working.field_id,
        selected_spans=selected_spans,
        selected_char_count=selected_chars,
        source_char_count=source_chars,
    )


def _cursor_after(cursor_before: str, characters: str, phase: str, page_index: int) -> str:
    payload = {
        "schema": "axon-cursor-state-v3",
        "cursor_before_hash": cursor_before,
        "characters_digest": _content_hash(characters),
        "phase": phase,
        "page_index": page_index,
    }
    return canonical_sha256(payload)


def _build_pages(
    *,
    core_id: str,
    tick_seq: int,
    phase: str,
    working_field_id: str,
    projection_id: str,
    board_id: str | None,
    effective_text: str,
    initial_cursor_hash: str,
    page_size: int = 96,
) -> tuple[ReadPageArtifactV3, ...]:
    total = len(effective_text)
    pages: list[ReadPageArtifactV3] = []
    cursor_before = initial_cursor_hash
    start = 0
    index = 0
    while start < total:
        end = min(total, start + page_size)
        characters = effective_text[start:end]
        cursor_after = _cursor_after(cursor_before, characters, phase, index)
        pages.append(
            ReadPageArtifactV3(
                core_id=core_id,
                tick_seq=tick_seq,
                phase=phase,
                working_field_id=working_field_id,
                projection_id=projection_id,
                board_id=board_id,
                page_index=index,
                start_offset=start,
                end_offset=end,
                characters=characters,
                cursor_before_hash=cursor_before,
                cursor_after_hash=cursor_after,
                total_effective_character_count=total,
            )
        )
        cursor_before = cursor_after
        start = end
        index += 1
    if total == 0:
        cursor_after = _cursor_after(cursor_before, "", phase, 0)
        pages.append(
            ReadPageArtifactV3(
                core_id=core_id,
                tick_seq=tick_seq,
                phase=phase,
                working_field_id=working_field_id,
                projection_id=projection_id,
                board_id=board_id,
                page_index=0,
                start_offset=0,
                end_offset=0,
                characters="",
                cursor_before_hash=cursor_before,
                cursor_after_hash=cursor_after,
                total_effective_character_count=0,
            )
        )
    return tuple(pages)


def _build_read_cycle(
    *,
    core_id: str,
    tick_seq: int,
    phase: str,
    working_field_id: str,
    projection_id: str,
    board_id: str | None,
    pages: tuple[ReadPageArtifactV3, ...],
) -> ReadCycleManifest:
    return ReadCycleManifest(
        core_id=core_id,
        tick_seq=tick_seq,
        phase=phase,
        working_field_id=working_field_id,
        projection_id=projection_id,
        board_id=board_id,
        page_view_hashes=tuple(page.page_id for page in pages),
        page_character_counts=tuple(len(page.characters) for page in pages),
        cursor_chain=tuple(page.cursor_after_hash for page in pages),
        expected_character_count=pages[0].total_effective_character_count if pages else 0,
        coverage_complete=True,
        selected_span_fingerprint=projection_id,
    )


def _synthetic_soul_hash(core_id: str, phase: str, tick_seq: int, seed: int) -> str:
    return _stable_text_hash("soul", core_id, phase, str(tick_seq), str(seed))


def _synthetic_rng_hash(core_id: str, phase: str, tick_seq: int, seed: int) -> str:
    return _stable_text_hash("rng", core_id, phase, str(tick_seq), str(seed))


def _synthetic_delta_text(core_id: str, phase: str, tick_seq: int, seed: int) -> str:
    return f"[{core_id}:{phase}:{tick_seq}:{seed}]"


def _build_field_delta(
    *,
    core_id: str,
    working: SharedFieldSnapshot,
    phase: str,
    tick_seq: int,
    seed: int,
) -> FieldDelta:
    return FieldDelta(
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        author_core_id=core_id,
        pass_id=f"{core_id}-{phase}-{tick_seq}",
        operations=(
            InsertText(
                region=LogicalRegion.RESPONSE_DRAFT,
                offset=0,
                text=_synthetic_delta_text(core_id, phase, tick_seq, seed),
                provenance=f"synthetic-{phase}",
            ),
        ),
        evidence=(f"synthetic-{phase}",),
    )


def _candidate_private_state(
    *,
    core_id: str,
    tick_seq: int,
    phase: str,
    substep: int,
    soul_sha256: str,
    cursor_state_sha256: str,
    rng_state_sha256: str | None,
    model_binding_epoch_id: str,
    parent_state_leaf_id: str,
    working_field_id: str,
    board_id: str | None,
    delta_id: str,
) -> PrivateStateArtifactV3:
    return PrivateStateArtifactV3(
        core_id=core_id,
        tick_seq=tick_seq,
        phase=phase,
        substep=substep,
        soul_sha256=soul_sha256,
        cursor_state_sha256=cursor_state_sha256,
        rng_state_sha256=rng_state_sha256,
        model_binding_epoch_id=model_binding_epoch_id,
        parent_state_leaf_id=parent_state_leaf_id,
        working_field_id=working_field_id,
        board_id=board_id,
        delta_id=delta_id,
    )


def _peer_delta_payloads(
    peers: tuple[str, ...],
    deltas_by_core: Mapping[str, FieldDelta],
    exclude: str | None = None,
) -> str:
    """Concatenate actual peer delta payloads in deterministic sorted order."""
    selected = [core_id for core_id in peers if core_id != exclude and core_id in deltas_by_core]
    return "".join(deltas_by_core[core_id].operations[0].text for core_id in sorted(selected))


def _peer_delta_payloads_from_records(
    peers: tuple[str, ...],
    pass_records: Mapping[str, CorePassRecord],
    deltas_by_id: Mapping[str, FieldDelta],
    exclude: str | None = None,
) -> str:
    selected = [core_id for core_id in peers if core_id != exclude and core_id in pass_records]
    return "".join(
        deltas_by_id[pass_records[core_id].delta_id].operations[0].text
        for core_id in sorted(selected)
    )


def compose_synthetic_tick_v3(
    head_snapshot: SharedFieldSnapshot,
    all_private_states: Sequence[PrivateStateArtifactV3],
    spec: SyntheticTickSpec,
) -> TickTransactionV3:
    """Build a complete deterministic three-pass v3 tick transaction."""

    head = _validate_shared_field_snapshot(head_snapshot, "head_snapshot")
    online, _ = _validate_spec(spec)
    offline = spec.offline_core_id
    consolidator = spec.consolidator_core_id
    state_by_core = _validate_private_states(
        all_private_states, online, offline, spec.model_binding_epoch_id
    )

    tick_seq = head.tick_id
    seed = spec.seed

    system_update = SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
        operations=(
            AppendSpans(
                region=LogicalRegion.USER_INPUT,
                spans=(
                    FieldSpan(
                        span_id=f"user-input-{tick_seq}",
                        text=f"user-input-{tick_seq}",
                        kind="text",
                        source="synthetic",
                        provenance="synthetic-system-update",
                    ),
                ),
            ),
        ),
        evidence=(f"tick-{tick_seq}",),
    )

    application = apply_system_update(head, system_update)
    working = application.working_snapshot
    no_core_delta_output = compose_tick_transaction(head, system_update, None).final_snapshot

    active_projection = _build_active_projection(working)

    input_states = tuple(
        sorted(
            (state_by_core[core_id] for core_id in (*online, offline)),
            key=lambda state: state.core_id,
        )
    )
    input_leaves = tuple(
        sorted((state.core_id, state.state_leaf_id) for state in input_states if state.core_id in online)
    )
    input_souls = tuple(
        sorted((state.core_id, state.soul_sha256) for state in input_states if state.core_id in online)
    )

    plan = TickPhasePlan(
        tick_seq=tick_seq,
        protocol_version=3,
        input_field_id=head.field_id,
        system_update_id=system_update.update_id,
        working_field_id=working.field_id,
        no_core_delta_output_field_id=no_core_delta_output.field_id,
        projection_id=active_projection.projection_id,
        online_core_ids=online,
        input_core_state_leaf_ids=input_leaves,
        input_soul_sha256_by_core=input_souls,
        offline_core_id=offline,
        consolidator_core_id=consolidator,
        model_binding_epoch_id=spec.model_binding_epoch_id,
    )

    # INITIAL passes.
    initial_deltas: dict[str, FieldDelta] = {}
    initial_cycles: dict[str, ReadCycleManifest] = {}
    initial_pages: list[ReadPageArtifactV3] = []
    initial_transitions: dict[str, SoulTransitionRecord] = {}
    initial_passes: list[CorePassRecord] = []
    initial_candidates: dict[str, PrivateStateArtifactV3] = {}

    for core_id in online:
        delta = _build_field_delta(
            core_id=core_id,
            working=working,
            phase="INITIAL",
            tick_seq=tick_seq,
            seed=seed,
        )
        initial_deltas[core_id] = delta
        pages = _build_pages(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="INITIAL",
            working_field_id=working.field_id,
            projection_id=active_projection.projection_id,
            board_id=None,
            effective_text=active_projection.projection_text,
            initial_cursor_hash=state_by_core[core_id].cursor_state_sha256,
        )
        initial_pages.extend(pages)
        cycle = _build_read_cycle(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="INITIAL",
            working_field_id=working.field_id,
            projection_id=active_projection.projection_id,
            board_id=None,
            pages=pages,
        )
        initial_cycles[core_id] = cycle
        candidate = _candidate_private_state(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="INITIAL",
            substep=0,
            soul_sha256=_synthetic_soul_hash(core_id, "INITIAL", tick_seq, seed),
            cursor_state_sha256=pages[-1].cursor_after_hash,
            rng_state_sha256=_synthetic_rng_hash(core_id, "INITIAL", tick_seq, seed),
            model_binding_epoch_id=spec.model_binding_epoch_id,
            parent_state_leaf_id=state_by_core[core_id].state_leaf_id,
            working_field_id=working.field_id,
            board_id=None,
            delta_id=delta.delta_id,
        )
        initial_candidates[core_id] = candidate
        transition = SoulTransitionRecord(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="INITIAL",
            substep=0,
            input_soul_sha256=state_by_core[core_id].soul_sha256,
            output_soul_sha256=candidate.soul_sha256,
            working_field_id=working.field_id,
            projection_id=active_projection.projection_id,
            read_cycle_id=cycle.cycle_id,
            board_id=None,
            delta_id=delta.delta_id,
            candidate_private_state_id=candidate.state_leaf_id,
            parent_transition_id=None,
            cursor_state_sha256=candidate.cursor_state_sha256,
            rng_state_sha256=candidate.rng_state_sha256,
            binding_manifest_id=spec.model_binding_epoch_id,
        )
        initial_transitions[core_id] = transition
        initial_passes.append(
            CorePassRecord(
                core_id=core_id,
                tick_seq=tick_seq,
                phase="INITIAL",
                working_field_id=working.field_id,
                board_id=None,
                read_cycle_id=cycle.cycle_id,
                delta_id=delta.delta_id,
                soul_transition_id=transition.transition_id,
                candidate_private_state_id=candidate.state_leaf_id,
            )
        )

    initial_board = ProposalBoardManifest(
        working_field_id=working.field_id,
        phase="INITIAL",
        pass_ids_by_author=tuple(
            sorted((record.core_id, record.pass_id) for record in initial_passes)
        ),
        required_authors=online,
    )

    # REFINE passes.
    refine_deltas: dict[str, FieldDelta] = {}
    refine_cycles: dict[str, ReadCycleManifest] = {}
    refine_pages: list[ReadPageArtifactV3] = []
    refine_transitions: dict[str, SoulTransitionRecord] = {}
    refine_passes: list[CorePassRecord] = []
    refine_candidates: dict[str, PrivateStateArtifactV3] = {}

    for core_id in online:
        peer_text = _peer_delta_payloads(online, initial_deltas, exclude=core_id)
        effective_text = active_projection.projection_text + peer_text
        delta = _build_field_delta(
            core_id=core_id,
            working=working,
            phase="REFINE",
            tick_seq=tick_seq,
            seed=seed,
        )
        refine_deltas[core_id] = delta
        pages = _build_pages(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="REFINE",
            working_field_id=working.field_id,
            projection_id=active_projection.projection_id,
            board_id=initial_board.board_id,
            effective_text=effective_text,
            initial_cursor_hash=initial_candidates[core_id].cursor_state_sha256,
        )
        refine_pages.extend(pages)
        cycle = _build_read_cycle(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="REFINE",
            working_field_id=working.field_id,
            projection_id=active_projection.projection_id,
            board_id=initial_board.board_id,
            pages=pages,
        )
        refine_cycles[core_id] = cycle
        candidate = _candidate_private_state(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="REFINE",
            substep=1,
            soul_sha256=_synthetic_soul_hash(core_id, "REFINE", tick_seq, seed),
            cursor_state_sha256=pages[-1].cursor_after_hash,
            rng_state_sha256=_synthetic_rng_hash(core_id, "REFINE", tick_seq, seed),
            model_binding_epoch_id=spec.model_binding_epoch_id,
            parent_state_leaf_id=initial_candidates[core_id].state_leaf_id,
            working_field_id=working.field_id,
            board_id=initial_board.board_id,
            delta_id=delta.delta_id,
        )
        refine_candidates[core_id] = candidate
        transition = SoulTransitionRecord(
            core_id=core_id,
            tick_seq=tick_seq,
            phase="REFINE",
            substep=1,
            input_soul_sha256=initial_candidates[core_id].soul_sha256,
            output_soul_sha256=candidate.soul_sha256,
            working_field_id=working.field_id,
            projection_id=active_projection.projection_id,
            read_cycle_id=cycle.cycle_id,
            board_id=initial_board.board_id,
            delta_id=delta.delta_id,
            candidate_private_state_id=candidate.state_leaf_id,
            parent_transition_id=initial_transitions[core_id].transition_id,
            cursor_state_sha256=candidate.cursor_state_sha256,
            rng_state_sha256=candidate.rng_state_sha256,
            binding_manifest_id=spec.model_binding_epoch_id,
        )
        refine_transitions[core_id] = transition
        refine_passes.append(
            CorePassRecord(
                core_id=core_id,
                tick_seq=tick_seq,
                phase="REFINE",
                working_field_id=working.field_id,
                board_id=initial_board.board_id,
                read_cycle_id=cycle.cycle_id,
                delta_id=delta.delta_id,
                soul_transition_id=transition.transition_id,
                candidate_private_state_id=candidate.state_leaf_id,
            )
        )

    refine_board = ProposalBoardManifest(
        working_field_id=working.field_id,
        phase="REFINE",
        pass_ids_by_author=tuple(
            sorted((record.core_id, record.pass_id) for record in refine_passes)
        ),
        required_authors=online,
    )

    # FINAL pass.
    final_peer_text = _peer_delta_payloads(online, refine_deltas)
    final_effective_text = active_projection.projection_text + final_peer_text
    final_delta = _build_field_delta(
        core_id=consolidator,
        working=working,
        phase="FINAL",
        tick_seq=tick_seq,
        seed=seed,
    )
    final_pages = _build_pages(
        core_id=consolidator,
        tick_seq=tick_seq,
        phase="FINAL",
        working_field_id=working.field_id,
        projection_id=active_projection.projection_id,
        board_id=refine_board.board_id,
        effective_text=final_effective_text,
        initial_cursor_hash=refine_candidates[consolidator].cursor_state_sha256,
    )
    final_cycle = _build_read_cycle(
        core_id=consolidator,
        tick_seq=tick_seq,
        phase="FINAL",
        working_field_id=working.field_id,
        projection_id=active_projection.projection_id,
        board_id=refine_board.board_id,
        pages=final_pages,
    )
    final_candidate = _candidate_private_state(
        core_id=consolidator,
        tick_seq=tick_seq,
        phase="FINAL",
        substep=2,
        soul_sha256=_synthetic_soul_hash(consolidator, "FINAL", tick_seq, seed),
        cursor_state_sha256=final_pages[-1].cursor_after_hash,
        rng_state_sha256=_synthetic_rng_hash(consolidator, "FINAL", tick_seq, seed),
        model_binding_epoch_id=spec.model_binding_epoch_id,
        parent_state_leaf_id=refine_candidates[consolidator].state_leaf_id,
        working_field_id=working.field_id,
        board_id=refine_board.board_id,
        delta_id=final_delta.delta_id,
    )
    final_transition = SoulTransitionRecord(
        core_id=consolidator,
        tick_seq=tick_seq,
        phase="FINAL",
        substep=2,
        input_soul_sha256=refine_candidates[consolidator].soul_sha256,
        output_soul_sha256=final_candidate.soul_sha256,
        working_field_id=working.field_id,
        projection_id=active_projection.projection_id,
        read_cycle_id=final_cycle.cycle_id,
        board_id=refine_board.board_id,
        delta_id=final_delta.delta_id,
        candidate_private_state_id=final_candidate.state_leaf_id,
        parent_transition_id=refine_transitions[consolidator].transition_id,
        cursor_state_sha256=final_candidate.cursor_state_sha256,
        rng_state_sha256=final_candidate.rng_state_sha256,
        binding_manifest_id=spec.model_binding_epoch_id,
    )

    all_deltas = tuple(
        initial_deltas[core_id] for core_id in online
    ) + tuple(refine_deltas[core_id] for core_id in online) + (final_delta,)
    all_candidate_states = tuple(
        initial_candidates[core_id] for core_id in online
    ) + tuple(refine_candidates[core_id] for core_id in online) + (final_candidate,)
    all_read_pages = tuple(initial_pages) + tuple(refine_pages) + tuple(final_pages)
    all_read_cycles = tuple(
        initial_cycles[core_id] for core_id in online
    ) + tuple(refine_cycles[core_id] for core_id in online) + (final_cycle,)
    all_transitions = tuple(
        initial_transitions[core_id] for core_id in online
    ) + tuple(refine_transitions[core_id] for core_id in online) + (final_transition,)
    all_passes = tuple(initial_passes) + tuple(refine_passes)

    # Dispositions and final leaves.
    dispositions: list[SoulTransitionDispositionRecord] = []
    final_leaves: list[tuple[str, str]] = []
    final_souls: list[tuple[str, str]] = []
    if spec.accepted:
        for core_id in online:
            dispositions.append(
                SoulTransitionDispositionRecord(
                    transition_id=initial_transitions[core_id].transition_id,
                    disposition="SUPERSEDED",
                    reason="refined by the next pass",
                    superseded_by_transition_id=refine_transitions[core_id].transition_id,
                )
            )
        for core_id in online:
            if core_id == consolidator:
                dispositions.append(
                    SoulTransitionDispositionRecord(
                        transition_id=refine_transitions[core_id].transition_id,
                        disposition="SUPERSEDED",
                        reason="consolidator authored FINAL",
                        superseded_by_transition_id=final_transition.transition_id,
                    )
                )
                final_leaves.append((core_id, final_candidate.state_leaf_id))
                final_souls.append((core_id, final_candidate.soul_sha256))
            else:
                dispositions.append(
                    SoulTransitionDispositionRecord(
                        transition_id=refine_transitions[core_id].transition_id,
                        disposition="COMMITTED",
                        reason="terminal online state installed",
                    )
                )
                final_leaves.append((core_id, refine_candidates[core_id].state_leaf_id))
                final_souls.append((core_id, refine_candidates[core_id].soul_sha256))
        dispositions.append(
            SoulTransitionDispositionRecord(
                transition_id=final_transition.transition_id,
                disposition="COMMITTED",
                reason="FINAL consolidator state installed",
            )
        )
        consolidator_delta = final_delta
        consolidator_author = consolidator
        output_snapshot = compose_tick_transaction(
            head, system_update, final_delta, consolidator_author_core_id=consolidator
        ).final_snapshot
    else:
        for transition in all_transitions:
            dispositions.append(
                SoulTransitionDispositionRecord(
                    transition_id=transition.transition_id,
                    disposition="REJECTED_NOT_INSTALLED",
                    reason="synthetic rejected tick",
                )
            )
        final_leaves = list(input_leaves)
        final_souls = list(input_souls)
        consolidator_delta = None
        consolidator_author = None
        output_snapshot = no_core_delta_output

    field_audit = compose_tick_transaction(
        head,
        system_update,
        consolidator_delta,
        consolidator_author_core_id=consolidator_author,
    )

    consolidation = ConsolidationRecord(
        core_id=consolidator,
        tick_seq=tick_seq,
        working_field_id=working.field_id,
        refinement_board_id=refine_board.board_id,
        final_delta_id=final_delta.delta_id,
        final_soul_transition_id=final_transition.transition_id,
        accepted=spec.accepted,
        reason="synthetic three-pass consolidation",
        invocation_request_ids=(),
    )

    commit = TickCommitRecordV3(
        tick_seq=tick_seq,
        plan_id=plan.plan_id,
        initial_board_id=initial_board.board_id,
        refine_board_id=refine_board.board_id,
        initial_pass_ids=initial_board.pass_ids_by_author,
        refine_pass_ids=refine_board.pass_ids_by_author,
        consolidation_id=consolidation.consolidation_id,
        disposition_ids_by_transition=tuple(
            sorted((disp.transition_id, disp.disposition_id) for disp in dispositions)
        ),
        final_core_state_leaf_ids=tuple(sorted(final_leaves)),
        field_transaction_audit_id=field_audit.audit_id,
        output_field_id=output_snapshot.field_id,
        invocation_request_ids=(),
    )

    return TickTransactionV3(
        system_update=system_update,
        projection=active_projection,
        plan=plan,
        input_private_states=input_states,
        candidate_private_states=all_candidate_states,
        read_pages=all_read_pages,
        read_cycles=all_read_cycles,
        deltas=all_deltas,
        soul_transitions=all_transitions,
        initial_passes=tuple(initial_passes),
        initial_board=initial_board,
        refine_passes=tuple(refine_passes),
        refine_board=refine_board,
        consolidation=consolidation,
        dispositions=tuple(dispositions),
        field_audit=field_audit,
        commit=commit,
    )


def _artifact_by_id(
    artifacts: Sequence[PrivateStateArtifactV3],
) -> dict[str, PrivateStateArtifactV3]:
    by_id: dict[str, PrivateStateArtifactV3] = {}
    for artifact in artifacts:
        if artifact.state_leaf_id in by_id:
            raise V3TransactionArtifactError("duplicate private-state artifact ID")
        by_id[artifact.state_leaf_id] = artifact
    return by_id


def _verify_private_state_against_transition(
    artifact: PrivateStateArtifactV3,
    transition: SoulTransitionRecord,
    input_by_core: Mapping[str, PrivateStateArtifactV3],
    candidate_by_id: Mapping[str, PrivateStateArtifactV3],
) -> None:
    checks = (
        (artifact.core_id, transition.core_id, "core_id"),
        (artifact.soul_sha256, transition.output_soul_sha256, "soul"),
        (artifact.cursor_state_sha256, transition.cursor_state_sha256, "cursor"),
        (artifact.rng_state_sha256, transition.rng_state_sha256, "rng"),
        (artifact.model_binding_epoch_id, transition.binding_manifest_id, "binding"),
        (artifact.working_field_id, transition.working_field_id, "working_field_id"),
        (artifact.board_id, transition.board_id, "board_id"),
        (artifact.delta_id, transition.delta_id, "delta_id"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise V3TransactionArtifactError(
                f"private-state artifact {label} does not match transition"
            )
    if artifact.tick_seq != transition.tick_seq:
        raise V3TransactionArtifactError("private-state artifact tick mismatch")
    if artifact.phase != transition.phase:
        raise V3TransactionArtifactError("private-state artifact phase mismatch")
    if artifact.substep != transition.substep:
        raise V3TransactionArtifactError("private-state artifact substep mismatch")

    # Parent leaf resolution: INITIAL points to the sealed input state; later
    # phases point to the previous phase candidate for the same core.
    if transition.phase == "INITIAL":
        expected_parent = input_by_core[transition.core_id].state_leaf_id
    elif transition.phase == "REFINE":
        parent_transition = next(
            t
            for t in candidate_by_id.values()
            if t.phase == "INITIAL" and t.core_id == transition.core_id
        )
        expected_parent = parent_transition.state_leaf_id
    else:  # FINAL
        parent_transition = next(
            t
            for t in candidate_by_id.values()
            if t.phase == "REFINE" and t.core_id == transition.core_id
        )
        expected_parent = parent_transition.state_leaf_id
    if artifact.parent_state_leaf_id != expected_parent:
        raise V3TransactionArtifactError(
            "private-state artifact parent leaf does not match transition"
        )


def _expected_effective_text(
    *,
    projection_text: str,
    phase: str,
    core_id: str,
    online: tuple[str, ...],
    consolidator: str,
    initial_delta_by_core: Mapping[str, FieldDelta],
    refine_delta_by_core: Mapping[str, FieldDelta],
) -> str:
    if phase == "INITIAL":
        return projection_text
    if phase == "REFINE":
        peer_text = _peer_delta_payloads(online, initial_delta_by_core, exclude=core_id)
        return projection_text + peer_text
    if phase == "FINAL":
        peer_text = _peer_delta_payloads(online, refine_delta_by_core)
        return projection_text + peer_text
    raise V3TransactionPageError(f"unknown phase {phase}")


def _verify_delta_shape(delta: FieldDelta, expected_author: str) -> None:
    if not isinstance(delta, FieldDelta):
        raise V3TransactionDeltaError("transaction.deltas must contain FieldDelta")
    if delta.author_core_id != expected_author:
        raise V3TransactionDeltaError("delta author mismatch")
    if len(delta.operations) != 1:
        raise V3TransactionDeltaError("synthetic delta must contain exactly one operation")
    operation = delta.operations[0]
    if not isinstance(operation, InsertText):
        raise V3TransactionDeltaError("synthetic delta operation must be InsertText")
    if operation.region != LogicalRegion.RESPONSE_DRAFT:
        raise V3TransactionDeltaError("synthetic delta must target response_draft")
    if operation.offset != 0:
        raise V3TransactionDeltaError("synthetic delta must insert at offset 0")
    if not operation.text:
        raise V3TransactionDeltaError("synthetic delta text must be non-empty")


def _verify_read_pages(
    transaction: TickTransactionV3,
    projection_text: str,
    online: tuple[str, ...],
    consolidator: str,
    initial_delta_by_core: Mapping[str, FieldDelta],
    refine_delta_by_core: Mapping[str, FieldDelta],
) -> None:
    cycle_ids = {
        transition.read_cycle_id: transition
        for transition in transaction.soul_transitions
    }
    pages_by_cycle_id: dict[str, list[ReadPageArtifactV3]] = {
        cycle_id: [] for cycle_id in cycle_ids
    }
    for page in transaction.read_pages:
        if page.phase not in _PASS_PHASES:
            raise V3TransactionPageError("page phase must be INITIAL/REFINE/FINAL")
        key = None
        for cycle_id, transition in cycle_ids.items():
            if (
                page.core_id == transition.core_id
                and page.phase == transition.phase
                and page.working_field_id == transition.working_field_id
                and page.projection_id == transition.projection_id
                and page.board_id == transition.board_id
            ):
                key = cycle_id
                break
        if key is None:
            raise V3TransactionPageError("orphan read page")
        pages_by_cycle_id[key].append(page)

    for cycle in transaction.read_cycles:
        if cycle.cycle_id not in pages_by_cycle_id:
            raise V3TransactionPageError("read cycle has no pages")
        pages = sorted(
            pages_by_cycle_id[cycle.cycle_id], key=lambda page: page.page_index
        )
        if cycle.page_view_hashes != tuple(page.page_id for page in pages):
            raise V3TransactionPageError("read cycle page IDs mismatch")
        if cycle.cursor_chain != tuple(page.cursor_after_hash for page in pages):
            raise V3TransactionPageError("read cycle cursor chain mismatch")
        if cycle.page_character_counts != tuple(len(page.characters) for page in pages):
            raise V3TransactionPageError("read cycle character counts mismatch")
        expected_total = cycle.expected_character_count
        covered = 0
        previous_after: str | None = None
        transition = cycle_ids[cycle.cycle_id]
        expected_text = _expected_effective_text(
            projection_text=projection_text,
            phase=cycle.phase,
            core_id=cycle.core_id,
            online=online,
            consolidator=consolidator,
            initial_delta_by_core=initial_delta_by_core,
            refine_delta_by_core=refine_delta_by_core,
        )
        actual_text_parts: list[str] = []
        for index, page in enumerate(pages):
            if page.page_index != index:
                raise V3TransactionPageError("page index not contiguous from zero")
            if page.start_offset != covered:
                raise V3TransactionPageError("page start offset gap/overlap")
            covered += len(page.characters)
            actual_text_parts.append(page.characters)
            if page.end_offset != covered:
                raise V3TransactionPageError("page end offset mismatch")
            if page.total_effective_character_count != expected_total:
                raise V3TransactionPageError("page total effective count mismatch")
            # Independently recompute cursor-after from exact page characters.
            recomputed_after = _cursor_after(
                page.cursor_before_hash, page.characters, page.phase, page.page_index
            )
            if recomputed_after != page.cursor_after_hash:
                raise V3TransactionPageError("page cursor_after_hash recomputation mismatch")
            if index == 0:
                if page.phase == "INITIAL":
                    input_state = next(
                        state
                        for state in transaction.input_private_states
                        if state.core_id == page.core_id
                    )
                    if page.cursor_before_hash != input_state.cursor_state_sha256:
                        raise V3TransactionPageError(
                            "INITIAL first cursor does not bind to input state"
                        )
                elif page.phase == "REFINE":
                    parent_transition = next(
                        t
                        for t in transaction.soul_transitions
                        if t.phase == "INITIAL" and t.core_id == page.core_id
                    )
                    if page.cursor_before_hash != parent_transition.cursor_state_sha256:
                        raise V3TransactionPageError(
                            "REFINE first cursor does not bind to INITIAL candidate"
                        )
                else:
                    parent_transition = next(
                        t
                        for t in transaction.soul_transitions
                        if t.phase == "REFINE" and t.core_id == page.core_id
                    )
                    if page.cursor_before_hash != parent_transition.cursor_state_sha256:
                        raise V3TransactionPageError(
                            "FINAL first cursor does not bind to REFINE candidate"
                        )
            else:
                if page.cursor_before_hash != previous_after:
                    raise V3TransactionPageError("page cursor linkage broken")
            previous_after = page.cursor_after_hash
        if covered != expected_total:
            raise V3TransactionPageError(
                "page coverage does not reach expected character count"
            )
        if "".join(actual_text_parts) != expected_text:
            raise V3TransactionPageError(
                "read page characters do not reconstruct the expected effective view"
            )
        if cycle.selected_span_fingerprint != transaction.projection.projection_id:
            raise V3TransactionPageError("read cycle projection fingerprint mismatch")
        if pages[-1].cursor_after_hash != transition.cursor_state_sha256:
            raise V3TransactionPageError(
                "terminal page cursor does not match transition candidate cursor"
            )


def _verify_deltas(
    transaction: TickTransactionV3,
    online: tuple[str, ...],
    consolidator: str,
) -> dict[str, FieldDelta]:
    delta_by_id: dict[str, FieldDelta] = {}
    for delta in transaction.deltas:
        if not isinstance(delta, FieldDelta):
            raise V3TransactionDeltaError("transaction.deltas must contain FieldDelta")
        if delta.delta_id in delta_by_id:
            raise V3TransactionDeltaError("duplicate delta in transaction.deltas")
        delta_by_id[delta.delta_id] = delta

    referenced: set[str] = set()
    initial_delta_by_core: dict[str, FieldDelta] = {}
    refine_delta_by_core: dict[str, FieldDelta] = {}
    for pass_record in transaction.initial_passes + transaction.refine_passes:
        if pass_record.delta_id not in delta_by_id:
            raise V3TransactionDeltaError("pass references missing delta")
        delta = delta_by_id[pass_record.delta_id]
        _verify_delta_shape(delta, pass_record.core_id)
        if delta.author_core_id != pass_record.core_id:
            raise V3TransactionDeltaError("delta author does not match pass")
        if delta.base_field_id != transaction.plan.working_field_id:
            raise V3TransactionDeltaError("delta base field mismatch")
        if delta.base_tick_id != transaction.plan.tick_seq:
            raise V3TransactionDeltaError("delta base tick mismatch")
        referenced.add(pass_record.delta_id)
        if pass_record.phase == "INITIAL":
            initial_delta_by_core[pass_record.core_id] = delta
        elif pass_record.phase == "REFINE":
            refine_delta_by_core[pass_record.core_id] = delta
        transition = next(
            t
            for t in transaction.soul_transitions
            if t.transition_id == pass_record.soul_transition_id
        )
        if transition.delta_id != pass_record.delta_id:
            raise V3TransactionDeltaError("pass delta mismatch with transition")

    final_transition = next(
        t for t in transaction.soul_transitions if t.phase == "FINAL"
    )
    if transaction.consolidation.final_delta_id not in delta_by_id:
        raise V3TransactionDeltaError("consolidation references missing final delta")
    final_delta = delta_by_id[transaction.consolidation.final_delta_id]
    _verify_delta_shape(final_delta, consolidator)
    if final_delta.author_core_id != consolidator:
        raise V3TransactionDeltaError("FINAL delta author must be consolidator")
    if final_delta.base_field_id != transaction.plan.working_field_id:
        raise V3TransactionDeltaError("FINAL delta base field mismatch")
    if final_delta.base_tick_id != transaction.plan.tick_seq:
        raise V3TransactionDeltaError("FINAL delta base tick mismatch")
    if final_transition.delta_id != final_delta.delta_id:
        raise V3TransactionDeltaError("FINAL transition delta mismatch")
    referenced.add(final_delta.delta_id)

    if referenced != set(delta_by_id):
        raise V3TransactionDeltaError("transaction contains orphan or unreferenced deltas")

    return delta_by_id


def validate_tick_transaction_v3(transaction: TickTransactionV3) -> TickReplayV3:
    """Validate a complete v3 tick transaction and return its replay evidence."""

    _exact_type(transaction, TickTransactionV3, "transaction")
    _exact_type(transaction.plan, TickPhasePlan, "transaction.plan")
    _exact_type(transaction.system_update, SystemFieldUpdate, "transaction.system_update")
    _exact_type(
        transaction.projection, ActiveProjectionArtifactV3, "transaction.projection"
    )
    _exact_type(transaction.field_audit, FieldTransactionAudit, "transaction.field_audit")
    _exact_type(transaction.commit, TickCommitRecordV3, "transaction.commit")
    _exact_type(
        transaction.consolidation, ConsolidationRecord, "transaction.consolidation"
    )
    _exact_type(
        transaction.initial_board, ProposalBoardManifest, "transaction.initial_board"
    )
    _exact_type(
        transaction.refine_board, ProposalBoardManifest, "transaction.refine_board"
    )

    # Exact tuple/container and element-type boundaries.
    _exact_tuple_of(
        transaction.input_private_states,
        PrivateStateArtifactV3,
        "transaction.input_private_states",
    )
    _exact_tuple_of(
        transaction.candidate_private_states,
        PrivateStateArtifactV3,
        "transaction.candidate_private_states",
    )
    _exact_tuple_of(
        transaction.read_pages, ReadPageArtifactV3, "transaction.read_pages"
    )
    _exact_tuple_of(
        transaction.read_cycles, ReadCycleManifest, "transaction.read_cycles"
    )
    _exact_tuple_of(transaction.deltas, FieldDelta, "transaction.deltas")
    _exact_tuple_of(
        transaction.soul_transitions, SoulTransitionRecord, "transaction.soul_transitions"
    )
    _exact_tuple_of(
        transaction.initial_passes, CorePassRecord, "transaction.initial_passes"
    )
    _exact_tuple_of(
        transaction.refine_passes, CorePassRecord, "transaction.refine_passes"
    )
    _exact_tuple_of(
        transaction.dispositions,
        SoulTransitionDispositionRecord,
        "transaction.dispositions",
    )

    # Invocation authority: this packet forbids typed invocations.
    if transaction.consolidation.invocation_request_ids:
        raise V3TransactionInvocationError(
            "consolidation must not carry invocation request IDs in this packet"
        )
    if transaction.commit.invocation_request_ids:
        raise V3TransactionInvocationError(
            "commit must not carry invocation request IDs in this packet"
        )

    plan = transaction.plan
    online = tuple(plan.online_core_ids)
    offline = plan.offline_core_id
    consolidator = plan.consolidator_core_id
    if not online or len(online) != len(set(online)):
        raise V3TransactionPopulationError("invalid online population in plan")
    if offline in online:
        raise V3TransactionPopulationError("offline core cannot be online")
    if consolidator not in online:
        raise V3TransactionPopulationError("consolidator must be online")

    # Complete population: input_private_states must be an exact tuple of exactly
    # len(online) + 1 PrivateStateArtifactV3 objects covering online + offline.
    expected_input_len = len(online) + 1
    if len(transaction.input_private_states) != expected_input_len:
        raise V3TransactionPopulationError(
            f"input_private_states must contain exactly {expected_input_len} states"
        )
    seen_input_core: set[str] = set()
    seen_input_leaf: set[str] = set()
    input_by_core: dict[str, PrivateStateArtifactV3] = {}
    for index, state in enumerate(transaction.input_private_states):
        if state.core_id in seen_input_core:
            raise V3TransactionPopulationError(
                f"input_private_states item {index} duplicates core_id {state.core_id}"
            )
        if state.state_leaf_id in seen_input_leaf:
            raise V3TransactionPopulationError(
                "duplicate state_leaf_id in input_private_states"
            )
        seen_input_core.add(state.core_id)
        seen_input_leaf.add(state.state_leaf_id)
        if state.model_binding_epoch_id != plan.model_binding_epoch_id:
            raise V3TransactionPopulationError(
                f"input_private_states item {index} has wrong model binding"
            )
        input_by_core[state.core_id] = state
    expected_population = set(online) | {offline}
    if set(input_by_core) != expected_population:
        raise V3TransactionPopulationError(
            "input_private_states population must be online cores plus offline core"
        )
    online_input_by_core = {core_id: input_by_core[core_id] for core_id in online}
    offline_input = input_by_core[offline]

    expected_input_leaves = tuple(
        sorted((state.core_id, state.state_leaf_id) for state in online_input_by_core.values())
    )
    if plan.input_core_state_leaf_ids != expected_input_leaves:
        raise V3TransactionPopulationError("plan input leaf IDs mismatch")
    expected_input_souls = tuple(
        sorted((state.core_id, state.soul_sha256) for state in online_input_by_core.values())
    )
    if plan.input_soul_sha256_by_core != expected_input_souls:
        raise V3TransactionPopulationError("plan input soul hashes mismatch")

    # Duplicate-ID rejection for remaining transaction collections.
    _candidate_ids = set()
    for candidate in transaction.candidate_private_states:
        if candidate.state_leaf_id in _candidate_ids:
            raise V3TransactionArtifactError("duplicate candidate private-state leaf ID")
        _candidate_ids.add(candidate.state_leaf_id)
    _page_ids = set()
    for page in transaction.read_pages:
        if page.page_id in _page_ids:
            raise V3TransactionPageError("duplicate read page ID")
        _page_ids.add(page.page_id)
    _cycle_ids = set()
    for cycle in transaction.read_cycles:
        if cycle.cycle_id in _cycle_ids:
            raise V3TransactionPageError("duplicate read cycle ID")
        _cycle_ids.add(cycle.cycle_id)
    _delta_ids = set()
    for delta in transaction.deltas:
        if delta.delta_id in _delta_ids:
            raise V3TransactionDeltaError("duplicate delta ID")
        _delta_ids.add(delta.delta_id)
    _transition_ids = set()
    for transition in transaction.soul_transitions:
        if transition.transition_id in _transition_ids:
            raise V3TransactionArtifactError("duplicate soul transition ID")
        _transition_ids.add(transition.transition_id)
    _initial_pass_ids = set()
    _initial_pass_cores = set()
    for pass_record in transaction.initial_passes:
        if pass_record.pass_id in _initial_pass_ids:
            raise V3TransactionArtifactError("duplicate INITIAL pass ID")
        if pass_record.core_id in _initial_pass_cores:
            raise V3TransactionArtifactError("duplicate INITIAL pass author")
        _initial_pass_ids.add(pass_record.pass_id)
        _initial_pass_cores.add(pass_record.core_id)
    _refine_pass_ids = set()
    _refine_pass_cores = set()
    for pass_record in transaction.refine_passes:
        if pass_record.pass_id in _refine_pass_ids:
            raise V3TransactionArtifactError("duplicate REFINE pass ID")
        if pass_record.core_id in _refine_pass_cores:
            raise V3TransactionArtifactError("duplicate REFINE pass author")
        _refine_pass_ids.add(pass_record.pass_id)
        _refine_pass_cores.add(pass_record.core_id)
    _disposition_ids = set()
    for disposition in transaction.dispositions:
        if disposition.disposition_id in _disposition_ids:
            raise V3TransactionArtifactError("duplicate disposition ID")
        _disposition_ids.add(disposition.disposition_id)

    # INITIAL/REFINE pass counts must exactly match the online population.
    if len(transaction.initial_passes) != len(online):
        raise V3TransactionArtifactError("INITIAL pass count must equal online count")
    if len(transaction.refine_passes) != len(online):
        raise V3TransactionArtifactError("REFINE pass count must equal online count")

    # Offline core must not author any pass, delta, transition, candidate, or disposition.
    for pass_record in transaction.initial_passes + transaction.refine_passes:
        if pass_record.core_id == offline:
            raise V3TransactionPopulationError("offline core authored a pass")
    for delta in transaction.deltas:
        if delta.author_core_id == offline:
            raise V3TransactionPopulationError("offline core authored a delta")
    for transition in transaction.soul_transitions:
        if transition.core_id == offline:
            raise V3TransactionPopulationError("offline core authored a transition")
    for candidate in transaction.candidate_private_states:
        if candidate.core_id == offline:
            raise V3TransactionPopulationError("offline core authored a candidate state")
    for disposition in transaction.dispositions:
        # Dispositions reference transitions; if no transition belongs to offline,
        # this is already covered, but we check explicitly for completeness.
        pass

    # Exact outer field binding: transaction.system_update, audit.system_update,
    # and plan.system_update_id must be the exact same content-addressed object.
    audit = transaction.field_audit
    if transaction.system_update.update_id != plan.system_update_id:
        raise V3TransactionArtifactError("transaction system_update_id mismatch with plan")
    if transaction.system_update != audit.system_update:
        raise V3TransactionArtifactError(
            "transaction system_update does not match audit system_update"
        )
    if audit.system_update.update_id != plan.system_update_id:
        raise V3TransactionArtifactError("audit system_update_id mismatch with plan")
    if audit.before_snapshot.field_id != plan.input_field_id:
        raise V3TransactionArtifactError("audit before field mismatch")
    if audit.before_snapshot.tick_id != plan.tick_seq:
        raise V3TransactionArtifactError("audit before tick mismatch")

    # Independently apply the system update to the audit before snapshot.
    recomputed_application = apply_system_update(audit.before_snapshot, audit.system_update)
    if recomputed_application.working_snapshot.field_id != plan.working_field_id:
        raise V3TransactionArtifactError("recomputed working field mismatch")
    if recomputed_application.working_snapshot != audit.working_snapshot:
        raise V3TransactionArtifactError(
            "recomputed working snapshot does not match audit.working_snapshot"
        )

    # Independently compose the no-core transaction.
    no_core_audit = compose_tick_transaction(
        audit.before_snapshot, audit.system_update, None
    )
    if no_core_audit.final_snapshot.field_id != plan.no_core_delta_output_field_id:
        raise V3TransactionArtifactError("no-core output field mismatch")

    # Reconstruct a fresh audit from its exact six semantic fields and require
    # exact equality/canonical identity, not merely matching final field IDs.
    reconstructed_audit = compose_tick_transaction(
        audit.before_snapshot,
        audit.system_update,
        audit.consolidator_delta,
        consolidator_author_core_id=audit.consolidator_author_core_id,
    )
    if reconstructed_audit.audit_id != audit.audit_id:
        raise V3TransactionArtifactError("reconstructed audit identity mismatch")
    if reconstructed_audit != audit:
        raise V3TransactionArtifactError(
            "reconstructed audit does not match supplied audit"
        )

    # Independently rebuild the active projection from the working snapshot.
    recomputed_projection = _build_active_projection(audit.working_snapshot)
    if recomputed_projection != transaction.projection:
        raise V3TransactionArtifactError(
            "recomputed active projection does not match transaction.projection"
        )
    if recomputed_projection.projection_id != plan.projection_id:
        raise V3TransactionArtifactError("recomputed projection_id mismatch")

    # Accepted/rejected consolidation semantics.
    final_transition = next(
        t for t in transaction.soul_transitions if t.phase == "FINAL"
    )
    final_delta_id = transaction.consolidation.final_delta_id
    if transaction.consolidation.accepted:
        if audit.consolidator_delta is None:
            raise V3TransactionArtifactError(
                "accepted transaction requires a consolidator delta"
            )
        if audit.consolidator_delta.delta_id != final_delta_id:
            raise V3TransactionArtifactError("audit consolidator delta mismatch")
        if audit.consolidator_author_core_id != consolidator:
            raise V3TransactionArtifactError("audit consolidator author mismatch")
        accepted_audit = compose_tick_transaction(
            audit.before_snapshot,
            audit.system_update,
            audit.consolidator_delta,
            consolidator_author_core_id=consolidator,
        )
        if accepted_audit.final_snapshot != audit.final_snapshot:
            raise V3TransactionArtifactError(
                "accepted audit final snapshot does not match recomposed output"
            )
    else:
        if audit.consolidator_delta is not None:
            raise V3TransactionArtifactError(
                "rejected transaction must not carry a consolidator delta"
            )
        if audit.consolidator_author_core_id is not None:
            raise V3TransactionArtifactError(
                "rejected transaction must not carry a consolidator author"
            )
        if audit.final_snapshot != no_core_audit.final_snapshot:
            raise V3TransactionArtifactError(
                "rejected audit final snapshot does not match no-core output"
            )

    # Commit output field must match the independently replayed field.
    if transaction.commit.output_field_id != audit.final_snapshot.field_id:
        raise V3TransactionArtifactError("commit output field mismatch")

    # Content-addressed artifact resolution.
    candidate_by_id = _artifact_by_id(transaction.candidate_private_states)
    transition_by_id = {
        transition.transition_id: transition
        for transition in transaction.soul_transitions
    }
    referenced_candidate_ids: set[str] = set()
    referenced_transition_ids: set[str] = set()
    for transition in transaction.soul_transitions:
        candidate = candidate_by_id.get(transition.candidate_private_state_id)
        if candidate is None:
            raise V3TransactionArtifactError(
                f"transition references missing candidate private state {transition.candidate_private_state_id}"
            )
        _verify_private_state_against_transition(
            candidate, transition, online_input_by_core, candidate_by_id
        )
        referenced_candidate_ids.add(candidate.state_leaf_id)
        referenced_transition_ids.add(transition.transition_id)
    if referenced_candidate_ids != set(candidate_by_id):
        raise V3TransactionArtifactError("transaction contains orphan candidates")

    # Pass/transition/delta consistency and artifact graph.
    try:
        validate_tick_artifact_graph(
            plan=plan,
            read_cycles=transaction.read_cycles,
            soul_transitions=transaction.soul_transitions,
            initial_passes=transaction.initial_passes,
            initial_board=transaction.initial_board,
            refine_passes=transaction.refine_passes,
            refine_board=transaction.refine_board,
            consolidation=transaction.consolidation,
            dispositions=transaction.dispositions,
            commit=transaction.commit,
        )
    except V3ContractError as exc:
        raise V3TransactionArtifactError(
            f"artifact graph validation failed: {exc}"
        ) from exc

    delta_by_id = _verify_deltas(transaction, online, consolidator)
    initial_delta_by_core = {
        pass_record.core_id: delta_by_id[pass_record.delta_id]
        for pass_record in transaction.initial_passes
    }
    refine_delta_by_core = {
        pass_record.core_id: delta_by_id[pass_record.delta_id]
        for pass_record in transaction.refine_passes
    }
    _verify_read_pages(
        transaction,
        projection_text=transaction.projection.projection_text,
        online=online,
        consolidator=consolidator,
        initial_delta_by_core=initial_delta_by_core,
        refine_delta_by_core=refine_delta_by_core,
    )

    # Field transaction replay.
    replay = replay_field_transaction(audit)
    if replay.field_id != transaction.commit.output_field_id:
        raise V3TransactionReplayError("replayed field ID does not match commit output")
    if replay.field_id != transaction.field_audit.final_field_id:
        raise V3TransactionReplayError("replayed field ID mismatch")
    if transaction.field_audit.audit_id != transaction.commit.field_transaction_audit_id:
        raise V3TransactionReplayError("audit ID mismatch")

    # Final leaves/souls derived from dispositions.
    disposition_by_transition = {
        disp.transition_id: disp for disp in transaction.dispositions
    }
    if set(disposition_by_transition) != referenced_transition_ids:
        raise V3TransactionArtifactError("dispositions do not exactly cover transitions")

    terminal_transition_by_core: dict[str, SoulTransitionRecord] = {}
    for transition in transaction.soul_transitions:
        disposition = disposition_by_transition[transition.transition_id]
        if disposition.disposition == "SUPERSEDED":
            continue
        existing = terminal_transition_by_core.get(transition.core_id)
        if existing is None or transition.substep > existing.substep:
            terminal_transition_by_core[transition.core_id] = transition

    expected_leaves: list[tuple[str, str]] = []
    expected_souls: list[tuple[str, str]] = []
    for core_id in online:
        if core_id not in terminal_transition_by_core:
            raise V3TransactionArtifactError("missing terminal transition")
        terminal = terminal_transition_by_core[core_id]
        disposition = disposition_by_transition[terminal.transition_id]
        if disposition.disposition in {"COMMITTED", "ACCEPTED_NOOP"}:
            expected_leaves.append((core_id, terminal.candidate_private_state_id))
            expected_souls.append((core_id, terminal.output_soul_sha256))
        elif disposition.disposition == "REJECTED_NOT_INSTALLED":
            input_state = online_input_by_core[core_id]
            expected_leaves.append((core_id, input_state.state_leaf_id))
            expected_souls.append((core_id, input_state.soul_sha256))
        else:
            raise V3TransactionArtifactError(
                f"unexpected terminal disposition {disposition.disposition}"
            )
    if transaction.commit.final_core_state_leaf_ids != tuple(sorted(expected_leaves)):
        raise V3TransactionArtifactError("commit final leaves mismatch")

    # Offline head must be preserved: no final leaf references the offline core.
    for core_id, _ in transaction.commit.final_core_state_leaf_ids:
        if core_id == offline:
            raise V3TransactionPopulationError("offline core appears in commit final leaves")

    final_souls = tuple(sorted(expected_souls))
    return TickReplayV3(
        final_snapshot=replay,
        final_core_state_leaf_ids=transaction.commit.final_core_state_leaf_ids,
        final_soul_sha256_by_core=final_souls,
        field_transaction_audit_id=transaction.field_audit.audit_id,
        commit_id=transaction.commit.commit_id,
    )


def replay_tick_transaction_v3(transaction: TickTransactionV3) -> TickReplayV3:
    """Replay the field transaction and return replay evidence."""

    if not isinstance(transaction, TickTransactionV3):
        raise V3TransactionError("transaction must be TickTransactionV3")
    replay_snapshot = replay_field_transaction(transaction.field_audit)
    if replay_snapshot.field_id != transaction.commit.output_field_id:
        raise V3TransactionReplayError("replay output field ID mismatch")
    return validate_tick_transaction_v3(transaction)


__all__ = [
    "V3TransactionError",
    "V3TransactionReplayError",
    "V3TransactionPopulationError",
    "V3TransactionArtifactError",
    "V3TransactionPageError",
    "V3TransactionDeltaError",
    "V3TransactionInvocationError",
    "PrivateStateArtifactV3",
    "ActiveProjectionArtifactV3",
    "ReadPageArtifactV3",
    "SyntheticTickSpec",
    "TickTransactionV3",
    "TickReplayV3",
    "compose_synthetic_tick_v3",
    "validate_tick_transaction_v3",
    "replay_tick_transaction_v3",
]
