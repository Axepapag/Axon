"""Tests for the isolated neural-free v3 protocol slice."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import pytest

from runtime.axon_runtime.v3_contracts import (
    MAX_COLLECTION_ITEMS,
    MAX_ONLINE_CORES,
    PROTOCOL_VERSION,
    ArtifactRejectionRecord,
    ConsolidationRecord,
    CorePassRecord,
    ProposalBoardManifest,
    ReadCycleManifest,
    SoulTransitionDispositionRecord,
    SoulTransitionRecord,
    TickCommitRecordV3,
    TickPhasePlan,
    V3BoardError,
    V3ContractError,
    V3PhaseError,
    V3PopulationError,
    V3SerializationError,
    V3SoulError,
    validate_consolidation_against_plan,
    validate_pass_against_read_cycle,
    validate_tick_artifact_graph,
)
from runtime.axon_runtime.v3_serde import (
    assert_round_trip_identity,
    canonical_json_text,
    deserialize_artifact_rejection,
    deserialize_consolidation,
    deserialize_core_pass,
    deserialize_proposal_board,
    deserialize_read_cycle_manifest,
    deserialize_soul_transition,
    deserialize_soul_transition_disposition,
    deserialize_tick_commit_v3,
    deserialize_tick_phase_plan,
    parse_json_object,
    serialize_artifact_rejection,
    serialize_consolidation,
    serialize_core_pass,
    serialize_proposal_board,
    serialize_read_cycle_manifest,
    serialize_soul_transition,
    serialize_soul_transition_disposition,
    serialize_tick_commit_v3,
    serialize_tick_phase_plan,
)


_HASH = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _plan(
    *,
    online: tuple[str, ...] = ("axon64-a", "axon128-a", "axon64-b"),
    offline: str = "axon128-b",
    consolidator: str = "axon64-a",
) -> TickPhasePlan:
    return TickPhasePlan(
        tick_seq=5,
        protocol_version=PROTOCOL_VERSION,
        input_field_id=_HASH,
        system_update_id="system-update-5",
        working_field_id=_HASH_B,
        no_core_delta_output_field_id=_digest("no-core-delta-output"),
        projection_id=_HASH_C,
        online_core_ids=online,
        input_core_state_leaf_ids=tuple(
            (core_id, f"{core_id}-existing-state")
            for core_id in online
        ),
        input_soul_sha256_by_core=tuple(
            (core_id, _digest(f"{core_id}-s0"))
            for core_id in online
        ),
        offline_core_id=offline,
        consolidator_core_id=consolidator,
        model_binding_epoch_id="binding-epoch-5",
    )


def _read_cycle(
    *,
    core_id: str = "axon64-a",
    phase: str = "INITIAL",
    complete: bool = True,
) -> ReadCycleManifest:
    return ReadCycleManifest(
        core_id=core_id,
        tick_seq=5,
        phase=phase,
        working_field_id=_HASH_B,
        projection_id=_HASH_C,
        board_id=None if phase == "INITIAL" else f"{phase.lower()}-board",
        page_view_hashes=(_HASH,),
        page_character_counts=(1,),
        cursor_chain=(_HASH,),
        expected_character_count=1 if complete else 2,
        coverage_complete=complete,
        selected_span_fingerprint=_HASH_B,
    )


def _soul_transition(
    *,
    core_id: str = "axon64-a",
    phase: str = "INITIAL",
    board_id: str | None = None,
    parent: str | None = None,
) -> SoulTransitionRecord:
    substep = {"INITIAL": 0, "REFINE": 1, "FINAL": 2}[phase]
    return SoulTransitionRecord(
        core_id=core_id,
        tick_seq=5,
        phase=phase,
        substep=substep,
        input_soul_sha256=_HASH,
        output_soul_sha256=_HASH_B,
        working_field_id=_HASH_C,
        projection_id="d" * 64,
        read_cycle_id="read-cycle-5",
        board_id=board_id,
        delta_id="delta-5",
        candidate_private_state_id="private-state-5",
        parent_transition_id=parent,
        cursor_state_sha256="d" * 64,
        rng_state_sha256="e" * 64,
        binding_manifest_id="binding-5",
    )


def _core_pass(
    *,
    core_id: str = "axon64-a",
    phase: str = "INITIAL",
    board_id: str | None = None,
) -> CorePassRecord:
    return CorePassRecord(
        core_id=core_id,
        tick_seq=5,
        phase=phase,
        working_field_id=_HASH_B,
        board_id=board_id,
        read_cycle_id="read-cycle-5",
        delta_id="delta-5",
        soul_transition_id="transition-5",
        candidate_private_state_id="private-state-5",
    )


def _initial_board() -> ProposalBoardManifest:
    return ProposalBoardManifest(
        working_field_id=_HASH_B,
        phase="INITIAL",
        pass_ids_by_author=(
            ("axon64-a", "pass-a-0"),
            ("axon64-b", "pass-b-0"),
            ("axon128-a", "pass-c-0"),
        ),
        required_authors=("axon64-a", "axon64-b", "axon128-a"),
    )


def _refine_board() -> ProposalBoardManifest:
    return ProposalBoardManifest(
        working_field_id=_HASH_B,
        phase="REFINE",
        pass_ids_by_author=(
            ("axon64-a", "pass-a-1"),
            ("axon64-b", "pass-b-1"),
            ("axon128-a", "pass-c-1"),
        ),
        required_authors=("axon64-a", "axon64-b", "axon128-a"),
    )


def _consolidation(*, accepted: bool = True) -> ConsolidationRecord:
    return ConsolidationRecord(
        core_id="axon64-a",
        tick_seq=5,
        working_field_id=_HASH_B,
        refinement_board_id="refine-board-5",
        final_delta_id="final-delta-5",
        final_soul_transition_id="final-transition-5",
        accepted=accepted,
        reason="consolidated refinement board",
        invocation_request_ids=("invoke-1", "invoke-2") if accepted else (),
    )


def _commit() -> TickCommitRecordV3:
    return TickCommitRecordV3(
        tick_seq=5,
        plan_id="plan-5",
        initial_board_id="initial-board-5",
        refine_board_id="refine-board-5",
        initial_pass_ids=(
            ("axon64-a", "pass-a-0"),
            ("axon64-b", "pass-b-0"),
            ("axon128-a", "pass-c-0"),
        ),
        refine_pass_ids=(
            ("axon64-a", "pass-a-1"),
            ("axon64-b", "pass-b-1"),
            ("axon128-a", "pass-c-1"),
        ),
        consolidation_id="consolidation-5",
        disposition_ids_by_transition=(
            ("transition-a", "disposition-a"),
        ),
        final_core_state_leaf_ids=(
            ("axon64-a", "leaf-a"),
            ("axon64-b", "leaf-b"),
            ("axon128-a", "leaf-c"),
        ),
        field_transaction_audit_id=_HASH,
        output_field_id=_HASH_B,
        invocation_request_ids=("invoke-1",),
    )


def _artifact_rejection() -> ArtifactRejectionRecord:
    return ArtifactRejectionRecord(
        artifact_kind="raw_pass_payload",
        reason="malformed JSON and missing core_id",
        source_context={"source": "ingress:5", "bytes": 128},
        raw_fingerprint_sha256="f" * 64,
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _valid_tick_graph(
    *,
    accepted: bool = True,
    online: tuple[str, ...] | None = None,
) -> dict:
    if online is None:
        plan = _plan()
    else:
        plan = _plan(
            online=online,
            offline="offline-core",
            consolidator=online[0],
        )
    initial_cycles: list[ReadCycleManifest] = []
    initial_transitions: dict[str, SoulTransitionRecord] = {}
    initial_passes: list[CorePassRecord] = []
    for core_id in plan.online_core_ids:
        cycle = ReadCycleManifest(
            core_id=core_id,
            tick_seq=plan.tick_seq,
            phase="INITIAL",
            working_field_id=plan.working_field_id,
            projection_id=plan.projection_id,
            board_id=None,
            page_view_hashes=(_digest(f"{core_id}-initial-page"),),
            page_character_counts=(95,),
            cursor_chain=(_digest(f"{core_id}-initial-cursor"),),
            expected_character_count=95,
            coverage_complete=True,
            selected_span_fingerprint=_digest(
                f"{core_id}-initial-selection"
            ),
        )
        transition = SoulTransitionRecord(
            core_id=core_id,
            tick_seq=plan.tick_seq,
            phase="INITIAL",
            substep=0,
            input_soul_sha256=_digest(f"{core_id}-s0"),
            output_soul_sha256=_digest(f"{core_id}-s1"),
            working_field_id=plan.working_field_id,
            projection_id=plan.projection_id,
            read_cycle_id=cycle.cycle_id,
            board_id=None,
            delta_id=f"{core_id}-initial-delta",
            candidate_private_state_id=f"{core_id}-state-s1",
            parent_transition_id=None,
            cursor_state_sha256=_digest(f"{core_id}-initial-state"),
            rng_state_sha256=_digest(f"{core_id}-initial-rng"),
            binding_manifest_id=plan.model_binding_epoch_id,
        )
        pass_record = CorePassRecord(
            core_id=core_id,
            tick_seq=plan.tick_seq,
            phase="INITIAL",
            working_field_id=plan.working_field_id,
            board_id=None,
            read_cycle_id=cycle.cycle_id,
            delta_id=transition.delta_id,
            soul_transition_id=transition.transition_id,
            candidate_private_state_id=transition.candidate_private_state_id,
        )
        initial_cycles.append(cycle)
        initial_transitions[core_id] = transition
        initial_passes.append(pass_record)
    initial_board = ProposalBoardManifest(
        working_field_id=plan.working_field_id,
        phase="INITIAL",
        pass_ids_by_author=tuple(
            (record.core_id, record.pass_id) for record in initial_passes
        ),
        required_authors=plan.online_core_ids,
    )

    refine_cycles: list[ReadCycleManifest] = []
    refine_transitions: dict[str, SoulTransitionRecord] = {}
    refine_passes: list[CorePassRecord] = []
    for core_id in plan.online_core_ids:
        initial_transition = initial_transitions[core_id]
        cycle = ReadCycleManifest(
            core_id=core_id,
            tick_seq=plan.tick_seq,
            phase="REFINE",
            working_field_id=plan.working_field_id,
            projection_id=plan.projection_id,
            board_id=initial_board.board_id,
            page_view_hashes=(_digest(f"{core_id}-refine-page"),),
            page_character_counts=(95,),
            cursor_chain=(_digest(f"{core_id}-refine-cursor"),),
            expected_character_count=95,
            coverage_complete=True,
            selected_span_fingerprint=_digest(
                f"{core_id}-refine-selection"
            ),
        )
        transition = SoulTransitionRecord(
            core_id=core_id,
            tick_seq=plan.tick_seq,
            phase="REFINE",
            substep=1,
            input_soul_sha256=initial_transition.output_soul_sha256,
            output_soul_sha256=_digest(f"{core_id}-s2"),
            working_field_id=plan.working_field_id,
            projection_id=plan.projection_id,
            read_cycle_id=cycle.cycle_id,
            board_id=initial_board.board_id,
            delta_id=f"{core_id}-refine-delta",
            candidate_private_state_id=f"{core_id}-state-s2",
            parent_transition_id=initial_transition.transition_id,
            cursor_state_sha256=_digest(f"{core_id}-refine-state"),
            rng_state_sha256=_digest(f"{core_id}-refine-rng"),
            binding_manifest_id=plan.model_binding_epoch_id,
        )
        pass_record = CorePassRecord(
            core_id=core_id,
            tick_seq=plan.tick_seq,
            phase="REFINE",
            working_field_id=plan.working_field_id,
            board_id=initial_board.board_id,
            read_cycle_id=cycle.cycle_id,
            delta_id=transition.delta_id,
            soul_transition_id=transition.transition_id,
            candidate_private_state_id=transition.candidate_private_state_id,
        )
        refine_cycles.append(cycle)
        refine_transitions[core_id] = transition
        refine_passes.append(pass_record)
    refine_board = ProposalBoardManifest(
        working_field_id=plan.working_field_id,
        phase="REFINE",
        pass_ids_by_author=tuple(
            (record.core_id, record.pass_id) for record in refine_passes
        ),
        required_authors=plan.online_core_ids,
    )

    consolidator_refine = refine_transitions[plan.consolidator_core_id]
    final_cycle = ReadCycleManifest(
        core_id=plan.consolidator_core_id,
        tick_seq=plan.tick_seq,
        phase="FINAL",
        working_field_id=plan.working_field_id,
        projection_id=plan.projection_id,
        board_id=refine_board.board_id,
        page_view_hashes=(_digest("final-page"),),
        page_character_counts=(95,),
        cursor_chain=(_digest("final-cursor"),),
        expected_character_count=95,
        coverage_complete=True,
        selected_span_fingerprint=_digest("final-selection"),
    )
    final_transition = SoulTransitionRecord(
        core_id=plan.consolidator_core_id,
        tick_seq=plan.tick_seq,
        phase="FINAL",
        substep=2,
        input_soul_sha256=consolidator_refine.output_soul_sha256,
        output_soul_sha256=_digest(
            f"{plan.consolidator_core_id}-s3"
        ),
        working_field_id=plan.working_field_id,
        projection_id=plan.projection_id,
        read_cycle_id=final_cycle.cycle_id,
        board_id=refine_board.board_id,
        delta_id="final-delta",
        candidate_private_state_id=(
            f"{plan.consolidator_core_id}-state-s3"
        ),
        parent_transition_id=consolidator_refine.transition_id,
        cursor_state_sha256=_digest("final-state"),
        rng_state_sha256=_digest("final-rng"),
        binding_manifest_id=plan.model_binding_epoch_id,
    )
    invocation_ids = ("invoke-1", "invoke-2") if accepted else ()
    consolidation = ConsolidationRecord(
        core_id=plan.consolidator_core_id,
        tick_seq=plan.tick_seq,
        working_field_id=plan.working_field_id,
        refinement_board_id=refine_board.board_id,
        final_delta_id=final_transition.delta_id,
        final_soul_transition_id=final_transition.transition_id,
        accepted=accepted,
        reason="synthetic graph fixture",
        invocation_request_ids=invocation_ids,
    )

    all_transitions = (
        *initial_transitions.values(),
        *refine_transitions.values(),
        final_transition,
    )
    dispositions: list[SoulTransitionDispositionRecord] = []
    if accepted:
        for core_id in plan.online_core_ids:
            dispositions.append(
                SoulTransitionDispositionRecord(
                    transition_id=initial_transitions[
                        core_id
                    ].transition_id,
                    disposition="SUPERSEDED",
                    reason="refined by the next pass",
                    superseded_by_transition_id=refine_transitions[
                        core_id
                    ].transition_id,
                )
            )
        for core_id in plan.online_core_ids:
            transition = refine_transitions[core_id]
            if core_id == plan.consolidator_core_id:
                dispositions.append(
                    SoulTransitionDispositionRecord(
                        transition_id=transition.transition_id,
                        disposition="SUPERSEDED",
                        reason="consolidator authored FINAL",
                        superseded_by_transition_id=(
                            final_transition.transition_id
                        ),
                    )
                )
            else:
                dispositions.append(
                    SoulTransitionDispositionRecord(
                        transition_id=transition.transition_id,
                        disposition="COMMITTED",
                        reason="terminal online state installed",
                    )
                )
        dispositions.append(
            SoulTransitionDispositionRecord(
                transition_id=final_transition.transition_id,
                disposition="COMMITTED",
                reason="FINAL consolidator state installed",
            )
        )
        final_leaves = tuple(
            (
                core_id,
                (
                    final_transition.candidate_private_state_id
                    if core_id == plan.consolidator_core_id
                    else refine_transitions[
                        core_id
                    ].candidate_private_state_id
                ),
            )
            for core_id in plan.online_core_ids
        )
    else:
        dispositions = [
            SoulTransitionDispositionRecord(
                transition_id=transition.transition_id,
                disposition="REJECTED_NOT_INSTALLED",
                reason="synthetic rejected tick",
            )
            for transition in all_transitions
        ]
        final_leaves = tuple(
            plan.input_core_state_leaf_ids
        )

    commit = TickCommitRecordV3(
        tick_seq=plan.tick_seq,
        plan_id=plan.plan_id,
        initial_board_id=initial_board.board_id,
        refine_board_id=refine_board.board_id,
        initial_pass_ids=initial_board.pass_ids_by_author,
        refine_pass_ids=refine_board.pass_ids_by_author,
        consolidation_id=consolidation.consolidation_id,
        disposition_ids_by_transition=tuple(
            (
                disposition.transition_id,
                disposition.disposition_id,
            )
            for disposition in dispositions
        ),
        final_core_state_leaf_ids=final_leaves,
        field_transaction_audit_id=_digest("field-audit"),
        output_field_id=(
            _digest("output-field")
            if accepted
            else plan.no_core_delta_output_field_id
        ),
        invocation_request_ids=invocation_ids,
    )
    return {
        "plan": plan,
        "read_cycles": (
            *initial_cycles,
            *refine_cycles,
            final_cycle,
        ),
        "soul_transitions": all_transitions,
        "initial_passes": tuple(initial_passes),
        "initial_board": initial_board,
        "refine_passes": tuple(refine_passes),
        "refine_board": refine_board,
        "consolidation": consolidation,
        "dispositions": tuple(dispositions),
        "commit": commit,
    }


# ---------------------------------------------------------------------------
# Deterministic canonical dictionaries and IDs
# ---------------------------------------------------------------------------


def test_canonical_dicts_are_deterministic_and_content_addressed() -> None:
    plan = _plan()
    plan2 = _plan()
    assert plan.to_canonical_dict() == plan2.to_canonical_dict()
    assert plan.plan_id == plan2.plan_id
    assert plan.plan_id.startswith("tick-phase-plan-")

    board = _initial_board()
    board_shuffled = ProposalBoardManifest(
        working_field_id=_HASH_B,
        phase="INITIAL",
        pass_ids_by_author=(
            ("axon128-a", "pass-c-0"),
            ("axon64-a", "pass-a-0"),
            ("axon64-b", "pass-b-0"),
        ),
        required_authors=("axon128-a", "axon64-a", "axon64-b"),
    )
    assert board.to_canonical_dict() == board_shuffled.to_canonical_dict()
    assert board.board_id == board_shuffled.board_id


# ---------------------------------------------------------------------------
# Strict round-trip for every record family
# ---------------------------------------------------------------------------


def test_round_trips_for_all_record_families() -> None:
    assert_round_trip_identity(_plan())
    assert_round_trip_identity(_read_cycle())
    assert_round_trip_identity(_soul_transition())
    assert_round_trip_identity(
        SoulTransitionDispositionRecord(
            transition_id="transition-5",
            disposition="COMMITTED",
            reason="accepted final transition",
        )
    )
    assert_round_trip_identity(_core_pass())
    assert_round_trip_identity(_initial_board())
    assert_round_trip_identity(_refine_board())
    assert_round_trip_identity(_consolidation())
    assert_round_trip_identity(_commit())
    assert_round_trip_identity(_artifact_rejection())


# ---------------------------------------------------------------------------
# Hash/ID tampering rejection
# ---------------------------------------------------------------------------


def test_derived_id_tampering_is_rejected() -> None:
    plan = _plan()
    serialized = serialize_tick_phase_plan(plan)
    serialized["plan_id"] = serialized["plan_id"][:-1] + "x"
    with pytest.raises(V3SerializationError, match="tick phase plan ID/hash mismatch"):
        deserialize_tick_phase_plan(serialized)

    board = _initial_board()
    serialized = serialize_proposal_board(board)
    serialized["board_id"] = "proposal-board-" + "0" * 64
    with pytest.raises(V3SerializationError, match="proposal board ID/hash mismatch"):
        deserialize_proposal_board(serialized)

    commit = _commit()
    serialized = serialize_tick_commit_v3(commit)
    serialized["commit_id"] = "tick-commit-v3-" + "0" * 64
    with pytest.raises(V3SerializationError, match="tick commit v3 ID/hash mismatch"):
        deserialize_tick_commit_v3(serialized)


# ---------------------------------------------------------------------------
# Missing/extra key rejection
# ---------------------------------------------------------------------------


def test_missing_and_extra_keys_are_rejected() -> None:
    plan = _plan()
    serialized = serialize_tick_phase_plan(plan)
    missing = dict(serialized)
    missing.pop("offline_core_id")
    with pytest.raises(V3SerializationError, match="fields mismatch"):
        deserialize_tick_phase_plan(missing)

    extra = dict(serialized)
    extra["extra_field"] = "forbidden"
    with pytest.raises(V3SerializationError, match="fields mismatch"):
        deserialize_tick_phase_plan(extra)


# ---------------------------------------------------------------------------
# Population invariants
# ---------------------------------------------------------------------------


def test_duplicate_online_core_rejection() -> None:
    with pytest.raises(V3PopulationError, match="duplicates"):
        _plan(online=("axon64-a", "axon64-a", "axon128-a"))


def test_offline_core_participation_rejection() -> None:
    with pytest.raises(V3PopulationError, match="offline core cannot be online"):
        _plan(online=("axon64-a", "axon128-a", "axon128-b"), offline="axon128-b")


def test_wrong_consolidator_rejection() -> None:
    with pytest.raises(V3PopulationError, match="consolidator must be online"):
        _plan(consolidator="axon128-b")


# ---------------------------------------------------------------------------
# Board author invariants
# ---------------------------------------------------------------------------


def test_missing_or_duplicate_board_author_rejection() -> None:
    with pytest.raises(V3BoardError, match="exactly match"):
        ProposalBoardManifest(
            working_field_id=_HASH_B,
            phase="INITIAL",
            pass_ids_by_author=(
                ("axon64-a", "pass-a-0"),
                ("axon64-b", "pass-b-0"),
            ),
            required_authors=("axon64-a", "axon64-b", "axon128-a"),
        )

    with pytest.raises(V3ContractError, match="duplicate keys"):
        ProposalBoardManifest(
            working_field_id=_HASH_B,
            phase="INITIAL",
            pass_ids_by_author=(
                ("axon64-a", "pass-a-0"),
                ("axon64-a", "pass-a-1"),
            ),
            required_authors=("axon64-a",),
        )


# ---------------------------------------------------------------------------
# INITIAL/REFINE parent-rule rejection
# ---------------------------------------------------------------------------


def test_initial_pass_cannot_claim_board_parent() -> None:
    with pytest.raises(V3PhaseError, match="INITIAL pass cannot claim"):
        _core_pass(phase="INITIAL", board_id="initial-board-5")


def test_refine_pass_requires_initial_board_parent() -> None:
    with pytest.raises(V3PhaseError, match="REFINE pass requires"):
        _core_pass(phase="REFINE", board_id=None)


# ---------------------------------------------------------------------------
# Read-cycle completeness
# ---------------------------------------------------------------------------


def test_incomplete_read_cycle_is_representable_but_not_usable() -> None:
    incomplete = _read_cycle(complete=False)
    assert incomplete.coverage_complete is False
    assert "read-cycle-" in incomplete.cycle_id

    complete_cycle = _read_cycle(complete=True)
    pass_record = CorePassRecord(
        core_id=complete_cycle.core_id,
        tick_seq=complete_cycle.tick_seq,
        phase=complete_cycle.phase,
        working_field_id=_HASH_B,
        board_id=None,
        read_cycle_id=complete_cycle.cycle_id,
        delta_id="delta-5",
        soul_transition_id="transition-5",
        candidate_private_state_id="private-state-5",
    )
    validate_pass_against_read_cycle(pass_record, complete_cycle)

    incomplete_pass = CorePassRecord(
        core_id=incomplete.core_id,
        tick_seq=incomplete.tick_seq,
        phase=incomplete.phase,
        working_field_id=_HASH_B,
        board_id=None,
        read_cycle_id=incomplete.cycle_id,
        delta_id="delta-5",
        soul_transition_id="transition-5",
        candidate_private_state_id="private-state-5",
    )
    with pytest.raises(V3ContractError, match="incomplete read cycle"):
        validate_pass_against_read_cycle(incomplete_pass, incomplete)


# ---------------------------------------------------------------------------
# Soul transition phase/substep/board consistency
# ---------------------------------------------------------------------------


def test_wrong_soul_transition_parent_substep_rejection() -> None:
    with pytest.raises(V3PhaseError, match="INITIAL transition substep must be 0"):
        SoulTransitionRecord(
            core_id="axon64-a",
            tick_seq=5,
            phase="INITIAL",
            substep=1,
            input_soul_sha256=_HASH,
            output_soul_sha256=_HASH_B,
            working_field_id=_HASH_C,
            projection_id="projection-5",
            read_cycle_id="read-cycle-5",
            board_id=None,
            delta_id="delta-5",
            candidate_private_state_id="private-state-5",
            parent_transition_id=None,
            cursor_state_sha256="d" * 64,
            rng_state_sha256="e" * 64,
            binding_manifest_id="binding-5",
        )

    with pytest.raises(V3PhaseError, match="REFINE transition substep must be 1"):
        SoulTransitionRecord(
            core_id="axon64-a",
            tick_seq=5,
            phase="REFINE",
            substep=0,
            input_soul_sha256=_HASH,
            output_soul_sha256=_HASH_B,
            working_field_id=_HASH_C,
            projection_id="d" * 64,
            read_cycle_id="read-cycle-5",
            board_id="initial-board-5",
            delta_id="delta-5",
            candidate_private_state_id="private-state-5",
            parent_transition_id=None,
            cursor_state_sha256="d" * 64,
            rng_state_sha256="e" * 64,
            binding_manifest_id="binding-5",
        )

    with pytest.raises(V3PhaseError, match="INITIAL transition cannot claim"):
        _soul_transition(phase="INITIAL", board_id="initial-board-5")


# ---------------------------------------------------------------------------
# Non-final invocation authority is structurally impossible
# ---------------------------------------------------------------------------


def test_only_final_consolidation_and_commit_carry_invocations() -> None:
    plan = _plan()
    assert "invocation" not in plan.to_canonical_dict()

    board = _initial_board()
    assert "invocation" not in board.to_canonical_dict()

    pass_record = _core_pass()
    assert "invocation" not in pass_record.to_canonical_dict()

    consolidation = _consolidation()
    assert consolidation.invocation_request_ids == ("invoke-1", "invoke-2")

    commit = _commit()
    assert commit.invocation_request_ids == ("invoke-1",)

    rejected = _consolidation(accepted=False)
    with pytest.raises(V3ContractError, match="rejected consolidation"):
        ConsolidationRecord(
            core_id=rejected.core_id,
            tick_seq=rejected.tick_seq,
            working_field_id=rejected.working_field_id,
            refinement_board_id=rejected.refinement_board_id,
            final_delta_id=rejected.final_delta_id,
            final_soul_transition_id=rejected.final_soul_transition_id,
            accepted=False,
            reason=rejected.reason,
            invocation_request_ids=("invoke-1",),
        )


# ---------------------------------------------------------------------------
# Semantic rejection via disposition record
# ---------------------------------------------------------------------------


def test_semantic_rejection_uses_disposition_record() -> None:
    disposition = SoulTransitionDispositionRecord(
        transition_id="transition-5",
        disposition="REJECTED_NOT_INSTALLED",
        reason="final consolidation rejected the delta",
    )
    assert disposition.disposition == "REJECTED_NOT_INSTALLED"
    assert disposition.disposition_id.startswith("soul-disposition-")
    assert_round_trip_identity(disposition)

    superseded = SoulTransitionDispositionRecord(
        transition_id="transition-5",
        disposition="SUPERSEDED",
        reason="overtaken by a later tick",
        superseded_by_transition_id="transition-6",
    )
    assert_round_trip_identity(superseded)


# ---------------------------------------------------------------------------
# Structural invalidity uses artifact rejection record
# ---------------------------------------------------------------------------


def test_structural_invalidity_uses_artifact_rejection() -> None:
    rejection = _artifact_rejection()
    assert rejection.rejection_id.startswith("artifact-rejection-")
    assert_round_trip_identity(rejection)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_consolidation_must_match_plan_consolidator() -> None:
    plan = _plan(consolidator="axon128-a")
    consolidation = _consolidation()
    with pytest.raises(V3ContractError, match="consolidator core_id"):
        validate_consolidation_against_plan(
            consolidation,
            plan,
            _refine_board(),
        )

    graph = _valid_tick_graph()
    validate_consolidation_against_plan(
        graph["consolidation"],
        graph["plan"],
        graph["refine_board"],
    )


# ---------------------------------------------------------------------------
# Strict input validation edge cases
# ---------------------------------------------------------------------------


def test_exact_nonnegative_int_rejects_booleans() -> None:
    with pytest.raises(V3ContractError, match="non-negative integer"):
        _plan().to_canonical_dict()
        TickPhasePlan(
            tick_seq=True,
            protocol_version=PROTOCOL_VERSION,
            input_field_id=_HASH,
            system_update_id="system-update-5",
            working_field_id=_HASH_B,
            no_core_delta_output_field_id=_HASH_C,
            projection_id=_HASH_C,
            online_core_ids=("axon64-a",),
            input_core_state_leaf_ids=(("axon64-a", "state-a"),),
            input_soul_sha256_by_core=(("axon64-a", _HASH),),
            offline_core_id="axon128-b",
            consolidator_core_id="axon64-a",
            model_binding_epoch_id="binding-5",
        )


def test_sha256_fields_require_exactly_64_hex() -> None:
    with pytest.raises(V3ContractError, match="64-character SHA-256"):
        TickPhasePlan(
            tick_seq=5,
            protocol_version=PROTOCOL_VERSION,
            input_field_id="not-a-hash",
            system_update_id="system-update-5",
            working_field_id=_HASH_B,
            no_core_delta_output_field_id=_HASH_C,
            projection_id=_HASH_C,
            online_core_ids=("axon64-a",),
            input_core_state_leaf_ids=(("axon64-a", "state-a"),),
            input_soul_sha256_by_core=(("axon64-a", _HASH),),
            offline_core_id="axon128-b",
            consolidator_core_id="axon64-a",
            model_binding_epoch_id="binding-5",
        )


def test_noncanonical_population_order_is_preserved() -> None:
    ordered = ("axon64-a", "axon128-a", "axon64-b")
    plan = _plan(online=ordered)
    assert plan.online_core_ids == ordered

    reversed_plan = _plan(online=tuple(reversed(ordered)))
    assert reversed_plan.online_core_ids == tuple(reversed(ordered))
    assert reversed_plan.plan_id != plan.plan_id


# ---------------------------------------------------------------------------
# Serde unknown-schema and wrong-protocol-version rejection
# ---------------------------------------------------------------------------


def test_deserializers_reject_unknown_schema_strings() -> None:
    plan = _plan()
    serialized = serialize_tick_phase_plan(plan)
    serialized["schema"] = "axon-tick-phase-plan-v99"
    with pytest.raises(V3SerializationError, match="unknown tick phase plan schema"):
        deserialize_tick_phase_plan(serialized)


def test_deserializer_rejects_wrong_protocol_version() -> None:
    plan = _plan()
    serialized = serialize_tick_phase_plan(plan)
    serialized["protocol_version"] = 2
    with pytest.raises(V3ContractError, match="protocol_version must be"):
        deserialize_tick_phase_plan(serialized)


def test_serde_rejects_noncanonical_aliases() -> None:
    plan_wire = serialize_tick_phase_plan(_plan())
    uppercase_id = copy.deepcopy(plan_wire)
    uppercase_id["plan_id"] = uppercase_id["plan_id"].upper()
    with pytest.raises(V3SerializationError, match="ID/hash mismatch"):
        deserialize_tick_phase_plan(uppercase_id)

    uppercase_sha = copy.deepcopy(plan_wire)
    uppercase_sha["input_field_id"] = uppercase_sha[
        "input_field_id"
    ].upper()
    with pytest.raises(V3SerializationError, match="canonical serialized form"):
        deserialize_tick_phase_plan(uppercase_sha)

    cycle_wire = serialize_read_cycle_manifest(_read_cycle())
    lowercase_phase = copy.deepcopy(cycle_wire)
    lowercase_phase["phase"] = "initial"
    with pytest.raises(V3SerializationError, match="canonical serialized form"):
        deserialize_read_cycle_manifest(lowercase_phase)

    board_wire = serialize_proposal_board(_initial_board())
    unsorted_authors = copy.deepcopy(board_wire)
    unsorted_authors["required_authors"].reverse()
    with pytest.raises(V3SerializationError, match="canonical serialized form"):
        deserialize_proposal_board(unsorted_authors)

    consolidation_wire = serialize_consolidation(_consolidation())
    unsorted_invocations = copy.deepcopy(consolidation_wire)
    unsorted_invocations["invocation_request_ids"].reverse()
    with pytest.raises(V3SerializationError, match="canonical serialized form"):
        deserialize_consolidation(unsorted_invocations)

    disposition_wire = serialize_soul_transition_disposition(
        SoulTransitionDispositionRecord(
            transition_id="transition-a",
            disposition="REJECTED_NOT_INSTALLED",
            reason="rejected",
        )
    )
    lowercase_disposition = copy.deepcopy(disposition_wire)
    lowercase_disposition["disposition"] = "rejected_not_installed"
    with pytest.raises(V3SerializationError, match="canonical serialized form"):
        deserialize_soul_transition_disposition(lowercase_disposition)


def test_strict_json_parser_rejects_duplicate_keys_and_constants() -> None:
    with pytest.raises(V3SerializationError, match="duplicate key"):
        parse_json_object('{"schema":"first","schema":"second"}')
    with pytest.raises(V3SerializationError, match="non-finite"):
        parse_json_object('{"value":NaN}')
    assert parse_json_object('{"items":[],"schema":"ok"}') == {
        "schema": "ok",
        "items": [],
    }


def test_strict_json_parser_wraps_limits_and_rejects_lone_surrogates() -> None:
    with pytest.raises(V3SerializationError, match="signed 64-bit"):
        parse_json_object('{"value":' + ("9" * 5000) + "}")
    with pytest.raises(V3SerializationError, match="depth limit"):
        parse_json_object(
            ('{"nested":' * 1500) + "null" + ("}" * 1500)
        )
    with pytest.raises(V3SerializationError, match="valid UTF-8"):
        parse_json_object('{"value":"\\ud800"}')
    with pytest.raises(V3SerializationError, match="valid UTF-8"):
        parse_json_object('{"\\ud800":"value"}')


def test_strict_json_parser_requires_exact_canonical_utf8_bytes() -> None:
    canonical = canonical_json_text(
        serialize_tick_phase_plan(_plan())
    )
    assert deserialize_tick_phase_plan(parse_json_object(canonical)) == _plan()
    assert deserialize_tick_phase_plan(
        parse_json_object(canonical.encode("utf-8"))
    ) == _plan()

    noncanonical_values = (
        canonical.replace("{", "{ ", 1),
        "\ufeff" + canonical,
        canonical.encode("utf-8-sig"),
        canonical.encode("utf-16"),
        '{"schema":"ok","items":[]}',
        '{"value":"\\u0061"}',
    )
    for serialized in noncanonical_values:
        with pytest.raises(V3SerializationError):
            parse_json_object(serialized)


def test_protocol_integers_and_json_depth_are_explicitly_bounded() -> None:
    with pytest.raises(V3ContractError, match="signed 64-bit range"):
        replace(_plan(), tick_seq=2**63)
    with pytest.raises(V3ContractError, match="signed 64-bit range"):
        replace(_read_cycle(), expected_character_count=2**63)

    cyclic_wire = serialize_tick_phase_plan(_plan())
    cyclic_wire["cycle"] = cyclic_wire
    with pytest.raises(V3SerializationError, match="JSON cycle"):
        deserialize_tick_phase_plan(cyclic_wire)

    deep: dict[str, object] = {}
    cursor = deep
    for _ in range(129):
        nested: dict[str, object] = {}
        cursor["nested"] = nested
        cursor = nested
    with pytest.raises(V3SerializationError, match="depth limit"):
        deserialize_artifact_rejection(deep)

    cyclic_context: dict[str, object] = {}
    cyclic_context["self"] = cyclic_context
    with pytest.raises(V3ContractError, match="JSON cycle"):
        ArtifactRejectionRecord(
            artifact_kind="raw",
            reason="cyclic context",
            source_context=cyclic_context,
            raw_fingerprint_sha256=_HASH,
        )


def test_constructor_and_parser_collection_budgets_are_round_trip_aligned() -> None:
    online = tuple(
        f"core-{index:04d}"
        for index in range(MAX_ONLINE_CORES)
    )
    boundary_plan = _plan(
        online=online,
        offline="offline-core",
        consolidator=online[0],
    )
    serialized = canonical_json_text(
        serialize_tick_phase_plan(boundary_plan)
    )
    assert deserialize_tick_phase_plan(
        parse_json_object(serialized)
    ) == boundary_plan

    over_limit = (*online, "one-core-too-many")
    with pytest.raises(V3PopulationError, match="online-core limit"):
        _plan(
            online=over_limit,
            offline="offline-core",
            consolidator=over_limit[0],
        )
    with pytest.raises(V3ContractError, match="collection-item limit"):
        ArtifactRejectionRecord(
            artifact_kind="raw",
            reason="oversized context",
            source_context={
                f"key-{index}": index
                for index in range(MAX_COLLECTION_ITEMS + 1)
            },
            raw_fingerprint_sha256=_HASH,
        )

    boundary_graph = _valid_tick_graph(online=online)
    assert len(boundary_graph["soul_transitions"]) == (
        2 * MAX_ONLINE_CORES + 1
    )
    validate_tick_artifact_graph(**boundary_graph)

    oversized_node_context = {
        f"key-{index}": list(range(12))
        for index in range(MAX_COLLECTION_ITEMS)
    }
    with pytest.raises(V3ContractError, match="JSON node limit"):
        ArtifactRejectionRecord(
            artifact_kind="raw",
            reason="oversized context tree",
            source_context=oversized_node_context,
            raw_fingerprint_sha256=_HASH,
        )

    deepest_context: dict[str, object] = {}
    cursor = deepest_context
    for _ in range(127):
        nested = {}
        cursor["nested"] = nested
        cursor = nested
    with pytest.raises(V3ContractError, match="JSON depth limit"):
        ArtifactRejectionRecord(
            artifact_kind="raw",
            reason="context leaves no envelope depth",
            source_context=deepest_context,
            raw_fingerprint_sha256=_HASH,
        )


def test_serde_and_constructors_reject_builtin_subclasses() -> None:
    class SneakyInt(int):
        pass

    class SneakyString(str):
        pass

    class SneakyDict(dict):
        pass

    plan_wire = serialize_tick_phase_plan(_plan())
    sneaky_int = copy.deepcopy(plan_wire)
    sneaky_int["tick_seq"] = SneakyInt(5)
    with pytest.raises(V3SerializationError, match="canonical JSON type"):
        deserialize_tick_phase_plan(sneaky_int)

    sneaky_string = copy.deepcopy(plan_wire)
    sneaky_string["plan_id"] = SneakyString(plan_wire["plan_id"])
    with pytest.raises(V3SerializationError, match="canonical JSON type"):
        deserialize_tick_phase_plan(sneaky_string)

    with pytest.raises(V3SerializationError, match="JSON object"):
        deserialize_tick_phase_plan(SneakyDict(plan_wire))

    with pytest.raises(V3ContractError, match="non-negative integer"):
        TickPhasePlan(
            tick_seq=SneakyInt(5),
            protocol_version=PROTOCOL_VERSION,
            input_field_id=_HASH,
            system_update_id="system-update-5",
            working_field_id=_HASH_B,
            no_core_delta_output_field_id=_HASH_C,
            projection_id=_HASH_C,
            online_core_ids=("axon64-a",),
            input_core_state_leaf_ids=(("axon64-a", "state-a"),),
            input_soul_sha256_by_core=(("axon64-a", _HASH),),
            offline_core_id="axon128-b",
            consolidator_core_id="axon64-a",
            model_binding_epoch_id="binding-5",
        )


def test_serializer_rejects_record_subclasses_and_invalid_utf8_text() -> None:
    class TickPhasePlanSubclass(TickPhasePlan):
        def to_dict(self) -> dict:
            return {"schema": "evil"}

    plan = _plan()
    subclass = TickPhasePlanSubclass(
        tick_seq=plan.tick_seq,
        protocol_version=plan.protocol_version,
        input_field_id=plan.input_field_id,
        system_update_id=plan.system_update_id,
        working_field_id=plan.working_field_id,
        no_core_delta_output_field_id=plan.no_core_delta_output_field_id,
        projection_id=plan.projection_id,
        online_core_ids=plan.online_core_ids,
        input_core_state_leaf_ids=plan.input_core_state_leaf_ids,
        input_soul_sha256_by_core=plan.input_soul_sha256_by_core,
        offline_core_id=plan.offline_core_id,
        consolidator_core_id=plan.consolidator_core_id,
        model_binding_epoch_id=plan.model_binding_epoch_id,
    )
    with pytest.raises(TypeError, match="requires TickPhasePlan"):
        serialize_tick_phase_plan(subclass)

    with pytest.raises(V3ContractError, match="valid UTF-8"):
        replace(plan, system_update_id="\ud800")


# ---------------------------------------------------------------------------
# Deep copy and immutability
# ---------------------------------------------------------------------------


def test_frozen_records_are_immutable() -> None:
    plan = _plan()
    with pytest.raises(AttributeError):
        plan.tick_seq = 99

    rejection = _artifact_rejection()
    original = rejection.to_dict()
    mutated = copy.deepcopy(original)
    mutated["source_context"]["extra"] = "injected"
    deserialized = deserialize_artifact_rejection(original)
    assert deserialized.source_context == {"source": "ingress:5", "bytes": 128}


def test_artifact_rejection_context_is_deeply_immutable() -> None:
    rejection = ArtifactRejectionRecord(
        artifact_kind="raw",
        reason="invalid",
        source_context={"nested": {"values": [1, 2]}},
        raw_fingerprint_sha256=_HASH,
    )
    original_id = rejection.rejection_id

    with pytest.raises(TypeError):
        rejection.source_context["new"] = 3
    with pytest.raises(TypeError):
        rejection.source_context["nested"]["new"] = 3
    with pytest.raises(AttributeError):
        rejection.source_context["nested"]["values"].append(3)
    with pytest.raises(TypeError):
        dict.__setitem__(rejection.source_context, "forced", 4)

    assert rejection.rejection_id == original_id
    assert rejection.to_canonical_dict()["source_context"] == {
        "nested": {"values": [1, 2]}
    }


def test_unordered_constructor_inputs_are_rejected() -> None:
    with pytest.raises(V3PopulationError, match="ordered sequence"):
        _plan(online={"axon64-a", "axon128-a", "axon64-b"})

    with pytest.raises(V3ContractError, match="ordered sequence"):
        ReadCycleManifest(
            core_id="axon64-a",
            tick_seq=5,
            phase="INITIAL",
            working_field_id=_HASH_B,
            projection_id=_HASH,
            board_id=None,
            page_view_hashes={_HASH, _HASH_B},
            page_character_counts=(1, 1),
            cursor_chain=(_HASH,),
            expected_character_count=2,
            coverage_complete=True,
            selected_span_fingerprint=_HASH_C,
        )


def test_soul_transition_parent_rules_are_fail_closed() -> None:
    graph = _valid_tick_graph()
    initial = next(
        transition
        for transition in graph["soul_transitions"]
        if transition.phase == "INITIAL"
    )
    refine = next(
        transition
        for transition in graph["soul_transitions"]
        if transition.phase == "REFINE"
    )
    final = next(
        transition
        for transition in graph["soul_transitions"]
        if transition.phase == "FINAL"
    )

    with pytest.raises(V3PhaseError, match="cannot claim a transition parent"):
        replace(initial, parent_transition_id="unexpected-parent")
    with pytest.raises(V3PhaseError, match="initial-board parent"):
        replace(refine, board_id=None)
    with pytest.raises(V3PhaseError, match="INITIAL transition parent"):
        replace(refine, parent_transition_id=None)
    with pytest.raises(V3PhaseError, match="refinement-board parent"):
        replace(final, board_id=None)
    with pytest.raises(V3PhaseError, match="REFINE transition parent"):
        replace(final, parent_transition_id=None)


def test_disposition_supersession_rules_are_fail_closed() -> None:
    with pytest.raises(V3SoulError, match="requires"):
        SoulTransitionDispositionRecord(
            transition_id="transition-a",
            disposition="SUPERSEDED",
            reason="missing successor",
        )
    with pytest.raises(V3SoulError, match="same transition"):
        SoulTransitionDispositionRecord(
            transition_id="transition-a",
            disposition="SUPERSEDED",
            reason="self cycle",
            superseded_by_transition_id="transition-a",
        )
    with pytest.raises(V3SoulError, match="only SUPERSEDED"):
        SoulTransitionDispositionRecord(
            transition_id="transition-a",
            disposition="COMMITTED",
            reason="wrong successor field",
            superseded_by_transition_id="transition-b",
        )


def test_empty_or_duplicate_board_and_commit_references_are_rejected() -> None:
    with pytest.raises(V3BoardError, match="at least one"):
        ProposalBoardManifest(
            working_field_id=_HASH,
            phase="REFINE",
            pass_ids_by_author=(),
            required_authors=(),
        )
    with pytest.raises(V3ContractError, match="duplicate values"):
        ProposalBoardManifest(
            working_field_id=_HASH,
            phase="INITIAL",
            pass_ids_by_author=(
                ("axon64-a", "same-pass"),
                ("axon64-b", "same-pass"),
            ),
            required_authors=("axon64-a", "axon64-b"),
        )
    with pytest.raises(V3ContractError, match="at least one"):
        TickCommitRecordV3(
            tick_seq=5,
            plan_id="plan",
            initial_board_id="initial",
            refine_board_id="refine",
            initial_pass_ids=(),
            refine_pass_ids=(),
            consolidation_id="consolidation",
            disposition_ids_by_transition=(
                ("transition", "disposition"),
            ),
            final_core_state_leaf_ids=(),
            field_transaction_audit_id=_HASH,
            output_field_id=_HASH_B,
        )


def test_read_cycle_seals_effective_input_and_character_coverage() -> None:
    initial = _read_cycle()
    assert initial.working_field_id == _HASH_B
    assert initial.board_id is None
    assert initial.page_character_counts == (1,)
    assert initial.expected_character_count == 1

    with pytest.raises(V3PhaseError, match="cannot reference"):
        replace(initial, board_id="unexpected-board")
    with pytest.raises(V3ContractError, match="equal lengths"):
        replace(initial, cursor_chain=())
    with pytest.raises(V3ContractError, match="every expected character"):
        replace(
            initial,
            page_character_counts=(),
            page_view_hashes=(),
            cursor_chain=(),
        )
    with pytest.raises(V3ContractError, match="more characters"):
        replace(initial, page_character_counts=(2,))
    with pytest.raises(V3ContractError, match="cannot cover every"):
        replace(initial, coverage_complete=False)

    refine_cycle = _read_cycle(phase="REFINE")
    wrong_board_pass = _core_pass(
        phase="REFINE",
        board_id="another-board",
    )
    with pytest.raises(V3ContractError, match="board parent"):
        validate_pass_against_read_cycle(
            wrong_board_pass,
            refine_cycle,
        )


@pytest.mark.parametrize("accepted", [True, False])
def test_complete_tick_artifact_graph_accepts_valid_fixture(
    accepted: bool,
) -> None:
    validate_tick_artifact_graph(**_valid_tick_graph(accepted=accepted))


def test_tick_graph_binds_prestate_rejection_and_dispositions() -> None:
    accepted_graph = _valid_tick_graph()
    plan = accepted_graph["plan"]
    first_author, second_author = plan.online_core_ids[:2]
    input_souls = dict(plan.input_soul_sha256_by_core)
    input_souls[first_author] = input_souls[second_author]
    donor_plan = replace(
        plan,
        input_soul_sha256_by_core=tuple(input_souls.items()),
    )
    donor_graph = dict(accepted_graph)
    donor_graph["plan"] = donor_plan
    donor_graph["commit"] = replace(
        accepted_graph["commit"],
        plan_id=donor_plan.plan_id,
    )
    with pytest.raises(V3ContractError, match="sealed prestate"):
        validate_tick_artifact_graph(**donor_graph)

    rejected_graph = _valid_tick_graph(accepted=False)
    terminal_by_author = {
        transition.core_id: transition
        for transition in rejected_graph["soul_transitions"]
        if transition.phase == "REFINE"
    }
    terminal_by_author[rejected_graph["plan"].consolidator_core_id] = next(
        transition
        for transition in rejected_graph["soul_transitions"]
        if transition.phase == "FINAL"
    )
    installed_rejected = dict(rejected_graph)
    installed_rejected["commit"] = replace(
        rejected_graph["commit"],
        final_core_state_leaf_ids=tuple(
            (
                author,
                transition.candidate_private_state_id,
            )
            for author, transition in terminal_by_author.items()
        ),
    )
    with pytest.raises(V3ContractError, match="preserve sealed input"):
        validate_tick_artifact_graph(**installed_rejected)

    changed_field = dict(rejected_graph)
    changed_field["commit"] = replace(
        rejected_graph["commit"],
        output_field_id=_digest("not-the-working-field"),
    )
    with pytest.raises(V3ContractError, match="no-core-delta successor"):
        validate_tick_artifact_graph(**changed_field)

    changed_dispositions = list(accepted_graph["dispositions"])
    changed_dispositions[0] = replace(
        changed_dispositions[0],
        reason="different durable outcome",
    )
    unsealed_disposition = dict(accepted_graph)
    unsealed_disposition["dispositions"] = tuple(changed_dispositions)
    with pytest.raises(V3ContractError, match="disposition references"):
        validate_tick_artifact_graph(**unsealed_disposition)


def test_tick_graph_rejects_false_accepted_noop() -> None:
    graph = _valid_tick_graph()
    terminal = next(
        transition
        for transition in graph["soul_transitions"]
        if (
            transition.phase == "REFINE"
            and transition.core_id != graph["plan"].consolidator_core_id
        )
    )
    dispositions = tuple(
        (
            SoulTransitionDispositionRecord(
                transition_id=disposition.transition_id,
                disposition="ACCEPTED_NOOP",
                reason="claims unchanged state",
            )
            if disposition.transition_id == terminal.transition_id
            else disposition
        )
        for disposition in graph["dispositions"]
    )
    false_noop = dict(graph)
    false_noop["dispositions"] = dispositions
    false_noop["commit"] = replace(
        graph["commit"],
        disposition_ids_by_transition=tuple(
            (
                disposition.transition_id,
                disposition.disposition_id,
            )
            for disposition in dispositions
        ),
    )
    with pytest.raises(V3ContractError, match="ACCEPTED_NOOP"):
        validate_tick_artifact_graph(**false_noop)


def test_tick_graph_rejects_population_board_and_reference_corruption() -> None:
    graph = _valid_tick_graph()

    missing_pass = dict(graph)
    missing_pass["initial_passes"] = graph["initial_passes"][:-1]
    with pytest.raises(V3ContractError, match="exactly one pass"):
        validate_tick_artifact_graph(**missing_pass)

    offline_board_graph = dict(graph)
    offline_board_graph["initial_board"] = ProposalBoardManifest(
        working_field_id=graph["plan"].working_field_id,
        phase="INITIAL",
        pass_ids_by_author=(("axon128-b", "offline-pass"),),
        required_authors=("axon128-b",),
    )
    with pytest.raises(V3ContractError, match="online population"):
        validate_tick_artifact_graph(**offline_board_graph)

    wrong_working = dict(graph)
    first_pass = replace(
        graph["initial_passes"][0],
        working_field_id=_HASH_C,
    )
    wrong_working["initial_passes"] = (
        first_pass,
        *graph["initial_passes"][1:],
    )
    with pytest.raises(V3ContractError, match="working field"):
        validate_tick_artifact_graph(**wrong_working)


def test_tick_graph_rejects_soul_final_commit_and_orphan_corruption() -> None:
    graph = _valid_tick_graph()
    final_index = next(
        index
        for index, transition in enumerate(graph["soul_transitions"])
        if transition.phase == "FINAL"
    )
    bad_final = replace(
        graph["soul_transitions"][final_index],
        parent_transition_id="wrong-refine-transition",
    )
    wrong_chain = dict(graph)
    wrong_chain["soul_transitions"] = tuple(
        bad_final if index == final_index else transition
        for index, transition in enumerate(graph["soul_transitions"])
    )
    wrong_chain["consolidation"] = replace(
        graph["consolidation"],
        final_soul_transition_id=bad_final.transition_id,
    )
    wrong_chain["commit"] = replace(
        graph["commit"],
        consolidation_id=wrong_chain["consolidation"].consolidation_id,
    )
    with pytest.raises(V3ContractError, match="transition parent"):
        validate_tick_artifact_graph(**wrong_chain)

    wrong_invocation = dict(graph)
    wrong_invocation["commit"] = replace(
        graph["commit"],
        invocation_request_ids=("different-invocation",),
    )
    with pytest.raises(V3ContractError, match="invocation requests"):
        validate_tick_artifact_graph(**wrong_invocation)

    wrong_leaf = dict(graph)
    wrong_leaf["commit"] = replace(
        graph["commit"],
        final_core_state_leaf_ids=tuple(
            (author, f"{author}-wrong-leaf")
            for author in graph["plan"].online_core_ids
        ),
    )
    with pytest.raises(V3ContractError, match="final state leaves"):
        validate_tick_artifact_graph(**wrong_leaf)

    orphan_cycle = ReadCycleManifest(
        core_id="axon64-b",
        tick_seq=graph["plan"].tick_seq,
        phase="FINAL",
        working_field_id=graph["plan"].working_field_id,
        projection_id=graph["plan"].projection_id,
        board_id=graph["refine_board"].board_id,
        page_view_hashes=(_digest("orphan-page"),),
        page_character_counts=(95,),
        cursor_chain=(_digest("orphan-cursor"),),
        expected_character_count=95,
        coverage_complete=True,
        selected_span_fingerprint=_digest("orphan-selection"),
    )
    orphan = dict(graph)
    orphan["read_cycles"] = (*graph["read_cycles"], orphan_cycle)
    with pytest.raises(V3ContractError, match="orphan read-cycle"):
        validate_tick_artifact_graph(**orphan)


def test_consolidation_validator_binds_working_field_and_refine_board() -> None:
    graph = _valid_tick_graph()
    validate_consolidation_against_plan(
        graph["consolidation"],
        graph["plan"],
        graph["refine_board"],
    )
    with pytest.raises(V3ContractError, match="working field"):
        validate_consolidation_against_plan(
            replace(graph["consolidation"], working_field_id=_HASH_C),
            graph["plan"],
            graph["refine_board"],
        )
    with pytest.raises(V3ContractError, match="supplied REFINE board"):
        validate_consolidation_against_plan(
            replace(
                graph["consolidation"],
                refinement_board_id="wrong-board",
            ),
            graph["plan"],
            graph["refine_board"],
        )
