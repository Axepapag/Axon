from __future__ import annotations

import dataclasses

import pytest

from runtime.field import (
    D64FieldCompiler,
    FieldDelta,
    InsertText,
    LogicalRegion,
    OverlappingDeltaError,
    ReplaceText,
    SharedFieldSnapshot,
)
from runtime.heart import (
    AuthorityClass,
    AuthorityGrant,
    AuthorityViolationError,
    BarrierNotReadyError,
    CoreCommitError,
    CoreDescriptor,
    CoreRegistry,
    CoreStatus,
    DuplicateCoreError,
    DuplicateProposalError,
    FinalCommitAlreadyMadeError,
    FrozenTickImage,
    HeartbeatClock,
    HeartbeatError,
    HeartTransactionBoundary,
    HeartTransactionError,
    IngressChannel,
    IngressDuringTickError,
    InvalidAuthorityGrantError,
    NoActiveParticipantsError,
    Proposal,
    ProposalBoard,
    ProposalBoardError,
    ProposalPass,
    RailMembershipError,
    RailWidthMismatchError,
    StaleBaseProposalError,
    StaleRailBindingError,
    TickBindingError,
    TickIdentity,
    UnknownCoreError,
    UnknownParticipantError,
    ValveDuringTickError,
)

D64 = 64


def _base() -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.SCRATCH: "seed notes",
            LogicalRegion.RESPONSE_DRAFT: "hello",
            LogicalRegion.USER_INPUT: "hi axon",
        },
        tick_id=0,
        source="test-heart",
        provenance="test-heart-fixture",
    )


def _delta(
    base: SharedFieldSnapshot,
    author: str,
    pass_id: str,
    operations: tuple,
) -> FieldDelta:
    return FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id=author,
        pass_id=pass_id,
        operations=operations,
    )


def _scratch_append(base: SharedFieldSnapshot, author: str, pass_id: str, text: str) -> FieldDelta:
    return _delta(
        base,
        author,
        pass_id,
        (InsertText(region=LogicalRegion.SCRATCH, offset=0, text=text),),
    )


def _registry() -> CoreRegistry:
    return CoreRegistry(
        (
            CoreDescriptor(core_id="core-alpha", d_model=D64),
            CoreDescriptor(core_id="core-beta", d_model=D64),
            CoreDescriptor(
                core_id="core-gamma",
                d_model=D64,
                status=CoreStatus.OFFLINE_TRAINING,
            ),
            CoreDescriptor(
                core_id="core-delta",
                d_model=D64,
                status=CoreStatus.DISABLED,
            ),
        )
    )


def _image(base: SharedFieldSnapshot):
    clock = HeartbeatClock()
    identity = clock.open_tick(base)
    compiled = D64FieldCompiler().compile(base)
    return clock, identity, FrozenTickImage.from_compiled(identity, {D64: compiled})


def _board(base: SharedFieldSnapshot):
    clock, identity, image = _image(base)
    registry = _registry()
    board = ProposalBoard(image, registry.declare_participants(D64))
    return clock, identity, image, registry, board


# --- authority classes -----------------------------------------------------


def test_authority_matrix_governs_exact_regions_per_class() -> None:
    user = AuthorityGrant.ingress(IngressChannel.USER)
    tool = AuthorityGrant.ingress(IngressChannel.TOOL)
    advisor = AuthorityGrant.ingress(IngressChannel.ADVISOR)
    assert user.governs(LogicalRegion.USER_INPUT)
    assert tool.governs(LogicalRegion.TOOL_RESULTS)
    assert advisor.governs(LogicalRegion.ADVISOR_INPUT)
    for grant in (user, tool, advisor):
        assert not grant.governs(LogicalRegion.SCRATCH)
        assert not grant.governs(LogicalRegion.CORTEX)
    assert not user.governs(LogicalRegion.TOOL_RESULTS)

    valve = AuthorityGrant.dormant_valve()
    for region in LogicalRegion:
        assert valve.governs(region) is (
            region is LogicalRegion.CORTEX
        )

    core = AuthorityGrant.core()
    assert core.governs(LogicalRegion.SCRATCH)
    assert core.governs(LogicalRegion.RESPONSE_DRAFT)
    assert not core.governs(LogicalRegion.USER_INPUT)
    widened = AuthorityGrant.core({LogicalRegion.TASK_STATE})
    assert widened.governs(LogicalRegion.TASK_STATE)
    assert not widened.governs(LogicalRegion.SCRATCH)

    consolidator = AuthorityGrant.consolidator()
    for region in LogicalRegion:
        assert consolidator.governs(region) is (region is not LogicalRegion.IDENTITY)

    identity_steward = AuthorityGrant.identity_steward()
    for region in LogicalRegion:
        assert identity_steward.governs(region) is (region is LogicalRegion.IDENTITY)


def test_invalid_grants_fail_closed() -> None:
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant(AuthorityClass.EXTERNAL_INGRESS)
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant(AuthorityClass.CONSOLIDATOR, channel=IngressChannel.USER)
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant(AuthorityClass.DORMANT_VALVE, permitted_regions=frozenset({LogicalRegion.SCRATCH}))
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant.core(())
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant.core({"not-a-region"})
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant("not-a-class")


def test_core_authority_rejects_ungoverned_delta() -> None:
    base = _base()
    delta = _delta(
        base,
        "core-alpha",
        "first",
        (InsertText(region=LogicalRegion.USER_INPUT, offset=0, text="forged"),),
    )
    with pytest.raises(AuthorityViolationError):
        AuthorityGrant.core().assert_delta_permitted(delta)
    AuthorityGrant.consolidator().assert_delta_permitted(delta)


# --- core registry ---------------------------------------------------------


def test_registry_declares_only_active_participants() -> None:
    registry = _registry()
    participants = registry.declare_participants(D64)
    assert tuple(d.core_id for d in participants) == ("core-alpha", "core-beta")
    assert all(d.status is CoreStatus.ACTIVE for d in participants)

    with pytest.raises(NoActiveParticipantsError):
        registry.declare_participants(128)
    with pytest.raises(UnknownCoreError):
        registry.get("core-ghost")
    with pytest.raises(DuplicateCoreError):
        registry.register(CoreDescriptor(core_id="core-alpha", d_model=D64))

    descriptor = registry.get("core-alpha")
    grant = descriptor.authority_grant()
    assert grant.authority_class is AuthorityClass.CORE
    assert grant.governs(LogicalRegion.SCRATCH)


# --- heartbeat / tick identity / frozen tick image -------------------------


def test_heartbeat_and_tick_identity_are_monotonic_and_bound() -> None:
    base = _base()
    clock = HeartbeatClock()
    assert clock.beat() == 1
    assert clock.beat() == 2
    first = clock.open_tick(base)
    second = clock.open_tick(base)
    assert first.heartbeat_id == 3
    assert second.heartbeat_id == 4
    assert second.tick_sequence == first.tick_sequence + 1
    assert first.base_field_id == base.field_id
    assert first.base_tick_id == base.tick_id
    assert first.tick_uid != second.tick_uid

    with pytest.raises(ValueError):
        TickIdentity(tick_sequence=0, heartbeat_id=1, base_field_id="x", base_tick_id=0)
    with pytest.raises(TypeError):
        clock.open_tick("not-a-snapshot")


def test_frozen_tick_image_is_immutable_and_bound_to_base() -> None:
    base = _base()
    _, identity, image = _image(base)

    assert image.base_field_id == base.field_id
    rail = image.require_rail(D64)
    assert rail.source_field_id == base.field_id
    assert rail.source_tick_id == base.tick_id
    assert image.image_id
    with pytest.raises(HeartbeatError):
        image.require_rail(128)

    with pytest.raises(dataclasses.FrozenInstanceError):
        image.identity = identity  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rail.rail_id = "tampered"  # type: ignore[misc]
    assert isinstance(image.rails, tuple)

    other = SharedFieldSnapshot.from_texts({LogicalRegion.SCRATCH: "different"})
    stale_compiled = D64FieldCompiler().compile(other)
    with pytest.raises(StaleRailBindingError):
        FrozenTickImage.from_compiled(identity, {D64: stale_compiled})
    with pytest.raises(HeartbeatError):
        FrozenTickImage.from_compiled(identity, {})


# --- proposal board --------------------------------------------------------


def test_board_barriers_and_participant_accounting() -> None:
    base = _base()
    _, _, _, _, board = _board(base)

    # Refinement is unreachable while the first pass is open.
    with pytest.raises(BarrierNotReadyError):
        board.submit(
            Proposal(
                delta=_scratch_append(base, "core-alpha", "refined", "early"),
                rail_d_model=D64,
                pass_kind=ProposalPass.REFINED,
            )
        )
    with pytest.raises(BarrierNotReadyError):
        board.first_pass_proposals()
    with pytest.raises(BarrierNotReadyError):
        board.assert_ready_for_consolidation()

    board.submit(
        Proposal(
            delta=_scratch_append(base, "core-alpha", "first", "alpha sees "),
            rail_d_model=D64,
            pass_kind=ProposalPass.FIRST,
        )
    )

    with pytest.raises(UnknownParticipantError):
        board.submit(
            Proposal(
                delta=_scratch_append(base, "core-ghost", "first", "ghost"),
                rail_d_model=D64,
                pass_kind=ProposalPass.FIRST,
            )
        )
    with pytest.raises(DuplicateProposalError):
        board.submit(
            Proposal(
                delta=_scratch_append(base, "core-alpha", "first", "again"),
                rail_d_model=D64,
                pass_kind=ProposalPass.FIRST,
            )
        )
    with pytest.raises(RailMembershipError):
        board.submit(
            Proposal(
                delta=_scratch_append(base, "core-beta", "first", "wrong rail"),
                rail_d_model=128,
                pass_kind=ProposalPass.FIRST,
            )
        )
    stale_base = SharedFieldSnapshot.from_texts({LogicalRegion.SCRATCH: "elsewhere"})
    with pytest.raises(StaleBaseProposalError):
        board.submit(
            Proposal(
                delta=_delta(
                    stale_base,
                    "core-beta",
                    "first",
                    (InsertText(region=LogicalRegion.SCRATCH, offset=0, text="x"),),
                ),
                rail_d_model=D64,
                pass_kind=ProposalPass.FIRST,
            )
        )

    # Authority classes bind core proposals even on the noncanonical board.
    with pytest.raises(AuthorityViolationError):
        board.submit(
            Proposal(
                delta=_delta(
                    base,
                    "core-beta",
                    "first",
                    (InsertText(region=LogicalRegion.USER_INPUT, offset=0, text="overreach"),),
                ),
                rail_d_model=D64,
                pass_kind=ProposalPass.FIRST,
            )
        )

    # The first-pass barrier refuses to close while beta is unaccounted.
    with pytest.raises(BarrierNotReadyError):
        board.close_first_pass()

    board.mark_failed("core-beta", ProposalPass.FIRST, detail="attend crashed")
    board.close_first_pass()
    assert board.first_pass_closed

    exposed = board.first_pass_proposals()
    assert [proposal.author_core_id for proposal in exposed] == ["core-alpha"]

    states = {
        record.core_id: record.state.value
        for record in board.participant_states(ProposalPass.FIRST)
    }
    assert states == {"core-alpha": "returned", "core-beta": "failed"}

    # Refinement: alpha returns, beta times out; then the barrier closes.
    board.submit(
        Proposal(
            delta=_scratch_append(base, "core-alpha", "refined", "refined note"),
            rail_d_model=D64,
            pass_kind=ProposalPass.REFINED,
        )
    )
    with pytest.raises(BarrierNotReadyError):
        board.close_refinement()
    board.mark_timed_out("core-beta", ProposalPass.REFINED, detail="budget exhausted")
    board.close_refinement()

    refined = board.refined_proposals()
    assert [proposal.author_core_id for proposal in refined] == ["core-alpha"]
    assert refined[0].delta.pass_id == "refined"
    board.assert_ready_for_consolidation()

    states = {
        record.core_id: record.state.value
        for record in board.participant_states(ProposalPass.REFINED)
    }
    assert states == {"core-alpha": "returned", "core-beta": "timed_out"}


def test_board_is_a_noncanonical_workspace() -> None:
    base = _base()
    before_id = base.field_id
    before_tick = base.tick_id
    before_hash = base.canonical_hash

    _, _, _, _, board = _board(base)
    board.submit(
        Proposal(
            delta=_scratch_append(base, "core-alpha", "first", "alpha"),
            rail_d_model=D64,
            pass_kind=ProposalPass.FIRST,
        )
    )
    board.mark_failed("core-beta", ProposalPass.FIRST)
    board.close_first_pass()

    # Board activity advances the workspace only; canonical state is untouched.
    assert base.field_id == before_id
    assert base.tick_id == before_tick
    assert base.canonical_hash == before_hash
    assert board.base_field_id == before_id
    assert not any(
        isinstance(value, SharedFieldSnapshot) for value in vars(board).values()
    )


def test_board_requires_active_participants_on_carried_rails() -> None:
    base = _base()
    _, _, image = _image(base)
    with pytest.raises(ProposalBoardError):
        ProposalBoard(image, ())
    with pytest.raises(ProposalBoardError):
        ProposalBoard(
            image,
            (CoreDescriptor(core_id="off", d_model=D64, status=CoreStatus.DISABLED),),
        )
    with pytest.raises(RailMembershipError):
        ProposalBoard(image, (CoreDescriptor(core_id="wide", d_model=128),))


# --- heart transaction boundary --------------------------------------------


def test_boundary_rejects_stale_base_proposals() -> None:
    base = _base()
    other = SharedFieldSnapshot.from_texts({LogicalRegion.SCRATCH: "elsewhere"})
    boundary = HeartTransactionBoundary()
    stale = _delta(
        other,
        "consolidator-01",
        "consolidation",
        (InsertText(region=LogicalRegion.SCRATCH, offset=0, text="stale"),),
    )
    with pytest.raises(StaleBaseProposalError):
        boundary.validate_proposal(base, stale, AuthorityGrant.consolidator())

    # A consolidator commit requires its matching in-flight tick; with that
    # tick open, a proposal bound to another base is still rejected as stale.
    _, identity, image = _image(base)
    boundary.note_tick_opened(image)
    with pytest.raises(StaleBaseProposalError):
        boundary.commit(base, stale, AuthorityGrant.consolidator(), tick=identity)
    boundary.note_tick_closed(identity)


def test_boundary_rejects_conflicting_sparse_edits() -> None:
    base = _base()
    _, identity, image = _image(base)
    boundary = HeartTransactionBoundary()
    boundary.note_tick_opened(image)
    grant = AuthorityGrant.consolidator()

    overlapping_replaces = _delta(
        base,
        "consolidator-01",
        "consolidation",
        (
            ReplaceText(region=LogicalRegion.SCRATCH, start=0, end=3, text="xy"),
            ReplaceText(region=LogicalRegion.SCRATCH, start=2, end=5, text="zz"),
        ),
    )
    with pytest.raises(OverlappingDeltaError):
        boundary.commit(base, overlapping_replaces, grant, tick=identity)

    colliding_inserts = _delta(
        base,
        "consolidator-01",
        "consolidation",
        (
            InsertText(region=LogicalRegion.SCRATCH, offset=1, text="a"),
            InsertText(region=LogicalRegion.SCRATCH, offset=1, text="b"),
        ),
    )
    with pytest.raises(OverlappingDeltaError):
        boundary.commit(base, colliding_inserts, grant, tick=identity)

    # Rejected commits leave the tick in flight; only a successful
    # consolidator commit or an explicit close ends it.
    assert boundary.tick_in_flight
    boundary.note_tick_closed(identity)


def test_validated_consolidator_decision_commits_via_canonical_delta_path() -> None:
    base = _base()
    _, identity, image, _, board = _board(base)
    boundary = HeartTransactionBoundary()
    boundary.note_tick_opened(image)

    with pytest.raises(HeartTransactionError):
        boundary.note_tick_opened(image)

    board.submit(
        Proposal(
            delta=_scratch_append(base, "core-alpha", "first", "alpha draft. "),
            rail_d_model=D64,
            pass_kind=ProposalPass.FIRST,
        )
    )
    board.mark_failed("core-beta", ProposalPass.FIRST, detail="no response")
    board.close_first_pass()
    board.submit(
        Proposal(
            delta=_scratch_append(base, "core-alpha", "refined", "refined draft. "),
            rail_d_model=D64,
            pass_kind=ProposalPass.REFINED,
        )
    )
    board.mark_timed_out("core-beta", ProposalPass.REFINED)
    board.close_refinement()
    board.assert_ready_for_consolidation()

    scratch_text = base.region(LogicalRegion.SCRATCH).text
    decision = _delta(
        base,
        "consolidator-01",
        "consolidation",
        (
            ReplaceText(
                region=LogicalRegion.SCRATCH,
                start=0,
                end=len(scratch_text),
                text="refined draft. consolidated",
                provenance="tick:1:consolidator:refined-board",
            ),
        ),
    )

    commit = boundary.commit(
        base,
        decision,
        AuthorityGrant.consolidator(),
        tick=identity,
    )
    successor = commit.successor

    assert successor.parent_field_id == base.field_id
    assert successor.tick_id == base.tick_id + 1
    assert successor.region(LogicalRegion.SCRATCH).text == "refined draft. consolidated"
    new_span = successor.region(LogicalRegion.SCRATCH).spans[0]
    assert new_span.source == "consolidator-01"
    assert new_span.provenance == "tick:1:consolidator:refined-board"
    assert commit.delta.delta_id == decision.delta_id
    assert commit.tick == identity
    assert commit.commit_id
    assert commit.to_canonical_dict()["authority_class"] == "consolidator"

    # The successor is valid canonical state: it recompiles with exact roundtrip.
    compiled = D64FieldCompiler().compile(successor)
    compiled.verify_roundtrip(successor)
    assert compiled.coverage.complete

    # The successful consolidator commit atomically ends the tick; a second
    # commit from the same tick identity fails closed.
    assert not boundary.tick_in_flight
    with pytest.raises(HeartTransactionError):
        boundary.note_tick_closed(identity)
    with pytest.raises(FinalCommitAlreadyMadeError):
        boundary.commit(base, decision, AuthorityGrant.consolidator(), tick=identity)

    # The base snapshot is untouched by the commit; replaying the same delta
    # against the successor under a fresh tick is stale and fails closed.
    assert base.tick_id == 0
    assert base.field_id != successor.field_id
    _, next_identity, next_image = _image(successor)
    boundary.note_tick_opened(next_image)
    with pytest.raises(StaleBaseProposalError):
        boundary.commit(
            successor,
            decision,
            AuthorityGrant.consolidator(),
            tick=next_identity,
        )
    boundary.note_tick_closed(next_identity)


def test_tick_commit_requires_the_in_flight_tick() -> None:
    base = _base()
    clock = HeartbeatClock()
    identity = clock.open_tick(base)
    other_identity = clock.open_tick(base)
    compiled = D64FieldCompiler().compile(base)
    image = FrozenTickImage.from_compiled(identity, {D64: compiled})
    boundary = HeartTransactionBoundary()
    decision = _scratch_append(base, "consolidator-01", "consolidation", "x")

    with pytest.raises(TickBindingError):
        boundary.commit(base, decision, AuthorityGrant.consolidator(), tick=identity)

    boundary.note_tick_opened(image)
    with pytest.raises(TickBindingError):
        boundary.commit(base, decision, AuthorityGrant.consolidator(), tick=other_identity)
    with pytest.raises(HeartTransactionError):
        boundary.note_tick_closed(other_identity)
    boundary.note_tick_closed(identity)
    assert not boundary.tick_in_flight


def test_ingress_and_valve_governance_at_the_boundary() -> None:
    base = _base()
    _, identity, image = _image(base)
    boundary = HeartTransactionBoundary()

    ingress_delta = _delta(
        base,
        "ingress:user",
        "intake",
        (InsertText(region=LogicalRegion.USER_INPUT, offset=0, text="jeff says hi"),),
    )
    valve_delta = _delta(
        base,
        "dormant-valve",
        "recall",
        (InsertText(region=LogicalRegion.CORTEX, offset=0, text="fact"),),
    )
    valve_overreach = _delta(
        base,
        "dormant-valve",
        "recall",
        (InsertText(region=LogicalRegion.SCRATCH, offset=0, text="fact"),),
    )

    # The valve may address only governed cortex.
    with pytest.raises(AuthorityViolationError):
        boundary.commit(base, valve_overreach, AuthorityGrant.dormant_valve())

    # Build B widens the canonical validator so the authority matrix can admit
    # ingress-owned regions and the dormant valve's cortex scope.
    commit = boundary.commit(base, ingress_delta, AuthorityGrant.ingress(IngressChannel.USER))
    assert commit.successor.region(LogicalRegion.USER_INPUT).text == "jeff says hihi axon"

    successor = commit.successor
    valve_delta_against_successor = _delta(
        successor,
        "dormant-valve",
        "recall",
        (InsertText(region=LogicalRegion.CORTEX, offset=0, text="fact"),),
    )
    valve_commit = boundary.commit(
        successor,
        valve_delta_against_successor,
        AuthorityGrant.dormant_valve(),
    )
    assert valve_commit.successor.region(LogicalRegion.CORTEX).text == "fact"

    # During an in-flight tick, ingress and the dormant valve must queue for
    # the next beat; only the consolidator's decision may cross the boundary.
    boundary.note_tick_opened(image)
    with pytest.raises(IngressDuringTickError):
        boundary.commit(base, ingress_delta, AuthorityGrant.ingress(IngressChannel.USER))
    with pytest.raises(ValveDuringTickError):
        boundary.commit(base, valve_delta, AuthorityGrant.dormant_valve())
    with pytest.raises(CoreCommitError):
        boundary.commit(
            base,
            _scratch_append(base, "core-alpha", "first", "bypass"),
            AuthorityGrant.core(),
        )
    boundary.note_tick_closed(identity)


# --- Build A.1 regression tests: doctrine-violation blockers ----------------


def test_core_grants_are_never_commit_capable() -> None:
    base = _base()
    boundary = HeartTransactionBoundary()
    decision = _scratch_append(base, "core-alpha", "first", "rogue")

    # Between ticks: cores only propose; a direct core commit fails closed.
    with pytest.raises(CoreCommitError):
        boundary.commit(base, decision, AuthorityGrant.core())
    assert not boundary.tick_in_flight

    # During a tick: a core grant is still never commit-capable.
    _, identity, image = _image(base)
    boundary.note_tick_opened(image)
    with pytest.raises(CoreCommitError):
        boundary.commit(base, decision, AuthorityGrant.core())
    assert boundary.tick_in_flight
    boundary.note_tick_closed(identity)


def test_consolidator_commit_requires_in_flight_tick_token_and_frozen_base() -> None:
    base = _base()
    _, identity, image = _image(base)
    boundary = HeartTransactionBoundary()
    decision = _scratch_append(base, "consolidator-01", "consolidation", "x")

    # No tick in flight: a consolidator decision cannot commit at all.
    with pytest.raises(TickBindingError):
        boundary.commit(base, decision, AuthorityGrant.consolidator())

    boundary.note_tick_opened(image)

    # The tick token is mandatory while a tick is in flight; omitting it
    # must not skip the identity/base binding checks.
    with pytest.raises(TickBindingError):
        boundary.commit(base, decision, AuthorityGrant.consolidator())

    # The commit base must be the in-flight tick's exact frozen base, even
    # when the delta itself is valid against another real snapshot.
    other = SharedFieldSnapshot.from_texts({LogicalRegion.SCRATCH: "elsewhere"})
    other_decision = _scratch_append(other, "consolidator-01", "consolidation", "y")
    with pytest.raises(TickBindingError):
        boundary.commit(
            other,
            other_decision,
            AuthorityGrant.consolidator(),
            tick=identity,
        )

    # Every rejection left the tick in flight.
    assert boundary.tick_in_flight
    boundary.note_tick_closed(identity)


def test_successful_consolidator_commit_consumes_the_tick() -> None:
    base = _base()
    _, identity, image = _image(base)
    boundary = HeartTransactionBoundary()
    boundary.note_tick_opened(image)
    decision = _scratch_append(base, "consolidator-01", "consolidation", "final")

    commit = boundary.commit(base, decision, AuthorityGrant.consolidator(), tick=identity)
    assert commit.tick == identity
    # The tick ends at the heart commit — and only there.
    assert not boundary.tick_in_flight

    # A second commit from the same tick identity/base fails closed as an
    # already-consumed tick.
    with pytest.raises(FinalCommitAlreadyMadeError):
        boundary.commit(base, decision, AuthorityGrant.consolidator(), tick=identity)

    # note_tick_closed remains correct for the no-commit close path: there is
    # nothing left to close after the commit consumed the tick.
    with pytest.raises(HeartTransactionError):
        boundary.note_tick_closed(identity)


def test_valve_may_not_commit_during_a_tick() -> None:
    base = _base()
    _, identity, image = _image(base)
    boundary = HeartTransactionBoundary()
    valve_delta = _delta(
        base,
        "dormant-valve",
        "recall",
        (InsertText(region=LogicalRegion.CORTEX, offset=0, text="fact"),),
    )

    # Between ticks the valve is admitted (subject to region authority and the
    # bootstrap delta seal), but during a tick it must queue for the next beat.
    boundary.note_tick_opened(image)
    with pytest.raises(ValveDuringTickError):
        boundary.commit(base, valve_delta, AuthorityGrant.dormant_valve())
    assert boundary.tick_in_flight
    boundary.note_tick_closed(identity)


def test_fake_wider_rail_labels_fail_closed() -> None:
    base = _base()
    clock = HeartbeatClock()
    identity = clock.open_tick(base)
    compiled = D64FieldCompiler().compile(base)
    assert compiled.rows.shape[1] == 64

    # Build A is 64D-first: no wider rail may be bound yet.
    with pytest.raises(RailWidthMismatchError):
        FrozenTickImage.from_compiled(identity, {128: compiled})
    with pytest.raises(RailWidthMismatchError):
        FrozenTickImage.from_compiled(identity, {1024: compiled})

    # The honest 64D binding still works.
    image = FrozenTickImage.from_compiled(identity, {D64: compiled})
    assert image.require_rail(D64).d_model == D64
