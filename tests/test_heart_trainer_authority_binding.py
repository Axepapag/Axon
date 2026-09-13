"""Scope A5 tests: trainer authority class and exact core generation binding.

Covers handoff steps 4 and 5:

- the TRAINER authority class governs exactly the two training regions plus
  the ratified training-lifecycle record kinds, fails closed at construction,
  and never widens the delta layer's CORE_WRITABLE_REGIONS seal;
- cores (including OFFLINE_TRAINING cores) hold only core-class proposal
  grants and can never self-commit training responses;
- a registered real core binds to exact architecture/parameter/optimizer/Soul
  generations via a hash-identified CoreBinding, phantom cores cannot
  auto-register without an operator-issued grant already in the registry, and
  entering OFFLINE_TRAINING requires a valid trainer grant plus a durable
  assignment claim reference.
"""
from __future__ import annotations

import dataclasses

import pytest

from runtime.field import (
    CORE_WRITABLE_REGIONS,
    TRAINING_REGIONS,
    D64FieldCompiler,
    FieldDelta,
    InsertText,
    LogicalRegion,
    SharedFieldSnapshot,
)
from runtime.heart import (
    TRAINER_GOVERNED_REGIONS,
    TRAINER_LIFECYCLE_RECORD_TYPES,
    AuthorityClass,
    AuthorityGrant,
    AuthorityViolationError,
    CoreBinding,
    CoreCommitError,
    CoreDescriptor,
    CoreRegistry,
    CoreStatus,
    DuplicateCoreError,
    FrozenTickImage,
    HeartbeatClock,
    HeartTransactionBoundary,
    InvalidAuthorityGrantError,
    InvalidCoreBindingError,
    InvalidModeTransitionError,
    OperatorCoreGrant,
    UngrantedCoreRegistrationError,
    UnknownCoreError,
)

D64 = 64


def _base() -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.TRAINER_INSTRUCTIONS: "assignment text",
            LogicalRegion.TRAINING_RESPONSES: "prior attempt",
            LogicalRegion.SCRATCH: "seed notes",
        },
        tick_id=0,
        source="test-trainer",
        provenance="test-trainer-fixture",
    )


def _delta(base: SharedFieldSnapshot, author: str, operations: tuple) -> FieldDelta:
    return FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id=author,
        pass_id="test",
        operations=operations,
    )


def _registry_with_bound_core() -> CoreRegistry:
    """One registered, generation-bound reasoning core (operator path)."""

    registry = CoreRegistry(
        (CoreDescriptor(core_id="core-alpha", d_model=D64),)
    )
    grant = OperatorCoreGrant(core_id="core-alpha", d_model=D64)
    registry.issue_operator_grant(grant)
    registry.bind_core(
        CoreBinding(
            core_id="core-alpha",
            architecture_id="untrained-reasoning-core-v1",
            parameter_generation="untrained",
            optimizer_generation="opt-gen-0",
            soul_id="soul-gen-0",
        ),
        grant=grant,
    )
    return registry


# --- trainer authority class -------------------------------------------------


def test_trainer_grant_governs_exactly_the_training_regions() -> None:
    trainer = AuthorityGrant.trainer()
    assert trainer.authority_class is AuthorityClass.TRAINER
    assert trainer.governed_regions == TRAINER_GOVERNED_REGIONS
    assert trainer.governed_regions == TRAINING_REGIONS
    assert trainer.governs(LogicalRegion.TRAINER_INSTRUCTIONS)
    assert trainer.governs(LogicalRegion.TRAINING_RESPONSES)
    for region in LogicalRegion:
        if region not in TRAINING_REGIONS:
            assert not trainer.governs(region)

    # Coercion from the raw enum value behaves identically.
    coerced = AuthorityGrant(AuthorityClass.TRAINER)
    assert coerced.governed_regions == trainer.governed_regions


def test_trainer_grant_governs_exactly_the_lifecycle_record_kinds() -> None:
    trainer = AuthorityGrant.trainer()
    for record_type in TRAINER_LIFECYCLE_RECORD_TYPES:
        assert trainer.governs_lifecycle_record(record_type)
        trainer.assert_governs_lifecycle_record(record_type)
    for record_type in ("identity", "ledger", "soul_payload", ""):
        assert not trainer.governs_lifecycle_record(record_type)
        with pytest.raises(AuthorityViolationError):
            trainer.assert_governs_lifecycle_record(record_type)
    for other in (
        AuthorityGrant.core(),
        AuthorityGrant.consolidator(),
        AuthorityGrant.dormant_valve(),
        AuthorityGrant.ingress("user"),
        AuthorityGrant.identity_steward(),
    ):
        for record_type in TRAINER_LIFECYCLE_RECORD_TYPES:
            assert not other.governs_lifecycle_record(record_type)
    with pytest.raises(TypeError):
        trainer.governs_lifecycle_record(42)  # type: ignore[arg-type]


def test_trainer_grant_fails_closed_at_construction() -> None:
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant(AuthorityClass.TRAINER, channel="user")
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant(
            AuthorityClass.TRAINER,
            permitted_regions=frozenset({LogicalRegion.TRAINER_INSTRUCTIONS}),
        )


def test_training_regions_are_not_core_writable_and_cannot_be_core_granted() -> None:
    # Defense in depth: the delta layer's bootstrap seal is untouched.
    assert not (CORE_WRITABLE_REGIONS & TRAINING_REGIONS)
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant.core({LogicalRegion.TRAINER_INSTRUCTIONS})
    with pytest.raises(InvalidAuthorityGrantError):
        AuthorityGrant.core(
            {LogicalRegion.SCRATCH, LogicalRegion.TRAINING_RESPONSES}
        )
    # The default core grant is unchanged.
    assert AuthorityGrant.core().governed_regions == CORE_WRITABLE_REGIONS


def test_trainer_authority_binds_deltas_to_the_training_regions() -> None:
    base = _base()
    trainer = AuthorityGrant.trainer()
    trainer_delta = _delta(
        base,
        "trainer-organ",
        (InsertText(region=LogicalRegion.TRAINER_INSTRUCTIONS, offset=0, text="go "),),
    )
    trainer.assert_delta_permitted(trainer_delta)

    mixed_delta = _delta(
        base,
        "trainer-organ",
        (
            InsertText(region=LogicalRegion.TRAINER_INSTRUCTIONS, offset=0, text="x"),
            InsertText(region=LogicalRegion.SCRATCH, offset=0, text="y"),
        ),
    )
    with pytest.raises(AuthorityViolationError):
        trainer.assert_delta_permitted(mixed_delta)
    with pytest.raises(AuthorityViolationError):
        trainer.assert_delta_permitted(
            _delta(
                base,
                "trainer-organ",
                (InsertText(region=LogicalRegion.SCRATCH, offset=0, text="z"),),
            )
        )
    # A core-class grant never admits the training regions.
    with pytest.raises(AuthorityViolationError):
        AuthorityGrant.core().assert_delta_permitted(trainer_delta)


def test_trainer_commit_passes_the_canonical_boundary_between_ticks_only() -> None:
    base = _base()
    boundary = HeartTransactionBoundary()
    trainer_grant = AuthorityGrant.trainer()
    response = _delta(
        base,
        "trainer-organ",
        (
            InsertText(
                region=LogicalRegion.TRAINING_RESPONSES, offset=0, text="attempt. "
            ),
        ),
    )

    # Heart-side trainer commits work between ticks through the unchanged
    # typed-delta machinery; the canonical field gains the response exactly.
    commit = boundary.commit(base, response, trainer_grant)
    successor = commit.successor
    assert (
        successor.region(LogicalRegion.TRAINING_RESPONSES).text
        == "attempt. prior attempt"
    )

    # During a tick, the trainer must queue like every non-consolidator class.
    clock = HeartbeatClock()
    identity = clock.open_tick(base)
    compiled = D64FieldCompiler().compile(base)
    image = FrozenTickImage.from_compiled(identity, {D64: compiled})
    boundary.note_tick_opened(image)
    with pytest.raises(AuthorityViolationError):
        boundary.commit(base, response, trainer_grant)
    boundary.note_tick_closed(identity)


def test_offline_training_core_cannot_self_commit_training_responses() -> None:
    base = _base()
    boundary = HeartTransactionBoundary()
    descriptor = CoreDescriptor(
        core_id="core-trainee",
        d_model=D64,
        status=CoreStatus.OFFLINE_TRAINING,
    )
    grant = descriptor.authority_grant()

    # A training core still holds only a core-class proposal grant: it does
    # not govern the training regions, and it is never commit-capable.
    assert grant.authority_class is AuthorityClass.CORE
    assert not grant.governs(LogicalRegion.TRAINER_INSTRUCTIONS)
    assert not grant.governs(LogicalRegion.TRAINING_RESPONSES)

    proposal = _delta(
        base,
        "core-trainee",
        (
            InsertText(
                region=LogicalRegion.TRAINING_RESPONSES, offset=0, text="my answer "
            ),
        ),
    )
    with pytest.raises(CoreCommitError):
        boundary.commit(base, proposal, grant)
    with pytest.raises(AuthorityViolationError):
        grant.assert_delta_permitted(proposal)


# --- operator-issued registration grants --------------------------------------


def test_operator_grant_identity_is_canonical_and_deterministic() -> None:
    grant = OperatorCoreGrant(core_id="core-alpha", d_model=D64)
    same = OperatorCoreGrant(core_id="core-alpha", d_model=D64)
    assert grant.grant_id == same.grant_id
    assert grant.to_canonical_dict()["schema"] == "axon-heart-operator-core-grant-v1"

    different_core = OperatorCoreGrant(core_id="core-beta", d_model=D64)
    different_width = OperatorCoreGrant(core_id="core-alpha", d_model=128)
    different_issuer = OperatorCoreGrant(
        core_id="core-alpha", d_model=D64, issued_by="operator-2"
    )
    for other in (different_core, different_width, different_issuer):
        assert other.grant_id != grant.grant_id


def test_operator_grant_validation_fails_closed() -> None:
    with pytest.raises(ValueError):
        OperatorCoreGrant(core_id="", d_model=D64)
    with pytest.raises(ValueError):
        OperatorCoreGrant(core_id="core-alpha", d_model=0)
    with pytest.raises(TypeError):
        OperatorCoreGrant(core_id="core-alpha", d_model=True)
    with pytest.raises(TypeError):
        OperatorCoreGrant(core_id="core-alpha", d_model="64")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OperatorCoreGrant(core_id="core-alpha", d_model=D64, issued_by="")


def test_registration_with_grant_refuses_phantom_core_ids() -> None:
    registry = CoreRegistry()
    descriptor = CoreDescriptor(core_id="core-alpha", d_model=D64)
    grant = OperatorCoreGrant(core_id="core-alpha", d_model=D64)

    # The grant must already be present in the registry; a core can never
    # introduce its own authorization.
    with pytest.raises(UngrantedCoreRegistrationError):
        registry.register(descriptor, grant=grant)

    # A grant for a different core id does not authorize this descriptor.
    registry.issue_operator_grant(grant)
    other_grant = OperatorCoreGrant(core_id="core-phantom", d_model=D64)
    registry.issue_operator_grant(other_grant)
    with pytest.raises(UngrantedCoreRegistrationError):
        registry.register(
            CoreDescriptor(core_id="core-phantom", d_model=128),
            grant=other_grant,
        )

    # The matching, present grant authorizes the non-operator path.
    registry.register(descriptor, grant=grant)
    assert registry.get("core-alpha") is descriptor

    # Duplicate grant issue and duplicate registration both fail closed.
    with pytest.raises(DuplicateCoreError):
        registry.issue_operator_grant(grant)
    with pytest.raises(DuplicateCoreError):
        registry.register(
            CoreDescriptor(core_id="core-alpha", d_model=D64), grant=grant
        )

    # The operator path (no grant argument) remains available to the operator.
    operator_registry = CoreRegistry((CoreDescriptor(core_id="core-op", d_model=D64),))
    assert "core-op" in operator_registry


# --- exact generation binding ---------------------------------------------------


def test_core_binding_identity_is_canonical_and_field_sensitive() -> None:
    binding = CoreBinding(
        core_id="core-alpha",
        architecture_id="arch-v1",
        parameter_generation="pg-1",
        optimizer_generation="og-1",
        soul_id="soul-1",
    )
    assert binding.binding_id
    assert binding.mode is CoreStatus.ACTIVE
    same = CoreBinding(
        core_id="core-alpha",
        architecture_id="arch-v1",
        parameter_generation="pg-1",
        optimizer_generation="og-1",
        soul_id="soul-1",
    )
    assert same.binding_id == binding.binding_id

    variants = (
        dataclasses.replace(binding, core_id="core-beta"),
        dataclasses.replace(binding, architecture_id="arch-v2"),
        dataclasses.replace(binding, parameter_generation="pg-2"),
        dataclasses.replace(binding, optimizer_generation="og-2"),
        dataclasses.replace(binding, soul_id="soul-2"),
        dataclasses.replace(binding, mode=CoreStatus.OFFLINE_TRAINING),
    )
    for variant in variants:
        assert variant.binding_id != binding.binding_id


def test_core_binding_fails_closed_on_malformed_input() -> None:
    with pytest.raises(ValueError):
        CoreBinding(
            core_id="",
            architecture_id="arch-v1",
            parameter_generation="pg-1",
            optimizer_generation="og-1",
            soul_id="soul-1",
        )
    for attr in (
        "architecture_id",
        "parameter_generation",
        "optimizer_generation",
        "soul_id",
    ):
        kwargs = {
            "core_id": "core-alpha",
            "architecture_id": "arch-v1",
            "parameter_generation": "pg-1",
            "optimizer_generation": "og-1",
            "soul_id": "soul-1",
            attr: "",
        }
        with pytest.raises(ValueError):
            CoreBinding(**kwargs)
    # DISABLED is a lifecycle status, not a binding mode.
    with pytest.raises(InvalidCoreBindingError):
        CoreBinding(
            core_id="core-alpha",
            architecture_id="arch-v1",
            parameter_generation="pg-1",
            optimizer_generation="og-1",
            soul_id="soul-1",
            mode=CoreStatus.DISABLED,
        )


def test_bind_core_requires_registered_matching_identity() -> None:
    registry = _registry_with_bound_core()

    # A phantom core id can never be bound.
    phantom = CoreBinding(
        core_id="core-phantom",
        architecture_id="arch-v1",
        parameter_generation="pg-1",
        optimizer_generation="og-1",
        soul_id="soul-1",
    )
    phantom_grant = OperatorCoreGrant(core_id="core-phantom", d_model=D64)
    registry.issue_operator_grant(phantom_grant)
    with pytest.raises(UnknownCoreError):
        registry.bind_core(phantom, grant=phantom_grant)

    # A binding that contradicts the registered descriptor fails closed.
    registry.register(CoreDescriptor(core_id="core-beta", d_model=D64))
    beta_grant = OperatorCoreGrant(core_id="core-beta", d_model=D64)
    registry.issue_operator_grant(beta_grant)
    with pytest.raises(InvalidCoreBindingError):
        registry.bind_core(
            CoreBinding(
                core_id="core-beta",
                architecture_id="different-arch",
                parameter_generation="untrained",
                optimizer_generation="og-0",
                soul_id="soul-0",
            ),
            grant=beta_grant,
        )

    # A binding whose mode does not match the descriptor status fails closed.
    with pytest.raises(InvalidCoreBindingError):
        registry.bind_core(
            CoreBinding(
                core_id="core-beta",
                architecture_id="untrained-reasoning-core-v1",
                parameter_generation="untrained",
                optimizer_generation="og-0",
                soul_id="soul-0",
                mode=CoreStatus.OFFLINE_TRAINING,
            ),
            grant=beta_grant,
        )

    # A mismatched or absent operator grant fails closed.
    mismatched = OperatorCoreGrant(core_id="core-alpha", d_model=D64)
    with pytest.raises(UngrantedCoreRegistrationError):
        registry.bind_core(
            CoreBinding(
                core_id="core-beta",
                architecture_id="untrained-reasoning-core-v1",
                parameter_generation="untrained",
                optimizer_generation="og-0",
                soul_id="soul-0",
            ),
            grant=mismatched,
        )
    with pytest.raises(TypeError):
        registry.bind_core(  # type: ignore[call-arg]
            CoreBinding(
                core_id="core-beta",
                architecture_id="untrained-reasoning-core-v1",
                parameter_generation="untrained",
                optimizer_generation="og-0",
                soul_id="soul-0",
            )
        )

    # The exact matching binding commits.
    binding = registry.bind_core(
        CoreBinding(
            core_id="core-beta",
            architecture_id="untrained-reasoning-core-v1",
            parameter_generation="untrained",
            optimizer_generation="og-0",
            soul_id="soul-0",
        ),
        grant=beta_grant,
    )
    assert registry.binding("core-beta") is binding
    assert [b.core_id for b in registry.bindings()] == ["core-alpha", "core-beta"]
    with pytest.raises(UnknownCoreError):
        registry.binding("core-phantom")


def test_registration_update_validates_the_binding() -> None:
    registry = _registry_with_bound_core()
    grant = OperatorCoreGrant(core_id="core-alpha", d_model=D64)

    # Updating a phantom is an auto-registration attempt and fails closed.
    with pytest.raises(UnknownCoreError):
        registry.update_descriptor(CoreDescriptor(core_id="core-ghost", d_model=D64))

    # A non-present grant cannot drive an update.
    ghost_grant = OperatorCoreGrant(core_id="core-alpha", d_model=D64, issued_by="impostor")
    with pytest.raises(UngrantedCoreRegistrationError):
        registry.update_descriptor(
            CoreDescriptor(
                core_id="core-alpha",
                d_model=D64,
                status=CoreStatus.OFFLINE_TRAINING,
            ),
            grant=ghost_grant,
        )

    # A status change that contradicts the bound mode fails closed.
    with pytest.raises(InvalidCoreBindingError):
        registry.update_descriptor(
            CoreDescriptor(
                core_id="core-alpha",
                d_model=D64,
                status=CoreStatus.OFFLINE_TRAINING,
            )
        )

    # A consistent status change commits through the operator path.
    registry.update_descriptor(
        CoreDescriptor(core_id="core-alpha", d_model=D64, status=CoreStatus.DISABLED)
    )
    assert registry.get("core-alpha").status is CoreStatus.DISABLED

    # The stale grant still authorizes the matching descriptor.
    registry.update_descriptor(
        CoreDescriptor(core_id="core-alpha", d_model=D64),
        grant=grant,
    )
    assert registry.get("core-alpha").status is CoreStatus.ACTIVE


# --- mode transition into offline training -------------------------------------


def test_enter_offline_training_requires_trainer_grant_and_assignment_claim() -> None:
    registry = _registry_with_bound_core()

    before = registry.binding("core-alpha")
    bound = registry.enter_offline_training(
        "core-alpha",
        grant=AuthorityGrant.trainer(),
        claim="claim-2026-09-11-0001",
    )
    assert bound.mode is CoreStatus.OFFLINE_TRAINING
    assert bound.binding_id != before.binding_id
    descriptor = registry.get("core-alpha")
    assert descriptor.status is CoreStatus.OFFLINE_TRAINING
    # The training core leaves the active participant set immediately.
    assert registry.active() == ()
    assert registry.active(D64) == ()


def test_enter_offline_training_rejects_every_invalid_combination() -> None:
    registry = _registry_with_bound_core()

    # Unknown / unbound cores fail closed.
    with pytest.raises(UnknownCoreError):
        registry.enter_offline_training(
            "core-phantom",
            grant=AuthorityGrant.trainer(),
            claim="claim-1",
        )
    registry.register(CoreDescriptor(core_id="core-unbound", d_model=D64))
    with pytest.raises(UnknownCoreError):
        registry.enter_offline_training(
            "core-unbound",
            grant=AuthorityGrant.trainer(),
            claim="claim-1",
        )

    # Only trainer-class authority may switch a core into training.
    for grant in (
        AuthorityGrant.core(),
        AuthorityGrant.consolidator(),
        AuthorityGrant.dormant_valve(),
        AuthorityGrant.identity_steward(),
        AuthorityGrant.ingress("user"),
    ):
        with pytest.raises(InvalidModeTransitionError):
            registry.enter_offline_training(
                "core-alpha", grant=grant, claim="claim-1"
            )
    with pytest.raises(TypeError):
        registry.enter_offline_training(
            "core-alpha", grant="trainer", claim="claim-1"  # type: ignore[arg-type]
        )

    # A durable assignment claim reference is mandatory.
    for bad_claim in (None, "", "   ", 42, object()):
        with pytest.raises(InvalidModeTransitionError):
            registry.enter_offline_training(
                "core-alpha",
                grant=AuthorityGrant.trainer(),
                claim=bad_claim,
            )


def test_enter_offline_training_accepts_duck_typed_assignment_claim_records() -> None:
    registry = _registry_with_bound_core()

    class FakeClaimRecord:
        assignment_claim_id = "claim-2026-09-11-0002"
        core_id = "core-alpha"

    bound = registry.enter_offline_training(
        "core-alpha",
        grant=AuthorityGrant.trainer(),
        claim=FakeClaimRecord(),
    )
    assert bound.mode is CoreStatus.OFFLINE_TRAINING

    # A claim record naming a different core fails closed.
    registry.register(CoreDescriptor(core_id="core-beta", d_model=D64))
    beta_grant = OperatorCoreGrant(core_id="core-beta", d_model=D64)
    registry.issue_operator_grant(beta_grant)
    registry.bind_core(
        CoreBinding(
            core_id="core-beta",
            architecture_id="untrained-reasoning-core-v1",
            parameter_generation="untrained",
            optimizer_generation="og-0",
            soul_id="soul-0",
        ),
        grant=beta_grant,
    )

    class WrongCoreClaim:
        claim_id = "claim-2026-09-11-0003"
        core_id = "core-alpha"

    with pytest.raises(InvalidModeTransitionError):
        registry.enter_offline_training(
            "core-beta",
            grant=AuthorityGrant.trainer(),
            claim=WrongCoreClaim(),
        )

    # A claim record carrying no durable identity fails closed.
    class EmptyClaim:
        core_id = "core-beta"

    with pytest.raises(InvalidModeTransitionError):
        registry.enter_offline_training(
            "core-beta",
            grant=AuthorityGrant.trainer(),
            claim=EmptyClaim(),
        )


def test_enter_offline_training_accepts_live_assignment_store_claims() -> None:
    from runtime.trainer.assignments import (
        AssignmentLease,
        TrainingAssignment,
    )

    registry = _registry_with_bound_core()
    claim = TrainingAssignment(
        kind="learning",
        curriculum_ref="abc-sequence",
        cohort_eligibility={"core_ids": ["core-alpha"]},
        field_binding={"field_id": "f" * 64, "view_id": "v" * 64},
        origin="test",
        created_at=1000.0,
        lease=AssignmentLease(
            holder_core_id="core-alpha",
            issued_at=1000.0,
            expires_at=2000.0,
            lease_epoch=1,
        ),
    )

    bound = registry.enter_offline_training(
        "core-alpha",
        grant=AuthorityGrant.trainer(),
        claim=claim,
    )
    assert bound.mode is CoreStatus.OFFLINE_TRAINING

    # A live assignment held by a different core fails closed.
    registry = _registry_with_bound_core()
    wrong_holder = dataclasses.replace(
        claim,
        lease=AssignmentLease(
            holder_core_id="core-beta",
            issued_at=1000.0,
            expires_at=2000.0,
            lease_epoch=1,
        ),
    )
    with pytest.raises(InvalidModeTransitionError):
        registry.enter_offline_training(
            "core-alpha",
            grant=AuthorityGrant.trainer(),
            claim=wrong_holder,
        )

    # A terminal assignment cannot authorize a mode switch.
    registry = _registry_with_bound_core()
    completed = dataclasses.replace(claim, status="completed")
    with pytest.raises(InvalidModeTransitionError):
        registry.enter_offline_training(
            "core-alpha",
            grant=AuthorityGrant.trainer(),
            claim=completed,
        )
