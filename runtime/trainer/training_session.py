"""End-to-end OFFLINE_TRAINING session orchestrator for one real core.

``TrainingSession`` runs the real loop required by handoff steps 9 and 10 on
the local worker, entirely on constructor-injected state roots — no network,
no HTTP, no threads:

1. claim + activate a durable assignment in the Heart-owned
   ``AssignmentStore`` (idempotently keyed, crash-recoverable);
2. compile a derived training-attention view of the current canonical HEAD
   (exact D64 rows and source addresses), then advance that view after each
   Heart-committed response;
3. bind the registered core to exact generations and enter OFFLINE_TRAINING
   through the registry under trainer authority;
4. per attempt: ``LocalTrainingWorker.execute_attempt`` inhales the exact
   Soul (the persisted hot-layer reader state conditions the recurrent
   forward as ``initial_state``), runs the real CompleteField64D forward,
   discrete categorical loss against the assignment's explicit target text
   when the trainer bound one (otherwise the attended instructions text),
   backward pass, real optimizer step, and real Soul exhale — worker numbers
   are evidence only;
5. the session measures the honest post-step metrics of the returned
   (committed-candidate) parameters, seals them into a rail-auth
   ``WorkerEvidenceBundle``, and ``GateEngine.evaluate`` independently
   recomputes everything: claimed loss, gradients, and accuracy that disagree
   with recomputation beyond tolerance are rejected outright.  Replay
   protection is mandatory at the gate boundary: a submission without a
   durable nonce cache fails closed before any other check;
6. the outcome is committed through the assignment store's idempotently keyed
   ``record_attempt_outcome`` — exactly once, preserving every attempt; a
   trailing failed series escalates to supervisor inspection, never fabricated
   competence.  When a ``HeartTransactionBoundary`` is injected, Heart alone
   then materializes the typed response into the canonical
   ``training_responses`` region under ``AuthorityGrant.trainer()`` through
   the canonical typed-delta machinery, and the commit's successor field/tick
   advances the canonical head that ``successor_view`` compiles the next
   view from;
7. after each ACCEPTED learning step the session publishes the accepted
   parameter/optimizer/Soul bundle through ``CandidateStepBundleCoordinator``
   (durable optimization receipt, verified candidate checkpoint carrying the
   real parameter and optimizer bytes, candidate Soul manifest, and a
   three-phase transition chain over the real exhaled hot payload) and records
   landmarks at meaningful points (acceptance gates met, accepted steps, mode
   transitions);
8. preemption is survived across completed attempt writes: a fresh session
   over the same state root runs ``recover()``, reclaims/resumes the assignment,
   asserts exact journal continuity, and rehydrates the latest pass-or-fail
   model/optimizer bytes from the rolling attempt workspace; accepted
   checkpoints remain sparse landmarks;
9. ``exit_to_reasoning`` leaves training only at an exact accepted bundle
   boundary through the registry's symmetric ``exit_offline_training``.

All identities are canonical sha256 digests; every durable write goes through
the existing stores' atomic, content-addressed, idempotently keyed commands.
The session touches the worker's private model/optimizer handles only where
the worker API has no public accessor (checkpoint capture and lineage
rehydration); it never fabricates a loss, gradient, Soul, or success.
"""
from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping

import torch

from runtime.field import (
    CanonicalStateBranch,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_sha256,
)
from runtime.field.delta import FieldDelta, InsertText
from runtime.field.training_view import (
    TrainingAttentionView,
    TrainingAttentionViewCompiler,
    TrainingHistoryWindow,
)
from runtime.heart.authority import AuthorityGrant
from runtime.heart.errors import UnknownCoreError
from runtime.heart.rail_auth import DEFAULT_TTL_SECONDS, seal_envelope
from runtime.heart.registry import (
    CoreBinding,
    CoreRegistry,
    CoreStatus,
    OperatorCoreGrant,
)
from runtime.heart.transaction import HeartCommit, HeartTransactionBoundary
from runtime.soul import (
    SoulLayer,
    SoulSnapshot,
    SoulStore,
    SoulTemperature,
    SoulTransition,
    apply_soul_transition,
)
from training.complete_field_64d import (
    ReaderConfig,
    sequence_cross_entropy,
    teacher_char_accuracy,
)

from .assignments import (
    AssignmentAttempt,
    AssignmentKind,
    AssignmentStatus,
    AssignmentStore,
    TrainingAssignment,
)
from .attempt_workspace import AttemptWorkspaceError, AttemptWorkspaceStore
from .contracts import OrganKind, ParameterModuleDescriptor
from .learning import GovernedLearningPolicy
from .lifecycle import OptimizationStepReceipt
from .local_worker import (
    EOS_TOKEN_MARKER,
    LOCAL_TRAINING_ARCHITECTURE,
    UNTRAINED_PARAMETER_GENERATION,
    AttemptEvidenceBundle,
    LocalTrainingWorker,
    decode_hot_payload,
)
from .soul_candidates import CandidateSoulWorkspace
from .step_bundle import AcceptedTrainingStepBundle, CandidateStepBundleCoordinator
from .store import TrainerStateStore
from .supervisory_gates import (
    GateDecision,
    GateEngine,
    GateVerdict,
    MechanismProbeReport,
    MemoryNonceCache,
    SubmissionKind,
    WorkerEvidenceBundle,
    assert_resume_continuity,
)

TRAINING_SESSION_SCHEMA = "axon-trainer-training-session-v1"
TRAINING_SESSION_OUTCOME_SCHEMA = "axon-trainer-session-outcome-v1"
HOMEWORK_VERDICT_SCHEMA = "axon-trainer-homework-verdict-v1"

INITIAL_PARAMETER_GENERATION = UNTRAINED_PARAMETER_GENERATION
INITIAL_OPTIMIZER_GENERATION = "optimizer-init"

_SOUL_PUBLICATION_PHASES = ("first", "refined", "consolidated")
_CORE_STATUSES = (AssignmentStatus.CLAIMED, AssignmentStatus.ACTIVE)


class TrainingSessionError(RuntimeError):
    """A session input is malformed, stale, or unauthenticated; fail closed."""


@dataclass(frozen=True, slots=True)
class HomeworkVerdict:
    """Assignment-level judgment, deliberately separate from optimizer validity."""

    completed: bool
    reason: str
    target_sha256: str | None
    response_sha256: str
    response_terminated: bool
    optimizer_step_accepted: bool
    schema: str = HOMEWORK_VERDICT_SCHEMA
    verdict_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _required(self.reason, "reason"))
        object.__setattr__(
            self,
            "verdict_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": self.schema,
            "completed": self.completed,
            "reason": self.reason,
            "target_sha256": self.target_sha256,
            "response_sha256": self.response_sha256,
            "response_terminated": self.response_terminated,
            "optimizer_step_accepted": self.optimizer_step_accepted,
        }
        if include_id:
            value["verdict_id"] = self.verdict_id
        return value


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not (-float("inf") < value < float("inf")):
        raise ValueError(f"{label} must be finite")
    return value


def _serialize_state_dict(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def evaluate_homework_response(
    *,
    target_text: str | None,
    response_text: str,
    response_terminated: bool,
    optimizer_step_accepted: bool,
) -> HomeworkVerdict:
    """Judge homework completion without borrowing the optimizer gate's result."""

    response_hash = canonical_sha256(response_text)
    target_hash = None if target_text is None else canonical_sha256(target_text)
    if target_text is None:
        return HomeworkVerdict(
            completed=False,
            reason="assignment has no exact completion target",
            target_sha256=None,
            response_sha256=response_hash,
            response_terminated=response_terminated,
            optimizer_step_accepted=optimizer_step_accepted,
        )
    if not optimizer_step_accepted:
        reason = "optimizer step was not accepted; homework remains open"
    elif not response_terminated:
        reason = "free-running response did not terminate; homework remains open"
    elif response_text != target_text:
        reason = "free-running response did not exactly match the assignment target"
    else:
        reason = "accepted optimizer step produced the exact terminated assignment target"
    return HomeworkVerdict(
        completed=(
            optimizer_step_accepted
            and response_terminated
            and response_text == target_text
        ),
        reason=reason,
        target_sha256=target_hash,
        response_sha256=response_hash,
        response_terminated=response_terminated,
        optimizer_step_accepted=optimizer_step_accepted,
    )


@dataclass(frozen=True, slots=True)
class TrainingSessionConfig:
    """Deterministic configuration for one core's training session."""

    core_id: str
    curriculum_ref: str
    cohort_core_ids: tuple[str, ...]
    origin: str = "trainer:local-training-session"
    assignment_kind: AssignmentKind | str = AssignmentKind.LEARNING
    history_window: TrainingHistoryWindow = field(
        default_factory=lambda: TrainingHistoryWindow.latest_n(8)
    )
    lease_seconds: float = 600.0
    failure_threshold: int = 3
    max_attempts: int = 8
    seed: int = 11
    learning_rate: float = 1e-3
    max_grad_norm: float = 100.0
    target_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _required(self.core_id, "core_id"))
        object.__setattr__(
            self, "curriculum_ref", _required(self.curriculum_ref, "curriculum_ref")
        )
        object.__setattr__(self, "origin", _required(self.origin, "origin"))
        cohort = tuple(self.cohort_core_ids)
        if not cohort or any(not isinstance(item, str) or not item for item in cohort):
            raise ValueError("cohort_core_ids must be a non-empty tuple of core ids")
        if self.core_id not in cohort:
            raise ValueError("the training core must be a member of its cohort")
        object.__setattr__(self, "cohort_core_ids", tuple(sorted(set(cohort))))
        kind = (
            self.assignment_kind
            if isinstance(self.assignment_kind, AssignmentKind)
            else AssignmentKind(self.assignment_kind)
        )
        object.__setattr__(self, "assignment_kind", kind)
        if not isinstance(self.history_window, TrainingHistoryWindow):
            raise TypeError("history_window must be a TrainingHistoryWindow")
        object.__setattr__(
            self, "lease_seconds", _finite(self.lease_seconds, "lease_seconds")
        )
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if (
            isinstance(self.failure_threshold, bool)
            or not isinstance(self.failure_threshold, int)
            or self.failure_threshold < 1
        ):
            raise ValueError("failure_threshold must be a positive integer")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        object.__setattr__(
            self, "learning_rate", _finite(self.learning_rate, "learning_rate")
        )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        object.__setattr__(
            self, "max_grad_norm", _finite(self.max_grad_norm, "max_grad_norm")
        )
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.target_text is not None:
            object.__setattr__(
                self, "target_text", _required(self.target_text, "target_text")
            )


@dataclass(frozen=True, slots=True)
class TrainingSessionStep:
    """The durable outcome of one session loop iteration."""

    attempt: AssignmentAttempt
    verdict: GateVerdict
    worker_bundle: AttemptEvidenceBundle
    accepted_bundle: AcceptedTrainingStepBundle | None = None
    homework_verdict: HomeworkVerdict | None = None
    landmark_id: str | None = None
    heart_commit: HeartCommit | None = None

    @property
    def accepted(self) -> bool:
        return self.accepted_bundle is not None

    @property
    def optimizer_step_valid(self) -> bool:
        return self.accepted_bundle is not None

    @property
    def homework_completed(self) -> bool:
        return bool(
            self.homework_verdict is not None and self.homework_verdict.completed
        )


@dataclass(frozen=True, slots=True)
class TrainingSessionResult:
    """The final state of one ``run`` invocation."""

    assignment: TrainingAssignment
    steps: tuple[TrainingSessionStep, ...]
    stopped_reason: str


class TrainingSession:
    """Run the real training loop for one bound core on injected state roots.

    The registry is the Heart's in-memory control plane; every file-backed
    store is rebuilt from ``state_root`` so a full process restart is
    simulated by constructing a fresh session over the same root.
    """

    def __init__(
        self,
        *,
        state_root: Path | str,
        registry: CoreRegistry,
        snapshot: SharedFieldSnapshot,
        config: TrainingSessionConfig,
        operator_grant: OperatorCoreGrant,
        clock: Callable[[], float] | None = None,
        rail_environ: Mapping[str, str] | None = None,
        heart: HeartTransactionBoundary | None = None,
        canonical_branch: CanonicalStateBranch | None = None,
    ) -> None:
        if not isinstance(registry, CoreRegistry):
            raise TypeError("TrainingSession requires a CoreRegistry")
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("TrainingSession requires a SharedFieldSnapshot")
        if not isinstance(config, TrainingSessionConfig):
            raise TypeError("TrainingSession requires a TrainingSessionConfig")
        if not isinstance(operator_grant, OperatorCoreGrant):
            raise TypeError("TrainingSession requires the operator grant for rebinds")
        if operator_grant.core_id != config.core_id:
            raise TrainingSessionError("operator grant belongs to a different core")
        if heart is not None and not isinstance(heart, HeartTransactionBoundary):
            raise TypeError("heart must be a HeartTransactionBoundary or None")
        if canonical_branch is not None and not isinstance(
            canonical_branch, CanonicalStateBranch
        ):
            raise TypeError("canonical_branch must be a CanonicalStateBranch or None")
        self._state_root = Path(state_root).resolve(strict=False)
        self._registry = registry
        self._snapshot = snapshot
        self._config = config
        self._operator_grant = operator_grant
        self._clock = clock if clock is not None else time.time
        self._rail_environ = rail_environ
        self._heart = heart
        self._canonical_branch = canonical_branch
        if canonical_branch is not None:
            if canonical_branch.initialized:
                durable_head = canonical_branch.load_head()
                if durable_head.field_id != snapshot.field_id:
                    raise TrainingSessionError(
                        "the injected snapshot is not the durable canonical branch HEAD"
                    )
            else:
                canonical_branch.initialize(snapshot)

        self._assignment_store = AssignmentStore.active(
            self._state_root, clock=self._clock
        )
        self._soul_store = SoulStore.active(self._state_root)
        self._nonce_cache = MemoryNonceCache(self._state_root)
        self._trainer_store = TrainerStateStore.active(state_root=self._state_root)
        self._coordinator = CandidateStepBundleCoordinator(self._state_root)
        self._attempt_workspace = AttemptWorkspaceStore(self._state_root, retain=3)
        self._soul_workspace = CandidateSoulWorkspace(self._state_root)
        self._gate = GateEngine(
            model_config=replace(
                ReaderConfig(dropout=0.0),
                field_schema_version=snapshot.schema_version,
            ),
            environ=rail_environ,
            now=self._clock,
        )
        self._worker = LocalTrainingWorker(
            assignment_store=self._assignment_store,
            seed=config.seed,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            clock=self._clock,
            rail_environ=rail_environ,
        )
        self._view = TrainingAttentionViewCompiler().compile(
            snapshot,
            core_id=config.core_id,
            cohort_core_ids=config.cohort_core_ids,
            history_window=config.history_window,
        )
        # The canonical head this session commits against through Heart.  It
        # starts at the injected snapshot (the issued view's source) and
        # advances along each accepted HeartCommit; the issued view itself
        # stays bound to its own source tick for the life of the assignment.
        self._canonical_head = snapshot
        self._policy = GovernedLearningPolicy(
            optimizer="adamw",
            learning_rate=config.learning_rate,
            weight_decay=0.0,
            gradient_clip_norm=config.max_grad_norm,
            precision="fp32",
        )
        self._plan_id = f"training-session-plan:{config.core_id}"
        self._authorization_id = f"training-session-authorization:{config.core_id}"

        # Live session lineage (rebuilt from durable state on resume).
        self._assignment_id: str | None = None
        self._current_binding: CoreBinding | None = None
        self._base_generation = INITIAL_PARAMETER_GENERATION
        self._previous_checkpoint_id: str | None = None
        self._prior_save_sha: str | None = None
        self._last_parameter_generation: str | None = None
        self._last_optimizer_generation: str | None = None
        self._prev_param_state: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------ reads

    @property
    def view(self) -> TrainingAttentionView:
        return self._view

    @property
    def objective_text(self) -> str:
        text = self._view.compiled.region_text(LogicalRegion.TRAINER_INSTRUCTIONS)
        if not text:
            raise TrainingSessionError(
                "the attended trainer_instructions region is empty; no objective"
            )
        return text

    @property
    def target_text(self) -> str | None:
        """The explicit expected target bound at claim time, if the trainer
        authored one.  When present, supervised loss and gate recomputation
        bind THAT text (explicit task semantics); the core's typed proposal
        remains a separate free-running decode.  When absent the objective
        remains instruction reconstruction over the attended instructions."""

        return self._config.target_text

    def _assignment_target(self, record: TrainingAssignment) -> str | None:
        """The authoritative explicit target of one durable assignment."""

        if record.target_text is not None:
            return record.target_text
        return self._config.target_text

    @property
    def assignment_store(self) -> AssignmentStore:
        return self._assignment_store

    @property
    def gate(self) -> GateEngine:
        return self._gate

    @property
    def nonce_cache(self) -> MemoryNonceCache:
        return self._nonce_cache

    @property
    def worker(self) -> LocalTrainingWorker:
        return self._worker

    def latest_accepted_bundle(self) -> AcceptedTrainingStepBundle | None:
        """The newest accepted step bundle for this core, from durable state."""

        root = self._trainer_store.candidates_dir / self._config.core_id
        if not root.is_dir():
            return None
        best: AcceptedTrainingStepBundle | None = None
        for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
            bundle = self._coordinator.latest_bundle(
                self._config.core_id, candidate.name
            )
            if bundle is not None and (best is None or bundle.step > best.step):
                best = bundle
        return best

    # -------------------------------------------------------------------- run

    def run(
        self,
        *,
        assignment_id: str | None = None,
        max_attempts: int | None = None,
    ) -> TrainingSessionResult:
        """Run the session loop: claim or resume, train, gate, commit, publish.

        With ``assignment_id=None`` a fresh assignment is claimed and
        activated; otherwise the durable assignment is resumed (see
        ``resume_lineage``).  The loop stops at the attempt limit or when the
        assignment leaves ACTIVE (pause on collapse, escalation on a failed
        series, completion).
        """

        limit = self._config.max_attempts if max_attempts is None else int(max_attempts)
        if limit < 1:
            raise ValueError("max_attempts must be a positive integer")
        if assignment_id is None:
            assignment = self._claim_and_activate()
            self._assignment_store.recover()
            self._enter_training_mode(assignment)
            self._anchor_worker_lineage()
        else:
            assignment = self.resume_lineage(assignment_id)

        steps: list[TrainingSessionStep] = []
        reason = "max_attempts"
        while True:
            record = self._assignment_store.get_assignment(assignment.assignment_id)
            if record.status is not AssignmentStatus.ACTIVE:
                reason = record.status.value
                break
            if len(steps) >= limit:
                reason = "max_attempts"
                break
            step = self._attempt_step(record)
            steps.append(step)
            if step.verdict.decision is GateDecision.ACCEPTED:
                continue
            if step.verdict.decision is GateDecision.PAUSED:
                reason = "paused"
                break
            fresh = self._assignment_store.get_assignment(assignment.assignment_id)
            if fresh.status is AssignmentStatus.ESCALATED:
                self._record_inspection(fresh, step)
                reason = "escalated"
                break
            if fresh.status is AssignmentStatus.PAUSED:
                reason = "paused"
                break
        final = self._assignment_store.get_assignment(assignment.assignment_id)
        return TrainingSessionResult(
            assignment=final, steps=tuple(steps), stopped_reason=reason
        )

    def resume_lineage(self, assignment_id: str) -> TrainingAssignment:
        """Recover, resume, and rehydrate an assignment after a stop.

        Runs ``recover()`` on the assignment store, reclaims an expired lease
        (preemption never deletes attempts and never decides outcomes), or
        resumes a paused assignment with a fresh lease, asserts exact resume
        continuity from the durable journal, rehydrates the exact accepted
        parameter/optimizer lineage from the verified checkpoint artifact,
        and re-enters OFFLINE_TRAINING under trainer authority.
        """

        self._assignment_store.recover()
        record = self._assignment_store.get_assignment(assignment_id)
        if (
            record.status in _CORE_STATUSES
            and record.lease is not None
            and record.lease.expired(self._clock())
        ):
            record = self._assignment_store.reclaim_expired(
                record.assignment_id,
                expected_revision=record.revision,
                idempotency_key=(
                    f"training-session-reclaim:{record.assignment_id}:{record.revision}"
                ),
                reason="preemption: lease expired before restart",
            )
        if record.status is AssignmentStatus.PAUSED:
            record = self._assignment_store.resume(
                record.assignment_id,
                core_id=self._config.core_id,
                expected_revision=record.revision,
                idempotency_key=(
                    f"training-session-resume:{record.assignment_id}:{record.attempt_count}"
                ),
                lease_seconds=self._config.lease_seconds,
                reason="session restart: exact lineage resume",
            )
        if record.status is not AssignmentStatus.ACTIVE:
            raise TrainingSessionError(
                f"assignment {record.assignment_id} is not resumable "
                f"(status={record.status.value})"
            )
        attempts = self._assignment_store.attempts_for(record.assignment_id)
        if attempts:
            # Fail closed unless the durable journal resumes exactly:
            # unique contiguous indices and an exact parameter chain.
            assert_resume_continuity(attempts)
            self._rehydrate_worker(record.assignment_id, attempts)
        self._enter_training_mode(record)
        self._anchor_worker_lineage()
        return record

    # --------------------------------------------------------- session phases

    def _anchor_worker_lineage(self) -> None:
        """Anchor the honest prior-parameter identity at the current bytes."""

        self._ensure_worker_model()
        self._prior_save_sha = hashlib.sha256(
            _serialize_state_dict(self._worker.model)
        ).hexdigest()
        self._prev_param_state = self._clone_param_state()

    def _claim_and_activate(self) -> TrainingAssignment:
        view = self._view
        assignment = self._assignment_store.create_assignment(
            kind=self._config.assignment_kind,
            curriculum_ref=self._config.curriculum_ref,
            cohort_eligibility={"core_ids": list(self._config.cohort_core_ids)},
            field_binding={
                "field_id": view.source_field_id,
                "view_id": view.view_id,
                "tick_id": view.source_tick_id,
            },
            holder_core_id=self._config.core_id,
            origin=self._config.origin,
            lease_seconds=self._config.lease_seconds,
            target_text=self._config.target_text,
            idempotency_key=(
                f"training-session-claim:{self._config.core_id}:"
                f"{view.source_field_id}:{view.source_tick_id}:"
                f"{self._config.curriculum_ref}"
            ),
            failure_threshold=self._config.failure_threshold,
        )
        return self._assignment_store.activate(
            assignment.assignment_id,
            holder_core_id=self._config.core_id,
            expected_revision=assignment.revision,
            idempotency_key=f"training-session-activate:{assignment.assignment_id}",
        )

    def _enter_training_mode(self, assignment: TrainingAssignment) -> None:
        core_id = self._config.core_id
        descriptor = self._registry.get(core_id)
        if descriptor.architecture_id != LOCAL_TRAINING_ARCHITECTURE:
            raise TrainingSessionError(
                f"core {core_id!r} is registered for architecture "
                f"{descriptor.architecture_id!r}, not {LOCAL_TRAINING_ARCHITECTURE!r}"
            )
        head = self._soul_store.ensure_core(
            core_id=core_id,
            architecture_id=LOCAL_TRAINING_ARCHITECTURE,
            parameter_generation=INITIAL_PARAMETER_GENERATION,
        )
        try:
            self._registry.binding(core_id)
        except UnknownCoreError:
            self._registry.bind_core(
                CoreBinding(
                    core_id=core_id,
                    architecture_id=LOCAL_TRAINING_ARCHITECTURE,
                    parameter_generation=INITIAL_PARAMETER_GENERATION,
                    optimizer_generation=INITIAL_OPTIMIZER_GENERATION,
                    soul_id=head.soul_id,
                    mode=CoreStatus.ACTIVE,
                ),
                grant=self._operator_grant,
            )
        entered = self._registry.enter_offline_training(
            core_id,
            grant=AuthorityGrant.trainer(),
            claim=assignment,
        )
        self._current_binding = entered
        self._assignment_id = assignment.assignment_id

    def _attempt_step(self, record: TrainingAssignment) -> TrainingSessionStep:
        core_id = self._config.core_id
        view = self._view
        objective = self.objective_text
        target = self._assignment_target(record)
        soul_branch = self._soul_store.branch(core_id)
        inhaled_soul = soul_branch.load_head()
        inhaled_hot = inhaled_soul.layer(SoulTemperature.HOT)
        worker_binding = self._worker_binding(inhaled_soul)

        # (4) real execution: Soul-conditioned forward, loss against the
        # explicit assignment target when one is bound, backward, optimizer
        # step, Soul exhale; the worker records the typed attempt as
        # evidence only.
        bundle = self._worker.execute_attempt(
            worker_binding, record, view, soul_branch, target_text=target
        )

        # (5) honest metrics of the RETURNED (post-step) parameters: these are
        # the claimed values, and the gate recomputes them independently
        # against the same explicit target (or the objective text when the
        # assignment binds no explicit target).
        supervised = target if target is not None else objective
        loss, accuracy, grad_norm = self._post_step_metrics(
            supervised,
            initial_state=self._decode_soul_initial_state(inhaled_hot.payload),
        )
        parameter_bytes = _serialize_state_dict(self._worker.model)
        prior_save_sha = self._prior_save_sha
        assert self._worker._optimizer is not None
        workspace = self._attempt_workspace.save(
            assignment_id=record.assignment_id,
            core_id=core_id,
            attempt_id=bundle.attempt_id,
            attempt_index=bundle.attempt_index,
            parameter_generation=bundle.parameter_generation_after,
            optimizer_generation=bundle.optimizer_generation_after,
            model=self._worker.model,
            optimizer=self._worker._optimizer,
        )

        submission = WorkerEvidenceBundle(
            core_id=core_id,
            assignment_id=record.assignment_id,
            view_id=view.view_id,
            field_id=view.source_field_id,
            tick_id=view.source_tick_id,
            submission_kind=SubmissionKind.LEARNING_RESULT,
            parameter_bytes=parameter_bytes,
            claimed_loss=loss,
            claimed_grad_norm=grad_norm,
            claimed_accuracy=accuracy,
            prior_parameter_sha256=prior_save_sha,
            optimizer_generation=bundle.optimizer_generation_after,
            emission_ref={
                "schema": "axon-trainer-attempt-emission-v1",
                "emission_id": bundle.emission.emission_id,
                "rail_id": view.compiled.rail_id,
                "view_id": view.view_id,
            },
            soul_lineage={
                "branch_id": soul_branch.branch_id,
                "soul_id_before": bundle.soul_id_before,
                "soul_id_after": bundle.soul_id_after,
                "generation_before": None,
                "soul_transition_id": None,
            },
            soul_payload=inhaled_hot.payload,
            soul_media_type=(inhaled_hot.media_type if inhaled_hot.payload else None),
            soul_tensor_layout=(inhaled_hot.tensor_layout if inhaled_hot.payload else None),
        )
        submission = replace(
            submission,
            envelope=seal_envelope(
                core_identity=core_id,
                assignment_id=record.assignment_id,
                payload=submission.sealed_payload_bytes,
                ttl_seconds=DEFAULT_TTL_SECONDS,
                now=int(self._clock()),
                environ=self._rail_environ,
            ),
        )
        verdict = self._gate.evaluate(
            submission,
            record,
            view,
            target_text=target if target is not None else objective,
            store=None,
            snapshot=self._snapshot,
            nonce_cache=self._nonce_cache,
        )
        response_text = "".join(
            character
            for character in bundle.emission.characters
            if character != EOS_TOKEN_MARKER
        )
        homework_verdict = evaluate_homework_response(
            target_text=target,
            response_text=response_text,
            response_terminated=bundle.emission.terminated,
            optimizer_step_accepted=(verdict.decision is GateDecision.ACCEPTED),
        )

        # (6) commit the outcome exactly once through the keyed store command,
        # preserving the worker's own evidence fields.
        attempt = self._assignment_store.get_attempt(bundle.attempt_id)
        outcome_evidence = {
            **attempt.evidence,
            "schema": TRAINING_SESSION_OUTCOME_SCHEMA,
            "bundle_sha256": submission.bundle_sha256,
            "parameter_sha256": bundle.parameter_generation_after,
            "prior_parameter_sha256": self._last_parameter_generation,
            "optimizer_generation": bundle.optimizer_generation_after,
            "attempt_workspace_id": workspace.record_id,
            "explicit_target_sha256": (
                canonical_sha256(target) if target is not None else None
            ),
            "decision": verdict.decision.value,
            "reasons": list(verdict.reasons),
            "verdict_id": verdict.verdict_id,
            "homework_verdict": homework_verdict.to_canonical_dict(),
            "observed_at": float(self._clock()),
        }
        if verdict.recomputed is not None:
            outcome_evidence["recomputed"] = verdict.recomputed.to_canonical_dict()
        fresh = self._assignment_store.get_assignment(record.assignment_id)
        committed, updated = self._assignment_store.record_attempt_outcome(
            bundle.attempt_id,
            outcome=verdict.outcome,
            evidence=outcome_evidence,
            expected_revision=fresh.revision,
            idempotency_key=f"training-session-outcome:{bundle.attempt_id}",
        )

        # Lineage advances on every real attempt (a rejected step still
        # changed parameter bytes; the journal chain must reflect that).
        self._prior_save_sha = hashlib.sha256(parameter_bytes).hexdigest()
        self._last_parameter_generation = bundle.parameter_generation_after
        self._last_optimizer_generation = bundle.optimizer_generation_after

        # (6) Heart alone materializes the typed response into the canonical
        # training_responses region; the successor field/tick it returns
        # drives the next view (see ``successor_view``).
        heart_commit = None
        if self._heart is not None:
            heart_commit = self._commit_training_response(
                record,
                bundle,
                verdict,
                target,
            )
            successor = self.successor_view()
            updated = self._assignment_store.adjust_curriculum(
                updated.assignment_id,
                author="trainer:training-session-heart",
                expected_revision=updated.revision,
                idempotency_key=f"training-session-successor-view:{bundle.attempt_id}",
                field_binding={
                    "field_id": successor.source_field_id,
                    "view_id": successor.view_id,
                    "tick_id": successor.source_tick_id,
                },
                reason="attend the Heart-committed response on the next training tick",
            )
            self._snapshot = self._canonical_head
            self._view = successor

        accepted = None
        landmark_id = None
        if verdict.decision is GateDecision.ACCEPTED:
            accepted = self._publish_accepted_step(bundle, verdict)
            landmark_id = self._mark_acceptance_landmark(accepted, bundle, verdict)
            self._rebind_binding(bundle, accepted)
            if homework_verdict.completed:
                updated = self._assignment_store.complete(
                    updated.assignment_id,
                    holder_core_id=core_id,
                    expected_revision=updated.revision,
                    idempotency_key=f"training-session-homework-complete:{bundle.attempt_id}",
                    reason=(
                        "exact terminated free-running response matched the assignment target; "
                        "optimizer admissibility was evaluated separately"
                    ),
                )

        if verdict.decision is GateDecision.PAUSED and (
            updated.status is AssignmentStatus.ACTIVE
        ):
            self._assignment_store.pause(
                updated.assignment_id,
                holder_core_id=core_id,
                expected_revision=updated.revision,
                idempotency_key=f"training-session-pause:{bundle.attempt_id}",
                reason="; ".join(verdict.reasons),
            )

        return TrainingSessionStep(
            attempt=committed,
            verdict=verdict,
            worker_bundle=bundle,
            accepted_bundle=accepted,
            homework_verdict=homework_verdict,
            landmark_id=landmark_id,
            heart_commit=heart_commit,
        )

    # ------------------------------------------------------ heart commitment

    def _commit_training_response(
        self,
        record: TrainingAssignment,
        bundle: AttemptEvidenceBundle,
        verdict: GateVerdict,
        target: str | None,
    ) -> HeartCommit:
        """Materialize one typed training response through the Heart boundary.

        The worker's typed emission (its actual answer characters, excluding
        the typed EOS terminal marker) becomes one attempt span in the
        canonical ``training_responses`` region — the exact convention the
        training-attention view history window selects by.  Heart commits
        under ``AuthorityGrant.trainer()``: the trainer class governs exactly
        the two training regions and the canonical typed-delta machinery
        binds every committed byte.  The returned commit's successor
        field/tick is the canonical head the next view compiles from.

        The typed emission identity, the exact objective/target identities,
        and the gate verdict bind to the span through provenance and
        container refs, so the canonical response dereferences back to the
        durable attempt record and the trainer-owned verdict.
        """

        assert self._heart is not None
        base = self._canonical_head
        region = base.region(LogicalRegion.TRAINING_RESPONSES)
        characters = [
            character
            for character in bundle.emission.characters
            if character != EOS_TOKEN_MARKER
        ]
        response_text = "".join(characters)
        if not response_text:
            raise TrainingSessionError(
                "the free-running emission terminated before producing response characters"
            )
        bundle_sha256 = canonical_sha256(bundle.to_canonical_dict())
        delta = FieldDelta(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            author_core_id=self._config.core_id,
            pass_id=f"training-response:{bundle.attempt_id}",
            operations=(
                InsertText(
                    LogicalRegion.TRAINING_RESPONSES,
                    offset=len(region.text),
                    text=response_text,
                    provenance=(
                        f"training_response:{record.assignment_id}:"
                        f"{bundle.attempt_id}:{verdict.decision.value}"
                    ),
                    container_refs=(
                        bundle.attempt_id,
                        bundle.emission.emission_id,
                        bundle_sha256,
                        verdict.verdict_id,
                    ),
                    edge_refs=(
                        canonical_sha256(
                            {
                                "objective_text_sha256": canonical_sha256(
                                    self.objective_text
                                ),
                                "explicit_target_sha256": (
                                    canonical_sha256(target)
                                    if target is not None
                                    else None
                                ),
                            }
                        ),
                    ),
                ),
            ),
            evidence=(
                bundle.attempt_id,
                verdict.verdict_id,
            ),
        )
        commit = self._heart.commit(
            base,
            delta,
            AuthorityGrant.trainer(),
            valve_provenance={
                "schema": "axon-trainer-training-response-v1",
                "assignment_id": record.assignment_id,
                "attempt_id": bundle.attempt_id,
                "attempt_index": bundle.attempt_index,
                "core_id": bundle.core_id,
                "decision": verdict.decision.value,
                "bundle_sha256": bundle_sha256,
                "emission_sha256": bundle.emission.emission_id,
            },
        )
        if self._canonical_branch is not None:
            persisted = self._canonical_branch.commit(
                delta,
                permitted_regions=AuthorityGrant.trainer().governed_regions,
                metadata={
                    "schema": "axon-trainer-training-response-v1",
                    "assignment_id": record.assignment_id,
                    "attempt_id": bundle.attempt_id,
                    "verdict_id": verdict.verdict_id,
                },
            )
            if persisted.field_id != commit.successor.field_id:
                raise TrainingSessionError(
                    "Heart successor differs from the durable canonical branch commit"
                )
            self._canonical_head = persisted
        else:
            self._canonical_head = commit.successor
        return commit

    @property
    def canonical_head(self) -> SharedFieldSnapshot:
        """The current canonical field head this session commits against."""

        return self._canonical_head

    def successor_view(
        self,
        *,
        history_window: TrainingHistoryWindow | None = None,
    ) -> TrainingAttentionView:
        """Compile the next beat's view from the Heart-committed successor.

        The view is derived from the current canonical head (the successor
        field of the latest committed training response), so its source tick
        advances exactly along the Heart commit chain and its history window
        can attend the core's own committed attempt spans.
        """

        return TrainingAttentionViewCompiler().compile(
            self._canonical_head,
            core_id=self._config.core_id,
            cohort_core_ids=self._config.cohort_core_ids,
            history_window=(
                self._config.history_window
                if history_window is None
                else history_window
            ),
        )

    # ------------------------------------------------------------- acceptance

    def _publish_accepted_step(
        self,
        bundle: AttemptEvidenceBundle,
        verdict: GateVerdict,
    ) -> AcceptedTrainingStepBundle:
        """Publish one accepted parameter/optimizer/Soul step boundary.

        Every artifact is real and durable: the optimization receipt enters
        the trainer journal, the checkpoint captures the exact parameter and
        optimizer bytes (content-addressed and hash-verified on every later
        load), and the three-phase Soul transition chain carries the exact
        hot-layer payload the worker exhaled into SoulStore.
        """

        if verdict.recomputed is None:
            raise TrainingSessionError(
                "accepted verdict carries no recomputed metrics"
            )
        core_id = self._config.core_id
        model = self._worker.model
        step = bundle.attempt_index + 1
        candidate_generation = bundle.parameter_generation_after
        changed, unchanged = self._changed_tensor_names(model)

        receipt = OptimizationStepReceipt(
            module_id=core_id,
            candidate_generation_id=candidate_generation,
            plan_id=self._plan_id,
            authorization_id=self._authorization_id,
            learning_policy_id=self._policy.policy_id,
            step=step,
            micro_step=1,
            microbatches_accumulated=1,
            loss=verdict.recomputed.loss,
            gradient_l2=verdict.recomputed.grad_norm or 0.0,
            gradient_clip_norm=self._config.max_grad_norm,
            learning_rate=self._config.learning_rate,
            weight_decay=0.0,
            precision_mode="fp32",
            update_l2=None,
            telemetry_frame_id=canonical_sha256(
                {
                    "schema": "axon-trainer-session-telemetry-v1",
                    "attempt_id": bundle.attempt_id,
                    "verdict_id": verdict.verdict_id,
                }
            ),
            changed_tensor_names=changed,
            unchanged_tensor_names=unchanged,
        )
        self._trainer_store.append_optimization_step(receipt)

        checkpoint = self._trainer_store.save_candidate_checkpoint(
            module=model,
            descriptor=ParameterModuleDescriptor(
                module_id=core_id,
                organ_kind=OrganKind.REASONING_CORE,
                generation_id=candidate_generation,
                architecture=LOCAL_TRAINING_ARCHITECTURE,
                d_model=64,
            ),
            base_generation_id=self._base_generation,
            plan_id=self._plan_id,
            authorization_id=self._authorization_id,
            learning_policy=self._policy,
            step=step,
            micro_step=1,
            accumulation_index=0,
            current_learning_rate=self._config.learning_rate,
            accumulated_loss_sum=verdict.recomputed.loss,
            optimizer=self._worker._optimizer,  # no public accessor exists
            previous_checkpoint_id=self._previous_checkpoint_id,
        )

        manifest = self._soul_workspace.prepare(
            candidate_id=candidate_generation,
            core_id=core_id,
            runtime_episode_session_id=f"training-session:{bundle.assignment_id}",
            whole_episode_split="train",
            soul_trajectory_ids=(bundle.attempt_id,),
            candidate_parameter_generation=candidate_generation,
        )
        branch = self._soul_workspace.branch(candidate_generation, core_id)
        head = branch.load_head()
        hot = self._soul_store.branch(core_id).load_head().layer(SoulTemperature.HOT)
        transitions: list[SoulTransition] = []
        cursor = head
        for phase in _SOUL_PUBLICATION_PHASES:
            transition = SoulTransition(
                core_id=core_id,
                architecture_id=LOCAL_TRAINING_ARCHITECTURE,
                parameter_generation=candidate_generation,
                before_soul_id=cursor.soul_id,
                before_generation=cursor.generation,
                tick_uid=f"{bundle.field_id}:{bundle.tick_id}",
                request_id=(
                    f"training-session:{bundle.assignment_id}:"
                    f"{bundle.attempt_index}:{phase}"
                ),
                phase=phase,
                updates=(
                    SoulLayer(
                        temperature=SoulTemperature.HOT,
                        payload=hot.payload,
                        media_type=hot.media_type,
                        tensor_layout=hot.tensor_layout,
                    ),
                ),
            )
            transitions.append(transition)
            cursor = apply_soul_transition(cursor, transition)

        accepted = self._coordinator.accept_step(
            optimization_receipt=receipt,
            checkpoint=checkpoint,
            soul_manifest=manifest,
            transitions=tuple(transitions),
        )
        self._base_generation = candidate_generation
        self._previous_checkpoint_id = checkpoint.checkpoint_id
        self._prev_param_state = self._clone_param_state()
        return accepted

    def _mark_acceptance_landmark(
        self,
        accepted: AcceptedTrainingStepBundle,
        bundle: AttemptEvidenceBundle,
        verdict: GateVerdict,
    ) -> str:
        label = (
            "acceptance-gates-met"
            if accepted.step == 1
            else f"accepted-step:{accepted.step}"
        )
        return self._coordinator.mark_landmark(
            accepted,
            label=label,
            evidence_ids=(bundle.attempt_id, verdict.verdict_id),
        )

    def _rebind_binding(
        self, bundle: AttemptEvidenceBundle, accepted: AcceptedTrainingStepBundle
    ) -> None:
        # The binding anchors to the Heart-ACCEPTED Soul generation (the
        # candidate publication chain), not the live working head; the worker
        # binding overrides the Soul with the live head per attempt.
        core_id = self._config.core_id
        binding = CoreBinding(
            core_id=core_id,
            architecture_id=LOCAL_TRAINING_ARCHITECTURE,
            parameter_generation=bundle.parameter_generation_after,
            optimizer_generation=bundle.optimizer_generation_after,
            soul_id=accepted.after_soul_id,
            mode=CoreStatus.OFFLINE_TRAINING,
        )
        self._registry.rebind_core(binding, grant=self._operator_grant)
        self._current_binding = binding

    # ------------------------------------------------------- mode transitions

    def exit_to_reasoning(self) -> CoreBinding:
        """Leave OFFLINE_TRAINING only at the exact accepted boundary."""

        accepted = self.latest_accepted_bundle()
        if accepted is None:
            raise TrainingSessionError(
                "no accepted bundle is published; refusing to leave training "
                "without an exact accepted boundary"
            )
        binding = self._registry.exit_offline_training(
            self._config.core_id,
            grant=AuthorityGrant.trainer(),
            accepted_bundle=accepted,
        )
        self._coordinator.mark_landmark(
            accepted,
            label="mode:reasoning-restored",
            evidence_ids=(accepted.bundle_id,),
        )
        self._current_binding = binding
        return binding

    # ----------------------------------------------------------------- probes

    def probe_output_sha256(self) -> str:
        """Deterministic forward-probe digest of the current worker model.

        Re-running this over the fixed attended view and objective after any
        stop/restart/mode round trip must reproduce the digest byte-exactly
        when identity is preserved.
        """

        torch.manual_seed(self._config.seed)
        model = self._worker.model
        model.eval()
        model.zero_grad(set_to_none=True)
        hot = self._soul_store.branch(self._config.core_id).load_head().layer(
            SoulTemperature.HOT
        )
        reader_state, memory, _manifest = model.read_compiled_with_memory(
            self._view.compiled,
            initial_state=self._decode_soul_initial_state(hot.payload),
        )
        logits, targets = model.decode_teacher(
            reader_state, self.objective_text, head=0, memory=memory
        )
        digest = hashlib.sha256()
        digest.update(
            logits.detach().to(device="cpu", dtype=torch.float64).numpy().tobytes()
        )
        digest.update(targets.detach().to(device="cpu").numpy().tobytes())
        model.zero_grad(set_to_none=True)
        return digest.hexdigest()

    def run_counterfactual_probes(
        self,
        claims: tuple[str, ...],
        *,
        probe_snapshot: SharedFieldSnapshot | None = None,
    ) -> tuple[MechanismProbeReport, ...]:
        """Counterfactual ablation probes: which claimed mechanisms are real."""

        if self._assignment_id is None:
            raise TrainingSessionError("run the session before probing")
        if not claims:
            raise ValueError("at least one counterfactual claim is required")
        record = self._assignment_store.get_assignment(self._assignment_id)
        hot = self._soul_store.branch(self._config.core_id).load_head().layer(
            SoulTemperature.HOT
        )
        submission = WorkerEvidenceBundle(
            core_id=self._config.core_id,
            assignment_id=record.assignment_id,
            view_id=self._view.view_id,
            field_id=self._view.source_field_id,
            tick_id=self._view.source_tick_id,
            submission_kind=SubmissionKind.COUNTERFACTUAL_PROBE,
            parameter_bytes=_serialize_state_dict(self._worker.model),
            claimed_loss=None,
            claimed_grad_norm=None,
            claimed_accuracy=None,
            prior_parameter_sha256=self._prior_save_sha,
            optimizer_generation=self._last_optimizer_generation,
            counterfactual_claims=claims,
            soul_payload=hot.payload,
            soul_media_type=hot.media_type if hot.payload else None,
            soul_tensor_layout=hot.tensor_layout if hot.payload else None,
        )
        submission = replace(
            submission,
            envelope=seal_envelope(
                core_identity=self._config.core_id,
                assignment_id=record.assignment_id,
                payload=submission.sealed_payload_bytes,
                ttl_seconds=DEFAULT_TTL_SECONDS,
                now=int(self._clock()),
                environ=self._rail_environ,
            ),
        )
        verdict = self._gate.evaluate(
            submission,
            record,
            self._view,
            target_text=self._assignment_target(record) or self.objective_text,
            store=None,
            snapshot=self._snapshot,
            nonce_cache=self._nonce_cache,
            probe_snapshot=probe_snapshot,
        )
        return verdict.mechanism_reports

    # -------------------------------------------------------------- internals

    def _worker_binding(self, live_head: SoulSnapshot) -> CoreBinding:
        """The binding handed to the worker for the next attempt.

        Carries the live SoulStore head and the worker's tracked parameter
        and optimizer generations; the registry binding remains the anchor
        that advances only along accepted bundles.
        """

        assert self._current_binding is not None
        binding = replace(self._current_binding, soul_id=live_head.soul_id)
        if self._last_parameter_generation is not None:
            binding = replace(
                binding,
                parameter_generation=self._last_parameter_generation,
                optimizer_generation=self._last_optimizer_generation,
            )
        return binding

    def _decode_soul_initial_state(self, payload: bytes) -> torch.Tensor | None:
        if not payload:
            return None
        array, _metrics = decode_hot_payload(payload)
        state = torch.from_numpy(array)
        expected = tuple(self._worker.model.initial_state.shape)
        if tuple(state.shape) == (1, *expected):
            state = state.squeeze(0)
        if tuple(state.shape) != expected:
            raise TrainingSessionError(
                f"Soul state shape {tuple(state.shape)} does not match {expected}"
            )
        return state.to(
            device=self._worker.model.device,
            dtype=self._worker.model.initial_state.dtype,
        )

    def _post_step_metrics(
        self,
        supervised_text: str,
        *,
        initial_state: torch.Tensor | None,
    ) -> tuple[float, float, float]:
        """Honest metrics of the returned parameters on the same target.

        Runs the identical forward the gate engine recomputes, so claimed
        values agree with recomputation to the bit on the same device.
        """

        model = self._worker.model
        model.eval()
        model.zero_grad(set_to_none=True)
        reader_state, memory, _manifest = model.read_compiled_with_memory(
            self._view.compiled,
            initial_state=initial_state,
        )
        logits, targets = model.decode_teacher(
            reader_state, supervised_text, head=0, memory=memory
        )
        loss = sequence_cross_entropy(logits, targets)
        loss_value = float(loss.item())
        accuracy = float(teacher_char_accuracy(logits, targets))
        loss.backward()
        total = torch.zeros((), dtype=torch.float64)
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            total = total + parameter.grad.detach().to(torch.float64).pow(2).sum()
        grad_norm = math.sqrt(float(total.item()))
        model.zero_grad(set_to_none=True)
        if not (-float("inf") < loss_value < float("inf")):
            raise TrainingSessionError(
                "post-step loss is not finite; refusing fabricated evidence"
            )
        return loss_value, accuracy, grad_norm

    def _ensure_worker_model(self) -> None:
        self._worker._ensure_model(self._view.source_schema_version)

    def _clone_param_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().to(device="cpu").clone()
            for name, parameter in self._worker.model.named_parameters()
        }

    def _changed_tensor_names(
        self, model: torch.nn.Module
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        prior = self._prev_param_state
        changed: list[str] = []
        unchanged: list[str] = []
        for name, parameter in model.named_parameters():
            if prior is None or not torch.equal(parameter.detach(), prior[name]):
                changed.append(name)
            else:
                unchanged.append(name)
        return tuple(sorted(changed)), tuple(sorted(unchanged))

    def _rehydrate_worker(
        self,
        assignment_id: str,
        attempts: tuple[AssignmentAttempt, ...],
    ) -> None:
        """Restore the exact in-progress parameter/optimizer lineage after a stop.

        The rolling attempt workspace advances for pass and fail outcomes, so
        the next attempt begins from the latest attempted bytes.  Accepted
        checkpoints still provide landmark ancestry and promotion authority.
        """

        try:
            recovered = self._attempt_workspace.load_latest(
                assignment_id=assignment_id,
                core_id=self._config.core_id,
            )
        except AttemptWorkspaceError as exc:
            raise TrainingSessionError(
                f"cannot recover the latest attempt workspace: {exc}"
            ) from exc
        if recovered is None:
            raise TrainingSessionError(
                "durable attempts exist but their recovery workspace is missing"
            )
        workspace, payload = recovered
        latest = attempts[-1]
        if (
            workspace.attempt_id != latest.attempt_id
            or workspace.attempt_index != latest.attempt_index
            or workspace.parameter_generation != latest.parameter_generation
            or workspace.optimizer_generation != latest.optimizer_generation
        ):
            raise TrainingSessionError(
                "latest attempt workspace does not match the durable attempt journal"
            )
        self._ensure_worker_model()
        self._worker.model.load_state_dict(payload["module_state_dict"], strict=True)
        assert self._worker._optimizer is not None
        self._worker._optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._worker._parameter_generation = workspace.parameter_generation
        self._last_parameter_generation = workspace.parameter_generation
        self._last_optimizer_generation = workspace.optimizer_generation

        accepted = self.latest_accepted_bundle()
        if accepted is None:
            return
        checkpoint = self._coordinator.checkpoint_for_bundle(accepted)
        self._base_generation = accepted.candidate_generation_id
        self._previous_checkpoint_id = accepted.checkpoint_id
        # Verify the accepted checkpoint remains readable even when later
        # failed WIP is the actual resume point.
        self._trainer_store.load_verified_candidate_checkpoint(checkpoint)

    def _record_inspection(
        self, record: TrainingAssignment, step: TrainingSessionStep
    ) -> None:
        """A failed series reached escalation: durable supervisor inspection."""

        self._assignment_store.add_critique(
            step.attempt.attempt_id,
            author="trainer:training-session",
            guidance=(
                f"failed series escalated after {record.consecutive_failures} "
                "consecutive failures; supervisor inspection required; every "
                "attempt is preserved and no competence is fabricated"
            ),
            idempotency_key=f"training-session-inspection:{step.attempt.attempt_id}",
        )


__all__ = [
    "HOMEWORK_VERDICT_SCHEMA",
    "INITIAL_OPTIMIZER_GENERATION",
    "INITIAL_PARAMETER_GENERATION",
    "TRAINING_SESSION_OUTCOME_SCHEMA",
    "TRAINING_SESSION_SCHEMA",
    "HomeworkVerdict",
    "TrainingSession",
    "TrainingSessionConfig",
    "TrainingSessionError",
    "TrainingSessionResult",
    "TrainingSessionStep",
    "evaluate_homework_response",
]
