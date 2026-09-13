"""Trainer-side evaluation and supervisory gates for real runtime training.

Worker-reported ``success``, loss, gradient norms, and accuracies are
EVIDENCE INPUTS, NEVER AUTHORITY.  ``GateEngine.evaluate`` independently:

1. authenticates the bundle's sealed payload through
   ``runtime.heart.rail_auth.verify_envelope`` against the expected core and
   assignment identities with a caller-supplied replay-retaining nonce cache —
   stale, replayed, malformed, wrong-core, wrong-tick, wrong-view, and
   unauthenticated submissions fail closed;
2. loads the returned parameter bytes into the real model
   (``training.complete_field_64d.CompleteField64D``) and RECOMPUTES the
   forward pass over the exact attended D64 rail of the issued training view,
   reproducing the typed objective contract — teacher-forced decode of the
   assignment target text with ``sequence_cross_entropy``.  A claimed loss
   that disagrees with the recomputed loss beyond the documented tolerance is
   rejected outright: a worker that lies about loss is rejected, full stop;
3. requires exact next-character accuracy to exceed the constant-output
   floor (always predict the most frequent training token);
4. runs counterfactual probes (zeroed rail / swapped Soul state /
   irrelevant-rail substitution) and reports which claimed mechanisms the
   core actually uses.  Probe submissions carry the
   ``counterfactual_only`` outcome and are never an acceptance;
5. detects collapse — non-finite recomputed metrics, zero gradients, or
   parameter bytes that did not change on a learning assignment — and pauses
   the assignment instead of accepting;
6. feeds every outcome back through ``runtime.trainer.assignments``
   idempotently keyed commands, preserving every attempt; a trailing failed
   series escalates to supervisor inspection.  Timeouts never decide
   outcomes and competence is never fabricated.

Resume-continuity helpers assert, from the durable assignment journal, that
attempt indices are unique and contiguous and that learning attempts form an
exact parameter-generation chain.

Everything is deterministic: identities are canonical sha256 digests over
canonical JSON, the engine never draws randomness in an identity path, and
all clocks, roots, secrets, and nonce retention are caller-injected.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime.field import canonical_json_bytes, canonical_sha256
from runtime.field.schema import SharedFieldSnapshot
from runtime.field.training_view import TrainingAttentionView
from runtime.heart.rail_auth import RailAuthError, verify_envelope
from runtime.trainer.assignments import (
    AssignmentAttempt,
    AssignmentKind,
    AssignmentStatus,
    AssignmentStore,
    AssignmentStoreError,
    AttemptOutcome,
    TrainingAssignment,
)
from training.complete_field_64d import (
    CompleteField64D,
    ReaderConfig,
    sequence_cross_entropy,
    teacher_char_accuracy,
)

WORKER_EVIDENCE_SCHEMA = "axon-trainer-worker-evidence-v1"
GATE_VERDICT_SCHEMA = "axon-trainer-gate-verdict-v1"
MECHANISM_PROBE_SCHEMA = "axon-trainer-mechanism-probe-v1"

# Documented tolerances.  The recomputation runs the same deterministic
# float32 forward graph the worker reports from; on an identical device the
# values agree bitwise, across devices float32 drift is bounded well under
# these tolerances.  Anything looser would let a lying worker through.
LOSS_RECOMPUTATION_TOLERANCE = 1e-3
GRAD_RECOMPUTATION_TOLERANCE = 1e-3
ACCURACY_RECOMPUTATION_TOLERANCE = 1e-9
# A counterfactual probe counts as "used" when it moves the typed output
# logits by more than this deterministic epsilon.
PROBE_DELTA_TOLERANCE = 1e-6

OBJECTIVE_CONTRACT = "teacher_forced_decode:sequence_cross_entropy:head0"

# Counterfactual probe kinds the engine can run locally against the real
# model and the issued view.
PROBE_RAIL = "rail"            # zeroed rail: does the output use the rail at all?
PROBE_RAIL_CONTENT = "rail_content"  # irrelevant-rail substitution: does it use THIS rail?
PROBE_SOUL = "soul"            # swapped (zeroed) recurrent Soul state
SUPPORTED_PROBES = (PROBE_RAIL, PROBE_RAIL_CONTENT, PROBE_SOUL)


class GateDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PAUSED = "paused"
    ESCALATED = "escalated"


class SubmissionKind(str, Enum):
    LEARNING_RESULT = "learning_result"
    COUNTERFACTUAL_PROBE = "counterfactual_probe"


class GateEvaluationError(RuntimeError):
    """A gate path failed closed in a way that cannot produce a verdict."""


class ResumeContinuityError(GateEvaluationError):
    """The durable attempt history does not resume exactly after restart."""


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not (-float("inf") < value < float("inf")):
        raise ValueError(f"{label} must be finite")
    return value


def _optional_finite(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_safe_number(value: float) -> float | str:
    """JSON cannot encode nan/inf; collapse evidence must still be durable."""

    value = float(value)
    return value if math.isfinite(value) else str(value)


def constant_output_floor(target_tokens: list[int]) -> float:
    """Accuracy floor of the constant-output baseline: always predict the
    single most frequent training token at every position."""

    if not target_tokens:
        raise ValueError("constant-output floor requires at least one target token")
    counts: dict[int, int] = {}
    for token in target_tokens:
        counts[int(token)] = counts.get(int(token), 0) + 1
    return max(counts.values()) / len(target_tokens)


@dataclass(frozen=True, slots=True)
class MechanismProbeReport:
    """Outcome of one counterfactual probe against one claimed mechanism."""

    name: str
    claimed: bool
    used: bool
    max_abs_delta: float
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required(self.name, "probe name"))
        object.__setattr__(self, "claimed", bool(self.claimed))
        object.__setattr__(self, "used", bool(self.used))
        object.__setattr__(self, "max_abs_delta", _finite(self.max_abs_delta, "max_abs_delta"))
        if self.note is not None:
            object.__setattr__(self, "note", str(self.note))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": MECHANISM_PROBE_SCHEMA,
            "name": self.name,
            "claimed": self.claimed,
            "used": self.used,
            "max_abs_delta": self.max_abs_delta,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RecomputedMetrics:
    """Metrics the gate engine computed locally; these alone are authority."""

    loss: float
    grad_norm: float | None
    accuracy: float
    output_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss", float(self.loss))
        object.__setattr__(self, "accuracy", float(self.accuracy))
        if self.grad_norm is not None:
            object.__setattr__(self, "grad_norm", float(self.grad_norm))
        object.__setattr__(self, "output_sha256", _required(self.output_sha256, "output_sha256"))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "loss": _json_safe_number(self.loss),
            "grad_norm": None if self.grad_norm is None else _json_safe_number(self.grad_norm),
            "accuracy": _json_safe_number(self.accuracy),
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True, slots=True)
class WorkerEvidenceBundle:
    """One typed worker submission, sealed by ``runtime.heart.rail_auth``.

    The worker returns exact parameter bytes (a ``torch.save`` state dict of
    the real model) plus its claimed metrics; every claim is evidence the
    gate independently recomputes.  ``envelope`` must seal exactly
    ``sealed_payload_bytes`` for ``core_id`` and ``assignment_id``.
    """

    core_id: str
    assignment_id: str
    view_id: str
    field_id: str
    tick_id: int
    submission_kind: SubmissionKind | str
    parameter_bytes: bytes
    claimed_loss: float | None
    claimed_grad_norm: float | None
    claimed_accuracy: float | None
    prior_parameter_sha256: str | None = None
    optimizer_generation: str | None = None
    counterfactual_claims: tuple[str, ...] = ()
    emission_ref: Mapping[str, Any] | None = None
    soul_lineage: Mapping[str, Any] | None = None
    soul_payload: bytes = b""
    soul_media_type: str | None = None
    soul_tensor_layout: str | None = None
    envelope: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _required(self.core_id, "core_id"))
        object.__setattr__(self, "assignment_id", _required(self.assignment_id, "assignment_id"))
        object.__setattr__(self, "view_id", _required(self.view_id, "view_id"))
        object.__setattr__(self, "field_id", _required(self.field_id, "field_id"))
        if isinstance(self.tick_id, bool) or not isinstance(self.tick_id, int) or self.tick_id < 0:
            raise ValueError("tick_id must be a non-negative integer")
        kind = (
            self.submission_kind
            if isinstance(self.submission_kind, SubmissionKind)
            else SubmissionKind(self.submission_kind)
        )
        object.__setattr__(self, "submission_kind", kind)
        if not isinstance(self.parameter_bytes, bytes) or not self.parameter_bytes:
            raise ValueError("parameter_bytes must be non-empty bytes")
        object.__setattr__(self, "claimed_loss", _optional_finite(self.claimed_loss, "claimed_loss"))
        object.__setattr__(self, "claimed_grad_norm", _optional_finite(self.claimed_grad_norm, "claimed_grad_norm"))
        object.__setattr__(self, "claimed_accuracy", _optional_finite(self.claimed_accuracy, "claimed_accuracy"))
        if self.prior_parameter_sha256 is not None:
            object.__setattr__(
                self,
                "prior_parameter_sha256",
                _required(self.prior_parameter_sha256, "prior_parameter_sha256"),
            )
        if self.optimizer_generation is not None:
            object.__setattr__(
                self, "optimizer_generation", _required(self.optimizer_generation, "optimizer_generation")
            )
        claims = tuple(sorted({_required(str(item), "counterfactual claim") for item in self.counterfactual_claims}))
        object.__setattr__(self, "counterfactual_claims", claims)
        if kind is SubmissionKind.COUNTERFACTUAL_PROBE and not claims:
            raise ValueError("counterfactual probe submissions require at least one claim")
        if kind is SubmissionKind.LEARNING_RESULT and (
            self.claimed_loss is None or self.claimed_accuracy is None
        ):
            raise ValueError("learning-result submissions require claimed_loss and claimed_accuracy")
        for label in ("emission_ref", "soul_lineage"):
            value = getattr(self, label)
            if value is not None:
                copied = dict(value)
                if not copied:
                    raise ValueError(f"{label} must be a non-empty mapping")
                canonical_json_bytes(copied)  # fail closed on non-JSON-safe content
                object.__setattr__(self, label, copied)
        if not isinstance(self.soul_payload, bytes):
            raise TypeError("soul_payload must be bytes")
        if self.soul_payload:
            object.__setattr__(
                self,
                "soul_media_type",
                _required(self.soul_media_type or "", "soul_media_type"),
            )
            object.__setattr__(
                self,
                "soul_tensor_layout",
                _required(self.soul_tensor_layout or "", "soul_tensor_layout"),
            )
        elif self.soul_media_type is not None or self.soul_tensor_layout is not None:
            raise ValueError("empty soul_payload may not carry Soul codec metadata")
        if self.envelope is not None and not isinstance(self.envelope, Mapping):
            raise TypeError("envelope must be the sealed rail-auth envelope mapping")

    @property
    def parameter_sha256(self) -> str:
        return _sha256_hex(self.parameter_bytes)

    @property
    def soul_payload_sha256(self) -> str:
        return _sha256_hex(self.soul_payload)

    def sealed_payload_dict(self) -> dict[str, Any]:
        """The exact typed payload the envelope binds and authenticates."""

        return {
            "schema": WORKER_EVIDENCE_SCHEMA,
            "core_id": self.core_id,
            "assignment_id": self.assignment_id,
            "view_id": self.view_id,
            "field_id": self.field_id,
            "tick_id": self.tick_id,
            "submission_kind": self.submission_kind.value,
            "parameter_bytes_sha256": self.parameter_sha256,
            "soul_payload_sha256": self.soul_payload_sha256,
            "soul_media_type": self.soul_media_type,
            "soul_tensor_layout": self.soul_tensor_layout,
            "prior_parameter_sha256": self.prior_parameter_sha256,
            "optimizer_generation": self.optimizer_generation,
            "claimed_loss": self.claimed_loss,
            "claimed_grad_norm": self.claimed_grad_norm,
            "claimed_accuracy": self.claimed_accuracy,
            "counterfactual_claims": list(self.counterfactual_claims),
            "emission_ref_sha256": (
                None if self.emission_ref is None else canonical_sha256(self.emission_ref)
            ),
            "soul_lineage_sha256": (
                None if self.soul_lineage is None else canonical_sha256(self.soul_lineage)
            ),
        }

    @property
    def sealed_payload_bytes(self) -> bytes:
        return canonical_json_bytes(self.sealed_payload_dict())

    @property
    def bundle_sha256(self) -> str:
        return _sha256_hex(self.sealed_payload_bytes)


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The deterministic supervisory decision for one evaluated bundle."""

    decision: GateDecision
    outcome: AttemptOutcome
    reasons: tuple[str, ...]
    assignment_id: str
    bundle_sha256: str
    attempt_id: str | None = None
    claimed_loss: float | None = None
    claimed_grad_norm: float | None = None
    claimed_accuracy: float | None = None
    recomputed: RecomputedMetrics | None = None
    accuracy_floor: float | None = None
    parameters_changed: bool | None = None
    mechanism_reports: tuple[MechanismProbeReport, ...] = ()
    verdict_id: str = field(init=False)

    def __post_init__(self) -> None:
        decision = self.decision if isinstance(self.decision, GateDecision) else GateDecision(self.decision)
        object.__setattr__(self, "decision", decision)
        outcome = self.outcome if isinstance(self.outcome, AttemptOutcome) else AttemptOutcome(self.outcome)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "assignment_id", _required(self.assignment_id, "assignment_id"))
        object.__setattr__(self, "bundle_sha256", _required(self.bundle_sha256, "bundle_sha256"))
        reasons = tuple(sorted({_required(str(item), "reason") for item in self.reasons}))
        object.__setattr__(self, "reasons", reasons)
        if self.attempt_id is not None:
            object.__setattr__(self, "attempt_id", _required(self.attempt_id, "attempt_id"))
        for label in ("claimed_loss", "claimed_grad_norm", "claimed_accuracy", "accuracy_floor"):
            object.__setattr__(self, label, _optional_finite(getattr(self, label), label))
        if self.parameters_changed is not None:
            object.__setattr__(self, "parameters_changed", bool(self.parameters_changed))
        reports = tuple(self.mechanism_reports)
        if not all(isinstance(item, MechanismProbeReport) for item in reports):
            raise TypeError("mechanism_reports must contain MechanismProbeReport values")
        object.__setattr__(self, "mechanism_reports", reports)
        if self.recomputed is not None and not isinstance(self.recomputed, RecomputedMetrics):
            raise TypeError("recomputed must be RecomputedMetrics or None")
        object.__setattr__(self, "verdict_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": GATE_VERDICT_SCHEMA,
            "decision": self.decision.value,
            "outcome": self.outcome.value,
            "reasons": list(self.reasons),
            "assignment_id": self.assignment_id,
            "bundle_sha256": self.bundle_sha256,
            "attempt_id": self.attempt_id,
            "claimed_loss": self.claimed_loss,
            "claimed_grad_norm": self.claimed_grad_norm,
            "claimed_accuracy": self.claimed_accuracy,
            "recomputed": None if self.recomputed is None else self.recomputed.to_canonical_dict(),
            "accuracy_floor": self.accuracy_floor,
            "parameters_changed": self.parameters_changed,
            "mechanism_reports": [item.to_canonical_dict() for item in self.mechanism_reports],
        }
        if include_id:
            value["verdict_id"] = self.verdict_id
        return value


class MemoryNonceCache:
    """Replay-retention hook for ``rail_auth.verify_envelope``.

    Returns ``True`` exactly when the nonce is fresh and has been recorded;
    any repeat returns ``False`` and fails the envelope closed.  When a state
    root is injected, every recorded nonce is durably appended (fsync) so
    replay protection survives a local restart; the cache is reloaded from
    that file on construction.
    """

    def __init__(self, state_root: Path | str | None = None) -> None:
        self._seen: set[str] = set()
        self._path: Path | None = None
        if state_root is not None:
            self._path = Path(state_root).resolve(strict=False) / "training" / "gate_nonces.jsonl"
            if self._path.is_file():
                for line in self._path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        self._seen.add(line)

    def record(self, nonce: str) -> bool:
        nonce = _required(nonce, "nonce")
        if nonce in self._seen:
            return False
        self._seen.add(nonce)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(nonce + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return True

    def __call__(self, nonce: str) -> bool:
        return self.record(nonce)


class GateEngine:
    """Independent trainer-side evaluation and supervisory gate engine.

    The engine is stateless apart from its injected configuration; every
    durable effect goes through the caller-supplied ``AssignmentStore``.
    """

    def __init__(
        self,
        *,
        model_config: ReaderConfig | None = None,
        environ: Mapping[str, str] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._model_config = model_config if model_config is not None else ReaderConfig()
        self._environ = environ
        self._now = now if now is not None else time.time

    # ------------------------------------------------------------------ entry

    def evaluate(
        self,
        bundle: WorkerEvidenceBundle,
        assignment: TrainingAssignment,
        view: TrainingAttentionView,
        *,
        target_text: str,
        nonce_cache: Callable[[str], bool],
        store: AssignmentStore | None = None,
        snapshot: SharedFieldSnapshot | None = None,
        probe_snapshot: SharedFieldSnapshot | None = None,
        now: float | None = None,
    ) -> GateVerdict:
        """Evaluate one sealed worker submission and return the verdict.

        ``target_text`` is the typed objective text of the assignment (the
        exact next-character training sequence).  ``snapshot`` is the current
        canonical source snapshot; when supplied, the issued view must prove
        fresh against it.  ``probe_snapshot`` supplies unrelated canonical
        content for the irrelevant-rail counterfactual.  ``nonce_cache`` is
        MANDATORY: replay protection can never be disabled at this boundary,
        so an absent cache fails the submission closed before any other
        check.  Outcomes are fed back through ``store`` with deterministic
        idempotency keys; without a store the engine evaluates only and
        records nothing.
        """

        if not isinstance(bundle, WorkerEvidenceBundle):
            raise TypeError("bundle must be a WorkerEvidenceBundle")
        if not isinstance(assignment, TrainingAssignment):
            raise TypeError("assignment must be a TrainingAssignment")
        if not isinstance(view, TrainingAttentionView):
            raise TypeError("view must be a TrainingAttentionView")
        if not isinstance(target_text, str) or not target_text:
            raise ValueError("target_text must be a non-empty string")
        if nonce_cache is None or not callable(nonce_cache):
            # Fail closed: replay protection is never caller-optional here.
            # The verdict carries the gate's canonical identity so the
            # rejection is durable evidence of the policy violation.
            return self._finalize(
                bundle,
                assignment,
                store,
                decision=GateDecision.REJECTED,
                outcome=AttemptOutcome.FAIL,
                reasons=(
                    "unauthenticated: a durable nonce cache is mandatory; "
                    "replay protection cannot be disabled at the gate boundary",
                ),
                observed_at=float(now if now is not None else self._now()),
            )
        observed_at = float(now if now is not None else self._now())

        binding_error = self._check_identity_binding(bundle, assignment, view, snapshot)
        if binding_error is not None:
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.REJECTED,
                outcome=AttemptOutcome.FAIL,
                reasons=(binding_error,),
                observed_at=observed_at,
            )

        auth_error = self._authenticate(bundle, nonce_cache, observed_at)
        if auth_error is not None:
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.REJECTED,
                outcome=AttemptOutcome.FAIL,
                reasons=(auth_error,),
                observed_at=observed_at,
            )

        model = self._load_model(bundle)
        if isinstance(model, str):
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.REJECTED,
                outcome=AttemptOutcome.FAIL,
                reasons=(model,),
                observed_at=observed_at,
            )

        soul_state = self._load_soul_state(bundle, model)
        if isinstance(soul_state, str):
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.REJECTED,
                outcome=AttemptOutcome.FAIL,
                reasons=(soul_state,),
                observed_at=observed_at,
            )

        if bundle.submission_kind is SubmissionKind.COUNTERFACTUAL_PROBE:
            return self._evaluate_probes(
                bundle, assignment, view, model, soul_state, target_text,
                store, probe_snapshot, observed_at,
            )
        return self._evaluate_result(
            bundle, assignment, view, model, soul_state, target_text, store, observed_at
        )

    # ------------------------------------------------------------- validation

    def _check_identity_binding(
        self,
        bundle: WorkerEvidenceBundle,
        assignment: TrainingAssignment,
        view: TrainingAttentionView,
        snapshot: SharedFieldSnapshot | None,
    ) -> str | None:
        """Fail-closed binding of the bundle to the assignment and the view."""

        if bundle.assignment_id != assignment.assignment_id:
            return "bundle belongs to a different assignment"
        if assignment.lease is not None and bundle.core_id != assignment.lease.holder_core_id:
            return "wrong-core: submitter does not hold the assignment lease"
        core_ids = assignment.cohort_eligibility.get("core_ids")
        if core_ids is not None and bundle.core_id not in core_ids:
            return "wrong-core: submitter is outside the assignment cohort"
        if bundle.field_id != view.source_field_id:
            return "wrong-field: bundle field identity does not match the issued view"
        if bundle.tick_id != view.source_tick_id:
            return "wrong-tick: bundle tick does not match the issued view"
        if bundle.view_id != view.view_id:
            return "wrong-view: bundle view identity does not match the issued view"
        if snapshot is not None:
            try:
                view.verify(snapshot)
            except Exception as exc:
                return f"stale-view: the issued view does not verify against the canonical snapshot: {exc}"
        return None

    def _authenticate(
        self,
        bundle: WorkerEvidenceBundle,
        nonce_cache: Callable[[str], bool] | None,
        observed_at: float,
    ) -> str | None:
        if bundle.envelope is None:
            return "unauthenticated: bundle carries no sealed envelope"
        try:
            verify_envelope(
                bundle.envelope,
                payload=bundle.sealed_payload_bytes,
                expected_core_identity=bundle.core_id,
                expected_assignment_id=bundle.assignment_id,
                nonce_cache=nonce_cache,
                now=observed_at,
                environ=self._environ,
            )
        except RailAuthError as exc:
            return f"unauthenticated: {exc}"
        return None

    def _load_model(self, bundle: WorkerEvidenceBundle) -> CompleteField64D | str:
        """Strictly load the returned parameters into the real model."""

        try:
            state_dict = torch.load(io.BytesIO(bundle.parameter_bytes), map_location="cpu")
        except Exception as exc:
            return f"malformed: returned parameter bytes are not a loadable state dict: {exc}"
        if not isinstance(state_dict, dict):
            return "malformed: returned parameters are not a state dict"
        model = CompleteField64D(self._model_config).eval()
        try:
            model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, KeyError) as exc:
            return f"malformed: returned parameters do not match the bound architecture: {exc}"
        return model

    @staticmethod
    def _load_soul_state(
        bundle: WorkerEvidenceBundle,
        model: CompleteField64D,
    ) -> torch.Tensor | str | None:
        """Decode the exact authenticated Soul payload used by the worker.

        Empty payload is the canonical generation-zero Soul.  Nonempty state
        must use the local conformance worker's explicit codec; architecture
        adapters will provide their own decoder when the living core replaces
        this conformance model.
        """

        if not bundle.soul_payload:
            return None
        if (
            bundle.soul_media_type != "application/x-axon-local-worker-soul-hot"
            or bundle.soul_tensor_layout != "reader-state-float32"
        ):
            return "malformed: submitted Soul codec does not match the bound worker architecture"
        try:
            # Local import avoids a module-load cycle: local_worker imports the
            # assignment types but the codec itself is transport independent.
            from runtime.trainer.local_worker import decode_hot_payload

            array, _metrics = decode_hot_payload(bundle.soul_payload)
        except Exception as exc:
            return f"malformed: submitted Soul payload cannot be decoded: {exc}"
        state = torch.from_numpy(array)
        expected = tuple(model.initial_state.shape)
        if tuple(state.shape) == (1, *expected):
            state = state.squeeze(0)
        if tuple(state.shape) != expected:
            return (
                "malformed: submitted Soul state shape "
                f"{tuple(state.shape)} does not match {expected}"
            )
        if not torch.isfinite(state).all():
            return "malformed: submitted Soul state contains non-finite values"
        return state.to(device=model.device, dtype=model.initial_state.dtype)

    # ------------------------------------------------------ recompute helpers

    def _forward_metrics(
        self,
        model: CompleteField64D,
        view: TrainingAttentionView,
        target_text: str,
        *,
        with_grad: bool,
        compiled_override: Any = None,
        soul_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float | None]:
        """Run the typed objective contract and return logits, targets, grad norm."""

        model.zero_grad(set_to_none=True)
        compiled = view.compiled if compiled_override is None else compiled_override
        reader_state, memory, _manifest = model.read_compiled_with_memory(
            compiled,
            initial_state=soul_override,
        )
        logits, targets = model.decode_teacher(reader_state, target_text, head=0, memory=memory)
        grad_norm: float | None = None
        if with_grad:
            loss = sequence_cross_entropy(logits, targets)
            loss.backward()
            total = torch.zeros((), dtype=torch.float64)
            for param in model.parameters():
                if param.grad is None:
                    continue
                total = total + param.grad.detach().to(torch.float64).pow(2).sum()
            grad_norm = math.sqrt(float(total.item()))
        return logits.detach(), targets.detach(), grad_norm

    @staticmethod
    def _output_sha256(logits: torch.Tensor, targets: torch.Tensor) -> str:
        digest = hashlib.sha256()
        digest.update(logits.detach().to(device="cpu", dtype=torch.float64).numpy().tobytes())
        digest.update(targets.detach().to(device="cpu").numpy().tobytes())
        return digest.hexdigest()

    # ------------------------------------------------------------ result gate

    def _evaluate_result(
        self,
        bundle: WorkerEvidenceBundle,
        assignment: TrainingAssignment,
        view: TrainingAttentionView,
        model: CompleteField64D,
        soul_state: torch.Tensor | None,
        target_text: str,
        store: AssignmentStore | None,
        observed_at: float,
    ) -> GateVerdict:
        logits, targets, grad_norm = self._forward_metrics(
            model, view, target_text, with_grad=True, soul_override=soul_state
        )
        recomputed_loss = float(sequence_cross_entropy(logits, targets).item())
        recomputed_accuracy = float(teacher_char_accuracy(logits, targets))
        floor = constant_output_floor([int(token) for token in targets.flatten().tolist()])
        recomputed = RecomputedMetrics(
            loss=recomputed_loss,
            grad_norm=grad_norm,
            accuracy=recomputed_accuracy,
            output_sha256=self._output_sha256(logits, targets),
        )

        # Collapse detection on the recomputed (authoritative) metrics first:
        # a collapsed core pauses; it is never accepted and never silently
        # fails either.
        if not math.isfinite(recomputed_loss) or not math.isfinite(recomputed_accuracy):
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.PAUSED,
                outcome=AttemptOutcome.FAIL,
                reasons=("collapse: recomputed metrics are non-finite",),
                observed_at=observed_at,
                recomputed=recomputed,
                accuracy_floor=floor,
            )
        if grad_norm is not None and (not math.isfinite(grad_norm) or grad_norm <= 0.0):
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.PAUSED,
                outcome=AttemptOutcome.FAIL,
                reasons=("collapse: recomputed gradient norm is zero or non-finite",),
                observed_at=observed_at,
                recomputed=recomputed,
                accuracy_floor=floor,
            )

        # Evidence-tampering detection: any claimed metric that disagrees
        # with recomputation beyond tolerance is a lying worker, full stop.
        tampered: list[str] = []
        if bundle.claimed_loss is not None and abs(bundle.claimed_loss - recomputed_loss) > LOSS_RECOMPUTATION_TOLERANCE:
            tampered.append(
                f"claimed loss {bundle.claimed_loss:.6f} disagrees with recomputed loss "
                f"{recomputed_loss:.6f} beyond tolerance {LOSS_RECOMPUTATION_TOLERANCE}"
            )
        if (
            bundle.claimed_accuracy is not None
            and abs(bundle.claimed_accuracy - recomputed_accuracy) > ACCURACY_RECOMPUTATION_TOLERANCE
        ):
            tampered.append(
                f"claimed accuracy {bundle.claimed_accuracy:.6f} disagrees with recomputed "
                f"accuracy {recomputed_accuracy:.6f}"
            )
        if (
            bundle.claimed_grad_norm is not None
            and grad_norm is not None
            and abs(bundle.claimed_grad_norm - grad_norm) > GRAD_RECOMPUTATION_TOLERANCE
        ):
            tampered.append(
                f"claimed gradient norm {bundle.claimed_grad_norm:.6f} disagrees with recomputed "
                f"gradient norm {grad_norm:.6f} beyond tolerance {GRAD_RECOMPUTATION_TOLERANCE}"
            )
        if tampered:
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.REJECTED,
                outcome=AttemptOutcome.FAIL,
                reasons=tuple(tampered),
                observed_at=observed_at,
                recomputed=recomputed,
                accuracy_floor=floor,
            )

        parameters_changed: bool | None = None
        if assignment.kind is AssignmentKind.LEARNING:
            if bundle.prior_parameter_sha256 is None:
                return self._finalize(
                    bundle, assignment, store,
                    decision=GateDecision.REJECTED,
                    outcome=AttemptOutcome.FAIL,
                    reasons=("malformed: learning submission carries no prior parameter identity",),
                    observed_at=observed_at,
                    recomputed=recomputed,
                    accuracy_floor=floor,
                )
            parameters_changed = bundle.parameter_sha256 != bundle.prior_parameter_sha256
            if not parameters_changed:
                return self._finalize(
                    bundle, assignment, store,
                    decision=GateDecision.PAUSED,
                    outcome=AttemptOutcome.FAIL,
                    reasons=("collapse: parameter bytes did not change on a learning assignment",),
                    observed_at=observed_at,
                    recomputed=recomputed,
                    accuracy_floor=floor,
                    parameters_changed=False,
                )

        if recomputed_accuracy <= floor:
            return self._finalize(
                bundle, assignment, store,
                decision=GateDecision.REJECTED,
                outcome=AttemptOutcome.FAIL,
                reasons=(
                    "below-floor: exact next-character accuracy "
                    f"{recomputed_accuracy:.6f} does not exceed the constant-output floor {floor:.6f}",
                ),
                observed_at=observed_at,
                recomputed=recomputed,
                accuracy_floor=floor,
                parameters_changed=parameters_changed,
            )

        return self._finalize(
            bundle, assignment, store,
            decision=GateDecision.ACCEPTED,
            outcome=AttemptOutcome.PASS,
            reasons=("verified: loss, gradients, parameter change, and accuracy recomputed and within gates",),
            observed_at=observed_at,
            recomputed=recomputed,
            accuracy_floor=floor,
            parameters_changed=parameters_changed,
        )

    # -------------------------------------------------------- probe-only gate

    def _evaluate_probes(
        self,
        bundle: WorkerEvidenceBundle,
        assignment: TrainingAssignment,
        view: TrainingAttentionView,
        model: CompleteField64D,
        soul_state: torch.Tensor | None,
        target_text: str,
        store: AssignmentStore | None,
        probe_snapshot: SharedFieldSnapshot | None,
        observed_at: float,
    ) -> GateVerdict:
        baseline_logits, _targets, _grad = self._forward_metrics(
            model, view, target_text, with_grad=False, soul_override=soul_state
        )
        baseline = baseline_logits.to(torch.float64)
        reports: list[MechanismProbeReport] = []
        for name in bundle.counterfactual_claims:
            reports.append(
                self._run_probe(
                    name, model, view, soul_state, target_text, baseline, probe_snapshot
                )
            )
        return self._finalize(
            bundle, assignment, store,
            decision=GateDecision.REJECTED,
            outcome=AttemptOutcome.COUNTERFACTUAL_ONLY,
            reasons=("counterfactual_only: probe evidence is never an acceptance",),
            observed_at=observed_at,
            mechanism_reports=tuple(reports),
        )

    def _run_probe(
        self,
        name: str,
        model: CompleteField64D,
        view: TrainingAttentionView,
        soul_state: torch.Tensor | None,
        target_text: str,
        baseline: torch.Tensor,
        probe_snapshot: SharedFieldSnapshot | None,
    ) -> MechanismProbeReport:
        if name == PROBE_RAIL:
            zeroed = replace(view.compiled, rows=np.zeros_like(view.compiled.rows))
            logits, _targets, _grad = self._forward_metrics(
                model,
                view,
                target_text,
                with_grad=False,
                compiled_override=zeroed,
                soul_override=soul_state,
            )
            delta = float((logits.to(torch.float64) - baseline).abs().max().item())
            return MechanismProbeReport(
                name=name,
                claimed=True,
                used=delta > PROBE_DELTA_TOLERANCE,
                max_abs_delta=delta,
                note="zeroed rail rows",
            )
        if name == PROBE_RAIL_CONTENT:
            if probe_snapshot is None:
                return MechanismProbeReport(
                    name=name,
                    claimed=True,
                    used=False,
                    max_abs_delta=0.0,
                    note="inconclusive: no unrelated probe snapshot supplied",
                )
            from runtime.field import D64FieldCompiler

            other = D64FieldCompiler().compile(probe_snapshot)
            rows = other.rows
            if other.row_count < view.compiled.row_count:
                repeats = math.ceil(view.compiled.row_count / max(1, other.row_count))
                rows = np.tile(rows, (repeats, 1))
            grafted = replace(view.compiled, rows=np.ascontiguousarray(rows[: view.compiled.row_count]))
            logits, _targets, _grad = self._forward_metrics(
                model,
                view,
                target_text,
                with_grad=False,
                compiled_override=grafted,
                soul_override=soul_state,
            )
            delta = float((logits.to(torch.float64) - baseline).abs().max().item())
            return MechanismProbeReport(
                name=name,
                claimed=True,
                used=delta > PROBE_DELTA_TOLERANCE,
                max_abs_delta=delta,
                note="irrelevant-rail substitution",
            )
        if name == PROBE_SOUL:
            if soul_state is None:
                return MechanismProbeReport(
                    name=name,
                    claimed=True,
                    used=False,
                    max_abs_delta=0.0,
                    note="inconclusive: the authenticated incoming Soul is generation-zero empty",
                )
            soul = torch.zeros_like(soul_state)
            logits, _targets, _grad = self._forward_metrics(
                model, view, target_text, with_grad=False, soul_override=soul
            )
            delta = float((logits.to(torch.float64) - baseline).abs().max().item())
            return MechanismProbeReport(
                name=name,
                claimed=True,
                used=delta > PROBE_DELTA_TOLERANCE,
                max_abs_delta=delta,
                note="swapped (zeroed) recurrent Soul state",
            )
        return MechanismProbeReport(
            name=name,
            claimed=True,
            used=False,
            max_abs_delta=0.0,
            note=f"unsupported probe kind; supported: {SUPPORTED_PROBES!r}",
        )

    # ------------------------------------------------------ store integration

    def _finalize(
        self,
        bundle: WorkerEvidenceBundle,
        assignment: TrainingAssignment,
        store: AssignmentStore | None,
        *,
        decision: GateDecision,
        outcome: AttemptOutcome,
        reasons: tuple[str, ...],
        observed_at: float,
        recomputed: RecomputedMetrics | None = None,
        accuracy_floor: float | None = None,
        parameters_changed: bool | None = None,
        mechanism_reports: tuple[MechanismProbeReport, ...] = (),
    ) -> GateVerdict:
        attempt_id: str | None = None
        final_decision = decision
        final_reasons = list(reasons)
        if store is not None:
            attempt_id, decision_after_store = self._commit_outcome(
                bundle,
                assignment,
                store,
                decision=decision,
                outcome=outcome,
                reasons=tuple(final_reasons),
                observed_at=observed_at,
                recomputed=recomputed,
            )
            if decision_after_store is not None:
                final_decision = decision_after_store
        return GateVerdict(
            decision=final_decision,
            outcome=outcome,
            reasons=tuple(final_reasons),
            assignment_id=assignment.assignment_id,
            bundle_sha256=bundle.bundle_sha256,
            attempt_id=attempt_id,
            claimed_loss=bundle.claimed_loss,
            claimed_grad_norm=bundle.claimed_grad_norm,
            claimed_accuracy=bundle.claimed_accuracy,
            recomputed=recomputed,
            accuracy_floor=accuracy_floor,
            parameters_changed=parameters_changed,
            mechanism_reports=mechanism_reports,
        )

    def _commit_outcome(
        self,
        bundle: WorkerEvidenceBundle,
        assignment: TrainingAssignment,
        store: AssignmentStore,
        *,
        decision: GateDecision,
        outcome: AttemptOutcome,
        reasons: tuple[str, ...],
        observed_at: float,
        recomputed: RecomputedMetrics | None,
    ) -> tuple[str | None, GateDecision | None]:
        """Record the attempt and outcome idempotently; escalate failed series.

        Returns the committed attempt id and a decision override when the
        store escalated the assignment.  Raises ``GateEvaluationError`` when
        an acceptance could not be made durable — an accepted outcome must be
        committed exactly once or not at all.
        """

        try:
            current = store.get_assignment(assignment.assignment_id)
            attempt = store.record_attempt(
                current.assignment_id,
                core_id=bundle.core_id,
                emission_ref=(
                    {"emission": "not_submitted"}
                    if bundle.emission_ref is None
                    else dict(bundle.emission_ref)
                ),
                soul_lineage=(
                    {"soul_lineage": "not_submitted", "verified": False}
                    if bundle.soul_lineage is None
                    else {**dict(bundle.soul_lineage), "verified": False}
                ),
                parameter_generation=bundle.parameter_sha256,
                optimizer_generation=bundle.optimizer_generation or "not_submitted",
                evidence={
                    "bundle_sha256": bundle.bundle_sha256,
                    "parameter_sha256": bundle.parameter_sha256,
                    "prior_parameter_sha256": bundle.prior_parameter_sha256,
                    "decision": decision.value,
                    "reasons": list(reasons),
                },
                expected_revision=current.revision,
                idempotency_key=f"gate-attempt:{bundle.bundle_sha256}",
            )
            outcome_evidence = {
                "bundle_sha256": bundle.bundle_sha256,
                "parameter_sha256": bundle.parameter_sha256,
                "prior_parameter_sha256": bundle.prior_parameter_sha256,
                "decision": decision.value,
                "reasons": list(reasons),
                "observed_at": observed_at,
            }
            if recomputed is not None:
                outcome_evidence["recomputed"] = recomputed.to_canonical_dict()
            fresh = store.get_assignment(attempt.assignment_id)
            _attempt, updated = store.record_attempt_outcome(
                attempt.attempt_id,
                outcome=outcome,
                evidence=outcome_evidence,
                expected_revision=fresh.revision,
                idempotency_key=f"gate-outcome:{bundle.bundle_sha256}",
            )
            if updated.status is AssignmentStatus.ESCALATED:
                return attempt.attempt_id, GateDecision.ESCALATED
            if decision is GateDecision.PAUSED and updated.status is AssignmentStatus.ACTIVE:
                paused = store.pause(
                    updated.assignment_id,
                    holder_core_id=bundle.core_id,
                    expected_revision=updated.revision,
                    idempotency_key=f"gate-pause:{bundle.bundle_sha256}",
                    reason="; ".join(reasons),
                )
                if paused.status is not AssignmentStatus.PAUSED:
                    raise GateEvaluationError("collapse pause did not take hold")
            return attempt.attempt_id, None
        except AssignmentStoreError as exc:
            if decision is GateDecision.ACCEPTED:
                raise GateEvaluationError(
                    f"accepted outcome could not be committed exactly once: {exc}"
                ) from exc
            return None, None


# ----------------------------------------------------------- resume continuity


def _sorted_attempts(attempts: tuple[AssignmentAttempt, ...] | list[AssignmentAttempt]) -> tuple[AssignmentAttempt, ...]:
    rows = tuple(attempts)
    if not all(isinstance(item, AssignmentAttempt) for item in rows):
        raise TypeError("attempts must contain AssignmentAttempt values")
    return tuple(sorted(rows, key=lambda item: (item.attempt_index, item.attempt_id)))


def assert_no_duplicate_attempt_indices(
    attempts: tuple[AssignmentAttempt, ...] | list[AssignmentAttempt],
) -> None:
    """Assert the durable journal has unique, gap-free attempt indices.

    Called after restart/preemption before any resume: a duplicate or gapped
    index means history did not survive exactly and resume must fail closed.
    """

    rows = _sorted_attempts(attempts)
    indices = [item.attempt_index for item in rows]
    if indices != list(range(len(rows))):
        raise ResumeContinuityError(
            f"attempt indices are not the exact contiguous range 0..{len(rows) - 1}: {indices!r}"
        )


def assert_exact_generation_lineage(
    attempts: tuple[AssignmentAttempt, ...] | list[AssignmentAttempt],
    *,
    expected_initial_parameter_sha256: str | None = None,
) -> None:
    """Assert learning attempts chain through exact parameter generations.

    Every attempt recorded by ``GateEngine`` carries its parameter identity
    and the prior parameter identity in its durable evidence.  After a
    restart the chain must hold exactly: attempt ``i + 1`` must begin from
    the parameter bytes attempt ``i`` returned.  Any break, fork, or unknown
    identity fails closed.
    """

    rows = _sorted_attempts(attempts)
    if expected_initial_parameter_sha256 is not None:
        if not rows:
            raise ResumeContinuityError("expected an initial parameter generation but no attempts exist")
        first_prior = rows[0].evidence.get("prior_parameter_sha256")
        if first_prior != expected_initial_parameter_sha256:
            raise ResumeContinuityError(
                "initial attempt does not begin at the expected parameter generation"
            )
    for index, attempt in enumerate(rows):
        current = attempt.evidence.get("parameter_sha256")
        if not isinstance(current, str) or not current:
            raise ResumeContinuityError(
                f"attempt {attempt.attempt_index} carries no parameter identity evidence"
            )
        if attempt.parameter_generation != current:
            raise ResumeContinuityError(
                f"attempt {attempt.attempt_index} parameter generation field disagrees with its evidence"
            )
        if index + 1 < len(rows):
            successor_prior = rows[index + 1].evidence.get("prior_parameter_sha256")
            if successor_prior is None:
                raise ResumeContinuityError(
                    f"attempt {rows[index + 1].attempt_index} carries no prior parameter identity"
                )
            if successor_prior != current:
                raise ResumeContinuityError(
                    f"generation lineage breaks at attempt {rows[index + 1].attempt_index}: "
                    "successor does not begin from the predecessor's returned parameters"
                )


def assert_resume_continuity(
    attempts: tuple[AssignmentAttempt, ...] | list[AssignmentAttempt],
    *,
    expected_initial_parameter_sha256: str | None = None,
) -> None:
    """Assert exact resume continuity: unique indices and exact lineage."""

    assert_no_duplicate_attempt_indices(attempts)
    assert_exact_generation_lineage(attempts, expected_initial_parameter_sha256=expected_initial_parameter_sha256)


__all__ = [
    "ACCURACY_RECOMPUTATION_TOLERANCE",
    "GATE_VERDICT_SCHEMA",
    "GRAD_RECOMPUTATION_TOLERANCE",
    "LOSS_RECOMPUTATION_TOLERANCE",
    "MECHANISM_PROBE_SCHEMA",
    "OBJECTIVE_CONTRACT",
    "PROBE_DELTA_TOLERANCE",
    "PROBE_RAIL",
    "PROBE_RAIL_CONTENT",
    "PROBE_SOUL",
    "SUPPORTED_PROBES",
    "WORKER_EVIDENCE_SCHEMA",
    "GateDecision",
    "GateEngine",
    "GateEvaluationError",
    "GateVerdict",
    "MechanismProbeReport",
    "MemoryNonceCache",
    "RecomputedMetrics",
    "ResumeContinuityError",
    "SubmissionKind",
    "WorkerEvidenceBundle",
    "assert_exact_generation_lineage",
    "assert_no_duplicate_attempt_indices",
    "assert_resume_continuity",
    "constant_output_floor",
]
