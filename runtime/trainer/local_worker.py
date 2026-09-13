"""Local OFFLINE_TRAINING worker: real model execution behind the rail boundary.

This is the LOCAL worker required by handoff step 7.  It runs FIRST, in
process, behind exactly the same authenticated transport boundary intended for
the Kaggle worker (``runtime.heart.rail_auth`` seal/verify), but every
mechanism it exercises is real:

- it inhales the exact private Soul payload from ``SoulStore`` and fails
  closed unless the inhaled Soul identity matches the ``CoreBinding``; the
  exact hot-layer reader state is decoded and passed as the recurrent
  ``initial_state`` of the real forward, so the persisted Soul genuinely
  conditions the next computation;
- it runs the real D64 motor (``training.complete_field_64d.CompleteField64D``
  with ``ReaderConfig(d_model=64)``) over the attended D64 rows of one
  ``TrainingAttentionView`` — no fixtures, no host-generated prose;
- the typed objective contract implemented here is exact next-character
  prediction: the discrete categorical loss (teacher-forced NLL over the
  frozen bank plus EOS) is computed on the SAME autograd graph as the
  forward pass, so the returned gradients are real and remain attached to
  the intended parameters.  When the assignment carries an explicit expected
  target (``target_text``), the loss binds THAT text (``ABC? -> D``
  semantics: solve the task, do not reconstruct the instructions); otherwise
  the attended ``trainer_instructions`` text is the objective.  The typed
  rail emission is separately decoded free-running, without target-prefix
  leakage;
- a real backward pass runs on every attempt (grad norm is evidence); a real
  optimizer step runs only when the assignment kind is LEARNING;
- the resulting recurrent reader state is exhaled as a real Soul transition
  through the defined ``SoulStore`` mechanism (prepare + finalize + receipt),
  producing a new content-addressed Soul generation;
- metrics are wired into the assignment store's idempotently keyed
  ``record_attempt`` command as EVIDENCE ONLY.  The worker never records an
  outcome: pass/fail remains Heart's decision, and worker-reported numbers
  are never authority.

Every artifact carries canonical sha256 identity; nothing here uses the
network, HTTP, threads, or RNG in identity paths.  Behavior is deterministic
under a fixed seed plus a fixed torch manual seed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from runtime.field import LogicalRegion, canonical_json_bytes, canonical_sha256
from runtime.field.training_view import TrainingAttentionView
from runtime.heart.rail_auth import DEFAULT_TTL_SECONDS, seal_envelope
from runtime.heart.registry import CoreBinding, CoreStatus
from runtime.soul import SoulBranch, SoulLayer, SoulSnapshot, SoulTemperature, SoulTransition
from training.complete_field_64d import CompleteField64D, ReaderConfig

from .assignments import AssignmentKind, AssignmentStore, TrainingAssignment

LOCAL_TRAINING_ARCHITECTURE = "complete-field-64d-v1"

# The canonical seed-derived initial parameter generation.  A binding that
# names any other generation must present the exact accepted checkpoint
# bytes (loaded by the session before the attempt); the worker never
# fabricates a fresh model for a bound hashed generation.
UNTRAINED_PARAMETER_GENERATION = "untrained"

ATTEMPT_EMISSION_SCHEMA = "axon-trainer-attempt-emission-v1"
ATTEMPT_METRICS_SCHEMA = "axon-trainer-attempt-metrics-v1"
ATTEMPT_EVIDENCE_BUNDLE_SCHEMA = "axon-trainer-attempt-evidence-bundle-v1"
ATTEMPT_EVIDENCE_SCHEMA = "axon-trainer-attempt-evidence-v1"
SOUL_HOT_PAYLOAD_SCHEMA = "axon-local-worker-soul-hot-payload-v1"

EOS_TOKEN_MARKER = "<eos>"

DEFAULT_SEED = 11
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_GRNORM_NORM = 100.0
DEFAULT_EMISSION_WORK_UNITS = 64


class LocalTrainingWorkerError(RuntimeError):
    """A training input is malformed, stale, or unauthenticated; fail closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalTrainingWorkerError(message)


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not (-float("inf") < value < float("inf")):
        raise LocalTrainingWorkerError(f"{label} must be finite; refusing fabricated evidence")
    return value


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _parameter_generation(model: CompleteField64D) -> str:
    """Content-addressed identity of the exact trainable parameter bytes."""

    return canonical_sha256(
        {name: _tensor_sha256(parameter) for name, parameter in model.named_parameters()}
    )


def _freeze_state(value: Any) -> Any:
    """Recursively convert an optimizer state fragment into JSON-safe content."""

    if isinstance(value, torch.Tensor):
        return _tensor_sha256(value)
    if isinstance(value, Mapping):
        return {str(key): _freeze_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze_state(item) for item in value]
    return value


def _optimizer_generation(optimizer: torch.optim.Optimizer) -> str:
    """Content-addressed identity of the exact optimizer state bytes."""

    return canonical_sha256(_freeze_state(optimizer.state_dict()))


def encode_hot_payload(*, reader_state: torch.Tensor, metrics: Mapping[str, Any]) -> bytes:
    """Deterministic opaque bytes for the exhale hot layer.

    Layout: canonical JSON header, one newline, then the exact float32
    reader-state bytes.  ``decode_hot_payload`` reverses this exactly.
    """

    array = reader_state.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    header = {
        "schema": SOUL_HOT_PAYLOAD_SCHEMA,
        "reader_state": {"dtype": "float32", "shape": list(array.shape)},
        "metrics": dict(metrics),
    }
    return canonical_json_bytes(header) + b"\n" + array.tobytes(order="C")


def decode_hot_payload(payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse one hot-layer payload produced by ``encode_hot_payload``."""

    if not isinstance(payload, bytes) or b"\n" not in payload:
        raise LocalTrainingWorkerError("soul hot payload is malformed")
    header_raw, _, raw = payload.partition(b"\n")
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalTrainingWorkerError("soul hot payload header is malformed") from exc
    if not isinstance(header, Mapping) or header.get("schema") != SOUL_HOT_PAYLOAD_SCHEMA:
        raise LocalTrainingWorkerError("soul hot payload schema mismatch")
    spec = header.get("reader_state")
    if not isinstance(spec, Mapping) or spec.get("dtype") != "float32" or not spec.get("shape"):
        raise LocalTrainingWorkerError("soul hot payload tensor spec is malformed")
    shape = tuple(int(dim) for dim in spec["shape"])
    array = np.frombuffer(raw, dtype=np.float32)
    expected_size = int(np.prod(shape)) if shape else 0
    if array.size != expected_size:
        raise LocalTrainingWorkerError("soul hot payload tensor byte length mismatch")
    return array.reshape(shape).copy(), dict(header.get("metrics") or {})


@dataclass(frozen=True, slots=True)
class TypedAttemptEmission:
    """Typed free-running model output over the frozen bank, plus optional EOS.

    This is the emission the worker would place on its perspective rail.  It
    is derived strictly from model logits (never host prose); each token id
    maps to one frozen-bank character or the typed EOS terminal marker.
    """

    token_ids: tuple[int, ...]
    characters: tuple[str, ...]
    supervised_positions: int
    terminated: bool = False
    schema: str = ATTEMPT_EMISSION_SCHEMA
    emission_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise LocalTrainingWorkerError("a typed emission requires at least one position")
        if len(self.token_ids) != len(self.characters):
            raise LocalTrainingWorkerError("emission token ids and characters must align")
        eos_positions = tuple(i for i, value in enumerate(self.characters) if value == EOS_TOKEN_MARKER)
        if self.terminated:
            _require(
                eos_positions == (len(self.characters) - 1,),
                "a terminated emission must contain one final EOS marker",
            )
        else:
            _require(not eos_positions, "an unterminated emission cannot contain EOS")
        _require(
            isinstance(self.supervised_positions, int)
            and not isinstance(self.supervised_positions, bool)
            and self.supervised_positions >= 1,
            "supervised_positions must be a positive integer",
        )
        object.__setattr__(self, "emission_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "token_ids": list(self.token_ids),
            "characters": list(self.characters),
            "supervised_positions": self.supervised_positions,
            "terminated": self.terminated,
        }
        if include_id:
            value["emission_id"] = self.emission_id
        return value


@dataclass(frozen=True, slots=True)
class AttemptMetrics:
    """Worker-measured evidence.  Numbers, never authority."""

    loss: float
    grad_norm: float
    exact_next_char_accuracy: float
    supervised_characters: int
    supervised_positions: int
    schema: str = ATTEMPT_METRICS_SCHEMA

    def __post_init__(self) -> None:
        _finite(self.loss, "loss")
        _finite(self.grad_norm, "grad_norm")
        _finite(self.exact_next_char_accuracy, "exact_next_char_accuracy")
        if not 0.0 <= self.exact_next_char_accuracy <= 1.0:
            raise LocalTrainingWorkerError("exact_next_char_accuracy must lie in [0, 1]")
        for name in ("supervised_characters", "supervised_positions"):
            value = getattr(self, name)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"{name} must be a positive integer",
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "exact_next_char_accuracy": self.exact_next_char_accuracy,
            "supervised_characters": self.supervised_characters,
            "supervised_positions": self.supervised_positions,
        }


@dataclass(frozen=True, slots=True)
class AttemptEvidenceBundle:
    """One exact committed attempt, sealed for the authenticated rail boundary.

    The bundle identity (``bundle_id``) is the canonical sha256 of the exact
    canonical dict.  ``sealed_payload()`` returns the wire bytes of a
    ``rail_auth`` envelope over those bytes, so the local path exercises the
    identical authenticated transport the Kaggle worker will use.  The secret
    itself is never stored here; ``rail_environ`` only names where the
    operator configured it.
    """

    assignment_id: str
    core_id: str
    attempt_index: int
    attempt_id: str
    view_id: str
    rail_id: str
    field_id: str
    tick_id: int
    architecture_generation_before: str
    architecture_generation_after: str
    parameter_generation_before: str
    parameter_generation_after: str
    optimizer_generation_before: str
    optimizer_generation_after: str
    soul_id_before: str
    soul_id_after: str
    metrics: AttemptMetrics
    emission: TypedAttemptEmission
    rail_environ: Mapping[str, str] | None = field(default=None, compare=False, repr=False)
    schema: str = ATTEMPT_EVIDENCE_BUNDLE_SCHEMA
    bundle_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "assignment_id",
            "core_id",
            "attempt_id",
            "view_id",
            "rail_id",
            "field_id",
            "architecture_generation_before",
            "architecture_generation_after",
            "parameter_generation_before",
            "parameter_generation_after",
            "optimizer_generation_before",
            "optimizer_generation_after",
            "soul_id_before",
            "soul_id_after",
        ):
            value = getattr(self, name)
            _require(isinstance(value, str) and value, f"{name} must be a non-empty string")
        _require(
            isinstance(self.attempt_index, int)
            and not isinstance(self.attempt_index, bool)
            and self.attempt_index >= 0,
            "attempt_index must be a non-negative integer",
        )
        _require(isinstance(self.metrics, AttemptMetrics), "metrics must be AttemptMetrics")
        _require(isinstance(self.emission, TypedAttemptEmission), "emission must be TypedAttemptEmission")
        _require(isinstance(self.tick_id, int) and not isinstance(self.tick_id, bool), "tick_id must be an integer")
        object.__setattr__(self, "bundle_id", canonical_sha256(self.to_canonical_dict()))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "assignment_id": self.assignment_id,
            "core_id": self.core_id,
            "attempt_index": self.attempt_index,
            "attempt_id": self.attempt_id,
            "view_id": self.view_id,
            "rail_id": self.rail_id,
            "field_id": self.field_id,
            "tick_id": self.tick_id,
            "architecture_generation_before": self.architecture_generation_before,
            "architecture_generation_after": self.architecture_generation_after,
            "parameter_generation_before": self.parameter_generation_before,
            "parameter_generation_after": self.parameter_generation_after,
            "optimizer_generation_before": self.optimizer_generation_before,
            "optimizer_generation_after": self.optimizer_generation_after,
            "soul_id_before": self.soul_id_before,
            "soul_id_after": self.soul_id_after,
            "metrics": self.metrics.to_canonical_dict(),
            "emission": self.emission.to_canonical_dict(),
        }

    def sealed_payload(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        now: float | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> bytes:
        """Seal this bundle into rail-auth wire bytes (canonical JSON envelope)."""

        envelope = seal_envelope(
            core_identity=self.core_id,
            assignment_id=self.assignment_id,
            payload=canonical_json_bytes(self.to_canonical_dict()),
            ttl_seconds=ttl_seconds,
            now=now,
            environ=environ if environ is not None else (self.rail_environ or os.environ),
        )
        return canonical_json_bytes(envelope)


class LocalTrainingWorker:
    """In-process OFFLINE_TRAINING worker executing the real D64 motor.

    Constructor-injected roots only: the assignment store, the operator rail
    secret location, the seed, the clock.  No network, no HTTP, no threads.
    The same worker instance persists parameters and optimizer state across
    attempts; parameter lineage is tracked in memory and any binding that
    does not match the tracked lineage fails closed as stale.
    """

    def __init__(
        self,
        *,
        assignment_store: AssignmentStore,
        seed: int = DEFAULT_SEED,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_grad_norm: float = DEFAULT_MAX_GRNORM_NORM,
        emission_work_units: int = DEFAULT_EMISSION_WORK_UNITS,
        reader_config: ReaderConfig | None = None,
        clock: Callable[[], float] | None = None,
        rail_environ: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(assignment_store, AssignmentStore):
            raise TypeError("LocalTrainingWorker requires an AssignmentStore")
        self.assignment_store = assignment_store
        self._seed = int(seed)
        self._learning_rate = _finite(float(learning_rate), "learning_rate")
        self._max_grad_norm = _finite(float(max_grad_norm), "max_grad_norm")
        _require(
            isinstance(emission_work_units, int)
            and not isinstance(emission_work_units, bool)
            and emission_work_units >= 1,
            "emission_work_units must be a positive integer",
        )
        self._emission_work_units = emission_work_units
        self._reader_config = reader_config
        self._clock = clock if clock is not None else time.time
        self._rail_environ = rail_environ
        self._model: CompleteField64D | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._schema_version: str | None = None
        self._parameter_generation: str | None = None

    # ------------------------------------------------------------------ model

    @property
    def model(self) -> CompleteField64D:
        if self._model is None:
            raise LocalTrainingWorkerError("worker model is not initialized; run one attempt first")
        return self._model

    def _ensure_model(self, schema_version: str) -> None:
        if self._model is not None:
            _require(
                self._schema_version == schema_version,
                f"worker is bound to schema {self._schema_version!r}, not {schema_version!r}",
            )
            return
        torch.manual_seed(self._seed)
        base = self._reader_config if self._reader_config is not None else ReaderConfig(dropout=0.0)
        cfg = replace(base, field_schema_version=schema_version)
        self._model = CompleteField64D(cfg)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._schema_version = schema_version

    # --------------------------------------------------------------- attempt

    def execute_attempt(
        self,
        binding: CoreBinding,
        assignment: TrainingAssignment,
        view: Any,
        soul_branch: SoulBranch,
        target_text: str | None = None,
    ) -> AttemptEvidenceBundle:
        """Execute one real attempt; commit evidence and exhale the real Soul.

        ``target_text`` is the optional explicit expected target of the
        assignment (for ``ABC? -> D`` style contracts).  When supplied, the
        supervised objective contract — teacher-forced decode, discrete
        categorical loss, and backward pass — is computed against THAT text
        while the view still attends the canonical training regions.  The
        typed proposal is decoded separately and causally.  When absent, the
        attended trainer_instructions text remains the supervised objective.
        """

        # ---- fail-closed validation of every bound identity ----------------
        _require(isinstance(binding, CoreBinding), "binding must be a CoreBinding")
        _require(
            binding.mode is CoreStatus.OFFLINE_TRAINING,
            "a training attempt requires an OFFLINE_TRAINING binding",
        )
        _require(
            binding.architecture_id == LOCAL_TRAINING_ARCHITECTURE,
            f"binding architecture {binding.architecture_id!r} is not this worker's "
            f"architecture {LOCAL_TRAINING_ARCHITECTURE!r}",
        )
        _require(isinstance(assignment, TrainingAssignment), "assignment must be a TrainingAssignment")
        _require(
            isinstance(view, TrainingAttentionView),
            "view must be a TrainingAttentionView compiled from the canonical field",
        )
        _require(isinstance(soul_branch, SoulBranch), "soul_branch must be a SoulBranch")
        core_id = binding.core_id
        _require(view.core_id == core_id, "view belongs to a different core")
        _require(soul_branch.core_id == core_id, "soul branch belongs to a different core")
        binding_field = str(assignment.field_binding.get("field_id", ""))
        binding_view = str(assignment.field_binding.get("view_id", ""))
        _require(
            binding_field == view.source_field_id,
            "assignment field binding does not match the view "
            f"({binding_field!r} != {view.source_field_id!r})",
        )
        _require(
            binding_view == view.view_id,
            "assignment view binding does not match the view "
            f"({binding_view!r} != {view.view_id!r})",
        )
        binding_tick = assignment.field_binding.get("tick_id")
        if binding_tick is not None:
            _require(int(binding_tick) == view.source_tick_id, "assignment tick binding does not match the view")
        current = self.assignment_store.get_assignment(assignment.assignment_id)
        _require(
            current.status.value in ("claimed", "active"),
            f"assignment is not live (status={current.status.value}); refusing the attempt",
        )
        _require(
            current.lease is not None and current.lease.holder_core_id == core_id,
            "assignment lease is not held by this core",
        )
        _require(not current.lease.expired(self._clock()), "assignment lease is expired; reclaim first")
        if self._parameter_generation is not None:
            _require(
                binding.parameter_generation == self._parameter_generation,
                "binding parameter generation is stale for this worker's tracked lineage",
            )
        elif self._model is None:
            # No accepted checkpoint has been loaded into this worker, so the
            # only honest state it can present is the canonical seed-derived
            # untrained generation; fail closed on any other bound generation
            # rather than fabricating a fresh model for hashed lineage.
            _require(
                binding.parameter_generation == UNTRAINED_PARAMETER_GENERATION,
                "binding parameter generation "
                f"{binding.parameter_generation!r} names accepted checkpoint "
                "lineage that this worker does not hold; load the exact "
                "accepted bytes before attempting",
            )

        # ---- (1) inhale the exact private Soul ----------------------------
        inhaled: SoulSnapshot = soul_branch.load_head()
        _require(
            inhaled.soul_id == binding.soul_id,
            "binding soul_id does not match the inhaled SoulStore head; stale or foreign Soul",
        )

        # ---- (2/3) real forward + discrete categorical loss on one graph ----
        self._ensure_model(view.source_schema_version)
        model = self._model
        assert model is not None and self._optimizer is not None
        parameter_before = _parameter_generation(model)
        optimizer_before = _optimizer_generation(self._optimizer)
        control_bytes = {
            name: tensor.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
            for name, tensor in model.named_buffers()
        }
        objective_text = view.compiled.region_text(LogicalRegion.TRAINER_INSTRUCTIONS)
        _require(bool(objective_text), "attended trainer_instructions text is empty; no objective")
        explicit_target = None
        if target_text is not None:
            _require(
                isinstance(target_text, str) and bool(target_text.strip()),
                "explicit target text must be a non-empty string",
            )
            explicit_target = target_text.strip()
        supervised_text = explicit_target if explicit_target is not None else objective_text
        # (1b) the inhaled Soul conditions the recurrent forward: the exact
        # hot-layer payload exhaled by the previous attempt is decoded and
        # passed as the reader's initial recurrent state, so the persisted
        # Soul genuinely drives this forward pass.
        hot = inhaled.layer(SoulTemperature.HOT)
        initial_state: torch.Tensor | None = None
        if hot.payload:
            reader_array, _soul_metrics = decode_hot_payload(hot.payload)
            initial_state = torch.from_numpy(reader_array)
            expected_shape = tuple(model.initial_state.shape)
            # The exhale persists the batched reader state (1, S, D); the
            # bound architecture shape is (S, D).  Both are exact encodings
            # of the same recurrent state; anything else is a foreign Soul.
            if tuple(initial_state.shape) == (1, *expected_shape):
                initial_state = initial_state.squeeze(0)
            _require(
                tuple(initial_state.shape) == expected_shape,
                f"inhaled Soul reader state has shape {tuple(initial_state.shape)}, "
                f"not the bound architecture shape {expected_shape}; refusing a foreign Soul",
            )

        torch.manual_seed(self._seed)
        model.train()
        reader_state, memory, _manifest = model.read_compiled_with_memory(
            view.compiled,
            initial_state=initial_state,
        )
        try:
            log_probabilities, targets = model.decode_teacher(
                reader_state,
                supervised_text,
                0,
                memory=memory,
            )
        except KeyError as exc:
            raise LocalTrainingWorkerError(
                f"objective text contains a character outside the frozen bank: {exc!r}"
            ) from exc
        loss = F.nll_loss(
            log_probabilities.reshape(-1, log_probabilities.shape[-1]),
            targets.reshape(-1),
        )
        loss_value = _finite(float(loss.detach().item()), "loss")

        predicted = log_probabilities.argmax(dim=-1)[0]
        target_flat = targets[0]
        supervised_positions = int(target_flat.shape[0])
        supervised_characters = len(supervised_text)
        char_correct = int((predicted[:-1] == target_flat[:-1]).sum().item())
        accuracy = char_correct / supervised_characters if supervised_characters else 0.0
        # The supervised logits above are loss evidence.  The rail emission
        # must be causal: it starts from the decoder BOS state and feeds back
        # only its own prior predictions, never the expected target prefix.
        attempted_text, terminated = model.decode_greedy(
            reader_state,
            head=0,
            work_units=self._emission_work_units,
            memory=memory,
        )
        token_ids = tuple(model.char_to_index[character] for character in attempted_text)
        characters = tuple(attempted_text)
        if terminated:
            token_ids = (*token_ids, model.eos_index)
            characters = (*characters, EOS_TOKEN_MARKER)
        emission = TypedAttemptEmission(
            token_ids=token_ids,
            characters=characters,
            supervised_positions=supervised_positions,
            terminated=terminated,
        )

        # ---- (4) real backward; optimizer step only for learning ----------
        is_learning = assignment.kind is AssignmentKind.LEARNING
        loss.backward()
        grad_norm = _finite(
            float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), self._max_grad_norm).item()
            ),
            "grad_norm",
        )
        if grad_norm <= 0.0:
            raise LocalTrainingWorkerError("zero gradient norm on the objective graph; refusing to proceed")
        if is_learning:
            self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

        parameter_after = _parameter_generation(model)
        optimizer_after = _optimizer_generation(self._optimizer)
        for name, tensor in model.named_buffers():
            observed = tensor.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
            _require(
                observed == control_bytes[name],
                f"frozen control buffer {name!r} changed during the attempt",
            )

        metrics = AttemptMetrics(
            loss=loss_value,
            grad_norm=grad_norm,
            exact_next_char_accuracy=accuracy,
            supervised_characters=supervised_characters,
            supervised_positions=supervised_positions,
        )

        # ---- (5) exhale the real Soul state through SoulStore --------------
        attempt_index = len(self.assignment_store.attempts_for(current.assignment_id))
        soul_after_id = inhaled.soul_id
        transition_id = ""
        if is_learning:
            hot_payload = encode_hot_payload(
                reader_state=reader_state,
                metrics={
                    "loss": metrics.loss,
                    "grad_norm": metrics.grad_norm,
                    "attempt_index": attempt_index,
                },
            )
            transition = SoulTransition(
                core_id=core_id,
                architecture_id=inhaled.architecture_id,
                parameter_generation=inhaled.parameter_generation,
                before_soul_id=inhaled.soul_id,
                before_generation=inhaled.generation,
                tick_uid=f"{view.source_field_id}:{view.source_tick_id}",
                request_id=f"training-attempt:{current.assignment_id}:{attempt_index}",
                phase="training",
                updates=(
                    SoulLayer(
                        temperature=SoulTemperature.HOT,
                        payload=hot_payload,
                        media_type="application/x-axon-local-worker-soul-hot",
                        tensor_layout="reader-state-float32",
                    ),
                ),
            )
            receipt = soul_branch.commit_transition(
                transition,
                commit_binding=f"training-attempt:{current.assignment_id}:{attempt_index}",
            )
            _require(
                receipt.before_soul_id == inhaled.soul_id,
                "soul exhale receipt does not bind the inhaled Soul",
            )
            soul_after_id = receipt.after_soul_id
            transition_id = receipt.transition_id

        # ---- evidence into the assignment store (outcomes remain Heart's) --
        evidence = {
            "schema": ATTEMPT_EVIDENCE_SCHEMA,
            "bundle_fields": {
                "view_id": view.view_id,
                "rail_id": view.compiled.rail_id,
                "field_id": view.source_field_id,
                "tick_id": view.source_tick_id,
            },
            "metrics": metrics.to_canonical_dict(),
            "emission_sha256": emission.emission_id,
            "soul_transition_id": transition_id or None,
            "learning_step_applied": is_learning,
            "explicit_target_sha256": (
                canonical_sha256(explicit_target)
                if explicit_target is not None
                else None
            ),
        }
        emission_ref = {
            "schema": ATTEMPT_EMISSION_SCHEMA,
            "emission_id": emission.emission_id,
            "rail_id": view.compiled.rail_id,
            "view_id": view.view_id,
        }
        soul_lineage = {
            "branch_id": soul_branch.branch_id,
            "soul_id_before": inhaled.soul_id,
            "soul_id_after": soul_after_id,
            "generation_before": inhaled.generation,
            "soul_transition_id": transition_id or None,
        }
        attempt = self.assignment_store.record_attempt(
            current.assignment_id,
            core_id=core_id,
            emission_ref=emission_ref,
            soul_lineage=soul_lineage,
            parameter_generation=parameter_after,
            optimizer_generation=optimizer_after,
            evidence=evidence,
            expected_revision=current.revision,
            idempotency_key=(
                f"local-worker:{core_id}:{current.assignment_id}:{attempt_index}"
            ),
        )

        self._parameter_generation = parameter_after

        # ---- (6) assemble the sealed evidence bundle -----------------------
        return AttemptEvidenceBundle(
            assignment_id=current.assignment_id,
            core_id=core_id,
            attempt_index=attempt.attempt_index,
            attempt_id=attempt.attempt_id,
            view_id=view.view_id,
            rail_id=view.compiled.rail_id,
            field_id=view.source_field_id,
            tick_id=view.source_tick_id,
            architecture_generation_before=binding.architecture_id,
            architecture_generation_after=binding.architecture_id,
            parameter_generation_before=parameter_before,
            parameter_generation_after=parameter_after,
            optimizer_generation_before=optimizer_before,
            optimizer_generation_after=optimizer_after,
            soul_id_before=inhaled.soul_id,
            soul_id_after=soul_after_id,
            metrics=metrics,
            emission=emission,
            rail_environ=self._rail_environ,
        )


__all__ = [
    "ATTEMPT_EMISSION_SCHEMA",
    "ATTEMPT_EVIDENCE_BUNDLE_SCHEMA",
    "ATTEMPT_EVIDENCE_SCHEMA",
    "ATTEMPT_METRICS_SCHEMA",
    "DEFAULT_EMISSION_WORK_UNITS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_GRNORM_NORM",
    "DEFAULT_SEED",
    "EOS_TOKEN_MARKER",
    "LOCAL_TRAINING_ARCHITECTURE",
    "SOUL_HOT_PAYLOAD_SCHEMA",
    "UNTRAINED_PARAMETER_GENERATION",
    "AttemptEvidenceBundle",
    "AttemptMetrics",
    "LocalTrainingWorker",
    "LocalTrainingWorkerError",
    "TypedAttemptEmission",
    "decode_hot_payload",
    "encode_hot_payload",
]
