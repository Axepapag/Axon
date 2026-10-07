"""E0 two-state loop: the one Heart path used for training and inference.

This module joins three already-tested pieces without editing any of them:

* ``core.e0_two_state`` supplies the E0 organism (a reasoning state, a response
  state, a 96-way character head and a 3-way WAIT/COMMIT/END control head);
* ``curriculum`` supplies the architecture-independent episode schema,
  validation, the version hash and the evaluation metrics/results row;
* ``runtime.heart.host`` supplies the HeartHost coordinator, which is the only
  writer of canonical state.

LAW NOTES (non-negotiable):

* Exactly the 95 frozen native characters are admitted. Episode text reaches
  the core only through ``substrate.native.encode_ids`` and any character
  outside the native 95 fails closed: nothing is converted, escaped,
  normalized or clamped. The core's own emitted characters are native by
  construction (``native_char``); an EMPTY (id 95) proposal emits nothing.
* The core only *proposes*. Character choices and the WAIT/COMMIT/END decision
  come from the core's own heads, and the readable text lives in the host's
  **private draft**. Committed canonical text exists only when a heart
  commit happens: the loop registers its core on the host and commits every
  emitted draft through ``host.submit_control(core_id, "commit")`` followed by
  ``host.commit(current_consolidator(), proposal_id)``. The core never touches
  the branch.
* WAIT performs zero operations: no draft mutation, no control submission and
  no commit. WAIT is never spelled as an EMPTY or filler payload.
* END finalizes the episode through ``host.submit_control(core_id, "end")``;
  the episode's end-of-walk host beat consumes it.
* Training and inference call the exact same ``run_episode``; ``train=True``
  supervises the **full response** and calls the caller's ``optimizer.step()``;
  inference never touches weights.

Phases (autoregressive protocol v2):

* **observe** - steps and query are context in BOTH modes. No draft mutations.
* **respond** - the first input is the existing EMPTY cell (a generation cue,
  never a WAIT payload). Training feeds the PREVIOUS target character, predicts
  the current one as COMMIT, then learns END on a separate terminal tick.
  WAIT/END-only examples receive control loss too. Inference feeds its own
  previous emitted character, with no answer text or answer-length input.
  Its control head drives WAIT (zero operations), COMMIT (a native character
  through the Heart), or END. A fixed caller-owned tick budget bounds it;
  exhaustion is reported, never disguised as a completed answer.

Training retains the observation/response autograd graph and applies ONE
optimizer update per complete episode (mean response loss). After a mid-episode
restore it rebuilds the prefix graph at unchanged weights without Heart writes
or optimizer updates. This gives response losses credit through the observed
memory while keeping checkpoints free of autograd objects. E0's GRU has no
stochastic layers. Results mark teacher-forced text separately from independent
generation; its exact-match score is not evidence of learned recall.

The results row keeps ``curriculum.metrics.RESULTS_FIELDS`` in order and adds
``"phases": {"observe": n, "respond": n}`` (observations and actual response
ticks); ``generation`` identifies mode, tick budget and termination. Training
rows additionally carry ``training`` with real optimizer steps, mean episode
``objective_loss``, the last 256 response loss samples, and
``final_loss == loss_samples[-1]`` (null when nothing was supervised).
Inference rows omit the training block entirely.

Mid-episode restart: ``save(tag)`` writes the core checkpoint (weights plus
both live states), the host checkpoint, ``optimizer.pt`` (when an optimizer has
been registered) and the RNG artifacts (``rng/torch_cpu``, ``rng/torch_cuda_<i>``
for every CUDA device, ``rng/numpy``, ``rng/python``) plus a loop cursor holding
both the legacy flat position keys and the structured
``axon-lab-execution-cursor-v1`` block.  ``load(tag)`` restores every artifact
that is present, so an interrupted training run continues with identical
weights, identical optimizer state and identical subsequent loss samples.
"""

from __future__ import annotations

import base64
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.e0_two_state import (
    ALPHABET_SIZE,
    CONTROL_COMMIT,
    CONTROL_END,
    CONTROL_WAIT,
    E0TwoStateCore,
    EMPTY_ID,
    load_checkpoint as load_core_checkpoint,
    save_checkpoint as save_core_checkpoint,
)
from curriculum.metrics import RESULTS_FIELDS, evaluate_episode
from curriculum.schema import CURRICULUM_VERSION, validate_episode
from curriculum.splits import curriculum_sha256
from runtime.field import LogicalRegion
from substrate.native import encode_ids, native_char

from .errors import DuplicateCoreError
from .host import HeartHost

#: Component id from ``core.manifests`` that identifies this architecture.
ARCHITECTURE_ID = "axon.core_reasoning_gru"

#: Control decisions are the same triple the curriculum schema uses.
WAIT = CONTROL_WAIT
COMMIT = CONTROL_COMMIT
END = CONTROL_END

#: Split names an evaluation row may record.
SPLITS = ("train", "validation", "test")

DEFAULT_CORE_ID = "e0"

CORE_CHECKPOINT_FILENAME = "core.pt"
HOST_CHECKPOINT_FILENAME = "host_checkpoint.json"
OPTIMIZER_CHECKPOINT_FILENAME = "optimizer.pt"
CURSOR_FILENAME = "loop_cursor.json"
RNG_DIRNAME = "rng"

#: Schema id of the structured execution cursor block.
CURSOR_SCHEMA = "axon-lab-execution-cursor-v1"
#: Schema id of one serialized RNG artifact under ``rng/``.
RNG_SCHEMA = "axon-e0-rng-state-v1"

#: Phase names recorded in the results row and the cursor.
PHASE_OBSERVE = "observe"
PHASE_RESPOND = "respond"

#: ``loss_samples`` keeps only the most recent samples.
LOSS_SAMPLE_CAP = 256
EXECUTION_PROTOCOL = "axon-e0-autoregressive-v2"
DEFAULT_RESPONSE_TICK_BUDGET = 256


def _safe_component(value: str) -> str:
    """One safe single path component for a checkpoint tag."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("tag must be a non-empty string")
    tag = value.strip()
    if tag in {".", ".."} or any(character in tag for character in '\\/:*?"<>|'):
        raise ValueError(f"tag {value!r} is not a safe path component")
    return tag


def _torch_rng_state_b64() -> str:
    return base64.b64encode(bytes(torch.get_rng_state().tolist())).decode("ascii")


def _restore_torch_rng_state(encoded: str) -> None:
    raw = base64.b64decode(encoded.encode("ascii"))
    torch.set_rng_state(torch.tensor(list(raw), dtype=torch.uint8))


def _torch_cuda_rng_state_b64(index: int) -> str:
    return base64.b64encode(bytes(torch.cuda.get_rng_state(index).tolist())).decode(
        "ascii"
    )


def _restore_torch_cuda_rng_state(index: int, encoded: str) -> None:
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(
            f"rng artifact names CUDA device {index} but this host has no such device"
        )
    raw = base64.b64decode(encoded.encode("ascii"))
    torch.cuda.set_rng_state(torch.tensor(list(raw), dtype=torch.uint8), index)


def _numpy_rng_state_json() -> list[Any]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return [
        str(name),
        base64.b64encode(np.asarray(keys, dtype=np.uint32).tobytes()).decode("ascii"),
        int(position),
        int(has_gauss),
        float(cached),
    ]



def _restore_numpy_rng_state(state: Iterable[Any]) -> None:
    name, keys_b64, position, has_gauss, cached = state
    keys = np.frombuffer(base64.b64decode(str(keys_b64).encode("ascii")), dtype=np.uint32).copy()
    np.random.set_state((str(name), keys, int(position), int(has_gauss), float(cached)))


def _py_rng_state_json() -> list[Any]:
    version, internal, gauss = random.getstate()
    return [int(version), [int(item) for item in internal], gauss]


def _restore_py_rng_state(state: Iterable[Any]) -> None:
    version, internal, gauss = state
    random.setstate((int(version), tuple(int(item) for item in internal), gauss))


class E0Loop:
    """Drive E0 episodes through one HeartHost, identically in train and infer.

    ``host`` must already be the single writer (call ``host.start()`` first).
    The loop registers its core on the host at construction; the host's own
    consolidator roster stays the only canonical committer, and the loop asks
    for ``host.current_consolidator()`` at every commit.

    Test seam: ``control_override`` may be set to a callable
    ``(phase, position, control_logits) -> int | None`` that forces the control
    decision for one tick (``None`` keeps the core's own argmax).  It never
    changes the supervision targets or the emission policy; it exists so tests
    can drive WAIT/COMMIT/END paths deterministically.  Production code leaves
    it ``None``.
    """

    def __init__(
        self,
        state_root: Path | str,
        host: HeartHost,
        core: E0TwoStateCore,
        episodes: list[dict],
        device: str = "cpu",
        *,
        response_tick_budget: int = DEFAULT_RESPONSE_TICK_BUDGET,
    ) -> None:
        if not isinstance(host, HeartHost):
            raise TypeError(f"host must be a HeartHost; got {type(host).__name__}")
        if not isinstance(core, E0TwoStateCore):
            raise TypeError(f"core must be an E0TwoStateCore; got {type(core).__name__}")

        self.state_root = Path(state_root).resolve(strict=False)
        self.host = host
        self.core = core
        self.episodes = [dict(episode) for episode in episodes]
        self.device = torch.device(device)
        if isinstance(response_tick_budget, bool) or not isinstance(response_tick_budget, int) or response_tick_budget < 1:
            raise ValueError("response_tick_budget must be a positive int")
        self.response_tick_budget = response_tick_budget

        #: Settable identity labels recorded in every results row.
        self.core_id = DEFAULT_CORE_ID
        self.architecture_id = ARCHITECTURE_ID
        #: None until ``save(tag)`` exists; then the tag of the last checkpoint.
        self.checkpoint_id: str | None = None

        self.curriculum_version = CURRICULUM_VERSION
        self.curriculum_sha256 = curriculum_sha256(self.episodes)

        #: Optional test seam (see the class docstring); None in production.
        self.control_override: Optional[Callable[[str, int, torch.Tensor], Optional[int]]] = None

        self.core.to(self.device)
        self.reasoning_state, self.response_state = self.core.initial_states(
            1, self.device
        )
        self.last_control: Optional[int] = None
        self.last_commit_id: Optional[str] = None
        #: Canonical commits this loop instance made for the current episode.
        self.commits = 0
        #: Telemetry of the most recent ``run_episode`` call (None before any).
        self.last_training: Optional[dict[str, Any]] = None
        self._episode_ended = False
        self._training: Optional[dict[str, Any]] = None
        self._optimizer: Optional[Any] = None
        self._execution: Optional[dict[str, Any]] = None
        self._losses: list[torch.Tensor] = []
        self._generation: dict[str, Any] = {}
        self._dataset: dict[str, Optional[str]] = {
            "preset": None,
            "curriculum_version": None,
            "curriculum_sha256": None,
            "split_manifest_sha256": None,
        }
        self._cursor: dict[str, Any] = {
            "episode_index": None,
            "step_index": 0,
            "char_position": 0,
        }
        self._register_core()

    # ------------------------------------------------------------------ public

    def register_optimizer(self, optimizer: Any) -> None:
        """Register the caller's optimizer so ``save(tag)`` persists its state.

        Only the state dict is stored (``optimizer.pt``); the optimizer object
        belongs to the caller and is never constructed or replaced by the loop.
        """

        if optimizer is None:
            raise ValueError("register_optimizer requires an optimizer")
        for name in ("state_dict", "load_state_dict", "step", "zero_grad"):
            if not callable(getattr(optimizer, name, None)):
                raise TypeError(
                    f"optimizer must provide {name}(); got {type(optimizer).__name__}"
                )
        self._optimizer = optimizer

    def bind_dataset(
        self,
        preset: Optional[str] = None,
        *,
        curriculum_version: Optional[str] = None,
        curriculum_sha256: Optional[str] = None,
        split_manifest_sha256: Optional[str] = None,
    ) -> dict[str, Optional[str]]:
        """Bind the dataset identity the cursor block reports.

        Every field is optional and unset fields stay ``None`` - the loop never
        invents a preset name or a manifest hash.  Callers that want the loop's
        own curriculum identity recorded pass it explicitly, for example
        ``loop.bind_dataset("e0-first", curriculum_version=loop.curriculum_version,
        curriculum_sha256=loop.curriculum_sha256,
        split_manifest_sha256=manifest["manifest_sha256"])``.
        """

        values = {
            "preset": preset,
            "curriculum_version": curriculum_version,
            "curriculum_sha256": curriculum_sha256,
            "split_manifest_sha256": split_manifest_sha256,
        }
        for name, value in values.items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"dataset {name} must be None or a non-empty string; got {value!r}"
                )
            self._dataset[name] = None if value is None else value.strip()
        return dict(self._dataset)

    def run_episode(
        self,
        episode: dict,
        train: bool = False,
        optimizer: Optional[Any] = None,
        *,
        split: Optional[str] = None,
    ) -> dict:
        """Walk one validated episode and return its results row.

        Both modes observe all context then enter the response phase. Training
        uses shifted teacher forcing and one full-episode gradient update;
        inference uses previous predictions and its own controls, bounded by
        ``response_tick_budget``. Expected answers are used only for training
        targets and post-generation scoring. See the module's protocol notes.
        """

        errors = validate_episode(episode)
        if errors:
            raise ValueError(
                f"episode {episode.get('episode_id')!r} failed curriculum validation: "
                + "; ".join(errors)
            )
        if train and optimizer is None:
            raise ValueError("train=True requires the caller's optimizer")
        resolved_split = self._resolve_split(episode, split)
        if train and resolved_split != "train":
            raise ValueError("training may only consume the train split")
        respond_ids = list(encode_ids(episode["expected"].get("text") or "")) if train else []
        plan = self._walk_plan(episode, respond_ids)
        if not train:
            plan = [entry for entry in plan if entry[0] == PHASE_OBSERVE]
            plan.extend((PHASE_RESPOND, "char", EMPTY_ID, pos) for pos in range(self.response_tick_budget))
        resume_from = self._resume_index(episode, plan)
        execution = {"protocol": EXECUTION_PROTOCOL, "train": train,
                     "response_tick_budget": self.response_tick_budget,
                     "episode_sha256": curriculum_sha256([episode])}
        if resume_from and self._execution != execution:
            raise ValueError("checkpoint execution protocol/mode/budget/episode mismatch; rebuild the episode explicitly")
        self._execution = execution

        self._training = None
        self._losses = []
        self._generation = {"mode": "teacher_forced_training" if train else "independent_generation",
                            "response_tick_budget": None if train else self.response_tick_budget,
                            "termination": None, "response_ticks": 0}
        if train:
            self.register_optimizer(optimizer)
            optimizer.zero_grad()
            self._training = {
                "optimizer_steps": 0,
                "loss_samples": deque(maxlen=LOSS_SAMPLE_CAP),
                "final_loss": None,
                "objective_loss": None,
            }
        if resume_from == 0:
            self.host.reset_episode()
            self.reasoning_state, self.response_state = self.core.initial_states(
                1, self.device
            )
            self.last_control = None
            self._episode_ended = False
            self.commits = 0
        elif train:
            # Serialized state values cannot carry autograd history. Recreate
            # the deterministic prefix at the checkpoint's unchanged weights.
            self._replay_training_prefix(episode, plan[:resume_from], respond_ids)
        self._cursor = {
            "episode_index": self._episode_index(episode),
            "step_index": resume_from,
            "char_position": 0,
        }

        for index in range(resume_from, len(plan)):
            phase, kind, char_id, phase_char = plan[index]
            # The cursor names the step about to run, so a save taken inside a
            # tick matches the live states exactly.
            self._cursor["step_index"] = index
            self._cursor["char_position"] = phase_char
            if kind == "reset":
                self.reasoning_state, self.response_state = self.core.initial_states(
                    1, self.device
                )
                continue
            with torch.set_grad_enabled(train):
                if phase == PHASE_OBSERVE:
                    self._observe_tick(char_id, train=train)
                elif train:
                    self._respond_tick(char_id, phase_char, len(respond_ids), optimizer,
                                       terminal_control=episode["expected"].get("control"))
                else:
                    self._generate_tick(phase_char)
            if self._episode_ended:
                # An observed END (inference) or the final respond step ends the
                # walk; the cursor below is marked finished either way.
                break
            self._cursor["char_position"] = phase_char + 1
            self._cursor["step_index"] = index + 1
        # The walk is over (completed or ended): nothing is left to resume.
        self._cursor["step_index"] = len(plan)
        if train:
            objective = torch.stack(self._losses).mean()
            self._training["objective_loss"] = float(objective.detach())
            objective.backward()
            optimizer.step()
            optimizer.zero_grad()
            self._training["optimizer_steps"] = 1
            self.reasoning_state = self.reasoning_state.detach()
            self.response_state = self.response_state.detach()
            self._losses = []
        elif not self._episode_ended:
            self._generation["termination"] = "budget_exhausted"
        self.host.beat()
        return self._results_row(episode, resolved_split, plan, train=train)

    def save(self, tag: str) -> Path:
        """Write every restart artifact under ``state_root/e0/<tag>/``.

        Artifacts: the core checkpoint (weights plus both live states, sha256
        sidecar), the host checkpoint (sha256 sidecar), ``optimizer.pt`` when an
        optimizer has been registered, the RNG artifacts ``rng/torch_cpu``,
        ``rng/torch_cuda_<i>`` (one per CUDA device), ``rng/numpy`` and
        ``rng/python``, and the loop cursor.  The cursor keeps the legacy flat
        position/RNG keys and adds the structured ``axon-lab-execution-cursor-v1``
        block; it is restart guidance, never canonical identity, so it is
        written atomically without a sidecar and a truncated cursor fails JSON
        parsing (fail closed).
        """

        directory = self._checkpoint_dir(tag)
        directory.mkdir(parents=True, exist_ok=True)
        save_core_checkpoint(
            directory / CORE_CHECKPOINT_FILENAME,
            self.core,
            self.reasoning_state,
            self.response_state,
            self.core.config(),
        )
        self.host.save_checkpoint(directory / HOST_CHECKPOINT_FILENAME)
        if self._optimizer is not None:
            torch.save(
                self._optimizer.state_dict(), directory / OPTIMIZER_CHECKPOINT_FILENAME
            )
        self._save_rng_artifacts(directory)
        cursor = {
            "episode_index": self._cursor.get("episode_index"),
            "step_index": int(self._cursor.get("step_index", 0)),
            "torch_rng_state": _torch_rng_state_b64(),
            "numpy_rng_state": _numpy_rng_state_json(),
            "py_rng_state": _py_rng_state_json(),
            "tag": tag,
            "cursor": self._cursor_block(),
            "execution": self._execution,
        }
        cursor_path = directory / CURSOR_FILENAME
        temporary = cursor_path.with_name(cursor_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(cursor, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cursor_path)
        self.checkpoint_id = tag
        return directory

    def load(self, tag: str) -> dict:
        """Restore the core, host and loop cursor saved under ``e0/<tag>/``."""

        directory = self._checkpoint_dir(tag)
        core_path = directory / CORE_CHECKPOINT_FILENAME
        loaded_core, reasoning_state, response_state, config = load_core_checkpoint(
            core_path, map_location=self.device
        )
        for name in ("width", "hidden_size", "max_chunk"):
            if config.get(name) != self.core.config().get(name):
                raise ValueError(
                    f"core checkpoint {name} {config.get(name)!r} does not match this "
                    f"loop's core {self.core.config().get(name)!r}"
                )
        self.core.load_state_dict(loaded_core.state_dict())
        self.core.to(self.device)
        self.reasoning_state = reasoning_state.to(self.device).detach()
        self.response_state = response_state.to(self.device).detach()

        host_report = self.host.load_checkpoint(directory / HOST_CHECKPOINT_FILENAME)

        cursor = json.loads(
            (directory / CURSOR_FILENAME).read_text(encoding="utf-8")
        )
        if not isinstance(cursor, dict):
            raise ValueError(f"loop cursor in {directory} must be a JSON object")
        self._execution = cursor.get("execution")
        if cursor.get("tag") != tag:
            raise ValueError(
                f"loop cursor tag {cursor.get('tag')!r} does not match {tag!r}"
            )
        episode_index = cursor.get("episode_index")
        if episode_index is not None:
            if isinstance(episode_index, bool) or not isinstance(episode_index, int):
                raise ValueError("loop cursor episode_index must be null or an int")
        step_index = cursor.get("step_index")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("loop cursor step_index must be a non-negative int")

        cursor_block = cursor.get("cursor")
        if cursor_block is not None:
            cursor_block = self._validate_cursor_block(cursor_block)

        optimizer_restored = False
        optimizer_path = directory / OPTIMIZER_CHECKPOINT_FILENAME
        if optimizer_path.is_file() and self._optimizer is not None:
            payload = torch.load(
                optimizer_path, map_location=self.device, weights_only=False
            )
            if not isinstance(payload, dict):
                raise ValueError(f"optimizer checkpoint must be a dict: {optimizer_path}")
            self._optimizer.load_state_dict(payload)
            optimizer_restored = True

        self._restore_rng_artifacts(directory, cursor)

        char_position = 0
        if cursor_block is not None:
            next_block = cursor_block["next"]
            if next_block["episode_index"] is not None:
                episode_index = int(next_block["episode_index"])
            step_index = int(next_block["step_index"])
            char_position = int(next_block["char_position"])
            dataset = cursor_block["dataset"]
            self._dataset = {
                "preset": dataset["preset"],
                "curriculum_version": dataset["curriculum_version"],
                "curriculum_sha256": dataset["curriculum_sha256"],
                "split_manifest_sha256": dataset["split_manifest_sha256"],
            }
            self.last_control = cursor_block["last_control"]
            self._episode_ended = bool(cursor_block["episode_ended"])

        self._cursor = {
            "episode_index": episode_index,
            "step_index": step_index,
            "char_position": char_position,
        }
        self.checkpoint_id = tag
        return {
            "tag": tag,
            "directory": str(directory),
            "episode_index": episode_index,
            "step_index": step_index,
            "char_position": char_position,
            "cursor": dict(cursor_block) if cursor_block is not None else None,
            "optimizer_restored": optimizer_restored,
            "host_restore": host_report,
        }

    # ----------------------------------------------------------------- internals

    def _register_core(self) -> None:
        try:
            self.host.register_core(self.core_id)
        except DuplicateCoreError:
            # Already registered (for example by an earlier loop on this host).
            pass

    def _checkpoint_dir(self, tag: str) -> Path:
        return self.state_root / "e0" / _safe_component(tag)

    def _resolve_split(self, episode: dict, split: Optional[str]) -> str:
        if split is None:
            declared = episode.get("split")
            split = declared if isinstance(declared, str) and declared else "train"
        if split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}; got {split!r}")
        return split

    def _episode_index(self, episode: dict) -> Optional[int]:
        for index, candidate in enumerate(self.episodes):
            if candidate.get("episode_id") == episode.get("episode_id"):
                return index
        return None

    def _resume_index(self, episode: dict, plan: list[tuple[str, str, int, int]]) -> int:
        """Where to resume this episode, or 0 for a fresh walk.

        The saved ``(step_index, char_position)`` pair must agree with this
        plan's entry: a cursor left by a different plan (for example a finished
        inference run of a shorter plan) therefore never resumes a training
        walk in the wrong place.
        """

        episode_index = self._cursor.get("episode_index")
        if episode_index is None or isinstance(episode_index, bool):
            return 0
        if not 0 <= episode_index < len(self.episodes):
            return 0
        if self.episodes[episode_index].get("episode_id") != episode.get("episode_id"):
            return 0
        step_index = self._cursor.get("step_index", 0)
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            return 0
        if not 0 < step_index < len(plan):
            return 0
        char_position = self._cursor.get("char_position", 0)
        if (
            isinstance(char_position, bool)
            or not isinstance(char_position, int)
            or char_position < 0
        ):
            return 0
        if plan[step_index][3] != char_position:
            return 0
        return step_index

    def _walk_plan(
        self, episode: dict, respond_ids: list[int]
    ) -> list[tuple[str, str, int, int]]:
        """The flat, deterministic step plan of one episode walk.

        Each entry is ``(phase, kind, char_id, phase_char_position)``:
        ``kind`` is ``"reset"`` or ``"char"``, and ``phase_char_position`` is the
        index of that character inside its own phase (for a reset step, the
        number of characters the phase has consumed so far).
        """

        plan: list[tuple[str, str, int, int]] = []
        observe_chars = 0
        for step in episode["steps"]:
            kind = step["kind"]
            if kind == "reset":
                plan.append((PHASE_OBSERVE, "reset", 0, observe_chars))
            elif kind == "tick":
                plan.append((PHASE_OBSERVE, "char", EMPTY_ID, observe_chars))
                observe_chars += 1
            else:
                for char_id in encode_ids(step["text"]):
                    plan.append((PHASE_OBSERVE, "char", char_id, observe_chars))
                    observe_chars += 1
        for char_id in encode_ids(episode["query"]):
            plan.append((PHASE_OBSERVE, "char", char_id, observe_chars))
            observe_chars += 1
        for position, char_id in enumerate(respond_ids):
            plan.append((PHASE_RESPOND, "char", char_id, position))
        # A separate terminal decision cannot swallow the last character.
        plan.append((PHASE_RESPOND, "terminal", EMPTY_ID, len(respond_ids)))
        return plan

    def _phase_counts(self, plan: list[tuple[str, str, int, int]]) -> dict[str, int]:
        """The phase sizes this mode runs, derived from the episode itself.

        Derived (not counted as the run goes) so a resumed run reports the same
        numbers as an uninterrupted one.
        """

        observe = sum(
            1 for phase, kind, _char, _pos in plan if phase == PHASE_OBSERVE and kind == "char"
        )
        respond = sum(1 for phase, _kind, _char, _pos in plan if phase == PHASE_RESPOND)
        return {"observe": observe, "respond": respond}

    def _decision(self, phase: str, position: int, control_logits: torch.Tensor) -> int:
        """The control decision for one tick: the core's argmax, or the seam."""

        override = self.control_override
        if override is not None:
            forced = override(phase, position, control_logits)
            if forced is not None:
                if (
                    isinstance(forced, bool)
                    or not isinstance(forced, int)
                    or forced not in (CONTROL_WAIT, CONTROL_COMMIT, CONTROL_END)
                ):
                    raise ValueError(
                        "control_override must return None or one of "
                        f"(0, 1, 2); got {forced!r}"
                    )
                return int(forced)
        return int(torch.argmax(control_logits, dim=-1).item())

    def _observe_tick(self, char_id: int, *, train: bool) -> None:
        """Context only in both modes; retain the graph when training memory."""

        ids = torch.tensor([char_id], dtype=torch.long, device=self.device)
        new_reasoning, new_response, char_logits, control_logits = self.core.step(
            ids, self.reasoning_state, self.response_state
        )
        decision = self._decision(
            PHASE_OBSERVE, int(self._cursor["char_position"]), control_logits
        )
        self.reasoning_state = new_reasoning if train else new_reasoning.detach()
        self.response_state = new_response if train else new_response.detach()
        self.last_control = decision

    def _respond_tick(
        self,
        char_id: int,
        position: int,
        total: int,
        optimizer: Optional[Any],
        *,
        terminal_control: Optional[int] = None,
    ) -> None:
        """Predict the current target from the PREVIOUS character, then stage it."""

        is_terminal = position == total
        draft = self._draft_text()
        previous = EMPTY_ID if position == 0 else encode_ids(draft[-1])[0]
        ids = torch.tensor([previous], dtype=torch.long, device=self.device)
        new_reasoning, new_response, char_logits, control_logits = self.core.step(
            ids, self.reasoning_state, self.response_state
        )
        decision = self._decision(PHASE_RESPOND, position, control_logits)

        target_control = (terminal_control if total == 0 and terminal_control is not None else CONTROL_END) if is_terminal else CONTROL_COMMIT
        control_target = torch.tensor(
            [target_control],
            dtype=torch.long,
            device=self.device,
        )
        loss = F.cross_entropy(control_logits, control_target)
        if not is_terminal:
            char_target = torch.tensor([char_id], dtype=torch.long, device=self.device)
            loss = loss + F.cross_entropy(char_logits, char_target)
        self._losses.append(loss)
        sample = float(loss.detach())
        training = self._training
        if training is not None:
            training["loss_samples"].append(sample)
            training["final_loss"] = sample

        self.reasoning_state = new_reasoning
        self.response_state = new_response
        self.last_control = decision
        self._generation["response_ticks"] += 1

        # The draft receives the EXPECTED character through the same
        # private-draft + consolidator-commit path inference uses.
        if not is_terminal:
            self._emit_character(char_id)
        else:
            self._generation["termination"] = "wait" if target_control == CONTROL_WAIT else "end"
            if target_control == CONTROL_END:
                self.host.submit_control(self.core_id, "end")
                self._episode_ended = True

    def _generate_tick(self, position: int) -> None:
        """An answer-independent tick, including feedback from the exact draft."""
        draft = self._draft_text()
        previous = encode_ids(draft[-1])[0] if draft else EMPTY_ID
        ids = torch.tensor([previous], dtype=torch.long, device=self.device)
        reasoning, response, chars, controls = self.core.step(ids, self.reasoning_state, self.response_state)
        self.reasoning_state, self.response_state = reasoning.detach(), response.detach()
        self.last_control = self._decision(PHASE_RESPOND, position, controls)
        self._generation["response_ticks"] = position + 1
        if self.last_control == CONTROL_COMMIT:
            self._emit_character(int(torch.argmax(chars, dim=-1).item()))
        elif self.last_control == CONTROL_END:
            self.host.submit_control(self.core_id, "end")
            self._episode_ended = True
            self._generation["termination"] = "end"
        # WAIT mutates recurrent state but produces zero draft/Heart operations.

    def _replay_training_prefix(self, episode: dict, prefix: list, respond_ids: list[int]) -> None:
        """Rebuild gradient history without repeating any canonical mutations."""
        self.reasoning_state, self.response_state = self.core.initial_states(1, self.device)
        # The previous input for each response is known from shifted targets;
        # replay must never temporarily overwrite the host's restored draft.
        with torch.enable_grad():
            for phase, kind, char_id, position in prefix:
                if kind == "reset":
                    self.reasoning_state, self.response_state = self.core.initial_states(1, self.device)
                    continue
                previous = char_id if phase == PHASE_OBSERVE else (EMPTY_ID if position == 0 else respond_ids[position - 1])
                ids = torch.tensor([previous], dtype=torch.long, device=self.device)
                r, s, chars, controls = self.core.step(ids, self.reasoning_state, self.response_state)
                self.reasoning_state, self.response_state = r, s
                if phase == PHASE_RESPOND:
                    control = torch.tensor([CONTROL_COMMIT], dtype=torch.long, device=self.device)
                    target = torch.tensor([char_id], dtype=torch.long, device=self.device)
                    loss = F.cross_entropy(chars, target) + F.cross_entropy(controls, control)
                    self._losses.append(loss)
                    self._training["loss_samples"].append(float(loss.detach()))
                    self._generation["response_ticks"] += 1

    def _emit_character(self, char_id: int) -> None:
        """Append one native character to the draft and commit it if it is new."""

        if not 0 <= char_id < ALPHABET_SIZE:
            return  # EMPTY proposes no character: nothing to append or commit
        text = native_char(char_id)
        current = self.host.private_draft()
        combined = (current["text"] if current is not None else "") + text
        self.host.set_draft(self.core_id, combined)
        if combined == self.host.committed_text(LogicalRegion.RESPONSE_DRAFT):
            # The canonical field already carries exactly this text: committing
            # it again would be a no-op write.
            return
        proposal_id = self.host.submit_control(self.core_id, "commit")
        ack = self.host.commit(self.host.current_consolidator(), proposal_id)
        self.commits += 1
        self.last_commit_id = ack.commit_id

    def _draft_text(self) -> str:
        draft = self.host.private_draft()
        return draft["text"] if draft is not None else ""

    def _cursor_block(self) -> dict[str, Any]:
        """The structured ``axon-lab-execution-cursor-v1`` block."""

        return {
            "schema": CURSOR_SCHEMA,
            "next": {
                "episode_index": self._cursor.get("episode_index"),
                "step_index": int(self._cursor.get("step_index", 0)),
                "char_position": int(self._cursor.get("char_position", 0)),
                "emission_count": len(self._draft_text()),
            },
            "dataset": dict(self._dataset),
            "last_control": self.last_control,
            "episode_ended": bool(self._episode_ended),
        }

    def _validate_cursor_block(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "next",
            "dataset",
            "last_control",
            "episode_ended",
        }:
            raise ValueError("structured cursor block has an unexpected shape")
        if value["schema"] != CURSOR_SCHEMA:
            raise ValueError(
                f"unsupported structured cursor schema {value['schema']!r}"
            )
        next_block = value["next"]
        if not isinstance(next_block, dict) or set(next_block) != {
            "episode_index",
            "step_index",
            "char_position",
            "emission_count",
        }:
            raise ValueError("structured cursor 'next' block has an unexpected shape")
        for name in ("step_index", "char_position", "emission_count"):
            item = next_block[name]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"structured cursor {name} must be a non-negative int")
        episode_index = next_block["episode_index"]
        if episode_index is not None and (
            isinstance(episode_index, bool) or not isinstance(episode_index, int)
        ):
            raise ValueError("structured cursor episode_index must be null or an int")
        dataset = value["dataset"]
        if not isinstance(dataset, dict) or set(dataset) != {
            "preset",
            "curriculum_version",
            "curriculum_sha256",
            "split_manifest_sha256",
        }:
            raise ValueError("structured cursor dataset block has an unexpected shape")
        for name, item in dataset.items():
            if item is not None and (not isinstance(item, str) or not item):
                raise ValueError(
                    f"structured cursor dataset {name} must be null or a non-empty string"
                )
        last_control = value["last_control"]
        if last_control is not None and (
            isinstance(last_control, bool)
            or not isinstance(last_control, int)
            or last_control not in (CONTROL_WAIT, CONTROL_COMMIT, CONTROL_END)
        ):
            raise ValueError("structured cursor last_control must be null or 0/1/2")
        if not isinstance(value["episode_ended"], bool):
            raise ValueError("structured cursor episode_ended must be a bool")
        return dict(value)

    def _save_rng_artifacts(self, directory: Path) -> None:
        rng_dir = directory / RNG_DIRNAME
        rng_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Any] = {
            "torch_cpu": _torch_rng_state_b64(),
            "numpy": _numpy_rng_state_json(),
            "python": _py_rng_state_json(),
        }
        for index in range(torch.cuda.device_count()):
            artifacts[f"torch_cuda_{index}"] = _torch_cuda_rng_state_b64(index)
        for name, state in sorted(artifacts.items()):
            path = rng_dir / name
            payload = json.dumps(
                {"schema": RNG_SCHEMA, "kind": name, "state": state},
                ensure_ascii=False,
                sort_keys=True,
            )
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)

    def _restore_rng_artifacts(self, directory: Path, cursor: dict) -> None:
        """Restore every RNG artifact present, else the flat legacy keys."""

        restored: set[str] = set()
        rng_dir = directory / RNG_DIRNAME
        if rng_dir.is_dir():
            restorers = {
                "torch_cpu": _restore_torch_rng_state,
                "numpy": _restore_numpy_rng_state,
                "python": _restore_py_rng_state,
            }
            for name in sorted(restorers):
                path = rng_dir / name
                if path.is_file():
                    restorers[name](self._rng_artifact_state(path, name))
                    restored.add(name)
            for index in range(torch.cuda.device_count()):
                name = f"torch_cuda_{index}"
                path = rng_dir / name
                if path.is_file():
                    _restore_torch_cuda_rng_state(
                        index, self._rng_artifact_state(path, name)
                    )
                    restored.add(name)
        fallbacks = (
            ("torch_cpu", "torch_rng_state", _restore_torch_rng_state),
            ("numpy", "numpy_rng_state", _restore_numpy_rng_state),
            ("python", "py_rng_state", _restore_py_rng_state),
        )
        for name, flat_key, restorer in fallbacks:
            if name not in restored and flat_key in cursor:
                restorer(cursor[flat_key])

    @staticmethod
    def _rng_artifact_state(path: Path, expected_kind: str) -> Any:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"rng artifact is unreadable: {path}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != RNG_SCHEMA
            or payload.get("kind") != expected_kind
            or "state" not in payload
        ):
            raise ValueError(f"rng artifact has an unexpected shape: {path}")
        return payload["state"]

    def _results_row(
        self,
        episode: dict,
        split: str,
        plan: list[tuple[str, str, int, int]],
        *,
        train: bool,
    ) -> dict:
        prediction_text = self._draft_text()
        row = {
            "episode_id": episode["episode_id"],
            "family": episode["family"],
            "split": split,
            "curriculum_version": self.curriculum_version,
            "curriculum_sha256": self.curriculum_sha256,
            "difficulty": episode["difficulty"],
            "prediction_text": prediction_text,
            # A nonempty response is a content COMMIT; its closing END is
            # reported separately in generation. No expected label is consulted.
            "prediction_control": CONTROL_COMMIT if prediction_text else self.last_control,
            "metrics": evaluate_episode(episode, prediction_text, CONTROL_COMMIT if prediction_text else self.last_control),
            "seed": episode["seed"],
            "architecture_id": self.architecture_id,
            "checkpoint_id": self.checkpoint_id,
        }
        row["phases"] = self._phase_counts(plan)
        row["phases"]["respond"] = self._generation["response_ticks"]
        row["generation"] = dict(self._generation)
        if train:
            training = self._training or {
                "optimizer_steps": 0,
                "loss_samples": deque(),
                "final_loss": None,
            }
            samples = [float(value) for value in training["loss_samples"]]
            row["training"] = {
                "optimizer_steps": int(training["optimizer_steps"]),
                "loss_samples": samples,
                "final_loss": samples[-1] if samples else None,
                "objective_loss": training.get("objective_loss"),
            }
        if tuple(row)[: len(RESULTS_FIELDS)] != RESULTS_FIELDS:
            raise RuntimeError(
                "results row does not start with curriculum RESULTS_FIELDS: "
                f"{tuple(row)[: len(RESULTS_FIELDS)]!r} != {RESULTS_FIELDS!r}"
            )
        extras = set(row) - set(RESULTS_FIELDS)
        if extras not in ({"phases", "generation"}, {"phases", "generation", "training"}):
            raise RuntimeError(f"results row carries unexpected extra fields {extras!r}")
        self.last_training = row.get("training")
        return row


__all__ = [
    "ARCHITECTURE_ID",
    "COMMIT",
    "CORE_CHECKPOINT_FILENAME",
    "CURSOR_FILENAME",
    "CURSOR_SCHEMA",
    "DEFAULT_RESPONSE_TICK_BUDGET",
    "DEFAULT_CORE_ID",
    "E0Loop",
    "EXECUTION_PROTOCOL",
    "END",
    "HOST_CHECKPOINT_FILENAME",
    "LOSS_SAMPLE_CAP",
    "OPTIMIZER_CHECKPOINT_FILENAME",
    "PHASE_OBSERVE",
    "PHASE_RESPOND",
    "RNG_DIRNAME",
    "RNG_SCHEMA",
    "SPLITS",
    "WAIT",
]
