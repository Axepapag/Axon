"""Core-owned PyTorch adapter for the E0 two-state organism.

E0 is the smallest organism in the ladder: one character in, one character
proposed out, with two pieces of recurrent state (a *reasoning* state and a
*response* state) plus a three-way control decision (WAIT / COMMIT / END).

LAW NOTES (non-negotiable):

* ``native_cells`` is the frozen 1024D lane bank itself, copied into a
  registered, non-trainable buffer. It is the substrate, not a learned
  projection of it: nothing in this module retunes, regenerates, re-derives,
  or learns the 96 codes. Changing a weight here can never change a cell.
* The surface this adapter feeds the core is built mechanically from
  character ids: lane 0 holds the character, lanes 1..lanes-1 hold EMPTY.
  That is occupancy rung 0 of Jeff's one-character-at-a-time ladder.
* Anything outside id range 0..95 fails closed with ``ValueError``. Ids are
  never clamped, wrapped, converted, or approximated.
* The core only *proposes* (character logits and a control logit triple).
  It never writes canonical state; the Heart is the sole writer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn

from substrate.substrate_1024 import lane_bank

#: Row ids of the frozen lane bank: 0..94 native characters, 95 EMPTY.
ALPHABET_SIZE = 95
EMPTY_ID = 95
ID_COUNT = 96

LANE_DIM = 16
SURFACE_WIDTHS = (256, 512, 768, 1024, 2048)

# Control decision triple produced by ``control_head``.
CONTROL_WAIT = 0
CONTROL_COMMIT = 1
CONTROL_END = 2

CHECKPOINT_FORMAT = "axon-e0-checkpoint-v1"

_DEFAULT_CONFIG: dict[str, Any] = {"width": 512, "hidden_size": 512, "max_chunk": 64}


class E0TwoStateCore(torch.nn.Module):
    """Two-state E0 core over the frozen 1024D lane substrate.

    ``native_cells`` (96, 16) float32 is the sealed lane bank registered as a
    buffer with ``requires_grad=False``: this IS the frozen substrate copied in
    for indexing, never a learned projection of it, and no optimizer will ever
    update it. The only learned parts are the two GRU cells and the two heads.

    The core proposes; it never writes canonical state.
    """

    def __init__(self, width: int = 512, hidden_size: int = 512, max_chunk: int = 64) -> None:
        super().__init__()
        if width not in SURFACE_WIDTHS:
            raise ValueError(
                f"width must be one of {SURFACE_WIDTHS} (a whole number of 16-wide lanes); got {width!r}"
            )
        if int(hidden_size) <= 0:
            raise ValueError(f"hidden_size must be positive; got {hidden_size!r}")
        if int(max_chunk) <= 0:
            raise ValueError(f"max_chunk must be positive; got {max_chunk!r}")

        self.width = int(width)
        self.hidden_size = int(hidden_size)
        self.max_chunk = int(max_chunk)
        self.lanes = self.width // LANE_DIM
        self.lane_dim = LANE_DIM

        bank = torch.tensor(lane_bank(), dtype=torch.float32)
        if tuple(bank.shape) != (ID_COUNT, LANE_DIM):
            raise ValueError(f"frozen lane bank has the wrong shape: {tuple(bank.shape)}")
        # Registered so it travels with .to()/.cuda(), pinned so it never trains.
        self.register_buffer("native_cells", bank, persistent=True)
        self.native_cells.requires_grad_(False)

        self.reasoning = torch.nn.GRUCell(self.width, self.hidden_size)
        self.response = torch.nn.GRUCell(self.hidden_size, self.hidden_size)
        self.char_head = torch.nn.Linear(self.hidden_size, ID_COUNT)
        self.control_head = torch.nn.Linear(self.hidden_size, 3)

    def config(self) -> dict[str, Any]:
        """Constructor arguments, suitable for ``save_checkpoint(config=...)``."""
        return {"width": self.width, "hidden_size": self.hidden_size, "max_chunk": self.max_chunk}

    def initial_states(self, batch: int, device: Optional[torch.device] = None) -> tuple:
        """Two zero states of shape (batch, hidden_size); no RNG, no warm start."""
        if int(batch) <= 0:
            raise ValueError(f"batch must be positive; got {batch!r}")
        if device is None:
            device = self.native_cells.device
        zeros = torch.zeros((int(batch), self.hidden_size), dtype=torch.float32, device=device)
        return zeros, zeros.clone()

    def surface_from_ids(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Character ids -> exact occupancy-rung-0 surface of shape (B, width).

        Lane 0 of each row is ``native_cells[char_id]``; lanes 1..lanes-1 are the
        EMPTY row (``native_cells[95]``). This is occupancy rung 0 of Jeff's
        one-character-at-a-time ladder; wider occupancy fills the prefix lanes
        with real characters instead of EMPTY. The layout is mechanical and
        reversible: no projection, no learned embedding, no tolerance.

        Fails closed with ``ValueError`` on any id outside 0..95. Ids are never
        clamped, wrapped, or converted.
        """
        self._check_ids(char_ids)
        cells = self.native_cells.to(device=char_ids.device)
        surface = cells[EMPTY_ID].expand(char_ids.shape[0], self.lanes, self.lane_dim).clone()
        surface[:, 0] = cells[char_ids]
        return surface.reshape(char_ids.shape[0], self.width)

    def step(
        self,
        char_ids: torch.Tensor,
        reasoning_state: torch.Tensor,
        response_state: torch.Tensor,
    ) -> tuple:
        """One character tick: (new_reasoning, new_response, char_logits, control_logits)."""
        self._check_ids(char_ids)
        batch = char_ids.shape[0]
        self._check_state(reasoning_state, "reasoning_state", batch, char_ids.device)
        self._check_state(response_state, "response_state", batch, char_ids.device)

        surface = self.surface_from_ids(char_ids)
        new_reasoning = self.reasoning(surface, reasoning_state)
        new_response = self.response(new_reasoning, response_state)
        char_logits = self.char_head(new_response)
        control_logits = self.control_head(new_response)
        return new_reasoning, new_response, char_logits, control_logits

    def _check_ids(self, char_ids: torch.Tensor) -> None:
        if not isinstance(char_ids, torch.Tensor):
            raise TypeError(f"char_ids must be a torch.Tensor; got {type(char_ids).__name__}")
        if char_ids.ndim != 1:
            raise ValueError(f"char_ids must be 1-D (B,); got shape {tuple(char_ids.shape)}")
        if char_ids.dtype == torch.bool or char_ids.dtype.is_floating_point or char_ids.dtype.is_complex:
            raise ValueError(f"char_ids must be a non-boolean integer tensor; got dtype {char_ids.dtype}")
        if char_ids.numel() == 0:
            raise ValueError("char_ids must hold at least one id")
        low = int(char_ids.min())
        high = int(char_ids.max())
        if low < 0 or high >= ID_COUNT:
            raise ValueError(
                f"character ids must be in 0..{ID_COUNT - 1} ({ALPHABET_SIZE} native + EMPTY); "
                f"got id range {low}..{high}. Failing closed: ids are never clamped or converted."
            )

    def _check_state(
        self, state: torch.Tensor, name: str, batch: int, device: torch.device
    ) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor; got {type(state).__name__}")
        if tuple(state.shape) != (batch, self.hidden_size):
            raise ValueError(
                f"{name} must have shape ({batch}, {self.hidden_size}); got {tuple(state.shape)}"
            )
        if state.dtype != torch.float32:
            raise ValueError(f"{name} must be float32; got {state.dtype}")
        if state.device != device:
            raise ValueError(f"{name} is on {state.device} but char_ids are on {device}")


def save_checkpoint(
    path: Any,
    core: E0TwoStateCore,
    reasoning_state: torch.Tensor,
    response_state: torch.Tensor,
    config: dict,
) -> Path:
    """Write a checkpoint plus its ``<file>.sha256`` sidecar.

    States are stored detached: a checkpoint carries values, never an autograd
    graph. The sidecar holds the digest of the written file bytes in the
    standard ``"<hex>  <filename>"`` form.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict; got {type(config).__name__}")
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": core.state_dict(),
        "reasoning_state": reasoning_state.detach().clone(),
        "response_state": response_state.detach().clone(),
        "config": dict(config),
    }
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return path


def load_checkpoint(
    path: Any,
    map_location: Any = None,
) -> tuple:
    """Verify, load, and rebuild: (core, reasoning_state, response_state, config).

    The sidecar digest is checked against the file bytes before anything is
    unpickled; a mismatch raises ``ValueError``.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"checkpoint not found: {path}")
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"checkpoint has no sha256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip().lower()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError(
            f"checkpoint sha256 mismatch for {path}: sidecar says {expected}, file hashes to {actual}"
        )

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        # No unsafe fallback: weights_only=True refuses any type outside its
        # allowlist, and our payloads (tensors plus plain str/int config) are
        # always inside it, so a failure here means a foreign or corrupt file.
        # The sha256 sidecar proves byte integrity, not trustworthy provenance;
        # it never justifies weights_only=False.
        raise ValueError(f"checkpoint could not be read: {path} ({exc})") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a dict; got {type(payload).__name__}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint format must be {CHECKPOINT_FORMAT!r}; got {payload.get('format')!r}"
        )
    for key in ("state_dict", "reasoning_state", "response_state", "config"):
        if key not in payload:
            raise ValueError(f"checkpoint is missing {key!r}: {path}")

    core = E0TwoStateCore(**_constructor_kwargs(payload["config"]))
    core.load_state_dict(payload["state_dict"])
    return core, payload["reasoning_state"], payload["response_state"], payload["config"]


def _constructor_kwargs(config: Any) -> dict[str, Any]:
    """Constructor kwargs from a saved config; unknown keys are carried through, not passed."""
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config must be a dict; got {type(config).__name__}")
    kwargs = {key: config[key] for key in _DEFAULT_CONFIG if key in config}
    if "width" not in kwargs or "hidden_size" not in kwargs:
        raise ValueError(f"checkpoint config must carry width and hidden_size; got {sorted(config)}")
    return kwargs
