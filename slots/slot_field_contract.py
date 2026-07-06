"""Axon slot field contract: ten regions, masking, materialization.

The shared field is divided into ten fixed regions. Each region holds a
contiguous sequence of 8192D slots. The field contract defines region names,
their slot allocations, masking for training (unused regions are masked so
the trainer controls exactly what the core can attend), and materialization
from container records into slot vectors and back.

A surfacing budget is a VIEW over the field, never a mutation. Budgets are
explicit, recorded, and tested — they never silently drop state.

Self-test: `python slots/slot_field_contract.py` round-trips a small field
with all ten regions and prints PASS/FAIL.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any,  Sequence

import numpy as np

# Ensure repo root is importable when run as a script
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from substrate import char_to_slot, get_letter_bank  # noqa: E402

try:
    from .slot_spec import (  # noqa: E402
        SLOT_WIDTH,
        Slot,
        ControlBlock,
        pack_slot,
        unpack_slot,
        pack_text_chain,
        unpack_chain,
        pack_region,
        unpack_region,
        KIND_TEXT,
        KIND_RESPONSE_DRAFT,
        KIND_CONTROL,
        KIND_EDGE_CONT,
        STATUS_ACTIVE,
        STATUS_DRAFT,
        STATUS_COMMITTED,
        STATUS_EMPTY,
        EDGE_FORM_NONE,
        format_edge_full,
    )
except ImportError:
    from slot_spec import (  # noqa: E402
        SLOT_WIDTH,
        Slot,
        ControlBlock,
        pack_slot,
        unpack_slot,
        pack_text_chain,
        unpack_chain,
        pack_region,
        unpack_region,
        KIND_TEXT,
        KIND_RESPONSE_DRAFT,
        KIND_CONTROL,
        KIND_EDGE_CONT,
        STATUS_ACTIVE,
        STATUS_DRAFT,
        STATUS_COMMITTED,
        STATUS_EMPTY,
        EDGE_FORM_NONE,
        format_edge_full,
    )

# --------------------------------------------------------------------------- #
# Region names — the locked ten
# --------------------------------------------------------------------------- #

REGION_NAMES = (
    "conversation_history",
    "user_input",
    "response_draft",
    "structured_knowledge",
    "tool_results",
    "scratch",
    "diary",
    "awareness",
    "task_state",
    "control",
)

# Context annotation is a tenth region (R6) that may be added later
REGION_CONTEXT_ANNOTATIONS = "context_annotations"

N_REGIONS = len(REGION_NAMES)

# --------------------------------------------------------------------------- #
# Field layout
# --------------------------------------------------------------------------- #

@dataclass
class FieldLayout:
    """Slot allocation for each region in the shared field.

    The field is a (total_slots, SLOT_WIDTH) array. Each region gets a
    contiguous slice. Regions may have different slot capacities.
    """
    region_slots: dict[str, int]

    def __post_init__(self) -> None:
        for name in REGION_NAMES:
            if name not in self.region_slots:
                raise ValueError(f"missing region allocation: {name}")
        total = sum(self.region_slots.values())
        self._offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in REGION_NAMES:
            n = self.region_slots[name]
            self._offsets[name] = (offset, offset + n)
            offset += n
        self.total_slots = total

    def region_slice(self, name: str) -> slice:
        """Return the slot slice [start:end) for a region name."""
        start, end = self._offsets[name]
        return slice(start, end)

    def region_bounds(self, name: str) -> tuple[int, int]:
        return self._offsets[name]

    @property
    def total_dim(self) -> int:
        return self.total_slots * SLOT_WIDTH


def default_layout() -> FieldLayout:
    """Default slot allocation for the ten regions.

    These numbers are starting defaults, not architecture-locked. They can
    be tuned per deployment. The key constraint is that the total field fits
    in CPU memory and the matmul budget stays light.
    """
    return FieldLayout(region_slots={
        "conversation_history": 16,   # 16 slots x 256 chars = ~4k chars
        "user_input": 4,              # current input for this tick
        "response_draft": 8,           # 8 slots for iterative draft
        "structured_knowledge": 12,   # 12 slots for surfaced facts
        "tool_results": 8,             # 8 slots for tool output
        "scratch": 8,                  # 8 slots for reasoning scratch
        "diary": 4,                    # 4 slots for reflections
        "awareness": 4,                # 4 slots for situational context
        "task_state": 4,               # 4 slots for task tracking
        "control": 2,                  # 2 slots for control/status
    })


# --------------------------------------------------------------------------- #
# Field — the full shared state
# --------------------------------------------------------------------------- #

@dataclass
class SlotField:
    """The active shared field: a (total_slots, SLOT_WIDTH) array + layout."""
    layout: FieldLayout
    data: np.ndarray  # (total_slots, SLOT_WIDTH) float32

    def __post_init__(self) -> None:
        expected = (self.layout.total_slots, SLOT_WIDTH)
        if self.data.shape != expected:
            raise ValueError(f"field data must be {expected}, got {self.data.shape}")
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

    @classmethod
    def empty(cls, layout: FieldLayout | None = None) -> "SlotField":
        layout = layout or default_layout()
        data = np.zeros((layout.total_slots, SLOT_WIDTH), dtype=np.float32)
        return cls(layout=layout, data=data)

    def get_region(self, name: str) -> np.ndarray:
        """Return the slot array for one region (N_region, SLOT_WIDTH)."""
        sl = self.layout.region_slice(name)
        return self.data[sl]

    def set_region(self, name: str, slots: list[Slot]) -> None:
        """Write a list of Slot objects into a region. Extra slots are truncated."""
        sl = self.layout.region_slice(name)
        max_n = sl.stop - sl.start
        for i, slot in enumerate(slots[:max_n]):
            self.data[sl.start + i] = slot.vector

    def get_region_slots(self, name: str) -> list[Slot]:
        """Unpack a region's slots into Slot objects."""
        sl = self.layout.region_slice(name)
        return unpack_region(self.data[sl])

    def clear_region(self, name: str) -> None:
        """Zero out a region."""
        sl = self.layout.region_slice(name)
        self.data[sl] = 0.0

    def mask(self, active_regions: set[str]) -> np.ndarray:
        """Return a boolean mask (total_slots,) — True for active regions.

        Used by trainers to mask off regions the core should not attend to.
        Inactive regions are zeroed in the mask, not in the data — the data
        is preserved; only attention is masked.
        """
        mask = np.zeros(self.layout.total_slots, dtype=bool)
        for name in active_regions:
            sl = self.layout.region_slice(name)
            mask[sl] = True
        return mask

    def all_mask(self) -> np.ndarray:
        """All regions active."""
        return np.ones(self.layout.total_slots, dtype=bool)

    def none_mask(self) -> np.ndarray:
        """No regions active (full mask off)."""
        return np.zeros(self.layout.total_slots, dtype=bool)


# --------------------------------------------------------------------------- #
# Materialization from container records
# --------------------------------------------------------------------------- #

@dataclass
class ContainerRecord:
    """A minimal container record for materialization into slots.

    In production, this is the container schema from curator/container_schema.py.
    This is a lightweight version for the field contract module.
    """
    kind: str = KIND_TEXT
    text: str = ""
    edges: list[str] = field(default_factory=list)  # list of formatted edge strings
    status: str = STATUS_ACTIVE
    edge_form: str = EDGE_FORM_NONE


def materialize_container(container: ContainerRecord, region_kind: str = "") -> list[Slot]:
    """Materialize a container record into slot(s).

    Text is packed into the text payload. Edges are joined and packed into
    the edge payload. Long text chains across slots. Dense edges chain
    edge-continuation slots.
    """
    edges_str = " ".join(container.edges) if container.edges else ""
    kind = region_kind or container.kind

    return pack_text_chain(
        text=container.text,
        edges=edges_str,
        kind=kind,
        status=container.status,
        edge_form=container.edge_form if edges_str else EDGE_FORM_NONE,
    )


def dematerialize_slots(slots: Sequence[Slot]) -> ContainerRecord:
    """Reconstruct a container record from a chain of slots."""
    text, edges = unpack_chain(slots)
    edge_list = edges.split(" ") if edges else []

    # Determine status and kind from the first non-continuation slot
    kind = KIND_TEXT
    status = STATUS_ACTIVE
    for slot in slots:
        if slot.control.kind != KIND_EDGE_CONT:
            kind = slot.control.kind
            status = slot.control.status
            break

    return ContainerRecord(
        kind=kind,
        text=text,
        edges=edge_list,
        status=status,
    )


# --------------------------------------------------------------------------- #
# Surfacing budget — a view, never a mutation
# --------------------------------------------------------------------------- #

@dataclass
class SurfacingBudget:
    """A budget for how many slots can be surfaced into a region.

    A budget is a VIEW over the full state. It limits how much is surfaced
    into active state, but it NEVER mutates or drops the underlying dormant
    state. Excess items are skipped and counted, not silently dropped.

    The count of skipped items is recorded for auditability.
    """
    max_slots: int
    surfaced: int = 0
    skipped: int = 0

    def can_surface(self) -> bool:
        return self.surfaced < self.max_slots

    def surface_one(self) -> bool:
        """Returns True if one more slot can be surfaced."""
        if self.surfaced < self.max_slots:
            self.surfaced += 1
            return True
        self.skipped += 1
        return False

    def skip_one(self) -> None:
        """Record a skipped item (over capacity)."""
        self.skipped += 1

    @property
    def report(self) -> str:
        return f"surfaced={self.surfaced}, skipped={self.skipped}, budget={self.max_slots}"


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def selftest() -> bool:
    """Round-trip a small field with all ten regions."""
    all_ok = True
    checks: list[tuple[str, bool]] = []

    # 1. Empty field creation
    layout = default_layout()
    field = SlotField.empty(layout)
    ok = field.data.shape == (layout.total_slots, SLOT_WIDTH)
    checks.append(("empty_field_shape", ok))
    all_ok &= ok

    # 2. All ten regions present
    ok = all(name in field.layout.region_slots for name in REGION_NAMES)
    checks.append(("all_regions_present", ok))
    all_ok &= ok

    # 3. Write and read a region
    slots_in = [
        pack_slot(text="Hello Axon.", kind=KIND_TEXT, status=STATUS_ACTIVE),
        pack_slot(text="How are you?", kind=KIND_TEXT, status=STATUS_ACTIVE),
    ]
    field.set_region("conversation_history", slots_in)
    slots_out = field.get_region_slots("conversation_history")
    ok = len(slots_out) >= 2 and slots_out[0].text == "Hello Axon." and slots_out[1].text == "How are you?"
    checks.append(("region_write_read", ok))
    all_ok &= ok

    # 4. Region masking
    mask = field.mask({"conversation_history", "response_draft"})
    conv_sl = field.layout.region_slice("conversation_history")
    draft_sl = field.layout.region_slice("response_draft")
    scratch_sl = field.layout.region_slice("scratch")
    ok = (mask[conv_sl].all() and mask[draft_sl].all() and not mask[scratch_sl].any())
    checks.append(("region_masking", ok))
    all_ok &= ok

    # 5. Materialize a container
    container = ContainerRecord(
        kind=KIND_TEXT,
        text="Dogs are loyal animals.",
        edges=[format_edge_full("is_a", "animal")],
        status=STATUS_ACTIVE,
        edge_form="F",
    )
    slots = materialize_container(container)
    recovered = dematerialize_slots(slots)
    ok = recovered.text == "Dogs are loyal animals." and len(recovered.edges) == 1
    checks.append(("materialize_roundtrip", ok))
    all_ok &= ok

    # 6. Surfacing budget
    budget = SurfacingBudget(max_slots=3)
    results = [budget.surface_one() for _ in range(5)]
    ok = results == [True, True, True, False, False] and budget.surfaced == 3 and budget.skipped == 2
    checks.append(("surfacing_budget", ok))
    all_ok &= ok

    # 7. Clear region
    field.clear_region("conversation_history")
    slots_after = field.get_region_slots("conversation_history")
    ok = all(s.control.kind == " " for s in slots_after) or all(s.text == "" for s in slots_after)
    checks.append(("clear_region", ok))
    all_ok &= ok

    # 8. Response draft region
    draft_slots = [pack_slot(text="I am Axon.", kind=KIND_RESPONSE_DRAFT, status=STATUS_DRAFT)]
    field.set_region("response_draft", draft_slots)
    out = field.get_region_slots("response_draft")
    ok = out[0].text == "I am Axon." and out[0].control.kind == KIND_RESPONSE_DRAFT
    checks.append(("response_draft_region", ok))
    all_ok &= ok

    # Print results
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")

    print()
    if all_ok:
        print(f"slot_field_contract self-test: PASS ({len(checks)} checks)")
    else:
        failed = [n for n, ok in checks if not ok]
        print(f"slot_field_contract self-test: FAIL ({len(failed)} failed: {failed})")
    return all_ok




# Salvaged from the rescinded write-head organ (convener order, 2026-07-03):
# diff-only commits survive doctrinally and belong to the field contract.

def commit_diff(
    decoded_slot_text: str,
    current_slot_text: str,
    length: int,
) -> list[dict[str, Any]]:
    """Pure-function diff: emit typed delta ops that touch only changed spans.

    The active prefix is ``decoded_slot_text[:length]``. Positions at or past
    ``length`` are treated as padding/cleared. The resulting ``update_span``
    records have ``start``, ``end``, and ``text``. An empty ``text`` means a
    deletion/clear of that span in the current slot.

    No silent truncation: every position that differs between the decoded
    active text and the current slot is represented in a delta.
    """
    active = (decoded_slot_text + " " * length)[:length]
    max_len = max(length, len(current_slot_text))
    deltas: list[dict[str, Any]] = []
    i = 0
    while i < max_len:
        a = active[i] if i < length else ""
        b = current_slot_text[i] if i < len(current_slot_text) else ""
        if a == b:
            i += 1
            continue
        start = i
        while i < max_len:
            a2 = active[i] if i < length else ""
            b2 = current_slot_text[i] if i < len(current_slot_text) else ""
            if a2 == b2:
                break
            i += 1
        end = i
        deltas.append({"update_span": {"start": start, "end": end, "text": active[start:end]}})
    return deltas


if __name__ == "__main__":
    ok = selftest()
    sys.exit(0 if ok else 1)
