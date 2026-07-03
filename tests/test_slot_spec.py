"""Lane 0: Slot integrity suite — blocking gate for all future training.

Tests pack/unpack exact text, exact edge payload both forms, chain
reconstruction (text and edge chains), overlength skip-and-count, invalid
symbol rejection, control-block decode, region masking, materialization
round-trip, and adapter gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from substrate import char_to_slot, default_alphabet, get_letter_bank
from slots.slot_spec import (
    SLOT_WIDTH,
    Slot,
    ControlBlock,
    pack_slot,
    unpack_slot,
    pack_text_chain,
    unpack_chain,
    pack_region,
    unpack_region,
    format_edge_full,
    format_edge_alias,
    format_edges,
    KIND_TEXT,
    KIND_RESPONSE_DRAFT,
    KIND_EDGE_CONT,
    KIND_EMPTY,
    STATUS_ACTIVE,
    STATUS_DRAFT,
    EDGE_FORM_FULL,
    EDGE_FORM_ALIAS,
    EDGE_FORM_NONE,
    EDGE_FORM_MIXED,
    MAX_TEXT_CHARS,
    MAX_EDGE_CHARS,
)
from slots.slot_field_contract import (
    REGION_NAMES,
    FieldLayout,
    default_layout,
    SlotField,
    ContainerRecord,
    materialize_container,
    dematerialize_slots,
    SurfacingBudget,
)


# -- Slot spec: pack/unpack exact text -- #

class TestSlotSpecText:
    def test_simple_roundtrip(self):
        text = "The dog is a good boy."
        slot = pack_slot(text=text)
        decoded = unpack_slot(slot.vector)
        assert decoded.text == text

    def test_empty_text(self):
        slot = pack_slot(text="")
        decoded = unpack_slot(slot.vector)
        assert decoded.text == ""

    def test_max_text_length(self):
        text = "A" * MAX_TEXT_CHARS
        slot = pack_slot(text=text)
        decoded = unpack_slot(slot.vector)
        assert decoded.text == text

    def test_all_alphabet_chars(self):
        alphabet = default_alphabet()
        text = "".join(alphabet)
        # Repeat to fill if needed
        text = text * (MAX_TEXT_CHARS // len(alphabet))
        slot = pack_slot(text=text[:MAX_TEXT_CHARS])
        decoded = unpack_slot(slot.vector)
        assert decoded.text == text[:MAX_TEXT_CHARS]

    def test_slot_width(self):
        slot = pack_slot(text="hello")
        assert slot.vector.shape == (SLOT_WIDTH,)
        assert slot.vector.dtype == np.float32


# -- Slot spec: edge payload -- #

class TestSlotSpecEdges:
    def test_full_edge_roundtrip(self):
        edges = format_edge_full("is_a", "animal")
        slot = pack_slot(text="dog", edges=edges, edge_form=EDGE_FORM_FULL)
        decoded = unpack_slot(slot.vector)
        assert decoded.edges == edges

    def test_alias_edge_roundtrip(self):
        edges = format_edge_alias("AA")
        slot = pack_slot(text="dog", edges=edges, edge_form=EDGE_FORM_ALIAS)
        decoded = unpack_slot(slot.vector)
        assert decoded.edges == edges

    def test_multiple_edges(self):
        edges = " ".join([
            format_edge_full("is_a", "animal"),
            format_edge_full("has_property", "furry"),
            format_edge_alias("K9"),
        ])
        slot = pack_slot(text="dog", edges=edges, edge_form=EDGE_FORM_MIXED)
        decoded = unpack_slot(slot.vector)
        assert decoded.edges == edges

    def test_no_edges(self):
        slot = pack_slot(text="dog")
        decoded = unpack_slot(slot.vector)
        assert decoded.edges == ""
        assert decoded.control.edge_form == EDGE_FORM_NONE


# -- Slot spec: chain reconstruction -- #

class TestSlotSpecChains:
    def test_text_chain_roundtrip(self):
        long_text = "This is a long paragraph. " * 15
        slots = pack_text_chain(long_text)
        full_text, _ = unpack_chain(slots)
        assert full_text == long_text

    def test_edge_chain_overflow(self):
        dense_edges = " ".join(format_edge_full("is_a", f"cat{i}") for i in range(30))
        slots = pack_text_chain("dog", edges=dense_edges, edge_form=EDGE_FORM_FULL)
        _, full_edges = unpack_chain(slots)
        assert full_edges == dense_edges

    def test_chain_metadata(self):
        long_text = "A" * 300
        slots = pack_text_chain(long_text)
        assert len(slots) == 2
        assert slots[0].control.chain_index == 0
        assert slots[1].control.chain_index == 1
        assert all(s.control.chain_total == 2 for s in slots)


# -- Slot spec: invalid input rejection -- #

class TestSlotSpecValidation:
    def test_invalid_text_rejected(self):
        with pytest.raises(ValueError, match="outside the substrate alphabet"):
            pack_slot(text="hello\tworld")

    def test_invalid_edge_rejected(self):
        with pytest.raises(ValueError, match="outside the substrate alphabet"):
            pack_slot(text="dog", edges="{is_a:animal}")


# -- Slot spec: control block -- #

class TestSlotSpecControl:
    def test_control_roundtrip(self):
        ctrl = ControlBlock(
            kind=KIND_RESPONSE_DRAFT, length=42, chain_index=1,
            chain_total=3, status=STATUS_DRAFT, edge_form=EDGE_FORM_ALIAS,
        )
        ctrl_str = ctrl.to_control_chars()
        ctrl2 = ControlBlock.from_control_chars(ctrl_str)
        assert ctrl2.kind == ctrl.kind
        assert ctrl2.length == ctrl.length
        assert ctrl2.chain_index == ctrl.chain_index
        assert ctrl2.chain_total == ctrl.chain_total
        assert ctrl2.status == ctrl.status
        assert ctrl2.edge_form == ctrl.edge_form

    def test_control_length_matches_text(self):
        slot = pack_slot(text="hello", kind=KIND_TEXT)
        assert slot.control.length == 5


# -- Slot spec: region pack/unpack -- #

class TestSlotSpecRegion:
    def test_region_roundtrip(self):
        slots_in = [pack_slot(text="hello"), pack_slot(text="world")]
        arr = pack_region(slots_in)
        slots_out = unpack_region(arr)
        assert len(slots_out) == 2
        assert slots_out[0].text == "hello"
        assert slots_out[1].text == "world"

    def test_empty_region(self):
        arr = pack_region([])
        assert arr.shape == (0, SLOT_WIDTH)


# -- Field contract: regions and masking -- #

class TestSlotFieldContract:
    def test_all_nine_regions(self):
        layout = default_layout()
        for name in REGION_NAMES:
            assert name in layout.region_slots

    def test_field_shape(self):
        layout = default_layout()
        field = SlotField.empty(layout)
        assert field.data.shape == (layout.total_slots, SLOT_WIDTH)

    def test_region_write_read(self):
        field = SlotField.empty()
        slots = [pack_slot(text="Hello Axon.", kind=KIND_TEXT)]
        field.set_region("conversation_history", slots)
        out = field.get_region_slots("conversation_history")
        assert out[0].text == "Hello Axon."

    def test_region_masking(self):
        field = SlotField.empty()
        mask = field.mask({"conversation_history", "response_draft"})
        conv_sl = field.layout.region_slice("conversation_history")
        scratch_sl = field.layout.region_slice("scratch")
        assert mask[conv_sl].all()
        assert not mask[scratch_sl].any()

    def test_materialize_roundtrip(self):
        container = ContainerRecord(
            kind=KIND_TEXT,
            text="Dogs are loyal animals.",
            edges=[format_edge_full("is_a", "animal")],
            edge_form=EDGE_FORM_FULL,
        )
        slots = materialize_container(container)
        recovered = dematerialize_slots(slots)
        assert recovered.text == "Dogs are loyal animals."
        assert len(recovered.edges) == 1

    def test_surfacing_budget(self):
        budget = SurfacingBudget(max_slots=3)
        results = [budget.surface_one() for _ in range(5)]
        assert results == [True, True, True, False, False]
        assert budget.surfaced == 3
        assert budget.skipped == 2

    def test_clear_region(self):
        field = SlotField.empty()
        field.set_region("scratch", [pack_slot(text="temp")])
        field.clear_region("scratch")
        slots = field.get_region_slots("scratch")
        assert all(s.text == "" for s in slots)


# -- Adapter gates -- #

class TestSlotAdapter:
    def test_snap_idempotence_64(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=64)
        report = adapter.check_snap_idempotence()
        assert report.passed, f"{report.n_changed} slots changed after snap"

    def test_snap_idempotence_128(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=128)
        report = adapter.check_snap_idempotence()
        assert report.passed

    def test_snap_idempotence_256(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=256)
        report = adapter.check_snap_idempotence()
        assert report.passed

    def test_separability_64(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=64)
        report = adapter.check_separability()
        assert report.passed, f"{report.n_collisions} collisions"

    def test_separability_128(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=128)
        report = adapter.check_separability()
        assert report.passed

    def test_separability_256(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=256)
        report = adapter.check_separability()
        assert report.passed

    def test_projection_shape(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=64)
        slot = pack_slot(text="hello")
        proj = adapter.project_down(slot.vector)
        assert proj.shape == (64,)

    def test_batch_projection_shape(self):
        from adapters.slot_adapter import SlotAdapter
        adapter = SlotAdapter(d_model=128)
        slots = np.stack([pack_slot(text="hello").vector,
                          pack_slot(text="world").vector])
        proj = adapter.project_down(slots)
        assert proj.shape == (2, 128)