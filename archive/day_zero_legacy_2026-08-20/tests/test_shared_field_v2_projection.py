from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from runtime.axon_runtime.config import load_runtime_config
from runtime.axon_runtime.identity_v2_config import (
    load_identity_v2_contract,
    parse_identity_v2_contract,
)
from runtime.field.schema_v2 import (
    CANONICAL_REGION_ORDER_V2,
    FieldSpanV2,
    IdentityCharterV2,
    LogicalRegionV2,
    RegionStateV2,
    RegionVisibilityV2,
    SharedFieldSnapshotV2,
)
from runtime.field.serde_v2 import serialize_shared_field_v2
from runtime.field import view_v2
from runtime.field.view_v2 import (
    CONTEXT_END_V2,
    PROPOSAL_START_V2,
    USER_END_V2,
    IdentityAwareFieldViewV2,
    V2ProjectionError,
    V2ReadCursor,
    V2SlotKind,
    audit_v2_read_cycle_coverage,
    collect_identity_aware_v2_read_cycle,
    compile_identity_aware_v2_read_page,
)
from substrate import default_alphabet, get_letter_bank


CONTRACT_PATH = Path("ops/axon_runtime.identity-v2.contract.json")
R2_CONTRACT_PATH = Path("ops/axon_runtime.identity-v2.contract.r2.json")
SMOKE_CONFIG_PATH = Path("ops/axon_runtime.cpu-smoke.json")
SHORT_IDENTITY = {
    "core_id": "a",
    "display_name": "a",
    "model_label": "d",
    "role_capabilities": ("p",),
    "current_role": "p",
}


@pytest.fixture
def contract():
    return load_identity_v2_contract(CONTRACT_PATH)


def _successor(
    contract,
    texts: dict[LogicalRegionV2, str] | None = None,
    *,
    visibility: dict[LogicalRegionV2, RegionVisibilityV2] | None = None,
    source_manifest_ids: tuple[str, ...] | None = None,
) -> SharedFieldSnapshotV2:
    texts = texts or {}
    visibility = visibility or {}
    base = contract.build_genesis_snapshot()
    regions: list[RegionStateV2] = []
    for region in base.regions:
        text = texts.get(region.name)
        if text is None:
            regions.append(region)
            continue
        spans = (
            FieldSpanV2(
                span_id=f"{region.name.value}:fixture",
                text=text,
                source="projection-test",
                provenance=f"{region.name.value}:fixture",
            ),
        )
        regions.append(
            RegionStateV2(
                name=region.name,
                spans=spans,
                visibility=visibility.get(region.name, region.visibility),
                write_policy=region.write_policy,
            )
        )
    return SharedFieldSnapshotV2(
        tick_id=1,
        regions=tuple(regions),
        parent_field_id=base.field_id,
        source_manifest_ids=(
            base.source_manifest_ids
            if source_manifest_ids is None
            else source_manifest_ids
        ),
    )


def _page(contract, snapshot, **overrides):
    kwargs = {**SHORT_IDENTITY, **overrides}
    return compile_identity_aware_v2_read_page(contract, snapshot, **kwargs)


def _raw_ref_keys(page: IdentityAwareFieldViewV2) -> list[tuple[str, str, int]]:
    return [
        (
            ref.logical_region.value,
            ref.span_id,
            ref.span_char_index,
        )
        for ref in page.slot_refs
        if (
            ref.kind is V2SlotKind.SPAN
            and ref.logical_region is not None
            and ref.span_id is not None
            and ref.span_char_index is not None
        )
    ]


def test_valid_genesis_is_deterministic_immutable_and_checkpoint_compatible(contract) -> None:
    snapshot = contract.build_genesis_snapshot()
    before = serialize_shared_field_v2(snapshot)
    first = _page(contract, snapshot)
    second = _page(contract, snapshot)

    assert first.view_hash == second.view_hash
    assert first.field16.shape == (384, 16)
    assert np.all(first.role_ids[:CONTEXT_END_V2] == 0)
    assert np.all(first.role_ids[CONTEXT_END_V2:USER_END_V2] == 1)
    assert np.all(first.role_ids[PROPOSAL_START_V2:] == 2)
    assert not np.any(first.write_mask[:PROPOSAL_START_V2])
    assert np.all(first.write_mask[PROPOSAL_START_V2:])
    assert np.all(first.attention_mask[PROPOSAL_START_V2:])
    assert np.any(first.logical_region_ids[:CONTEXT_END_V2] == 10)
    assert serialize_shared_field_v2(snapshot) == before
    assert not Path(r"D:\Axon\State\axon_runtime_identity_v2").exists()

    with pytest.raises(ValueError):
        first.field16[0, 0] = 1.0
    with pytest.raises(ValueError):
        first.role_ids.setflags(write=True)


def test_identity_envelope_is_context_derived_not_raw_charter(contract) -> None:
    page = _page(contract, contract.build_genesis_snapshot())
    derived = [
        ref for ref in page.slot_refs if ref.kind is V2SlotKind.DERIVED_IDENTITY
    ]
    raw_identity = [
        ref
        for ref in page.slot_refs
        if ref.kind is V2SlotKind.SPAN
        and ref.logical_region is LogicalRegionV2.IDENTITY
    ]

    assert derived
    assert not raw_identity
    assert all(ref.role.value == 0 for ref in derived)
    assert all(ref.logical_region is LogicalRegionV2.IDENTITY for ref in derived)
    assert [ref.identity_envelope_char_index for ref in derived] == list(
        range(len(derived))
    )
    assert all(ref.span_id is None for ref in derived)
    assert any(
        omission.logical_region is LogicalRegionV2.IDENTITY
        and omission.reason == "context_region_not_selected"
        for omission in page.omissions
    )


def test_full_95_character_substrate_roundtrips_selected_span(contract) -> None:
    alphabet = "".join(default_alphabet())
    snapshot = _successor(
        contract,
        {LogicalRegionV2.CONVERSATION_HISTORY: alphabet},
    )
    page = _page(contract, snapshot)
    refs = [
        ref
        for ref in page.slot_refs
        if ref.kind is V2SlotKind.SPAN
        and ref.logical_region is LogicalRegionV2.CONVERSATION_HISTORY
    ]
    assert [ref.span_char_index for ref in refs] == list(range(95))
    rows = page.field16[[ref.slot_index for ref in refs]]
    assert get_letter_bank().decode_sequence(rows, blanks_as="") == alphabet
    assert not any(
        omission.logical_region is LogicalRegionV2.CONVERSATION_HISTORY
        for omission in page.omissions
    )


def test_complete_cycle_covers_every_attended_raw_character_once(contract) -> None:
    texts = {
        LogicalRegionV2.CONVERSATION_HISTORY: "history-" * 45,
        LogicalRegionV2.USER_INPUT: "question-" * 14,
        LogicalRegionV2.STRUCTURED_KNOWLEDGE: "fact-" * 31,
        LogicalRegionV2.SITUATION_AWARENESS: "situation-" * 15,
        LogicalRegionV2.TOOL_RESULTS: "tool-" * 20,
        LogicalRegionV2.ADVISOR_INPUT: "advisor-" * 15,
        LogicalRegionV2.TASK_STATE: "task-" * 20,
        LogicalRegionV2.SCRATCH: "scratch-" * 18,
        LogicalRegionV2.RESPONSE_DRAFT: "draft-" * 22,
        LogicalRegionV2.DIARY: "diary-" * 25,
    }
    snapshot = _successor(contract, texts)
    pages = collect_identity_aware_v2_read_cycle(
        contract,
        snapshot,
        **SHORT_IDENTITY,
    )
    coverage = audit_v2_read_cycle_coverage(contract, snapshot, pages)

    assert len(pages) > 4
    assert pages[-1].read_cycle_complete
    assert coverage.passed
    assert coverage.expected_characters == coverage.observed_characters
    assert coverage.duplicate_characters == 0
    assert not coverage.missing_refs
    assert any(
        ref.kind is V2SlotKind.SPAN
        and ref.logical_region is LogicalRegionV2.IDENTITY
        for page in pages
        for ref in page.slot_refs
    )


def test_cycle_finishes_when_user_outlives_all_context(contract) -> None:
    snapshot = _successor(
        contract,
        {LogicalRegionV2.USER_INPUT: "u" * 1000},
    )
    pages = collect_identity_aware_v2_read_cycle(
        contract,
        snapshot,
        **SHORT_IDENTITY,
    )
    coverage = audit_v2_read_cycle_coverage(contract, snapshot, pages)

    assert len(pages) > 4
    assert pages[-1].read_cycle_complete
    assert coverage.passed
    assert coverage.expected_characters == coverage.observed_characters


def test_cycle_finishes_when_proposal_outlives_all_context(contract) -> None:
    snapshot = _successor(
        contract,
        {LogicalRegionV2.RESPONSE_DRAFT: "p" * 1000},
    )
    pages = collect_identity_aware_v2_read_cycle(
        contract,
        snapshot,
        **SHORT_IDENTITY,
    )
    coverage = audit_v2_read_cycle_coverage(contract, snapshot, pages)

    assert len(pages) > 4
    assert pages[-1].read_cycle_complete
    assert coverage.passed
    assert coverage.expected_characters == coverage.observed_characters


def test_cycle_audit_rejects_noncanonical_cursor_sequence(contract) -> None:
    pages = collect_identity_aware_v2_read_cycle(
        contract,
        contract.build_genesis_snapshot(),
        **SHORT_IDENTITY,
    )
    with pytest.raises(V2ProjectionError, match="must start at canonical cursor"):
        audit_v2_read_cycle_coverage(
            contract,
            contract.build_genesis_snapshot(),
            pages[1:],
        )
    early_complete = replace(pages[0], read_cycle_complete=True)
    with pytest.raises(V2ProjectionError, match="completion marker"):
        audit_v2_read_cycle_coverage(
            contract,
            contract.build_genesis_snapshot(),
            (early_complete, *pages[1:]),
        )


def test_cycle_audit_requires_exact_contract_compiler_output(contract) -> None:
    snapshot = _successor(
        contract,
        {
            LogicalRegionV2.CONVERSATION_HISTORY: "a" * 300,
            LogicalRegionV2.USER_INPUT: "u" * 1000,
            LogicalRegionV2.RESPONSE_DRAFT: "p" * 1000,
        },
    )
    pages = collect_identity_aware_v2_read_cycle(
        contract,
        snapshot,
        **SHORT_IDENTITY,
    )
    assert audit_v2_read_cycle_coverage(contract, snapshot, pages).passed

    with pytest.raises(V2ProjectionError, match="parent provenance"):
        audit_v2_read_cycle_coverage(
            contract,
            snapshot,
            (replace(pages[0], source_parent_field_id=None), *pages[1:]),
        )
    with pytest.raises(V2ProjectionError, match="tick provenance"):
        audit_v2_read_cycle_coverage(
            contract,
            snapshot,
            (replace(pages[0], source_tick_id=snapshot.tick_id + 1), *pages[1:]),
        )

    fabricated_omissions = tuple(
        replace(item, reason="fabricated") for item in pages[0].omissions
    )
    with pytest.raises(V2ProjectionError, match="exact contract compiler output"):
        audit_v2_read_cycle_coverage(
            contract,
            snapshot,
            (replace(pages[0], omissions=fabricated_omissions), *pages[1:]),
        )

    raw_refs = [
        ref
        for ref in pages[0].slot_refs
        if (
            ref.kind is V2SlotKind.SPAN
            and ref.logical_region is LogicalRegionV2.CONVERSATION_HISTORY
            and ref.rendered_char == "a"
        )
    ]
    assert len(raw_refs) >= 2
    left, right = raw_refs[:2]
    altered_refs = list(pages[0].slot_refs)
    altered_refs[left.slot_index] = replace(
        left,
        span_char_index=right.span_char_index,
        region_char_index=right.region_char_index,
    )
    altered_refs[right.slot_index] = replace(
        right,
        span_char_index=left.span_char_index,
        region_char_index=left.region_char_index,
    )
    with pytest.raises(V2ProjectionError, match="exact contract compiler output"):
        audit_v2_read_cycle_coverage(
            contract,
            snapshot,
            (replace(pages[0], slot_refs=tuple(altered_refs)), *pages[1:]),
        )

    with pytest.raises(V2ProjectionError, match="exact contract compiler output"):
        audit_v2_read_cycle_coverage(
            contract,
            snapshot,
            (*pages[:-1], replace(pages[-1], next_cursor=V2ReadCursor())),
        )

    fake_contract_id = "f" * 64
    fake_manifest_snapshot = _successor(
        contract,
        {LogicalRegionV2.CONVERSATION_HISTORY: "source"},
        source_manifest_ids=(
            *contract.build_genesis_snapshot().source_manifest_ids,
            f"axon-runtime-identity-v2-contract-v1:{fake_contract_id}",
        ),
    )
    fake_pages = collect_identity_aware_v2_read_cycle(
        contract,
        fake_manifest_snapshot,
        **SHORT_IDENTITY,
    )
    with pytest.raises(V2ProjectionError, match="verified authority"):
        audit_v2_read_cycle_coverage(
            contract,
            fake_manifest_snapshot,
            tuple(replace(page, contract_id=fake_contract_id) for page in fake_pages),
        )


def test_cursor_omissions_and_masking_are_explicit(contract) -> None:
    snapshot = _successor(
        contract,
        {
            LogicalRegionV2.CONVERSATION_HISTORY: "a" * 300,
            LogicalRegionV2.USER_INPUT: "uvwxyz",
            LogicalRegionV2.RESPONSE_DRAFT: "proposal",
            LogicalRegionV2.STRUCTURED_KNOWLEDGE: "masked",
        },
        visibility={LogicalRegionV2.STRUCTURED_KNOWLEDGE: RegionVisibilityV2.MASKED},
    )
    page = _page(
        contract,
        snapshot,
        cursor=V2ReadCursor(
            context_region=LogicalRegionV2.CONVERSATION_HISTORY,
            context_offset=2,
            user_offset=1,
            proposal_offset=2,
        ),
    )
    reasons = {item.reason for item in page.omissions}
    assert {
        "context_cursor_before",
        "context_capacity",
        "user_cursor_before",
        "proposal_cursor_before",
        "region_masked",
    } <= reasons
    assert any(
        item.logical_region is LogicalRegionV2.IDENTITY
        and item.reason == "context_region_not_selected"
        for item in page.omissions
    )


def test_hash_binding_changes_with_core_role_cursor_and_source(contract) -> None:
    snapshot = _successor(
        contract,
        {LogicalRegionV2.CONVERSATION_HISTORY: "hello"},
    )
    base = _page(contract, snapshot)
    other_core = _page(contract, snapshot, core_id="b")
    other_role = _page(
        contract,
        snapshot,
        role_capabilities=("c", "p"),
        current_role="c",
    )
    other_cursor = _page(
        contract,
        snapshot,
        cursor=V2ReadCursor(context_offset=1),
    )
    other_source = _page(
        contract,
        _successor(contract, {LogicalRegionV2.CONVERSATION_HISTORY: "goodbye"}),
    )

    assert len(
        {
            base.view_hash,
            other_core.view_hash,
            other_role.view_hash,
            other_cursor.view_hash,
            other_source.view_hash,
        }
    ) == 5


def test_real_128d_consolidator_identity_over_budget_fails_closed(contract) -> None:
    with pytest.raises(V2ProjectionError, match="over budget"):
        _page(
            contract,
            contract.build_genesis_snapshot(),
            core_id="axon128-a",
            display_name="Axon 128D A",
            model_label="128D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="consolidator",
        )


def test_r2_contract_fits_all_declared_cores_without_changing_checkpoint_geometry() -> None:
    r1 = load_identity_v2_contract(CONTRACT_PATH)
    r2 = load_identity_v2_contract(R2_CONTRACT_PATH)
    runtime_config = load_runtime_config(SMOKE_CONFIG_PATH)
    model_by_id = {model.model_id: model for model in runtime_config.models}
    r2_snapshot = r2.build_genesis_snapshot()

    assert r1.envelope_char_budget == 128
    assert r2.envelope_char_budget == 146
    assert r1.contract_id != r2.contract_id
    assert not r1.state_root.exists()
    assert not r2.state_root.exists()

    real_128d_consolidator = None
    for core in runtime_config.cores:
        model = model_by_id[core.model_id]
        assert core.role_capabilities == tuple(sorted(core.role_capabilities))
        for role in core.role_capabilities:
            page = compile_identity_aware_v2_read_page(
                r2,
                r2_snapshot,
                core_id=core.core_id,
                display_name=core.display_name,
                model_label=f"{model.d_model}D",
                role_capabilities=core.role_capabilities,
                current_role=role,
            )
            assert page.field16.shape == (384, 16)
            assert set(page.role_ids.tolist()) == {0, 1, 2}
            assert np.any(page.logical_region_ids[:CONTEXT_END_V2] == 10)
            if core.core_id == "axon128-a" and role == "consolidator":
                real_128d_consolidator = page

    assert real_128d_consolidator is not None
    derived = [
        ref
        for ref in real_128d_consolidator.slot_refs
        if ref.kind is V2SlotKind.DERIVED_IDENTITY
    ]
    assert len(derived) == 146

    short_descriptor = r2.to_canonical_descriptor()
    short_descriptor["identity"]["envelope_char_budget"] = 145
    budget_145 = parse_identity_v2_contract(json.dumps(short_descriptor))
    with pytest.raises(V2ProjectionError, match="over budget"):
        compile_identity_aware_v2_read_page(
            budget_145,
            budget_145.build_genesis_snapshot(),
            core_id="axon128-a",
            display_name="Axon 128D A",
            model_label="128D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="consolidator",
        )

    alphabet = "".join(default_alphabet())
    alphabet_snapshot = _successor(
        r2,
        {LogicalRegionV2.CONVERSATION_HISTORY: alphabet},
    )
    alphabet_pages = collect_identity_aware_v2_read_cycle(
        r2,
        alphabet_snapshot,
        core_id="axon128-a",
        display_name="Axon 128D A",
        model_label="128D",
        role_capabilities=("consolidator", "proposer", "sleeper"),
        current_role="consolidator",
    )
    coverage = audit_v2_read_cycle_coverage(r2, alphabet_snapshot, alphabet_pages)
    rows_by_index = sorted(
        (
            ref.span_char_index,
            page.field16[ref.slot_index],
        )
        for page in alphabet_pages
        for ref in page.slot_refs
        if (
            ref.kind is V2SlotKind.SPAN
            and ref.logical_region is LogicalRegionV2.CONVERSATION_HISTORY
            and ref.span_char_index is not None
        )
    )
    assert coverage.passed
    assert [index for index, _ in rows_by_index] == list(range(len(alphabet)))
    assert get_letter_bank().decode_sequence(
        np.stack([row for _, row in rows_by_index]),
        blanks_as="",
    ) == alphabet

    with pytest.raises(V2ProjectionError, match="contract source manifest"):
        compile_identity_aware_v2_read_page(
            r2,
            r1.build_genesis_snapshot(),
            core_id="a",
            display_name="a",
            model_label="d",
            role_capabilities=("p",),
            current_role="p",
        )
    with pytest.raises(V2ProjectionError, match="contract source manifest"):
        compile_identity_aware_v2_read_page(
            r1,
            r2.build_genesis_snapshot(),
            core_id="a",
            display_name="a",
            model_label="d",
            role_capabilities=("p",),
            current_role="p",
        )
    assert not r1.state_root.exists()
    assert not r2.state_root.exists()


def test_hostile_snapshot_boundaries_fail_closed(contract) -> None:
    snapshot = contract.build_genesis_snapshot()
    wrong_hash = replace(snapshot)
    object.__setattr__(wrong_hash, "field_id", "0" * 64)
    with pytest.raises(V2ProjectionError, match="supplied snapshot hashes"):
        _page(contract, wrong_hash)

    class LyingString(str):
        def __ne__(self, other):  # pragma: no cover - the type check must win
            return False

    lying = replace(snapshot)
    object.__setattr__(lying, "canonical_hash", LyingString(snapshot.canonical_hash))
    with pytest.raises(V2ProjectionError, match="exact lower-case"):
        _page(contract, lying)

    forged = object.__new__(SharedFieldSnapshotV2)
    with pytest.raises(V2ProjectionError, match="missing required primitive"):
        _page(contract, forged)

    class EvilSnapshot(SharedFieldSnapshotV2):
        pass

    evil = object.__new__(EvilSnapshot)
    with pytest.raises(V2ProjectionError, match="exact SharedFieldSnapshotV2"):
        _page(contract, evil)


def test_contract_and_snapshot_identity_disagreement_fail_closed(contract) -> None:
    base = contract.build_genesis_snapshot()
    other_charter = IdentityCharterV2(
        text="A different but frozen alphabet supported identity charter."
    )
    replacement_identity = other_charter.build_identity_region()
    regions = tuple(
        replacement_identity
        if region.name is LogicalRegionV2.IDENTITY
        else region
        for region in base.regions
    )
    mismatch = SharedFieldSnapshotV2(
        tick_id=1,
        regions=regions,
        parent_field_id=base.field_id,
        source_manifest_ids=(
            other_charter.source_manifest_id,
            f"axon-runtime-identity-v2-contract-v1:{contract.contract_id}",
        ),
    )
    with pytest.raises(V2ProjectionError, match="sealed contract charter"):
        _page(contract, mismatch)

    missing_contract_manifest = SharedFieldSnapshotV2(
        tick_id=1,
        regions=base.regions,
        parent_field_id=base.field_id,
        source_manifest_ids=(contract.charter.source_manifest_id,),
    )
    with pytest.raises(V2ProjectionError, match="contract source manifest"):
        _page(contract, missing_contract_manifest)


def test_invalid_regions_cursors_and_masks_fail_closed(contract) -> None:
    snapshot = _successor(
        contract,
        {LogicalRegionV2.SCRATCH: "scratch"},
    )
    with pytest.raises(V2ProjectionError, match="cannot be user_input"):
        _page(
            contract,
            snapshot,
            cursor=V2ReadCursor(context_region=LogicalRegionV2.USER_INPUT),
        )
    with pytest.raises(V2ProjectionError, match="current proposal"):
        _page(
            contract,
            snapshot,
            cursor=V2ReadCursor(
                context_region=LogicalRegionV2.RESPONSE_DRAFT,
            ),
        )
    with pytest.raises(V2ProjectionError, match="must be scratch or response_draft"):
        _page(
            contract,
            snapshot,
            proposal_region=LogicalRegionV2.CONVERSATION_HISTORY,
        )
    with pytest.raises(V2ProjectionError, match="non-negative"):
        V2ReadCursor(context_offset=-1)
    forged_cursor = object.__new__(V2ReadCursor)
    with pytest.raises(V2ProjectionError, match="missing required primitive"):
        _page(contract, snapshot, cursor=forged_cursor)

    page = _page(contract, snapshot)
    bad_roles = page.role_ids.copy()
    bad_roles[0] = 3
    with pytest.raises(V2ProjectionError, match="0/1/2"):
        replace(page, role_ids=bad_roles)
    bad_write = page.write_mask.copy()
    bad_write[0] = True
    with pytest.raises(V2ProjectionError, match="write mask"):
        replace(page, write_mask=bad_write)
    bad_vector = page.field16.copy()
    bad_vector[0] = 0.0
    with pytest.raises(V2ProjectionError, match="slot vector"):
        replace(page, field16=bad_vector)
    with pytest.raises(V2ProjectionError, match="projection plan hash"):
        replace(page, projection_plan_hash="0" * 64)
    with pytest.raises(V2ProjectionError, match="field_id and canonical_hash"):
        replace(page, source_canonical_hash="0" * 64)

    forged_page = object.__new__(IdentityAwareFieldViewV2)
    with pytest.raises(V2ProjectionError, match="missing required fields"):
        audit_v2_read_cycle_coverage(contract, snapshot, (forged_page,))


def test_hostile_contract_and_post_construction_page_mutations_fail_closed(contract) -> None:
    snapshot = _successor(
        contract,
        {LogicalRegionV2.CONVERSATION_HISTORY: "mutation-check"},
    )
    pages = collect_identity_aware_v2_read_cycle(
        contract,
        snapshot,
        **SHORT_IDENTITY,
    )

    forged_contract = object.__new__(type(contract))
    with pytest.raises(V2ProjectionError, match="authority verification failed"):
        audit_v2_read_cycle_coverage(forged_contract, snapshot, pages)

    class EvilContract(type(contract)):
        pass

    evil_contract = object.__new__(EvilContract)
    with pytest.raises(V2ProjectionError, match="exact IdentityV2Contract"):
        audit_v2_read_cycle_coverage(evil_contract, snapshot, pages)

    bad_hash = replace(pages[0])
    object.__setattr__(bad_hash, "view_hash", "0" * 64)
    with pytest.raises(V2ProjectionError, match="view_hash"):
        audit_v2_read_cycle_coverage(contract, snapshot, (bad_hash, *pages[1:]))

    bad_field = replace(pages[0])
    field16 = bad_field.field16.copy()
    field16[0] = 0.0
    object.__setattr__(bad_field, "field16", field16)
    with pytest.raises(V2ProjectionError, match="slot vector"):
        audit_v2_read_cycle_coverage(contract, snapshot, (bad_field, *pages[1:]))


def test_module_isolated_from_v1_compiler_and_no_hidden_relaxation() -> None:
    source = inspect.getsource(view_v2)
    assert "compile_field_view" not in source
    assert "CyclingFieldViewProvider" not in source
    assert "AXON_RELAX_READ_VIEW_HASH" not in source
    assert "open(" not in source
    assert "mkdir(" not in source
    assert "FieldView" not in view_v2.__dict__


def test_source_character_partition_has_no_double_counting(contract) -> None:
    snapshot = _successor(
        contract,
        {
            LogicalRegionV2.CONVERSATION_HISTORY: "x" * 250,
            LogicalRegionV2.USER_INPUT: "user",
            LogicalRegionV2.RESPONSE_DRAFT: "answer",
        },
    )
    page = _page(contract, snapshot)
    selected = _raw_ref_keys(page)
    assert len(selected) == len(set(selected))

    omitted = set()
    for omission in page.omissions:
        state = snapshot.region(omission.logical_region)
        span = next(item for item in state.spans if item.span_id == omission.span_id)
        text = span.text[omission.span_char_start : omission.span_char_end]
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == omission.omitted_text_sha256
        keys = {
            (omission.logical_region.value, omission.span_id, index)
            for index in range(omission.span_char_start, omission.span_char_end)
        }
        assert not omitted & keys
        omitted |= keys

    all_raw = {
        (region.value, span.span_id, index)
        for region in CANONICAL_REGION_ORDER_V2
        for span in snapshot.region(region).spans
        for index, _ in enumerate(span.text)
    }
    assert not set(selected) & omitted
    assert set(selected) | omitted == all_raw
