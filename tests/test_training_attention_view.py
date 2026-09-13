from __future__ import annotations

import json

import numpy as np
import pytest

from runtime.field import (
    D64FieldCompiler,
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
    StaleCompiledFieldError,
    canonical_json_bytes,
    canonical_sha256,
)
from runtime.field.training_view import (
    TRAINING_VIEW_SCHEMA,
    TrainingAttentionView,
    TrainingAttentionViewCompiler,
    TrainingHistoryWindow,
    TrainingViewError,
)
from substrate import transport_token_cell16


def _responses_state(core_id: str, attempts: list[str], others: list[str] | None = None) -> RegionState:
    spans = [
        FieldSpan(
            span_id=f"attempt:{core_id}:{index}",
            text=text,
            kind="training_attempt",
            source=core_id,
            provenance="heart:committed-attempt",
        )
        for index, text in enumerate(attempts)
    ]
    for index, text in enumerate(others or ()):
        spans.append(
            FieldSpan(
                span_id=f"feedback:trainer:{index}",
                text=text,
                kind="trainer_feedback",
                source="trainer",
                provenance="heart:committed-feedback",
            )
        )
    return RegionState(name=LogicalRegion.TRAINING_RESPONSES, spans=tuple(spans))


def _snapshot(
    attempts: list[str] | None = None,
    *,
    core_id: str = "core-alpha",
    assignment: str = "Predict the next character after 'ABC'.",
    others: list[str] | None = None,
    extra_regions: dict[str, str] | None = None,
    tick_id: int = 7,
) -> SharedFieldSnapshot:
    regions: list[RegionState] = [
        RegionState.from_text(
            LogicalRegion.TRAINER_INSTRUCTIONS,
            assignment,
            source="trainer",
            provenance="heart:committed-assignment",
        ),
        _responses_state(core_id, attempts or [], others),
    ]
    for name, text in (extra_regions or {}).items():
        regions.append(RegionState.from_text(name, text))
    return SharedFieldSnapshot(tick_id=tick_id, regions=tuple(regions))


def _compile(snapshot: SharedFieldSnapshot, **kwargs) -> TrainingAttentionView:
    options = {"core_id": "core-alpha", "cohort_core_ids": ("core-alpha",)}
    options.update(kwargs)
    return TrainingAttentionViewCompiler().compile(snapshot, **options)


def test_history_window_none_attends_assignment_but_no_responses() -> None:
    snapshot = _snapshot(["first attempt", "second attempt"])
    view = _compile(snapshot, history_window=TrainingHistoryWindow.none())
    assert view.compiled.region_text("trainer_instructions") == "Predict the next character after 'ABC'."
    assert view.compiled.region_text("training_responses") == ""
    assert view.valid_addresses and all(
        address.region is LogicalRegion.TRAINER_INSTRUCTIONS for address in view.valid_addresses
    )
    view.verify(snapshot)


def test_history_window_all_attends_every_core_attempt() -> None:
    snapshot = _snapshot(["alpha one", "alpha two"])
    view = _compile(snapshot, history_window=TrainingHistoryWindow.all())
    assert view.compiled.region_text("training_responses") == "alpha onealpha two"
    assert view.policy_for("training_responses").kind == "last_n_spans"
    assert view.policy_for("training_responses").limit == 2
    view.verify(snapshot)


def test_history_window_latest_n_attends_only_newest_attempts() -> None:
    snapshot = _snapshot(["one", "two", "three", "four"])
    view = _compile(snapshot, history_window=TrainingHistoryWindow.latest_n(2))
    assert view.compiled.region_text("training_responses") == "threefour"
    assert view.policy_for("training_responses").limit == 2
    view.verify(snapshot)


def test_history_window_latest_n_exceeding_history_attends_everything() -> None:
    snapshot = _snapshot(["one", "two"])
    view = _compile(snapshot, history_window=TrainingHistoryWindow.latest_n(10))
    assert view.compiled.region_text("training_responses") == "onetwo"
    view.verify(snapshot)


def test_history_window_over_empty_responses_compiles() -> None:
    snapshot = _snapshot([])
    for window in (
        TrainingHistoryWindow.none(),
        TrainingHistoryWindow.all(),
        TrainingHistoryWindow.latest_n(3),
    ):
        view = _compile(snapshot, history_window=window)
        assert view.compiled.region_text("training_responses") == ""
        view.verify(snapshot)


def test_history_window_excludes_other_cores_and_trainer_feedback() -> None:
    core_id = "core-alpha"
    snapshot = _snapshot(["alpha attempt"], core_id=core_id)
    spans = [
        FieldSpan(span_id="attempt:core-beta:0", text="beta attempt ", source="core-beta"),
        FieldSpan(span_id="feedback:trainer:0", text="trainer critique ", source="trainer"),
        FieldSpan(span_id="feedback:trainer:1", text="trainer receipt ", source="trainer"),
        *snapshot.region("training_responses").spans,
    ]
    snapshot = SharedFieldSnapshot(
        tick_id=snapshot.tick_id,
        regions=(
            snapshot.region("trainer_instructions"),
            RegionState(name=LogicalRegion.TRAINING_RESPONSES, spans=tuple(spans)),
        ),
    )
    view = _compile(snapshot, core_id=core_id, history_window=TrainingHistoryWindow.all())
    assert view.compiled.region_text("training_responses") == "alpha attempt"
    assert "beta attempt" not in view.compiled.region_text("training_responses")
    assert "trainer critique" not in view.compiled.region_text("training_responses")
    view.verify(snapshot)


def test_non_trailing_attempt_window_fails_closed() -> None:
    core_id = "core-alpha"
    snapshot = _snapshot(["alpha attempt"], core_id=core_id)
    spans = list(snapshot.region("training_responses").spans)
    spans.append(FieldSpan(span_id="feedback:trainer:0", text="late feedback", source="trainer"))
    snapshot = SharedFieldSnapshot(
        tick_id=snapshot.tick_id,
        regions=(
            snapshot.region("trainer_instructions"),
            RegionState(name=LogicalRegion.TRAINING_RESPONSES, spans=tuple(spans)),
        ),
    )
    with pytest.raises(TrainingViewError, match="trailing span set"):
        _compile(snapshot, core_id=core_id, history_window=TrainingHistoryWindow.all())


def test_only_training_regions_are_attended() -> None:
    snapshot = _snapshot(
        ["prior attempt"],
        extra_regions={"user_input": "user secret", "cortex": "cortex secret", "diary": "diary secret"},
    )
    view = _compile(snapshot, history_window=TrainingHistoryWindow.all())
    for region in ("user_input", "cortex", "diary"):
        assert view.compiled.region_text(region) == ""
        assert view.compiled.region_addresses(region) == ()
    assert {address.region for address in view.valid_addresses} == {
        LogicalRegion.TRAINER_INSTRUCTIONS,
        LogicalRegion.TRAINING_RESPONSES,
    }
    # Masking changed attendance, never the canonical body.
    assert snapshot.region("user_input").text == "user secret"
    view.verify(snapshot)


def test_identical_specs_are_byte_identical() -> None:
    snapshot = _snapshot(["one", "two", "three"])
    window = TrainingHistoryWindow.latest_n(2)
    first = _compile(snapshot, history_window=window)
    second = _compile(snapshot, history_window=TrainingHistoryWindow.latest_n(2))
    assert first.view_id == second.view_id
    assert first.mask_policy_id == second.mask_policy_id
    assert first.compiled.rail_id == second.compiled.rail_id
    assert first.compiled.rows.tobytes() == second.compiled.rows.tobytes()
    assert first.address_mapping == second.address_mapping
    assert first.to_canonical_dict() == second.to_canonical_dict()


def test_view_id_binds_source_tick_and_every_spec_field() -> None:
    snapshot = _snapshot(["one", "two"])
    baseline = _compile(snapshot, history_window=TrainingHistoryWindow.all())
    successor = SharedFieldSnapshot(
        tick_id=snapshot.tick_id + 1,
        regions=tuple(snapshot.regions),
    )
    variants = [
        _compile(successor, history_window=TrainingHistoryWindow.all()),
        _compile(snapshot, history_window=TrainingHistoryWindow.none()),
        _compile(snapshot, history_window=TrainingHistoryWindow.latest_n(1)),
        _compile(
            snapshot,
            history_window=TrainingHistoryWindow.all(),
            core_id="core-beta",
            cohort_core_ids=("core-alpha", "core-beta"),
        ),
        _compile(snapshot, history_window=TrainingHistoryWindow.all(), cohort_core_ids=("core-alpha", "core-beta")),
    ]
    for variant in variants:
        assert variant.view_id != baseline.view_id
    assert baseline.view_id == canonical_sha256(baseline._identity_dict())
    assert baseline.to_canonical_dict()["schema"] == TRAINING_VIEW_SCHEMA


def test_source_snapshot_is_unchanged_after_compile() -> None:
    snapshot = _snapshot(["one", "two"], extra_regions={"user_input": "hello"})
    before = canonical_json_bytes(snapshot.to_canonical_dict())
    before_hash = snapshot.canonical_hash
    regions_before = snapshot.regions
    _compile(snapshot, history_window=TrainingHistoryWindow.all())
    assert snapshot.canonical_hash == before_hash
    assert canonical_json_bytes(snapshot.to_canonical_dict()) == before
    assert snapshot.regions == regions_before
    # The canonical unmasked compile still roundtrips the untouched source.
    D64FieldCompiler().compile(snapshot).verify_roundtrip(snapshot)


def test_verify_proves_every_row_dereferences_to_canonical_addresses() -> None:
    snapshot = _snapshot(["réponse ✓"], extra_regions={"user_input": "context"})
    view = _compile(snapshot, history_window=TrainingHistoryWindow.all())
    view.verify(snapshot)
    for address in view.valid_addresses:
        state = snapshot.region(address.region)
        span = next(item for item in state.spans if item.span_id == address.span_id)
        assert state.text[address.region_position] == address.character
        assert span.text[address.span_position] == address.character
        assert span.source == address.source
        cell = view.compiled.lane_cell16(address.row_index, address.lane_index)
        assert np.array_equal(cell, transport_token_cell16(address.transport_token_id))


def test_address_mapping_roundtrip_through_receipts() -> None:
    snapshot = _snapshot(["attempt text"])
    view = _compile(snapshot, history_window=TrainingHistoryWindow.all())
    for address in view.valid_addresses:
        receipt = view.dereference(address, snapshot)
        assert receipt == address
        assert receipt.region is address.region
        assert receipt.span_id == address.span_id
        assert receipt.region_position == address.region_position
    tampered = view.dereference(view.valid_addresses[0], snapshot)
    forged = tampered.__class__(**{**tampered.to_canonical_dict(), "character": "Z"})
    with pytest.raises(TrainingViewError):
        view.dereference(forged, snapshot)


def test_verify_fails_closed_on_stale_snapshot() -> None:
    snapshot = _snapshot(["one"])
    view = _compile(snapshot, history_window=TrainingHistoryWindow.all())
    successor = SharedFieldSnapshot(
        tick_id=snapshot.tick_id + 1,
        regions=tuple(snapshot.regions),
    )
    with pytest.raises(StaleCompiledFieldError):
        view.verify(successor)
    with pytest.raises(StaleCompiledFieldError):
        view.compiled.assert_fresh(successor)


def test_verify_detects_attended_text_tampering() -> None:
    core_id = "core-alpha"
    snapshot = _snapshot(["one", "two"], core_id=core_id)
    view = _compile(snapshot, core_id=core_id, history_window=TrainingHistoryWindow.all())
    swapped = SharedFieldSnapshot(
        tick_id=snapshot.tick_id,
        regions=(
            snapshot.region("trainer_instructions"),
            RegionState(
                name=LogicalRegion.TRAINING_RESPONSES,
                spans=tuple(reversed(snapshot.region("training_responses").spans)),
            ),
        ),
    )
    # Reordering attempts changes the canonical hash, so freshness fails first.
    assert swapped.canonical_hash != view.source_canonical_hash
    with pytest.raises((TrainingViewError, StaleCompiledFieldError)):
        view.verify(swapped)


def test_compile_rejects_schema_without_training_regions() -> None:
    legacy = SharedFieldSnapshot(
        tick_id=0,
        regions=(RegionState.from_text("user_input", "hello"),),
        schema_version="shared-field-v3",
    )
    assert legacy.schema_version == "shared-field-v3"
    with pytest.raises(TrainingViewError, match="no training regions"):
        _compile(legacy)


def test_malformed_specs_fail_closed() -> None:
    snapshot = _snapshot(["one"])
    with pytest.raises(ValueError):
        TrainingHistoryWindow(kind="bogus")
    with pytest.raises(ValueError):
        TrainingHistoryWindow(kind="latest_n", limit=0)
    with pytest.raises(ValueError):
        TrainingHistoryWindow(kind="none", limit=3)
    with pytest.raises(TypeError):
        TrainingHistoryWindow(kind="latest_n", limit="2")
    with pytest.raises(ValueError):
        _compile(snapshot, core_id="")
    with pytest.raises(ValueError):
        _compile(snapshot, cohort_core_ids=("core-beta",))
    with pytest.raises(TypeError):
        _compile(snapshot, cohort_core_ids="core-alpha")
    with pytest.raises(TypeError):
        _compile(snapshot, history_window="latest_n")


def test_view_serializes_to_canonical_json() -> None:
    snapshot = _snapshot(["one", "two"])
    view = _compile(snapshot, history_window=TrainingHistoryWindow.latest_n(1))
    payload = view.to_canonical_dict()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["view_id"] == view.view_id
    assert decoded["source_field_id"] == snapshot.field_id
    assert decoded["source_tick_id"] == snapshot.tick_id
    assert decoded["core_id"] == "core-alpha"
    assert decoded["history_window"] == {"kind": "latest_n", "limit": 1}
    assert decoded["mask_policy_id"] == view.mask_policy_id
    assert decoded["rail_id"] == view.compiled.rail_id


def test_view_dataclass_rejects_mismatched_compiled_source() -> None:
    first = _snapshot(["one"])
    second = _snapshot(["two"])
    compiled = (
        TrainingAttentionViewCompiler()
        .compile(
            first,
            core_id="core-alpha",
            cohort_core_ids=("core-alpha",),
        )
        .compiled
    )
    with pytest.raises(TrainingViewError):
        TrainingAttentionView(
            source_field_id=second.field_id,
            source_tick_id=second.tick_id,
            source_schema_version=second.schema_version,
            source_canonical_hash=second.canonical_hash,
            core_id="core-alpha",
            cohort_core_ids=("core-alpha",),
            history_window=TrainingHistoryWindow.none(),
            region_policies=(),
            mask_policy_id="x" * 64,
            compiled=compiled,
        )
