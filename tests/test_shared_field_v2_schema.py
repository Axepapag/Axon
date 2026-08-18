from __future__ import annotations

import math

import pytest

from runtime.field.schema_v2 import (
    CHARTER_MAX_CHARS_V2,
    CORE_WRITABLE_REGIONS_V2,
    CANONICAL_REGION_ORDER_V2,
    LOGICAL_REGION_IDS_V2,
    SEALED_REGIONS_V2,
    CoreIdentityViewV2,
    FieldSpanV2,
    IdentityCharterV2,
    LogicalRegionV2,
    PhysicalRoleV2,
    RegionStateV2,
    RegionVisibilityV2,
    SharedFieldSnapshotV2,
    WritePolicyV2,
    render_core_identity_view_v2,
)


CHARTER_TEXT = (
    "You are one of Axon's many cores, not Axon as a whole.\n"
    "You attend to the shared field, propose and review evidence-bound deltas, "
    "and improve the response draft across ticks.\n"
    "You assist Jeffrey Glickman as a trusted digital collaborator under "
    "authorized policy.\n"
    "Dexter version 2 is user-supplied history.\n"
    "Tools, advisors, browsing, and account actions require separate configured "
    "policy and authorized instruction."
)


@pytest.fixture
def charter() -> IdentityCharterV2:
    return IdentityCharterV2(text=CHARTER_TEXT)


def _base_regions(charter: IdentityCharterV2) -> tuple[RegionStateV2, ...]:
    return tuple(
        RegionStateV2(name=region)
        for region in CANONICAL_REGION_ORDER_V2
        if region is not LogicalRegionV2.IDENTITY
    )


def _identity_span(
    charter: IdentityCharterV2,
    **overrides: object,
) -> FieldSpanV2:
    defaults = {
        "span_id": charter.span_id,
        "text": charter.text,
        "kind": "identity_charter",
        "source": charter.source_manifest_id,
        "provenance": "genesis",
        "confidence": 1.0,
        "container_refs": (),
        "edge_refs": (),
    }
    defaults.update(overrides)
    return FieldSpanV2(**defaults)


class TestV2RegionModel:
    def test_v2_has_eleven_ordered_regions_with_v1_ids_preserved(self) -> None:
        assert len(CANONICAL_REGION_ORDER_V2) == 11

        v1_names = [
            "conversation_history",
            "user_input",
            "structured_knowledge",
            "situation_awareness",
            "tool_results",
            "advisor_input",
            "task_state",
            "scratch",
            "response_draft",
            "diary",
        ]
        for index, expected_name in enumerate(v1_names):
            region = CANONICAL_REGION_ORDER_V2[index]
            assert region.value == expected_name
            assert LOGICAL_REGION_IDS_V2[region] == index

        identity = LogicalRegionV2.IDENTITY
        assert CANONICAL_REGION_ORDER_V2[-1] is identity
        assert LOGICAL_REGION_IDS_V2[identity] == 10

    def test_only_scratch_and_response_draft_are_core_writable(self) -> None:
        assert CORE_WRITABLE_REGIONS_V2 == {
            LogicalRegionV2.SCRATCH,
            LogicalRegionV2.RESPONSE_DRAFT,
        }
        assert SEALED_REGIONS_V2 == frozenset(
            set(CANONICAL_REGION_ORDER_V2) - CORE_WRITABLE_REGIONS_V2
        )
        assert LogicalRegionV2.IDENTITY in SEALED_REGIONS_V2

    def test_physical_roles_unchanged(self) -> None:
        assert PhysicalRoleV2.CONTEXT == 0
        assert PhysicalRoleV2.USER == 1
        assert PhysicalRoleV2.PROPOSAL == 2


class TestV2Snapshot:
    def test_genesis_binds_one_sealed_attended_identity_charter(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        snapshot = charter.build_genesis_snapshot()

        assert len(snapshot.regions) == 11
        assert tuple(r.name for r in snapshot.regions) == CANONICAL_REGION_ORDER_V2

        identity = snapshot.region(LogicalRegionV2.IDENTITY)
        assert identity.visibility is RegionVisibilityV2.ATTENDED
        assert identity.write_policy is WritePolicyV2.SEALED
        assert len(identity.spans) == 1

        span = identity.spans[0]
        assert span.text == CHARTER_TEXT
        assert span.span_id == charter.span_id
        assert span.source == charter.source_manifest_id
        assert span.kind == "identity_charter"

        assert charter.source_manifest_id in snapshot.source_manifest_ids
        assert snapshot.source_manifest_ids == (charter.source_manifest_id,)

    def test_identity_region_is_required_and_not_auto_filled(self) -> None:
        other_regions = tuple(
            RegionStateV2(name=region)
            for region in CANONICAL_REGION_ORDER_V2
            if region is not LogicalRegionV2.IDENTITY
        )
        with pytest.raises(ValueError, match="v2 snapshot requires exactly 11 regions"):
            SharedFieldSnapshotV2(tick_id=0, regions=other_regions)

    def test_duplicate_regions_rejected(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        base = list(_base_regions(charter))
        # Place identity at the first index as well as its canonical last index,
        # producing a wrong-order error that detects the duplicate name.
        duplicate_regions = (identity,) + tuple(base[1:]) + (identity,)
        with pytest.raises(ValueError, match="canonical v2 order"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=duplicate_regions,
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_unknown_region_rejected(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="unknown logical region"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=(identity,) + (RegionStateV2(name="dormant"),),
            )

    def test_mutable_identity_policy_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="always sealed in v2"):
            bad_identity = RegionStateV2(
                name=LogicalRegionV2.IDENTITY,
                spans=(_identity_span(charter),),
                write_policy=WritePolicyV2.CORE_WRITABLE,
            )
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (bad_identity,),
            )

    def test_masked_identity_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        bad_identity = RegionStateV2(
            name=LogicalRegionV2.IDENTITY,
            spans=(_identity_span(charter),),
            visibility=RegionVisibilityV2.MASKED,
        )
        with pytest.raises(ValueError, match="identity region must be attended"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (bad_identity,),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_empty_identity_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        bad_identity = RegionStateV2(
            name=LogicalRegionV2.IDENTITY,
            spans=(),
        )
        with pytest.raises(ValueError, match="exactly one charter span"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (bad_identity,),
            )

    def test_forged_span_id_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        bad_identity = RegionStateV2(
            name=LogicalRegionV2.IDENTITY,
            spans=(_identity_span(charter, span_id="forged"),),
        )
        with pytest.raises(ValueError, match="span_id does not match"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (bad_identity,),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_forged_provenance_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        bad_identity = RegionStateV2(
            name=LogicalRegionV2.IDENTITY,
            spans=(_identity_span(charter, provenance="forged"),),
        )
        with pytest.raises(ValueError, match="provenance does not match"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (bad_identity,),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_forged_confidence_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        bad_identity = RegionStateV2(
            name=LogicalRegionV2.IDENTITY,
            spans=(_identity_span(charter, confidence=0.5),),
        )
        with pytest.raises(ValueError, match="byte-for-byte"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (bad_identity,),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_forged_refs_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        bad_identity = RegionStateV2(
            name=LogicalRegionV2.IDENTITY,
            spans=(_identity_span(charter, container_refs=("ref",)),),
        )
        with pytest.raises(ValueError, match="byte-for-byte"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (bad_identity,),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_missing_charter_manifest_rejected_with_full_regions(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="missing the charter source manifest"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (identity,),
                source_manifest_ids=(),
            )

    def test_source_manifest_ids_are_strict_and_preserve_order(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        identity = charter.build_identity_region()
        manifests = ["z:1", charter.source_manifest_id, "a:2", "b:3"]
        snapshot = SharedFieldSnapshotV2(
            tick_id=0,
            regions=_base_regions(charter) + (identity,),
            source_manifest_ids=manifests,
        )
        assert list(snapshot.source_manifest_ids) == manifests

    def test_source_manifest_ids_reject_duplicates(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        identity = charter.build_identity_region()
        manifests = ["z:1", charter.source_manifest_id, "a:2", charter.source_manifest_id]
        with pytest.raises(ValueError, match="must not contain duplicates"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (identity,),
                source_manifest_ids=manifests,
            )

    def test_source_manifest_ids_reject_non_string_items(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="source_manifest_ids"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (identity,),
                source_manifest_ids=(charter.source_manifest_id, 123),
            )

    def test_field_id_is_canonical_hash(self, charter: IdentityCharterV2) -> None:
        snapshot = charter.build_genesis_snapshot()
        assert snapshot.field_id == snapshot.canonical_hash
        assert len(snapshot.field_id) == 64

    def test_genesis_requires_none_parent(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="genesis snapshot .* parent_field_id=None"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (identity,),
                parent_field_id="0" * 64,
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_non_genesis_requires_parent_hash(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="non-genesis snapshot must declare"):
            SharedFieldSnapshotV2(
                tick_id=1,
                regions=_base_regions(charter) + (identity,),
                parent_field_id=None,
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_non_genesis_rejects_arbitrary_parent(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="64-character lowercase hex"):
            SharedFieldSnapshotV2(
                tick_id=1,
                regions=_base_regions(charter) + (identity,),
                parent_field_id="not-a-hash",
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_non_genesis_rejects_short_parent_hash(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="64-character lowercase hex"):
            SharedFieldSnapshotV2(
                tick_id=1,
                regions=_base_regions(charter) + (identity,),
                parent_field_id="0" * 63,
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_non_genesis_rejects_uppercase_parent_hash(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(ValueError, match="64-character lowercase hex"):
            SharedFieldSnapshotV2(
                tick_id=1,
                regions=_base_regions(charter) + (identity,),
                parent_field_id="A" * 64,
                source_manifest_ids=(charter.source_manifest_id,),
            )


class TestFieldSpanStrictness:
    def test_boolean_confidence_rejected(self) -> None:
        with pytest.raises(TypeError, match="finite float"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                confidence=True,
            )

    def test_integer_confidence_rejected(self) -> None:
        with pytest.raises(TypeError, match="finite float"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                confidence=1,
            )

    def test_confidence_must_be_finite_number(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                confidence=float("inf"),
            )

    def test_container_refs_must_be_list_or_tuple_of_strings(self) -> None:
        with pytest.raises(TypeError, match="list or tuple"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                container_refs="not-a-list",
            )

    def test_container_refs_reject_non_string_items(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                container_refs=["ok", 123],
            )

    def test_refs_are_not_sorted_or_deduplicated(self) -> None:
        span = FieldSpanV2(
            span_id="s1",
            text="hello",
            container_refs=["b", "a", "b"],
        )
        assert span.container_refs == ("b", "a", "b")

    def test_negative_zero_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"canonical \+0\.0"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                confidence=-0.0,
            )

    def test_positive_zero_confidence_accepted(self) -> None:
        span = FieldSpanV2(
            span_id="s1",
            text="hello",
            confidence=+0.0,
        )
        assert span.confidence == 0.0
        assert math.copysign(1.0, span.confidence) == 1.0


class TestIdentityCharter:
    def test_charter_hash_is_canonical_sha256(self, charter: IdentityCharterV2) -> None:
        assert charter.charter_hash == (
            "bd01b48e40feccb5cee351060ae49698344221770a5217abb5c9618d62030438"
        )

    def test_empty_charter_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            IdentityCharterV2(text="")

    def test_oversized_charter_rejected(self) -> None:
        with pytest.raises(ValueError, match=f"{CHARTER_MAX_CHARS_V2}"):
            IdentityCharterV2(text="x" * (CHARTER_MAX_CHARS_V2 + 1))

    def test_unsupported_character_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            IdentityCharterV2(text="hello\x00world")

    def test_version_must_be_exactly_one(self) -> None:
        with pytest.raises(ValueError, match="must be 1"):
            IdentityCharterV2(text=CHARTER_TEXT, charter_version=2)

    def test_version_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 1"):
            IdentityCharterV2(text=CHARTER_TEXT, charter_version=0)

    def test_charter_span_and_manifest_derived_from_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        assert charter.span_id == f"charter:{charter.charter_hash}"
        assert charter.source_manifest_id == (
            f"axon-identity-charter-v1:{charter.charter_hash}"
        )

    def test_subclass_instances_rejected(self) -> None:
        class ForgedCharter(IdentityCharterV2):
            def to_canonical_dict(self) -> dict[str, object]:
                return {"forged": True}

            def build_genesis_snapshot(
                self,
                *,
                source_manifest_ids: tuple[str, ...] = (),
            ):
                return None

        with pytest.raises(TypeError, match="exact base IdentityCharterV2"):
            ForgedCharter(text=CHARTER_TEXT)


class TestCoreIdentityView:
    def test_same_inputs_produce_same_envelope_and_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        view_a = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        view_b = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        assert view_a.envelope == view_b.envelope
        assert view_a.view_hash == view_b.view_hash
        assert "h=" in view_a.envelope
        assert charter.charter_hash in view_a.envelope

    def test_different_core_id_changes_view_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        view_a = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        view_b = render_core_identity_view_v2(
            charter,
            core_id="axon64-b",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        assert view_a.view_hash != view_b.view_hash

    def test_different_current_role_changes_view_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        view_a = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        view_b = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="consolidator",
            envelope_char_budget=192,
        )
        assert view_a.view_hash != view_b.view_hash

    def test_too_small_budget_fails_closed(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="exceeds budget"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("consolidator", "proposer", "sleeper"),
                current_role="proposer",
                envelope_char_budget=1,
            )

    def test_unsupported_character_fails_closed(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a\x00",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_budget_above_max_rejected(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="envelope_char_budget"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=193,
            )


class TestCoreIdentityViewDirectConstruction:
    def test_valid_direct_construction_accepts_matching_envelope(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        rendered = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        direct = CoreIdentityViewV2(
            charter_hash=charter.charter_hash,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer"),
            current_role="proposer",
            envelope=rendered.envelope,
        )
        assert direct.view_hash == rendered.view_hash
        assert direct.role_capabilities == ("consolidator", "proposer")

    def test_unsorted_capabilities_rejected(self, charter: IdentityCharterV2) -> None:
        envelope = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer"),
            current_role="proposer",
            envelope_char_budget=192,
        ).envelope
        with pytest.raises(ValueError, match="sorted and unique"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer", "consolidator"),
                current_role="proposer",
                envelope=envelope,
            )

    def test_duplicate_capabilities_rejected(self, charter: IdentityCharterV2) -> None:
        envelope = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("proposer",),
            current_role="proposer",
            envelope_char_budget=192,
        ).envelope
        with pytest.raises(ValueError, match="sorted and unique"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer", "proposer"),
                current_role="proposer",
                envelope=envelope,
            )

    def test_empty_capabilities_rejected(self, charter: IdentityCharterV2) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=(),
                current_role="proposer",
                envelope="irrelevant",
            )

    def test_delimiter_injection_rejected(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="delimiter"):
            render_core_identity_view_v2(
                charter,
                core_id="axon;64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_line_break_injection_rejected(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="delimiter"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon\n64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_uppercase_charter_hash_rejected(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        uppercase_hash = charter.charter_hash.upper()
        with pytest.raises(ValueError, match="lowercase hex"):
            CoreIdentityViewV2(
                charter_hash=uppercase_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope="irrelevant",
            )

    def test_stale_envelope_rejected(self, charter: IdentityCharterV2) -> None:
        envelope = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("proposer",),
            current_role="proposer",
            envelope_char_budget=192,
        ).envelope
        with pytest.raises(ValueError, match="envelope does not match"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("consolidator", "proposer"),
                current_role="consolidator",
                envelope=envelope,
            )

    def test_view_is_frozen(self, charter: IdentityCharterV2) -> None:
        view = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("proposer",),
            current_role="proposer",
            envelope_char_budget=192,
        )
        with pytest.raises(AttributeError):
            view.core_id = "tampered"


class TestV2ExactSubclassAttacks:
    def test_evil_span_rejected_in_region(self, charter: IdentityCharterV2) -> None:
        class EvilSpan(FieldSpanV2):
            def to_canonical_dict(self) -> dict[str, object]:
                return {"confidence": 1}

            @property
            def canonical_hash(self) -> str:
                return "0" * 64

        with pytest.raises(TypeError, match="exact base FieldSpanV2"):
            RegionStateV2(
                name=LogicalRegionV2.SCRATCH,
                spans=(EvilSpan(span_id="s1", text="ok"),),
            )

    def test_evil_region_rejected_in_snapshot(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        class EvilRegion(RegionStateV2):
            def to_canonical_dict(self) -> dict[str, object]:
                return {"forged": True}

        base = _base_regions(charter)
        with pytest.raises(TypeError, match="exact base RegionStateV2"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=base + (EvilRegion(name=LogicalRegionV2.IDENTITY),),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_evil_snapshot_rejected_by_serializer(self, charter: IdentityCharterV2) -> None:
        class EvilSnapshot(SharedFieldSnapshotV2):
            def to_dict(self) -> dict[str, object]:
                return {"forged": True}

        snapshot = charter.build_genesis_snapshot()
        evil = object.__new__(EvilSnapshot)
        object.__setattr__(evil, "tick_id", snapshot.tick_id)
        object.__setattr__(evil, "regions", snapshot.regions)
        object.__setattr__(evil, "parent_field_id", snapshot.parent_field_id)
        object.__setattr__(
            evil,
            "source_manifest_ids",
            snapshot.source_manifest_ids,
        )
        object.__setattr__(evil, "field_id", snapshot.field_id)
        object.__setattr__(evil, "canonical_hash", snapshot.canonical_hash)

        from runtime.field.serde_v2 import serialize_shared_field_v2

        with pytest.raises(TypeError, match="exact base SharedFieldSnapshotV2"):
            serialize_shared_field_v2(evil)

    def test_evil_view_rejected_at_construction(self, charter: IdentityCharterV2) -> None:
        class EvilView(CoreIdentityViewV2):
            def to_canonical_dict(self, *, include_envelope: bool = False) -> dict[str, object]:
                return {"forged": True}

        rendered = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("proposer",),
            current_role="proposer",
            envelope_char_budget=192,
        )
        with pytest.raises(TypeError, match="exact base CoreIdentityViewV2"):
            EvilView(
                charter_hash=rendered.charter_hash,
                core_id=rendered.core_id,
                display_name=rendered.display_name,
                model_label=rendered.model_label,
                role_capabilities=rendered.role_capabilities,
                current_role=rendered.current_role,
                envelope=rendered.envelope,
            )


class TestV2SnapshotCanonicality:
    def test_reversed_full_region_order_rejected(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        identity = charter.build_identity_region()
        base = _base_regions(charter)
        reversed_regions = base[::-1] + (identity,)
        with pytest.raises(ValueError, match="canonical v2 order"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=reversed_regions,
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_permuted_regions_rejected(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        base = list(_base_regions(charter))
        base[0], base[1] = base[1], base[0]
        with pytest.raises(ValueError, match="canonical v2 order"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=tuple(base) + (identity,),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_generator_regions_rejected(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        base = _base_regions(charter)

        def _gen():
            for region in base:
                yield region
            yield identity

        with pytest.raises(TypeError, match="list or tuple"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_gen(),
                source_manifest_ids=(charter.source_manifest_id,),
            )


class TestRegionStateRawTypes:
    def test_spans_reject_string(self) -> None:
        with pytest.raises(TypeError, match="must be a list or tuple"):
            RegionStateV2(
                name=LogicalRegionV2.SCRATCH,
                spans="not-a-list",
            )

    def test_spans_reject_mapping(self) -> None:
        with pytest.raises(TypeError, match="must be a list or tuple"):
            RegionStateV2(
                name=LogicalRegionV2.SCRATCH,
                spans={"span_id": "s1", "text": "ok"},
            )

    def test_spans_reject_generator(self) -> None:
        def _gen():
            yield FieldSpanV2(span_id="s1", text="ok")

        with pytest.raises(TypeError, match="must be a list or tuple"):
            RegionStateV2(
                name=LogicalRegionV2.SCRATCH,
                spans=_gen(),
            )


class TestIdentityCharterGenesisStrictness:
    def test_string_extra_manifests_rejected(self, charter: IdentityCharterV2) -> None:
        with pytest.raises(TypeError, match="list or tuple of strings"):
            charter.build_genesis_snapshot(source_manifest_ids="bad")  # type: ignore[arg-type]

    def test_mapping_extra_manifests_rejected(self, charter: IdentityCharterV2) -> None:
        with pytest.raises(TypeError, match="list or tuple of strings"):
            charter.build_genesis_snapshot(source_manifest_ids={"x": "y"})  # type: ignore[arg-type]

    def test_generator_extra_manifests_rejected(self, charter: IdentityCharterV2) -> None:
        def _gen():
            yield "a:1"

        with pytest.raises(TypeError, match="list or tuple of strings"):
            charter.build_genesis_snapshot(source_manifest_ids=_gen())  # type: ignore[arg-type]

    def test_duplicate_extra_manifests_rejected(self, charter: IdentityCharterV2) -> None:
        with pytest.raises(ValueError, match="must not contain duplicates"):
            charter.build_genesis_snapshot(source_manifest_ids=("a:1", "a:1"))

    def test_extra_manifests_preserve_supplied_order(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        extras = ("z:1", "a:2", "m:3")
        snapshot = charter.build_genesis_snapshot(source_manifest_ids=extras)
        assert snapshot.source_manifest_ids == (
            charter.source_manifest_id,
            "z:1",
            "a:2",
            "m:3",
        )


class TestRendererAdversarial:
    def test_fake_charter_rejected(self, charter: IdentityCharterV2) -> None:
        class FakeCharter:
            charter_hash = charter.charter_hash

        with pytest.raises(TypeError, match="exact base IdentityCharterV2"):
            render_core_identity_view_v2(
                FakeCharter(),  # type: ignore[arg-type]
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_unsorted_capabilities_rejected_by_renderer(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="sorted and unique"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer", "consolidator"),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_duplicate_capabilities_rejected_by_renderer(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="sorted and unique"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer", "proposer"),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_non_string_capabilities_rejected_by_renderer(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(TypeError, match="list or tuple of strings"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=(1,),  # type: ignore[arg-type]
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_current_role_outside_capabilities_rejected_by_renderer(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="current_role must be present"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="consolidator",
                envelope_char_budget=192,
            )

    def test_valid_capabilities_only_when_already_sorted(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        view = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        assert view.role_capabilities == ("consolidator", "proposer", "sleeper")



class EvilStrWithLyingIterator(str):
    """A str subclass whose iterator lies about the actual characters."""

    def __iter__(self):
        for _ in range(len(self)):
            yield "a"


class EvilInt(int):
    pass


class EvilFloat(float):
    pass


class TestV2ExactPrimitiveTypes:
    def test_str_subclass_span_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            FieldSpanV2(
                span_id=EvilStrWithLyingIterator("s1"),  # type: ignore[arg-type]
                text="hello",
            )

    def test_str_subclass_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            FieldSpanV2(
                span_id="s1",
                text=EvilStrWithLyingIterator("hello\x00"),  # type: ignore[arg-type]
            )

    def test_str_subclass_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                kind=EvilStrWithLyingIterator("text"),  # type: ignore[arg-type]
            )

    def test_str_subclass_ref_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                container_refs=[EvilStrWithLyingIterator("ref")],  # type: ignore[list-item]
            )

    def test_int_subclass_tick_id_rejected(self, charter: IdentityCharterV2) -> None:
        identity = charter.build_identity_region()
        with pytest.raises(TypeError, match="must be an integer"):
            SharedFieldSnapshotV2(
                tick_id=EvilInt(0),  # type: ignore[arg-type]
                regions=_base_regions(charter) + (identity,),
                source_manifest_ids=(charter.source_manifest_id,),
            )

    def test_float_subclass_negative_zero_rejected(self) -> None:
        with pytest.raises(TypeError, match="finite float"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                confidence=EvilFloat(-0.0),  # type: ignore[arg-type]
            )

    def test_float_subclass_positive_confidence_rejected(self) -> None:
        with pytest.raises(TypeError, match="finite float"):
            FieldSpanV2(
                span_id="s1",
                text="hello",
                confidence=EvilFloat(1.0),  # type: ignore[arg-type]
            )


class TestV2ForgedExactBaseObjects:
    def test_forged_exact_base_span_with_bad_confidence_rejected_in_region(
        self,
    ) -> None:
        forged = object.__new__(FieldSpanV2)
        object.__setattr__(forged, "span_id", "s1")
        object.__setattr__(forged, "text", "ok")
        object.__setattr__(forged, "kind", "text")
        object.__setattr__(forged, "source", "")
        object.__setattr__(forged, "provenance", "")
        object.__setattr__(forged, "confidence", EvilFloat(-0.0))
        object.__setattr__(forged, "container_refs", ())
        object.__setattr__(forged, "edge_refs", ())

        with pytest.raises(TypeError, match="finite float"):
            RegionStateV2(
                name=LogicalRegionV2.SCRATCH,
                spans=(forged,),
            )

    def test_forged_exact_base_span_with_str_subclass_text_rejected_in_region(
        self,
    ) -> None:
        forged = object.__new__(FieldSpanV2)
        object.__setattr__(forged, "span_id", "s1")
        object.__setattr__(
            forged,
            "text",
            EvilStrWithLyingIterator("ok\x00"),
        )
        object.__setattr__(forged, "kind", "text")
        object.__setattr__(forged, "source", "")
        object.__setattr__(forged, "provenance", "")
        object.__setattr__(forged, "confidence", 1.0)
        object.__setattr__(forged, "container_refs", ())
        object.__setattr__(forged, "edge_refs", ())

        with pytest.raises(ValueError, match="non-empty string"):
            RegionStateV2(
                name=LogicalRegionV2.SCRATCH,
                spans=(forged,),
            )

    def test_forged_exact_base_region_missing_attributes_rejected_in_snapshot(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        forged_region = object.__new__(RegionStateV2)
        object.__setattr__(forged_region, "name", LogicalRegionV2.IDENTITY)

        with pytest.raises(TypeError, match="missing or invalid attributes"):
            SharedFieldSnapshotV2(
                tick_id=0,
                regions=_base_regions(charter) + (forged_region,),
                source_manifest_ids=(charter.source_manifest_id,),
            )


class TestV2ForgedCharterReconstruction:
    def test_renderer_rejects_forged_exact_base_charter_with_missing_text(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        forged = object.__new__(IdentityCharterV2)
        object.__setattr__(forged, "charter_version", 1)

        with pytest.raises((TypeError, ValueError), match="missing required primitive fields"):
            render_core_identity_view_v2(
                forged,  # type: ignore[arg-type]
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_renderer_rejects_forged_exact_base_charter_with_invalid_version(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        forged = object.__new__(IdentityCharterV2)
        object.__setattr__(forged, "text", CHARTER_TEXT)
        object.__setattr__(forged, "charter_version", 2)

        with pytest.raises((TypeError, ValueError), match="must be 1"):
            render_core_identity_view_v2(
                forged,  # type: ignore[arg-type]
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_renderer_rejects_empty_forged_charter_without_attribute_error(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        forged = object.__new__(IdentityCharterV2)

        with pytest.raises((TypeError, ValueError)):
            render_core_identity_view_v2(
                forged,  # type: ignore[arg-type]
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )
