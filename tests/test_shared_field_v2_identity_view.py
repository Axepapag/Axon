from __future__ import annotations

import pytest

from runtime.field.schema_v2 import (
    CoreIdentityViewV2,
    IdentityCharterV2,
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


class TestDeterministicIdentityEnvelope:
    def test_same_inputs_same_envelope_and_hash(
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
        base = {
            "charter": charter,
            "display_name": "Axon 64D A",
            "model_label": "64D",
            "role_capabilities": ("consolidator", "proposer", "sleeper"),
            "current_role": "proposer",
            "envelope_char_budget": 192,
        }
        view_a = render_core_identity_view_v2(core_id="axon64-a", **base)
        view_b = render_core_identity_view_v2(core_id="axon64-b", **base)
        assert view_a.view_hash != view_b.view_hash

    def test_different_role_changes_view_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        base = {
            "charter": charter,
            "core_id": "axon64-a",
            "display_name": "Axon 64D A",
            "model_label": "64D",
            "role_capabilities": ("consolidator", "proposer", "sleeper"),
            "envelope_char_budget": 192,
        }
        view_a = render_core_identity_view_v2(current_role="proposer", **base)
        view_b = render_core_identity_view_v2(current_role="consolidator", **base)
        assert view_a.view_hash != view_b.view_hash


class TestIdentityEnvelopeFailsClosed:
    def test_too_small_budget_does_not_truncate(
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
                envelope_char_budget=50,
            )

    def test_unsupported_input_character_does_not_normalize(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="unsupported characters"):
            render_core_identity_view_v2(
                charter,
                core_id="axon64-a",
                display_name="Axon 64D A\u00e9",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="proposer",
                envelope_char_budget=192,
            )

    def test_budget_above_hard_max_rejected(
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


class TestDirectIdentityViewValidation:
    def test_direct_construction_requires_matching_envelope(
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

    def test_direct_construction_rejects_unsorted_capabilities(
        self,
        charter: IdentityCharterV2,
    ) -> None:
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

    def test_direct_construction_rejects_stale_envelope(
        self,
        charter: IdentityCharterV2,
    ) -> None:
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

    def test_delimiter_injection_in_descriptor_rejected(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        for bad_value, label in (
            ("axon;64", "core_id"),
            ("name=bad", "display_name"),
            ("64,D", "model_label"),
            ("propo,ser", "role_capabilities"),
            ("propo\nser", "current_role"),
        ):
            kwargs = {
                "charter": charter,
                "core_id": "axon64-a",
                "display_name": "Axon 64D A",
                "model_label": "64D",
                "role_capabilities": ("proposer",),
                "current_role": "proposer",
                "envelope_char_budget": 192,
            }
            if label == "role_capabilities":
                kwargs["role_capabilities"] = (bad_value,)
            else:
                kwargs[label] = bad_value
            with pytest.raises(ValueError, match="delimiter|non-empty string|unsupported"):
                render_core_identity_view_v2(**kwargs)

    def test_view_object_is_immutable(
        self,
        charter: IdentityCharterV2,
    ) -> None:
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
            view.envelope = "tampered"
        with pytest.raises(AttributeError):
            view.role_capabilities.append("extra")  # type: ignore[attr-defined]


class TestDirectIdentityViewAdversarial:
    def test_direct_rejects_non_string_capabilities(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(TypeError, match="list or tuple of strings"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=(1,),  # type: ignore[arg-type]
                current_role="proposer",
                envelope="irrelevant",
            )

    def test_direct_rejects_current_role_outside_capabilities(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="current_role must be present"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer",),
                current_role="consolidator",
                envelope="irrelevant",
            )

    def test_direct_rejects_unsorted_capabilities(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="sorted and unique"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer", "consolidator"),
                current_role="proposer",
                envelope="irrelevant",
            )

    def test_direct_rejects_duplicate_capabilities(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        with pytest.raises(ValueError, match="sorted and unique"):
            CoreIdentityViewV2(
                charter_hash=charter.charter_hash,
                core_id="axon64-a",
                display_name="Axon 64D A",
                model_label="64D",
                role_capabilities=("proposer", "proposer"),
                current_role="proposer",
                envelope="irrelevant",
            )

    def test_direct_preserves_supplied_capability_order(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        rendered = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope_char_budget=192,
        )
        direct = CoreIdentityViewV2(
            charter_hash=charter.charter_hash,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("consolidator", "proposer", "sleeper"),
            current_role="proposer",
            envelope=rendered.envelope,
        )
        assert direct.role_capabilities == ("consolidator", "proposer", "sleeper")



class TestRendererRejectsForgedCharter:
    def test_renderer_rejects_object_new_charter_missing_text(
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

    def test_renderer_rejects_object_new_charter_invalid_version(
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

    def test_renderer_rejects_empty_object_new_charter_no_attribute_error(
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

    def test_renderer_uses_validated_charter_clone_for_hash(
        self,
        charter: IdentityCharterV2,
    ) -> None:
        view = render_core_identity_view_v2(
            charter,
            core_id="axon64-a",
            display_name="Axon 64D A",
            model_label="64D",
            role_capabilities=("proposer",),
            current_role="proposer",
            envelope_char_budget=192,
        )
        assert charter.charter_hash in view.envelope
