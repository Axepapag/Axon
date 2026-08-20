from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from runtime.axon_runtime.config import load_runtime_config
from runtime.axon_runtime.identity_v2_config import (
    EXPECTED_BASE_V1_CONFIG_ID,
    IDENTITY_V2_CONTRACT_SCHEMA,
    IdentityV2ConfigError,
    IdentityV2Contract,
    load_identity_v2_contract,
    parse_identity_v2_contract,
)
from runtime.field.schema_v2 import IdentityCharterV2, canonical_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "ops" / "axon_runtime.identity-v2.contract.json"
R2_CONTRACT_PATH = REPO_ROOT / "ops" / "axon_runtime.identity-v2.contract.r2.json"
SMOKE_CONFIG_PATH = REPO_ROOT / "ops" / "axon_runtime.cpu-smoke.json"

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

CHARTER_HASH = (
    "bd01b48e40feccb5cee351060ae49698344221770a5217abb5c9618d62030438"
)

CONTRACT_ID = (
    "aaab37e4c7795508773875c11663fb120ab424eabc33cde1137e584ccbaba485"
)

R2_CONTRACT_ID = (
    "a252fdfd55fdcd3e4144b3c00d9b505c76b38c002efa03bb52dec01ccbefcd86"
)


def _load_mapping() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class TestV2ContractParsing:
    def test_sample_contract_parses_and_binds_hash(self) -> None:
        contract = load_identity_v2_contract(CONTRACT_PATH)

        assert isinstance(contract, IdentityV2Contract)
        assert contract.base_v1_config_id == EXPECTED_BASE_V1_CONFIG_ID
        assert contract.field_schema == "shared-field-v2"
        assert str(contract.state_root) == r"D:\Axon\State\training\regression\identity_v2"
        assert contract.sealed is True
        assert contract.physical_role == "context"
        assert contract.envelope_char_budget == 128
        assert contract.v1_auto_migration is False
        assert contract.migration_authority == "human_authorized"

        assert contract.charter.text == CHARTER_TEXT
        assert contract.charter.charter_hash == CHARTER_HASH
        assert contract.contract_id == CONTRACT_ID


    def test_load_does_not_create_state_root(self, tmp_path: Path) -> None:
        mapping = _load_mapping()
        future_root = tmp_path / "not_created" / "axon_runtime_identity_v2"
        paths = mapping["state_root"]
        assert isinstance(paths, str)
        mapping["state_root"] = str(future_root)

        contract = parse_identity_v2_contract(json.dumps(mapping))
        assert contract.state_root == future_root.resolve()
        assert not future_root.exists()

    def test_build_genesis_snapshot_is_in_memory_only(
        self,
        tmp_path: Path,
    ) -> None:
        mapping = _load_mapping()
        future_root = tmp_path / "not_created" / "axon_runtime_identity_v2"
        mapping["state_root"] = str(future_root)
        contract = parse_identity_v2_contract(json.dumps(mapping))

        snapshot = contract.build_genesis_snapshot()
        assert snapshot.tick_id == 0
        assert snapshot.region("identity").spans[0].text == CHARTER_TEXT
        assert not future_root.exists()

    def test_genesis_includes_contract_and_base_v1_provenance(
        self,
    ) -> None:
        contract = load_identity_v2_contract(CONTRACT_PATH)
        snapshot = contract.build_genesis_snapshot()
        manifests = list(snapshot.source_manifest_ids)
        assert f"{IDENTITY_V2_CONTRACT_SCHEMA}:{contract.contract_id}" in manifests
        assert f"axon-runtime-bootstrap-config-v1:{contract.base_v1_config_id}" in manifests
        assert contract.charter.source_manifest_id in manifests


class TestV2ContractR2:
    def test_r2_is_a_new_isolated_contract_without_mutating_r1(self) -> None:
        r1 = load_identity_v2_contract(CONTRACT_PATH)
        r2 = load_identity_v2_contract(R2_CONTRACT_PATH)

        assert r1.contract_id == CONTRACT_ID
        assert r1.envelope_char_budget == 128
        assert r2.contract_id == R2_CONTRACT_ID
        assert r2.contract_id != r1.contract_id
        assert r2.envelope_char_budget == 146
        assert r2.base_v1_config_id == r1.base_v1_config_id
        assert r2.field_schema == r1.field_schema == "shared-field-v2"
        assert r2.charter.to_canonical_dict() == r1.charter.to_canonical_dict()
        assert r2.sealed is True
        assert r2.physical_role == "context"
        assert r2.v1_auto_migration is False
        assert r2.migration_authority == "human_authorized"
        assert str(r2.state_root) == r"D:\Axon\State\training\regression\identity_v2_r2"
        assert r2.state_root != r1.state_root
        assert not r1.state_root.exists()
        assert not r2.state_root.exists()

    def test_r2_genesis_uses_its_own_contract_manifest_in_memory_only(self) -> None:
        r1 = load_identity_v2_contract(CONTRACT_PATH)
        r2 = load_identity_v2_contract(R2_CONTRACT_PATH)
        r1_genesis = r1.build_genesis_snapshot()
        r2_genesis = r2.build_genesis_snapshot()

        assert r1_genesis.field_id != r2_genesis.field_id
        assert (
            f"{IDENTITY_V2_CONTRACT_SCHEMA}:{r1.contract_id}"
            in r1_genesis.source_manifest_ids
        )
        assert (
            f"{IDENTITY_V2_CONTRACT_SCHEMA}:{r2.contract_id}"
            in r2_genesis.source_manifest_ids
        )
        assert (
            f"{IDENTITY_V2_CONTRACT_SCHEMA}:{r1.contract_id}"
            not in r2_genesis.source_manifest_ids
        )
        assert (
            f"{IDENTITY_V2_CONTRACT_SCHEMA}:{r2.contract_id}"
            not in r1_genesis.source_manifest_ids
        )
        assert not r1.state_root.exists()
        assert not r2.state_root.exists()


class TestV2ContractValidation:
    def test_wrong_base_v1_config_id_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["base_v1_config_id"] = "0" * 64
        with pytest.raises(IdentityV2ConfigError, match="current v1 smoke config"):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_v1_state_root_collision_rejected(self, tmp_path: Path) -> None:
        mapping = _load_mapping()
        mapping["state_root"] = r"D:\Axon\State\axon_runtime"
        with pytest.raises(IdentityV2ConfigError, match="protected v1 runtime root"):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_underneath_v1_root_rejected(self, tmp_path: Path) -> None:
        mapping = _load_mapping()
        mapping["state_root"] = r"D:\Axon\State\axon_runtime\identity_v2"
        with pytest.raises(
            IdentityV2ConfigError,
            match="underneath the protected v1 runtime root",
        ):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_missing_identity_v2_marker_rejected(self, tmp_path: Path) -> None:
        mapping = _load_mapping()
        mapping["state_root"] = r"D:\Axon\State\axon_runtime_v2"
        with pytest.raises(
            IdentityV2ConfigError,
            match="must contain 'identity_v2'",
        ):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_v1_auto_migration_true_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["migration"]["v1_auto_migration"] = True
        with pytest.raises(
            IdentityV2ConfigError,
            match="v1_auto_migration must be false",
        ):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_sealed_false_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["identity"]["sealed"] = False
        with pytest.raises(IdentityV2ConfigError, match="identity.sealed must be true"):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_physical_role_not_context_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["identity"]["physical_role"] = "proposal"
        with pytest.raises(
            IdentityV2ConfigError,
            match="identity.physical_role must be 'context'",
        ):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_wrong_charter_hash_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["identity"]["charter_hash"] = "0" * 64
        with pytest.raises(
            IdentityV2ConfigError,
            match="charter_hash does not match",
        ):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_unsupported_charter_schema_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["identity"]["charter_schema"] = "axon-identity-charter-v2"
        with pytest.raises(IdentityV2ConfigError, match="unsupported charter_schema"):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_unknown_top_level_key_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["extra"] = True
        with pytest.raises(IdentityV2ConfigError, match="keys mismatch"):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_literal_secret_key_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["api_key"] = "literal-secret"
        with pytest.raises(
            IdentityV2ConfigError,
            match="forbidden literal secret field",
        ):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_duplicate_json_keys_rejected(self) -> None:
        text = CONTRACT_PATH.read_text(encoding="utf-8").replace(
            '"schema": "axon-runtime-identity-v2-contract-v1",',
            (
                '"schema": "axon-runtime-identity-v2-contract-v1",\n'
                '  "schema": "axon-runtime-identity-v2-contract-v1",'
            ),
            1,
        )
        with pytest.raises(IdentityV2ConfigError, match="duplicate JSON key"):
            parse_identity_v2_contract(text)

    def test_non_finite_number_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["identity"]["envelope_char_budget"] = float("nan")
        with pytest.raises(IdentityV2ConfigError, match="non-finite"):
            parse_identity_v2_contract(json.dumps(mapping))


class TestV2ContractDirectConstruction:
    def test_direct_construction_with_valid_fields_succeeds(self) -> None:
        contract = IdentityV2Contract(
            base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
            state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
            field_schema="shared-field-v2",
            charter=IdentityCharterV2(text=CHARTER_TEXT),
            sealed=True,
            physical_role="context",
            envelope_char_budget=128,
            migration_authority="human_authorized",
            v1_auto_migration=False,
        )
        assert contract.contract_id == CONTRACT_ID

    def test_direct_construction_rejects_wrong_base_id(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="current v1 smoke config"):
            IdentityV2Contract(
                base_v1_config_id="0" * 64,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_direct_construction_rejects_v1_root(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="protected v1 runtime root"):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\axon_runtime"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_direct_construction_rejects_unsealed(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="sealed must be true"):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=False,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_direct_construction_rejects_auto_migration(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="v1_auto_migration must be false"):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=True,
            )

    def test_direct_construction_rejects_non_exact_charter_type(self) -> None:
        class FakeCharter:
            text = CHARTER_TEXT
            charter_hash = CHARTER_HASH
            charter_version = 1

            def to_canonical_dict(self) -> dict[str, object]:
                return {"forged": True}

            def build_genesis_snapshot(
                self,
                *,
                source_manifest_ids: tuple[str, ...] = (),
            ):
                return None

        with pytest.raises(
            IdentityV2ConfigError,
            match="exact base IdentityCharterV2",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=FakeCharter(),  # type: ignore[arg-type]
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )


class TestV2ContractHashAndVersionStrictness:
    def test_uppercase_base_v1_config_id_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["base_v1_config_id"] = EXPECTED_BASE_V1_CONFIG_ID.upper()
        with pytest.raises(IdentityV2ConfigError, match="lowercase hexadecimal"):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_uppercase_charter_hash_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["identity"]["charter_hash"] = CHARTER_HASH.upper()
        with pytest.raises(IdentityV2ConfigError, match="lowercase hexadecimal"):
            parse_identity_v2_contract(json.dumps(mapping))

    @pytest.mark.parametrize("bad_version", ["01", "+1", " 1", "1 ", "2", "01"])
    def test_noncanonical_charter_version_rejected(
        self,
        bad_version: str,
    ) -> None:
        mapping = _load_mapping()
        mapping["identity"]["charter_version"] = bad_version
        with pytest.raises(IdentityV2ConfigError, match="raw string '1'"):
            parse_identity_v2_contract(json.dumps(mapping))

    def test_integer_charter_version_rejected(self) -> None:
        mapping = _load_mapping()
        mapping["identity"]["charter_version"] = 1
        with pytest.raises(IdentityV2ConfigError, match="raw string '1'"):
            parse_identity_v2_contract(json.dumps(mapping))


class TestV2ContractResolvedConfig:
    def test_resolved_config_is_deeply_immutable(self) -> None:
        contract = load_identity_v2_contract(CONTRACT_PATH)
        config = contract.resolved_config
        assert isinstance(config, MappingProxyType)
        with pytest.raises(TypeError):
            config["contract_id"] = "tampered"
        with pytest.raises(TypeError):
            config["descriptor"]["identity"]["sealed"] = False

    def test_mutating_returned_descriptor_does_not_affect_contract(self) -> None:
        contract = load_identity_v2_contract(CONTRACT_PATH)
        descriptor = contract.to_canonical_descriptor()
        descriptor["identity"]["sealed"] = False
        assert contract.sealed is True
        assert contract.contract_id == CONTRACT_ID


class TestV2ContractGenesisProvenance:
    def test_distinct_descriptors_produce_distinct_genesis_provenance(
        self,
        tmp_path: Path,
    ) -> None:
        mapping_a = _load_mapping()
        mapping_a["identity"]["envelope_char_budget"] = 128
        mapping_b = _load_mapping()
        mapping_b["identity"]["envelope_char_budget"] = 129

        contract_a = parse_identity_v2_contract(json.dumps(mapping_a))
        contract_b = parse_identity_v2_contract(json.dumps(mapping_b))
        assert contract_a.contract_id != contract_b.contract_id

        genesis_a = contract_a.build_genesis_snapshot()
        genesis_b = contract_b.build_genesis_snapshot()
        assert genesis_a.source_manifest_ids != genesis_b.source_manifest_ids
        assert genesis_a.field_id != genesis_b.field_id


class TestV1SmokeConfigBinding:
    def test_cpu_smoke_config_id_matches_expected_base_v1_id(self) -> None:
        config = load_runtime_config(SMOKE_CONFIG_PATH)
        assert config.config_id == EXPECTED_BASE_V1_CONFIG_ID


class TestV2ContractForgedCharter:
    def test_object_new_charter_with_incompatible_fields_rejected(self) -> None:
        forged = object.__new__(IdentityCharterV2)
        object.__setattr__(forged, "text", CHARTER_TEXT)
        object.__setattr__(forged, "charter_version", 2)

        with pytest.raises(
            IdentityV2ConfigError,
            match="valid exact base IdentityCharterV2",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=forged,
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_mutating_supplied_charter_after_contract_construction_is_neutral(
        self,
        tmp_path: Path,
    ) -> None:
        charter = IdentityCharterV2(text=CHARTER_TEXT)
        contract = IdentityV2Contract(
            base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
            state_root=tmp_path / "axon_runtime_identity_v2",
            field_schema="shared-field-v2",
            charter=charter,
            sealed=True,
            physical_role="context",
            envelope_char_budget=128,
            migration_authority="human_authorized",
            v1_auto_migration=False,
        )
        original_contract_id = contract.contract_id
        original_genesis_id = contract.build_genesis_snapshot().field_id

        # Mutate the originally supplied charter after contract construction.
        object.__setattr__(charter, "text", "forged text")

        assert contract.contract_id == original_contract_id
        assert contract.build_genesis_snapshot().field_id == original_genesis_id
        assert contract.charter.text == CHARTER_TEXT

    def test_mutating_contract_charter_fails_closed_or_is_neutral(
        self,
        tmp_path: Path,
    ) -> None:
        contract = IdentityV2Contract(
            base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
            state_root=tmp_path / "axon_runtime_identity_v2",
            field_schema="shared-field-v2",
            charter=IdentityCharterV2(text=CHARTER_TEXT),
            sealed=True,
            physical_role="context",
            envelope_char_budget=128,
            migration_authority="human_authorized",
            v1_auto_migration=False,
        )
        original_contract_id = contract.contract_id
        original_genesis_id = contract.build_genesis_snapshot().field_id

        # Attempt to mutate the contract-held charter directly.
        object.__setattr__(contract.charter, "text", "forged text")

        # The genesis snapshot must still derive from the immutable resolved config.
        assert contract.contract_id == original_contract_id
        assert contract.build_genesis_snapshot().field_id == original_genesis_id


class TestV2ContractWindowsStateRoot:
    def _mapping_with_state_root(self, state_root: str) -> str:
        mapping = _load_mapping()
        mapping["state_root"] = state_root
        return json.dumps(mapping)

    def test_trailing_space_component_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"D:\Axon\State\axon_runtime \identity_v2"
        )
        with pytest.raises(IdentityV2ConfigError, match="unsafe path component"):
            parse_identity_v2_contract(text)

    def test_trailing_dot_component_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"D:\Axon\State\axon_runtime.\identity_v2"
        )
        with pytest.raises(IdentityV2ConfigError, match="unsafe path component"):
            parse_identity_v2_contract(text)

    def test_extended_path_prefix_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"\\?\D:\Axon\State\axon_runtime\identity_v2"
        )
        with pytest.raises(
            IdentityV2ConfigError,
            match="extended.*device path prefix",
        ):
            parse_identity_v2_contract(text)

    def test_device_path_prefix_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"\\.\D:\Axon\State\axon_runtime\identity_v2"
        )
        with pytest.raises(
            IdentityV2ConfigError,
            match="extended.*device path prefix",
        ):
            parse_identity_v2_contract(text)

    def test_unc_path_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"\\server\share\axon_runtime_identity_v2"
        )
        with pytest.raises(
            IdentityV2ConfigError,
            match="UNC|extended or device path prefix",
        ):
            parse_identity_v2_contract(text)

    def test_relative_path_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"State\axon_runtime_identity_v2"
        )
        with pytest.raises(IdentityV2ConfigError, match="must be absolute"):
            parse_identity_v2_contract(text)

    def test_nul_in_path_rejected(self) -> None:
        text = self._mapping_with_state_root(
            "D:\\Axon\\State\\axon_runtime_identity_v2\x00"
        )
        with pytest.raises(IdentityV2ConfigError, match="forbidden control"):
            parse_identity_v2_contract(text)

    def test_normal_sample_root_still_parses(self) -> None:
        contract = load_identity_v2_contract(CONTRACT_PATH)
        assert str(contract.state_root) == r"D:\Axon\State\training\regression\identity_v2"


class TestV2ContractRootTypeStrictness:
    def test_null_root_raises_config_error(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="must be a string-keyed object"):
            parse_identity_v2_contract("null")

    def test_boolean_root_raises_config_error(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="must be a string-keyed object"):
            parse_identity_v2_contract("true")

    def test_integer_root_raises_config_error(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="must be a string-keyed object"):
            parse_identity_v2_contract("42")

    def test_string_root_raises_config_error(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="must be a string-keyed object"):
            parse_identity_v2_contract('"not-an-object"')

    def test_array_root_raises_config_error(self) -> None:
        with pytest.raises(IdentityV2ConfigError, match="must be a string-keyed object"):
            parse_identity_v2_contract('[{}]')



class TestV2ContractR6AdsAndWindowsRoot:
    def _mapping_with_state_root(self, state_root: str) -> str:
        mapping = _load_mapping()
        mapping["state_root"] = state_root
        return json.dumps(mapping)

    def test_ads_colon_in_path_component_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"D:\Axon\State\axon_runtime:identity_v2"
        )
        with pytest.raises(IdentityV2ConfigError, match="alternate data stream"):
            parse_identity_v2_contract(text)

    def test_ads_colon_with_data_stream_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"D:\Axon\State\axon_runtime:identity_v2:$DATA"
        )
        with pytest.raises(IdentityV2ConfigError, match="alternate data stream"):
            parse_identity_v2_contract(text)

    def test_ads_colon_in_non_final_component_rejected(self) -> None:
        text = self._mapping_with_state_root(
            r"D:\Axon\State\axon_runtime:identity_v2\extra"
        )
        with pytest.raises(IdentityV2ConfigError, match="alternate data stream"):
            parse_identity_v2_contract(text)

    def test_normal_temporary_root_still_parses(self, tmp_path: Path) -> None:
        mapping = _load_mapping()
        future_root = tmp_path / "not_created" / "axon_runtime_identity_v2"
        mapping["state_root"] = str(future_root)
        contract = parse_identity_v2_contract(json.dumps(mapping))
        assert contract.state_root == future_root.resolve()
        assert not future_root.exists()


class TestV2ContractSealedAuthority:
    def test_subclass_rejected_at_construction(self) -> None:
        class EvilContract(IdentityV2Contract):
            def to_canonical_descriptor(self) -> dict[str, object]:
                return {"forged": True}

        with pytest.raises(
            IdentityV2ConfigError,
            match="does not permit subclass instances",
        ):
            EvilContract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_forged_base_contract_has_no_descriptor_authority(self) -> None:
        forged = object.__new__(IdentityV2Contract)
        object.__setattr__(forged, "base_v1_config_id", EXPECTED_BASE_V1_CONFIG_ID)
        object.__setattr__(forged, "contract_id", CONTRACT_ID)
        object.__setattr__(
            forged,
            "resolved_config",
            {"forged": True},
        )

        with pytest.raises(
            IdentityV2ConfigError,
            match="no bound descriptor authority",
        ):
            forged.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="no bound descriptor authority",
        ):
            forged.build_genesis_snapshot()

    def test_forged_base_contract_populated_fields_has_no_authority(self) -> None:
        forged = object.__new__(IdentityV2Contract)
        for field_name in (
            "base_v1_config_id",
            "state_root",
            "field_schema",
            "charter",
            "sealed",
            "physical_role",
            "envelope_char_budget",
            "migration_authority",
            "v1_auto_migration",
            "contract_id",
            "resolved_config",
        ):
            object.__setattr__(forged, field_name, "forged")
        object.__setattr__(forged, "charter", IdentityCharterV2(text=CHARTER_TEXT))
        object.__setattr__(forged, "sealed", True)
        object.__setattr__(forged, "v1_auto_migration", False)
        object.__setattr__(forged, "envelope_char_budget", 128)
        object.__setattr__(
            forged,
            "state_root",
            Path(r"D:\Axon\State\training\regression\identity_v2"),
        )

        with pytest.raises(
            IdentityV2ConfigError,
            match="no bound descriptor authority",
        ):
            forged.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="no bound descriptor authority",
        ):
            forged.build_genesis_snapshot()


class TestV2ContractMutationResistance:
    def _make_contract(self, tmp_path: Path) -> IdentityV2Contract:
        return IdentityV2Contract(
            base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
            state_root=tmp_path / "axon_runtime_identity_v2",
            field_schema="shared-field-v2",
            charter=IdentityCharterV2(text=CHARTER_TEXT),
            sealed=True,
            physical_role="context",
            envelope_char_budget=128,
            migration_authority="human_authorized",
            v1_auto_migration=False,
        )

    def test_mutating_supplied_charter_does_not_alter_descriptor_or_genesis(
        self,
        tmp_path: Path,
    ) -> None:
        charter = IdentityCharterV2(text=CHARTER_TEXT)
        contract = IdentityV2Contract(
            base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
            state_root=tmp_path / "axon_runtime_identity_v2",
            field_schema="shared-field-v2",
            charter=charter,
            sealed=True,
            physical_role="context",
            envelope_char_budget=128,
            migration_authority="human_authorized",
            v1_auto_migration=False,
        )
        original_contract_id = contract.contract_id
        original_descriptor = contract.to_canonical_descriptor()
        original_genesis_id = contract.build_genesis_snapshot().field_id

        object.__setattr__(charter, "text", "forged charter text")

        assert contract.to_canonical_descriptor() == original_descriptor
        assert contract.build_genesis_snapshot().field_id == original_genesis_id
        assert contract.contract_id == original_contract_id

    def test_mutating_contract_held_charter_does_not_alter_descriptor_or_genesis(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        original_contract_id = contract.contract_id
        original_descriptor = contract.to_canonical_descriptor()
        original_genesis_id = contract.build_genesis_snapshot().field_id

        object.__setattr__(contract.charter, "text", "forged charter text")

        assert contract.to_canonical_descriptor() == original_descriptor
        assert contract.build_genesis_snapshot().field_id == original_genesis_id
        assert contract.contract_id == original_contract_id

    def test_replacing_contract_charter_does_not_alter_descriptor_or_genesis(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        original_contract_id = contract.contract_id
        original_descriptor = contract.to_canonical_descriptor()
        original_genesis_id = contract.build_genesis_snapshot().field_id

        object.__setattr__(
            contract,
            "charter",
            IdentityCharterV2(text="forged charter text"),
        )

        assert contract.to_canonical_descriptor() == original_descriptor
        assert contract.build_genesis_snapshot().field_id == original_genesis_id
        assert contract.contract_id == original_contract_id

    def test_mutating_field_schema_does_not_alter_descriptor(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        original_contract_id = contract.contract_id
        original_descriptor = contract.to_canonical_descriptor()

        object.__setattr__(contract, "field_schema", "forged-field-schema")

        assert contract.to_canonical_descriptor() == original_descriptor
        assert contract.contract_id == original_contract_id

    def test_mutating_resolved_config_does_not_alter_descriptor_or_genesis(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        original_contract_id = contract.contract_id
        original_descriptor = contract.to_canonical_descriptor()
        original_genesis_id = contract.build_genesis_snapshot().field_id

        object.__setattr__(contract, "resolved_config", {"forged": True})

        assert contract.to_canonical_descriptor() == original_descriptor
        assert contract.build_genesis_snapshot().field_id == original_genesis_id
        assert contract.contract_id == original_contract_id

    def test_mutating_contract_id_is_detected(self, tmp_path: Path) -> None:
        contract = self._make_contract(tmp_path)
        object.__setattr__(contract, "contract_id", "0" * 64)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.build_genesis_snapshot()


class TestV2ContractR6PrimitiveStrictness:
    def test_state_root_str_subclass_rejected_directly(self, tmp_path: Path) -> None:
        class EvilStr(str):
            pass

        with pytest.raises(IdentityV2ConfigError, match="must be a non-empty string"):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=EvilStr(str(tmp_path / "axon_runtime_identity_v2")),  # type: ignore[arg-type]
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_envelope_budget_int_subclass_rejected_directly(self) -> None:
        class EvilInt(int):
            pass

        with pytest.raises(
            IdentityV2ConfigError,
            match="envelope_char_budget must be an integer",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=EvilInt(128),  # type: ignore[arg-type]
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )


class TestV2ContractR7ExactDirectConstruction:
    """Direct construction must reject equality-lying str subclasses for
    authority-bearing text fields before any descriptor is emitted.
    """

    def test_base_v1_config_id_lying_str_subclass_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        class LyingSha(str):
            def __eq__(self, other: object) -> bool:
                return True

        with pytest.raises(
            IdentityV2ConfigError,
            match="must be a 64-character lowercase hexadecimal SHA-256",
        ):
            IdentityV2Contract(
                base_v1_config_id=LyingSha("0" * 64),  # type: ignore[arg-type]
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_field_schema_lying_str_subclass_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        class LyingStr(str):
            def __eq__(self, other: object) -> bool:
                return True

        with pytest.raises(
            IdentityV2ConfigError,
            match="must be a non-empty string",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema=LyingStr("forged-field-schema"),  # type: ignore[arg-type]
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_physical_role_lying_str_subclass_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        class LyingStr(str):
            def __eq__(self, other: object) -> bool:
                return True

        with pytest.raises(
            IdentityV2ConfigError,
            match="must be a non-empty string",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role=LyingStr("forged-role"),  # type: ignore[arg-type]
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_migration_authority_lying_str_subclass_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        class LyingStr(str):
            def __eq__(self, other: object) -> bool:
                return True

        with pytest.raises(
            IdentityV2ConfigError,
            match="must be a non-empty string",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=Path(r"D:\Axon\State\training\regression\identity_v2"),
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority=LyingStr("forged-authority"),  # type: ignore[arg-type]
                v1_auto_migration=False,
            )


class TestV2ContractR7BoundAuthority:
    """The bound descriptor text must be re-hashed and verified before any
    descriptor or genesis material is emitted.
    """

    def _make_contract(self, tmp_path: Path) -> IdentityV2Contract:
        return IdentityV2Contract(
            base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
            state_root=tmp_path / "axon_runtime_identity_v2",
            field_schema="shared-field-v2",
            charter=IdentityCharterV2(text=CHARTER_TEXT),
            sealed=True,
            physical_role="context",
            envelope_char_budget=128,
            migration_authority="human_authorized",
            v1_auto_migration=False,
        )

    def test_mutating_bound_descriptor_text_to_forged_json_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        original_contract_id = contract.contract_id

        forged_dict = contract.to_canonical_descriptor()
        forged_dict["identity"]["envelope_char_budget"] = 1
        forged_text = json.dumps(
            forged_dict,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        object.__setattr__(contract, "_bound_descriptor_text", forged_text)

        assert contract.contract_id == original_contract_id
        assert contract._bound_descriptor_hash == original_contract_id

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.build_genesis_snapshot()

    def test_mutating_bound_descriptor_text_to_invalid_json_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        invalid_text = "{not valid json"
        invalid_hash = hashlib.sha256(invalid_text.encode("utf-8")).hexdigest()
        object.__setattr__(contract, "_bound_descriptor_text", invalid_text)
        object.__setattr__(contract, "_bound_descriptor_hash", invalid_hash)
        object.__setattr__(contract, "contract_id", invalid_hash)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor text is not valid canonical JSON",
        ):
            contract.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor text is not valid canonical JSON",
        ):
            contract.build_genesis_snapshot()

    def test_mutating_bound_descriptor_hash_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        object.__setattr__(contract, "_bound_descriptor_hash", "0" * 64)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.build_genesis_snapshot()

    def test_mutating_contract_id_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        object.__setattr__(contract, "contract_id", "0" * 64)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.build_genesis_snapshot()

    def test_forged_object_with_non_str_bound_fields_fails_closed(
        self,
    ) -> None:
        forged = object.__new__(IdentityV2Contract)
        object.__setattr__(forged, "_bound_descriptor_text", b'{"forged":true}')
        object.__setattr__(forged, "_bound_descriptor_hash", "0" * 64)
        object.__setattr__(forged, "contract_id", "0" * 64)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority fields are not exact strings",
        ):
            forged.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority fields are not exact strings",
        ):
            forged.build_genesis_snapshot()

    def test_forged_object_with_bound_fields_but_mismatched_hash_fails_closed(
        self,
    ) -> None:
        forged = object.__new__(IdentityV2Contract)
        object.__setattr__(forged, "_bound_descriptor_text", '{"forged":true}')
        object.__setattr__(forged, "_bound_descriptor_hash", "0" * 64)
        object.__setattr__(forged, "contract_id", "0" * 64)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            forged.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            forged.build_genesis_snapshot()


class TestV2ContractR8AtomicAuthorityAndCanonicalValidation:
    """R8: atomic authority capture and canonical descriptor validation."""

    def _make_contract(self, tmp_path: Path) -> IdentityV2Contract:
        return IdentityV2Contract(
            base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
            state_root=tmp_path / "axon_runtime_identity_v2",
            field_schema="shared-field-v2",
            charter=IdentityCharterV2(text=CHARTER_TEXT),
            sealed=True,
            physical_role="context",
            envelope_char_budget=128,
            migration_authority="human_authorized",
            v1_auto_migration=False,
        )

    def _forged_contract_from_descriptor(
        self,
        descriptor: dict[str, object],
    ) -> IdentityV2Contract:
        """Build a forged contract whose text/hash/ID form a coherent triple."""

        text = canonical_json_bytes(descriptor).decode("utf-8")
        hash_value = hashlib.sha256(text.encode("utf-8")).hexdigest()
        forged = object.__new__(IdentityV2Contract)
        object.__setattr__(forged, "_bound_descriptor_text", text)
        object.__setattr__(forged, "_bound_descriptor_hash", hash_value)
        object.__setattr__(forged, "contract_id", hash_value)
        return forged

    def test_no_trace_hook_field_or_reference(self) -> None:
        assert not hasattr(IdentityV2Contract, "_trace_hook")
        contract_fields = {f.name for f in IdentityV2Contract.__dataclass_fields__.values()}
        assert "_trace_hook" not in contract_fields

    def test_verified_authority_carries_no_mutable_descriptor_mapping(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        verified = contract._verify_bound_authority()
        assert isinstance(verified.canonical_text, str)
        assert isinstance(verified.contract_id, str)
        assert isinstance(verified.base_v1_config_id, str)
        assert isinstance(verified.charter_text, str)
        # No dict/list/mapping attribute may carry descriptor state.
        for attr_name in ("canonical_text", "contract_id", "base_v1_config_id", "charter_text"):
            attr_value = getattr(verified, attr_name)
            assert not isinstance(attr_value, (dict, list))
            assert not isinstance(attr_value, MappingProxyType)

    def test_post_verification_contract_id_mutation_with_sys_settrace(
        self,
        tmp_path: Path,
    ) -> None:
        contract = self._make_contract(tmp_path)
        original_contract_id = contract.contract_id
        original_genesis_id = contract.build_genesis_snapshot().field_id

        prior_trace = sys.gettrace()
        target_frame_id: list[int | None] = [None]

        def trace_fn(frame, event, arg):
            # Mutate contract_id immediately after _verify_bound_authority returns.
            if (
                event == "return"
                and frame.f_code.co_name == "_verify_bound_authority"
            ):
                target_frame_id[0] = id(frame)
                object.__setattr__(contract, "contract_id", "0" * 64)
            return trace_fn

        sys.settrace(trace_fn)
        try:
            snapshot = contract.build_genesis_snapshot()
        finally:
            sys.settrace(prior_trace)

        assert target_frame_id[0] is not None
        assert contract.contract_id == "0" * 64
        assert (
            f"{IDENTITY_V2_CONTRACT_SCHEMA}:{original_contract_id}"
            in snapshot.source_manifest_ids
        )
        assert (
            f"axon-runtime-bootstrap-config-v1:{EXPECTED_BASE_V1_CONFIG_ID}"
            in snapshot.source_manifest_ids
        )
        assert snapshot.field_id == original_genesis_id

        # Later calls fail closed because contract_id no longer matches the
        # bound descriptor authority.
        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor authority does not match contract_id",
        ):
            contract.to_canonical_descriptor()

    def test_coherent_noncanonical_json_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        descriptor = self._make_contract(tmp_path).to_canonical_descriptor()
        pretty_text = json.dumps(descriptor, sort_keys=True, indent=2)
        pretty_hash = hashlib.sha256(pretty_text.encode("utf-8")).hexdigest()

        forged = object.__new__(IdentityV2Contract)
        object.__setattr__(forged, "_bound_descriptor_text", pretty_text)
        object.__setattr__(forged, "_bound_descriptor_hash", pretty_hash)
        object.__setattr__(forged, "contract_id", pretty_hash)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor text is not canonical",
        ):
            forged.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor text is not canonical",
        ):
            forged.build_genesis_snapshot()

    def test_coherent_malformed_descriptor_missing_migration_key_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        descriptor = self._make_contract(tmp_path).to_canonical_descriptor()
        del descriptor["migration"]
        forged = self._forged_contract_from_descriptor(descriptor)

        with pytest.raises(IdentityV2ConfigError, match="keys mismatch"):
            forged.to_canonical_descriptor()

        with pytest.raises(IdentityV2ConfigError, match="keys mismatch"):
            forged.build_genesis_snapshot()

    def test_coherent_forged_base_v1_config_id_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        descriptor = self._make_contract(tmp_path).to_canonical_descriptor()
        descriptor["base_v1_config_id"] = "0" * 64
        forged = self._forged_contract_from_descriptor(descriptor)

        with pytest.raises(
            IdentityV2ConfigError,
            match="base_v1_config_id does not match the current v1 smoke config",
        ):
            forged.to_canonical_descriptor()

    def test_coherent_forged_charter_hash_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        descriptor = self._make_contract(tmp_path).to_canonical_descriptor()
        descriptor["identity"]["charter_hash"] = "0" * 64
        forged = self._forged_contract_from_descriptor(descriptor)

        with pytest.raises(
            IdentityV2ConfigError,
            match="charter_hash does not match canonical charter hash",
        ):
            forged.to_canonical_descriptor()

    def test_coherent_forged_physical_role_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        descriptor = self._make_contract(tmp_path).to_canonical_descriptor()
        descriptor["identity"]["physical_role"] = "proposal"
        forged = self._forged_contract_from_descriptor(descriptor)

        with pytest.raises(
            IdentityV2ConfigError,
            match="identity.physical_role must be 'context'",
        ):
            forged.to_canonical_descriptor()

    def test_coherent_forged_migration_authority_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        descriptor = self._make_contract(tmp_path).to_canonical_descriptor()
        descriptor["migration"]["authority"] = "auto"
        forged = self._forged_contract_from_descriptor(descriptor)

        with pytest.raises(
            IdentityV2ConfigError,
            match="migration.authority must be 'human_authorized'",
        ):
            forged.to_canonical_descriptor()

    def test_coherent_forged_schema_fails_closed(
        self,
        tmp_path: Path,
    ) -> None:
        descriptor = self._make_contract(tmp_path).to_canonical_descriptor()
        descriptor["schema"] = "axon-runtime-identity-v2-contract-v2"
        forged = self._forged_contract_from_descriptor(descriptor)

        with pytest.raises(
            IdentityV2ConfigError,
            match="unknown identity v2 contract schema",
        ):
            forged.to_canonical_descriptor()

    def test_coherent_duplicate_keys_fails_closed(self) -> None:
        # A text/hash/ID triple where the JSON object contains duplicate keys
        # is syntactically valid but must be rejected by the strict parser.
        forged_text = (
            '{"schema": "'
            + IDENTITY_V2_CONTRACT_SCHEMA
            + '", "schema": "'
            + IDENTITY_V2_CONTRACT_SCHEMA
            + '"}'
        )
        forged_hash = hashlib.sha256(
            forged_text.encode("utf-8")
        ).hexdigest()
        forged = object.__new__(IdentityV2Contract)
        object.__setattr__(forged, "_bound_descriptor_text", forged_text)
        object.__setattr__(forged, "_bound_descriptor_hash", forged_hash)
        object.__setattr__(forged, "contract_id", forged_hash)

        with pytest.raises(
            IdentityV2ConfigError,
            match="duplicate JSON key",
        ):
            forged.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="duplicate JSON key",
        ):
            forged.build_genesis_snapshot()

    def test_bound_text_with_unpaired_surrogate_fails_closed(self) -> None:
        # Valid JSON containing an unpaired surrogate as a raw character.
        forged_text = '{"forged": "\ud800"}'
        forged_bytes = forged_text.encode("utf-8", "surrogatepass")
        forged_hash = hashlib.sha256(forged_bytes).hexdigest()

        forged = object.__new__(IdentityV2Contract)
        object.__setattr__(forged, "_bound_descriptor_text", forged_text)
        object.__setattr__(forged, "_bound_descriptor_hash", forged_hash)
        object.__setattr__(forged, "contract_id", forged_hash)

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor text is not valid UTF-8",
        ):
            forged.to_canonical_descriptor()

        with pytest.raises(
            IdentityV2ConfigError,
            match="bound descriptor text is not valid UTF-8",
        ):
            forged.build_genesis_snapshot()



class TestV2ContractR9PathLikeExceptionNormalization:
    """R9: every ordinary PathLike conversion failure is IdentityV2ConfigError."""

    @pytest.mark.parametrize("exc_cls", [RuntimeError, OSError, ValueError])
    def test_direct_pathlike_fspath_exception_normalized(
        self,
        exc_cls: type[Exception],
    ) -> None:
        class EvilPath(os.PathLike[str]):
            def __fspath__(self) -> str:
                raise exc_cls("attacker-controlled conversion")

        with pytest.raises(
            IdentityV2ConfigError,
            match="state_root could not be converted to a path",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=EvilPath(),  # type: ignore[arg-type]
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_direct_pathlike_fspath_custom_exception_normalized(self) -> None:
        class CustomError(Exception):
            pass

        class EvilPath(os.PathLike[str]):
            def __fspath__(self) -> str:
                raise CustomError("attacker-controlled conversion")

        with pytest.raises(
            IdentityV2ConfigError,
            match="state_root could not be converted to a path",
        ):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=EvilPath(),  # type: ignore[arg-type]
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    def test_direct_pathlike_fspath_keyboard_interrupt_not_swallowed(self) -> None:
        class EvilPath(os.PathLike[str]):
            def __fspath__(self) -> str:
                raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            IdentityV2Contract(
                base_v1_config_id=EXPECTED_BASE_V1_CONFIG_ID,
                state_root=EvilPath(),  # type: ignore[arg-type]
                field_schema="shared-field-v2",
                charter=IdentityCharterV2(text=CHARTER_TEXT),
                sealed=True,
                physical_role="context",
                envelope_char_budget=128,
                migration_authority="human_authorized",
                v1_auto_migration=False,
            )

    @pytest.mark.parametrize("exc_cls", [RuntimeError, OSError, ValueError])
    def test_load_contract_pathlike_fspath_exception_normalized(
        self,
        exc_cls: type[Exception],
    ) -> None:
        class EvilPath(os.PathLike[str]):
            def __fspath__(self) -> str:
                raise exc_cls("attacker-controlled conversion")

        with pytest.raises(
            IdentityV2ConfigError,
            match="contract path could not be converted",
        ):
            load_identity_v2_contract(EvilPath())  # type: ignore[arg-type]

    def test_load_contract_pathlike_fspath_custom_exception_normalized(self) -> None:
        class CustomError(Exception):
            pass

        class EvilPath(os.PathLike[str]):
            def __fspath__(self) -> str:
                raise CustomError("attacker-controlled conversion")

        with pytest.raises(
            IdentityV2ConfigError,
            match="contract path could not be converted",
        ):
            load_identity_v2_contract(EvilPath())  # type: ignore[arg-type]

    def test_load_contract_pathlike_fspath_keyboard_interrupt_not_swallowed(
        self,
    ) -> None:
        class EvilPath(os.PathLike[str]):
            def __fspath__(self) -> str:
                raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            load_identity_v2_contract(EvilPath())  # type: ignore[arg-type]

    def test_load_contract_missing_file_still_raises_file_not_found(self) -> None:
        missing = Path(r"D:\Axon\State\not_created\axon_runtime_identity_v2.json")
        with pytest.raises(FileNotFoundError):
            load_identity_v2_contract(missing)
