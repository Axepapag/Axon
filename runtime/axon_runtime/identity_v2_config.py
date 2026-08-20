"""Strict, side-effect-free parser for the isolated v2 identity contract.

Importing, constructing, or parsing this module never creates directories,
loads models, opens a database, or starts a runtime.  It validates a sealed
v2 identity descriptor and can build an in-memory v2 genesis snapshot only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from runtime.field.schema_v2 import (
    CHARTER_SCHEMA_V2,
    IdentityCharterV2,
    SharedFieldSnapshotV2,
    _reconstruct_charter,
    canonical_json_bytes,
    canonical_sha256,
)

IDENTITY_V2_CONTRACT_SCHEMA = "axon-runtime-identity-v2-contract-v1"
EXPECTED_BASE_V1_CONFIG_ID = (
    "17ce7dab76f466565e3c35214c27072aaa225945832ff803ff9575464ed27979"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LITERAL_SECRET_KEY_PARTS = (
    "secret",
    "password",
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "credential",
)


class IdentityV2ConfigError(ValueError):
    """The v2 identity contract descriptor is malformed or unsafe."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise IdentityV2ConfigError(f"{label} must be a string-keyed object")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise IdentityV2ConfigError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _nonempty(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise IdentityV2ConfigError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise IdentityV2ConfigError(
            f"{label} must be a 64-character lowercase hexadecimal SHA-256"
        )
    return value


def _has_forbidden_path_prefix(path_str: str) -> bool:
    """Reject Windows extended, device, and UNC/network path prefixes."""

    normalized = path_str.replace("/", "\\")
    return bool(
        normalized.startswith("\\\\?\\")
        or normalized.startswith("\\\\.\\")
        or normalized.startswith("\\\\")
    )


def _has_unsafe_path_component(path_str: str) -> bool:
    """Detect unsafe components: trailing spaces/dots or colons beyond the drive letter."""

    parts = re.split(r"[\\/]", path_str)
    for index, part in enumerate(parts):
        if part in ("", ".", ".."):
            continue
        if index == 0 and len(part) == 2 and part[1] == ":":
            continue
        if ":" in part:
            return True
        if part != part.rstrip(" ."):
            return True
    return False


def _absolute_path(value: Any, label: str) -> Path:
    try:
        raw_value = os.fspath(value) if isinstance(value, os.PathLike) else value
    except Exception as exc:
        raise IdentityV2ConfigError(
            f"{label} could not be converted to a path"
        ) from exc
    raw = _nonempty(raw_value, label)
    if type(raw) is not str:
        raise IdentityV2ConfigError(f"{label} must be a string")
    if "\x00" in raw or "\r" in raw or "\n" in raw:
        raise IdentityV2ConfigError(f"{label} contains forbidden control characters")

    if _has_forbidden_path_prefix(raw):
        raise IdentityV2ConfigError(
            f"{label} must be a safe local absolute path; "
            "UNC, extended, and device path prefixes are forbidden"
        )

    if _has_unsafe_path_component(raw):
        raise IdentityV2ConfigError(
            f"{label} contains an unsafe path component "
            "(trailing spaces/dots or alternate data stream colon)"
        )

    path = Path(raw)
    if not path.is_absolute():
        raise IdentityV2ConfigError(f"{label} must be absolute")

    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise IdentityV2ConfigError(f"{label} could not be resolved: {exc}") from exc

    if not resolved.is_absolute():
        raise IdentityV2ConfigError(f"{label} resolved to a relative path")

    if _has_forbidden_path_prefix(str(resolved)):
        raise IdentityV2ConfigError(
            f"{label} resolved to a forbidden UNC or extended path"
        )

    return resolved


def _reject_literal_secret_fields(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _LITERAL_SECRET_KEY_PARTS):
                raise IdentityV2ConfigError(
                    f"{path}.{key} is a forbidden literal secret field"
                )
            _reject_literal_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_literal_secret_fields(item, f"{path}[{index}]")


def _validate_state_root(path: Path) -> None:
    """Keep identity-v2 declarations from becoming a competing Axon State root."""

    try:
        retired_v1_root = Path(r"D:\Axon\State\axon_runtime").resolve(strict=False)
        axon_repo_root = Path(r"D:\Axon").resolve(strict=False)
        regression_root = Path(r"D:\Axon\State\training\regression").resolve(strict=False)
    except OSError as exc:
        raise IdentityV2ConfigError(
            f"could not resolve canonical Axon state boundaries: {exc}"
        ) from exc

    normalized_v1 = os.path.normcase(str(retired_v1_root))
    normalized_repo = os.path.normcase(str(axon_repo_root))
    normalized_regression = os.path.normcase(str(regression_root))
    normalized_candidate = os.path.normcase(str(path))

    if normalized_candidate == normalized_v1:
        raise IdentityV2ConfigError(
            "state_root must not be the protected v1 runtime root"
        )

    try:
        v1_common = os.path.commonpath((normalized_v1, normalized_candidate))
    except ValueError:
        v1_common = None

    if v1_common is not None and os.path.normcase(v1_common) == normalized_v1:
        raise IdentityV2ConfigError(
            "state_root must not be underneath the protected v1 runtime root"
        )

    if "identity_v2" not in path.name:
        raise IdentityV2ConfigError(
            "state_root final path component must contain 'identity_v2'"
        )

    # Pure parser tests may use temporary roots outside the Axon repository.
    # Any declaration inside D:\Axon, however, must remain an explicitly
    # non-authoritative regression branch beneath the one canonical State tree.
    try:
        repo_common = os.path.commonpath((normalized_repo, normalized_candidate))
    except ValueError:
        repo_common = None
    if repo_common is not None and os.path.normcase(repo_common) == normalized_repo:
        try:
            regression_common = os.path.commonpath(
                (normalized_regression, normalized_candidate)
            )
        except ValueError:
            regression_common = None
        if (
            regression_common is None
            or os.path.normcase(regression_common) != normalized_regression
            or normalized_candidate == normalized_regression
        ):
            raise IdentityV2ConfigError(
                "Axon identity-v2 state_root must be under "
                "D:\\Axon\\State\\training\\regression"
            )


def _freeze(value: Any) -> Any:
    """Return a deeply immutable view of a resolved-config value."""

    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class _VerifiedAuthority:
    """Immutable local authority record produced by _verify_bound_authority().

    Carries only primitive values derived from the bound descriptor authority.
    No mutable dict, list, or mapping is retained for later descriptor/genesis
    composition.
    """

    canonical_text: str
    contract_id: str
    base_v1_config_id: str
    charter_text: str


def _strict_json_loads(text: str | bytes, label: str = "bound descriptor") -> Any:
    """Parse JSON with duplicate-key and non-finite-constant rejection."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise IdentityV2ConfigError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise IdentityV2ConfigError(f"non-finite JSON constant {token}")

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except IdentityV2ConfigError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IdentityV2ConfigError(
            f"{label} text is not valid canonical JSON"
        ) from exc


@dataclass(frozen=True, slots=True)
class IdentityV2Contract:
    """A parsed, hash-bound v2 identity contract descriptor.

    The contract is a sealed authority: descriptor bytes and ``contract_id`` are
    bound at construction.  Public-field mutation or forged construction cannot
    make ``to_canonical_descriptor`` or ``build_genesis_snapshot`` emit material
    that contradicts the bound authority under an unchanged ``contract_id``.
    """

    base_v1_config_id: str
    state_root: Path
    field_schema: str
    charter: IdentityCharterV2
    sealed: bool
    physical_role: str
    envelope_char_budget: int
    migration_authority: str
    v1_auto_migration: bool
    contract_id: str = field(init=False)
    resolved_config: Mapping[str, Any] = field(init=False)
    _bound_descriptor_text: str = field(init=False, repr=False)
    _bound_descriptor_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is not IdentityV2Contract:
            raise IdentityV2ConfigError(
                "IdentityV2Contract does not permit subclass instances"
            )

        base_v1_config_id = _sha256(self.base_v1_config_id, "base_v1_config_id")
        if base_v1_config_id != EXPECTED_BASE_V1_CONFIG_ID:
            raise IdentityV2ConfigError(
                "base_v1_config_id does not match the current v1 smoke config"
            )

        state_root = _absolute_path(self.state_root, "state_root")
        _validate_state_root(state_root)

        field_schema = _nonempty(self.field_schema, "field_schema")
        if field_schema != "shared-field-v2":
            raise IdentityV2ConfigError("field_schema must be 'shared-field-v2'")

        try:
            safe_charter = _reconstruct_charter(self.charter)
        except (TypeError, ValueError) as exc:
            raise IdentityV2ConfigError(
                "charter must be a valid exact base IdentityCharterV2"
            ) from exc
        object.__setattr__(self, "charter", safe_charter)

        if type(self.sealed) is not bool or self.sealed is not True:
            raise IdentityV2ConfigError("sealed must be true")

        physical_role = _nonempty(self.physical_role, "physical_role")
        if physical_role != "context":
            raise IdentityV2ConfigError("physical_role must be 'context'")

        budget = self.envelope_char_budget
        if (
            type(budget) is not int
            or budget <= 0
            or budget > 192
        ):
            raise IdentityV2ConfigError(
                "envelope_char_budget must be an integer in (0, 192]"
            )

        if (
            type(self.v1_auto_migration) is not bool
            or self.v1_auto_migration is not False
        ):
            raise IdentityV2ConfigError("v1_auto_migration must be false")

        authority = _nonempty(self.migration_authority, "migration_authority")
        if authority != "human_authorized":
            raise IdentityV2ConfigError(
                "migration_authority must be 'human_authorized'"
            )

        bound_descriptor = self._build_bound_descriptor(
            base_v1_config_id=base_v1_config_id,
            state_root=state_root,
            field_schema=field_schema,
            charter=safe_charter,
            sealed=True,
            physical_role=physical_role,
            envelope_char_budget=budget,
            migration_authority=authority,
            v1_auto_migration=False,
        )
        descriptor_bytes = canonical_json_bytes(bound_descriptor)
        descriptor_hash = hashlib.sha256(descriptor_bytes).hexdigest()

        object.__setattr__(self, "base_v1_config_id", base_v1_config_id)
        object.__setattr__(self, "state_root", state_root)
        object.__setattr__(self, "field_schema", field_schema)
        object.__setattr__(self, "physical_role", physical_role)
        object.__setattr__(self, "envelope_char_budget", budget)
        object.__setattr__(self, "migration_authority", authority)
        object.__setattr__(
            self,
            "_bound_descriptor_text",
            descriptor_bytes.decode("utf-8"),
        )
        object.__setattr__(self, "_bound_descriptor_hash", descriptor_hash)
        object.__setattr__(self, "contract_id", descriptor_hash)
        object.__setattr__(
            self,
            "resolved_config",
            _freeze(self._build_resolved_config()),
        )

    @staticmethod
    def _build_bound_descriptor(
        *,
        base_v1_config_id: str,
        state_root: Path,
        field_schema: str,
        charter: IdentityCharterV2,
        sealed: bool,
        physical_role: str,
        envelope_char_budget: int,
        migration_authority: str,
        v1_auto_migration: bool,
    ) -> dict[str, Any]:
        return {
            "schema": IDENTITY_V2_CONTRACT_SCHEMA,
            "base_v1_config_id": base_v1_config_id,
            "field_schema": field_schema,
            "state_root": str(state_root),
            "identity": {
                "charter_schema": CHARTER_SCHEMA_V2,
                "charter_version": str(charter.charter_version),
                "charter_text": charter.text,
                "charter_hash": charter.charter_hash,
                "charter_max_chars": 512,
                "sealed": sealed,
                "physical_role": physical_role,
                "envelope_char_budget": envelope_char_budget,
            },
            "migration": {
                "authority": migration_authority,
                "v1_auto_migration": v1_auto_migration,
            },
        }

    def _verify_bound_authority(self) -> _VerifiedAuthority:
        if type(self) is not IdentityV2Contract:
            raise IdentityV2ConfigError(
                "IdentityV2Contract does not permit subclass instances"
            )
        try:
            bound_text = self._bound_descriptor_text
            bound_hash = self._bound_descriptor_hash
            contract_id = self.contract_id
        except AttributeError as exc:
            raise IdentityV2ConfigError(
                "IdentityV2Contract has no bound descriptor authority"
            ) from exc
        if (
            type(bound_text) is not str
            or type(bound_hash) is not str
            or type(contract_id) is not str
        ):
            raise IdentityV2ConfigError(
                "bound descriptor authority fields are not exact strings"
            )
        try:
            text_bytes = bound_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise IdentityV2ConfigError(
                "bound descriptor text is not valid UTF-8"
            ) from exc
        computed_hash = hashlib.sha256(text_bytes).hexdigest()
        if computed_hash != bound_hash or computed_hash != contract_id:
            raise IdentityV2ConfigError(
                "bound descriptor authority does not match contract_id"
            )
        parsed = _strict_json_loads(bound_text, label="bound descriptor")
        try:
            descriptor = _validate_descriptor_value(parsed)
        except IdentityV2ConfigError:
            raise
        except (TypeError, ValueError, AttributeError) as exc:
            raise IdentityV2ConfigError(
                "bound descriptor is malformed"
            ) from exc
        try:
            rebuilt_bytes = canonical_json_bytes(descriptor)
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise IdentityV2ConfigError(
                "bound descriptor could not be re-canonicalized"
            ) from exc
        if rebuilt_bytes != text_bytes:
            raise IdentityV2ConfigError(
                "bound descriptor text is not canonical"
            )
        return _VerifiedAuthority(
            canonical_text=bound_text,
            contract_id=contract_id,
            base_v1_config_id=descriptor["base_v1_config_id"],
            charter_text=descriptor["identity"]["charter_text"],
        )

    def to_canonical_descriptor(self) -> dict[str, Any]:
        """Return the bound canonical descriptor.

        The returned dict is a fresh copy derived from the verified bound
        authority; mutating it does not affect the contract's sealed authority.
        """

        verified = self._verify_bound_authority()
        parsed = _strict_json_loads(verified.canonical_text, label="bound descriptor")
        return _validate_descriptor_value(parsed)

    def _build_resolved_config(self) -> dict[str, Any]:
        return {
            "schema": "axon-resolved-identity-v2-config-v1",
            "contract_id": self.contract_id,
            "descriptor": self.to_canonical_descriptor(),
        }

    def build_genesis_snapshot(self) -> SharedFieldSnapshotV2:
        """Build the canonical v2 genesis snapshot in memory only.

        The charter text is taken from the bound descriptor authority, so later
        mutation of ``self.charter`` or any other public field cannot emit a
        different genesis under the original ``contract_id``.

        After ``_verify_bound_authority()`` returns, this method uses only the
        captured local primitive values in ``verified``; it does not read any
        ``self.*`` field before returning or failing.
        """

        verified = self._verify_bound_authority()
        safe_charter = IdentityCharterV2(
            text=verified.charter_text, charter_version=1
        )
        extra_manifests = (
            f"{IDENTITY_V2_CONTRACT_SCHEMA}:{verified.contract_id}",
            f"axon-runtime-bootstrap-config-v1:{verified.base_v1_config_id}",
        )
        return IdentityCharterV2.build_genesis_snapshot(
            safe_charter, source_manifest_ids=extra_manifests
        )


def _validate_descriptor_value(value: Any) -> dict[str, Any]:
    """Validate a parsed v2 identity contract descriptor.

    Returns the canonical descriptor dict produced by ``_build_bound_descriptor``.
    All malformed shape/type/key/index errors normalize to
    IdentityV2ConfigError.
    """

    value = _mapping(value, "contract")

    _reject_literal_secret_fields(value)

    _exact_keys(
        value,
        {
            "schema",
            "base_v1_config_id",
            "field_schema",
            "state_root",
            "identity",
            "migration",
        },
        "contract",
    )

    if value["schema"] != IDENTITY_V2_CONTRACT_SCHEMA:
        raise IdentityV2ConfigError("unknown identity v2 contract schema")

    base_v1_config_id = _sha256(value["base_v1_config_id"], "base_v1_config_id")
    if base_v1_config_id != EXPECTED_BASE_V1_CONFIG_ID:
        raise IdentityV2ConfigError(
            "base_v1_config_id does not match the current v1 smoke config"
        )

    field_schema = _nonempty(value["field_schema"], "field_schema")
    if field_schema != "shared-field-v2":
        raise IdentityV2ConfigError("field_schema must be 'shared-field-v2'")

    state_root = _absolute_path(value["state_root"], "state_root")
    _validate_state_root(state_root)

    identity = _mapping(value["identity"], "identity")
    _exact_keys(
        identity,
        {
            "charter_schema",
            "charter_version",
            "charter_text",
            "charter_hash",
            "charter_max_chars",
            "sealed",
            "physical_role",
            "envelope_char_budget",
        },
        "identity",
    )

    if identity["charter_schema"] != CHARTER_SCHEMA_V2:
        raise IdentityV2ConfigError("unsupported charter_schema")

    if identity["charter_max_chars"] != 512:
        raise IdentityV2ConfigError("charter_max_chars must be 512")

    if type(identity["sealed"]) is not bool or identity["sealed"] is not True:
        raise IdentityV2ConfigError("identity.sealed must be true")

    physical_role = _nonempty(identity["physical_role"], "identity.physical_role")
    if physical_role != "context":
        raise IdentityV2ConfigError("identity.physical_role must be 'context'")

    envelope_char_budget = identity["envelope_char_budget"]
    if (
        isinstance(envelope_char_budget, bool)
        or type(envelope_char_budget) is not int
        or envelope_char_budget <= 0
        or envelope_char_budget > 192
    ):
        raise IdentityV2ConfigError(
            "envelope_char_budget must be an integer in (0, 192]"
        )

    charter_text = _nonempty(identity["charter_text"], "charter_text")
    charter_version = identity["charter_version"]
    if charter_version != "1":
        raise IdentityV2ConfigError(
            "charter_version must be the raw string '1'"
        )

    try:
        charter = IdentityCharterV2(text=charter_text, charter_version=1)
    except (TypeError, ValueError) as exc:
        raise IdentityV2ConfigError(
            "charter must be a valid exact base IdentityCharterV2"
        ) from exc

    declared_hash = _sha256(identity["charter_hash"], "charter_hash")
    if declared_hash != charter.charter_hash:
        raise IdentityV2ConfigError(
            "charter_hash does not match canonical charter hash"
        )

    migration = _mapping(value["migration"], "migration")
    _exact_keys(migration, {"authority", "v1_auto_migration"}, "migration")

    authority = _nonempty(migration["authority"], "migration.authority")
    if authority != "human_authorized":
        raise IdentityV2ConfigError(
            "migration.authority must be 'human_authorized'"
        )

    v1_auto_migration = migration["v1_auto_migration"]
    if type(v1_auto_migration) is not bool or v1_auto_migration is not False:
        raise IdentityV2ConfigError("migration.v1_auto_migration must be false")

    try:
        return IdentityV2Contract._build_bound_descriptor(
            base_v1_config_id=base_v1_config_id,
            state_root=state_root,
            field_schema=field_schema,
            charter=charter,
            sealed=True,
            physical_role=physical_role,
            envelope_char_budget=envelope_char_budget,
            migration_authority=authority,
            v1_auto_migration=False,
        )
    except (TypeError, ValueError) as exc:
        raise IdentityV2ConfigError(
            "could not rebuild canonical descriptor"
        ) from exc


def parse_identity_v2_contract(text: str | bytes) -> IdentityV2Contract:
    """Parse strict JSON, rejecting duplicate keys and non-finite numbers."""

    parsed = _strict_json_loads(text, label="identity v2 contract")
    try:
        descriptor = _validate_descriptor_value(parsed)
    except IdentityV2ConfigError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise IdentityV2ConfigError(
            "identity v2 contract descriptor is malformed"
        ) from exc

    return IdentityV2Contract(
        base_v1_config_id=descriptor["base_v1_config_id"],
        state_root=Path(descriptor["state_root"]),
        field_schema=descriptor["field_schema"],
        charter=IdentityCharterV2(
            text=descriptor["identity"]["charter_text"],
            charter_version=1,
        ),
        sealed=True,
        physical_role="context",
        envelope_char_budget=descriptor["identity"]["envelope_char_budget"],
        migration_authority="human_authorized",
        v1_auto_migration=False,
    )


def load_identity_v2_contract(path: str | Path) -> IdentityV2Contract:
    """Read and parse one contract file; perform no other filesystem work."""

    try:
        source = Path(path)
    except Exception as exc:
        raise IdentityV2ConfigError(
            "contract path could not be converted"
        ) from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    return parse_identity_v2_contract(source.read_bytes())


__all__ = [
    "IDENTITY_V2_CONTRACT_SCHEMA",
    "EXPECTED_BASE_V1_CONFIG_ID",
    "IdentityV2ConfigError",
    "IdentityV2Contract",
    "parse_identity_v2_contract",
    "load_identity_v2_contract",
]
