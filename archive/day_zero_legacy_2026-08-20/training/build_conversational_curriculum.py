#!/usr/bin/env python3
"""Build a deterministic, public-only conversational curriculum.

The builder is intentionally a pure transformation until
``build_conversational_curriculum`` (or the CLI) is called.  Importing this
module performs no filesystem writes and never accesses a network service.

Input contract
==============

The input is UTF-8 JSONL with one *complete conversation* per source record::

    {
      "schema": "axon_public_conversation_source_v1",
      "source_id": "public-conversation-001",
      "source_sha256": "...",
      "license": {
        "spdx_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "redistribution_allowed": true,
        "derivatives_allowed": true,
        "commercial_use_allowed": true
      },
      "provenance": {
        "publisher": "Example publisher",
        "source_name": "Example public corpus",
        "source_url": "https://example.org/corpus/001",
        "retrieved_utc": "2026-07-28T00:00:00Z"
      },
      "privacy": {
        "classification": "public",
        "local_only": false,
        "cloud_export_allowed": true,
        "contains_personal_data": false
      },
      "conversation": [
        {"role": "user", "text": "Hello."},
        {"role": "assistant", "text": "Hello there."}
      ]
    }

``source_sha256`` is SHA-256 over canonical JSON for the complete record with
the ``source_sha256`` member omitted.  Use :func:`source_record_sha256` to
produce it.

Messages alternate ``user`` and ``assistant`` and a complete conversation
ends with an assistant turn.  ``kind`` is optional for ordinary messages.
An assistant message may instead explicitly use
``"kind": "structural_noop"``.  Its non-empty ``text`` is an audit reason,
not a blank response target.

Every ordinary assistant turn becomes one complete ``response_draft``
replacement of 1--64 characters.  All ten canonical field regions are
present.  The current gold answer is never copied into a visible region.
Complete source conversations are assigned to train/dev/test *before*
examples are derived, so no source conversation can cross splits.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import sys
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.field import CANONICAL_REGION_ORDER  # noqa: E402
from substrate import (  # noqa: E402
    assert_supported_text,
    default_alphabet,
    get_letter_bank,
)


SOURCE_SCHEMA = "axon_public_conversation_source_v1"
EXAMPLE_SCHEMA = "axon_conversational_field_example_v1"
MANIFEST_SCHEMA = "axon_conversational_curriculum_manifest_v1"
BUILDER_VERSION = "1"

EXPECTED_CANONICAL_REGIONS: tuple[str, ...] = (
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
)
CANONICAL_REGIONS: tuple[str, ...] = tuple(
    region.value for region in CANONICAL_REGION_ORDER
)
if CANONICAL_REGIONS != EXPECTED_CANONICAL_REGIONS:
    raise RuntimeError(
        "runtime canonical regions drifted from the frozen ten-region "
        "conversational curriculum contract"
    )
RESPONSE_REGION = "response_draft"
MAX_TARGET_CHARS = 64
SPLIT_MODULUS = 10_000
SPLIT_THRESHOLDS: tuple[tuple[str, int], ...] = (
    ("train", 8_000),
    ("dev", 9_000),
    ("test", SPLIT_MODULUS),
)
SPLITS: tuple[str, ...] = tuple(item[0] for item in SPLIT_THRESHOLDS)
EXPORTABLE_SPDX_IDS = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "CC0-1.0",
        "ISC",
        "MIT",
        "ODC-BY-1.0",
        "PDDL-1.0",
        "Unlicense",
    }
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "source_id",
        "source_sha256",
        "license",
        "provenance",
        "privacy",
        "conversation",
    }
)
_LICENSE_KEYS = frozenset(
    {
        "spdx_id",
        "license_url",
        "redistribution_allowed",
        "derivatives_allowed",
        "commercial_use_allowed",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "publisher",
        "source_name",
        "source_url",
        "retrieved_utc",
    }
)
_PRIVACY_KEYS = frozenset(
    {
        "classification",
        "local_only",
        "cloud_export_allowed",
        "contains_personal_data",
    }
)
_MESSAGE_KEYS = frozenset({"role", "kind", "text"})
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
_SPDX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,127}$")
_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_D00_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])D:[\\/]+00(?:[\\/]|$)",
    re.IGNORECASE,
)
_RELATIVE_PATH_SEGMENT_RE = re.compile(r"(?:^|[\\/])\.{1,2}(?:[\\/]|$)")
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")


class CurriculumContractError(ValueError):
    """Raised when a source or derived example violates the curriculum contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes without a trailing newline."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return an uppercase SHA-256 digest."""

    return hashlib.sha256(value).hexdigest().upper()


def source_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash a source record, excluding its self-referential hash member."""

    payload = dict(record)
    payload.pop("source_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    actual = set(value)
    missing = expected - actual - optional
    extra = actual - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if extra:
            details.append(f"unexpected={sorted(extra)!r}")
        raise CurriculumContractError(
            f"{label} must use the exact source contract ({', '.join(details)})"
        )


def _expect_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CurriculumContractError(f"{label} must be a JSON object")
    return value


def _expect_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurriculumContractError(f"{label} must be a non-empty string")
    if not value.strip(" \n"):
        raise CurriculumContractError(
            f"{label} must contain a non-whitespace character"
        )
    return value


def _expect_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CurriculumContractError(f"{label} must be a boolean")
    return value


def _validate_public_url(value: Any, label: str) -> str:
    url = _expect_nonempty_string(value, label)
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise CurriculumContractError(
            f"{label} must be a public http or https URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise CurriculumContractError(
            f"{label} must not contain embedded credentials"
        )
    try:
        parsed.port
    except ValueError as exc:
        raise CurriculumContractError(f"{label} contains an invalid port") from exc
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith(
        (".local", ".internal", ".lan", ".home")
    ):
        raise CurriculumContractError(f"{label} must not identify a local host")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
    ):
        raise CurriculumContractError(f"{label} must not identify a private host")
    if address is None and "." not in hostname:
        raise CurriculumContractError(
            f"{label} must not identify a single-label private host"
        )
    host_labels = hostname.split(".")
    legacy_numeric_host = bool(host_labels) and all(
        re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]+)", host_label)
        for host_label in host_labels
    )
    if address is None and legacy_numeric_host:
        raise CurriculumContractError(
            f"{label} must not use a noncanonical numeric host"
        )
    return url


def _path_like_provenance_violation(
    value: Any,
    *,
    key_path: tuple[str, ...] = (),
) -> str | None:
    """Return a violation for D:\\00 or private/local path provenance."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            label = str(key)
            lowered = label.casefold()
            nested_path = (*key_path, label)
            if (
                any(token in lowered for token in ("path", "directory"))
                or lowered in {"file", "filename", "root"}
            ) and nested not in (None, ""):
                return ".".join(nested_path)
            violation = _path_like_provenance_violation(
                nested,
                key_path=nested_path,
            )
            if violation is not None:
                return violation
        return None
    if isinstance(value, list):
        for index, nested in enumerate(value):
            violation = _path_like_provenance_violation(
                nested,
                key_path=(*key_path, str(index)),
            )
            if violation is not None:
                return violation
        return None
    if not isinstance(value, str):
        return None

    if _D00_RE.search(value):
        return ".".join(key_path) or "<value>"
    lowered = value.casefold()
    field_name = key_path[-1].casefold() if key_path else ""
    if (
        lowered.startswith("file:")
        or _WINDOWS_ABSOLUTE_RE.match(value)
        or value.startswith("\\\\")
        or value.startswith("/")
        or _DRIVE_PREFIX_RE.match(value)
        or _RELATIVE_PATH_SEGMENT_RE.search(value)
        or lowered.startswith(("~/", "~\\"))
        or (
            field_name != "source_url"
            and ("\\" in value or "/" in value)
        )
    ):
        return ".".join(key_path) or "<value>"
    return None


def _validate_license(value: Any) -> dict[str, Any]:
    license_record = _expect_object(value, "license")
    _expect_exact_keys(license_record, _LICENSE_KEYS, "license")
    spdx_id = _expect_nonempty_string(
        license_record["spdx_id"],
        "license.spdx_id",
    )
    if not _SPDX_RE.fullmatch(spdx_id):
        raise CurriculumContractError(
            "license.spdx_id must be an SPDX-style identifier"
        )
    if spdx_id not in EXPORTABLE_SPDX_IDS:
        raise CurriculumContractError(
            "license.spdx_id is not in the reviewed exportable-license allowlist"
        )
    license_url = _validate_public_url(
        license_record["license_url"],
        "license.license_url",
    )
    redistribution_allowed = _expect_bool(
        license_record["redistribution_allowed"],
        "license.redistribution_allowed",
    )
    derivatives_allowed = _expect_bool(
        license_record["derivatives_allowed"],
        "license.derivatives_allowed",
    )
    commercial_use_allowed = _expect_bool(
        license_record["commercial_use_allowed"],
        "license.commercial_use_allowed",
    )
    if not redistribution_allowed or not derivatives_allowed:
        raise CurriculumContractError(
            "license must explicitly allow redistribution and derivative works"
        )
    if not commercial_use_allowed:
        raise CurriculumContractError(
            "license must explicitly allow commercial use for exportable artifacts"
        )
    return {
        "spdx_id": spdx_id,
        "license_url": license_url,
        "redistribution_allowed": redistribution_allowed,
        "derivatives_allowed": derivatives_allowed,
        "commercial_use_allowed": commercial_use_allowed,
    }


def _validate_provenance(value: Any) -> dict[str, str]:
    provenance = _expect_object(value, "provenance")
    violation = _path_like_provenance_violation(provenance)
    if violation is not None:
        raise CurriculumContractError(
            "private or local path provenance is forbidden "
            f"(at provenance.{violation})"
        )
    _expect_exact_keys(provenance, _PROVENANCE_KEYS, "provenance")
    publisher = _expect_nonempty_string(
        provenance["publisher"],
        "provenance.publisher",
    )
    source_name = _expect_nonempty_string(
        provenance["source_name"],
        "provenance.source_name",
    )
    source_url = _validate_public_url(
        provenance["source_url"],
        "provenance.source_url",
    )
    retrieved_utc = _expect_nonempty_string(
        provenance["retrieved_utc"],
        "provenance.retrieved_utc",
    )
    if not _UTC_RE.fullmatch(retrieved_utc):
        raise CurriculumContractError(
            "provenance.retrieved_utc must be an ISO-8601 UTC timestamp ending Z"
        )
    try:
        datetime.fromisoformat(retrieved_utc[:-1] + "+00:00")
    except ValueError as exc:
        raise CurriculumContractError(
            "provenance.retrieved_utc is not a valid timestamp"
        ) from exc
    return {
        "publisher": publisher,
        "source_name": source_name,
        "source_url": source_url,
        "retrieved_utc": retrieved_utc,
    }


def _validate_privacy(value: Any) -> dict[str, Any]:
    privacy = _expect_object(value, "privacy")
    _expect_exact_keys(privacy, _PRIVACY_KEYS, "privacy")
    classification = _expect_nonempty_string(
        privacy["classification"],
        "privacy.classification",
    )
    local_only = _expect_bool(privacy["local_only"], "privacy.local_only")
    cloud_export_allowed = _expect_bool(
        privacy["cloud_export_allowed"],
        "privacy.cloud_export_allowed",
    )
    contains_personal_data = _expect_bool(
        privacy["contains_personal_data"],
        "privacy.contains_personal_data",
    )
    if classification != "public":
        raise CurriculumContractError(
            "privacy.classification must be exactly 'public'"
        )
    if local_only:
        raise CurriculumContractError("local_only source records are forbidden")
    if not cloud_export_allowed:
        raise CurriculumContractError(
            "source records with cloud_export_allowed=false are forbidden"
        )
    if contains_personal_data:
        raise CurriculumContractError(
            "public curriculum sources must declare contains_personal_data=false"
        )
    return {
        "classification": classification,
        "local_only": local_only,
        "cloud_export_allowed": cloud_export_allowed,
        "contains_personal_data": contains_personal_data,
    }


def _validate_substrate_text(text: Any, label: str) -> str:
    validated = _expect_nonempty_string(text, label)
    try:
        assert_supported_text(validated)
    except ValueError as exc:
        unsupported = sorted({ord(char) for char in validated if ord(char) > 127})
        suffix = (
            f"; unsupported code points={unsupported!r}" if unsupported else ""
        )
        raise CurriculumContractError(
            f"{label} violates the exact 95-character substrate{suffix}"
        ) from exc
    return validated


def _substrate_contract() -> dict[str, Any]:
    """Audit-bind the exact ordered character and decoder-class contracts."""

    alphabet = tuple(default_alphabet())
    bank = get_letter_bank()
    bank_chars = tuple(bank.chars)
    if len(alphabet) != 95 or len(set(alphabet)) != 95:
        raise CurriculumContractError(
            "substrate alphabet must contain exactly 95 unique characters"
        )
    if bank.empty_index != 95:
        raise CurriculumContractError(
            "letter-bank empty class must remain at index 95"
        )
    if len(bank_chars) != 96 or bank_chars[:95] != alphabet:
        raise CurriculumContractError(
            "letter-bank class order no longer extends the frozen alphabet"
        )
    if bank_chars[95] != "<empty>":
        raise CurriculumContractError(
            "letter-bank class 95 must remain <empty>"
        )
    return {
        "character_count": len(alphabet),
        "ordered_alphabet_sha256": sha256_bytes(
            canonical_json_bytes(list(alphabet))
        ),
        "decoder_class_count": len(bank_chars),
        "ordered_letter_bank_sha256": sha256_bytes(
            canonical_json_bytes(list(bank_chars))
        ),
        "empty_index": bank.empty_index,
    }


def _validate_conversation(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise CurriculumContractError(
            "conversation must be a non-empty JSON array"
        )
    if len(value) % 2:
        raise CurriculumContractError(
            "complete conversation must end with an assistant turn"
        )

    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(value):
        label = f"conversation[{index}]"
        message = _expect_object(raw_message, label)
        _expect_exact_keys(
            message,
            _MESSAGE_KEYS,
            label,
            optional=frozenset({"kind"}),
        )
        role = _expect_nonempty_string(message["role"], f"{label}.role")
        expected_role = "user" if index % 2 == 0 else "assistant"
        if role != expected_role:
            raise CurriculumContractError(
                f"{label}.role must be {expected_role!r}; "
                "conversation roles must alternate"
            )
        default_kind = "utterance" if role == "user" else "response"
        kind = message.get("kind", default_kind)
        if not isinstance(kind, str):
            raise CurriculumContractError(f"{label}.kind must be a string")
        if role == "user" and kind != "utterance":
            raise CurriculumContractError(
                f"{label}.kind must be 'utterance' for user messages"
            )
        if role == "assistant" and kind not in {
            "response",
            "structural_noop",
        }:
            raise CurriculumContractError(
                f"{label}.kind must be 'response' or 'structural_noop'"
            )
        text_label = (
            f"{label}.structural_noop_reason"
            if kind == "structural_noop"
            else f"{label}.text"
        )
        text = _validate_substrate_text(message["text"], text_label)
        if role == "assistant" and len(text) > MAX_TARGET_CHARS:
            raise CurriculumContractError(
                f"{text_label} has {len(text)} characters; "
                f"the complete target limit is {MAX_TARGET_CHARS}"
            )
        messages.append({"role": role, "kind": kind, "text": text})
    return messages


def validate_source_record(
    raw_record: Mapping[str, Any],
    *,
    line_number: int | None = None,
) -> dict[str, Any]:
    """Validate and return a normalized copy of one complete source record."""

    line_label = f"line {line_number}" if line_number is not None else "source"
    record = _expect_object(raw_record, line_label)
    _expect_exact_keys(record, _TOP_LEVEL_KEYS, line_label)
    if record["schema"] != SOURCE_SCHEMA:
        raise CurriculumContractError(
            f"{line_label}.schema must be {SOURCE_SCHEMA!r}"
        )
    source_id = _expect_nonempty_string(
        record["source_id"],
        f"{line_label}.source_id",
    )
    if not _SOURCE_ID_RE.fullmatch(source_id):
        raise CurriculumContractError(
            f"{line_label}.source_id must be an ASCII identifier"
        )
    claimed_hash = _expect_nonempty_string(
        record["source_sha256"],
        f"{line_label}.source_sha256",
    )
    if not _SHA256_RE.fullmatch(claimed_hash):
        raise CurriculumContractError(
            f"{line_label}.source_sha256 must contain 64 hexadecimal digits"
        )

    license_record = _validate_license(record["license"])
    provenance = _validate_provenance(record["provenance"])
    privacy = _validate_privacy(record["privacy"])
    conversation = _validate_conversation(record["conversation"])

    actual_hash = source_record_sha256(record)
    if claimed_hash.upper() != actual_hash:
        raise CurriculumContractError(
            f"{line_label}.source_sha256 mismatch: "
            f"claimed {claimed_hash.upper()}, computed {actual_hash}"
        )
    conversation_sha256 = sha256_bytes(canonical_json_bytes(conversation))
    return {
        "schema": SOURCE_SCHEMA,
        "source_id": source_id,
        "source_sha256": actual_hash,
        "conversation_sha256": conversation_sha256,
        "license": license_record,
        "provenance": provenance,
        "privacy": privacy,
        "conversation": conversation,
    }


def split_for_conversation(conversation_sha256: str) -> str:
    """Assign one complete conversation to a stable source-level split."""

    if not isinstance(conversation_sha256, str) or not _SHA256_RE.fullmatch(
        conversation_sha256
    ):
        raise CurriculumContractError(
            "conversation_sha256 must contain 64 hexadecimal digits"
        )
    bucket = int(conversation_sha256[:16], 16) % SPLIT_MODULUS
    for split, upper_bound in SPLIT_THRESHOLDS:
        if bucket < upper_bound:
            return split
    raise AssertionError("split thresholds do not cover the modulus")


def _parse_json_object(line: str, *, line_number: int) -> Mapping[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CurriculumContractError(
                    f"line {line_number} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CurriculumContractError(
            f"line {line_number} contains non-finite JSON value {value}"
        )

    try:
        value = json.loads(
            line,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except CurriculumContractError:
        raise
    except json.JSONDecodeError as exc:
        raise CurriculumContractError(
            f"line {line_number} is not valid JSON: {exc.msg}"
        ) from exc
    return _expect_object(value, f"line {line_number}")


def load_source_records(input_jsonl: str | Path) -> tuple[list[dict[str, Any]], bytes]:
    """Read and validate all sources without deriving any training row."""

    path = Path(input_jsonl)
    _reject_d00_argument(path, "input_jsonl")
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise CurriculumContractError(
            f"unable to read input_jsonl {path}: {exc}"
        ) from exc
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CurriculumContractError(
            "input_jsonl must be strict UTF-8"
        ) from exc
    if source_text.startswith("\ufeff"):
        raise CurriculumContractError("input_jsonl must not contain a UTF-8 BOM")
    if not source_text:
        raise CurriculumContractError("input_jsonl must not be empty")

    records: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    source_hashes: set[str] = set()
    for line_number, line in enumerate(source_text.splitlines(), start=1):
        if not line:
            raise CurriculumContractError(
                f"line {line_number} is blank; blank JSONL rows are forbidden"
            )
        record = validate_source_record(
            _parse_json_object(line, line_number=line_number),
            line_number=line_number,
        )
        if record["source_id"] in source_ids:
            raise CurriculumContractError(
                f"duplicate source_id {record['source_id']!r}"
            )
        if record["source_sha256"] in source_hashes:
            raise CurriculumContractError(
                f"duplicate source_sha256 {record['source_sha256']}"
            )
        source_ids.add(record["source_id"])
        source_hashes.add(record["source_sha256"])
        records.append(record)
    if not records:
        raise CurriculumContractError(
            "input_jsonl must contain at least one source record"
        )
    return records, source_bytes


def _render_history(messages: Sequence[Mapping[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        if message["kind"] == "structural_noop":
            continue
        role_label = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {message['text']}")
    rendered = "\n".join(lines)
    if rendered:
        try:
            assert_supported_text(rendered)
        except ValueError as exc:  # Defensive: labels are frozen substrate text.
            raise CurriculumContractError(
                "rendered conversation history left the exact substrate"
            ) from exc
    return rendered


def _blank_active_field() -> dict[str, str]:
    return {region: "" for region in CANONICAL_REGIONS}


def _assert_gold_hidden(
    gold: str,
    active_field: Mapping[str, str],
    *,
    source_id: str,
    assistant_turn_index: int,
) -> None:
    leaked_regions = [
        region
        for region, text in active_field.items()
        if gold and gold in text
    ]
    if leaked_regions:
        raise CurriculumContractError(
            "gold target appears in visible field for "
            f"{source_id} assistant turn {assistant_turn_index}: "
            f"{leaked_regions!r}"
        )


def _example_id(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _derive_examples(
    records: Sequence[Mapping[str, Any]],
    source_splits: Mapping[str, str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (item["conversation_sha256"], item["source_id"]),
    ):
        source_id = str(record["source_id"])
        split = source_splits[source_id]
        messages = record["conversation"]
        derived_index = 0
        for assistant_index in range(1, len(messages), 2):
            user_message = messages[assistant_index - 1]
            assistant_message = messages[assistant_index]
            active_field = _blank_active_field()
            active_field["conversation_history"] = _render_history(
                messages[: assistant_index - 1]
            )
            active_field["user_input"] = user_message["text"]
            gold = assistant_message["text"]
            _assert_gold_hidden(
                gold,
                active_field,
                source_id=source_id,
                assistant_turn_index=assistant_index,
            )

            base: dict[str, Any] = {
                "schema": EXAMPLE_SCHEMA,
                "example_kind": (
                    "structural_noop"
                    if assistant_message["kind"] == "structural_noop"
                    else "response_draft_replacement"
                ),
                "split": split,
                "source": {
                    "source_id": source_id,
                    "source_sha256": record["source_sha256"],
                    "conversation_sha256": record["conversation_sha256"],
                    "assistant_turn_index": assistant_index,
                    "derived_example_index": derived_index,
                    "license": deepcopy(record["license"]),
                    "provenance": deepcopy(record["provenance"]),
                    "privacy": deepcopy(record["privacy"]),
                },
                "active_field": active_field,
                "gold_exclusion_audit": {
                    "passed": True,
                    "visible_region_names": list(CANONICAL_REGIONS),
                    "gold_present_in_visible_regions": [],
                },
            }
            if assistant_message["kind"] == "structural_noop":
                base["target_delta"] = None
                base["structural_noop"] = {
                    "op": "no_op",
                    "reason": gold,
                    "reason_char_count": len(gold),
                }
            else:
                base["target_delta"] = {
                    "op": "replace",
                    "region": RESPONSE_REGION,
                    "text": gold,
                    "char_count": len(gold),
                    "complete_replacement": True,
                }
                base["structural_noop"] = None
            base["example_id"] = _example_id(base)
            examples.append(base)
            derived_index += 1
    return examples


def _reject_d00_argument(path: Path, label: str) -> None:
    # Check both the caller spelling and the normalized resolved location.
    # Otherwise a path such as D:\Axon\..\00\source.jsonl (or a junction
    # resolving there) can bypass a lexical D:\00 check.
    candidates = (str(path), _resolved(path))
    if any(_D00_RE.search(candidate) for candidate in candidates):
        raise CurriculumContractError(
            f"{label} must not access the private D:\\00 tree"
        )


def _resolved(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _write_artifact_pair_atomic(
    output_path: Path,
    output_payload: bytes,
    manifest_path: Path,
    manifest_payload: bytes,
) -> None:
    """Install output and manifest as one rollback-safe logical pair.

    Unique, exclusively created sibling temporaries prevent a caller-controlled
    input filename from aliasing an internal staging path. Existing artifacts
    are moved to unique backups before either replacement is installed. Any
    ordinary process-level failure restores the previous pair or removes the
    newly installed partial pair.
    """

    artifacts = (
        (output_path, output_payload),
        (manifest_path, manifest_payload),
    )
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temporaries: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []

    try:
        for destination, payload in artifacts:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not destination.is_file():
                raise CurriculumContractError(
                    f"artifact destination is not a regular file: {destination}"
                )
            temporary = destination.with_name(
                f".{destination.name}.axon-tmp-{token}"
            )
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporaries[destination] = temporary

        for destination, _ in artifacts:
            if destination.exists():
                backup = destination.with_name(
                    f".{destination.name}.axon-backup-{token}"
                )
                os.replace(destination, backup)
                backups[destination] = backup

        for destination, _ in artifacts:
            os.replace(temporaries[destination], destination)
            installed.append(destination)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for destination in reversed(installed):
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"remove partial {destination}: {rollback_exc}"
                )
        for destination, backup in reversed(tuple(backups.items())):
            try:
                os.replace(backup, destination)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"restore {destination} from {backup}: {rollback_exc}"
                )
        if rollback_errors:
            raise CurriculumContractError(
                "artifact-pair installation failed and rollback was incomplete; "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink()
            except FileNotFoundError:
                pass
    finally:
        for temporary in temporaries.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _build_manifest(
    *,
    input_bytes: bytes,
    output_bytes: bytes,
    records: Sequence[Mapping[str, Any]],
    source_splits: Mapping[str, str],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sources_by_split = {
        split: sum(source_splits[record["source_id"]] == split for record in records)
        for split in SPLITS
    }
    examples_by_split = {
        split: sum(example["split"] == split for example in examples)
        for split in SPLITS
    }
    replacements = sum(
        example["example_kind"] == "response_draft_replacement"
        for example in examples
    )
    noops = sum(
        example["example_kind"] == "structural_noop"
        for example in examples
    )
    source_manifest = []
    for record in sorted(records, key=lambda item: item["source_id"]):
        source_manifest.append(
            {
                "source_id": record["source_id"],
                "source_sha256": record["source_sha256"],
                "conversation_sha256": record["conversation_sha256"],
                "split": source_splits[record["source_id"]],
                "derived_example_count": sum(
                    example["source"]["source_id"] == record["source_id"]
                    for example in examples
                ),
                "license": deepcopy(record["license"]),
                "provenance": deepcopy(record["provenance"]),
                "privacy": deepcopy(record["privacy"]),
            }
        )

    manifest_without_hash: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "source_contract": {
            "schema": SOURCE_SCHEMA,
            "complete_conversation_before_split": True,
            "public_only": True,
            "local_only_allowed": False,
            "cloud_export_required": True,
            "private_path_provenance_allowed": False,
            "source_hash_contract": (
                "SHA-256 uppercase canonical JSON with source_sha256 omitted"
            ),
        },
        "example_contract": {
            "schema": EXAMPLE_SCHEMA,
            "canonical_regions": list(CANONICAL_REGIONS),
            "target_region": RESPONSE_REGION,
            "target_operation": "complete replacement",
            "target_char_range": [1, MAX_TARGET_CHARS],
            "structural_noop_is_separate": True,
            "empty_answer_targets_allowed": False,
            "gold_visible_in_active_field": False,
            "unsupported_character_policy": "reject without substitution",
            "substrate": _substrate_contract(),
        },
        "split_policy": {
            "unit": "complete source conversation",
            "assignment_key": "conversation_sha256",
            "algorithm": (
                "first 64 hash bits modulo 10000; "
                "train 0-7999, dev 8000-8999, test 9000-9999"
            ),
            "thresholds": {
                "train": [0, 7_999],
                "dev": [8_000, 8_999],
                "test": [9_000, 9_999],
            },
        },
        "artifacts": {
            "input_jsonl": {
                "bytes": len(input_bytes),
                "sha256": sha256_bytes(input_bytes),
            },
            "examples_jsonl": {
                "bytes": len(output_bytes),
                "sha256": sha256_bytes(output_bytes),
            },
        },
        "realized_counts": {
            "sources": len(records),
            "conversations": len(records),
            "examples": len(examples),
            "response_draft_replacements": replacements,
            "structural_noops": noops,
            "sources_by_split": sources_by_split,
            "examples_by_split": examples_by_split,
        },
        "sources": source_manifest,
    }
    manifest = deepcopy(manifest_without_hash)
    manifest["manifest_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(manifest_without_hash)
    )
    return manifest


def build_conversational_curriculum(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build deterministic examples and a hash-complete manifest.

    Source records are all validated and assigned a source-level split before
    any training example is derived.  No output is written if validation,
    derivation, leakage checking, or source immutability checking fails.
    """

    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    if manifest_path is None:
        manifest_output = output_path.with_name(
            f"{output_path.stem}.manifest.json"
        )
    else:
        manifest_output = Path(manifest_path)

    _reject_d00_argument(input_path, "input_jsonl")
    _reject_d00_argument(output_path, "output_jsonl")
    _reject_d00_argument(manifest_output, "manifest_path")
    resolved_paths = {
        _resolved(input_path),
        _resolved(output_path),
        _resolved(manifest_output),
    }
    if len(resolved_paths) != 3:
        raise CurriculumContractError(
            "input_jsonl, output_jsonl, and manifest_path must be distinct"
        )

    # Phase 1: validate every complete source conversation.
    records, input_bytes = load_source_records(input_path)

    # Phase 2: assign every source before deriving any assistant-turn row.
    source_splits = {
        record["source_id"]: split_for_conversation(
            record["conversation_sha256"]
        )
        for record in records
    }

    # Phase 3: derive deterministic examples and enforce visible-gold absence.
    examples = _derive_examples(records, source_splits)
    if not examples:
        raise CurriculumContractError(
            "validated sources did not yield any examples"
        )
    output_bytes = b"".join(
        canonical_json_bytes(example) + b"\n" for example in examples
    )
    manifest = _build_manifest(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        records=records,
        source_splits=source_splits,
        examples=examples,
    )
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    # Verify the input remained byte-identical before committing artifacts.
    try:
        after_bytes = input_path.read_bytes()
    except OSError as exc:
        raise CurriculumContractError(
            f"unable to re-verify input_jsonl {input_path}: {exc}"
        ) from exc
    if after_bytes != input_bytes:
        raise CurriculumContractError(
            "input_jsonl changed during the build; no artifact was written"
        )

    _write_artifact_pair_atomic(
        output_path,
        output_bytes,
        manifest_output,
        manifest_bytes,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic public-only ten-region conversational "
            "curriculum."
        )
    )
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Manifest path (default: <output stem>.manifest.json beside output)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_conversational_curriculum(
        args.input_jsonl,
        args.output_jsonl,
        args.manifest,
    )
    print(
        json.dumps(
            {
                "manifest_payload_sha256": manifest[
                    "manifest_payload_sha256"
                ],
                "realized_counts": manifest["realized_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
