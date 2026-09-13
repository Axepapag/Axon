from __future__ import annotations

import copy
import json

import pytest

from runtime.field import D64FieldCompiler, LogicalRegion, RegionMaskPolicy, SharedFieldSnapshot
from runtime.heart.rail_auth import (
    DEFAULT_TTL_SECONDS,
    RAIL_AUTH_SCHEMA,
    RailAuthError,
    seal_envelope,
    verify_envelope,
)
from runtime.heart.remote_rail import decode_attended_rail, rail_wire_bytes

_SECRET = "k" * 48
_ENVIRON = {"AXON_RAIL_SECRET": _SECRET}


def _seal(**overrides):
    params = {
        "core_identity": "core-training-alpha",
        "assignment_id": "assignment-abc-0001",
        "payload": b"exact typed rail payload",
        "now": 1_000_000,
        "environ": _ENVIRON,
    }
    params.update(overrides)
    return seal_envelope(**params)


def _verify(envelope, **overrides):
    params = {
        "payload": b"exact typed rail payload",
        "expected_core_identity": "core-training-alpha",
        "expected_assignment_id": "assignment-abc-0001",
        "now": 1_000_000 + 10,
        "environ": _ENVIRON,
    }
    params.update(overrides)
    return verify_envelope(envelope, **params)


def test_roundtrip_seal_and_verify_returns_envelope_fields():
    envelope = _seal()
    assert envelope["schema"] == RAIL_AUTH_SCHEMA
    assert envelope["core_identity"] == "core-training-alpha"
    assert envelope["assignment_id"] == "assignment-abc-0001"
    assert len(envelope["nonce"]) == 32
    assert envelope["expires_at"] == envelope["issued_at"] + DEFAULT_TTL_SECONDS
    assert len(envelope["signature"]) == 64
    verified = _verify(envelope)
    assert verified == envelope


@pytest.mark.parametrize(
    "field,value",
    [
        ("core_identity", "core-impostor"),
        ("assignment_id", "assignment-other"),
        ("nonce", "ab" * 16),
        ("issued_at", 999_999),
        ("expires_at", 2_000_000),
        ("payload_sha256", "0" * 64),
    ],
)
def test_verify_rejects_any_tampered_signed_field(field, value):
    envelope = _seal()
    envelope[field] = value
    with pytest.raises(RailAuthError):
        _verify(envelope)


def test_verify_rejects_tampered_signature():
    envelope = _seal()
    flipped = "0" if envelope["signature"][0] != "0" else "1"
    envelope["signature"] = flipped + envelope["signature"][1:]
    with pytest.raises(RailAuthError, match="signature mismatch"):
        _verify(envelope)


def test_verify_rejects_wrong_payload_bytes():
    envelope = _seal()
    with pytest.raises(RailAuthError, match="payload digest mismatch"):
        _verify(envelope, payload=b"substituted payload")


def test_verify_rejects_wrong_expected_identity():
    envelope = _seal()
    with pytest.raises(RailAuthError, match="core identity mismatch"):
        _verify(envelope, expected_core_identity="core-training-beta")
    with pytest.raises(RailAuthError, match="assignment identity mismatch"):
        _verify(envelope, expected_assignment_id="assignment-abc-0002")


def test_verify_rejects_expired_envelope():
    envelope = _seal(now=1_000_000)
    with pytest.raises(RailAuthError, match="expired"):
        _verify(envelope, now=1_000_000 + DEFAULT_TTL_SECONDS)


def test_verify_rejects_envelope_from_the_future():
    envelope = _seal(now=5_000_000)
    with pytest.raises(RailAuthError, match="future"):
        _verify(envelope, now=1_000_000)


def test_seal_rejects_non_positive_ttl():
    with pytest.raises(RailAuthError, match="ttl_seconds"):
        _seal(ttl_seconds=-5)
    with pytest.raises(RailAuthError, match="ttl_seconds"):
        _seal(ttl_seconds=0)


def test_replay_rejected_through_caller_nonce_cache():
    envelope = _seal()
    seen: set[str] = set()

    def nonce_cache(nonce: str) -> bool:
        if nonce in seen:
            return False
        seen.add(nonce)
        return True

    _verify(envelope, nonce_cache=nonce_cache)
    with pytest.raises(RailAuthError, match="replay"):
        _verify(envelope, nonce_cache=nonce_cache)


def test_verify_fails_closed_when_nonce_cache_hook_fails():
    envelope = _seal()

    def broken_cache(nonce: str) -> bool:
        raise OSError("nonce store unavailable")

    with pytest.raises(RailAuthError, match="nonce cache hook"):
        _verify(envelope, nonce_cache=broken_cache)


def test_verify_without_nonce_cache_skips_replay_detection():
    envelope = _seal()
    _verify(envelope)
    _verify(envelope)  # documented: retention is the caller's responsibility


def test_missing_secret_fails_closed_for_seal_and_verify():
    empty_environ: dict[str, str] = {}
    with pytest.raises(RailAuthError, match="secret"):
        seal_envelope(
            core_identity="core-training-alpha",
            assignment_id="assignment-abc-0001",
            payload=b"x",
            environ=empty_environ,
        )
    envelope = _seal()
    with pytest.raises(RailAuthError, match="secret"):
        _verify(envelope, environ=empty_environ)


def test_short_secret_fails_closed():
    with pytest.raises(RailAuthError, match="secret"):
        _seal(environ={"AXON_RAIL_SECRET": "too-short"})


def test_secret_loads_from_file_path_and_takes_precedence(tmp_path):
    secret_file = tmp_path / "rail_secret.bin"
    secret_file.write_bytes((("f" * 40) + "\n").encode("ascii"))
    environ = {"AXON_RAIL_SECRET_PATH": str(secret_file), "AXON_RAIL_SECRET": "wrong-" + "w" * 40}
    envelope = _seal(environ=environ)
    _verify(envelope, environ=environ)
    with pytest.raises(RailAuthError, match="signature mismatch"):
        _verify(envelope)


def test_unreadable_secret_path_fails_closed(tmp_path):
    environ = {"AXON_RAIL_SECRET_PATH": str(tmp_path / "does-not-exist.bin")}
    with pytest.raises(RailAuthError, match="AXON_RAIL_SECRET_PATH"):
        _seal(environ=environ)


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"schema": RAIL_AUTH_SCHEMA},
        None,
        "not-a-mapping",
        {"schema": "axon-rail-auth-envelope-v0"},
    ],
)
def test_verify_rejects_malformed_envelope_structures(envelope):
    with pytest.raises(RailAuthError):
        _verify(envelope)


def test_verify_rejects_malformed_field_values():
    envelope = _seal()
    bad_nonce = copy.deepcopy(envelope)
    bad_nonce["nonce"] = "zz" * 16
    with pytest.raises(RailAuthError, match="nonce"):
        _verify(bad_nonce)

    short_nonce = copy.deepcopy(envelope)
    short_nonce["nonce"] = "abcd"
    with pytest.raises(RailAuthError, match="nonce"):
        _verify(short_nonce)

    bool_timestamp = copy.deepcopy(envelope)
    bool_timestamp["issued_at"] = True
    with pytest.raises(RailAuthError, match="issued_at"):
        _verify(bool_timestamp)

    bad_digest = copy.deepcopy(envelope)
    bad_digest["payload_sha256"] = "xyz"
    with pytest.raises(RailAuthError, match="payload_sha256"):
        _verify(bad_digest)

    bad_signature = copy.deepcopy(envelope)
    bad_signature["signature"] = 1234
    with pytest.raises(RailAuthError, match="signature"):
        _verify(bad_signature)

    inverted = copy.deepcopy(envelope)
    inverted["expires_at"] = inverted["issued_at"]
    inverted["signature"] = _resign(inverted)
    with pytest.raises(RailAuthError, match="expires_at"):
        _verify(inverted)


def _resign(envelope: dict) -> str:
    import hashlib
    import hmac

    from runtime.field import canonical_json_bytes

    body = {k: v for k, v in envelope.items() if k != "signature"}
    return hmac.new(_SECRET.encode(), canonical_json_bytes(body), hashlib.sha256).hexdigest()


def test_rail_wire_bytes_seal_verify_decode_end_to_end():
    field = SharedFieldSnapshot.from_texts({
        LogicalRegion.USER_INPUT: "ABC? λ🧠\n",
        LogicalRegion.CONVERSATION_HISTORY: "private dormant material never sent",
    })
    rail = D64FieldCompiler().compile(field, region_masks={
        LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("none"),
    })
    payload = rail_wire_bytes(rail)
    envelope = _seal(payload=payload, now=1_000_000)
    verified = _verify(envelope, payload=payload, now=1_000_010)
    assert verified["payload_sha256"] == envelope["payload_sha256"]
    restored = decode_attended_rail(json.loads(payload), expected_rail_id=rail.rail_id)
    assert restored.region_text(LogicalRegion.USER_INPUT) == "ABC? λ🧠\n"
    assert restored.region_text(LogicalRegion.CONVERSATION_HISTORY) == ""

    tampered = bytearray(payload)
    tampered[-3] ^= 0x01
    with pytest.raises(RailAuthError, match="payload digest mismatch"):
        _verify(envelope, payload=bytes(tampered), now=1_000_010)


def test_seal_rejects_non_bytes_payload():
    with pytest.raises(RailAuthError, match="payload"):
        _seal(payload="text is not bytes")
