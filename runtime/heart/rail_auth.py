"""Authenticated, replay-protected envelopes for remote rail transport.

A sealed envelope binds one exact typed payload (an encoded attended D64 rail,
a typed emission, or a checkpoint bundle) to a core identity and an assignment
identity, with expiry, a random nonce, and an HMAC-SHA256 signature over the
canonical JSON bytes of the envelope body.  Verification fails closed on
malformed, wrong-signature, wrong-identity, wrong-payload, expired, and
replayed envelopes.  Replay detection uses a caller-supplied nonce cache hook
so the Heart, not this module, owns durable nonce retention.

The shared secret is never hardcoded and never enters source control.  It is
loaded per call from ``AXON_RAIL_SECRET_PATH`` (a file readable only by the
operator) or, failing that, the ``AXON_RAIL_SECRET`` environment variable.
Both seal and verify refuse to run when no usable secret is configured.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from runtime.field import canonical_json_bytes

RAIL_AUTH_SCHEMA = "axon-rail-auth-envelope-v1"
NONCE_BYTES = 16
MIN_SECRET_BYTES = 32
DEFAULT_TTL_SECONDS = 300
MAX_CLOCK_SKEW_SECONDS = 300

_ENVELOPE_FIELDS = (
    "schema",
    "core_identity",
    "assignment_id",
    "nonce",
    "issued_at",
    "expires_at",
    "payload_sha256",
    "signature",
)


class RailAuthError(ValueError):
    """An envelope is unauthenticated, expired, replayed, or malformed."""


def _load_secret(environ: Mapping[str, str]) -> bytes:
    """Load the HMAC secret from operator-controlled configuration, fail closed."""

    path = environ.get("AXON_RAIL_SECRET_PATH")
    if path:
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise RailAuthError(
                f"AXON_RAIL_SECRET_PATH is unreadable: {path!r}; refusing to authenticate"
            ) from exc
    else:
        raw = (environ.get("AXON_RAIL_SECRET") or "").encode("utf-8")
    secret = raw.strip()
    if len(secret) < MIN_SECRET_BYTES:
        raise RailAuthError(
            f"rail auth secret is absent or shorter than {MIN_SECRET_BYTES} bytes; "
            "refusing to authenticate"
        )
    return secret


def _require_identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RailAuthError(f"{name} must be a non-empty string")
    return value


def _require_timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RailAuthError(f"{name} must be an integer epoch second")
    if value < 0:
        raise RailAuthError(f"{name} must be non-negative")
    return value


def _require_nonce(value: Any) -> str:
    if not isinstance(value, str) or len(value) < NONCE_BYTES * 2:
        raise RailAuthError("nonce must be a hex string of at least 16 random bytes")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RailAuthError("nonce must be hex-encoded random bytes") from exc
    return value


def _require_payload(payload: bytes | bytearray | memoryview) -> bytes:
    if isinstance(payload, (bytearray, memoryview)):
        return bytes(payload)
    if not isinstance(payload, bytes):
        raise RailAuthError("payload must be bytes; encode text explicitly before sealing")
    return payload


def _signing_bytes(envelope: Mapping[str, Any]) -> bytes:
    body = {name: envelope[name] for name in _ENVELOPE_FIELDS if name != "signature"}
    return canonical_json_bytes(body)


def seal_envelope(
    *,
    core_identity: str,
    assignment_id: str,
    payload: bytes | bytearray | memoryview,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    now: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Seal one exact payload into a signed, expiring, replay-protected envelope."""

    if environ is None:
        environ = os.environ
    core = _require_identity(core_identity, "core_identity")
    assignment = _require_identity(assignment_id, "assignment_id")
    body = _require_payload(payload)
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds < 1:
        raise RailAuthError("ttl_seconds must be a positive integer")
    issued_at = _require_timestamp(int(now if now is not None else time.time()), "now")
    secret = _load_secret(environ)
    envelope: dict[str, Any] = {
        "schema": RAIL_AUTH_SCHEMA,
        "core_identity": core,
        "assignment_id": assignment,
        "nonce": secrets.token_bytes(NONCE_BYTES).hex(),
        "issued_at": issued_at,
        "expires_at": issued_at + ttl_seconds,
        "payload_sha256": hashlib.sha256(body).hexdigest(),
    }
    envelope["signature"] = hmac.new(secret, _signing_bytes(envelope), hashlib.sha256).hexdigest()
    return envelope


def verify_envelope(
    envelope: Mapping[str, Any],
    *,
    payload: bytes | bytearray | memoryview,
    expected_core_identity: str,
    expected_assignment_id: str,
    nonce_cache: Callable[[str], bool] | None = None,
    now: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify and return a sealed envelope, failing closed on any defect.

    ``nonce_cache`` is the caller's replay-retention hook: it receives the
    envelope nonce and must return ``True`` exactly when the nonce is fresh and
    has been durably recorded.  Any other result, or a hook exception, rejects
    the envelope.  Passing ``None`` skips replay detection and is the caller's
    explicit responsibility.
    """

    if environ is None:
        environ = os.environ
    if not isinstance(envelope, Mapping):
        raise RailAuthError("envelope must be a mapping")
    missing = [name for name in _ENVELOPE_FIELDS if name not in envelope]
    if missing:
        raise RailAuthError(f"envelope is missing required fields: {missing!r}")
    if envelope["schema"] != RAIL_AUTH_SCHEMA:
        raise RailAuthError("envelope schema is unknown or unsupported")
    core = _require_identity(envelope["core_identity"], "core_identity")
    assignment = _require_identity(envelope["assignment_id"], "assignment_id")
    nonce = _require_nonce(envelope["nonce"])
    issued_at = _require_timestamp(envelope["issued_at"], "issued_at")
    expires_at = _require_timestamp(envelope["expires_at"], "expires_at")
    if expires_at <= issued_at:
        raise RailAuthError("envelope expires_at must be later than issued_at")
    payload_sha256 = envelope["payload_sha256"]
    if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
        raise RailAuthError("payload_sha256 must be a 64-character hex digest")
    signature = envelope["signature"]
    if not isinstance(signature, str) or len(signature) != 64:
        raise RailAuthError("signature must be a 64-character hex digest")
    body = _require_payload(payload)
    expected_core = _require_identity(expected_core_identity, "expected_core_identity")
    expected_assignment = _require_identity(expected_assignment_id, "expected_assignment_id")
    observed_at = int(now if now is not None else time.time())

    # Signature before anything else: an unauthenticated envelope must not
    # consume caller resources such as nonce-cache slots.
    secret = _load_secret(environ)
    expected_signature = hmac.new(secret, _signing_bytes(envelope), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise RailAuthError("envelope signature mismatch")
    if core != expected_core:
        raise RailAuthError("envelope core identity mismatch")
    if assignment != expected_assignment:
        raise RailAuthError("envelope assignment identity mismatch")
    if not hmac.compare_digest(payload_sha256, hashlib.sha256(body).hexdigest()):
        raise RailAuthError("envelope payload digest mismatch")
    if issued_at > observed_at + MAX_CLOCK_SKEW_SECONDS:
        raise RailAuthError("envelope was issued in the future")
    if observed_at >= expires_at:
        raise RailAuthError("envelope is expired")
    if nonce_cache is not None:
        try:
            fresh = nonce_cache(nonce)
        except Exception as exc:
            raise RailAuthError("nonce cache hook failed closed") from exc
        if fresh is not True:
            raise RailAuthError("envelope nonce replay detected")
    return dict(envelope)


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "MAX_CLOCK_SKEW_SECONDS",
    "MIN_SECRET_BYTES",
    "NONCE_BYTES",
    "RAIL_AUTH_SCHEMA",
    "RailAuthError",
    "seal_envelope",
    "verify_envelope",
]
