"""Fail-closed typed invocation protocol for Axon's runtime.

The four human-friendly markers are lexical sugar only:

``$$`` direct/explicit shell execution
``##`` rooted filesystem operations
``@@`` registered advisor calls
``&&`` pinned scripts from a registered tools root

Text becomes an invocation proposal only inside an exact
``:::axon-invoke/v1`` envelope.  Durable :class:`InvocationPolicy` data grants
authority; neither the marker nor a model/consolidator's authorship grants it.
Execution is disabled by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Mapping, Sequence

from .advisors import (
    AdvisorError,
    AdvisorCallResult,
    AdvisorRegistry,
    HttpPost,
    ProcessRunner,
    invoke_registered_advisor,
    redact_text,
)


START_MARKER = ":::axon-invoke/v1"
END_MARKER = ":::end"
ACTION_MARKERS = {
    "$$": "shell",
    "##": "filesystem",
    "@@": "advisor",
    "&&": "tool",
}
PARSABLE_SOURCES = frozenset(
    {"user_input", "model_output", "consolidator_output"}
)
INERT_SOURCES = frozenset(
    {"tool_results", "advisor_input", "runtime_result", "dormant_state"}
)
_ACTION_LINE = re.compile(r"^(\$\$|##|@@|&&) ([^\r\n]+)$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class InvocationError(RuntimeError):
    """Base error for parsing, policy, idempotency, or execution."""


class InvocationParseError(InvocationError):
    """An invocation envelope is ambiguous or malformed."""


class InvocationValidationError(InvocationError):
    """A typed request does not satisfy its strict schema."""


class InvocationPolicyError(InvocationError):
    """Durable policy does not grant the requested capability."""


class RootedPathError(InvocationPolicyError):
    """A path escapes or violates its configured root."""


class IdempotencyConflict(InvocationError):
    """One idempotency key was reused for a different canonical request."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvocationParseError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def strict_json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                InvocationParseError(f"invalid JSON constant {item!r}")
            ),
        )
    except InvocationParseError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise InvocationParseError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvocationParseError(f"{label} must be a JSON object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise InvocationValidationError(
            f"{label} has unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise InvocationValidationError(
            f"{label} is missing fields: {sorted(missing)}"
        )


def _string(
    value: Mapping[str, Any],
    key: str,
    *,
    label: str,
    default: str | None = None,
    allow_empty: bool = False,
) -> str:
    raw = value.get(key, default)
    if not isinstance(raw, str) or (not allow_empty and not raw):
        raise InvocationValidationError(f"{label}.{key} must be a string")
    return raw


def _positive_int(
    value: Mapping[str, Any],
    key: str,
    *,
    label: str,
    default: int,
    ceiling: int,
) -> int:
    raw = value.get(key, default)
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int)
        or raw <= 0
        or raw > ceiling
    ):
        raise InvocationValidationError(
            f"{label}.{key} must be in 1..{ceiling}"
        )
    return raw


def _normalize_common(
    payload: dict[str, Any],
    *,
    index: int,
    label: str,
) -> dict[str, Any]:
    result = dict(payload)
    action_id = result.get("action_id", f"action-{index + 1}")
    if not isinstance(action_id, str) or not _SAFE_ID.fullmatch(action_id):
        raise InvocationValidationError(f"{label}.action_id is invalid")
    result["action_id"] = action_id
    if "idempotency_key" in result:
        key = result["idempotency_key"]
        if not isinstance(key, str) or not _SAFE_ID.fullmatch(key):
            raise InvocationValidationError(
                f"{label}.idempotency_key is invalid"
            )
    return result


def _validate_shell(payload: dict[str, Any], index: int) -> dict[str, Any]:
    label = "shell action"
    allowed = {
        "action_id",
        "idempotency_key",
        "mode",
        "program",
        "argv",
        "command",
        "root",
        "cwd",
        "stdin",
        "timeout_ms",
        "output_limit_bytes",
    }
    _strict_keys(payload, allowed=allowed, required=set(), label=label)
    result = _normalize_common(payload, index=index, label=label)
    mode = _string(result, "mode", label=label, default="direct")
    if mode not in {"direct", "powershell", "cmd"}:
        raise InvocationValidationError(
            "shell action.mode must be direct, powershell, or cmd"
        )
    result["mode"] = mode
    result["root"] = _string(
        result, "root", label=label, default="workspace"
    )
    result["cwd"] = _string(
        result, "cwd", label=label, default=".", allow_empty=False
    )
    result["timeout_ms"] = _positive_int(
        result,
        "timeout_ms",
        label=label,
        default=30_000,
        ceiling=3_600_000,
    )
    result["output_limit_bytes"] = _positive_int(
        result,
        "output_limit_bytes",
        label=label,
        default=1_048_576,
        ceiling=16 * 1024 * 1024,
    )
    if "stdin" in result and result["stdin"] is not None and not isinstance(
        result["stdin"], str
    ):
        raise InvocationValidationError("shell action.stdin must be string or null")
    if mode == "direct":
        result["program"] = _string(result, "program", label=label)
        argv = result.get("argv", [])
        if not isinstance(argv, list) or not all(
            isinstance(item, str) for item in argv
        ):
            raise InvocationValidationError(
                "shell action.argv must be a string array"
            )
        result["argv"] = list(argv)
        if "command" in result:
            raise InvocationValidationError(
                "direct shell action cannot contain command"
            )
    else:
        result["command"] = _string(result, "command", label=label)
        if "program" in result or "argv" in result:
            raise InvocationValidationError(
                f"{mode} command mode cannot contain program/argv"
            )
    return result


def _validate_filesystem(payload: dict[str, Any], index: int) -> dict[str, Any]:
    label = "filesystem action"
    base = {"action_id", "idempotency_key", "op", "root", "path"}
    op = payload.get("op")
    if op not in {"read_text", "list", "stat", "write_text", "mkdir"}:
        raise InvocationValidationError(
            "filesystem action.op must be read_text, list, stat, write_text, or mkdir"
        )
    extras = {
        "read_text": {"encoding", "max_bytes"},
        "list": {"max_entries"},
        "stat": set(),
        "write_text": {"text", "encoding", "expected_sha256"},
        "mkdir": {"parents", "exist_ok"},
    }[str(op)]
    required = {
        "read_text": set(),
        "list": set(),
        "stat": set(),
        "write_text": {"text", "expected_sha256"},
        "mkdir": set(),
    }[str(op)]
    _strict_keys(
        payload,
        allowed=base | extras,
        required={"op", "root", "path"} | required,
        label=label,
    )
    result = _normalize_common(payload, index=index, label=label)
    result["op"] = str(op)
    result["root"] = _string(result, "root", label=label)
    result["path"] = _string(result, "path", label=label)
    if op == "read_text":
        result["encoding"] = _string(
            result, "encoding", label=label, default="utf-8"
        )
        result["max_bytes"] = _positive_int(
            result,
            "max_bytes",
            label=label,
            default=1_048_576,
            ceiling=16 * 1024 * 1024,
        )
    elif op == "list":
        result["max_entries"] = _positive_int(
            result,
            "max_entries",
            label=label,
            default=1_000,
            ceiling=100_000,
        )
    elif op == "write_text":
        if not isinstance(result["text"], str):
            raise InvocationValidationError(
                "filesystem action.text must be a string"
            )
        result["encoding"] = _string(
            result, "encoding", label=label, default="utf-8"
        )
        expected = result["expected_sha256"]
        if expected is not None and (
            not isinstance(expected, str) or not _SHA256.fullmatch(expected)
        ):
            raise InvocationValidationError(
                "expected_sha256 must be a SHA-256 string or null"
            )
        result["expected_sha256"] = (
            expected.lower() if isinstance(expected, str) else None
        )
    elif op == "mkdir":
        for key, default in (("parents", True), ("exist_ok", False)):
            raw = result.get(key, default)
            if not isinstance(raw, bool):
                raise InvocationValidationError(
                    f"filesystem action.{key} must be boolean"
                )
            result[key] = raw
    return result


def _validate_advisor(payload: dict[str, Any], index: int) -> dict[str, Any]:
    label = "advisor action"
    allowed = {
        "action_id",
        "idempotency_key",
        "advisor",
        "model",
        "prompt",
        "timeout_ms",
        "output_limit_bytes",
    }
    _strict_keys(
        payload,
        allowed=allowed,
        required={"advisor", "prompt"},
        label=label,
    )
    result = _normalize_common(payload, index=index, label=label)
    result["advisor"] = _string(result, "advisor", label=label)
    result["prompt"] = _string(
        result, "prompt", label=label, allow_empty=False
    )
    if "model" in result and result["model"] is not None and not isinstance(
        result["model"], str
    ):
        raise InvocationValidationError(
            "advisor action.model must be string or null"
        )
    result["timeout_ms"] = _positive_int(
        result,
        "timeout_ms",
        label=label,
        default=120_000,
        ceiling=3_600_000,
    )
    result["output_limit_bytes"] = _positive_int(
        result,
        "output_limit_bytes",
        label=label,
        default=1_048_576,
        ceiling=16 * 1024 * 1024,
    )
    return result


def _validate_tool(payload: dict[str, Any], index: int) -> dict[str, Any]:
    label = "tool action"
    allowed = {
        "action_id",
        "idempotency_key",
        "tool",
        "args",
        "timeout_ms",
        "output_limit_bytes",
    }
    _strict_keys(
        payload,
        allowed=allowed,
        required={"tool", "args"},
        label=label,
    )
    result = _normalize_common(payload, index=index, label=label)
    result["tool"] = _string(result, "tool", label=label)
    args = result["args"]
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise InvocationValidationError("tool action.args must be a string array")
    result["args"] = list(args)
    result["timeout_ms"] = _positive_int(
        result,
        "timeout_ms",
        label=label,
        default=60_000,
        ceiling=3_600_000,
    )
    result["output_limit_bytes"] = _positive_int(
        result,
        "output_limit_bytes",
        label=label,
        default=1_048_576,
        ceiling=16 * 1024 * 1024,
    )
    return result


@dataclass(frozen=True, slots=True)
class InvocationAction:
    kind: str
    marker: str
    payload: Mapping[str, Any]
    request_hash: str

    @property
    def action_id(self) -> str:
        return str(self.payload["action_id"])

    @property
    def idempotency_key(self) -> str:
        return str(self.payload.get("idempotency_key") or self.request_hash)


@dataclass(frozen=True, slots=True)
class InvocationBatch:
    actions: tuple[InvocationAction, ...]
    source_kind: str
    batch_hash: str


def _require_absolute_local_executable(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty path")
    windows = PureWindowsPath(value)
    native = Path(value)
    if not windows.is_absolute() and not native.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if value.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\")):
        raise ValueError(f"{label} cannot be a UNC or device path")


def _validate_registry_limits(
    timeout_ms_max: int,
    output_limit_bytes: int,
    label: str,
) -> None:
    if (
        isinstance(timeout_ms_max, bool)
        or not isinstance(timeout_ms_max, int)
        or not 1 <= timeout_ms_max <= 3_600_000
    ):
        raise ValueError(f"{label} timeout_ms_max must be in 1..3600000")
    if (
        isinstance(output_limit_bytes, bool)
        or not isinstance(output_limit_bytes, int)
        or not 1 <= output_limit_bytes <= 16 * 1024 * 1024
    ):
        raise ValueError(
            f"{label} output_limit_bytes must be in 1..16777216"
        )


def _markdown_fence_mask(lines: Sequence[str]) -> list[bool]:
    """Return True for lines inside a Markdown fence, including fence lines."""

    fence_character: str | None = None
    fence_length = 0
    mask: list[bool] = []
    for line in lines:
        if fence_character is None:
            opener = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            is_fence = opener is not None
            mask.append(is_fence)
            if opener is not None:
                fence = opener.group(1)
                # CommonMark forbids backticks in a backtick-fence info
                # string.  A malformed would-be opener remains ordinary text.
                if fence[0] == "`" and "`" in opener.group(2):
                    mask[-1] = False
                    continue
                fence_character = fence[0]
                fence_length = len(fence)
            continue

        closer = re.match(
            rf"^ {{0,3}}{re.escape(fence_character)}"
            rf"{{{fence_length},}}[ \t]*$",
            line,
        )
        mask.append(True)
        if closer is not None:
            fence_character = None
            fence_length = 0
    return mask


def parse_invocation_envelope(
    text: str,
    *,
    source_kind: str = "user_input",
) -> InvocationBatch | None:
    """Parse at most one strict envelope.

    Tool/advisor/runtime results are unconditionally inert.  User, model, and
    consolidator text may form proposals; policy independently decides whether
    those proposals have execution authority.
    """

    if source_kind in INERT_SOURCES:
        return None
    if source_kind not in PARSABLE_SOURCES:
        raise InvocationValidationError(f"unknown invocation source {source_kind!r}")
    lines = text.splitlines()
    fence_mask = _markdown_fence_mask(lines)
    starts = [
        index
        for index, line in enumerate(lines)
        if not fence_mask[index] and line == START_MARKER
    ]
    if not starts:
        return None
    if len(starts) != 1:
        raise InvocationParseError("exactly one invocation envelope is allowed")
    ends = [
        index
        for index, line in enumerate(lines)
        if not fence_mask[index] and line == END_MARKER
    ]
    if len(ends) != 1 or ends[0] <= starts[0]:
        raise InvocationParseError(
            "invocation envelope must have exactly one later :::end"
        )
    start = starts[0]
    end = ends[0]
    body = lines[start + 1 : end]
    if not body:
        raise InvocationParseError("invocation envelope is empty")
    actions: list[InvocationAction] = []
    action_ids: set[str] = set()
    validators = {
        "$$": _validate_shell,
        "##": _validate_filesystem,
        "@@": _validate_advisor,
        "&&": _validate_tool,
    }
    for body_index, line in enumerate(body):
        match = _ACTION_LINE.fullmatch(line)
        if match is None:
            raise InvocationParseError(
                "every envelope line must be MARKER + space + one-line JSON"
            )
        marker, raw = match.groups()
        parsed = strict_json_object(
            raw,
            label=f"action line {body_index + 1}",
        )
        payload = validators[marker](parsed, body_index)
        action_id = str(payload["action_id"])
        if action_id in action_ids:
            raise InvocationValidationError(
                f"duplicate action_id {action_id!r}"
            )
        action_ids.add(action_id)
        kind = ACTION_MARKERS[marker]
        request_hash = canonical_sha256(
            {
                "schema": "axon.invocation-request.v1",
                "kind": kind,
                "payload": payload,
            }
        )
        actions.append(
            InvocationAction(
                kind=kind,
                marker=marker,
                payload=payload,
                request_hash=request_hash,
            )
        )
    batch_hash = canonical_sha256(
        {
            "schema": "axon.invocation-batch.v1",
            "source_kind": source_kind,
            "requests": [action.request_hash for action in actions],
        }
    )
    return InvocationBatch(
        actions=tuple(actions),
        source_kind=source_kind,
        batch_hash=batch_hash,
    )


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    program_id: str
    executable: str
    enabled: bool = False
    timeout_ms_max: int = 120_000
    output_limit_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.program_id):
            raise ValueError("invalid program_id")
        _require_absolute_local_executable(self.executable, "program executable")
        _validate_registry_limits(
            self.timeout_ms_max,
            self.output_limit_bytes,
            "program",
        )


@dataclass(frozen=True, slots=True)
class CommandModeSpec:
    mode: str
    executable: str
    enabled: bool = False
    prefix: tuple[str, ...] = ()
    timeout_ms_max: int = 120_000
    output_limit_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if self.mode not in {"powershell", "cmd"}:
            raise ValueError("command mode must be powershell or cmd")
        _require_absolute_local_executable(
            self.executable,
            "command-mode executable",
        )
        _validate_registry_limits(
            self.timeout_ms_max,
            self.output_limit_bytes,
            "command mode",
        )


@dataclass(frozen=True, slots=True)
class ScriptSpec:
    tool_id: str
    relative_path: str
    sha256: str
    interpreter: str = sys.executable
    interpreter_args: tuple[str, ...] = ()
    enabled: bool = False
    timeout_ms_max: int = 120_000
    output_limit_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.tool_id):
            raise ValueError("invalid tool_id")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("script sha256 must be a SHA-256 string")
        _require_absolute_local_executable(
            self.interpreter,
            "script interpreter",
        )
        _validate_registry_limits(
            self.timeout_ms_max,
            self.output_limit_bytes,
            "script",
        )


@dataclass(slots=True)
class InvocationPolicy:
    """Durable capability policy.  Every execution switch defaults to off."""

    enabled: bool = False
    allowed_sources: frozenset[str] = frozenset()
    roots: dict[str, Path] = field(default_factory=dict)
    filesystem_ops: frozenset[str] = frozenset()
    programs: dict[str, ProgramSpec] = field(default_factory=dict)
    command_modes: dict[str, CommandModeSpec] = field(default_factory=dict)
    advisors: AdvisorRegistry = field(default_factory=AdvisorRegistry)
    tools_root: Path | None = None
    scripts: dict[str, ScriptSpec] = field(default_factory=dict)
    timeout_ms_max: int = 300_000
    output_limit_bytes: int = 1_048_576
    redact_env_names: tuple[str, ...] = ()


def _validate_raw_relative_path(raw: str) -> tuple[str, ...]:
    if not raw or "\x00" in raw:
        raise RootedPathError("path is empty or contains NUL")
    if raw.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\")):
        raise RootedPathError("UNC/device paths are forbidden")
    win = PureWindowsPath(raw)
    if win.is_absolute() or win.drive or ":" in raw:
        raise RootedPathError("absolute, drive, and ADS paths are forbidden")
    pieces = tuple(part for part in raw.replace("\\", "/").split("/") if part)
    if not pieces:
        if raw == ".":
            return (".",)
        raise RootedPathError("path is empty")
    if any(part == ".." for part in pieces):
        raise RootedPathError("parent traversal is forbidden")
    if any(part in {"", "/"} for part in pieces):
        raise RootedPathError("invalid path component")
    return pieces


def _inside_root(root: Path, candidate: Path) -> bool:
    root_text = os.path.normcase(str(root))
    candidate_text = os.path.normcase(str(candidate))
    try:
        return os.path.commonpath([root_text, candidate_text]) == root_text
    except ValueError:
        return False


def resolve_rooted_path(
    policy: InvocationPolicy,
    root_id: str,
    relative_path: str,
    *,
    must_exist: bool = False,
) -> Path:
    try:
        configured = Path(policy.roots[root_id])
    except KeyError as exc:
        raise RootedPathError(f"root {root_id!r} is not registered") from exc
    root = configured.resolve(strict=True)
    if not root.is_dir():
        raise RootedPathError(f"root {root_id!r} is not a directory")
    pieces = _validate_raw_relative_path(relative_path)
    candidate = root if pieces == (".",) else root.joinpath(*pieces)
    resolved = candidate.resolve(strict=False)
    if not _inside_root(root, resolved):
        raise RootedPathError("resolved path escapes its configured root")
    if must_exist and not resolved.exists():
        raise RootedPathError(f"path does not exist: {relative_path}")
    return resolved


def _resolve_under_tools(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve(strict=True)
    pieces = _validate_raw_relative_path(relative_path)
    candidate = root_resolved.joinpath(*pieces).resolve(strict=True)
    if not _inside_root(root_resolved, candidate):
        raise RootedPathError("registered tool path escapes tools_root")
    if not candidate.is_file():
        raise RootedPathError("registered tool is not a file")
    return candidate


def _verify_expected_hash(path: Path, expected: str | None) -> None:
    if expected is None:
        if path.exists():
            raise InvocationPolicyError(
                "expected_sha256=null requires the target to be absent"
            )
        return
    if not path.is_file():
        raise InvocationPolicyError(
            "expected_sha256 requires an existing regular file"
        )
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise InvocationPolicyError(
            f"expected_sha256 mismatch: expected {expected}, got {actual}"
        )


@dataclass(frozen=True, slots=True)
class PreparedAction:
    action: InvocationAction
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class InvocationResult:
    action_id: str
    kind: str
    request_hash: str
    status: str
    target_region: str
    started_at: str
    finished_at: str
    summary: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    truncated: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_hash",
            canonical_sha256(self.to_dict(include_hash=False)),
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": "axon.invocation-result.v1",
            "action_id": self.action_id,
            "kind": self.kind,
            "request_hash": self.request_hash,
            "status": self.status,
            "target_region": self.target_region,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "truncated": self.truncated,
            "details": dict(self.details),
        }
        if include_hash:
            value["result_hash"] = self.result_hash
        return value


@dataclass(slots=True)
class _IdempotencyEntry:
    request_hash: str
    result: InvocationResult


class IdempotencyStore:
    """Thread-safe helper; a durable backend can implement the same contract."""

    def __init__(self) -> None:
        self._entries: dict[str, _IdempotencyEntry] = {}
        self._lock = threading.Lock()

    def check(
        self,
        key: str,
        request_hash: str,
    ) -> InvocationResult | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.request_hash != request_hash:
                raise IdempotencyConflict(
                    f"idempotency key {key!r} belongs to another request"
                )
            return entry.result

    def record(
        self,
        key: str,
        request_hash: str,
        result: InvocationResult,
    ) -> None:
        with self._lock:
            prior = self._entries.get(key)
            if prior is not None and prior.request_hash != request_hash:
                raise IdempotencyConflict(
                    f"idempotency key {key!r} belongs to another request"
                )
            self._entries[key] = _IdempotencyEntry(request_hash, result)


def _decode_output(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _cap_utf8(text: str, limit: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text, False
    marker = f"\n[output truncated at {limit} bytes]"
    marker_bytes = marker.encode("utf-8")
    keep = max(0, limit - len(marker_bytes))
    return encoded[:keep].decode("utf-8", errors="ignore") + marker, True


def _cap_output_pair(
    stdout: str,
    stderr: str,
    limit: int,
) -> tuple[str, str, bool]:
    total = len(stdout.encode("utf-8")) + len(stderr.encode("utf-8"))
    if total <= limit:
        return stdout, stderr, False
    if stderr:
        stdout_budget = max(1, limit // 2)
        stderr_budget = max(1, limit - stdout_budget)
    else:
        stdout_budget = limit
        stderr_budget = 1
    stdout, cut_a = _cap_utf8(stdout, stdout_budget)
    stderr, cut_b = _cap_utf8(stderr, stderr_budget)
    return stdout, stderr, cut_a or cut_b


class InvocationExecutor:
    """Validate an entire batch, then execute enabled actions sequentially."""

    def __init__(
        self,
        policy: InvocationPolicy,
        *,
        idempotency: IdempotencyStore | None = None,
        runner: ProcessRunner | None = None,
        http_post: HttpPost | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.policy = policy
        self.idempotency = idempotency or IdempotencyStore()
        self.runner = runner or subprocess.run
        self.http_post = http_post
        self.environment = dict(
            os.environ if environment is None else environment
        )

    def _limits(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_max: int | None = None,
        output_max: int | None = None,
    ) -> tuple[int, int]:
        timeout = int(payload.get("timeout_ms", 30_000))
        output = int(payload.get("output_limit_bytes", 1_048_576))
        if timeout > self.policy.timeout_ms_max or (
            timeout_max is not None and timeout > timeout_max
        ):
            raise InvocationPolicyError("requested timeout exceeds policy")
        if output > self.policy.output_limit_bytes or (
            output_max is not None and output > output_max
        ):
            raise InvocationPolicyError("requested output limit exceeds policy")
        return timeout, output

    def _prepare(self, action: InvocationAction) -> PreparedAction:
        payload = action.payload
        if action.kind == "shell":
            cwd = resolve_rooted_path(
                self.policy,
                str(payload["root"]),
                str(payload["cwd"]),
                must_exist=True,
            )
            if not cwd.is_dir():
                raise RootedPathError("shell cwd must be a directory")
            mode = str(payload["mode"])
            if mode == "direct":
                program_id = str(payload["program"])
                try:
                    spec = self.policy.programs[program_id]
                except KeyError as exc:
                    raise InvocationPolicyError(
                        f"program {program_id!r} is not registered"
                    ) from exc
                if not spec.enabled:
                    raise InvocationPolicyError(
                        f"program {program_id!r} is disabled"
                    )
                timeout, output = self._limits(
                    payload,
                    timeout_max=spec.timeout_ms_max,
                    output_max=spec.output_limit_bytes,
                )
                argv = [spec.executable, *list(payload["argv"])]
            else:
                try:
                    command_spec = self.policy.command_modes[mode]
                except KeyError as exc:
                    raise InvocationPolicyError(
                        f"{mode} command mode is not registered"
                    ) from exc
                if not command_spec.enabled:
                    raise InvocationPolicyError(
                        f"{mode} command mode is disabled"
                    )
                timeout, output = self._limits(
                    payload,
                    timeout_max=command_spec.timeout_ms_max,
                    output_max=command_spec.output_limit_bytes,
                )
                default_prefix = (
                    ("-NoProfile", "-NonInteractive", "-Command")
                    if mode == "powershell"
                    else ("/d", "/s", "/c")
                )
                prefix = command_spec.prefix or default_prefix
                argv = [
                    command_spec.executable,
                    *prefix,
                    str(payload["command"]),
                ]
            return PreparedAction(
                action,
                {
                    "cwd": cwd,
                    "argv": tuple(argv),
                    "timeout_ms": timeout,
                    "output_limit_bytes": output,
                },
            )

        if action.kind == "filesystem":
            op = str(payload["op"])
            if op not in self.policy.filesystem_ops:
                raise InvocationPolicyError(
                    f"filesystem operation {op!r} is disabled"
                )
            path = resolve_rooted_path(
                self.policy,
                str(payload["root"]),
                str(payload["path"]),
                must_exist=op in {"read_text", "list", "stat"},
            )
            if op == "read_text" and not path.is_file():
                raise RootedPathError("read_text target must be a file")
            if op == "list" and not path.is_dir():
                raise RootedPathError("list target must be a directory")
            if op == "write_text":
                _verify_expected_hash(
                    path,
                    payload["expected_sha256"],  # type: ignore[arg-type]
                )
            return PreparedAction(action, {"path": path})

        if action.kind == "advisor":
            try:
                config, selected_model = self.policy.advisors.resolve(
                    str(payload["advisor"]),
                    (
                        None
                        if payload.get("model") is None
                        else str(payload["model"])
                    ),
                )
            except AdvisorError as exc:
                raise InvocationPolicyError(str(exc)) from exc
            timeout, output = self._limits(
                payload,
                timeout_max=config.timeout_ms_max,
                output_max=config.output_limit_bytes,
            )
            return PreparedAction(
                action,
                {
                    "model": selected_model,
                    "timeout_ms": timeout,
                    "output_limit_bytes": output,
                },
            )

        if action.kind == "tool":
            tool_id = str(payload["tool"])
            try:
                spec = self.policy.scripts[tool_id]
            except KeyError as exc:
                raise InvocationPolicyError(
                    f"tool {tool_id!r} is not registered"
                ) from exc
            if not spec.enabled:
                raise InvocationPolicyError(f"tool {tool_id!r} is disabled")
            if self.policy.tools_root is None:
                raise InvocationPolicyError("tools_root is not configured")
            script_path = _resolve_under_tools(
                Path(self.policy.tools_root),
                spec.relative_path,
            )
            actual = file_sha256(script_path)
            if actual.lower() != spec.sha256.lower():
                raise InvocationPolicyError(
                    f"tool {tool_id!r} SHA-256 mismatch"
                )
            timeout, output = self._limits(
                payload,
                timeout_max=spec.timeout_ms_max,
                output_max=spec.output_limit_bytes,
            )
            argv = [
                spec.interpreter,
                *spec.interpreter_args,
                str(script_path),
                *list(payload["args"]),
            ]
            return PreparedAction(
                action,
                {
                    "spec": spec,
                    "script_path": script_path,
                    "argv": tuple(argv),
                    "cwd": Path(self.policy.tools_root).resolve(strict=True),
                    "timeout_ms": timeout,
                    "output_limit_bytes": output,
                },
            )
        raise InvocationPolicyError(f"unsupported action kind {action.kind!r}")

    def validate_batch(self, batch: InvocationBatch) -> tuple[PreparedAction, ...]:
        if not self.policy.enabled:
            raise InvocationPolicyError("typed invocation execution is disabled")
        if batch.source_kind not in self.policy.allowed_sources:
            raise InvocationPolicyError(
                f"source {batch.source_kind!r} has no execution authority"
            )
        prepared: list[PreparedAction] = []
        local_keys: dict[str, str] = {}
        for action in batch.actions:
            key = action.idempotency_key
            prior_hash = local_keys.get(key)
            if prior_hash is not None and prior_hash != action.request_hash:
                raise IdempotencyConflict(
                    f"batch reuses idempotency key {key!r}"
                )
            local_keys[key] = action.request_hash
            cached = self.idempotency.check(key, action.request_hash)
            if cached is not None:
                # An exact replay has already crossed its policy/precondition
                # boundary and causes no new effect.  In particular, do not
                # re-evaluate a create-only write after that write succeeded.
                prepared.append(
                    PreparedAction(action, {"cached_result": cached})
                )
                continue
            prepared.append(self._prepare(action))
        return tuple(prepared)

    def _secret_values(self) -> tuple[str, ...]:
        return tuple(
            self.environment[name]
            for name in self.policy.redact_env_names
            if self.environment.get(name)
        )

    def _run_process(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdin: str | None,
        timeout_ms: int,
        output_limit_bytes: int,
    ) -> tuple[str, int | None, str, str, bool]:
        try:
            completed = self.runner(
                list(argv),
                cwd=str(cwd),
                input=None if stdin is None else stdin.encode("utf-8"),
                capture_output=True,
                text=False,
                timeout=timeout_ms / 1000.0,
                shell=False,
                env=dict(self.environment),
            )
            status = (
                "succeeded"
                if int(getattr(completed, "returncode", 0)) == 0
                else "failed"
            )
            code: int | None = int(getattr(completed, "returncode", 0))
            stdout = _decode_output(getattr(completed, "stdout", b""))
            stderr = _decode_output(getattr(completed, "stderr", b""))
        except subprocess.TimeoutExpired as exc:
            status = "timed_out"
            code = None
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
        secrets = self._secret_values()
        stdout = redact_text(stdout, secrets)
        stderr = redact_text(stderr, secrets)
        stdout, stderr, truncated = _cap_output_pair(
            stdout,
            stderr,
            output_limit_bytes,
        )
        return status, code, stdout, stderr, truncated

    def _result(
        self,
        prepared: PreparedAction,
        *,
        started: str,
        status: str,
        summary: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        truncated: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        return InvocationResult(
            action_id=prepared.action.action_id,
            kind=prepared.action.kind,
            request_hash=prepared.action.request_hash,
            status=status,
            target_region=(
                "advisor_input"
                if prepared.action.kind == "advisor"
                else "tool_results"
            ),
            started_at=started,
            finished_at=utc_now(),
            summary=summary,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            truncated=truncated,
            details=details or {},
        )

    def _execute_filesystem(
        self,
        prepared: PreparedAction,
        started: str,
    ) -> InvocationResult:
        payload = prepared.action.payload
        op = str(payload["op"])
        # Resolve again at the effect boundary.  The first resolution is the
        # whole-batch preflight; this second one catches a junction/symlink
        # swap made between validation and execution.
        path = resolve_rooted_path(
            self.policy,
            str(payload["root"]),
            str(payload["path"]),
            must_exist=op in {"read_text", "list", "stat"},
        )
        if path != Path(prepared.metadata["path"]):
            raise RootedPathError(
                "filesystem path changed after batch validation"
            )
        if op == "read_text":
            limit = int(payload["max_bytes"])
            raw = path.read_bytes()
            clipped = raw[:limit]
            text = clipped.decode(str(payload["encoding"]), errors="replace")
            truncated = len(raw) > limit
            if truncated:
                text += f"\n[output truncated at {limit} bytes]"
            return self._result(
                prepared,
                started=started,
                status="succeeded",
                summary=f"read {min(len(raw), limit)} bytes from {payload['path']}",
                stdout=redact_text(text, self._secret_values()),
                truncated=truncated,
                details={
                    "path": str(payload["path"]),
                    "bytes": len(raw),
                    "sha256": file_sha256(path),
                },
            )
        if op == "list":
            entries = sorted(item.name for item in path.iterdir())
            limit = int(payload["max_entries"])
            clipped = entries[:limit]
            truncated = len(entries) > limit
            output = canonical_json(clipped)
            if truncated:
                output += f"\n[entries truncated at {limit}]"
            return self._result(
                prepared,
                started=started,
                status="succeeded",
                summary=f"listed {len(clipped)} entries under {payload['path']}",
                stdout=output,
                truncated=truncated,
                details={"path": str(payload["path"]), "entries": len(entries)},
            )
        if op == "stat":
            info = path.stat()
            details: dict[str, Any] = {
                "path": str(payload["path"]),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
            if path.is_file():
                details["sha256"] = file_sha256(path)
            return self._result(
                prepared,
                started=started,
                status="succeeded",
                summary=f"statted {payload['path']}",
                stdout=canonical_json(details),
                details=details,
            )
        if op == "write_text":
            expected = payload["expected_sha256"]
            _verify_expected_hash(
                path,
                expected if isinstance(expected, str) else None,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = str(payload["text"]).encode(str(payload["encoding"]))
            descriptor, temporary = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.axon-",
                suffix=".tmp",
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
            digest = file_sha256(path)
            return self._result(
                prepared,
                started=started,
                status="succeeded",
                summary=f"atomically wrote {len(encoded)} bytes to {payload['path']}",
                details={
                    "path": str(payload["path"]),
                    "bytes": len(encoded),
                    "sha256": digest,
                    "atomic": True,
                },
            )
        if op == "mkdir":
            path.mkdir(
                parents=bool(payload["parents"]),
                exist_ok=bool(payload["exist_ok"]),
            )
            return self._result(
                prepared,
                started=started,
                status="succeeded",
                summary=f"created directory {payload['path']}",
                details={"path": str(payload["path"])},
            )
        raise InvocationPolicyError(f"unsupported filesystem op {op!r}")

    def _execute_one(self, prepared: PreparedAction) -> InvocationResult:
        started = utc_now()
        action = prepared.action
        payload = action.payload
        try:
            if action.kind == "filesystem":
                return self._execute_filesystem(prepared, started)
            if action.kind == "shell":
                cwd = resolve_rooted_path(
                    self.policy,
                    str(payload["root"]),
                    str(payload["cwd"]),
                    must_exist=True,
                )
                if cwd != Path(prepared.metadata["cwd"]) or not cwd.is_dir():
                    raise RootedPathError(
                        "shell cwd changed after batch validation"
                    )
                status, code, stdout, stderr, truncated = self._run_process(
                    prepared.metadata["argv"],  # type: ignore[arg-type]
                    cwd=cwd,
                    stdin=(
                        payload.get("stdin")
                        if isinstance(payload.get("stdin"), str)
                        else None
                    ),
                    timeout_ms=int(prepared.metadata["timeout_ms"]),
                    output_limit_bytes=int(
                        prepared.metadata["output_limit_bytes"]
                    ),
                )
                return self._result(
                    prepared,
                    started=started,
                    status=status,
                    summary=f"shell action {status}",
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=code,
                    truncated=truncated,
                    details={"mode": payload["mode"]},
                )
            if action.kind == "advisor":
                call: AdvisorCallResult = invoke_registered_advisor(
                    self.policy.advisors,
                    advisor_id=str(payload["advisor"]),
                    model=(
                        None
                        if payload.get("model") is None
                        else str(payload["model"])
                    ),
                    prompt=str(payload["prompt"]),
                    timeout_ms=int(prepared.metadata["timeout_ms"]),
                    output_limit_bytes=int(
                        prepared.metadata["output_limit_bytes"]
                    ),
                    runner=self.runner,
                    http_post=self.http_post,
                    environment=self.environment,
                )
                return self._result(
                    prepared,
                    started=started,
                    status=call.status,
                    summary=f"advisor {payload['advisor']} {call.status}",
                    stdout=call.output,
                    stderr=call.stderr,
                    exit_code=call.exit_code,
                    truncated=call.truncated,
                    details={
                        "advisor": payload["advisor"],
                        "model": prepared.metadata["model"],
                        **dict(call.provider_metadata),
                    },
                )
            if action.kind == "tool":
                spec = prepared.metadata["spec"]
                if self.policy.tools_root is None:
                    raise InvocationPolicyError("tools_root is not configured")
                script_path = _resolve_under_tools(
                    Path(self.policy.tools_root),
                    spec.relative_path,
                )
                if script_path != Path(prepared.metadata["script_path"]):
                    raise RootedPathError(
                        "registered tool path changed after validation"
                    )
                if file_sha256(script_path).lower() != spec.sha256.lower():
                    raise InvocationPolicyError(
                        f"tool {payload['tool']!r} changed after validation"
                    )
                status, code, stdout, stderr, truncated = self._run_process(
                    prepared.metadata["argv"],  # type: ignore[arg-type]
                    cwd=Path(prepared.metadata["cwd"]),
                    stdin=None,
                    timeout_ms=int(prepared.metadata["timeout_ms"]),
                    output_limit_bytes=int(
                        prepared.metadata["output_limit_bytes"]
                    ),
                )
                return self._result(
                    prepared,
                    started=started,
                    status=status,
                    summary=f"tool {payload['tool']} {status}",
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=code,
                    truncated=truncated,
                    details={
                        "tool": payload["tool"],
                        "script_sha256": spec.sha256.lower(),
                    },
                )
        except Exception as exc:
            error = redact_text(
                f"{type(exc).__name__}: {exc}",
                self._secret_values(),
            )
            return self._result(
                prepared,
                started=started,
                status="failed",
                summary=f"{action.kind} action failed",
                stderr=error,
            )
        raise InvocationPolicyError(f"unsupported action kind {action.kind!r}")

    def execute_batch(
        self,
        batch: InvocationBatch,
    ) -> tuple[InvocationResult, ...]:
        # Crucial fail-closed boundary: every action is structurally and
        # policy-validated before the first effect.
        prepared = self.validate_batch(batch)
        results: list[InvocationResult] = []
        for item in prepared:
            action = item.action
            cached = self.idempotency.check(
                action.idempotency_key,
                action.request_hash,
            )
            if cached is not None:
                results.append(cached)
                continue
            result = self._execute_one(item)
            self.idempotency.record(
                action.idempotency_key,
                action.request_hash,
                result,
            )
            results.append(result)
        return tuple(results)

    def execute_text(
        self,
        text: str,
        *,
        source_kind: str = "user_input",
    ) -> tuple[InvocationResult, ...]:
        batch = parse_invocation_envelope(text, source_kind=source_kind)
        if batch is None:
            return ()
        return self.execute_batch(batch)


def render_results_for_field(results: Sequence[InvocationResult]) -> str:
    """Render sealed-region records; this output is never an invocation source."""

    header = canonical_json(
        {
            "schema": "axon.invocation-results-field.v1",
            "count": len(results),
        }
    )
    rows = [canonical_json(result.to_dict()) for result in results]
    return "\n".join([header, *rows])


__all__ = [
    "START_MARKER",
    "END_MARKER",
    "InvocationError",
    "InvocationParseError",
    "InvocationValidationError",
    "InvocationPolicyError",
    "RootedPathError",
    "IdempotencyConflict",
    "InvocationAction",
    "InvocationBatch",
    "ProgramSpec",
    "CommandModeSpec",
    "ScriptSpec",
    "InvocationPolicy",
    "InvocationResult",
    "IdempotencyStore",
    "InvocationExecutor",
    "canonical_json",
    "canonical_sha256",
    "file_sha256",
    "strict_json_object",
    "parse_invocation_envelope",
    "resolve_rooted_path",
    "render_results_for_field",
]
