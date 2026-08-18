"""Registered advisor providers for the Axon typed-invocation runtime.

Provider configuration is deliberately data-only.  It may name environment
variables that contain credentials, but it may never contain credential
values.  Callers choose a registered advisor and an allowed model; they cannot
override the executable, endpoint, or credential source at invocation time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
from typing import Any, Callable, Mapping
from urllib import request as urllib_request


class AdvisorError(RuntimeError):
    """Base error for advisor registration, policy, or execution failures."""


class AdvisorConfigError(AdvisorError):
    """An advisor descriptor is malformed or contains forbidden data."""


class AdvisorDisabledError(AdvisorError):
    """The requested advisor or model is not enabled by durable policy."""


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ADVISOR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_FORBIDDEN_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
}


def _require_absolute_local_program(value: str, label: str) -> None:
    windows = PureWindowsPath(value)
    native = Path(value)
    if not value or "\x00" in value:
        raise AdvisorConfigError(f"{label} must be a non-empty path")
    if not windows.is_absolute() and not native.is_absolute():
        raise AdvisorConfigError(f"{label} must be an absolute path")
    if value.startswith(("\\\\", "//", "\\\\?\\", "\\\\.\\")):
        raise AdvisorConfigError(f"{label} cannot be a UNC or device path")


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
        raise AdvisorConfigError(f"{label} has unknown fields: {sorted(unknown)}")
    if missing:
        raise AdvisorConfigError(f"{label} is missing fields: {sorted(missing)}")
    forbidden = {key for key in value if key.lower() in _FORBIDDEN_CONFIG_KEYS}
    if forbidden:
        raise AdvisorConfigError(
            f"{label} may reference secret environment names but not secret values: "
            f"{sorted(forbidden)}"
        )


def _as_tuple_of_strings(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AdvisorConfigError(f"{label} must be a non-empty string array")
    return tuple(value)


def _cap_utf8(text: str, limit: int) -> tuple[str, bool]:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return text, False
    marker = f"\n[output truncated at {limit} bytes]"
    marker_raw = marker.encode("utf-8")
    keep = max(0, limit - len(marker_raw))
    clipped = raw[:keep].decode("utf-8", errors="ignore") + marker
    return clipped, True


def redact_text(text: str, secret_values: tuple[str, ...]) -> str:
    """Replace configured secret values without ever logging their names/values."""

    redacted = text
    for value in sorted(
        {item for item in secret_values if item},
        key=len,
        reverse=True,
    ):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted


@dataclass(frozen=True, slots=True)
class AdvisorConfig:
    """One immutable, registered advisor descriptor."""

    advisor_id: str
    adapter: str
    enabled: bool = False
    allowed_models: tuple[str, ...] = ()
    default_model: str = ""
    timeout_ms_max: int = 120_000
    output_limit_bytes: int = 1_048_576
    secret_env_names: tuple[str, ...] = ()
    argv_template: tuple[str, ...] = ()
    endpoint: str | None = None
    api_key_env: str | None = None
    # Header values are always loaded from environment variables.  Literal
    # credential-bearing headers never belong in durable runtime config.
    extra_header_env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not _ADVISOR_ID.fullmatch(self.advisor_id):
            raise AdvisorConfigError(f"invalid advisor_id {self.advisor_id!r}")
        if self.adapter not in {"cli", "openai_http"}:
            raise AdvisorConfigError("adapter must be cli or openai_http")
        if not self.allowed_models:
            raise AdvisorConfigError("allowed_models must not be empty")
        if self.default_model not in self.allowed_models:
            raise AdvisorConfigError("default_model must be in allowed_models")
        if (
            isinstance(self.timeout_ms_max, bool)
            or self.timeout_ms_max <= 0
            or self.timeout_ms_max > 3_600_000
        ):
            raise AdvisorConfigError("timeout_ms_max must be in 1..3600000")
        if (
            isinstance(self.output_limit_bytes, bool)
            or self.output_limit_bytes <= 0
            or self.output_limit_bytes > 16 * 1024 * 1024
        ):
            raise AdvisorConfigError(
                "output_limit_bytes must be in 1..16777216"
            )
        for name in self.secret_env_names:
            if not _ENV_NAME.fullmatch(name):
                raise AdvisorConfigError(f"invalid secret environment name {name!r}")
        for header, environment_name in self.extra_header_env:
            if (
                not header
                or any(character in header for character in "\r\n:")
                or not _ENV_NAME.fullmatch(environment_name)
            ):
                raise AdvisorConfigError(
                    "extra_header_env must map safe header names to "
                    "environment variable names"
                )
            if header.lower() in {"authorization", "content-type"}:
                raise AdvisorConfigError(
                    f"extra header {header!r} is reserved"
                )
        if self.api_key_env is not None and not _ENV_NAME.fullmatch(
            self.api_key_env
        ):
            raise AdvisorConfigError("api_key_env must be an environment name")
        if self.adapter == "cli":
            if not self.argv_template:
                raise AdvisorConfigError("CLI advisor requires argv_template")
            _require_absolute_local_program(
                self.argv_template[0],
                "CLI advisor executable",
            )
            joined = "\n".join(self.argv_template)
            if "{prompt}" not in joined:
                raise AdvisorConfigError(
                    "CLI argv_template must contain {prompt}"
                )
            if self.endpoint is not None or self.api_key_env is not None:
                raise AdvisorConfigError(
                    "CLI advisor cannot define HTTP endpoint/api_key_env"
                )
        else:
            if not self.endpoint or not self.endpoint.startswith(
                ("http://", "https://")
            ):
                raise AdvisorConfigError(
                    "openai_http advisor requires an http(s) endpoint"
                )
            if self.argv_template:
                raise AdvisorConfigError(
                    "openai_http advisor cannot define argv_template"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AdvisorConfig":
        allowed = {
            "advisor_id",
            "adapter",
            "enabled",
            "allowed_models",
            "default_model",
            "timeout_ms_max",
            "output_limit_bytes",
            "secret_env_names",
            "argv_template",
            "endpoint",
            "api_key_env",
            "extra_header_env",
        }
        _strict_keys(
            value,
            allowed=allowed,
            required={"advisor_id", "adapter", "allowed_models", "default_model"},
            label="advisor config",
        )
        allowed_models = _as_tuple_of_strings(
            value["allowed_models"], label="allowed_models"
        )
        secret_names_raw = value.get("secret_env_names", ())
        if not isinstance(secret_names_raw, (list, tuple)) or not all(
            isinstance(item, str) for item in secret_names_raw
        ):
            raise AdvisorConfigError("secret_env_names must be a string array")
        argv_raw = value.get("argv_template", ())
        if not isinstance(argv_raw, (list, tuple)) or not all(
            isinstance(item, str) and item for item in argv_raw
        ):
            raise AdvisorConfigError("argv_template must be a string array")
        headers_raw = value.get("extra_header_env", {})
        if not isinstance(headers_raw, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in headers_raw.items()
        ):
            raise AdvisorConfigError(
                "extra_header_env must map header names to environment names"
            )
        return cls(
            advisor_id=str(value["advisor_id"]),
            adapter=str(value["adapter"]),
            enabled=bool(value.get("enabled", False)),
            allowed_models=allowed_models,
            default_model=str(value["default_model"]),
            timeout_ms_max=int(value.get("timeout_ms_max", 120_000)),
            output_limit_bytes=int(
                value.get("output_limit_bytes", 1_048_576)
            ),
            secret_env_names=tuple(secret_names_raw),
            argv_template=tuple(argv_raw),
            endpoint=(
                None
                if value.get("endpoint") is None
                else str(value["endpoint"])
            ),
            api_key_env=(
                None
                if value.get("api_key_env") is None
                else str(value["api_key_env"])
            ),
            extra_header_env=tuple(
                sorted((str(key), str(item)) for key, item in headers_raw.items())
            ),
        )


@dataclass(slots=True)
class AdvisorRegistry:
    """Fail-closed registry keyed by stable advisor IDs."""

    _items: dict[str, AdvisorConfig] = field(default_factory=dict)

    def register(self, config: AdvisorConfig) -> None:
        if config.advisor_id in self._items:
            raise AdvisorConfigError(
                f"duplicate advisor_id {config.advisor_id!r}"
            )
        self._items[config.advisor_id] = config

    def get(self, advisor_id: str) -> AdvisorConfig:
        try:
            return self._items[advisor_id]
        except KeyError as exc:
            raise AdvisorDisabledError(
                f"advisor {advisor_id!r} is not registered"
            ) from exc

    def resolve(
        self,
        advisor_id: str,
        model: str | None,
    ) -> tuple[AdvisorConfig, str]:
        config = self.get(advisor_id)
        if not config.enabled:
            raise AdvisorDisabledError(f"advisor {advisor_id!r} is disabled")
        selected = model or config.default_model
        if selected not in config.allowed_models:
            raise AdvisorDisabledError(
                f"model {selected!r} is not allowed for advisor {advisor_id!r}"
            )
        return config, selected


@dataclass(frozen=True, slots=True)
class AdvisorCallResult:
    status: str
    output: str
    stderr: str = ""
    exit_code: int | None = None
    truncated: bool = False
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)


ProcessRunner = Callable[..., Any]
HttpPost = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Any]


def _secret_values(
    config: AdvisorConfig,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    names = set(config.secret_env_names)
    if config.api_key_env:
        names.add(config.api_key_env)
    names.update(
        environment_name for _, environment_name in config.extra_header_env
    )
    return tuple(
        environment[name]
        for name in sorted(names)
        if environment.get(name)
    )


def invoke_cli_advisor(
    config: AdvisorConfig,
    *,
    model: str,
    prompt: str,
    timeout_ms: int,
    output_limit_bytes: int,
    runner: ProcessRunner | None = None,
    environment: Mapping[str, str] | None = None,
) -> AdvisorCallResult:
    if config.adapter != "cli":
        raise AdvisorConfigError("invoke_cli_advisor requires a CLI config")
    if timeout_ms > config.timeout_ms_max:
        raise AdvisorDisabledError("advisor timeout exceeds registered maximum")
    limit = min(output_limit_bytes, config.output_limit_bytes)
    if limit <= 0:
        raise AdvisorConfigError("output limit must be positive")
    argv = [
        part.replace("{model}", model).replace("{prompt}", prompt)
        for part in config.argv_template
    ]
    run = runner or subprocess.run
    env = dict(os.environ if environment is None else environment)
    secrets = _secret_values(config, env)
    try:
        completed = run(
            argv,
            capture_output=True,
            text=False,
            timeout=timeout_ms / 1000.0,
            shell=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raw_stdout = exc.stdout or b""
        raw_stderr = exc.stderr or b""
        stdout = (
            raw_stdout.decode("utf-8", errors="replace")
            if isinstance(raw_stdout, bytes)
            else str(raw_stdout)
        )
        stderr = (
            raw_stderr.decode("utf-8", errors="replace")
            if isinstance(raw_stderr, bytes)
            else str(raw_stderr)
        )
        stdout, out_cut = _cap_utf8(redact_text(stdout, secrets), limit)
        stderr, err_cut = _cap_utf8(redact_text(stderr, secrets), limit)
        return AdvisorCallResult(
            status="timed_out",
            output=stdout,
            stderr=stderr,
            exit_code=None,
            truncated=out_cut or err_cut,
        )
    stdout_raw = getattr(completed, "stdout", b"") or b""
    stderr_raw = getattr(completed, "stderr", b"") or b""
    stdout = (
        stdout_raw.decode("utf-8", errors="replace")
        if isinstance(stdout_raw, bytes)
        else str(stdout_raw)
    )
    stderr = (
        stderr_raw.decode("utf-8", errors="replace")
        if isinstance(stderr_raw, bytes)
        else str(stderr_raw)
    )
    stdout, out_cut = _cap_utf8(redact_text(stdout, secrets), limit)
    stderr, err_cut = _cap_utf8(redact_text(stderr, secrets), limit)
    code = int(getattr(completed, "returncode", 0))
    return AdvisorCallResult(
        status="succeeded" if code == 0 else "failed",
        output=stdout,
        stderr=stderr,
        exit_code=code,
        truncated=out_cut or err_cut,
    )


def _default_http_post(
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    req = urllib_request.Request(
        endpoint,
        data=encoded,
        method="POST",
        headers=dict(headers),
    )
    with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
        return response.read()


def invoke_openai_http_advisor(
    config: AdvisorConfig,
    *,
    model: str,
    prompt: str,
    timeout_ms: int,
    output_limit_bytes: int,
    http_post: HttpPost | None = None,
    environment: Mapping[str, str] | None = None,
) -> AdvisorCallResult:
    if config.adapter != "openai_http":
        raise AdvisorConfigError(
            "invoke_openai_http_advisor requires an openai_http config"
        )
    if timeout_ms > config.timeout_ms_max:
        raise AdvisorDisabledError("advisor timeout exceeds registered maximum")
    limit = min(output_limit_bytes, config.output_limit_bytes)
    env = dict(os.environ if environment is None else environment)
    secrets = _secret_values(config, env)
    headers = {
        "Content-Type": "application/json",
    }
    for header, environment_name in config.extra_header_env:
        value = env.get(environment_name)
        if not value:
            raise AdvisorDisabledError(
                f"advisor header environment {environment_name!r} is unset"
            )
        headers[header] = value
    if config.api_key_env:
        key = env.get(config.api_key_env)
        if not key:
            raise AdvisorDisabledError(
                f"advisor credential environment {config.api_key_env!r} is unset"
            )
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    post = http_post or _default_http_post
    try:
        response = post(
            config.endpoint or "",
            headers,
            payload,
            timeout_ms / 1000.0,
        )
        if isinstance(response, bytes):
            parsed = json.loads(response.decode("utf-8"))
        elif isinstance(response, str):
            parsed = json.loads(response)
        elif isinstance(response, Mapping):
            parsed = dict(response)
        else:
            raise AdvisorError("HTTP advisor returned an unsupported response")
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AdvisorError("HTTP advisor response has no choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise AdvisorError("HTTP advisor choice is malformed")
        message = first.get("message")
        if not isinstance(message, Mapping) or not isinstance(
            message.get("content"), str
        ):
            raise AdvisorError("HTTP advisor choice has no text content")
        output, truncated = _cap_utf8(
            redact_text(message["content"], secrets),
            limit,
        )
        metadata = {
            key: parsed[key]
            for key in ("id", "created", "usage")
            if key in parsed
        }
        return AdvisorCallResult(
            status="succeeded",
            output=output,
            truncated=truncated,
            provider_metadata=metadata,
        )
    except TimeoutError:
        return AdvisorCallResult(status="timed_out", output="")
    except AdvisorError:
        raise
    except Exception as exc:
        raise AdvisorError(f"HTTP advisor call failed: {type(exc).__name__}") from exc


def invoke_registered_advisor(
    registry: AdvisorRegistry,
    *,
    advisor_id: str,
    model: str | None,
    prompt: str,
    timeout_ms: int,
    output_limit_bytes: int,
    runner: ProcessRunner | None = None,
    http_post: HttpPost | None = None,
    environment: Mapping[str, str] | None = None,
) -> AdvisorCallResult:
    config, selected_model = registry.resolve(advisor_id, model)
    if config.adapter == "cli":
        return invoke_cli_advisor(
            config,
            model=selected_model,
            prompt=prompt,
            timeout_ms=timeout_ms,
            output_limit_bytes=output_limit_bytes,
            runner=runner,
            environment=environment,
        )
    return invoke_openai_http_advisor(
        config,
        model=selected_model,
        prompt=prompt,
        timeout_ms=timeout_ms,
        output_limit_bytes=output_limit_bytes,
        http_post=http_post,
        environment=environment,
    )


__all__ = [
    "AdvisorError",
    "AdvisorConfigError",
    "AdvisorDisabledError",
    "AdvisorConfig",
    "AdvisorRegistry",
    "AdvisorCallResult",
    "redact_text",
    "invoke_cli_advisor",
    "invoke_openai_http_advisor",
    "invoke_registered_advisor",
]
