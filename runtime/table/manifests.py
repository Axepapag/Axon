"""Agent manifest load/validate for the Round Table Orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ManifestDir = Path(__file__).parent / "manifests"


@dataclass(frozen=True)
class SessionSpec:
    parse_regex: str | None = None


@dataclass(frozen=True)
class WakeSpec:
    argv: list[str]
    prompt_via: Literal["argv", "stdin", "file"]
    timeout_seconds: int
    workdir: str
    resume_argv: list[str] | None = None
    model_arg: str | None = None
    type: str = "cli"                 # "cli" | "terminal" (v1.2)
    term_id: str | None = None        # bus client id of the terminal bridge


@dataclass(frozen=True)
class AgentManifest:
    client_id: str
    display_name: str
    wake: WakeSpec
    capabilities: list[str]
    identity_stamp: str
    enabled: bool
    notes: str
    model: str | None = None
    session: SessionSpec | None = None

    @property
    def is_manual(self) -> bool:
        """A manual seat waits for a reply file instead of invoking a subprocess."""
        return not self.wake.argv and self.wake.type == "cli"

    @property
    def is_terminal(self) -> bool:
        """A terminal seat is driven through a live TermBridge on the bus."""
        return self.wake.type == "terminal"

    @property
    def parse_regex(self) -> str | None:
        return self.session.parse_regex if self.session else None


def _require_str(d: dict[str, Any], key: str, path: Path) -> str:
    if key not in d:
        raise ValueError(f"manifest {path.name}: missing {key!r}")
    if not isinstance(d[key], str):
        raise ValueError(f"manifest {path.name}: {key!r} must be a string")
    return d[key]


def _require_int(d: dict[str, Any], key: str, path: Path, default: int | None = None) -> int:
    val = d.get(key, default)
    if val is None:
        raise ValueError(f"manifest {path.name}: missing {key!r}")
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValueError(f"manifest {path.name}: {key!r} must be an integer")
    return val


def load_manifest(path: str | Path) -> AgentManifest:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest {p.name}: must be a JSON object")

    wake = data.get("wake", {})
    if not isinstance(wake, dict):
        raise ValueError(f"manifest {p.name}: 'wake' must be an object")

    wake_type = wake.get("type", "cli")
    if wake_type not in ("cli", "terminal"):
        raise ValueError(f"manifest {p.name}: wake.type must be cli|terminal")

    term_id = wake.get("term_id")
    if term_id is not None and not isinstance(term_id, str):
        raise ValueError(f"manifest {p.name}: wake.term_id must be a string")
    if wake_type == "terminal" and not term_id:
        raise ValueError(f"manifest {p.name}: wake.type=terminal requires wake.term_id")

    argv = wake.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ValueError(f"manifest {p.name}: wake.argv must be a list of strings")

    resume_argv = wake.get("resume_argv")
    if resume_argv is not None and (
        not isinstance(resume_argv, list) or not all(isinstance(a, str) for a in resume_argv)
    ):
        raise ValueError(f"manifest {p.name}: wake.resume_argv must be a list of strings")

    model_arg = wake.get("model_arg")
    if model_arg is not None and not isinstance(model_arg, str):
        raise ValueError(f"manifest {p.name}: wake.model_arg must be a string")

    prompt_via = wake.get("prompt_via", "argv")
    if prompt_via not in ("argv", "stdin", "file"):
        raise ValueError(f"manifest {p.name}: wake.prompt_via must be argv|stdin|file")

    # A manual seat has empty argv and ignores prompt_via.
    placeholders = {p for arg in argv for p in ("{prompt}", "{prompt_file}") if p in arg}
    if argv and prompt_via == "argv" and placeholders and "{prompt}" not in placeholders:
        raise ValueError(f"manifest {p.name}: wake.argv must contain '{{prompt}}' placeholder when prompt_via=argv")

    timeout = _require_int(wake, "timeout_seconds", p, default=600)
    workdir = wake.get("workdir", str(Path.cwd()))
    if not isinstance(workdir, str):
        raise ValueError(f"manifest {p.name}: wake.workdir must be a string")

    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(f"manifest {p.name}: model must be a string")

    for template in [argv, resume_argv or []]:
        joined = " ".join(template)
        if "{model}" in joined and model is None and model_arg is None:
            raise ValueError(
                f"manifest {p.name}: '{{model}}' placeholder requires a manifest 'model' or 'wake.model_arg' value"
            )

    session = data.get("session")
    session_spec: SessionSpec | None = None
    if session is not None:
        if not isinstance(session, dict):
            raise ValueError(f"manifest {p.name}: session must be an object")
        parse_regex = session.get("parse_regex")
        if parse_regex is not None and not isinstance(parse_regex, str):
            raise ValueError(f"manifest {p.name}: session.parse_regex must be a string")
        session_spec = SessionSpec(parse_regex=parse_regex or None)

    return AgentManifest(
        client_id=_require_str(data, "client_id", p),
        display_name=_require_str(data, "display_name", p),
        wake=WakeSpec(
            argv=argv,
            prompt_via=prompt_via,
            timeout_seconds=timeout,
            workdir=workdir,
            resume_argv=resume_argv,
            model_arg=model_arg,
            type=wake_type,
            term_id=term_id,
        ),
        capabilities=list(data.get("capabilities", [])),
        identity_stamp=_require_str(data, "identity_stamp", p),
        enabled=bool(data.get("enabled", True)),
        notes=str(data.get("notes", "")),
        model=model,
        session=session_spec,
    )


def load_all_manifests(manifests_dir: str | Path | None = None) -> dict[str, AgentManifest]:
    d = Path(manifests_dir) if manifests_dir else ManifestDir
    out: dict[str, AgentManifest] = {}
    for path in sorted(d.glob("*.json")):
        manifest = load_manifest(path)
        out[manifest.client_id] = manifest
    return out


def stable_ids() -> set[str]:
    """Return the bus stable-ID list from AGENT_BUS_ACCESS.md."""
    return {
        "codex",
        "kimi",
        "hermes",
        "claude",
        "chatgpt",
        "gptu-bridge",
        "phone",
        "browser-ui",
        "axon",
        # orchestrator's own table identity is not a seated participant.
    }
