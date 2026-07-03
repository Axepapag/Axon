"""Agent manifest load/validate for the Round Table Orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ManifestDir = Path(__file__).parent / "manifests"


@dataclass(frozen=True)
class WakeSpec:
    argv: list[str]
    prompt_via: Literal["argv", "stdin", "file"]
    timeout_seconds: int
    workdir: str


@dataclass(frozen=True)
class AgentManifest:
    client_id: str
    display_name: str
    wake: WakeSpec
    capabilities: list[str]
    identity_stamp: str
    enabled: bool
    notes: str

    @property
    def is_manual(self) -> bool:
        """A manual seat waits for a reply file instead of invoking a subprocess."""
        return not self.wake.argv


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

    argv = wake.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ValueError(f"manifest {p.name}: wake.argv must be a list of strings")

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

    return AgentManifest(
        client_id=_require_str(data, "client_id", p),
        display_name=_require_str(data, "display_name", p),
        wake=WakeSpec(
            argv=argv,
            prompt_via=prompt_via,
            timeout_seconds=timeout,
            workdir=workdir,
        ),
        capabilities=list(data.get("capabilities", [])),
        identity_stamp=_require_str(data, "identity_stamp", p),
        enabled=bool(data.get("enabled", True)),
        notes=str(data.get("notes", "")),
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
