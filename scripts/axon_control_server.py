"""Minimal authenticated control plane for a cloud-hosted Axon body.

This server is intentionally transport-only. It reads canonical runtime/trainer
surfaces and exposes them to Axon Home; it does not bypass Heart or Trainer
mutation authority and never falls back to shell commands.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from runtime.trainer.organ import TrainerCommandKind, TrainerOrgan, TrainerOrganCommand

STARTED = time.monotonic()
STATE_ROOT = Path(os.environ.get("AXON_STATE_ROOT", r"D:\Axon\State")).resolve(strict=False)
TOKEN = os.environ.get("AXON_CONTROL_TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("AXON_CONTROL_TOKEN must be set before starting the control plane")

app = FastAPI(title="Axon Control Plane", version="1.0.0")
trainer = TrainerOrgan(state_root=STATE_ROOT)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _auth(authorization: str | None = Header(default=None)) -> None:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="bearer token required")
    presented = authorization[len(prefix) :]
    if not secrets.compare_digest(presented, TOKEN):
        raise HTTPException(status_code=403, detail="invalid bearer token")


def _heart_health() -> dict[str, Any] | None:
    return _read_json(STATE_ROOT / "active" / "heart" / "health_latest.json")


def _field_head() -> dict[str, Any] | None:
    return _read_json(STATE_ROOT / "active" / "branches" / "active" / "HEAD.json")


def _training_progress() -> dict[str, Any] | None:
    candidates = [
        STATE_ROOT / "axon_observability" / "trainer" / "current.json",
        STATE_ROOT.parent / "axon_observability" / "trainer" / "current.json",
    ]
    for path in candidates:
        value = _read_json(path)
        if value is not None:
            return value
    return None


@app.get("/v1/health", dependencies=[Depends(_auth)])
def health() -> dict[str, Any]:
    heart = _heart_health()
    return {
        "schema": "axon-controlplane-health-v1",
        "control_plane": {
            "status": "ok" if heart is not None else "degraded",
            "version": app.version,
            "api_versions": ["v1"],
            "uptime_seconds": int(time.monotonic() - STARTED),
        },
        "state_root": str(STATE_ROOT),
        "heart": heart,
    }


@app.get("/v1/field/head", dependencies=[Depends(_auth)])
def field_head() -> dict[str, Any]:
    value = _field_head()
    if value is None:
        raise HTTPException(status_code=404, detail="canonical branch HEAD unavailable")
    return value


@app.get("/v1/trainer/status", dependencies=[Depends(_auth)])
def trainer_status() -> dict[str, Any]:
    result = trainer.dispatch(
        TrainerOrganCommand(kind=TrainerCommandKind.STATUS, arguments={"detail": "summary"}, requested_by="axon-home")
    )
    return result.to_canonical_dict()


@app.get("/v1/training/progress", dependencies=[Depends(_auth)])
def training_progress() -> dict[str, Any]:
    value = _training_progress()
    return value or {
        "schema": "axon-training-progress-unavailable-v1",
        "status": "idle",
        "detail": "No durable training progress event is present yet.",
    }


@app.get("/v1/runtime/summary", dependencies=[Depends(_auth)])
def runtime_summary() -> dict[str, Any]:
    status = trainer.dispatch(
        TrainerOrganCommand(kind=TrainerCommandKind.STATUS, arguments={"detail": "summary"}, requested_by="axon-home")
    )
    return {
        "schema": "axon-runtime-summary-v1",
        "heart": _heart_health(),
        "field_head": _field_head(),
        "trainer": status.to_canonical_dict(),
        "training_progress": _training_progress(),
    }
