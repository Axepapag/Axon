"""First live Axon Lab API: evidence and registries, without a fabricated trainer."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import datetime as dt
import hashlib
import json
from pathlib import Path
import threading
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from substrate.native import ALPHABET
from core.manifests import component_manifests
from tools.training_preflight import run_preflight
from .store import Registry, canonical, local_state
from .validation import validate_graph
from .core_checks import check_core

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "axon-lab-api-v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class ApiProblem(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        self.status, self.code, self.message, self.details = status, code, message, details or {}


def create_app(*, root: Path = ROOT, state_path: Path | None = None,
               components: list[dict] | None = None, preflight=run_preflight,
               frontend_path: Path | None = None, core_probe=check_core) -> FastAPI:
    root = Path(root).resolve()
    registry = Registry(state_path or local_state(root) / "registry.sqlite3")
    # The architecture owner supplies real contracts. Names alone are not adapters.
    catalog = components if components is not None else component_manifests() + [
        {"id": kind, "name": name, "version": None, "status": "not_integrated",
         "execution_eligible": False, "reason": "Core owner's versioned component contract and adapter are pending."}
        for kind, name in (("gru", "GRU"), ("ffn", "Feed-forward network"),
                           ("transformer", "Transformer / attention"), ("custom_recurrent", "Custom recurrent cell"))
    ]
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="axon-preflight")
    running = threading.Lock()
    core_evidence = []

    @asynccontextmanager
    async def lifespan(app):
        nonlocal core_evidence
        if any(c["id"] == "axon.core_reasoning_gru" for c in catalog):
            core_evidence = await asyncio.to_thread(core_probe)
        yield
        # A worker uses finite probe timeouts; wait without blocking the event loop.
        await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=False)

    app = FastAPI(title="Axon Lab", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry

    @app.exception_handler(ApiProblem)
    async def problem(request, exc):
        return JSONResponse(status_code=exc.status, content={"schema_version": SCHEMA,
            "code": exc.code, "message": exc.message, "details": exc.details,
            "retryable": exc.status == 503, "request_id": str(uuid.uuid4())})

    @app.exception_handler(HTTPException)
    async def missing(request, exc):
        return await problem(request, ApiProblem(exc.status_code, "not_found", str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return await problem(request, ApiProblem(422, "invalid_request", "Request fields are invalid."))

    def items(kind):
        return {"schema_version": SCHEMA, "items": registry.items(kind)}

    def latest_check():
        complete = [o for o in registry.items("operation") if o["status"] in ("completed", "failed", "interrupted") and o.get("kind") == "preflight"]
        return complete[-1] if complete else None

    def readiness_data():
        op = latest_check()
        report = op.get("result") if op else None
        checks = [{"id": "foundation", "name": "Selected-device foundation checks",
                   "status": ("passed" if report.get("foundation_passed") else "failed") if report else ("failed" if op else "not_checked"),
                   "reason": op.get("error", {}).get("message", "Review the actual preflight result; it does not authorize training.") if op else "Run a selected-device foundation check.",
                   "evidence": op["operation_id"] if op else None, "updated_at": op.get("finished_at") if op else None}]
        checks += core_evidence
        checks += [{"id": name, "status": "not_integrated", "reason": reason}
                   for name, reason in (("core_runtime", "Core adapter exists; HeartHost execution coordinator is pending."),
                       ("dataset_split", "Inventory is not a frozen deduplicated training/test split."),
                       ("checkpoint_resume", "Complete live-state recovery is not implemented yet."),
                       ("backup_restore", "No valuable-artifact backup/restore drill is verified."),
                       ("operator_controls", "The complete phone/desktop training path is not accepted."))]
        return {"schema_version": SCHEMA, "training_authorized": False,
                "allowed_actions": ["preflight"], "checks": checks}

    @app.get("/api/v1/health")
    def health():
        return {"schema_version": SCHEMA, "status": "ok", "backend_version": "0.1.0",
                "workspace_identity": hashlib.sha256(str(root).casefold().encode()).hexdigest()[:12]}

    @app.get("/api/v1/capabilities")
    def capabilities():
        devices = [{"id": name, "name": label, "status": "unavailable", "reason": "Run an explicit device preflight first."}
                   for name, label in (("cpu", "CPU"), ("cuda:0", "CUDA GPU 0"))]
        op = latest_check()
        if op:
            for check in op.get("result", {}).get("checks", []):
                if check.get("id") == "device" and check.get("status") == "pass":
                    info = check.get("details", {})
                    for device in devices:
                        if device["id"] == info.get("selected_device"):
                            device.update(status="available", reason="A real tensor operation passed in the recorded preflight.", evidence=op["operation_id"])
        return {"schema_version": SCHEMA, "backend_version": "0.1.0", "components": catalog,
                "devices": devices, "providers": [{"id": "local", "name": "Local execution", "status": "available", "reason": "Foundation service available; training adapter pending."},
                    {"id": "kaggle", "name": "Kaggle", "status": "not_integrated", "reason": "No launch adapter configured."},
                    {"id": "colab", "name": "Colab", "status": "not_integrated", "reason": "No launch adapter configured."}],
                "native_alphabet": ALPHABET, "supported_actions": ["preflight", "validate_architecture", "register_architecture"],
                "feature_flags": {"training": False, "inference": False, "tensor_inspection": False}}

    @app.get("/api/v1/readiness")
    def readiness():
        return readiness_data()

    @app.post("/api/v1/preflight", status_code=202)
    def start_preflight(spec: dict):
        command_id, device = spec.get("command_id"), spec.get("device", "auto")
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 128 or device not in ("auto", "cpu", "cuda:0"):
            raise ApiProblem(422, "invalid_preflight", "Supply command_id and device auto, cpu or cuda:0.")
        normalized = {"kind": "preflight", "device": device}
        with registry.lock:
            proposed = {"schema_version": SCHEMA, "operation_id": str(uuid.uuid4()), "kind": "preflight", "status": "queued", "device": device, "created_at": now()}
            # Existing command retries must work even while their worker is busy.
            with registry.connect() as db:
                prior = db.execute("SELECT operation_id FROM commands WHERE id=?", (command_id,)).fetchone()
            if not prior and not running.acquire(blocking=False):
                raise ApiProblem(409, "preflight_busy", "A foundation check is already running.")
            try:
                operation = registry.begin_command(command_id, normalized, proposed)
            except ValueError as exc:
                if not prior: running.release()
                raise ApiProblem(409, "command_conflict", str(exc))
            except Exception:
                if not prior: running.release()
                raise
            if prior:
                return operation
            def work():
                try:
                    operation.update(status="running", started_at=now())
                    registry.put("operation", operation["operation_id"], operation, replace=True)
                    result = preflight(root=root, device=device)
                    operation.update(status="completed", result=result, finished_at=now())
                except Exception as exc:
                    operation.update(status="failed", error={"code": "preflight_failed", "message": str(exc)}, finished_at=now())
                finally:
                    registry.put("operation", operation["operation_id"], operation, replace=True)
                    running.release()
            acknowledgement = dict(proposed)
            try:
                pool.submit(work)
            except RuntimeError:
                operation.update(status="failed", error={"code": "service_stopping", "message": "The service is stopping."}, finished_at=now())
                registry.put("operation", operation["operation_id"], operation, replace=True)
                running.release()
                raise ApiProblem(503, "service_stopping", "The service is stopping.")
            return acknowledgement

    @app.get("/api/v1/operations/{identity}")
    def operation(identity: str):
        result = registry.get("operation", identity)
        if not result: raise ApiProblem(404, "operation_not_found", "No such operation.")
        return result

    @app.get("/api/v1/architectures")
    def architectures():
        return items("architecture")

    @app.get("/api/v1/architectures/{identity}")
    def architecture(identity: str):
        item = registry.get("architecture", identity)
        if not item: raise ApiProblem(404, "architecture_not_found", "No such architecture.")
        return item

    @app.post("/api/v1/architectures/validate")
    def validate(graph: dict):
        return {"schema_version": SCHEMA, **validate_graph(graph, catalog)}

    @app.post("/api/v1/architectures", status_code=201)
    def register(graph: dict):
        command_id = graph.get("command_id")
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 128:
            raise ApiProblem(422, "invalid_command", "Architecture registration requires a unique command_id.")
        graph = {key: value for key, value in graph.items() if key != "command_id"}
        validation = validate_graph(graph, catalog)
        if not validation["valid"]:
            raise ApiProblem(422, "invalid_architecture", "Architecture does not match integrated component contracts.", validation)
        name, version = graph.get("name"), graph.get("version")
        if not isinstance(name, str) or not 1 <= len(name) <= 200 or not isinstance(version, str) or not 1 <= len(version) <= 64:
            raise ApiProblem(422, "invalid_identity", "Architecture needs a name and version.")
        saved = {**graph, "architecture_id": str(uuid.uuid4()), "schema_version": SCHEMA,
                 "execution_eligible": False, "created_at": now()}
        saved["architecture_hash"] = hashlib.sha256(canonical(graph).encode()).hexdigest()
        try:
            return registry.register_architecture(command_id, graph, saved)
        except ValueError as exc:
            raise ApiProblem(409, "command_conflict", str(exc))

    @app.get("/api/v1/datasets")
    def datasets():
        return items("dataset")

    @app.get("/api/v1/curricula")
    def curricula():
        return items("curriculum")

    @app.get("/api/v1/inventory")
    def inventory():
        path = root / "State/curriculum_inventory/legacy_raw_manifest.json"
        if not path.exists(): return {"schema_version": SCHEMA, "status": "not_available"}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"schema_version": SCHEMA, "status": "inventoried_not_training_ready", **{k: data.get(k) for k in ("generated_at", "file_count", "total_bytes", "audit_cross_check", "dedupe_summary")}}

    @app.get("/api/v1/runs")
    def runs(): return items("run")

    @app.get("/api/v1/checkpoints")
    def checkpoints(): return items("checkpoint")

    @app.get("/api/v1/backup/status")
    def backup():
        return {"schema_version": SCHEMA, "status": "not_verified", "configured_destination": None,
                "verified_artifact": None, "restore_drill": None,
                "reason": "No backup destination and completed restore drill have been registered."}

    @app.api_route("/api/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def not_integrated(path: str):
        raise ApiProblem(503, "not_integrated", "This training/runtime capability is not integrated yet.", {"path": path})

    frontend = Path(frontend_path) if frontend_path is not None else root / "lab/frontend"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
