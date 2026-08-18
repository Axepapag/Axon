"""Council runtime web server (FastAPI).

Serves the operator dashboard and exposes the REST + WebSocket surface
specified in runtime/council/CONTRACT.md:

    GET  /                 -> static/index.html
    GET  /api/status
    GET  /api/config       / POST /api/config
    POST /api/chat         {"text": str}
    POST /api/control      {"action": "start"|"stop"|"pause"|"resume"}
    GET  /api/cores
    WS   /ws               -> every engine event broadcast to all clients

The server constructs a CouncilEngine from engine.py (same directory) and
wires its event_sink into the WebSocket broadcast hub.

Run:  uvicorn server:app --host 127.0.0.1 --port 8788
(see start_council.bat)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "council_config.json"
INDEX_HTML = BASE_DIR / "static" / "index.html"

# engine.py is built in parallel; tolerate its absence so the dashboard and
# tests can still boot. Once engine.py exists this import resolves normally.
try:  # pragma: no cover - exercised implicitly
    from engine import CouncilEngine  # type: ignore
except ImportError:  # pragma: no cover
    CouncilEngine = None  # type: ignore


# --------------------------------------------------------------------------
# Event hub: engine event_sink -> asyncio queue -> all connected websockets
# --------------------------------------------------------------------------

class EventHub:
    """Fan-out of engine events to every connected WebSocket client.

    The engine calls ``sink`` (a plain sync callable) possibly from its own
    task/thread; the hub re-enters the server loop safely and a broadcaster
    task pushes events to all clients.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None

    def sink(self, event: dict) -> None:
        """Engine-facing event sink. Safe to call from any thread."""
        if not isinstance(event, dict):
            return
        event.setdefault("ts", time.time())
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._broadcaster())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._clients.clear()

    def add(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def discard(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def _broadcaster(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(event)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

def _validate_config(posted: dict, defaults: dict) -> dict:
    """Validate posted config keys/types against defaults. Returns cleaned dict."""
    if not isinstance(posted, dict):
        raise HTTPException(status_code=422, detail="config body must be a JSON object")
    cleaned: dict = {}
    unknown = sorted(k for k in posted if k not in defaults)
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown config keys: {unknown}")
    for key, value in posted.items():
        default = defaults[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise HTTPException(status_code=422, detail=f"{key} must be a boolean")
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(status_code=422, detail=f"{key} must be a number")
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise HTTPException(status_code=422, detail=f"{key} must be a string")
        elif isinstance(default, dict):  # regions
            if not isinstance(value, dict) or not all(
                isinstance(k, str) and isinstance(v, bool) for k, v in value.items()
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f"{key} must be an object of region-name -> boolean",
                )
        elif isinstance(default, list):  # advisors
            if not isinstance(value, list):
                raise HTTPException(status_code=422, detail=f"{key} must be a list")
            for i, adv in enumerate(value):
                cleaned[key] = cleaned.get(key, value)
                if not isinstance(adv, dict):
                    raise HTTPException(
                        status_code=422, detail=f"advisors[{i}] must be an object"
                    )
                for field, ftype in (
                    ("name", str),
                    ("endpoint", str),
                    ("api_key", str),
                    ("model", str),
                    ("enabled", bool),
                    ("temperature", (int, float)),
                ):
                    if field in adv and not isinstance(adv[field], ftype):
                        raise HTTPException(
                            status_code=422,
                            detail=f"advisors[{i}].{field} has wrong type",
                        )
        cleaned[key] = value
    return cleaned


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------

def create_app(engine=None, hub: Optional[EventHub] = None,
               config_path: Path = CONFIG_PATH) -> FastAPI:
    hub = hub or EventHub()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await hub.start()
        yield
        if engine is not None:
            try:
                if engine.status().get("running"):
                    await engine.stop()
            except Exception:
                pass
        await hub.stop()

    app = FastAPI(title="Axon Council Runtime", lifespan=lifespan)

    def require_engine():
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="council engine unavailable (engine.py not importable)",
            )
        return engine

    # ---- dashboard -------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def dashboard():
        return FileResponse(INDEX_HTML)

    # ---- REST ------------------------------------------------------------

    @app.get("/api/status")
    async def get_status():
        eng = require_engine()
        return eng.status()

    @app.get("/api/cores")
    async def get_cores():
        eng = require_engine()
        st = eng.status()
        return {
            "tick": st.get("tick", 0),
            "consolidator_core": st.get("consolidator_core"),
            "cores": st.get("cores", []),
        }

    @app.get("/api/config")
    async def get_config():
        eng = require_engine()
        return eng.status().get("config", {})

    @app.post("/api/config")
    async def post_config(payload: dict = Body(...)):
        eng = require_engine()
        defaults = eng.default_config()
        cleaned = _validate_config(payload, defaults)
        # Merge against the REAL config (status() masks advisor api_keys);
        # masked "...xxxx" placeholders posted back by the UI restore the
        # stored key rather than clobbering it.
        raw = eng.raw_config() if hasattr(eng, "raw_config") else eng.status().get("config", {})
        if "advisors" in cleaned:
            stored_by_name = {
                str(a.get("name")): str(a.get("api_key") or "")
                for a in (raw.get("advisors") or [])
                if isinstance(a, dict)
            }
            for adv in cleaned["advisors"]:
                key = str(adv.get("api_key") or "")
                if key.startswith("..."):
                    adv["api_key"] = stored_by_name.get(str(adv.get("name")), "")
        merged = {**defaults, **raw, **cleaned}
        changed = [k for k in cleaned if raw.get(k) != merged[k]]

        # validate -> save -> hot-apply (contract order)
        try:
            config_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"config save failed: {exc}")

        needs_restart = list(eng.apply_config(merged) or [])
        applied = [k for k in changed if k not in needs_restart]
        return {"applied": applied, "needs_restart": needs_restart}

    @app.post("/api/chat")
    async def post_chat(payload: dict = Body(...)):
        eng = require_engine()
        text = (payload or {}).get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text must be a non-empty string")
        await eng.submit_user_message(text.strip())
        return {"ok": True}

    # ---- Shared field & dormant state (Jeff's visibility + control) -------

    @app.get("/api/field")
    async def get_field():
        eng = require_engine()
        return eng.field_view()

    @app.post("/api/field/mask")
    async def post_field_mask(payload: dict = Body(...)):
        eng = require_engine()
        payload = payload or {}
        try:
            return eng.set_mask(
                str(payload.get("region", "")),
                mode=payload.get("mode"),
                offset=payload.get("offset"),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/field/region")
    async def post_field_region(payload: dict = Body(...)):
        eng = require_engine()
        payload = payload or {}
        try:
            return eng.set_region(
                str(payload.get("region", "")), payload.get("content", "")
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/control")
    async def post_control(payload: dict = Body(...)):
        eng = require_engine()
        action = (payload or {}).get("action", "")
        if action == "start":
            await eng.start()
        elif action == "stop":
            await eng.stop()
        elif action in ("pause", "resume"):
            fn = getattr(eng, action, None)
            if not callable(fn):
                raise HTTPException(
                    status_code=501,
                    detail=f"engine does not support '{action}'",
                )
            result = fn()
            if inspect.isawaitable(result):
                await result
        else:
            raise HTTPException(
                status_code=422,
                detail="action must be one of: start, stop, pause, resume",
            )
        return {"ok": True, "action": action, "status": eng.status()}

    # ---- WebSocket --------------------------------------------------------

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.add(websocket)
        # Fresh clients get an immediate status snapshot so they can render
        # before the next engine event arrives.
        if engine is not None:
            try:
                snapshot = {"type": "status", "ts": time.time(), **engine.status()}
                await websocket.send_json(snapshot)
            except Exception:
                pass
        try:
            while True:
                # Keep the connection alive; all traffic is server -> client.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            hub.discard(websocket)

    return app


# --------------------------------------------------------------------------
# Module-level app (uvicorn server:app)
# --------------------------------------------------------------------------

def _build_default_engine():
    if CouncilEngine is None:
        return None
    return CouncilEngine(config_path=CONFIG_PATH, event_sink=default_hub.sink)


default_hub = EventHub()
default_engine = _build_default_engine()
app = create_app(engine=default_engine, hub=default_hub, config_path=CONFIG_PATH)


def main() -> None:
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8788, reload=False)


if __name__ == "__main__":
    main()
