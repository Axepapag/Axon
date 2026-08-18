"""Tests for runtime/council/server.py against a FakeEngine.

Contract verification gates 1 and 3 (server side, no GPU, no real engine):

    gate 1: py_compile every new file            -> run via: python -m py_compile ...
    gate 3: uvicorn boot on 8788 with FakeEngine wired, GET / 200,
            POST /api/chat round-trip, WS events observed, config round-trip.

The FakeEngine implements the CONTRACT's documented CouncilEngine API:
    __init__(config_path, event_sink), default_config(), status(),
    start(), stop(), submit_user_message(text), apply_config(cfg)
plus optional pause()/resume() and emits canned tick_start / core_delta /
soul / consolidated / canonical / chat events.

Run (from D:\\Axon\\runtime\\council):
    python tests_council_server.py            # in-process tests + uvicorn boot test
Uvicorn boot target (used by this file itself):
    uvicorn tests_council_server:fake_app --host 127.0.0.1 --port 8788
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
import server  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
REGIONS = [
    "conversation_history", "user_input", "response_draft",
    "structured_knowledge", "situation_awareness", "scratch",
    "tool_results", "advisor_input", "task_state", "diary",
]
HOT_KEYS = {"tick_delay_ms", "stable_ticks", "max_ticks",
            "temperature_spread", "regions", "advisors", "log_path"}

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# FakeEngine — contract API, canned events
# ---------------------------------------------------------------------------

class FakeEngine:
    """Implements the CONTRACT CouncilEngine API; emits canned tick events."""

    def __init__(self, config_path: Path, event_sink, tick_delay: float = 0.02):
        self.config_path = Path(config_path)
        self._sink = event_sink
        self.tick_delay = tick_delay
        self.running = False
        self.paused = False
        self.tick = 0
        self.consolidator_core = 0
        self._task: asyncio.Task | None = None
        self._stable = 0
        self._turn_active = False
        if self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        else:
            self.config = self.default_config()
            self.config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        self.canonical_state = {r: "" for r in REGIONS}
        self._masks = {r: {"mode": "tail", "offset": 0} for r in REGIONS}
        self.cores = [
            {"id": i, "soul_norm": 1.0 + 0.1 * i, "last_delta": "", "last_conf": 0.0}
            for i in range(int(self.config["cores"]))
        ]

    @staticmethod
    def default_config() -> dict:
        return {
            "checkpoint": r"D:\Axon\runs\conversational_cpu_autopilot\ckpt_440500.pt",
            "device": "cpu",
            "cores": 3,
            "soul_noise": 0.01,
            "temperature_spread": 0.1,
            "tick_delay_ms": 50,
            "stable_ticks": 3,
            "max_ticks": 0,
            "regions": {r: True for r in REGIONS},
            "advisors": [],
            "log_path": "runtime/council/council.log",
        }

    def status(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "tick": self.tick,
            "consolidator_core": self.consolidator_core,
            "cores": [dict(c) for c in self.cores],
            "canonical_state": dict(self.canonical_state),
            "config": json.loads(json.dumps(self.config)),
        }

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.paused = False
        self._task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def pause(self) -> None:
        self.paused = True

    async def resume(self) -> None:
        self.paused = False

    async def submit_user_message(self, text: str) -> None:
        self.canonical_state["user_input"] = text
        self.canonical_state["response_draft"] = ""
        self._turn_active = True
        self._stable = 0
        hist = self.canonical_state["conversation_history"]
        self.canonical_state["conversation_history"] = (hist + f"\nUser: {text}").strip()
        self._emit({"type": "chat", "role": "user", "text": text})
        self._emit({"type": "canonical", "state": dict(self.canonical_state)})

    def apply_config(self, cfg: dict) -> list[str]:
        needs_restart = []
        for k, v in cfg.items():
            if self.config.get(k) != v and k not in HOT_KEYS:
                needs_restart.append(k)
            self.config[k] = v
        return sorted(needs_restart)

    # -- shared field / masks (contract parity with the real engine) --------

    def _fake_offset(self, region: str) -> int:
        mask = self._masks[region]
        content = self.canonical_state[region]
        if mask["mode"] == "tail":
            return max(0, len(content) - 256)
        return max(0, min(int(mask["offset"]), len(content)))

    def field_view(self) -> dict:
        regions = {}
        for region in REGIONS:
            content = self.canonical_state[region]
            offset = self._fake_offset(region)
            regions[region] = {
                "content": content,
                "dormant": content[:offset],
                "active": content[offset:],
                "mask_offset": offset,
                "mask_mode": self._masks[region]["mode"],
                "total_chars": len(content),
                "dormant_chars": offset,
                "active_chars": len(content) - offset,
                "visible": True,
            }
        return {"regions": regions, "active_region_chars": 256,
                "model_window_chars": 128}

    def set_mask(self, region: str, mode: str | None = None,
                 offset: int | None = None) -> dict:
        if region not in self._masks:
            raise ValueError(f"unknown region {region!r}")
        mask = self._masks[region]
        if mode is not None:
            if mode not in ("tail", "manual"):
                raise ValueError("mode must be 'tail' or 'manual'")
            mask["mode"] = mode
        if offset is not None:
            mask["mode"] = "manual"
            mask["offset"] = max(0, int(offset))
        return self.field_view()["regions"][region]

    def set_region(self, region: str, content: str) -> dict:
        if region not in self.canonical_state:
            raise ValueError(f"unknown region {region!r}")
        self.canonical_state[region] = str(content)
        return self.field_view()["regions"][region]

    # -- internals ----------------------------------------------------------

    def _emit(self, event: dict) -> None:
        event.setdefault("tick", self.tick)
        event.setdefault("ts", time.time())
        self._sink(event)

    async def _tick_loop(self) -> None:
        n = len(self.cores)
        while self.running:
            if self.paused:
                await asyncio.sleep(self.tick_delay)
                continue
            self.consolidator_core = self.tick % n
            self._emit({"type": "tick_start", "consolidator": self.consolidator_core})
            for phase in ("A", "B"):
                for c in self.cores:
                    self._emit({"type": "soul", "core": c["id"], "phase": "inhale",
                                "soul_norm": c["soul_norm"]})
                    text = f"core{c['id']} d{phase} t{self.tick}"
                    conf = round(0.5 + 0.1 * c["id"] + (0.05 if phase == "B" else 0), 3)
                    c["last_delta"], c["last_conf"] = text, conf
                    self._emit({"type": "core_delta", "core": c["id"],
                                "phase": phase, "text": text, "conf": conf})
                    c["soul_norm"] = round(c["soul_norm"] + 0.001, 6)
                    self._emit({"type": "soul", "core": c["id"], "phase": "exhale",
                                "soul_norm": c["soul_norm"]})
            # consolidation
            if self._turn_active:
                drafts = ["I", "I am", "I am doing well.", "I am doing well."]
                draft = drafts[min(self.tick, len(drafts) - 1)]
            else:
                draft = self.canonical_state["response_draft"]
            con = self.consolidator_core
            prev = self.canonical_state["response_draft"]
            self._stable = self._stable + 1 if draft == prev else 0
            self.canonical_state["response_draft"] = draft
            self._emit({"type": "consolidated", "core": con, "text": draft,
                        "conf": 0.9, "stable": self._stable})
            self._emit({"type": "canonical", "state": dict(self.canonical_state)})
            # stable halt: turn completes
            if (self._turn_active
                    and self._stable >= int(self.config["stable_ticks"])):
                hist = self.canonical_state["conversation_history"]
                self.canonical_state["conversation_history"] = (
                    hist + f"\nAxon: {draft}").strip()
                self.canonical_state["user_input"] = ""
                self._turn_active = False
                self._emit({"type": "chat", "role": "axon", "text": draft})
                self._emit({"type": "canonical", "state": dict(self.canonical_state)})
            self._emit({"type": "status", **self.status()})
            self.tick += 1
            max_ticks = int(self.config.get("max_ticks", 0))
            if max_ticks and self.tick >= max_ticks:
                self.running = False
                break
            await asyncio.sleep(self.tick_delay)


class NoPauseFakeEngine(FakeEngine):
    pause = None
    resume = None


def rotation_ok(consolidators: list[int], n: int = 3) -> bool:
    """Round-robin property: each tick's consolidator advances by 1 mod n,
    and all n cores have held the crown within the observed window."""
    if len(set(consolidators)) < n:
        return False
    return all(b == (a + 1) % n for a, b in zip(consolidators, consolidators[1:]))


# ---------------------------------------------------------------------------
# App instance for the uvicorn boot test (gate 3)
# ---------------------------------------------------------------------------

_tmpdir = Path(tempfile.mkdtemp(prefix="council_test_"))
TEST_CONFIG_PATH = _tmpdir / "council_config.json"
fake_hub = server.EventHub()
fake_engine = FakeEngine(TEST_CONFIG_PATH, fake_hub.sink)
fake_app = server.create_app(engine=fake_engine, hub=fake_hub,
                             config_path=TEST_CONFIG_PATH)


# ---------------------------------------------------------------------------
# In-process tests (fastapi TestClient)
# ---------------------------------------------------------------------------

def test_in_process() -> None:
    from fastapi.testclient import TestClient

    print("== in-process tests (TestClient) ==")
    with TestClient(fake_app) as client:
        # GET / serves dashboard
        r = client.get("/")
        report("GET / 200 + html", r.status_code == 200 and "<title>Axon Council" in r.text)

        # GET /api/status shape
        r = client.get("/api/status")
        st = r.json()
        ok = r.status_code == 200 and all(
            k in st for k in ("running", "paused", "tick", "consolidator_core",
                              "cores", "canonical_state", "config"))
        report("GET /api/status shape", ok, json.dumps({k: st.get(k) for k in ("running", "tick")}))

        # GET /api/cores
        r = client.get("/api/cores")
        body = r.json()
        report("GET /api/cores 3 cores",
               r.status_code == 200 and len(body["cores"]) == 3
               and body["consolidator_core"] is not None)

        # GET /api/config
        r = client.get("/api/config")
        cfg = r.json()
        report("GET /api/config keys",
               r.status_code == 200 and set(FakeEngine.default_config()) <= set(cfg))

        # POST /api/config hot key
        r = client.post("/api/config", json={"tick_delay_ms": 25})
        body = r.json()
        report("POST /api/config hot key applied",
               r.status_code == 200 and body["applied"] == ["tick_delay_ms"]
               and body["needs_restart"] == [], json.dumps(body))

        # POST /api/config restart key
        r = client.post("/api/config", json={"cores": 4})
        body = r.json()
        report("POST /api/config restart key flagged",
               r.status_code == 200 and body["needs_restart"] == ["cores"]
               and body["applied"] == [], json.dumps(body))
        # restore cores=3
        client.post("/api/config", json={"cores": 3})

        # config persisted to disk
        saved = json.loads(TEST_CONFIG_PATH.read_text(encoding="utf-8"))
        report("config saved to disk", saved.get("tick_delay_ms") == 25)

        # unknown key rejected
        r = client.post("/api/config", json={"bogus_key": 1})
        report("POST /api/config unknown key -> 422", r.status_code == 422)

        # wrong type rejected
        r = client.post("/api/config", json={"cores": "three"})
        report("POST /api/config wrong type -> 422", r.status_code == 422)

        # advisor CRUD round-trip
        advisors = [{"name": "gpt-x", "endpoint": "https://api.example.com/v1",
                     "api_key": "sk-test", "model": "m1", "enabled": True,
                     "temperature": 0.7}]
        r = client.post("/api/config", json={"advisors": advisors})
        got = client.get("/api/config").json().get("advisors")
        report("advisor CRUD round-trip", r.status_code == 200 and got == advisors)
        client.post("/api/config", json={"advisors": []})

        # POST /api/chat validation
        r = client.post("/api/chat", json={"text": ""})
        report("POST /api/chat empty -> 422", r.status_code == 422)

        # shared field: view, mask move, region edit, validation
        r = client.get("/api/field")
        fv = r.json()
        report("GET /api/field shape",
               r.status_code == 200
               and set(fv.get("regions", {})) == set(REGIONS)
               and fv.get("model_window_chars") == 128)
        r = client.post("/api/field/region",
                        json={"region": "scratch", "content": "x" * 300})
        report("POST /api/field/region edit",
               r.status_code == 200 and r.json()["total_chars"] == 300)
        r = client.post("/api/field/mask",
                        json={"region": "scratch", "offset": 100})
        m = r.json()
        report("POST /api/field/mask manual offset",
               r.status_code == 200 and m["mask_offset"] == 100
               and m["mask_mode"] == "manual"
               and m["dormant_chars"] == 100 and m["active_chars"] == 200)
        r = client.post("/api/field/mask",
                        json={"region": "scratch", "mode": "tail"})
        m = r.json()
        report("POST /api/field/mask follow tail",
               r.status_code == 200 and m["mask_mode"] == "tail"
               and m["mask_offset"] == 44)
        r = client.post("/api/field/mask", json={"region": "nope", "offset": 1})
        report("POST /api/field/mask unknown region -> 422", r.status_code == 422)
        client.post("/api/field/region", json={"region": "scratch", "content": ""})

        # control: start
        r = client.post("/api/control", json={"action": "start"})
        report("POST /api/control start",
               r.status_code == 200 and r.json()["status"]["running"] is True)

        # control: pause/resume
        r = client.post("/api/control", json={"action": "pause"})
        paused = r.json()["status"]["paused"]
        r = client.post("/api/control", json={"action": "resume"})
        report("pause/resume pass-through",
               paused is True and r.json()["status"]["paused"] is False)

        # control: bad action
        r = client.post("/api/control", json={"action": "explode"})
        report("POST /api/control bad action -> 422", r.status_code == 422)

        # WS: snapshot + full canned event flow + consolidator rotation
        with client.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            report("WS status snapshot on connect", first.get("type") == "status")

            r = client.post("/api/chat", json={"text": "How are you?"})
            report("POST /api/chat ok", r.status_code == 200 and r.json().get("ok"))

            seen_types: set[str] = set()
            consolidators: list[int] = []
            chat_roles: list[str] = []
            history_has_turn = False
            deadline = time.time() + 15
            while time.time() < deadline and not (
                    {"tick_start", "core_delta", "soul", "consolidated",
                     "canonical", "chat"} <= seen_types
                    and rotation_ok(consolidators) and chat_roles.count("axon") >= 1
                    and history_has_turn):
                ev = ws.receive_json()
                seen_types.add(ev.get("type"))
                if ev.get("type") == "tick_start":
                    consolidators.append(ev["consolidator"])
                if ev.get("type") == "chat":
                    chat_roles.append(ev["role"])
                if ev.get("type") == "canonical":
                    hist = ev["state"].get("conversation_history", "")
                    if "Axon: I am doing well." in hist and "User: How are you?" in hist:
                        history_has_turn = True

            report("WS event types observed",
                   {"tick_start", "core_delta", "soul", "consolidated",
                    "canonical", "chat"} <= seen_types,
                   ",".join(sorted(seen_types)))
            report("WS consolidator rotates round-robin",
                   rotation_ok(consolidators), str(consolidators[:6]))
            report("WS chat user+axon events", "user" in chat_roles and "axon" in chat_roles)
            report("turn landed in conversation_history", history_has_turn)

        # control: stop
        r = client.post("/api/control", json={"action": "stop"})
        report("POST /api/control stop",
               r.status_code == 200 and r.json()["status"]["running"] is False)

    # engine without pause/resume -> 501
    nop_hub = server.EventHub()
    nop_eng = NoPauseFakeEngine(_tmpdir / "council_config2.json", nop_hub.sink)
    nop_app = server.create_app(engine=nop_eng, hub=nop_hub,
                                config_path=_tmpdir / "council_config2.json")
    from fastapi.testclient import TestClient
    with TestClient(nop_app) as client:
        r = client.post("/api/control", json={"action": "pause"})
        report("pause unsupported -> 501", r.status_code == 501)


# ---------------------------------------------------------------------------
# Gate 3: real uvicorn boot on 8788 with the FakeEngine wired
# ---------------------------------------------------------------------------

def test_uvicorn_boot() -> None:
    import websockets

    print("== uvicorn boot test on 127.0.0.1:8788 (gate 3) ==")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests_council_server:fake_app",
         "--host", "127.0.0.1", "--port", "8788"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # wait for the port
        up = False
        for _ in range(60):
            try:
                r = httpx.get("http://127.0.0.1:8788/", timeout=1.0)
                if r.status_code == 200:
                    up = True
                    break
            except Exception:
                time.sleep(0.5)
        report("uvicorn boot, GET / 200", up)
        if not up:
            return

        # start engine, WS observe, chat round-trip
        r = httpx.post("http://127.0.0.1:8788/api/control",
                       json={"action": "start"}, timeout=5)
        report("boot: control start", r.status_code == 200)

        async def flow():
            async with websockets.connect("ws://127.0.0.1:8788/ws") as ws:  # noqa: F821
                first = json.loads(await asyncio.wait_for(ws.recv(), 5))
                types = {first.get("type")}
                consolidators, chat_roles = [], []
                # chat round-trip while engine ticks
                rr = httpx.post("http://127.0.0.1:8788/api/chat",
                                json={"text": "How are you?"}, timeout=5)
                assert rr.status_code == 200, rr.text
                end = time.time() + 15
                while time.time() < end:
                    ev = json.loads(await asyncio.wait_for(ws.recv(), max(0.1, end - time.time())))
                    types.add(ev.get("type"))
                    if ev.get("type") == "tick_start":
                        consolidators.append(ev["consolidator"])
                    if ev.get("type") == "chat":
                        chat_roles.append(ev["role"])
                    if ({"tick_start", "core_delta", "soul", "consolidated",
                         "canonical", "chat"} <= types
                            and len(set(consolidators)) >= 3
                            and "axon" in chat_roles):
                        break
                return types, consolidators, chat_roles

        types, consolidators, chat_roles = asyncio.run(flow())
        want = {"tick_start", "core_delta", "soul", "consolidated", "canonical", "chat"}
        report("boot: WS events observed over the wire", want <= types,
               ",".join(sorted(types)))
        report("boot: chat round-trip (user+axon)",
               "user" in chat_roles and "axon" in chat_roles, str(chat_roles))
        report("boot: consolidator rotation", rotation_ok(consolidators),
               str(consolidators[:6]))

        # config round-trip over the wire
        r = httpx.get("http://127.0.0.1:8788/api/config", timeout=5)
        cfg = r.json()
        cfg["stable_ticks"] = 4
        r = httpx.post("http://127.0.0.1:8788/api/config", json=cfg, timeout=5)
        body = r.json()
        r2 = httpx.get("http://127.0.0.1:8788/api/config", timeout=5)
        report("boot: config round-trip",
               r.status_code == 200 and "stable_ticks" in body["applied"]
               and r2.json()["stable_ticks"] == 4, json.dumps(body))

        httpx.post("http://127.0.0.1:8788/api/control",
                   json={"action": "stop"}, timeout=5)
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        tail = "\n".join((out or "").splitlines()[-5:])
        print("  uvicorn log tail:\n    " + tail.replace("\n", "\n    "))


def main() -> int:
    test_in_process()
    test_uvicorn_boot()
    failed = [r for r in _results if r[0] == FAIL]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
