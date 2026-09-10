"""Axon live organ demo server — drives the production runtime for presentations.

Everything on screen comes from real organ APIs on an isolated demo state root:
  - Heart valve ingress (user valve -> durable spool -> final gate -> canonical)
  - Heartbeat ticks with dormant recall (dormant valve -> CORTEX surfacing)
  - Two-barrier reasoning circulation with fixture cores (scripted Q&A answers
    committed through the consolidator path -- explicitly NOT learned cores)
  - Heart-owned durable region mask sliders (view changes, body never moves)
  - D64 field compiler: 16D cells packed 4-per-64-lane row, exact roundtrip

The demo state root lives under State/tmp/organ_demo (gitignored) with its own
small demo-authored dormant corpus, so no private memory is ever displayed and
the real State/ root is never touched (separate single-writer lease).

Usage (from D:/Axon, Python 3.12 full install):
    PYTHONUTF8=1 python scripts/demo_organ_server.py            # port 9201
    PYTHONUTF8=1 python scripts/demo_organ_server.py --rebuild  # wipe demo root
Open http://127.0.0.1:9201
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from runtime.dormant import DormantEvidenceIndex  # noqa: E402
from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot  # noqa: E402
from runtime.heart import (  # noqa: E402
    BeatConfig,
    CategoricalTextFrame,
    CoreDescriptor,
    CoreRegistry,
    HeartHost,
    HeartHostConfig,
    ReasoningDecision,
    ReasoningEmission,
    ReasoningOperationEmission,
    ReasoningOperationKind,
    ReasoningPassRequest,
    ReasoningPassResult,
)
from runtime.soul import SoulLayer, SoulTemperature, SoulTransition  # noqa: E402

DEMO_STATE_ROOT = REPO_ROOT / "State" / "tmp" / "organ_demo"
PORT = 9201

# Regions the UI exposes sliders for (identity region is unmaskable by law).
SLIDER_REGIONS = [
    LogicalRegion.CONVERSATION_HISTORY,
    LogicalRegion.CORTEX,
    LogicalRegion.RESPONSE_DRAFT,
    LogicalRegion.USER_INPUT,
]

# ---------------------------------------------------------------------------
# Demo dormant corpus (demo-authored facts; provenance strings are honest:
# these records were written FOR this demo, by scripts/demo_organ_server.py).
# ---------------------------------------------------------------------------

_CORPUS = [
    {
        "container_id": "c-demo-sky-color",
        "kind": "fact",
        "text": "The color of the sky is blue",
        "normalized_text": "the color of the sky is blue",
        "letters": "The color of the sky is blue",
        "edges": [
            {
                "edge_type": "has_color",
                "target": "blue",
                "confidence": 0.99,
                "provenance": "organ-demo:edge:sky-blue",
                "status": "dormant",
            }
        ],
        "source": "organ-demo:facts:1",
        "provenance": "organ-demo:container:sky-color",
        "confidence": 0.99,
        "status": "dormant",
        "metadata": {"topic": "sky"},
    },
    {
        "container_id": "c-demo-sky-scatter",
        "kind": "fact",
        "text": "The sky appears blue because air scatters shorter wavelengths of light more than longer ones",
        "normalized_text": "the sky appears blue because air scatters shorter wavelengths of light more than longer ones",
        "letters": "The sky appears blue because air scatters shorter wavelengths of light more than longer ones",
        "edges": [],
        "source": "organ-demo:facts:2",
        "provenance": "organ-demo:container:sky-scatter",
        "confidence": 0.9,
        "status": "dormant",
        "metadata": {"topic": "sky"},
    },
    {
        "container_id": "c-demo-axon-definition",
        "kind": "concept",
        "text": "Axon is a stateful AI runtime that remembers exactly",
        "normalized_text": "axon is a stateful ai runtime that remembers exactly",
        "letters": "Axon is a stateful AI runtime that remembers exactly",
        "edges": [
            {
                "edge_type": "has_property",
                "target": "stateful",
                "confidence": 0.95,
                "provenance": "organ-demo:edge:axon-stateful",
                "status": "dormant",
            }
        ],
        "source": "organ-demo:concepts:1",
        "provenance": "organ-demo:container:axon-definition",
        "confidence": 0.95,
        "status": "dormant",
        "metadata": {},
    },
    {
        "container_id": "c-demo-heart-writer",
        "kind": "concept",
        "text": "The Heart is the sole canonical writer of the shared field",
        "normalized_text": "the heart is the sole canonical writer of the shared field",
        "letters": "The Heart is the sole canonical writer of the shared field",
        "edges": [],
        "source": "organ-demo:concepts:2",
        "provenance": "organ-demo:container:heart-writer",
        "confidence": 0.9,
        "status": "dormant",
        "metadata": {},
    },
    {
        "container_id": "c-demo-cell16",
        "kind": "concept",
        "text": "Axon packs one character into one frozen 16 dimensional transport cell",
        "normalized_text": "axon packs one character into one frozen 16 dimensional transport cell",
        "letters": "Axon packs one character into one frozen 16 dimensional transport cell",
        "edges": [],
        "source": "organ-demo:concepts:3",
        "provenance": "organ-demo:container:cell16",
        "confidence": 0.9,
        "status": "dormant",
        "metadata": {},
    },
    {
        "container_id": "c-demo-lane64",
        "kind": "concept",
        "text": "Axon packs four 16 dimensional character cells into one 64 lane rail row",
        "normalized_text": "axon packs four 16 dimensional character cells into one 64 lane rail row",
        "letters": "Axon packs four 16 dimensional character cells into one 64 lane rail row",
        "edges": [
            {
                "edge_type": "packs_into",
                "target": "rail",
                "confidence": 0.9,
                "provenance": "organ-demo:edge:lane64-rail",
                "status": "dormant",
            }
        ],
        "source": "organ-demo:concepts:4",
        "provenance": "organ-demo:container:lane64",
        "confidence": 0.9,
        "status": "dormant",
        "metadata": {},
    },
    {
        "container_id": "c-demo-checkpoint",
        "kind": "fact",
        "text": "The candidate paused at step 120 at an exact accepted checkpoint and is renewable",
        "normalized_text": "the candidate paused at step 120 at an exact accepted checkpoint and is renewable",
        "letters": "The candidate paused at step 120 at an exact accepted checkpoint and is renewable",
        "edges": [],
        "source": "organ-demo:facts:3",
        "provenance": "organ-demo:container:checkpoint",
        "confidence": 0.9,
        "status": "dormant",
        "metadata": {},
    },
    {
        "container_id": "c-demo-sky-episode",
        "kind": "episode",
        "text": "Asked what color the sky is, the answer given was blue",
        "normalized_text": "asked what color the sky is the answer given was blue",
        "letters": "Asked what color the sky is, the answer given was blue",
        "edges": [],
        "source": "organ-demo:episodes:1",
        "provenance": "organ-demo:container:sky-episode",
        "confidence": 0.85,
        "status": "dormant",
        "metadata": {},
    },
]

_SEMANTIC_EDGES = [
    {
        "source_container_id": "c-demo-sky-color",
        "source_text": "The color of the sky is blue",
        "edge_type": "has_color",
        "target": "blue",
        "provenance": "organ-demo:edge:sky-blue:standalone",
        "confidence": 0.99,
        "status": "dormant",
    },
    {
        "source_container_id": "c-demo-lane64",
        "source_text": "Axon packs four 16 dimensional character cells into one 64 lane rail row",
        "edge_type": "packs_into",
        "target": "rail",
        "provenance": "organ-demo:edge:lane64-rail:standalone",
        "confidence": 0.9,
        "status": "dormant",
    },
]

_QA = [
    (("sky", "color"), "The sky is blue."),
    (("sky",), "The sky is blue."),
    (("axon",), "Axon is a stateful AI runtime that remembers exactly."),
    (("who", "writes"), "The Heart is the sole canonical writer."),
    (("heart",), "The Heart is the sole canonical writer of the shared field."),
    (("checkpoint",), "The candidate paused at step 120 — an exact accepted checkpoint, renewable."),
    (("step", "120"), "The candidate paused at step 120 — an exact accepted checkpoint, renewable."),
    (("64",), "Four 16D character cells pack into one 64-lane rail row."),
    (("16",), "One character packs into one frozen 16D transport cell."),
]


def scripted_answer(user_text: str) -> str | None:
    text = user_text.lower()
    for needles, answer in _QA:
        if all(needle in text for needle in needles):
            return answer
    return None


# ---------------------------------------------------------------------------
# Fixture reasoning cores (the circulation-test pattern: deterministic ports
# through the REAL two-barrier governance; explicitly not learned models).
# ---------------------------------------------------------------------------

class DemoFixtureCore:
    """Scripted consolidator: answers predetermined questions via real DELTA ops."""

    def __init__(self, core_id: str) -> None:
        self.core_id = core_id

    @staticmethod
    def _result(request: ReasoningPassRequest, emission: ReasoningEmission) -> ReasoningPassResult:
        hot = SoulLayer(
            SoulTemperature.HOT,
            f"{request.descriptor.core_id}:{request.phase}:{request.soul.generation + 1}".encode(),
            tensor_layout="fixture-hot-v1",
        )
        return ReasoningPassResult(
            emission=emission,
            soul_transition=SoulTransition(
                core_id=request.descriptor.core_id,
                architecture_id=request.descriptor.architecture_id,
                parameter_generation=request.descriptor.parameter_generation,
                before_soul_id=request.soul.soul_id,
                before_generation=request.soul.generation,
                tick_uid=request.image.identity.tick_uid,
                request_id=request.request_id,
                phase=request.phase,
                updates=(hot,),
            ),
        )

    def _emission(self, request: ReasoningPassRequest, decision: ReasoningDecision,
                  operations=(), detail: str | None = None) -> ReasoningEmission:
        return ReasoningEmission(
            base_field_id=request.image.identity.base_field_id,
            base_tick_id=request.image.identity.base_tick_id,
            author_core_id=request.descriptor.core_id,
            pass_id=request.phase,
            rail_d_model=request.descriptor.d_model,
            decision=decision,
            operations=tuple(operations),
            detail=None if not detail else CategoricalTextFrame.from_text(
                detail, d_model=request.descriptor.d_model
            ),
        )

    def emit(self, request: ReasoningPassRequest) -> ReasoningPassResult:
        if request.phase == "consolidated":
            user_text = request.rail.exact_surface.region_text(
                LogicalRegion.USER_INPUT.value
            ).strip()
            answer = scripted_answer(user_text)
            if answer is None:
                # The consolidator must commit a delta; unscripted input gets an
                # honest fixture placeholder, never a fabricated answer.
                answer = "[fixture core] no scripted answer is registered for this input"
            current = request.rail.exact_surface.region_text(
                LogicalRegion.RESPONSE_DRAFT.value
            )
            return self._result(
                request,
                self._emission(
                    request,
                    ReasoningDecision.DELTA,
                    operations=(
                        ReasoningOperationEmission(
                            kind=ReasoningOperationKind.REPLACE,
                            region=LogicalRegion.RESPONSE_DRAFT,
                            start=0,
                            end=len(current),
                            payload=CategoricalTextFrame.from_text(
                                answer, d_model=request.descriptor.d_model
                            ),
                        ),
                    ),
                ),
            )
        return self._result(
            request,
            self._emission(request, ReasoningDecision.NO_OP, detail="witness core observing"),
        )


# ---------------------------------------------------------------------------
# Demo state root construction
# ---------------------------------------------------------------------------

def _jsonl_line(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def build_demo_root(rebuild: bool) -> Path:
    state_root = DEMO_STATE_ROOT
    dormant = state_root / "dormant"
    index_path = dormant / ".derived" / "evidence_v1" / "index.sqlite3"
    if rebuild and state_root.exists():
        shutil.rmtree(state_root)
    if index_path.exists():
        return state_root

    dormant.mkdir(parents=True, exist_ok=True)
    with (dormant / "containers.jsonl").open("wb") as handle:
        for record in _CORPUS:
            handle.write(_jsonl_line(record))
    with (dormant / "semantic_edges.jsonl").open("wb") as handle:
        for edge in _SEMANTIC_EDGES:
            handle.write(_jsonl_line(edge))
    for name in ("kg_cache_50k.jsonl", "layout_groups.jsonl", "symbol_registry.jsonl"):
        (dormant / name).write_bytes(b"")

    import hashlib

    manifest = {
        "kind": "axon_recovered_corpus_manifest",
        "version": 1,
        "source_files": [
            {
                "path": "organ-demo-authored-facts",
                "size_bytes": 0,
                "sha256": hashlib.sha256(b"organ-demo-authored-facts").hexdigest(),
                "included": True,
            }
        ],
        "output_counts": {
            "containers": len(_CORPUS),
            "semantic_edges": len(_SEMANTIC_EDGES),
        },
    }
    (dormant / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

    index = DormantEvidenceIndex.build(state_root)
    index.close()
    return state_root


# ---------------------------------------------------------------------------
# Server core
# ---------------------------------------------------------------------------

class DemoRuntime:
    def __init__(self, rebuild: bool) -> None:
        self.state_root = build_demo_root(rebuild)
        # All organ operations run on ONE dedicated worker thread: sqlite
        # objects (dormant index, heart state) are thread-affine, and the demo
        # must never touch them from HTTP handler threads.
        self._queue: "queue.Queue[tuple[str, dict, object]]" = queue.Queue()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._worker = threading.Thread(target=self._worker_main, name="organ-demo-runtime", daemon=True)
        self._worker.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("organ demo runtime did not start")
        if self._error is not None:
            raise RuntimeError(f"organ demo runtime failed to start: {self._error!r}")

    def _worker_main(self) -> None:
        try:
            registry = CoreRegistry(
                (
                    CoreDescriptor(core_id="demo-consolidator", d_model=64),
                    CoreDescriptor(core_id="demo-witness", d_model=64),
                )
            )
            self.host = HeartHost(
                state_root=self.state_root,
                beat_config=BeatConfig(),
                host_config=HeartHostConfig(idle_interval_seconds=3600.0),
                core_registry=registry,
                reasoning_ports=(DemoFixtureCore("demo-consolidator"), DemoFixtureCore("demo-witness")),
            )
            self.host.start()
            self.compiler = D64FieldCompiler()
            self.beat_log: list[dict] = []
        except Exception as exc:  # noqa: BLE001
            self._error = exc
        finally:
            self._ready.set()
        while True:
            item = self._queue.get()
            if item is None:
                break
            name, kwargs, result_box = item
            try:
                if self._error is not None:
                    raise self._error
                result_box["value"] = getattr(self, "_op_" + name)(**kwargs)
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc
            finally:
                result_box["done"].set()

    def _call(self, name: str, **kwargs):
        if not self._ready.wait(timeout=60):
            raise RuntimeError("organ demo runtime did not start")
        if self._error is not None and name != "shutdown":
            raise RuntimeError(f"organ demo runtime failed to start: {self._error!r}")
        box: dict = {"done": threading.Event()}
        self._queue.put((name, kwargs, box))
        if not box["done"].wait(timeout=300):
            raise TimeoutError(f"organ op {name} timed out")
        if "error" in box:
            raise box["error"]
        return box["value"]

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=10)
        self.host.stop()

    # -- helpers ------------------------------------------------------------

    def _mask_policies(self) -> dict:
        return self.host.region_mask_state().policies

    def _compile_view(self, field: SharedFieldSnapshot):
        return self.compiler.compile(field, region_masks=self._mask_policies())

    def _region_payload(self, field: SharedFieldSnapshot, compiled) -> list[dict]:
        out = []
        for region in LogicalRegion:
            state = field.region(region)
            if state is None:
                continue
            policy = self._mask_policies().get(region)
            attended = compiled.region_text(region)
            canonical = state.text
            if region.value == "identity":
                slider = None
            elif policy is not None and policy.kind == "tail_percent":
                slider = policy.limit
            else:
                slider = 100
            out.append({
                "region": region.value,
                "canonical": canonical,
                "attended": attended,
                "slider": slider,
                "masked_canonical": canonical != attended,
                "span_count": len(state.spans),
            })
        return out

    def _rail_payload(self, field: SharedFieldSnapshot, compiled) -> dict:
        rows = []
        max_rows = min(compiled.row_count, 40)
        for row in range(max_rows):
            lanes = []
            for lane in range(4):
                addr = compiled.address(row, lane)
                if addr is None:
                    lanes.append(None)
                    continue
                cell = compiled.lane_cell16(row, lane)
                lanes.append({
                    "char": addr.character,
                    "region": addr.region.value,
                    "token": int(addr.transport_token_id),
                    "kind": addr.transport_kind,
                    "unit": int(addr.transport_unit_index),
                    "unit_count": int(addr.transport_unit_count),
                    "vec": [round(float(v), 4) for v in cell],
                })
            rows.append(lanes)
        return {
            "rail_id": compiled.rail_id,
            "row_count": compiled.row_count,
            "shown_rows": max_rows,
            "coverage_complete": compiled.coverage.complete,
            "rows": rows,
        }

    def _cortex_hits(self, field: SharedFieldSnapshot) -> list[dict]:
        hits = []
        state = field.region(LogicalRegion.CORTEX)
        if state is None:
            return hits
        for span in state.spans:
            provenance = span.provenance
            try:
                prov = json.loads(provenance)
            except (TypeError, ValueError):
                prov = {"provenance": provenance}
            hits.append({
                "span_id": span.span_id,
                "text": span.text,
                "kind": span.kind,
                "source": span.source,
                "container_refs": list(span.container_refs or ()),
                "provenance": prov,
            })
        return hits

    # -- API operations (worker thread only) ---------------------------------

    def _op_state(self) -> dict:
        field = self.host.coordinator.current_field
        compiled = self._compile_view(field)
        return {
            "field_id": field.field_id,
            "tick_id": field.tick_id,
            "heartbeats": len(self.beat_log),
            "regions": self._region_payload(field, compiled),
            "rail": self._rail_payload(field, compiled),
            "cortex_hits": self._cortex_hits(field),
            "health": self.host.health(),
        }

    def _op_ingress(self, text: str) -> dict:
        decision = self.host.submit_user(text, provenance="organ-demo-ui")
        if not decision.admitted:
            return {"admitted": False, "reason": decision.reason}
        beat = self.host.heartbeat()
        return self._beat_payload(beat, ingress=text)

    def _op_beat(self) -> dict:
        beat = self.host.heartbeat()
        return self._beat_payload(beat, ingress=None)

    def _op_mask(self, region: str, percent: int) -> dict:
        logical = LogicalRegion(region)
        self.host.set_region_unmasked_percent(logical, percent)
        field = self.host.coordinator.current_field
        compiled = self._compile_view(field)
        return {
            "region": region,
            "percent": percent,
            "field_id": field.field_id,
            "regions": self._region_payload(field, compiled),
            "rail": self._rail_payload(field, compiled),
        }

    def _op_roundtrip(self) -> dict:
        field = self.host.coordinator.current_field
        # Roundtrip is a claim about the CANONICAL BODY, so verify against a
        # full-body compile (no region masks). The attention view is a lens;
        # masked characters remain dormant in place and still roundtrip.
        canonical = self.compiler.compile(field)
        canonical.assert_fresh(field)
        canonical.verify_roundtrip(field)
        view = self._compile_view(field)
        regions = []
        for region in LogicalRegion:
            state = field.region(region)
            if state is None or not state.text:
                continue
            regions.append({
                "region": region.value,
                "chars": len(state.text),
                "decoded": canonical.region_text(region),
                "exact": canonical.region_text(region) == state.text,
            })
        return {
            "roundtrip_exact": True,
            "field_id": field.field_id,
            "rows": canonical.row_count,
            "lanes": canonical.row_count * 64,
            "valid_lanes": int(canonical.lane_valid.sum()),
            "view_rows": view.row_count,
            "masked_active": canonical.row_count != view.row_count,
            "regions": regions,
        }

    def _beat_payload(self, beat, ingress: str | None) -> dict:
        field = beat.field
        compiled = self._compile_view(field)
        reasoning = None
        if beat.reasoning_result is not None:
            result = beat.reasoning_result
            reasoning = {
                "consolidator_core_id": result.consolidator_core_id,
                "first_pass": [
                    {"core_id": rec.core_id, "state": rec.state.value}
                    for rec in result.first_records
                ],
                "refined_pass": [
                    {"core_id": rec.core_id, "state": rec.state.value}
                    for rec in result.refined_records
                ],
                "response": field.region(LogicalRegion.RESPONSE_DRAFT).text,
            }
        payload = {
            "admitted": True,
            "ingress": ingress,
            "heartbeat_sequence": beat.heartbeat_sequence,
            "state": beat.state.value,
            "field_id": field.field_id,
            "tick_id": field.tick_id,
            "commits": [
                {"commit_id": commit.commit_id} for commit in beat.commits
            ],
            "tick_view": (None if beat.tick_image is None
                          else beat.tick_image.view_id),
            "reasoning": reasoning,
            "regions": self._region_payload(field, compiled),
            "rail": self._rail_payload(field, compiled),
            "cortex_hits": self._cortex_hits(field),
        }
        self.beat_log.append({
            "heartbeat_sequence": payload["heartbeat_sequence"],
            "state": payload["state"],
            "ingress": ingress,
            "commits": payload["commits"],
            "reasoning": reasoning is not None,
            "cortex_hits": [hit["container_refs"] for hit in payload["cortex_hits"]],
        })
        return payload


RUNTIME: DemoRuntime | None = None


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            try:
                self._json(200, RUNTIME._call("state"))
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": repr(exc)})
        elif self.path == "/api/roundtrip":
            try:
                self._json(200, RUNTIME._call("roundtrip"))
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": repr(exc)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        try:
            if self.path == "/api/ingress":
                self._json(200, RUNTIME._call("ingress", text=str(req.get("text", ""))[:512]))
            elif self.path == "/api/beat":
                self._json(200, RUNTIME._call("beat"))
            elif self.path == "/api/mask":
                self._json(200, RUNTIME._call("mask", region=str(req.get("region")), percent=int(req.get("percent"))))
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})


# ---------------------------------------------------------------------------
# UI (single page, dark, gliksbot palette)
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Axon Organ Demo — Live Runtime</title>
<style>
:root{
  --bg:#06131c; --bg2:#0b1e2c; --ink:#eef1fa; --muted:#a3adc2; --faint:#5f6d80;
  --accent:#37d6ff; --violet:#9b6bff; --amber:#ffb547; --teal:#00c2b8;
  --border:rgba(163,173,194,.18); --mono:ui-monospace,Consolas,monospace;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:14px/1.45 'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:18px;padding:10px 18px;border-bottom:1px solid var(--border);background:var(--bg2)}
header .logo{font-weight:700;letter-spacing:.18em;color:var(--accent)}
header .fid{font-family:var(--mono);font-size:12px;color:var(--muted)}
header .chip{font-family:var(--mono);font-size:12px;border:1px solid var(--border);border-radius:8px;padding:3px 10px;color:var(--muted)}
header .chip.ok{color:var(--teal);border-color:rgba(0,194,184,.4)}
header .chip.beats{color:var(--accent)}
main{flex:1;display:grid;grid-template-columns:340px 1fr 380px;min-height:0}
.col{overflow:auto;padding:14px;border-right:1px solid var(--border)}
.col:last-child{border-right:none}
h2{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--faint);margin:6px 0 10px}
.region{border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-bottom:10px;background:var(--bg2)}
.region .rname{display:flex;justify-content:space-between;align-items:center;font-family:var(--mono);font-size:12px;color:var(--accent);margin-bottom:6px}
.region .rname .spans{color:var(--faint)}
.region pre{font-family:var(--mono);font-size:12px;white-space:pre-wrap;word-break:break-word;color:var(--ink);min-height:16px}
.region pre .masked{color:var(--faint)}
.region .slider-row{display:flex;align-items:center;gap:8px;margin-top:8px}
.region .slider-row input[type=range]{flex:1;accent-color:var(--accent)}
.region .slider-row .pct{font-family:var(--mono);font-size:12px;color:var(--muted);width:42px;text-align:right}
.region .warn{color:var(--amber);font-size:11px;margin-top:4px;font-family:var(--mono)}
.railbar{display:flex;align-items:center;gap:10px;margin-bottom:5px}
.railbar .rlabel{font-family:var(--mono);font-size:10px;color:var(--faint);width:52px;text-align:right;flex:none}
.railbar .chars{font-family:var(--mono);font-size:13px;color:var(--accent);width:74px;flex:none;overflow:hidden;white-space:nowrap}
.lanes{display:flex;gap:3px;flex:1}
.lane{display:flex;flex-direction:column;gap:2px;align-items:center}
.lane .glyph{font-family:var(--mono);font-size:12px;color:var(--ink);height:16px}
.lane canvas{border:1px solid rgba(55,214,255,.25);border-radius:2px;cursor:pointer}
.lane.byte canvas{border-color:rgba(155,107,255,.5)}
.lane.pad canvas{border-style:dashed;opacity:.35;cursor:default}
.railmeta{font-family:var(--mono);font-size:11px;color:var(--faint);margin:8px 0}
#inspector{border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-top:10px;background:var(--bg2);font-family:var(--mono);font-size:12px}
#inspector .bars{display:flex;gap:2px;align-items:flex-end;height:44px;margin-top:8px}
#inspector .bars i{flex:1;background:var(--accent);opacity:.8;border-radius:1px}
.legend{display:flex;gap:14px;font-size:11px;color:var(--muted);margin:6px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.log{font-family:var(--mono);font-size:11.5px}
.beat{border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin-bottom:8px;background:var(--bg2)}
.beat .bh{color:var(--accent);display:flex;justify-content:space-between}
.beat .bl{color:var(--muted);margin-top:4px;word-break:break-all}
.hit{border-left:2px solid var(--teal);padding:6px 8px;margin:6px 0;background:rgba(0,194,184,.05);border-radius:0 6px 6px 0}
.hit .src{color:var(--faint);font-size:10.5px;word-break:break-all}
.hit .txt{color:var(--ink)}
footer{border-top:1px solid var(--border);padding:10px 18px;display:flex;gap:10px;align-items:center;background:var(--bg2)}
footer input[type=text]{flex:1;background:#030d14;border:1px solid var(--border);border-radius:8px;color:var(--ink);padding:9px 12px;font:14px 'Segoe UI',system-ui}
button{background:rgba(55,214,255,.12);border:1px solid rgba(55,214,255,.45);color:var(--accent);border-radius:8px;padding:9px 14px;font:600 13px 'Segoe UI',system-ui;cursor:pointer}
button:hover{background:rgba(55,214,255,.22)}
button.sec{background:transparent;border-color:var(--border);color:var(--muted)}
#status{font-family:var(--mono);font-size:11px;color:var(--faint);margin-left:8px;min-width:180px}
.try{font-size:11px;color:var(--faint);margin-top:8px;font-family:var(--mono)}
.truth{font-size:11px;color:var(--amber);border:1px solid rgba(255,181,71,.3);border-radius:8px;padding:6px 10px;margin-bottom:10px;background:rgba(255,181,71,.05)}
</style></head><body>
<header>
  <span class="logo">AXON · LIVE ORGANS</span>
  <span class="chip beats" id="beats">beats 0</span>
  <span class="chip" id="tick">tick —</span>
  <span class="chip" id="fid">field —</span>
  <span class="chip ok" id="rt">roundtrip —</span>
  <span class="chip" style="margin-left:auto">fixture cores · not learned · not serving</span>
</header>
<main>
  <div class="col">
    <h2>Shared Field · regions + attention sliders</h2>
    <div class="truth">Sliders move the VIEW only — masked characters go dormant in place; the canonical body and field_id never change.</div>
    <div id="regions"></div>
    <div class="try">try: slide conversation_history to 50 and watch the rail view shrink while the body stays.</div>
  </div>
  <div class="col">
    <h2>D64 Rail · 4 × 16D cells per 64-lane row</h2>
    <div class="legend"><span><i style="background:rgba(55,214,255,.45)"></i>native 16D char cell</span><span><i style="background:rgba(155,107,255,.45)"></i>UTF-8 byte cell</span><span><i style="background:rgba(55,214,255,.08)"></i>padding lane</span></div>
    <div id="rail"></div>
    <div class="railmeta" id="railmeta"></div>
    <div id="inspector">click any lane to inspect its 16D cell</div>
  </div>
  <div class="col">
    <h2>Heart · beats, commits, reasoning, dormant recall</h2>
    <div id="log"><div class="beat"><div class="bl">no beats yet — send ingress or press beat</div></div></div>
  </div>
</main>
<footer>
  <input type="text" id="say" placeholder="type ingress — e.g. what color is the sky?" autocomplete="off">
  <button id="send">valve → heart</button>
  <button class="sec" id="beat">beat</button>
  <button class="sec" id="roundtrip">roundtrip check</button>
  <span id="status"></span>
</footer>
<div class="try" style="padding:0 18px 8px">try: “what color is the sky?” · “what is axon?” · “who writes the field?” · “what happened at step 120?”</div>
<script>
const $=s=>document.querySelector(s);
let STATE=null;
function api(path,body){return fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined}).then(r=>r.json())}
function esc(t){return (t||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function drawRegions(regions){
  $('#regions').innerHTML=regions.map(r=>{
    const showSlider=r.slider!==null;
    const body=r.masked_canonical
      ?`<pre><span class="masked">[${r.attended}]</span></pre>`
      :`<pre>${esc(r.attended)||'<span style="color:var(--faint)">—</span>'}</pre>`;
    return `<div class="region"><div class="rname"><span>${r.region}</span><span class="spans">${r.span_count} spans</span></div>${body}
      ${showSlider?`<div class="slider-row"><input type="range" min="0" max="100" value="${r.slider}" data-region="${r.region}"><span class="pct">${r.slider}%</span></div>`:''}
      ${r.masked_canonical?'<div class="warn">view differs from canonical body — body untouched</div>':''}</div>`;
  }).join('');
  document.querySelectorAll('input[type=range]').forEach(el=>{
    el.oninput=()=>{el.parentElement.querySelector('.pct').textContent=el.value+'%'};
    el.onchange=async()=>{const p=await api('/api/mask',{region:el.dataset.region,percent:+el.value});apply(p);flash('mask '+el.dataset.region+' → '+el.value+'%')};
  });
}
function drawRail(rail){
  const railEl=$('#rail');railEl.innerHTML='';
  for(let r=0;r<rail.rows.length;r++){
    const bar=document.createElement('div');bar.className='railbar';
    const chars=(rail.rows[r]||[]).map(l=>l?l.char:'·').join('');
    bar.innerHTML=`<span class="rlabel">row ${r}</span><span class="chars" title="${esc(chars)}">${esc(chars)}</span>`;
    const lanesEl=document.createElement('div');lanesEl.className='lanes';
    (rail.rows[r]||[]).forEach((lane,li)=>{
      const laneEl=document.createElement('div');laneEl.className='lane';
      if(!lane){laneEl.classList.add('pad');
        laneEl.innerHTML=`<span class="glyph">&nbsp;</span>`;
        const c=document.createElement('canvas');c.width=16;c.height=16;c.title='padding lane (no character)';c.style.width='32px';c.style.height='32px';
        const g=c.getContext('2d');g.fillStyle='rgba(55,214,255,.05)';g.fillRect(0,0,16,16);
        laneEl.appendChild(c);lanesEl.appendChild(laneEl);return;}
      if(lane.kind!=='native_16d_cell')laneEl.classList.add('byte');
      const glyph=document.createElement('span');glyph.className='glyph';
      glyph.textContent=lane.char==='\n'?'⏎':lane.char;glyph.title=JSON.stringify(lane.char);
      const c=document.createElement('canvas');c.width=16;c.height=16;c.style.width='32px';c.style.height='32px';
      const g=c.getContext('2d');
      const mx=Math.max(...lane.vec.map(Math.abs),1e-6);
      for(let y=0;y<4;y++)for(let x=0;x<4;x++){
        const v=lane.vec[y*4+x]||0;const a=Math.abs(v)/mx;
        g.fillStyle=v>=0?`rgba(55,214,255,${(a*.85+.05).toFixed(2)})`:`rgba(255,181,71,${(a*.85+.05).toFixed(2)})`;
        g.fillRect(x*4,y*4,4,4);
      }
      c.title=`${JSON.stringify(lane.char)} · ${lane.region} · token ${lane.token} · ${lane.kind}`;
      c.onclick=()=>inspect(lane,r,li);
      laneEl.appendChild(glyph);laneEl.appendChild(c);lanesEl.appendChild(laneEl);
    });
    bar.appendChild(lanesEl);railEl.appendChild(bar);
  }
  $('#railmeta').textContent=`rail ${rail.rail_id.slice(0,18)}… · ${rail.row_count} rows · 4 × 16D cells per row · coverage complete: ${rail.coverage_complete}`;
}
function inspect(lane,row,laneIdx){
  const vals=lane.vec.map(Math.abs), mx=Math.max(...vals,1e-6);
  $('#inspector').innerHTML=`<b style="color:var(--accent)">row ${row} · lane ${laneIdx}</b> — char ${JSON.stringify(lane.char)} · ${lane.region}<br>
    token ${lane.token} · ${lane.kind} · unit ${lane.unit}/${lane.unit_count}<br>16D cell (cyan + / amber −):
    <div class="bars">${lane.vec.map(v=>`<i style="height:${Math.max(4,Math.abs(v)/mx*100)}%;background:${v>=0?'var(--accent)':'var(--amber)'}"></i>`).join('')}</div>
    <span style="color:var(--faint)">[${lane.vec.map(v=>v.toFixed(2)).join(', ')}]</span>`;
}
function drawLog(){
  $('#beats').textContent='beats '+STATE.heartbeats;
  $('#tick').textContent='tick '+STATE.tick_id;
  $('#fid').textContent='field '+STATE.field_id.slice(0,14)+'…';
}
function apply(p){ if(p.error){flash('ERROR '+p.error);return}
  STATE=p; drawRegions(p.regions); drawRail(p.rail); drawLog();
  if(p.cortex_hits&&p.cortex_hits.length) drawHits(p.cortex_hits);
  if(p.reasoning) drawBeat(p);
}
function drawHits(hits){
  const el=$('#log');const div=document.createElement('div');
  div.innerHTML=`<div class="beat"><div class="bh">dormant recall surfaced</div>${hits.map(h=>`<div class="hit"><div class="txt">${esc(h.text)}</div><div class="src">${esc(h.kind)} · ${esc(h.source)} · containers ${esc((h.container_refs||[]).join(','))}</div></div>`).join('')}</div>`;
  el.prepend(div);
}
function drawBeat(p){
  const el=$('#log');const div=document.createElement('div');
  const parts=[];
  parts.push(`ingress: ${esc(p.ingress)}`);
  parts.push(`state ${p.state} · commits ${p.commits.length}${p.tick_view?' · view '+p.tick_view.slice(0,10)+'…':''}`);
  if(p.reasoning) parts.push(`${p.reasoning.consolidator_core_id} → “${esc(p.reasoning.response)}”`);
  div.className='beat';
  div.innerHTML=`<div class="bh"><span>beat ${p.heartbeat_sequence}</span><span>tick ${p.tick_id}</span></div><div class="bl">${parts.join('<br>')}</div>`;
  el.prepend(div);
}
function flash(t){$('#status').textContent=t;setTimeout(()=>{if($('#status').textContent===t)$('#status').textContent=''},4000)}
$('#send').onclick=async()=>{const t=$('#say').value.trim();if(!t)return;$('#say').value='';flash('valve admitting…');apply(await api('/api/ingress',{text:t}));flash('committed')};
$('#say').onkeydown=e=>{if(e.key==='Enter')$('#send').onclick()};
$('#beat').onclick=async()=>{flash('heartbeat…');apply(await api('/api/beat'));flash('beat done')};
$('#roundtrip').onclick=async()=>{const r=await api('/api/roundtrip');if(r.error){flash('ROUNDTRIP FAILED '+r.error);return}
  $('#rt').textContent=`roundtrip exact · ${r.valid_lanes}/${r.lanes} lanes`;flash('roundtrip exact: True — '+r.rows+' rows, all regions decode exactly')};
(async()=>{apply(await api('/api/state'))})();
</script></body></html>"""


def main() -> int:
    global RUNTIME
    parser = argparse.ArgumentParser(description="Axon live organ demo server")
    parser.add_argument("--rebuild", action="store_true", help="wipe and rebuild the demo state root")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    RUNTIME = DemoRuntime(rebuild=args.rebuild)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    print(f"organ demo state root: {RUNTIME.state_root}")
    print(f"organ demo live: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        RUNTIME.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())