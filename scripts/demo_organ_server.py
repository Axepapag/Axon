"""Axon live organ demo server — the REAL runtime, on the REAL state.

Presents the production organs in a browser:
  - Real state root (default D:/Axon/State) with its real dormant corpus
    (59,875 recovered records, 4.2 GB derived index) and real canonical branch.
  - Heart valve ingress (user valve -> durable spool -> final gate -> canonical
    user_input commit -> autobiography deposit).
  - A heartbeat on every field change AND on every attention-mask move
    (mask dirtiness freezes a new tick view; the canonical body never moves).
  - Primitive dormant recall (the cortex valve): user input queries the real
    dormant index; relevant records surface into CORTEX with provenance.
  - NO reasoning cores are registered: response_draft stays canonically empty
    (the UI shows a "Reasoning Cores Coming Soon" ghost), user_input
    accumulates, and every tick closes as a null tick. Nothing is scripted.
  - The D64 rail is rendered in full — no row caps — grouped by region, with
    per-region visual collapse (presentation only; cores attend everything).

Usage (from D:/Axon, Python 3.12 full install):
    PYTHONUTF8=1 python scripts/demo_organ_server.py
    PYTHONUTF8=1 python scripts/demo_organ_server.py --state-root D:/Axon/State
    PYTHONUTF8=1 python scripts/demo_organ_server.py --demo   # isolated demo root
Open http://127.0.0.1:9201
"""

from __future__ import annotations

import argparse
import hashlib
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
from runtime.dormant.relevance import DormantRelevanceAuditor, DormantRelevancePolicy  # noqa: E402
from runtime.field import (  # noqa: E402
    D64FieldCompiler,
    DeleteText,
    FieldDelta,
    FieldSpan,
    InsertText,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    RegionVisibility,
    SharedFieldSnapshot,
    canonical_sha256,
    replacement_delta,
)
from runtime.heart import (  # noqa: E402
    BeatConfig,
    HeartHost,
    HeartHostConfig,
    TickIdentity,
)
from runtime.heart.authority import AuthorityGrant  # noqa: E402

DEFAULT_REAL_STATE_ROOT = REPO_ROOT / "State"
DEMO_STATE_ROOT = REPO_ROOT / "State" / "tmp" / "organ_demo"
PORT = 9201

LANES_PER_ROW = 4  # D64_LANES_PER_ROW: four 16D character cells per 64-wide row

# Every region gets an attention slider except identity (unmaskable by law).
SLIDER_REGIONS = [region for region in LogicalRegion if region.value != "identity"]


# ---------------------------------------------------------------------------
# Demo corpus (used ONLY with --demo; the real state uses its own corpus)
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
    records = [
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
            "container_id": "c-demo-axon-definition",
            "kind": "concept",
            "text": "Axon is a stateful AI runtime that remembers exactly",
            "normalized_text": "axon is a stateful ai runtime that remembers exactly",
            "letters": "Axon is a stateful AI runtime that remembers exactly",
            "edges": [],
            "source": "organ-demo:concepts:1",
            "provenance": "organ-demo:container:axon-definition",
            "confidence": 0.95,
            "status": "dormant",
            "metadata": {},
        },
        {
            "container_id": "c-demo-jeff",
            "kind": "concept",
            "text": "Jeff Gliksman is the founder of DexterGliksbot and the convener of Axon",
            "normalized_text": "jeff gliksman is the founder of dextergliksbot and the convener of axon",
            "letters": "Jeff Gliksman is the founder of DexterGliksbot and the convener of Axon",
            "edges": [],
            "source": "organ-demo:concepts:2",
            "provenance": "organ-demo:container:jeff",
            "confidence": 0.95,
            "status": "dormant",
            "metadata": {},
        },
    ]
    with (dormant / "containers.jsonl").open("wb") as handle:
        for record in records:
            handle.write(_jsonl_line(record))
    with (dormant / "semantic_edges.jsonl").open("wb") as handle:
        handle.write(_jsonl_line({
            "source_container_id": "c-demo-sky-color",
            "source_text": "The color of the sky is blue",
            "edge_type": "has_color",
            "target": "blue",
            "provenance": "organ-demo:edge:sky-blue:standalone",
            "confidence": 0.99,
            "status": "dormant",
        }))
    for name in ("kg_cache_50k.jsonl", "layout_groups.jsonl", "symbol_registry.jsonl"):
        (dormant / name).write_bytes(b"")
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
        "output_counts": {"containers": len(records), "semantic_edges": 1},
    }
    (dormant / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    index = DormantEvidenceIndex.build(state_root)
    index.close()
    return state_root


# ---------------------------------------------------------------------------
# Server core — all organ ops on one worker thread (sqlite is thread-affine)
# ---------------------------------------------------------------------------

class DemoRuntime:
    def __init__(self, state_root: Path, is_demo: bool) -> None:
        self.state_root = state_root
        self.is_demo = is_demo
        self.last_view_id: str | None = None
        self.last_tick_sequence: int | None = None
        self.beat_log: list[dict] = []
        # Cortex engine config (operator-controlled via /api/cortex-config):
        # the cortex is a bounded working surface fed by semantic edges from
        # every OTHER region, newest-first, deduped, char-budgeted.
        self.cortex_budget_chars = 1600
        self.cortex_auto = True
        self.cortex_cadence_s = 5.0
        self.last_cortex_tick: dict | None = None
        self.cortex_ticks_fired = 0
        self._last_seen_field_id: str | None = None
        self._stop_ticker = threading.Event()
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._worker = threading.Thread(target=self._worker_main, name="organ-demo-runtime", daemon=True)
        self._worker.start()
        if not self._ready.wait(timeout=600):
            raise RuntimeError("organ demo runtime did not start")
        if self._error is not None:
            raise RuntimeError(f"organ demo runtime failed to start: {self._error!r}")
        self._ticker = threading.Thread(target=self._auto_ticker, name="cortex-auto-tick", daemon=True)
        self._ticker.start()

    def _worker_main(self) -> None:
        try:
            self.host = HeartHost(
                state_root=self.state_root,
                # The cortex belongs to the cortex engine in this demo: the
                # per-beat primitive recall is disabled (0 items) so the
                # cortex only changes through cortex ticks — which still
                # commit through the Heart boundary under DORMANT_VALVE
                # authority. Null ticks auto-close; nothing scripted.
                beat_config=BeatConfig(recall_items_per_materialization=0),
                host_config=HeartHostConfig(idle_interval_seconds=3600.0),
                # No core registry, no reasoning ports: zero cores. Null ticks
                # auto-close; user_input accumulates; response_draft stays empty.
            )
            self.host.start()
            self.compiler = D64FieldCompiler()
            # Warm the dormant bridge now (verifies the corpus binding once) so
            # the first on-stage recall is fast.
            self.host.coordinator._ensure_bridge()
        except Exception as exc:  # noqa: BLE001
            self._error = exc
        finally:
            self._ready.set()
        while True:
            item = self._queue.get()
            if item is None:
                break
            name, kwargs, box = item
            try:
                box["value"] = getattr(self, "_op_" + name)(**kwargs)
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                box["done"].set()

    def _call(self, name: str, **kwargs):
        box: dict = {"done": threading.Event()}
        self._queue.put((name, kwargs, box))
        if not box["done"].wait(timeout=900):
            raise TimeoutError(f"organ op {name} timed out")
        if "error" in box:
            raise box["error"]
        return box["value"]

    # -- helpers (worker thread only) ----------------------------------------

    def _mask_policies(self) -> dict:
        return self.host.region_mask_state().policies

    def _compile_view(self, field: SharedFieldSnapshot):
        return self.compiler.compile(field, region_masks=self._mask_policies())

    def _region_payload(self, field: SharedFieldSnapshot, compiled) -> list[dict]:
        policies = self._mask_policies()
        out = []
        for region in LogicalRegion:
            state = field.region(region)
            if state is None:
                continue
            policy = policies.get(region)
            attended = compiled.region_text(region)
            canonical = state.text
            if region.value in ("identity", "cortex", "user_input"):
                slider = None
            elif policy is not None and policy.kind == "tail_percent":
                slider = policy.limit
            elif region.value == "conversation_history":
                # The history slider is expressed in percent but masks by
                # turns; the UI shows the resolved turn label. Turn spans
                # are durable-marked by their source (the delta applier
                # rebuilds spans as delta_insert, so span.kind is lost).
                turns = [s for s in state.spans if s.source == "heart-turn-rotation"]
                if policy is None or policy.kind == "all":
                    slider = 100
                elif policy.kind == "none":
                    slider = 0
                elif policy.kind == "last_n_spans":
                    slider = round(100 * policy.limit / len(turns)) if turns else 100
                else:
                    slider = 100
            else:
                slider = 100
            out.append({
                "region": region.value,
                "canonical": canonical,
                "attended": attended,
                "slider": slider,
                "masked_canonical": canonical != attended,
                "span_count": len(state.spans),
                "turn_count": (
                    len([s for s in state.spans if s.kind == "conversation_turn"])
                    if region.value == "conversation_history" else None
                ),
                "mask_label": (
                    ({"all": "all turns", "none": "0 turns"}.get(
                        policy.kind,
                        (f"newest {policy.limit} turns" if policy.kind == "last_n_spans" else None),
                    ) if policy is not None else None)
                    if region.value == "conversation_history" else None
                ),
            })
        return out

    def _grouped_rail(self, compiled) -> dict:
        """Full rail — no row caps — grouped by region of each row's first lane."""
        groups: dict[str, list] = {}
        order: list[str] = []
        valid_lanes = 0
        for row in range(compiled.row_count):
            lanes = []
            row_region = None
            for lane in range(LANES_PER_ROW):
                addr = compiled.address(row, lane)
                if addr is None:
                    lanes.append(None)
                    continue
                valid_lanes += 1
                if row_region is None:
                    row_region = addr.region.value
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
            if row_region is None:
                row_region = "padding"
            if row_region not in groups:
                groups[row_region] = []
                order.append(row_region)
            groups[row_region].append({"row": row, "lanes": lanes})
        return {
            "rail_id": compiled.rail_id,
            "row_count": compiled.row_count,
            "valid_lanes": valid_lanes,
            "coverage_complete": compiled.coverage.complete,
            "order": order,
            "groups": groups,
        }

    def _cortex_hits(self, field: SharedFieldSnapshot) -> list[dict]:
        hits = []
        state = field.region(LogicalRegion.CORTEX)
        if state is None:
            return hits
        for span in state.spans:
            try:
                prov = json.loads(span.provenance)
            except (TypeError, ValueError):
                prov = {"provenance": span.provenance}
            hits.append({
                "span_id": span.span_id,
                "text": span.text,
                "kind": span.kind,
                "source": span.source,
                "container_refs": list(span.container_refs or ()),
                "provenance": prov,
            })
        return hits

    def _beat_payload(self, beat, ingress: str | None) -> dict:
        field = beat.field
        compiled = self._compile_view(field)
        if beat.tick_image is not None:
            self.last_view_id = beat.tick_image.view_id
            self.last_tick_sequence = beat.tick_image.identity.tick_sequence
        payload = {
            "admitted": True,
            "ingress": ingress,
            "heartbeat_sequence": beat.heartbeat_sequence,
            "state": beat.state.value,
            "deferred_event_id": beat.deferred_event_id,
            "field_id": field.field_id,
            "tick_id": field.tick_id,
            "view_id": self.last_view_id,
            "commits": [{"commit_id": c.commit_id} for c in beat.commits],
            "cortex_budget": self.cortex_budget_chars,
            "cortex_chars": len(field.region(LogicalRegion.CORTEX).text),
            "regions": self._region_payload(field, compiled),
            "rail": self._grouped_rail(compiled),
            "cortex_hits": self._cortex_hits(field),
        }
        self.beat_log.append({
            "n": beat.heartbeat_sequence,
            "state": payload["state"],
            "ingress": ingress,
            "commits": len(beat.commits),
            "hits": len(payload["cortex_hits"]),
        })
        return payload

    # -- API operations (worker thread) --------------------------------------

    def _op_state(self) -> dict:
        field = self.host.coordinator.current_field
        compiled = self._compile_view(field)
        return {
            "field_id": field.field_id,
            "tick_id": field.tick_id,
            "view_id": self.last_view_id,
            "beats": len(self.beat_log),
            "cortex_budget": self.cortex_budget_chars,
            "cortex_chars": len(field.region(LogicalRegion.CORTEX).text),
            "last_cortex_tick": self.last_cortex_tick,
            "cortex_ticks_fired": self.cortex_ticks_fired,
            "regions": self._region_payload(field, compiled),
            "rail": self._grouped_rail(compiled),
            "cortex_hits": self._cortex_hits(field),
        }

    def _op_summary(self) -> dict:
        """Lightweight poll target: lets the UI notice changes without
        re-downloading the whole rail."""
        field = self.host.coordinator.current_field
        return {
            "field_id": field.field_id,
            "tick_id": field.tick_id,
            "view_id": self.last_view_id,
            "beats": len(self.beat_log),
            "cortex_budget": self.cortex_budget_chars,
            "cortex_chars": len(field.region(LogicalRegion.CORTEX).text),
            "cortex_ticks_fired": self.cortex_ticks_fired,
            "last_cortex_tick": self.last_cortex_tick,
            "cortex_top": field.region(LogicalRegion.CORTEX).text[:200],
            "rail_rows": self.compiler.compile(field).row_count,
            "server_time": __import__("datetime").datetime.now().strftime("%H:%M:%S"),
        }

    def _op_ingress(self, text: str) -> dict:
        # ---- Turn rotation (Jeff's design) --------------------------------
        # user_input holds ONLY the latest input. Before admitting new
        # ingress, the PREVIOUS user_input content rotates into
        # conversation_history as one discrete turn span ("Jeff: ..."),
        # through the Heart boundary under consolidator authority with a
        # real in-flight tick - the same path production turn finalization
        # uses. user = 1 turn; when Axon later responds, that response will
        # be its own turn.
        rotated = None
        field = self.host.coordinator.current_field
        prior = field.region(LogicalRegion.USER_INPUT).text.strip()
        if prior:
            history = field.region(LogicalRegion.CONVERSATION_HISTORY)
            turn_text = f'Jeff: "{prior}"\n'
            turn_span = FieldSpan(
                span_id=f"turn:{canonical_sha256({'text': prior})[:16]}",
                text=turn_text,
                kind="conversation_turn",
                source="organ-demo-turn-rotation",
                provenance=f"turn-rotation:jeff:{field.tick_id}",
            )
            rotated = RegionState(
                name=LogicalRegion.CONVERSATION_HISTORY,
                spans=tuple(history.spans) + (turn_span,),
                visibility=history.visibility,
                write_policy=history.write_policy,
            )
            user_state = field.region(LogicalRegion.USER_INPUT)
            delta = FieldDelta(
                base_field_id=field.field_id,
                base_tick_id=field.tick_id,
                author_core_id="heart-turn-rotation",
                pass_id="turn_rotation",
                operations=(
                    DeleteText(
                        region=LogicalRegion.USER_INPUT,
                        start=0,
                        end=len(user_state.text),
                        provenance="heart-turn-rotation:clear",
                    ),
                ),
            )
            identity = self.host.identity_store.next_tick()
            tick_identity = TickIdentity(
                tick_sequence=identity.tick_sequence,
                heartbeat_id=identity.heartbeat_sequence,
                base_field_id=field.field_id,
                base_tick_id=field.tick_id,
            )
            self.host.coordinator.freeze_tick(field, tick_identity)
            commit = self.host.coordinator.commit_consolidator_delta(
                delta, tick=tick_identity,
                metadata={"engine": "turn-rotation"},
            )
            field = commit.successor
            # Append the turn into history through a second consolidator
            # commit on a fresh tick (history is sealed to consolidator).
            history2 = field.region(LogicalRegion.CONVERSATION_HISTORY)
            delta2 = FieldDelta(
                base_field_id=field.field_id,
                base_tick_id=field.tick_id,
                author_core_id="heart-turn-rotation",
                pass_id="turn_rotation",
                operations=(
                    InsertText(
                        region=LogicalRegion.CONVERSATION_HISTORY,
                        offset=len(history2.text),
                        text=turn_text,
                        provenance="heart-turn-rotation:append",
                    ),
                ),
            )
            identity2 = self.host.identity_store.next_tick()
            tick2 = TickIdentity(
                tick_sequence=identity2.tick_sequence,
                heartbeat_id=identity2.heartbeat_sequence,
                base_field_id=field.field_id,
                base_tick_id=field.tick_id,
            )
            self.host.coordinator.freeze_tick(field, tick2)
            commit2 = self.host.coordinator.commit_consolidator_delta(
                delta2, tick=tick2,
                metadata={"engine": "turn-rotation"},
            )
            field = commit2.successor

        # ---- Normal valve ingress -----------------------------------------
        decision = self.host.submit_user(text, provenance="organ-demo-ui")
        if not decision.admitted:
            return {"admitted": False, "reason": decision.reason, "rotated": rotated is not None}
        beat = self.host.heartbeat()
        # The per-beat budget may defer a long item; keep beating like the
        # permanent host would until it is processed (bounded loop).
        attempts = 0
        while beat.deferred_event_id is not None and attempts < 8:
            beat = self.host.heartbeat()
            attempts += 1
        payload = self._beat_payload(beat, ingress=text)
        payload["rotated_turn"] = rotated is not None
        # The cortex ticks IMMEDIATELY on fresh ingress (its own cadence
        # continues in the background); surface the result in the same
        # response so the UI shows the cortex react to what you just said.
        try:
            payload["cortex_tick"] = self._op_cortex_tick(trigger="ingress")
        except Exception:  # noqa: BLE001
            payload["cortex_tick"] = {"ticked": False, "reason": "tick error"}
        return payload

    def _op_beat(self) -> dict:
        beat = self.host.heartbeat()
        return self._beat_payload(beat, ingress=None)

    def _op_mask(self, region: str, percent: int) -> dict:
        logical = LogicalRegion(region)
        if logical in (LogicalRegion.IDENTITY, LogicalRegion.CORTEX, LogicalRegion.USER_INPUT):
            return {
                "region": region,
                "percent": percent,
                "rejected": True,
                "reason": (
                    "user_input holds only the latest input and is never masked; "
                    "cortex is the engine surface; identity is unmaskable by law"
                ),
            }
        if logical is LogicalRegion.CONVERSATION_HISTORY:
            # Jeff's design: the history slider masks BY TURNS, not percent.
            # Turn spans are the ones the turn rotation committed (durable
            # marker: source == 'heart-turn-rotation'; the delta applier
            # rebuilds spans as delta_insert, so span.kind can't be used).
            field = self.host.coordinator.current_field
            turns = [
                span for span in field.region(LogicalRegion.CONVERSATION_HISTORY).spans
                if span.source == "heart-turn-rotation"
            ]
            total = len(turns)
            if percent >= 100:
                n_turns = total if total else 10_000  # "all"
                policy = RegionMaskPolicy("all", 0) if total else RegionMaskPolicy("none", 0)
            elif percent <= 0:
                n_turns = 0
                policy = RegionMaskPolicy("none", 0)
            else:
                n_turns = max(1, round(total * percent / 100)) if total else 1
                policy = RegionMaskPolicy("last_n_spans", n_turns)
            self.host.set_region_mask_policy(logical, policy)
            label = (
                f"all {total} turns" if percent >= 100
                else ("0 turns" if percent <= 0 else f"newest {n_turns} of {total} turns")
            )
        else:
            self.host.set_region_unmasked_percent(logical, percent)
            label = f"{percent}%"
        # Mask dirtiness means the next heartbeat freezes a NEW tick view:
        # the heart beats because attention changed (canonical body never does).
        beat = self.host.heartbeat()
        payload = self._beat_payload(beat, ingress=f"mask {region} -> {label}")
        payload["mask_label"] = label
        return payload

    # -- cortex engine (worker thread) ---------------------------------------

    def _auto_ticker(self) -> None:
        """Fire a cortex tick on any field change OR at the configured cadence."""
        import time

        last_fire = time.monotonic()
        while not self._stop_ticker.wait(0.5):
            if not self.cortex_auto:
                continue
            try:
                changed = self._call("cortex_field_changed")
            except Exception:  # noqa: BLE001
                changed = False
            due = (time.monotonic() - last_fire) >= max(self.cortex_cadence_s, 1.0)
            if changed or due:
                try:
                    result = self._call("cortex_tick", trigger="auto" if changed else "cadence")
                    self.cortex_ticks_fired += 1
                    if result.get("ticked"):
                        self.last_cortex_tick = result
                except Exception:  # noqa: BLE001
                    pass
                last_fire = time.monotonic()

    def _op_cortex_field_changed(self) -> bool:
        """Compare-only check (worker thread): has the field changed since the
        last cortex tick consumed it? The tick itself updates the marker —
        whether or not it commits — so a noise-gated null tick settles."""
        field = self.host.coordinator.current_field
        return field.field_id != self._last_seen_field_id

    def _cortex_regions(self) -> tuple[LogicalRegion, ...]:
        """Every region EXCEPT cortex and identity feeds the cortex."""
        return tuple(
            region for region in LogicalRegion
            if region not in (LogicalRegion.CORTEX, LogicalRegion.IDENTITY)
        )

    def _op_cortex_tick(self, trigger: str = "manual") -> dict:
        """One cortex tick: gather semantic edges from all other regions.

        For each non-cortex region, build a query from its attended text and
        retrieve from the real dormant index (edges included). Keep the newest
        records first, skip anything already present in the attended cortex
        (dedup by container id), trim to the char budget at span boundaries,
        and commit through the Heart boundary under DORMANT_VALVE authority.
        Evicted spans leave the CANONICAL cortex region but their records stay
        in the dormant state — the cortex is a working cache, not the archive.
        """
        field = self.host.coordinator.current_field
        bridge = self.host.coordinator._ensure_bridge()
        index = bridge.index
        budget = self.cortex_budget_chars

        # 1. Gather queries from every other region's ATTENDED text — the
        #    attention mask is real: masked characters are dormant and MUST
        #    NOT drive cortex queries (this was the noise bug: the engine
        #    read canonical text, so sliders changed the display but not
        #    what the cortex attended). Uses the production pattern
        #    (region_state.with_policy(policy).attended_text) — the same
        #    path the coordinator's _attended_text uses for rail compile.
        #    The query is the region's newest attended sentence.
        policies = self._mask_policies()
        queries: list[tuple[str, str]] = []
        for region in self._cortex_regions():
            state = field.region(region)
            policy = policies.get(region)
            attended = (
                state.with_policy(policy).attended_text
                if policy is not None
                else state.attended_text
            ).strip()
            if not attended:
                continue
            import re as _sentence_re
            sentences = [s.strip() for s in _sentence_re.split(r"[.!?\n]+", attended) if s.strip()]
            query = sentences[-1] if sentences else attended
            if len(query) > 240:
                query = query[-240:]
            queries.append((region.value, query))

        # 2. Retrieve candidates per region query, then score them through the
        #    production relevance auditor with a STRICT threshold and NO
        #    fallback: silence beats noise. Only genuinely relevant records
        #    surface; if nothing clears the bar, the cortex simply does not
        #    change this tick.
        seen_containers: set[str] = set()
        selected: list = []
        per_region_stats: list[dict] = []
        current_cortex_pre = field.region(LogicalRegion.CORTEX)
        auditor = DormantRelevanceAuditor(
            DormantRelevancePolicy(items_per_materialization=3, target_chars=budget, min_score=0.42)
        )
        # The auditor's novelty/overlap model must see what is ATTENDED, not
        # the canonical body: build a derived view where every non-cortex
        # region carries exactly its attended text. Cortex keeps its real
        # spans (the dedup refs and already-active scoring live there).
        attended_regions = []
        for region_state in field.regions:
            if region_state.name is LogicalRegion.CORTEX:
                attended_regions.append(region_state)
                continue
            policy = policies.get(region_state.name)
            attended = (
                region_state.with_policy(policy).attended_text
                if policy is not None
                else region_state.attended_text
            )
            attended_regions.append(
                RegionState(
                    name=region_state.name,
                    spans=(FieldSpan(
                        span_id=f"attended-view:{region_state.name.value}",
                        text=attended,
                    ),) if attended else (),
                    visibility=region_state.visibility,
                    write_policy=region_state.write_policy,
                )
            )
        attended_field = SharedFieldSnapshot(
            tick_id=field.tick_id,
            regions=tuple(attended_regions),
            source_manifest_ids=field.source_manifest_ids,
        )
        for region_name, query in queries:
            try:
                evidence = index.retrieve(query, limit=10, include_graph=True)
            except Exception:  # noqa: BLE001
                per_region_stats.append({"region": region_name, "error": "retrieve failed"})
                continue
            if not evidence:
                per_region_stats.append({"region": region_name, "hits": 0, "kept": 0})
                continue
            # A name-like query ("who is Jeff?") should pull the subject's own
            # records: retry with the strongest proper noun as the query.
            import re as _re
            tokens = _re.findall(r"[A-Za-z][A-Za-z']+", query)
            proper = [w for w in tokens if w[0].isupper() and w.lower() not in {
                "who", "what", "when", "where", "why", "how", "the", "a", "an",
                "is", "are", "was", "were", "tell", "show", "about",
            }]
            known_subjects = {"jeff", "jeffrey", "gliksman", "gliksbot", "dextergliksbot", "kimmy", "codex"}
            subject = None
            for w in proper:
                if w.lower() in known_subjects:
                    subject = w
                    break
            if subject is None and proper:
                subject = proper[-1]  # "who is Jeff?" -> "Jeff"
            if subject:
                try:
                    subject_hits = index.retrieve(subject, limit=6, include_graph=True)
                    evidence = tuple(subject_hits) + tuple(evidence)
                except Exception:  # noqa: BLE001
                    pass
            decision = auditor.select(query, evidence, attended_field)
            fresh = [
                item for item in decision.selected
                if item.container.container_id not in seen_containers
            ]
            kept = fresh[:2]
            for item in kept:
                seen_containers.add(item.container.container_id)
            per_region_stats.append({
                "region": region_name,
                "query_chars": len(query),
                "hits": len(evidence),
                "audited": len(decision.selected),
                "below_threshold": len(decision.below_threshold),
                "fallback_used": decision.fallback_used,
                "kept": len(kept),
            })
            selected.extend(kept)

        # 3. Char-budget trim at span boundaries (whole spans only).
        budget = self.cortex_budget_chars
        fresh_spans: list[tuple[FieldSpan, int]] = []
        for item in selected:
            text = item.container.text
            if not text:
                continue
            prov_payload = {
                "engine": "organ-demo-cortex-tick",
                "trigger": trigger,
                "index_id": index.index_id,
                "byte_offset": item.container.byte_offset,
                "byte_length": item.container.byte_length,
                "raw_sha256": item.container.raw_sha256,
                "text_sha256": item.container.text_sha256,
                "source": item.container.source,
                "record_provenance": item.container.provenance,
            }
            fresh_spans.append((
                FieldSpan(
                    span_id=f"cortex:{item.container.container_id}:{item.container.raw_sha256[:12]}",
                    text=text,
                    kind=f"cortex_{item.container.kind or 'evidence'}",
                    source=item.container.source,
                    provenance=json.dumps(prov_payload, sort_keys=True),
                    confidence=item.container.confidence,
                    container_refs=(item.container.container_id,),
                    edge_refs=tuple(edge.edge_id for edge in item.edges[:4]),
                ),
                len(text),
            ))
        final_spans_list: list[FieldSpan] = []
        used = 0
        for span, text_len in fresh_spans:
            if used + text_len > budget:
                break
            final_spans_list.append(span)
            used += text_len
        # Jeff's design: every tick WIPES the cortex and replaces it with the
        # latest hits. Duplicates (same container set as before) leave the
        # region untouched; zero matches wipes it EMPTY. The region is the
        # engine's focus surface, not an accumulating log - evicted records
        # remain in the dormant archive.
        final_spans = tuple(final_spans_list)
        previous_ids = sorted(
            ref for span in current_cortex_pre.spans for ref in (span.container_refs or ())
        )
        new_ids = sorted(
            ref for span in final_spans for ref in (span.container_refs or ())
        )
        wiped = bool(current_cortex_pre.spans) and not final_spans
        same_surface = (
            bool(current_cortex_pre.spans)
            and bool(final_spans)
            and previous_ids == new_ids
            and current_cortex_pre.text == "".join(s.text for s in final_spans)
        )

        if same_surface:
            self._last_seen_field_id = field.field_id  # null tick settles the change marker
            self.cortex_ticks_fired += 1
            self.last_cortex_tick = {
                "ticked": False,
                "trigger": trigger,
                "reason": "duplicate surface — identical hits, region left untouched",
                "per_region": per_region_stats,
            }
            return {
                "ticked": False,
                "reason": "duplicate surface — identical hits, region left untouched",
                "trigger": trigger,
                "per_region": per_region_stats,
            }
        if wiped:
            verdict = "zero matches — cortex wiped empty"
        else:
            verdict = "cortex replaced with latest hits"

        replacement = RegionState(
            name=LogicalRegion.CORTEX,
            spans=final_spans,
            visibility=RegionVisibility.ATTENDED,
        )
        compiled = self.compiler.compile(field)
        if wiped:
            # A full wipe is a DeleteText over the whole prior region text
            # (ReplaceText cannot be an empty no-op by delta law).
            prior_text = current_cortex_pre.text
            delta = FieldDelta(
                base_field_id=field.field_id,
                base_tick_id=field.tick_id,
                author_core_id="cortex-engine",
                pass_id="cortex_tick",
                operations=(
                    DeleteText(
                        region=LogicalRegion.CORTEX,
                        start=0,
                        end=len(prior_text),
                        provenance=f"cortex-engine-tick:{trigger}:wipe",
                    ),
                ),
            )
        else:
            delta = replacement_delta(
                field,
                compiled,
                region=LogicalRegion.CORTEX,
                text=replacement.text,
                author_core_id="cortex-engine",
                pass_id="cortex_tick",
                evidence=tuple(sorted({ref for s in final_spans for ref in s.container_refs})),
                provenance=f"cortex-engine-tick:{trigger}",
                container_refs=tuple(sorted({ref for s in final_spans for ref in s.container_refs})),
            )
        commit = self.host.coordinator.commit_heart_delta(
            field,
            delta,
            AuthorityGrant.dormant_valve(),
            valve_provenance={
                "valve_id": "dormant_recall",
                "valve_version": 1,
                "source_id": "dormant_valve",
                "item_id": canonical_sha256({"engine": "cortex-engine", "trigger": trigger, "base": field.field_id}),
                "provenance": f"cortex-engine:{trigger}",
                "authority_class": "dormant_valve",
                "governed_regions": [LogicalRegion.CORTEX.value],
                "engine": "cortex-engine",
                "per_region": per_region_stats,
            },
        )
        self._last_seen_field_id = commit.successor.field_id
        self.cortex_ticks_fired += 1
        self.last_cortex_tick = {
            "ticked": True,
            "trigger": trigger,
            "chars": len(replacement.text),
            "spans": len(final_spans),
            "budget": budget,
            "verdict": verdict,
            "wiped": wiped,
        }
        return {
            "ticked": True,
            "trigger": trigger,
            "chars": len(replacement.text),
            "spans": len(final_spans),
            "budget": budget,
            "verdict": verdict,
            "wiped": wiped,
            "per_region": per_region_stats,
            "field_id": commit.successor.field_id,
            "commit_id": commit.commit_id,
        }

    def _op_cortex_config(self, budget_chars: int, auto: bool, cadence_s: float) -> dict:
        self.cortex_budget_chars = max(200, min(int(budget_chars), 20000))
        self.cortex_auto = bool(auto)
        self.cortex_cadence_s = max(0.0, float(cadence_s))
        return {"budget_chars": self.cortex_budget_chars, "auto": self.cortex_auto,
                "cadence_s": self.cortex_cadence_s}

    def _op_roundtrip(self) -> dict:
        field = self.host.coordinator.current_field
        # Roundtrip is a claim about the CANONICAL BODY: verify against a
        # full-body compile (no masks). The attention view is a lens; masked
        # characters remain dormant in place and still roundtrip.
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
                "exact": canonical.region_text(region) == state.text,
            })
        return {
            "roundtrip_exact": True,
            "field_id": field.field_id,
            "rows": canonical.row_count,
            "lanes": canonical.row_count * LANES_PER_ROW,
            "valid_lanes": int(canonical.lane_valid.sum()),
            "view_rows": view.row_count,
            "masked_active": canonical.row_count != view.row_count,
            "regions": regions,
        }

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=30)
        self.host.stop()


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
        elif self.path == "/api/cortex":
            try:
                self._json(200, {
                    "config": {
                        "budget_chars": RUNTIME.cortex_budget_chars,
                        "auto": RUNTIME.cortex_auto,
                        "cadence_s": RUNTIME.cortex_cadence_s,
                    },
                    "last_tick": RUNTIME.last_cortex_tick,
                })
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": repr(exc)})
        elif self.path == "/api/summary":
            try:
                self._json(200, RUNTIME._call("summary"))
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
                self._json(200, RUNTIME._call("ingress", text=str(req.get("text", ""))))
            elif self.path == "/api/beat":
                self._json(200, RUNTIME._call("beat"))
            elif self.path == "/api/mask":
                self._json(200, RUNTIME._call("mask", region=str(req.get("region")), percent=int(req.get("percent"))))
            elif self.path == "/api/cortex-tick":
                self._json(200, RUNTIME._call("cortex_tick", trigger=str(req.get("trigger", "manual"))))
            elif self.path == "/api/cortex-config":
                self._json(200, RUNTIME._call(
                    "cortex_config",
                    budget_chars=int(req.get("budget_chars", RUNTIME.cortex_budget_chars)),
                    auto=bool(req.get("auto", RUNTIME.cortex_auto)),
                    cadence_s=float(req.get("cadence_s", RUNTIME.cortex_cadence_s)),
                ))
            else:
                self._json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Axon Organ Demo — Live Runtime (real state)</title>
<style>
:root{
  --bg:#06131c; --bg2:#0b1e2c; --ink:#eef1fa; --muted:#a3adc2; --faint:#5f6d80;
  --accent:#37d6ff; --violet:#9b6bff; --amber:#ffb547; --teal:#00c2b8;
  --border:rgba(163,173,194,.18); --mono:ui-monospace,Consolas,monospace;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:14px/1.45 'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:14px;padding:10px 18px;border-bottom:1px solid var(--border);background:var(--bg2);flex-wrap:wrap}
header .logo{font-weight:700;letter-spacing:.18em;color:var(--accent)}
header .chip{font-family:var(--mono);font-size:12px;border:1px solid var(--border);border-radius:8px;padding:3px 10px;color:var(--muted)}
header .chip.ok{color:var(--teal);border-color:rgba(0,194,184,.4)}
header .chip.beats{color:var(--accent)}
header .chip.real{color:var(--amber);border-color:rgba(255,181,71,.4)}
header .chip.right{margin-left:auto}
main{flex:1;display:grid;grid-template-columns:330px 1fr 360px;min-height:0}
.col{overflow:auto;padding:14px;border-right:1px solid var(--border)}
.col:last-child{border-right:none}
h2{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--faint);margin:6px 0 10px}
.region{border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-bottom:10px;background:var(--bg2)}
.region .rname{display:flex;justify-content:space-between;align-items:center;font-family:var(--mono);font-size:12px;color:var(--accent);margin-bottom:6px}
.region .rname .spans{color:var(--faint)}
.region pre{font-family:var(--mono);font-size:12px;white-space:pre-wrap;word-break:break-word;color:var(--ink);min-height:14px}
.region pre .masked{color:var(--faint)}
.region .ghost{color:var(--faint);font-style:italic;font-family:'Segoe UI',sans-serif;font-size:12px}
.region .slider-row{display:flex;align-items:center;gap:8px;margin-top:8px}
.region .slider-row input[type=range]{flex:1;accent-color:var(--accent)}
.region .slider-row .pct{font-family:var(--mono);font-size:12px;color:var(--muted);width:42px;text-align:right}
.region .warn{color:var(--amber);font-size:11px;margin-top:4px;font-family:var(--mono)}
/* rail */
.railgroup{margin-bottom:14px}
.railgroup summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:10px;padding:6px 10px;background:var(--bg2);border:1px solid var(--border);border-radius:8px;font-family:var(--mono);font-size:12px;color:var(--accent);user-select:none}
.railgroup summary .cnt{color:var(--faint)}
.railgroup summary .hide-note{margin-left:auto;color:var(--faint);font-size:10.5px}
.railgroup[open] summary{border-bottom-left-radius:0;border-bottom-right-radius:0}
.railgroup .rows{border:1px solid var(--border);border-top:none;border-radius:0 0 8px 8px;padding:8px 10px}
.railbar{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.railbar .rlabel{font-family:var(--mono);font-size:10px;color:var(--faint);width:52px;text-align:right;flex:none}
.railbar .chars{font-family:var(--mono);font-size:13px;color:var(--accent);width:72px;flex:none;overflow:hidden;white-space:nowrap}
.lanes{display:flex;gap:3px;flex:1;flex-wrap:wrap}
.lane{display:flex;flex-direction:column;gap:2px;align-items:center}
.lane .glyph{font-family:var(--mono);font-size:12px;color:var(--ink);height:16px}
.lane canvas{border:1px solid rgba(55,214,255,.25);border-radius:2px;cursor:pointer}
.lane.byte canvas{border-color:rgba(155,107,255,.5)}
.lane.pad canvas{border-style:dashed;opacity:.35;cursor:default}
.railmeta{font-family:var(--mono);font-size:11px;color:var(--faint);margin:8px 0}
#inspector{border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin-top:10px;background:var(--bg2);font-family:var(--mono);font-size:12px;position:sticky;top:0;z-index:5}
#inspector .bars{display:flex;gap:2px;align-items:flex-end;height:44px;margin-top:8px}
#inspector .bars i{flex:1;border-radius:1px}
.legend{display:flex;gap:14px;font-size:11px;color:var(--muted);margin:6px 0;flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.log{font-family:var(--mono);font-size:11.5px}
.beat{border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin-bottom:8px;background:var(--bg2)}
.beat .bh{color:var(--accent);display:flex;justify-content:space-between}
.beat .bl{color:var(--muted);margin-top:4px;word-break:break-word}
.hit{border-left:2px solid var(--teal);padding:6px 8px;margin:6px 0;background:rgba(0,194,184,.05);border-radius:0 6px 6px 0}
.hit .src{color:var(--faint);font-size:10.5px;word-break:break-all}
.hit .txt{color:var(--ink)}
footer{border-top:1px solid var(--border);padding:10px 18px;display:flex;gap:10px;align-items:center;background:var(--bg2)}
footer input[type=text]{flex:1;background:#030d14;border:1px solid var(--border);border-radius:8px;color:var(--ink);padding:9px 12px;font:14px 'Segoe UI',system-ui}
button{background:rgba(55,214,255,.12);border:1px solid rgba(55,214,255,.45);color:var(--accent);border-radius:8px;padding:9px 14px;font:600 13px 'Segoe UI',system-ui;cursor:pointer}
button:hover{background:rgba(55,214,255,.22)}
button.sec{background:transparent;border-color:var(--border);color:var(--muted)}
#status{font-family:var(--mono);font-size:11px;color:var(--faint);margin-left:8px;min-width:220px}
.try{font-size:11px;color:var(--faint);margin-top:8px;font-family:var(--mono)}
.truth{font-size:11px;color:var(--amber);border:1px solid rgba(255,181,71,.3);border-radius:8px;padding:6px 10px;margin-bottom:10px;background:rgba(255,181,71,.05)}
</style></head><body>
<header>
  <span class="logo">AXON · LIVE ORGANS</span>
  <span class="chip real" id="rootchip">state —</span>
  <span class="chip beats" id="beats">beats 0</span>
  <span class="chip" id="tick">tick —</span>
  <span class="chip" id="view">view —</span>
  <span class="chip" id="fid">field —</span>
  <span class="chip ok" id="rt">roundtrip —</span>
  <span class="chip right">no cores registered · nothing scripted · null ticks</span>
</header>
<main>
  <div class="col">
    <h2>Shared Field · attention sliders (durable)</h2>
    <div class="truth">Moving a slider beats the heart and freezes a NEW tick view. Masked characters go dormant in place — the canonical body and field_id never change. Cores attend everything; collapse below only hides it from observers.</div>
    <div id="regions"></div>
    <h2 style="margin-top:14px">Cortex engine · bounded working surface</h2>
    <div class="region">
      <div class="rname"><span>cortex ticks</span><span class="spans" id="cortex-last">—</span></div>
      <div style="font-size:11px;color:var(--muted);margin-bottom:8px">Ticks on every field change and on cadence. Each tick WIPES the cortex and replaces it with the latest hits: it searches semantic edges in the dormant state for every OTHER region (through their attention masks), scores through the relevance auditor (strict bar, no fallback), and trims whole spans to the char budget. Zero matches wipes the region empty; identical hits leave it untouched (duplicate, no churn). Evicted records stay in the dormant archive. The cortex has NO attention slider — it is the engine's surface, always fully attended.</div>
      <div class="slider-row"><span style="font-family:var(--mono);font-size:11px;color:var(--faint);width:86px">char budget</span><input type="range" id="cx-budget" min="400" max="6000" step="200" value="1600"><span class="pct" id="cx-budget-v">1600</span></div>
      <div class="slider-row"><span style="font-family:var(--mono);font-size:11px;color:var(--faint);width:86px">cadence s</span><input type="range" id="cx-cadence" min="2" max="60" step="1" value="5"><span class="pct" id="cx-cadence-v">5</span></div>
      <div style="display:flex;gap:8px;margin-top:8px;align-items:center">
        <label style="font-size:11px;color:var(--muted);font-family:var(--mono)"><input type="checkbox" id="cx-auto" checked> auto-tick</label>
        <button class="sec" id="cx-tick" style="padding:5px 10px;font-size:11px">tick now</button>
        <span id="cx-status" style="font-family:var(--mono);font-size:10.5px;color:var(--faint)"></span>
      </div>
    </div>
    <div class="try">try: “who is Jeff?” → the cortex surfaces records about Jeff at its next tick, pushes them to the top, and older entries fall deeper.</div>
  </div>
  <div class="col">
    <h2>D64 Rail · grouped by region · 4 × 16D cells per row · no caps</h2>
    <div class="legend"><span><i style="background:rgba(55,214,255,.45)"></i>native 16D char cell</span><span><i style="background:rgba(155,107,255,.45)"></i>UTF-8 byte cell</span><span><i style="background:rgba(55,214,255,.06);border:1px dashed rgba(55,214,255,.4)"></i>padding lane</span><span style="color:var(--faint)">collapse a region below — cores still attend it</span></div>
    <div id="rail"></div>
    <div class="railmeta" id="railmeta"></div>
    <div id="inspector">click any lane cell to inspect its real 16D vector</div>
  </div>
  <div class="col">
    <h2>Heart · beats, commits, dormant recall</h2>
    <div id="log"><div class="beat"><div class="bl">no beats yet — send ingress or press beat</div></div></div>
  </div>
</main>
<footer>
  <input type="text" id="say" placeholder="type ingress — it commits to user_input through the valve, then the heart beats" autocomplete="off">
  <button id="send">valve → heart</button>
  <button class="sec" id="beat">beat</button>
  <button class="sec" id="roundtrip">roundtrip check</button>
  <span id="status"></span>
</footer>
<script>
const $=s=>document.querySelector(s);
let STATE=null;
function api(path,body){return fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined}).then(r=>r.json())}
function esc(t){return (t||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function drawRegions(regions){
  $('#regions').innerHTML=regions.map(r=>{
    const showSlider=r.slider!==null;
    let body;
    if(r.region==='response_draft' && !r.attended){
      body=`<pre class="ghost">Reasoning Cores Coming Soon</pre><pre style="color:var(--faint)">(canonical: empty)</pre>`;
    } else if(r.masked_canonical){
      body=`<pre><span class="masked">[${r.attended}]</span></pre>`;
    } else {
      body=`<pre>${esc(r.attended)||'<span style="color:var(--faint)">—</span>'}</pre>`;
    }
    const head = r.region==='conversation_history'
      ? `<span>${r.region}</span><span class="spans">${r.mask_label||('all turns')} · ${r.turn_count??0} turns</span>`
      : r.region==='user_input'
      ? `<span>${r.region}</span><span class="spans">latest input only</span>`
      : `<span>${r.region}</span><span class="spans">${r.span_count} spans · ${r.canonical.length} chars</span>`;
    return `<div class="region"><div class="rname">${head}</div>${body}
      ${showSlider?`<div class="slider-row"><input type="range" min="0" max="100" value="${r.slider}" data-region="${r.region}"><span class="pct" id="sl-${r.region}">${r.region==='conversation_history'?(r.mask_label||'all turns'):r.slider+'%'}</span></div>`:''}
      ${r.region==='user_input'?'<div class="try" style="margin-top:6px">holds ONLY the latest input — older inputs rotate into conversation_history as turns</div>':''}
      ${r.masked_canonical?'<div class="warn">masked from attention — body untouched</div>':''}</div>`;
  }).join('');
  document.querySelectorAll('input[type=range]').forEach(el=>{
    const region=el.dataset.region;
    el.oninput=()=>{
      if(region==='conversation_history'){
        // Turn preview while dragging is resolved server-side; show raw %.
        el.parentElement.querySelector('.pct').textContent=el.value+'%';
      } else {
        el.parentElement.querySelector('.pct').textContent=el.value+'%';
      }
    };
    el.onchange=async()=>{
      flash('mask '+region+' → '+el.value+'% · heart beating…');
      const p=await api('/api/mask',{region,percent:+el.value});
      if(p.rejected){flash('MASK REJECTED: '+p.reason);return}
      apply(p); flash('new tick view frozen — '+short(p.view_id)+(p.mask_label?' · '+p.mask_label:''));
    };
  });
}
function short(id){return id?id.slice(0,10)+'…':'—'}
function drawRail(rail){
  const railEl=$('#rail');railEl.innerHTML='';
  for(const rg of rail.order){
    const rows=rail.groups[rg]||[];
    const det=document.createElement('details');det.className='railgroup';det.open=true;
    const sum=document.createElement('summary');
    sum.innerHTML=`<span>${esc(rg)}</span><span class="cnt">${rows.length} rows · ${rows.length*4} lanes</span><span class="hide-note">click to collapse (cores still attend)</span>`;
    const wrap=document.createElement('div');wrap.className='rows';
    for(const r of rows){
      const bar=document.createElement('div');bar.className='railbar';
      const chars=(r.lanes||[]).map(l=>l?l.char:'·').join('');
      bar.innerHTML=`<span class="rlabel">row ${r.row}</span><span class="chars" title="${esc(chars)}">${esc(chars)}</span>`;
      const lanesEl=document.createElement('div');lanesEl.className='lanes';
      (r.lanes||[]).forEach((lane,li)=>{
        const laneEl=document.createElement('div');laneEl.className='lane';
        if(!lane){laneEl.classList.add('pad');
          const c=document.createElement('canvas');c.width=16;c.height=16;c.style.width='28px';c.style.height='28px';c.title='padding lane (no character)';
          const g=c.getContext('2d');g.fillStyle='rgba(55,214,255,.05)';g.fillRect(0,0,16,16);
          laneEl.appendChild(c);lanesEl.appendChild(laneEl);return;}
        if(lane.kind!=='native_16d_cell')laneEl.classList.add('byte');
        const glyph=document.createElement('span');glyph.className='glyph';
        glyph.textContent=lane.char==='\n'?'⏎':lane.char;glyph.title=JSON.stringify(lane.char);
        const c=document.createElement('canvas');c.width=16;c.height=16;c.style.width='28px';c.style.height='28px';
        const g=c.getContext('2d');
        const mx=Math.max(...lane.vec.map(Math.abs),1e-6);
        for(let y=0;y<4;y++)for(let x=0;x<4;x++){
          const v=lane.vec[y*4+x]||0;const a=Math.abs(v)/mx;
          g.fillStyle=v>=0?`rgba(55,214,255,${(a*.85+.05).toFixed(2)})`:`rgba(255,181,71,${(a*.85+.05).toFixed(2)})`;
          g.fillRect(x*4,y*4,4,4);
        }
        c.title=`${JSON.stringify(lane.char)} · ${lane.region} · token ${lane.token} · ${lane.kind}`;
        c.onclick=()=>inspect(lane,r.row,li);
        laneEl.appendChild(glyph);laneEl.appendChild(c);lanesEl.appendChild(laneEl);
      });
      bar.appendChild(lanesEl);wrap.appendChild(bar);
    }
    det.appendChild(sum);det.appendChild(wrap);railEl.appendChild(det);
  }
  $('#railmeta').textContent=`rail ${rail.rail_id.slice(0,18)}… · ${rail.row_count} rows · ${rail.valid_lanes} valid lanes · coverage complete: ${rail.coverage_complete}`;
}
function inspect(lane,row,laneIdx){
  const vals=lane.vec.map(Math.abs), mx=Math.max(...vals,1e-6);
  $('#inspector').innerHTML=`<b style="color:var(--accent)">row ${row} · lane ${laneIdx}</b> — char ${JSON.stringify(lane.char)} · ${lane.region}<br>
    token ${lane.token} · ${lane.kind} · unit ${lane.unit}/${lane.unit_count}<br>16D cell (cyan + / amber −):
    <div class="bars">${lane.vec.map(v=>`<i style="height:${Math.max(4,Math.abs(v)/mx*100)}%;background:${v>=0?'var(--accent)':'var(--amber)'}"></i>`).join('')}</div>
    <span style="color:var(--faint)">[${lane.vec.map(v=>v.toFixed(2)).join(', ')}]</span>`;
}
function drawChips(){
  $('#beats').textContent='beats '+STATE.beats;
  $('#tick').textContent='tick '+STATE.tick_id;
  $('#view').textContent='view '+short(STATE.view_id);
  $('#fid').textContent='field '+STATE.field_id.slice(0,14)+'…';
}
function drawHits(hits){
  if(!hits||!hits.length)return;
  const el=$('#log');const div=document.createElement('div');
  div.innerHTML=`<div class="beat"><div class="bh">dormant recall surfaced</div>${hits.map(h=>`<div class="hit"><div class="txt">${esc(h.text)}</div><div class="src">${esc(h.kind)} · ${esc(h.source)}</div></div>`).join('')}</div>`;
  el.prepend(div);
}
function drawBeat(p){
  const el=$('#log');const div=document.createElement('div');
  const parts=[p.ingress?`ingress: ${esc(p.ingress)}`:'operator beat'];
  parts.push(`state ${p.state} · commits ${p.commits.length}`);
  if(p.deferred_event_id) parts.push('<span style="color:var(--amber)">item deferred by beat budget — beating again…</span>');
  div.className='beat';
  div.innerHTML=`<div class="bh"><span>beat ${p.heartbeat_sequence}</span><span>tick ${p.tick_id}</span></div><div class="bl">${parts.join('<br>')}</div>`;
  el.prepend(div);
}
function drawCortexInfo(){
  const lt=STATE.last_cortex_tick;
  if(lt){
    $('#cortex-last').textContent=lt.ticked
      ?('tick '+lt.trigger+' · '+lt.chars+' chars'+(lt.wiped?' · WIPED':''))
      :(lt.reason||'null tick').slice(0,34);
  }
}
function apply(p){
  if(!p){return}
  if(p.error){flash('ERROR '+p.error);return}
  if(p.admitted===false){flash('VALVE REJECTED: '+p.reason);return}
  STATE=p; drawRegions(p.regions); drawRail(p.rail); drawChips(); drawHits(p.cortex_hits); if(p.heartbeat_sequence!==undefined)drawBeat(p); drawCortexInfo();
}
// Live poll: notice server-side cortex ticks / heartbeats and refresh the
// whole view when something changed. Cheap summary call every 2.5 s.
let POLL=null;
function startPoll(){
  if(POLL)return;
  POLL=setInterval(async()=>{
    try{
      const s=await api('/api/summary');
      if(!STATE||s.tick_id!==STATE.tick_id||s.cortex_ticks_fired!==(STATE.cortex_ticks_fired||0)||s.cortex_chars!==(STATE.cortex_chars||0)){
        apply(await api('/api/state'));
      }
    }catch(e){/* server restarting */}
  },2500);
}
function flash(t){$('#status').textContent=t;setTimeout(()=>{if($('#status').textContent===t)$('#status').textContent=''},5000)}
$('#send').onclick=async()=>{const t=$('#say').value.trim();if(!t)return;$('#say').value='';flash('valve admitting… heart beating…');apply(await api('/api/ingress',{text:t}));flash('committed')};
$('#say').onkeydown=e=>{if(e.key==='Enter')$('#send').onclick()};
$('#beat').onclick=async()=>{flash('heartbeat…');apply(await api('/api/beat'));flash('beat done')};
$('#cx-budget').oninput=()=>{$('#cx-budget-v').textContent=$('#cx-budget').value};
$('#cx-budget').onchange=async()=>{await api('/api/cortex-config',{budget_chars:+$('#cx-budget').value,auto:$('#cx-auto').checked,cadence_s:+$('#cx-cadence').value});flash('cortex char budget → '+$('#cx-budget').value)};
$('#cx-cadence').oninput=()=>{$('#cx-cadence-v').textContent=$('#cx-cadence').value};
$('#cx-cadence').onchange=async()=>{await api('/api/cortex-config',{budget_chars:+$('#cx-budget').value,auto:$('#cx-auto').checked,cadence_s:+$('#cx-cadence').value});flash('cortex cadence → '+$('#cx-cadence').value+'s')};
$('#cx-auto').onchange=async()=>{await api('/api/cortex-config',{budget_chars:+$('#cx-budget').value,auto:$('#cx-auto').checked,cadence_s:+$('#cx-cadence').value});flash('cortex auto-tick '+(($('#cx-auto').checked)?'on':'off'))};
$('#cx-tick').onclick=async()=>{flash('cortex ticking…');const r=await api('/api/cortex-tick',{trigger:'manual'});if(r.error){flash('TICK ERROR '+r.error);return}
  const s=await api('/api/state');s.last_cortex_tick=r.ticked?{ticked:true,trigger:'manual',chars:r.chars,wiped:r.wiped}:r;apply(s);flash(r.ticked?(r.verdict+' — '+r.chars+' chars, '+r.spans+' spans'):('null tick — '+r.reason))};
$('#roundtrip').onclick=async()=>{const r=await api('/api/roundtrip');if(r.error){flash('ROUNDTRIP FAILED '+r.error);return}
  $('#rt').textContent=`roundtrip exact · ${r.valid_lanes}/${r.lanes} lanes`;flash('canonical body roundtrips exactly — '+r.rows+' rows'+(r.masked_active?' (view masked to '+r.view_rows+')':''))};
(async()=>{STATE=await api('/api/state');$('#rootchip').textContent='state '+(STATE&&STATE.field_id?'attached':'down');if(STATE&&!STATE.error){$('#cx-budget').value=STATE.cortex_budget;$('#cx-budget-v').textContent=STATE.cortex_budget+'/'+STATE.cortex_budget;apply(STATE);startPoll()}})();
</script></body></html>"""


def main() -> int:
    global RUNTIME
    parser = argparse.ArgumentParser(description="Axon live organ demo server (real state)")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_REAL_STATE_ROOT,
                        help="state root to attach to (default: the real D:/Axon/State)")
    parser.add_argument("--demo", action="store_true",
                        help="use the isolated demo root (State/tmp/organ_demo) with a demo-authored corpus instead")
    parser.add_argument("--rebuild-demo", action="store_true",
                        help="with --demo: wipe and rebuild the demo state root")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.demo:
        state_root = build_demo_root(rebuild=args.rebuild_demo)
        is_demo = True
    else:
        state_root = args.state_root.resolve()
        is_demo = False
        if args.rebuild_demo:
            raise SystemExit("--rebuild-demo requires --demo; refusing to touch a real state root")

    RUNTIME = DemoRuntime(state_root, is_demo)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    mode = "DEMO root (isolated)" if is_demo else "REAL state root"
    print(f"organ demo [{mode}]: {RUNTIME.state_root}")
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