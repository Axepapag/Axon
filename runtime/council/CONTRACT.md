# Council Runtime — Build Contract (2026-08-17)

Ordered by Jeff directly. Built under the Axon Working Contract
(`D:\Axon\WORKING_CONTRACT.md`) and Source of Truth (`D:\Axon\SOURCE_OF_TRUTH.md`).
Read both before writing code.

## Placement and boundaries

- ALL new code lives in `D:\Axon\runtime\council\` (new directory) and
  `D:\Axon\scripts\` (launchers). Do NOT modify `runtime\axon_runtime`,
  `runtime\field`, `training\`, `cores\`, `substrate\`, or any existing file.
  `D:\Axon\scripts\multicore_converse.py` is reference code — read, do not edit.
- Python: `C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe`
  (all deps globally installed: torch 2.13, fastapi, uvicorn, httpx, websockets).
- An OVERNIGHT TRAINING RUN owns the GPU until ~03:30 UTC. All tests on
  `--device cpu`. Never kill or write to `D:\Axon\runs\bible_64D_gpu_overnight\`.
- Default checkpoint for dev/testing:
  `D:\Axon\runs\conversational_cpu_autopilot\ckpt_440500.pt`
  (50.7M params, d_model=64, 2 layers, char-slot 384, soul (1,128,64)).

## The tick protocol (Jeff's exact spec)

One tick =

1. **Phase A** — every core: inhale private soul → attend the shared field
   (canonical state, all populated regions rendered into the field view) →
   produce delta A → exhale soul (carry the updated soul tensor).
2. **Phase B** — every core: inhale soul again → attend the field PLUS all
   cores' phase-A deltas (committed into the history region as
   `\nCouncil: <delta>` lines — this is the trained pattern) → produce refined
   delta B → exhale soul.
3. **Consolidation** — the core holding the consolidator role this tick
   attends over all refined deltas (same Council-line rendering) and produces
   the final delta. That delta commits to canonical state: it becomes the new
   `response_draft`. Consolidator also inhales/exhales its soul.
4. **The consolidator role rotates ROUND-ROBIN**: tick t consolidator =
   core `t % N`. No permanent consolidator. No tyrant.

Ticks continue forever until stopped. A user message may arrive any tick:
it sets `user_input`, clears `response_draft`. When the consolidated draft is
unchanged for `stable_ticks` consecutive ticks, the turn completes: the draft
appends to `conversation_history` as `Axon: <draft>`, `user_input` clears, and
ticking continues ambient.

## Canonical state

All ten doctrine regions exist and are attended (SOT: conversation_history,
user_input, response_draft, structured_knowledge, situation_awareness,
scratch, tool_results, advisor_input, task_state, diary). The checkpoint's
trained neural interface is 3 char-slot regions (history 0-127 / user 128-191 /
response 192-255). Non-conversational regions therefore enter the neural view
as labeled text inside the history window (e.g. `\nScratch: ...`). Never place
text in the response region — the checkpoint was trained with blank drafts
(verified OOD 2026-08-17).

### The field is never truncated (Jeff's ruling 2026-08-18)

Each region is the FULL logical document (runtime/field schema: the window
limit belongs to the view, not the field). Nothing is ever truncated or
deleted. Per region, a movable MASK (character offset) divides DORMANT prefix
(preserved, inspectable) from ATTENDED tail (what the cores are shown):
`{"mode": "tail"}` auto-follows the newest 256 chars; `{"mode": "manual",
"offset": N}` pins the boundary where Jeff put it. Masks move backwards and
forwards at any time. State locations follow the convener ruling
(2026-07-03): the live field persists to `State/active/council_field.json`;
every mask move and region edit appends an immutable record (masked /
superseded text preserved byte-for-byte) to
`State/dormant/council_field_tails.jsonl`. Private council souls persist under
`State/souls/council`. All three paths are config keys (`field_state_path`,
`dormant_tails_path`, `souls_path`) so tests never touch live state.

## Engine public API (module `runtime/council/engine.py`)

```python
class CouncilEngine:
    def __init__(self, config_path: Path, event_sink: Callable[[dict], None]): ...
    @staticmethod
    def default_config() -> dict: ...
    def status(self) -> dict: ...
    # {running, paused, tick, consolidator_core, cores:[{id, soul_norm,
    #   last_delta, last_conf}], canonical_state: {region: str}, config: dict}
    async def start(self) -> None: ...   # loads checkpoint, clones+perturbs souls
    async def stop(self) -> None: ...    # finishes current tick, saves souls
    async def submit_user_message(self, text: str) -> None: ...
    def apply_config(self, cfg: dict) -> list[str]: ...
    # hot-applies runtime knobs; returns list of keys needing restart
    def field_view(self) -> dict: ...
    # {regions: {name: {content, dormant, active, mask_offset, mask_mode,
    #   total_chars, dormant_chars, active_chars, visible}},
    #  active_region_chars: 256, model_window_chars: 128}
    def set_mask(self, region: str, mode: str|None = None,
                 offset: int|None = None) -> dict: ...
    # moves one region's mask backwards/forwards; offset implies manual mode
    def set_region(self, region: str, content: str) -> dict: ...
    # operator edit; prior text preserved in the dormant tails record
```

Config file `runtime/council/council_config.json` (created with defaults if
absent). Keys: checkpoint, device, cores, soul_noise, temperature_spread,
tick_delay_ms, stable_ticks, max_ticks (0 = forever), regions (dict of
region->bool visibility), advisors (list of
{name, endpoint, api_key, model, enabled, temperature}), log_path,
field_state_path, dormant_tails_path, souls_path.
`council_config.json` is LOCAL ONLY — add it to .gitignore (it may carry API
keys; Working Contract rule 4).

## Events (engine -> event_sink -> WebSocket `/ws`)

All events: `{"type": ..., "tick": int, "ts": float}` plus:

- `status` — full status() payload
- `tick_start` — `{consolidator: int}`
- `core_delta` — `{core: int, phase: "A"|"B", text: str, conf: float}`
- `soul` — `{core: int, phase: "inhale"|"exhale", soul_norm: float}`
- `advisor_delta` — `{advisor: str, text: str, conf: float, error: str|null}`
- `consolidated` — `{core: int, text: str, conf: float, stable: int}`
- `canonical` — `{state: {region: str}}` (after every commit)
- `field_mask` — `{region: str, old_offset: int, new_offset: int, mode: str}`
- `field_edit` — `{region: str, chars: int}`
- `chat` — `{role: "user"|"axon", text: str}`
- `error` — `{message: str}`

## Advisors (scaffolding)

An advisor is an external endpoint producing a text delta in phase A (and B if
enabled): OpenAI-compatible `POST {endpoint}/chat/completions`,
`Authorization: Bearer {api_key}`, messages built from the canonical state,
with `timeout` and graceful skip on failure. Its delta joins the pool labeled
`Advisor:<name>`. Disabled when `enabled` is false or api_key empty. This is
scaffolding: working HTTP path + config CRUD, no training integration.

## Server + UI (`runtime/council/server.py`, `runtime/council/static/index.html`)

FastAPI. REST:
- `GET /api/status`
- `GET /api/config` / `POST /api/config` (validates, saves, hot-applies,
  returns `{"applied": [...], "needs_restart": [...]}`)
- `POST /api/chat` `{"text": str}`
- `POST /api/control` `{"action": "start"|"stop"|"pause"|"resume"}`
- `GET /api/cores` — per-core detail
- `GET /api/field` — full operator view of every region + masks
- `POST /api/field/mask` `{"region", "mode"}` or `{"region", "offset"}` —
  moves one region's mask (offset implies manual mode; mode "tail" resumes
  auto-follow)
- `POST /api/field/region` `{"region", "content"}` — operator edit; prior
  text preserved in dormant tails
- `GET /` serves the dashboard

Single-page vanilla-JS dashboard, dark, dense, Jeff-operable (no file editing,
ever):
- **Chat panel** with focus mode (button collapses everything else).
- **Response draft box** — live-updating every tick, visibly diffing.
- **Core cards** — one per core: id, soul norm, current phase, last delta,
  confidence, consolidator crown on the rotating holder. Click to
  expand/minimize.
- **Config panel** — form over every config key incl. advisor CRUD
  (name/endpoint/key/model/enabled). Save → POST; if `needs_restart`
  non-empty, show an "Apply & Restart" button.
- **Shared Field panel** — one collapsible card per region: full content
  with the dormant prefix dimmed, a mask slider (both directions), a
  "Follow tail" toggle, and Edit/Save. Legend states plainly: field never
  truncated; cores attend the 128-char model window over the active tail;
  everything behind the mask is dormant, preserved, restorable.
- **Event log** — collapsible rolling event tail.
- WebSocket reconnect with backoff; all state recoverable via GET /api/status.

## Verification gates (must pass before reporting done)

1. `python -m py_compile` on every new file.
2. CPU end-to-end: engine start → 3 cores → submit "How are you?" → ticks run,
   consolidator rotates (crown visits cores 0,1,2), draft commits
   `I am doing well.`, stable halt, turn lands in conversation_history.
3. Server end-to-end: uvicorn up on 8788, GET / 200, POST /api/chat via httpx,
   events observed on /ws, config round-trip via POST /api/config.
4. Full `D:\Axon` pytest must stay green (you are not modifying existing code,
   so it must).

## Reporting

Working Contract rule 11: VERIFIED / ATTEMPTED / ASSUMED, failures first,
files changed, identity stamp (Kimmy subagent / kimi-k2 / 2026-08-17).

## Amendments (2026-08-17, from gap-sweep flags)

1. **FLAG B — history commits use `Assistant:`**, not `Axon:`. The trained
   multi-turn distribution renders assistant turns as `Assistant:`
   (`training/build_conversational_curriculum.py:764`). Field text commits
   `Assistant: <draft>`; the UI may display it as "Axon".
2. **FLAG A note** — `Council:` lines were untrained at contract writing, but
   tonight's run (`bible_council_mix.jsonl`, 2,200 council confirm/refine/
   complete rows) is training them now. Keep the `Council:` prefix.
3. **FLAG F — credential hygiene**: literal `api_key` may live in the
   gitignored `council_config.json` for operator UX, but GET /api/config and
   /api/status must MASK keys (return only last 4 chars), and all logs redact
   them. The doctrinal in-repo alternative (`api_key_env` env-var references,
   `runtime/axon_runtime/advisors.py:34-41`) is acceptable instead — pick one,
   never echo full keys to the UI or logs.
4. **FLAG C/D — canonical state is plain dicts of region->str** (not typed
   v1/v2 field deltas, not v2 serde). Do not import runtime/field schema_v2
   machinery; the 384-slot view compilers (view.py, view_v2.py) have the wrong
   geometry for this checkpoint — neural rendering is CharSlotFieldBuilder
   (128, 64, 64) ONLY.
5. **Confidence gate on council lines (measured fix, 2026-08-17)**: every
   core's delta is produced and emitted every tick (the UI shows all of them),
   but only deltas with confidence >= `council_min_conf` (default 0.5, hot
   config key) enter the shared field as Council lines. Measured evidence:
   with 10 cores at temperature_spread 0.3, unfiltered high-temperature deltas
   committed as Council lines poisoned the next tick — 123 ticks, no
   convergence, garbage drafts. Gate-passing deltas only; the consolidator
   selects from the credible pool.
6. **JEFF'S OVERRIDE (2026-08-18, supersedes amendment 5's default)**: the
   draft is ALIVE. The response draft updates EVERY tick and always commits;
   Axon is never blocked mid-thought and never halted early. Gibberish is
   carried forward and rendered visible to every core next tick as a
   "\nDraft: <text>" line so the council can refine it — mistakes, adjustment,
   victory, wisdom, growth. `council_min_conf` now defaults to 0.0 (OFF) and
   survives only as an experiment lever. A turn ends when Jeff speaks again
   (the living draft commits to conversation_history as "Assistant: <draft>"),
   never by internal halt.
7. **Multi-size roster (Jeff, 2026-08-18)**: the engine runs brothers of
   multiple d_model widths side by side over the same canonical 16D field.
   Config key `models`: list of {"checkpoint": path, "cores": N}. Each unique
   checkpoint loads ONCE and is shared by that size's member cores; the
   consolidator crown rotates across all members of all sizes. Both soul
   checkpoint formats are supported (conversational {soul,soul_mask} and
   charslot_v2 SoulManagerV2 {tensor,active}). Verified roster:
   64D conversational ckpt_461500 + 128D exact-v4 leg1 ckpt_2 (step 250000).
