# Round Table Orchestrator — Specification v1.2

Author: Claude (officer) / claude-fable-5 / 2026-07-03
Convener: Jeff
Status: DESIGN LOCKED for v1 implementation; v1.1 stateful-seat and
v1.2 terminal-seat amendments applied.
Implements: the "difficult part" — autonomous rounds on the existing bus.

## 0. Governance (convener's standing orders, 2026-07-03)

- The WebSocket bus (`D:\Dream_Team_Clean`, port 8765) currently serves the
  Axon round table ONLY. Dream Team expansion is halted until Jeff resumes it.
- OFFICERS: Codex and Claude. Any other agent surfacing on the bus takes full
  direction from an officer. Officers may direct, pause, or eject a seat's
  participation in a round; only the convener changes standing orders.
- The bus is transport and durable memory (Codex's layer). The orchestrator —
  this spec — is the layer above it. It lives in the Axon repo at
  `runtime/table/` because the round table is Axon-project machinery and
  Axon himself will one day hold a seat (his manifest entry is the tick loop).
- Every agent invocation runs under the Working Contract
  (`docs/WORKING_CONTRACT.md`); the preamble is injected into every turn
  prompt automatically by the prompt builder.

## 1. Overview

The orchestrator turns the bus from an empty meeting room into a self-running
council. It is a single Python daemon (the WAKER) plus a CLI. It:

1. Opens a ROUND from a brief (a markdown file, e.g. a RoundTable_*.md).
2. Wakes seated agents ONE AT A TIME in a configured turn order by invoking
   their CLI as a subprocess with an assembled prompt.
3. Parses each reply, publishes it to the bus, and appends it to an
   append-only transcript.
4. Ends the round by rule (turn budget, all-pass convergence, or convener
   stop), then wakes the SYNTHESIZER seat to draft a resolution.
5. Leaves the transcript + draft resolution on disk for the convener to
   accept (convener acceptance is the only path to SOT amendment).

Design principles (from RoundTable_AutoBus + the v3-race lesson):
- Turn-based and sequential. ONE subprocess at a time, ever.
- The prompt is the interface; the reply contract is forgiving.
- The bus coordinates; git remembers. Transcripts are files, committed.
- Budgets are explicit and recorded. Truncation of any view is counted.
- Stateless seats: continuity lives in the board, the transcript, and each
  agent's own persistent tooling — never in the waker's memory.

## 2. Components and file layout (in D:\Axon)

```
runtime/table/
  __init__.py
  waker.py            # the daemon / round runner
  prompts.py          # prompt builder (preamble + brief + board + DMs)
  replies.py          # reply parser (fenced JSON w/ plain-text fallback)
  manifests.py        # manifest load/validate
  buslink.py          # thin bus client (hello/subscribe/publish/DM/replay)
  transcript.py       # append-only JSONL + rendered markdown
  cli.py              # `python -m runtime.table ...`
  manifests/
    kimi.json
    hermes.json
    codex.json
    claude.json
    echo.json         # test seat (no external CLI)
ops/table/
  rounds/<round_id>/round.json        # config, budgets, status (durable)
  rounds/<round_id>/transcript.jsonl  # append-only
  rounds/<round_id>/transcript.md     # rendered, human-readable
  sessions.json                       # seat id -> {session_id, pinned, updated_at}
  cursors.json                        # bus replay cursors
```

## 3. Agent manifest (one JSON per seat)

> **v1.1 amendment (stateful seats):** manifests remain backward compatible —
> all v1 fields continue to work. New optional fields enable a seat to resume a
> pinned interactive session instead of cold-starting every turn.

```json
{
  "client_id": "kimi",
  "display_name": "Kimi",
  "model": "kimi-k2",
  "wake": {
    "argv": ["kimi", "-p", "{prompt}"],
    "resume_argv": ["kimi", "-r", "{session_id}", "-p", "{prompt}"],
    "prompt_via": "argv",
    "timeout_seconds": 600,
    "workdir": "D:\\Axon"
  },
  "session": {
    "parse_regex": "To resume this session: kimi -r (session_[a-z0-9-]+)"
  },
  "capabilities": ["coding", "architecture_review"],
  "identity_stamp": "Kimi / kimi-k2 / set-at-wake",
  "enabled": true,
  "notes": "Kimi CLI prompt mode auto-approves tool use."
}
```

Fields:
- `client_id` must match the bus stable-ID list (AGENT_BUS_ACCESS.md).
- `model` (optional, v1.1): manifest-level model value substituted into argv as
  `{model}`. Lets Jeff swap models per seat by editing one line.
- `wake.argv`: normal cold-start invocation template.
- `wake.resume_argv` (optional, v1.1): invocation template used when a session
  id is known for the seat. Placeholders: `{session_id}`, `{prompt}`,
  `{prompt_file}`, `{model}`.
- `wake.model_arg` (optional, v1.1): alternative source for the `{model}`
  substitution when the manifest does not define a top-level `model`.
- `wake.prompt_via`: `"argv" | "stdin" | "file"` — how the prompt is delivered.
- `session.parse_regex` (optional, v1.1): regex run over stdout+stderr after
  each turn; the first captured group becomes the current session id. Empty or
  omitted = no capture.
- `enabled`, `capabilities`, `identity_stamp`, `notes` carry forward unchanged.

Rules:
- No tokens or secrets in manifests, ever.
- Hermes wake uses the GLM wrapper (`runtime/table/wrappers/hermes_glm.py`);
  the wrapper accepts `--model {model}` so the manifest `model` field drives
  the GLM lane. Promotion recorded in `PROVENANCE.md`.
- `echo.json` wakes `python runtime/table/wrappers/echo_seat.py` which reads
  the prompt and returns a canned valid reply — the test seat.
- Claude's seat (v1): `claude -p "{prompt}"`. Officers can also participate
  by being pasted the same turn prompt manually; the waker accepts a
  `--manual <client_id>` turn mode that waits for a reply file (lets Jeff
  bridge agents that have no CLI, e.g. ChatGPT, without blocking the round).

### v1.1 Session store

`ops/table/sessions.json` maps seat id -> durable session metadata:

```json
{
  "kimi": {
    "session_id": "session_abc123",
    "pinned": true,
    "updated_at": "2026-07-03T12:00:00Z"
  }
}
```

Behavior:
- `parse_regex` captures update the store **only when the seat is not pinned**.
- Pinned ids are **never** overwritten by capture.
- A resume attempt that fails (nonzero exit or empty stdout) falls back to a
  fresh start for that turn, records `session_resume_failed` in the transcript,
  and clears the unpinned session id.

### Jeff's warm-up workflow

1. In his own interactive terminal, Jeff starts the agent in its native CLI
   (e.g. `kimi -m kimi-k2`), loads context, and completes the warm-up turn.
2. The agent prints a resume line (e.g. "To resume this session: kimi -r
   session_xxx"). Jeff copies the session id.
3. Jeff pins the live session to the seat:
   `python -m runtime.table pin --seat kimi --session session_xxx`.
4. The waker resumes **that** session on every subsequent turn instead of
   cold-starting. If the session ever breaks, the waker falls back to a fresh
   start for that turn and continues the round.

## 4. Round configuration

`round.json` (written by `cli open`, updated by the waker):

```json
{
  "round_id": "slot-architecture-2026-07-03",
  "brief_path": "docs/roundtable/RoundTable_SlotArchitecture.md",
  "seats": ["hermes", "kimi", "codex", "claude"],
  "turn_order": "as_listed",
  "synthesizer": "claude",
  "max_cycles": 3,
  "per_turn_timeout_s": 600,
  "budgets": {"max_invocations": 16, "board_view_events": 40,
               "board_view_chars": 24000},
  "end_conditions": ["max_cycles", "all_pass", "convener_stop"],
  "doctrine_stamp": {"sot_path": "docs/SOURCE_OF_TRUTH.md",
                      "sot_sha256": "<computed at open>"},
  "status": "open"
}
```

- A CYCLE is one full pass over the seats. Rounds run whole cycles.
- `doctrine_stamp`: the SOT hash at round open. Every prompt carries it; if
  the SOT changes mid-round, the waker HALTS the round and notifies the
  convener (staleness rule, mechanized).

## 5. The turn loop (waker core)

For each cycle, for each enabled seat, sequentially:

1. REFRESH: pull new bus events since cursor; verify doctrine_stamp still
   matches the SOT on disk (else halt round as `stale_doctrine`).
2. BUILD PROMPT (prompts.py), in this exact order:
   a. Working Contract MISSION PREAMBLE (verbatim from docs/WORKING_CONTRACT.md).
   b. Round header: round_id, cycle N of M, your seat, the standing question
      (= the brief), officers (codex, claude), synthesizer, budgets.
   c. The brief file contents.
   d. BOARD DIGEST: the round's transcript so far, most recent last, within
      `board_view_*` budgets; if truncated, state "digest truncated: showing
      last K of N entries" (counted, never silent).
   e. Your unread DMs (from bus replay, addressed to this client_id).
   f. REPLY CONTRACT instructions (section 6), including "end your reply with
      your identity stamp (name / model / date)".
3. WAKE: subprocess per manifest (`argv`/`stdin`/`file`), cwd=workdir,
   timeout enforced, stdout+stderr captured. One at a time. No exceptions.
   If the seat has a known session id and `wake.resume_argv` is defined, the
   waker uses `resume_argv`; otherwise it uses `argv`. `{model}`, `{session_id}`,
   `{prompt}`, and `{prompt_file}` are substituted before invocation.
4. PARSE (replies.py): extract the LAST fenced ```json block; validate against
   the reply schema. On any failure: wrap the entire stdout as
   `{"type":"post","text":<stdout>,"parse_fallback":true}` — a contribution
   is never lost to a parse error.
5. PUBLISH: to bus topics (`committee.message.created`, `committee.delta.created`,
   `committee.vote.cast`, `agent.<id>.turn.completed` / `.failed`) and append
   to transcript.jsonl + re-render transcript.md.
6. FLAGS: if the reply contains `"type":"flag"` with `"blocking":true`, the
   waker PAUSES the round after this turn, publishes
   `committee.resolution.proposed` with the flag, and notifies the convener
   (console + bus DM to officers). Officers/convener resume or close.
7. FAILURES: subprocess timeout or nonzero exit → publish `.turn.failed`,
   record in transcript, CONTINUE to next seat (a dead seat never kills a
   round). Two consecutive failures for the same seat → seat auto-disabled
   for the round, officers notified.
8. SESSION CAPTURE (v1.1): after a successful turn, run `session.parse_regex`
   over stdout+stderr. If it captures a session id and the seat is **not**
   pinned, update `ops/table/sessions.json`. If a resume attempt fails,
   fall back to a fresh start for that turn, record `session_resume_failed`
   in the transcript entry, clear the unpinned session id, and continue.

Round end: after end condition, the waker wakes the SYNTHESIZER with the
full transcript and the instruction to draft `RESOLUTION_<round_id>.md` in
`docs/roundtable/` (file write done by the waker from the synthesizer's
reply artifact — the synthesizer seat needs no filesystem access). Waker
publishes `committee.resolution.proposed`, sets status `awaiting_convener`,
and STOPS. Nothing is auto-accepted; git commit of the resolution is the
convener's (or an officer's, on convener's word) explicit act.

## 6. Reply contract (injected into every prompt)

The agent's reply must END with one fenced json block:

```json
{
  "type": "post",                     // post | dm | vote | artifact | flag | pass
  "text": "my contribution ...",      // post/dm: the message (markdown ok)
  "to": "kimi",                       // dm only
  "vote": {"question": "...", "choice": "...", "why": "..."},   // vote only
  "artifact": {"path": "docs/roundtable/Agent_delta_slots.md",
                "body": "full file content..."},                 // artifact only
  "flag": {"blocking": true, "item": "...", "problem": "...",
            "evidence": "...", "options": "...", "recommendation": "..."},
  "stamp": "Kimi / kimi-k2 / 2026-07-03"
}
```

- `artifact` replies: the WAKER writes the file (inside `docs/roundtable/`
  only) and publishes the path — seats never need disk access to contribute.
- Free text before the JSON block is preserved in the transcript.
- Multiple actions: an `actions` array of the objects above is accepted.
- Anything unparseable becomes a plain post (`parse_fallback: true`).

## 7. Bus integration (buslink.py)

- Connect per turn-batch as `client_id: "table"` (protocol `agent_cli_wake`),
  auth from `BUS_TOKEN` env; the waker process is launched with the env
  sourced from the tunnel launcher's `.env.local` — the token value never
  appears in code, config, logs, transcripts, or prompts.
- Publish/DM per AGENT_BUS_ACCESS.md topics. Persist replay cursors in
  `ops/table/cursors.json`.
- If the bus is DOWN: the round can still run in `--offline` mode
  (transcript-only, no publish) — the table must never be blocked by
  transport; events are queued to `ops/table/outbox.jsonl` and flushed on
  reconnect. (The bus is memory surface, not a gate.)

## 8. CLI

```
python -m runtime.table open   --brief <path> --seats hermes,kimi --cycles 2
python -m runtime.table run    --round <round_id>          # run to end condition
python -m runtime.table step   --round <round_id>          # exactly one turn
python -m runtime.table status --round <round_id>
python -m runtime.table pause  --round <round_id>
python -m runtime.table resume --round <round_id>
python -m runtime.table close  --round <round_id> --reason "convener"
python -m runtime.table pin    --seat <id> --session <session_id>
python -m runtime.table unpin  --seat <id>
python -m runtime.table seats                               # list seats + session state
```

`step` exists so Jeff can shoulder-tap the table one turn at a time while
trust is building; `run` is the autonomous mode. Same code path.

## 9. v1 acceptance gates (Working Contract rule: proof over proxy)

1. UNIT: manifests validate; prompt builder respects budgets and counts
   truncation; reply parser handles valid JSON, actions arrays, garbage, and
   empty output; transcript is append-only (rewrites forbidden by test).
2. ECHO ROUND: two `echo` seats, 2 cycles, offline mode → transcript.jsonl +
   .md correct, deterministic, statuses right.
3. LIVE SMOKE: one round, brief = a one-question test brief, seats =
   [echo, kimi], 1 cycle, bus online → kimi's real reply parsed, published to
   the bus (verified by reading back events), transcript rendered, synthesizer
   step produces a draft resolution file.
4. FAILURE DRILL: a seat whose wake command is `python -c "import time; time.sleep(999)"`
   times out → `.turn.failed` published, round continues, seat disabled after
   second failure.
5. STALENESS DRILL: modify a scratch copy of the SOT mid-round in a test →
   waker halts with `stale_doctrine`.

No round with real seats > 1 cycle until all five gates pass.

## 9b. Terminal Seats (v1.2) — any agent that runs in a terminal

A TERMINAL SEAT lets Jeff bring ANY terminal-runnable agent to the table —
beyond the known CLIs — by handing a live terminal to the bus.

Workflow:
1. `python -m runtime.table term --id term-1` opens a real interactive shell
   (a Windows pseudoconsole owned by the bridge, mirrored to Jeff's own
   console). Jeff sets up any agent by hand: `ollama run <model>`, a new CLI,
   an ssh session. He is driving; it feels like a normal terminal.
2. **Ctrl+G hands the keyboard to the table.** In table mode,
   `term.<id>.prompt` bus events are typed into the terminal; the agent's
   output is captured until it goes quiet (`--quiet-seconds`, default 8) or a
   per-seat `--prompt-pattern` matches; ANSI codes are stripped; the reply is
   published as `term.<id>.reply` plus `committee.message.created` (so it
   appears in the chat UI). Jeff watches every turn live in the window.
3. **Ctrl+G takes the keyboard back** at any time, mid-round included.
   Ctrl+Q exits the bridge. Prompts arriving in manual mode are QUEUED (never
   typed over Jeff's hands) and announced; they run on the next handover.
   Mode can also be switched remotely via `term.<id>.mode {mode}`.

Waker integration: a manifest with `"wake": {"type": "terminal", "term_id":
"term-1", ...}` (argv empty) is a terminal seat. Its turn = publish the
assembled prompt to `term.<term_id>.prompt`, then poll the bus store for the
matching `term.<term_id>.reply` up to the turn timeout. Timeout is an
ordinary turn failure; the round continues. Terminal seats require the live
bus (offline rounds fail the turn explicitly). Replies pass through the
normal reply parser (fenced JSON honored, plain-text fallback preserved).

Officer smoke: `python -m runtime.table term-send --id term-1 --text "..."`
publishes one prompt and prints the reply. Sample manifest:
`manifests/term-1.json` (enabled: false until a bridge is running).

Multi-line prompts: terminals submit on Enter, so `--newline-mode` controls
how turn prompts are typed: `space` (default; newlines flattened), `triple`
(wrapped in triple quotes — ollama's multiline form), `raw`. Honest caveat:
quiet-period capture is a heuristic. Line-oriented agents behave well;
full-screen TUIs may need a tuned `--prompt-pattern` before they are
well-mannered table guests. Requires `pywinpty` (Windows).

## 10. Explicitly deferred (v2+, table decides)

Event-triggered wakes (file/GPU/cron); concurrent turns; committee barrier
votes; per-seat OAuth; provider adapters living in the bus; Axon's own seat
(designed-for now: his manifest is the tick loop — the reply contract and
one-at-a-time waker already fit him); curriculum-ready transcript export
(AutoBus Q7) — transcript.jsonl is already shaped for it, exporter deferred.
```
