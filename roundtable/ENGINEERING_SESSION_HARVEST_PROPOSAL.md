# Engineering-Session Harvest — Proposal

Author: Kimmy / Kimi Code CLI / 2026-08-29
Requested by: Jeff (convener), 2026-08-29
Audience: Codex and ChatGPT (reviewers), Jeff (final authority)
Status: PROPOSAL ONLY. Nothing here is doctrine until Jeff ratifies. No code,
gate, schema, budget, or Source of Truth text is changed by this document.

---

## 1. What Jeff asked for

Jeff's directive, 2026-08-29: harvest small, medium, and large training
sessions **from** the engineers' own session folders —
`C:\Users\axema\.kimi-code` and `C:\Users\axema\.codex` — because these are
valuable engineering sessions from the engineers working on Axon.

This proposal covers: what is actually on disk (verified, not assumed), how
the harvest maps onto the existing Dormant import and Trainer session
contracts, how small/medium/large tiers should be cut, the outcome-evidence
rules, and the two flags that gate the work. It ends with review questions.

## 2. Verified source survey (2026-08-29, read-only)

### 2.1 Codex (`C:\Users\axema\.codex`)

- Transcripts: `sessions\YYYY\MM\DD\rollout-*.jsonl` — 3 files, ~44 MB.
- Record kinds observed (first 400 lines of the 2026-08-17 rollout):
  `session_meta`, `turn_context`, `response_item/message`,
  `response_item/reasoning`, `response_item/custom_tool_call` +
  `custom_tool_call_output`, `event_msg/user_message`,
  `event_msg/agent_message`, `event_msg/agent_reasoning`,
  `event_msg/task_started`, `event_msg/task_complete`,
  `event_msg/token_count`, `event_msg/patch_apply_end`, `compacted`,
  `event_msg/context_compacted`, `world_state`.
- Side stores present: `thread_history_1.sqlite`, `goals_1.sqlite`,
  `memories_1.sqlite`, `logs_2.sqlite`, `queue_1.sqlite`,
  `session_index.jsonl`, `transcription-history.jsonl`.
- Credential-bearing stores present in the same folder tree: `auth.json`,
  `config.toml`, `.sandbox-secrets\`.

### 2.2 Kimi Code (`C:\Users\axema\.kimi-code`)

- Sessions: `sessions\wd_axon_1fe2d78b6eb4\session_<uuid>\` — 6 Axon
  sessions, ~24 MB total.
- Per session: `state.json` (id, version, cwd, timestamps, archive flag,
  agent metadata), `agents\main\wire.jsonl` (the full transcript),
  `logs\kimi-code.log`, `tasks\` (background-task outputs).
- Indexes: `session_index.jsonl`, `user-history\`.
- Credential-bearing stores present: `credentials\`, `config.toml`.

Total raw material across both: ~68 MB of transcripts. Small enough that the
entire harvest, import, and verification is CPU-cheap.

## 3. Harvest design (extends the ratified D00 import pattern)

### Step 1 — Importer: `curator/import_engineering_sessions.py`

Modeled directly on `curator/import_d00_memories.py`:

- Sources opened **read-only**; every transcript file snapshotted
  byte-for-byte into a content-addressed Dormant source snapshot
  (`RecoveredSourceSnapshot`: sha256, size, source path) before any parsing.
- Parsed into `ExperienceRecord`s under a new import label (suggested:
  `engineering-sessions-v1`) with new record kinds, one per observed event
  family:
  - `engineer_session_meta` (agent, model, cwd, timestamps, session id)
  - `engineer_user_turn`, `engineer_assistant_turn`
  - `engineer_assistant_reasoning` (observed-only; see §5)
  - `engineer_tool_call`, `engineer_tool_result`
  - `engineer_task_boundary` (task_started / task_complete / turn markers)
  - `engineer_compaction_marker` (compact/context_compacted events)
- Every record carries `source_pointer` = source file + line number, bound to
  the raw snapshot's SHA256, per the D00 provenance contract.
- One `ExperienceImportManifest` with exact record-kind counts; oversize or
  unparseable items are skipped **and** counted **and** reported (no silent
  truncation, Working Contract §7).

### Step 2 — Tiering: measured, not invented

- The split unit is always the **whole session** (whole-episode doctrine; no
  partial conversations, no train/heldout leakage across a session).
- Measure every session in transport-expanded canonical characters (the
  351-category Unicode transport already covers all content).
- Cut small/medium/large at the measured distribution's natural knees rather
  than inventing bands a priori. Expected shape:
  - *small* — single-issue Q&A / fix turns;
  - *medium* — multi-step task sessions (one mission, start to ledger
    closeout);
  - *large* — whole campaigns spanning multiple sessions and commits
    (e.g. the Candidate-A turn), chained as one trajectory with lineage
    links, never truncated.
- Publish the measured histogram with the import report; tier cutoffs are a
  recorded decision, revisable by the table.

### Step 3 — Compilation into Trainer sessions

- A compiler in the `LivedExperienceSessionCompiler` pattern
  (`runtime/trainer/sessions.py`) turns the imported records into immutable
  Trainer sessions, stamped `serving_promotion_eligible=False`.
- Outcome labels come **only from explicit evidence**, per the standing
  TARGET-QUALITY GAP flag and the Soul promotion doctrine
  (`VETTED_OUTCOME_QUALITIES` = success / corrected / endorsed):
  - success: `task_complete` events, landed commit hashes, passing test
    output, ledger closeout events;
  - corrected: explicit user corrections inside the session (verbatim
    follow-up messages that retract or redirect);
  - endorsed: Jeff's explicit approvals.
- Everything else remains observed-only evidence — importable, queryable,
  never supervision.

### Step 4 — Representation

Already solved: the additive 351-category Unicode transport
(`substrate/unicode_transport.py`, ratified 2026-08-28) covers every scalar
in these logs without normalization or destructive replacement. Expansion
factors are measured and reported per tier.

### Step 5 — Verification

- Exact transcript roundtrip: rebuild each session's event stream from the
  imported records and compare against the raw snapshot.
- Record counts reconciled against source line counts; every skip counted.
- Credential-screen quarantine counts reported (see §4).
- Full repository suite green before any curriculum consumes the import.

## 4. FLAG [BLOCKING] — credential screen before any import

Both session folders co-locate credential stores (`auth.json`,
`credentials\`, `.sandbox-secrets\`, keyed `config.toml` files), and
transcript *content* routinely contains tool output that can dump
environment variables, config files, or tokens.

Rules proposed:

1. The known credential stores are never opened as sources — excluded by
   path policy, not by inspection.
2. A pattern screen runs over every parsed record (token formats, private-key
   headers, `api_key`/`authorization`/`bearer` shapes, connection strings).
3. Any hit **quarantines** the record (excluded + counted + reported); the
   screen fails closed — ambiguity means quarantine, never import.
4. The screen ships with its own tests, including planted-canary fixtures,
   before the importer's first real run.

No engineering-session import runs before this exists and passes.

## 5. FLAG [ADVISORY] — self-reference and stale doctrine

These sessions are Axon engineers working *on Axon*. That is high-value
autobiography, but:

- Sessions contain superseded proposals and stale doctrine. Every record
  must carry its source timestamp; curriculum construction must apply the
  standing rule that current canonical evidence overrides stale proposals.
- Agent reasoning streams (`agent_reasoning`, wire thinking) are imported as
  **observed-only** records — they are process evidence, not supervision
  targets.
- Codex sqlite side stores may duplicate rollout content; dedupe by content
  hash before counting coverage.

## 6. Review questions

- **Q1 — Record granularity.** One record per wire/rollout event (proposed),
  or aggregated per-turn records with event manifests? Per-event maximizes
  provenance; per-turn halves record count.
- **Q2 — Tier cutoffs.** Agree with measured distribution knees, or should
  small/medium/large be fixed character bands tied to page budgets?
- **Q3 — Outcome evidence.** Are `task_complete` + landed commit + passing
  tests + ledger closeout + verbatim Jeff corrections the right success/
  corrected/endorsed signals? What is missing or wrongly included?
- **Q4 — Credential screen.** Quarantine whole record on any hit (proposed),
  or redact-and-import with a redaction receipt? Kimmy's position: quarantine;
  redaction rewrites evidence.
- **Q5 — Codex side stores.** Import `thread_history`/`goals`/`memories`
  sqlite as separate sources with dedupe, or treat rollouts as canonical and
  side stores as indexes only?
- **Q6 — Self-reference.** Should doctrine-discussion content be eligible for
  supervision at all, or observed-only until the lived-outcome adjudication
  (TARGET-QUALITY GAP) is resolved?
- **Q7 — Large-tier chaining.** Should multi-session campaigns chain into one
  trajectory with lineage links (proposed), or stay per-session with
  cross-references?
- **Q8 — Ordering.** This harvest versus the already-queued lived-episode
  curriculum (next-step 2): does this import become the evidence base for
  that curriculum, or a parallel track?

## 7. Sequencing and spend

1. Jeff rules on this proposal (and Q1–Q8 after Codex/ChatGPT review).
2. Build + test the credential screen (§4) — hard gate.
3. Importer + raw snapshots + manifest; measured tier histogram.
4. Session compiler with explicit-evidence labels only.
5. Roundtrip/count verification; full suite; ledger closeout.

Spend estimate: local CPU only, no GPU, no cloud; the corpus is ~68 MB raw.
The dominant cost is engineering time on the credential screen and compiler.

— Kimmy / Kimi Code CLI / 2026-08-29
