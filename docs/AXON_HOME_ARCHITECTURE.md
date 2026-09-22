# Axon Home — Architecture (Phase 0)

> **Deployment status (2026-09-21):** This document remains the control-plane
> and operator-surface baseline. Its original cloud-authoritative deployment is
> superseded as the intended target by `PHONE_SOVEREIGN_AXON.md`. The verified
> organism still lives at `D:\Axon` today. Authority must not move until the
> phone-host acceptance gates in that document pass against an exact recovery
> copy. The Android APK is always a client; it never owns a parallel body.

**Status: WORKING ARCHITECTURE — product architecture, not ratified Axon doctrine. `docs/SOURCE_OF_TRUTH.md` remains the only locked doctrine.**

- **Date:** 2026-09-21
- **Author:** Kimmy (Kimi K3) + swarm, mission from Jeff
- **Repo basis:** Axepapag/Axon `origin/main` @ `843bbedd25347d367da1f979488a6ecb77128ab7` (2026-09-20)
- **Governing authority hierarchy** (per `docs/WORKING_CONTRACT.md` §0): 1. Jeff (the convener) → 2. `docs/SOURCE_OF_TRUTH.md` (locked doctrine) → 3. Roundtable RESOLUTION documents → 4. the Axon Home mission prompt → 5. implementer judgment. Where this document and the repository disagree, the repository wins; where this document and locked doctrine disagree, doctrine wins and this document must be corrected.

---

## 1. Purpose & product philosophy

Axon Home is a native Android APK that becomes Jeff's permanent operator console for the Axon organism. The product thesis, restated as architecture:

- **The phone is the permanent surface; everything else is disposable.** A cloud worker, a VM, a Kaggle kernel, even the current Windows host (`D:\Axon`) may vanish tomorrow. The expected response to losing a worker is "provision another one," never "we lost Axon."
- **Disposable compute.** Training and runtime execution happen on whatever provider is available (local PC, Kaggle, Colab, VM, TPU, future providers) behind provider-neutral adapter contracts (§7). No Axon identity or state may depend on one machine surviving.
- **Durable state is primary plus verified replicas.** The active Heart host owns the one canonical branch. Content-addressed, self-verifying recovery capsules (§9) carry the complete organism continuation boundary to at least one independent encrypted replica with per-member SHA-256 manifests, verifiable offline and restorable on a fresh host.
- **GitHub is the engineering home.** Code, doctrine, ledger, and Roundtable collaboration live in the repo; the app is a client of that repo, not a parallel record.
- **The APK is not a compute node or canonical body.** In the phone-sovereign target, the real Python organism runs in a separate phone-local Linux service and the APK observes, commands, and verifies it over loopback. Heavy neural computation may remain on disposable workers.

This document is Phase 0 of the mission sequence (§18). It designs boundaries for the eventual system — dozens of heterogeneous Cores, terabytes of Dormant material, multiple training workers, several engineering agents — while the delivered slice stays small and honest. Everything described here that does not exist yet is labeled **designed, not built**.

---

## 2. Trust & authority boundaries

These boundaries are non-negotiable. The app must not accidentally redesign Axon's cognition or become a second authority.

**Central invariant (mission brief, verbatim):**

> **Complexity belongs behind the authority boundary, not inside the freedom of thought. Cores propose language; Heart owns state.**

### 2.1 Canonical authority model (from the repo, not negotiable)

| Authority | Owns | Repo anchor |
|---|---|---|
| **Heart** | Sole canonical writer of the Shared Field. The only commit path is `HeartTransactionBoundary` (`runtime/heart/transaction.py`, `axon-heart-commit-v1`), reached through the permanent single-writer process `HeartHost` (`runtime/heart/host.py`, entry point `scripts/run_axon_heart.py`). | `runtime/heart/` |
| **Trainer** | Sole parameter authority. All parameter mutation flows through `TrainerOrgan` (`runtime/trainer/organ.py`) under the `TrainerWriterLease`; gates (preflight authorization, execution guards, supervisory recompute, promotion) are locked doctrine and must never be weakened. | `runtime/trainer/` |
| **Cores** | Propose ordinary-language FIRST/REFINED proposals and a rotating consolidator's tagged FINAL verdict. Cores never directly mutate the canonical Shared Field; only `scratch` and `response_draft` are core-writable regions, and even those only via Heart-validated deltas. | `runtime/heart/circulation.py`, `runtime/field/schema.py` |

### 2.2 Rules the app must preserve

- Cores reason in ordinary language. FIRST and REFINED are language. FINAL is tagged desired-state language (`#regionTag#` verdicts, `axon-tagged-final-verdict-v2`). Heart derives internal typed mutations and commits atomically.
- EOS is ordinary language termination, never permission to think or act.
- Never reintroduce DELTA/NO_OP/ABSTAIN cognition or learned operation/region/address/termination control anatomy (archived at `archive/legacy_typed_reasoning_20260919/`; the 2026-09-19 English-proposal correction supersedes it).
- Keep strict infrastructure outside the freedom of cognition. The app exposes controls *around* Heart, Trainer, runtime, storage, Rails, masks, scheduling — it must not silently move any of those controls into Core cognition.

### 2.3 The app and Control Plane are clients, never authorities

- **The app never mutates canonical state except through existing governed seams:**
  - Canonical field mutation → `HeartTransactionBoundary` via `HeartHost` (and only via Heart's own valve/ingress/circulation paths).
  - Mask policy → `HeartHost.set_region_mask_policy` / `HeartRegionMaskController` (`runtime/heart/masks.py`, durable `axon-heart-region-masks-v3`). Masks are a *derived view*, not canonical state — changing them advances a mask revision and produces a new `view_id`, never a field rewrite.
  - Training → `TrainerOrgan` command boundary (`TrainerOrganCommand`, fail-closed).
- The **Control Plane governs operations/infrastructure** (machines, jobs, artifacts, streaming). **Heart governs organism state.** Those are different authority domains; the Control Plane never commits to the Shared Field itself.
- The app never writes to the Engineer's Ledger directly (§14); never writes to `State/` files directly; never fabricates activity.
### 2.4 Mutation authority matrix

Quick reference for every mutation surface the app touches:

| Surface | Canonical? | Only legal mutation path | App/Control Plane role |
|---|---|---|---|
| Shared Field content | Yes | `HeartTransactionBoundary` via Heart circulation/ingress valves | Read + inject user input via governed ingress; never write |
| Region mask policy | No (derived view) | `HeartRegionMaskController` / `HeartHost.set_region_mask_policy` | Propose → confirm → apply (§5.5, §10.F) |
| Parameters / optimizer / lineage | Yes (Trainer domain) | `TrainerOrganCommand` under `TrainerWriterLease` + gates | Dispatch commands; render `UNAVAILABLE` honestly |
| Soul state | Yes (private per Core) | Heart/Trainer transitions with receipts (`apply_soul_transition`, generation+1) | Read-only metadata |
| Dormant corpus | Yes (Dormant authority) | Curator/dormant store machinery; derived index is disposable | Read-only; never edit authoritative JSONL |
| Capacity budgets | Locked doctrine | Ratified change to `configs/source_of_truth/capacity_policy.json` only | Display only |
| Engineer's Ledger | Canonical history | `scripts/append_engineers_ledger_event.py` by a participating agent | Read-only (§14) |
| Machines / jobs / artifacts / audit | Operational (non-canonical) | Control Plane | Full CRUD within auth + audit |

- **UI rule:** a button that is not wired to a real governed seam must say "Not implemented." No fake operations, ever.

---

## 3. Current Axon integration surface (as of `843bbedd`)

### 3.1 What exists today

| Surface | Mechanism | Exact anchor |
|---|---|---|
| Runtime control | In-process Python object `HeartHost`; production CLI `scripts/run_axon_heart.py` (`--state-root`, `--mask REGION=PERCENT`, `--user`, `--once`) | `runtime/heart/host.py` |
| Canonical state | JSON files under the single State root (`D:\Axon\State` on the Windows host; `State/*` gitignored by design) | `runtime/field/state_branch.py` (`DEFAULT_STATE_ROOT`) |
| Training control | `TrainerOrgan` in-process command boundary; only `status` is wired (`scripts/axon_trainer.py status`); all mutation kinds return `TrainerCommandStatus.UNAVAILABLE` — fail-closed, no shell fallback | `runtime/trainer/organ.py` |
| Training telemetry | `AXON_PROGRESS` single-line JSON events on stdout + atomic `current.json` (`axon-training-progress-event-v1`) | `runtime/trainer/progress.py` |
| Health journals | `health.jsonl` / `health_latest.json` (`axon-heart-health-v2`), written every beat, success or failure | `runtime/heart/health.py` |
| Remote rail transport | Exact categorical wire codec `axon-remote-attended-d64-v1` inside HMAC-SHA256 sealed envelopes `axon-rail-auth-envelope-v1` (secret via `AXON_RAIL_SECRET_PATH`/`AXON_RAIL_SECRET`) | `runtime/heart/remote_rail.py`, `rail_auth.py` |
| Cloud training machinery | Provider-neutral job/packet records, deterministic tar.gz bundles, quarantine-on-mismatch verification | `runtime/trainer/cloud_jobs.py`, `cloud_bundle.py` |
| Demo web server | `scripts/demo_organ_server.py` (69 KB), 127.0.0.1:9201 — a 2026-09-10 demo server, **not** a control API | `scripts/` |

### 3.2 What does NOT exist today

- **No network control API.** Heart is a Python object; nothing serves HTTP/WebSocket for control or observation. `fastapi`, `uvicorn`, `websockets` are declared in `pyproject.toml` but essentially unused in active code.
- **No scheduler, no job reconciler.** The ledger's "loop is open" finding (2026-09-17) stands: 34 local job records, 10 stuck `submitted` with `provider_status: null`, no reconciliation loop.
- **No Google Drive / S3 / GCS integration** (zero code). **No active SSH automation.**
- **No mid-run sync ever proven** (`AXON_KAGGLE_SYNC` secret never attached; `SyncCredentialsMissing` receipts). End-of-run bundle fetch is the only working telemetry return path.

### 3.3 Broken `.bat` pointers (superseded by this product)

| Root file | Points at | State on current main |
|---|---|---|
| `AXON_KAGGLE.bat` | `scripts\axon_kaggle_control.ps1` | **Broken** — script archived to `archive/legacy_typed_reasoning_20260919/scripts/` on 2026-09-19 |
| `MONITOR_AXON_KAGGLE.bat` | `scripts\axon_kaggle_monitor.ps1` | **Broken** — archived |
| `AXON_TRAINING_WATCH.bat` | `scripts\axon_training_watch.py` | **Broken** — archived |
| `TRAIN_AXON.bat` | `scripts\axon_trainer.py status --state-root State` | **Works** (read-only status; mutations fail closed) |

`docs/KAGGLE_TRAINING_GUIDE.md` and the README badge describe the pre-2026-09-19 layout; trust the archived tree, not those docs, for Kaggle CLI mechanics. Axon Home supersedes these desktop shortcuts (§10).

### 3.3a Tick pipeline quick reference (as implemented)

The screens in §10.B/C and the events in §6 follow this exact pipeline (`HeartHost.heartbeat()` over `BeatCoordinator`):

1. **Identity reserve** — `HeartIdentityStore.next_heartbeat()/next_tick()` (durable, gap-tolerant, never reused).
2. **Ingress (between ticks only)** — spool front → valve final gate → `FieldDelta` committed under `AuthorityGrant.ingress(channel)` → autobiography deposit → spool ack.
3. **Dormant recall** — attended-region query → `DormantEvidenceBridge` retrieve + relevance audit → whole-region CORTEX replacement under `AuthorityGrant.dormant_valve()` with full valve provenance.
4. **Freeze** — `D64FieldCompiler.compile(field, region_masks)` → masked roundtrip verification → semantic surface → `FrozenTickImage` (view_id derives from masks; only d_model=64 permitted).
5. **Circulation** — FIRST pass → barrier → render first workspace → REFINED pass → barrier → round-robin consolidator (`participants[(tick_sequence-1) % len(participants)]`) → `TechnicalFinalVerdict` required.
6. **Materialize + commit** — verdict → typed `FieldDelta` → boundary validation → `apply_delta` → branch commit → tick consumed (a second commit raises `FinalCommitAlreadyMadeError`); consolidator Soul finalized with `commit_binding`.
7. **Episode deposit** — `axon-runtime-reasoning-episode-v3` autobiography record with full circulation dict.
8. **Health write** — every beat, success or failure (`axon-heart-health-v2`).

### 3.4 State-root files the Control Plane will read

All paths relative to the State root. All canonical JSON, content-addressed by SHA-256 where IDs exist.

| Path | Schema | Feeds |
|---|---|---|
| `active/branches/active/HEAD.json` | `axon-canonical-state-branch-head-v1` | Home, Shared Field HEAD |
| `active/branches/active/journal.jsonl` | `axon-canonical-state-branch-event-v1` | Tick history, commit provenance |
| `active/branches/active/snapshots/<field_id>.json` | `shared-field-v4` | Shared Field explorer, tick inspector |
| `active/branches/active/deltas/<delta_id>.json` | `shared-field-delta-v1` | Tick diffs, mutation provenance |
| `active/heart/health.jsonl`, `health_latest.json` | `axon-heart-health-v2` | Home heartbeat, Heart status |
| `active/heart/identity.json` | `axon-heart-identity-v2` | Epoch/heartbeat/tick sequences |
| `active/heart/region_masks.json` | `axon-heart-region-masks-v3` | Mask/attention panel |
| `active/heart/ingress/{ingress.jsonl, ack_cursor.json, quarantine.jsonl, rejections.jsonl}` | — | Ingress queues, alerts |
| `active/souls/<core_id>/branches/live/{HEAD.json, journal.jsonl, snapshots/, transitions/, prepared/, receipts/}` | `axon-private-soul-*-v1` family | Soul Observatory |
| `dormant/experience_v1/imports/...` | incl. `axon-runtime-reasoning-episode-v3` | Dormant browser, episode replay |
| `dormant/corpus_manifest.json` + `.derived/evidence_v1/index.sqlite3.manifest.json` | `axon_recovered_corpus_manifest`, `axon-dormant-evidence-index-manifest-v1` | Dormant status |
| `training/trainer/latest_*.json` (lifecycle, step, microstep, telemetry, learning_policy) | `axon-trainer-*` family | Trainer Control Center, Live View |
| `training/trainer/candidates/<module>/<gen>/accepted_steps/{pointer.json, checkpoint_done.json, bundles/}` | `axon-accepted-training-step-pointer-v1`, `axon-accepted-training-step-sentinel-v1`, `axon-accepted-reasoning-training-step-v1` | Lineage, recovery |
| `training/trainer/candidates/*/*/latest_checkpoint.json` | `axon-trainer-candidate-checkpoint-v2` | Checkpoints |
| `training/trainer/tranches/*.json`, `tranches/continuations/*.json` | `axon-trainer-resource-tranche-v1`, `axon-trainer-tranche-continuation-v1` | Tranche progress |
| `training/trainer/authority/lease.json` | writer lease | Trainer authority display |
| `training/reasoning/english/<candidate_generation>/report.json` | `axon-english-reasoning-smoke-trainer-v1` | Run reports |
| `axon_observability/trainer/{events.jsonl, current.json}`, `axon_observability/runner_events.jsonl`, `axon_job_result.json` | `axon-training-progress-event-v1` | Cloud job watch |
| `training/cloud/jobs/<job-id>/...` | `axon-cloud-bundle-manifest-v1`, `axon-cloud-bundle-verification-v1` | Cloud bundles, quarantine |

---

## 4. Mobile architecture

### 4.1 Technology stack (per mission §3)

- **Native Android, Kotlin, Jetpack Compose, Material 3, coroutines/Flow.** No giant WebView; WebViews only where unavoidable for third-party notebook/provider UIs (e.g., Colab deep links).
- **Dark-first theme** with system/light/dark support.
- **Single-activity + Navigation (Compose Navigation).**
- **Persistent navigation:** bottom nav with five primary destinations — **Home, Field, Cores, Trainer, More** — plus a navigation drawer and a global Command Palette ("Search commands, Cores, sessions, ticks, machines, artifacts, agents...") reaching all 20 areas (§10).
- **Phone + tablet responsive:** adaptive layouts (list-detail panes on tablets/foldables), landscape mode for graphs and terminals, pinch/zoom on rail and metric visualizations, collapsible raw-JSON views behind every visualization.

### 4.2 Module map

```
:core            Pure Kotlin. Domain model (Shared Field, ticks, Cores, Souls, training
                 lineage, recovery manifests) + the deterministic simulator (§11).
                 NO Android dependencies. JVM-testable.
:controlplane    Pure Kotlin. Control Plane client: endpoint DTOs, event-stream codecs
                 (§6), provider capability models (§7), schema-version validation,
                 contract-test fixtures. NO Android dependencies. JVM-testable.
:app             Android UI. Compose screens, ViewModels, Room/DataStore cache,
                 WorkManager sync, Keystore vault, notification glue.
```

Rationale: `:core` and `:controlplane` compile and test on the JVM in seconds; the simulator that powers demo/offline mode is the same code that validates the UI against the real event contract.

### 4.3 Data flow

- **Unidirectional data flow:** `ViewModel` exposes immutable `StateFlow<ScreenState>`; UI emits intents; repositories (backed by `:controlplane` clients or the simulator) are the single source of truth per screen.
- **Cache:** Room for structured mirrors (tick index, machine registry, manifests, audit history); DataStore for preferences and endpoint profiles. Cache is a *mirror*, never an authority (§12).
- **Background sync:** WorkManager periodic + expedited jobs for health/HEAD polling and event-stream reconnect; respects connectivity and battery constraints.
- **State labeling everywhere:** every screen region carries a `DataFreshness` enum — `LIVE`, `CACHED`, `STALE`, `DISCONNECTED` — rendered as a status chip. Cached state is never presented as current. Simulation adds a fifth, visually unmistakable state: `SIMULATION` (§11).

---

## 5. Control Plane architecture

### 5.1 What it is

The **Axon Control Plane** is a headless, provider-neutral service that the APK talks to. The APK never needs direct filesystem access to every remote worker.

- **Implementation (designed, not built):** Python **FastAPI** service living **in-repo** (e.g., `runtime/controlplane/` — subject to `tests/test_day_zero_hygiene.py` pinned-surface review before landing; see Risks §17). All dependencies are already declared in `pyproject.toml` (`fastapi>=0.115`, `starlette>=1.3`, `uvicorn[standard]>=0.30`, `websockets>=13`); no new runtime dependency is required.
- **Why separate from Heart:** Heart governs organism state (canonical commits). The Control Plane governs operations/infrastructure (machines, jobs, artifacts, streaming, auth). It is a **client of the governed seams** — it reads State-root JSON, dispatches `TrainerOrganCommand`s, and calls `HeartHost` methods — it never becomes a second writer. Per mission §12: "Heart governs organism state. The Control Plane governs operations/infrastructure. Those are not the same authority domain."

### 5.2 Responsibilities (mission §12)

Authenticated mobile API; machine registry; worker heartbeats; runtime status; Trainer control (via `TrainerOrgan`); artifact catalog; recovery capsule catalog; Shared Field snapshot streaming; tick event streaming; logs; provider orchestration; agent routing; command audit.

### 5.3 Deployment topology

- Runs **beside the Heart host** on the machine that owns the State root (initially the Windows PC, later any worker).
- **Binds localhost by default.** Remote access via SSH tunnel / Tailscale-style overlay, or explicit bind config.
- **NEVER publicly exposed without explicit config + auth.** Per the mission working rules: "Do not silently expose the Control Plane publicly."
- Authentication: bearer token minted during secure onboarding (§8); all mutating endpoints require it; audit every command (§14).

### 5.4 Protocol

- **HTTPS REST** for commands/resources; **WebSocket (primary) / SSE (fallback)** for streams.
- **Versioned JSON schemas** on every payload; server rejects unknown schema versions with a typed error.
- **Optional SSH transport** for bootstrap/emergency administration (first contact with a fresh worker before the Control Plane is running).
- gRPC is a future option if profiling justifies it; not now.

### 5.6 Versioning & error model

- Every request/response schema carries an explicit version string; the server rejects unknown versions with `426`-style typed errors listing supported versions. Client (`:controlplane`) ships a compatibility matrix per API version; contract tests pin it.
- Error envelope: `{schema, error{code, message, retryable, upstream_status}}`. Governed failures pass through verbatim — e.g. Trainer `UNAVAILABLE`, `REJECTED`, stale-delta and sealed-region errors — so the operator always sees the organism's real answer, not a UI-rewritten one.
- All mutating endpoints are idempotent via `Idempotency-Key`; retries never double-apply. Commands are audited before and after execution (§14).

### 5.5 Endpoint map (v1) — designed, not built

Base: `/v1`. All responses carry a `schema` field; all mutating endpoints accept an `Idempotency-Key` header and produce an audit entry.

| Endpoint | Method | Maps to / notes |
|---|---|---|
| `/v1/health` | GET | Control Plane liveness + Heart health summary (`axon-heart-health-v2` distilled) |
| `/v1/field/head` | GET | `active/branches/active/HEAD.json` + `health_latest.json` |
| `/v1/field/snapshot/{fieldId}` | GET | `snapshots/<field_id>.json` (`shared-field-v4`) + spans |
| `/v1/field/ticks/{n}` | GET | Last *n* journal entries (`axon-canonical-state-branch-event-v1`) with deltas |
| `/v1/masks` | GET | `region_masks.json` (`axon-heart-region-masks-v3`) |
| `/v1/masks` | PUT | **Confirm semantics:** PUT returns a *proposed* new mask state with a `confirm_token`; a second PUT with `{confirm_token, confirmed: true}` applies via `HeartHost.set_region_mask_policy`. Never mutates on first touch. Recorded in audit. |
| `/v1/cores` | GET | `CoreRegistry` descriptors + bindings (status, architecture_id, parameter_generation, soul head) |
| `/v1/souls/{coreId}` | GET | Soul branch HEAD, journal tail, generations, receipt chain (`axon-private-soul-*` metadata only — payloads are opaque) |
| `/v1/trainer/status` | GET | `inspect_trainer_state` (`axon-trainer-inspection-v1`) |
| `/v1/trainer/commands` | POST | Body `{kind, args, requested_by}` → `TrainerOrganCommand`. Kinds: `status, inventory, preflight, configure, start, pause, resume, evaluate, export_cloud_packet, import_cloud_result, compare, request_promotion, rollback`. **Fail-closed passthrough:** if the organ returns `UNAVAILABLE` ("no governed handler; no action was taken"), the API returns `409` with that status verbatim — the app must render it as "Not implemented," never as success. |
| `/v1/compute/workers` | GET/POST | Machine registry: provider, endpoint, capabilities, last heartbeat, active job |
| `/v1/storage/...` | GET/POST | Storage profiles, artifact catalog, upload/download ticket minting, hash verification |
| `/v1/capsules` | GET/POST | Recovery capsule catalog, export request, restore-verification reports (§9) |
| `/v1/audit` | GET | Operational audit log (§14) |
| `/v1/agents/roundtable` | GET/POST | Agent registry, routed messages (provider adapters §7); never carries secrets |
| `/v1/events` | WS (SSE fallback) | Typed event stream (§6) with seq-gapped reconnect |

---

## 6. Typed event stream (v1) — designed, not built

The app never polls dozens of files; the Control Plane derives a typed, versioned, replayable event stream from State-root journals (journal.jsonl, health.jsonl, AXON_PROGRESS events, ledger metadata) and pushes it to clients.

### 6.1 Envelope

```json
{
  "schema": "axon-home-event-v1",
  "event_id": "evt-<ulid>",
  "seq": 10234,
  "timestamp": "2026-09-21T00:00:00.000000Z",
  "type": "FieldCommitted",
  "payload": { "...": "per-type payload, schema-referenced" }
}
```

- `seq` is a monotonic per-stream sequence. On reconnect the client sends `last_seq`; the server replays from the retained ring buffer. If `last_seq` is older than retention, the server responds `SEQ_GAP` and the client re-syncs via REST snapshot before resuming the stream. A visible `seq` gap is never silently skipped.
- Payloads reference existing Axon schemas rather than inventing parallel ones.

### 6.2 Event types and payload schema references

| # | Event | Payload schema ref (existing repo schema) |
|---|---|---|
| 1 | `TickStarted` | `axon-heart-tick-identity-v1` |
| 2 | `CoreFirstReturned` | `axon-english-proposal-v1` + participant record |
| 3 | `FirstBarrierComplete` | `axon-heart-english-proposal-workspace-v2` ref |
| 4 | `CoreRefinedReturned` | `axon-english-proposal-v1` |
| 5 | `RefinedBarrierComplete` | `axon-heart-english-proposal-workspace-v2` ref |
| 6 | `ConsolidatorSelected` | core_id + tick_sequence (round-robin derivation) |
| 7 | `FinalReturned` | `axon-tagged-final-verdict-v2` |
| 8 | `HeartValidationStarted` | `axon-heart-frozen-tick-image-v2` ref |
| 9 | `FieldCommitted` | `axon-heart-commit-v1` + `axon-canonical-state-branch-event-v1` |
| 10 | `SoulTransitionAccepted` | `axon-private-soul-commit-receipt-v1` |
| 11 | `TrainerStepAccepted` | `axon-accepted-reasoning-training-step-v1` + pointer `axon-accepted-training-step-pointer-v1` |
| 12 | `CheckpointWritten` | `axon-trainer-candidate-checkpoint-v2` (record, not payload) |
| 13 | `HeldoutEvaluationCompleted` | evaluation record id + gate requirements result |
| 14 | `ComputeWorkerConnected` | worker registry record + capabilities |
| 15 | `ComputeWorkerLost` | worker id + last heartbeat + active job |
| 16 | `RecoveryCapsulePublished` | capsule manifest (`axon-cloud-bundle-manifest-v1`-compatible) |
| 17 | `AgentMessageReceived` | Roundtable message ref (no secrets) |
| 18 | `AlertRaised` | severity, source, message, related event ids |

Retention: ring buffer sized for replay of a phone's typical offline window; full history remains in the canonical journals, which the app can always re-read via REST.

---

## 7. Provider adapter contracts

Provider logic never lives in UI screens. Each adapter implements a typed contract and **advertises capabilities**; the UI renders controls **only** from advertised capabilities. A provider that cannot do something simply does not show the control.

### 7.1 Contracts

| Contract | Responsibility | Capability advertisement (examples) |
|---|---|---|
| `ComputeProvider` | Provision/connect/launch/stop/destroy workers and jobs | `supportsLaunch, supportsStop, supportsStatus, supportsLogs, supportsGpuMetrics, supportsProvision, supportsDestroy` |
| `StorageProvider` | Browse/upload/download/mirror/hash-verify artifacts and capsules | `supportsBrowse, supportsUpload, supportsDownload, supportsHashVerify, supportsArtifacts` |
| `AgentProvider` | Route engineering-agent conversations | `supportsStreaming, supportsTools, supportsSystemPrompt, supportsModelList` |
| `GitProvider` | Repo status/diff/pull/push/PR links | `supportsPush, supportsPullRequest, supportsDiff, supportsCommit` |
| `AxonRuntimeProvider` | Heart runtime attach (local in-process, future remote rail) | `supportsPause, supportsMaskControl, supportsIngress, supportsLiveStream` |
| `TrainerProvider` | Trainer organ transport (local in-process first) | per-`TrainerCommandKind` availability map mirroring the fail-closed registry |
| `NotebookProvider` | Kaggle/Jupyter/Colab workspaces | `supportsLaunch, supportsStop, supportsStatus, supportsLogs, supportsFiles, supportsTerminal, supportsGpuMetrics, supportsArtifacts` |

Capability models live in `:controlplane` as serializable Kotlin data classes; contract tests assert every adapter's advertisement parses and every UI control maps to a declared capability.

### 7.2 Initial adapters (mission §11)

- **Compute/notebooks:** generic SSH Linux worker; Kaggle (where its API genuinely supports the action); Jupyter endpoint; Google Colab via deep-links/Drive workflow where API limitations prevent direct control — the adapter advertises reduced capabilities rather than faking control.
- **Storage:** Google Drive (first obvious backend), SFTP, generic HTTP artifact endpoint.
- **Git:** GitHub.
- **Agents:** OpenAI-compatible endpoint adapter + generic REST abstraction; multiple agent-provider configurations, no hardcoded model brands.

If a provider does not expose a legitimate API for an action, the app must not pretend it does.

---

## 8. Security model

The app will hold GitHub tokens, SSH private keys, cloud provider credentials, API keys, storage credentials, and private Axon state. Rules:

- **Android Keystore-backed secret storage:** all vault keys are AES-GCM keys generated inside the Keystore (`KeyGenParameterSpec`, `PURPOSE_ENCRYPT/DECRYPT`, `setUserAuthenticationRequired(true)` where biometrics available); key material never leaves secure hardware.
- **Encrypted credential database:** secrets stored only as ciphertext (Keystore-wrapped DEKs per entry); per-provider scoping (a Kaggle token is never usable where a GitHub token is expected); least privilege by default.
- **Biometric / device-auth unlock** where available; **auto-lock** on inactivity and screen-off; configurable lock timeout.
- **Never** commit secrets; never sync raw secrets to GitHub; never put secrets in logs, crash reports, or Roundtable transcripts; mask secrets in UI after entry (show last-4 at most).
- **Clipboard minimization:** secrets are copied only on explicit action, flagged sensitive (`ClipDescription.EXTRA_IS_SENSITIVE`), never auto-copied.
- **SSH:** host fingerprint verification with visible TOFU prompt and pinned fingerprints per profile; private keys imported/generated into the vault only; keys never printed in logs.
- **TLS:** certificate errors are visible, explainable, and **never silently accepted**; no "accept all certs" toggle exists.
- **Remote session revocation:** per-device session list, one-tap revoke; revoking a Control Plane session invalidates its bearer token server-side.
- **Auditable destructive actions:** every destructive or billable action produces an audit entry (§14) and a confirmation screen (cost authority: provider + expected cost shown before creating billable infrastructure).
- **Explicit export/import policy:** vault export is encrypted, passphrase-protected, off by default, and warned; non-secret configuration export is separate (§10.T).
- **Do not invent your own cryptography.** Keystore + platform `Cipher` (AES-GCM) + vetted TLS/SSH libraries only.

### Secrets Vault screen

Sections: GitHub, cloud providers, storage providers, SSH keys, agent API keys, Control Plane endpoints. Each entry: label, scope, created/last-used, masked value, actions (reveal-with-biometric, rotate, revoke sessions, delete with confirm). Empty states explain scope and least privilege. A global "lock now" lives in the Command Palette.

---

## 9. Recovery capsule architecture

### 9.1 What a capsule is

Every important accepted boundary is exportable into a **self-verifying recovery capsule**: the minimum complete state required to continue that organism on any fresh worker. The app makes it impossible to casually confuse three different things:

1. **Code backup** — a git revision. Recoverable from GitHub alone.
2. **Checkpoint backup** — parameter payload + record. Restores weights, not the organism.
3. **Full organism recovery capsule** — the complete governed continuation boundary below.

### 9.2 Capsule contents (mission §15 mapped to real repo records)

| Component | Repo record / schema |
|---|---|
| Model checkpoint record | `axon-trainer-candidate-checkpoint-v2` (module, base/candidate generation, plan_id, authorization_id, learning_policy_id, step, artifact_relpath/sha256/bytes, optimizer/gradient/scaler inclusion flags, previous_checkpoint_id, checkpoint_id) |
| Checkpoint payload | `axon-trainer-candidate-checkpoint-payload-v2` (`torch.save` dict: descriptor, learning_policy, parameter_manifest, module/optimizer/scaler/gradient state dicts), content-addressed `checkpoints/<sha256>.pt` |
| Optimizer state | inside payload (`optimizer_state_dict`, `optimizer_included` required when step>0) |
| Private Soul state | `SoulSnapshot` `axon-private-soul-snapshot-v1` — 4 layers (`axon-private-soul-layer-v1`, temperatures HOT/WARM/COLD/DEEP_COLD, opaque payloads, sha256 per layer) |
| Accepted bundle | `axon-accepted-reasoning-training-step-v1` + pointer `axon-accepted-training-step-pointer-v1` + sentinel `axon-accepted-training-step-sentinel-v1` (crash-safe publication order: sentinel → pointer → sentinel; pointer rebuilt from immutable bundles on recovery) |
| Resource tranche + continuation | `axon-trainer-resource-tranche-v1` + `axon-trainer-tranche-continuation-v1` (parent bundle/checkpoint/optimizer receipt/soul/global step) |
| Learning policy | `axon-trainer-learning-policy-v1` (`GovernedLearningPolicy`) |
| Mutation plan | `axon-parameter-mutation-plan-v2` |
| Heldout evaluation | evaluation record (`training/reasoning/english/<gen>/evaluations/<evaluation_id>.json`) |
| Run report | `axon-english-reasoning-smoke-trainer-v1` report.json |
| Architecture identity | `architecture_id` (`"living-d64-english-" + sha256(config)[:24]`) from `LivingReasoningCoreConfig` |
| Git commit | HEAD sha at export |
| **Manifest** | per-member `{path: {sha256, bytes}}` + archive digest, **written LAST** (precedent: `runtime/trainer/cloud_bundle.py` detached `axon-cloud-bundle-manifest-v1`; dormant index build: temp → `os.replace` → manifest sidecar last) |

### 9.3 Restore verification order (fail closed on any disagreement)

1. Verify **every SHA-256** (archive digest, then per-member, both directions of member-set equality).
2. Verify **architecture identity** (config hash → `architecture_id` match).
3. Verify **candidate generation** (base vs candidate distinct; lineage ids match).
4. Verify **optimizer policy** (learning-policy equality; an explicit objective transition may change *only* `objective_program_id` — per `runtime/trainer/execution.py` restore rules).
5. Verify **Soul/checkpoint pairing** (`before_soul_id`/`after_soul_id` chain, generation+1, receipt chain exact).
6. Verify **lineage** (plan_id equality — "checkpoint plan lineage mismatch"; tranche/continuation chain; pointer vs bundles).
7. **Fail closed:** any mismatch quarantines the whole capsule and reports; nothing half-verified reaches canonical State.

Precedent in repo: `verify_and_extract` for cloud bundles routes all mismatches to `jobs/<job-id>/quarantine/` with a `axon-cloud-bundle-verification-v1` report — "nothing half-verified reaches canonical outputs." Capsule restore adopts the same discipline. Note: mid-run sync checkpoints are **observation-only** in current doctrine — they never authorize continuation; only accepted-boundary state may seed a restore.

---

## 10. Screen map (mission areas A–T)

Honest stub policy everywhere: unimplemented controls render "Not implemented"; nothing pretends to have performed an operation. Data sources reference §3.4 paths and §5.5 endpoints. The stale `.bat` pointers (§3.3) are superseded: everything they did (or failed to do) is absorbed into these screens.

| Area | Screen | Primary data sources |
|---|---|---|
| A | **Home / Organism Overview** — heartbeat visualization (animation only from real state), online/offline/degraded, current tick, Heart status, field id/hash, active/dormant cores, trainer activity, lineage, latest checkpoint, Soul continuity, compute/storage/agents, alerts, last capsule, git branch/commit | `/v1/health`, `/v1/field/head`, `health_latest.json`, HEAD.json, trainer latest_*.json, machine registry; stream `/v1/events` |
| B | **Shared Field Explorer** — 13 regions, exact sizes, hashes, tick diffs, span provenance (which Core proposed what, consolidator FINAL, Heart delta, accept/reject), raw JSON | snapshots/<field_id>.json, deltas/<delta_id>.json, journal.jsonl; `/v1/field/*` |
| C | **Tick Inspector / Time Machine** — freeze any tick: F_t, participants, FIRST/REFINED boards, consolidator, FINAL, parse, mutations, validation, F_t+1, soul receipts, timing, errors. Replay is immutable evidence, never live mutation | journal.jsonl + `axon-runtime-reasoning-episode-v3` records; `/v1/field/ticks/{n}` |
| D | **Core Observatory** — per-core profile (id, architecture, d_model=64, heads, layers, FFN, state tokens, params, architecture_id, parameter generation, soul id/generation, temperature status, role, consolidator eligibility, placement, training history, evaluations, Soul probes). Controls: activate/deactivate, pause, checkpoint, export, migrate, assign training, compare, restore — each wired only to governed seams or stubbed honestly | `/v1/cores`, `/v1/souls/{coreId}`, trainer inspection |
| E | **Rail Observatory** — D16 substrate, packing into D64 rails (4 lanes/row), page traversal, attended intervals, coverage manifests, Unicode transport receipts. Truthful-only rule: exact state, deterministic transport, learned representation, and private Soul state are visually distinguished; opaque tensors are never rendered as fake semantics | field compiler manifests (`axon-field-compiler-d64-v3`), semantic surface (`axon-d64-semantic-surface-v2`) |
| F | **Attention / Region Controls** — real mask sliders per region (tail_percent 0–100 = newest-suffix policy; all/none/last_n_spans), live preview of *proposed* view, diagnostic vs production distinction, confirm-before-apply (`/v1/masks` PUT two-step), revert to defaults, audit history | `region_masks.json` (`axon-heart-region-masks-v3`); `/v1/masks` |
| G | **Semantic Cortex / Surfacing Inspector** — what was surfaced toward the field: source, item, provenance, relevance metadata, target region, surfaced-vs-committed distinction (retrieved/surfaced/proposed/accepted/committed are not synonyms), consuming Core, referencing proposal | dormant recall provenance in deltas/journal; Cortext is reserved/inactive — screen says so |
| H | **Soul Observatory** — per-core Soul id, generation, 4 temperature layers, sizes, continuity receipts, inhale/exhale events, parameter-generation compatibility, causal probes, migrations. Structural/causal facts only — latent tensors are opaque and labeled as such | `active/souls/<core>/branches/live/*`, `axon-private-soul-*-v1` family |
| I | **Dormant** — corpus sources, provenance, grouping, ingestion state, curriculum eligibility, quarantine status, search, retrieved experiences, storage usage; remote-storage-backed (no Windows-path assumption) | corpus_manifest.json, evidence index manifest, experience_v1 imports |
| J | **Trainer Control Center** — lineages, candidate generation, accepted step, parameter/Soul generations, optimizer, lr, objective program, curriculum, manifests, tranche progress, losses, gradient norms, heldout metrics, checkpoint retention, worker/GPU, cost. Controls: preflight, start, pause-at-boundary, stop, emergency terminate, evaluate, resume, compare, promote (governed only), export capsule. Global red STOP | `/v1/trainer/status`, `/v1/trainer/commands`, tranches/, accepted_steps/ |
| K | **Training Live View** — step, lived experiences, Soul transitions, objective components, loss, grad L2, lr, GPU util/VRAM, CPU/RAM, elapsed, checkpoints, heldout evals, logs, free-running samples, latest accepted boundary; zoomable graphs | `AXON_PROGRESS` stream, `axon-parameter-telemetry-frame-v1`, `/v1/events` |
| L | **Compute Fleet** — provider-neutral cards (SSH hosts, Kaggle, Colab, VMs, GPUs, TPUs): provider, endpoint, machine, CPU/RAM, accelerator/VRAM, region, status, uptime, cost, storage, worker version, git commit, active job. Ops: provision/connect/bootstrap/sync/restore/launch/monitor/stop/destroy — all behind capability flags + cost-confirmation screen | `/v1/compute/workers`, provider adapters |
| M | **Notebook / Remote Workspace** — Kaggle status, Colab links/status, Jupyter views, terminal, logs, files, GPU status, notebook URL; capability-driven controls only | `NotebookProvider` adapters |
| N | **Storage Center** — Drive/SFTP/S3-compatible/GCS/local adapters: browse, upload, download, mirror, hash verify, restore, capsule/checkpoint/Dormant archives, reports, logs | `/v1/storage/...`, `StorageProvider` |
| O | **Recovery Capsules** — export per §9.2, verify per §9.3, catalog, quarantine view; visually distinguishes code backup / checkpoint backup / full capsule | `/v1/capsules`, manifests |
| P | **Git / Repository** — status, branch, commit, ahead/behind, changed files, diff viewer, recent commits, pull/fetch, commit/push, PR links. No force push; destructive ops confirmed. Flow: verify → review screen → commit/push | `GitProvider` (GitHub adapter) |
| Q | **Engineers / Roundtable** — participants, online/offline, provider, model, capabilities, active task, messages, proposals, reviews, ledger events; initiate agent conversations via adapters | `/v1/agents/roundtable`, repo roundtable/ mirror |
| R | **Agent Studio** — agent configs: name, provider, base URL, vault key *reference* (never raw key), model, system instructions, tools, repo/compute/storage permissions, timeout, context limits, temperature, avatar/color, role, active. OpenAI-compatible first, generic abstraction; keys never exposed to prompts | local config + vault refs; `AgentProvider` |
| S | **SSH / Terminal** — connection profiles, key import/generation, passphrases, host fingerprints, terminal sessions, history, SFTP where practical, port forwarding later | vault-backed keys; SSH adapter |
| T | **Configuration** — Control Plane/runtime/trainer endpoints, GitHub, Drive, providers, storage, compute, agents, SSH, refresh rates, retention, telemetry, themes, security, notifications; non-secret config export | DataStore profiles |

---

## 11. Simulation mode

A deterministic simulator lives in `:core` (pure Kotlin, no Android deps) so the APK is fully developable and testable with no live Axon server.

- **Simulates:** 4 heterogeneous Cores (varying architecture descriptors within the D64 contract), heartbeat ticks, full Shared Field transitions (FIRST → REFINED → consolidator FINAL → Heart commit), a training job with accepted steps/checkpoints/evaluations, GPU metrics, storage artifacts, agent activity, and injected failures (worker loss, seq gaps, quarantined bundles, stale cache).
- **Event-stream driven:** the simulator emits the exact §6 envelope (`axon-home-event-v1`), so `:app` consumes simulation through the same client contract as production. Contract tests run the same event fixtures against both.
- **Seeded:** a user-visible seed makes any session reproducible; seeds appear in bug reports.
- **Visually unmistakable:** a persistent `SIMULATION` banner, watermarked status chips, and a distinct theme accent. Simulated data can never masquerade as a live organism; the `DataFreshness` chip reads `SIMULATION`, not `LIVE`.

---

## 12. Offline & mirror model

- **Cached verified mirror on the phone:** Shared Field HEAD + recent tick snapshots/deltas, health history, trainer status, machine status, recovery manifests, agent/Roundtable history, configuration. Only hash-verified content enters the mirror; verification failures mark the cache entry invalid, never silently stale.
- **Metadata mirror vs full artifact mirror:** the UI always distinguishes the two. Large tensors/checkpoint payloads stay remote and **content-addressed** (`checkpoints/<sha256>.pt`); the phone holds manifests, records, and hashes — enough to verify, catalog, and request restore, not enough to fill storage.
- **Labels everywhere:** `LIVE` (streamed/just-fetched and verified), `CACHED` (verified mirror, timestamp shown), `STALE` (past freshness threshold, timestamp shown), `DISCONNECTED` (no transport; mirror read-only). Cached state is never presented as current.
- **Reconnect:** seq-gapped stream semantics (§6) + REST re-sync; mutations attempted while disconnected are queued locally as *drafts*, clearly labeled, and require re-confirmation on reconnect — never silently replayed.

---

## 13. Stop semantics

Four distinct stops, each honest about its boundary:

| Control | Semantics | Boundary / mechanism |
|---|---|---|
| **PAUSE AXON** | Finish/abort the current beat at the atomic runtime boundary and prevent new ticks | Heart host boundary (`BeatState` idle); today this is a host-level operation on the Heart machine — via Control Plane, designed not built |
| **STOP TRAINING** | Stop at a recoverable Trainer boundary, preserving full recoverability | Tranche boundary: today `CandidateOptimizationSession.pause(reason, checkpoint_id)` exists in-process only (requires committed gradient accumulation); a mid-segment interactive pause is a **future `TrainerOrgan` handler** — until wired, the organ returns `UNAVAILABLE` and the app shows exactly that |
| **STOP COMPUTE** | Terminate a selected worker/VM/provider job | Provider adapter `supportsStop`; only rendered when advertised |
| **EMERGENCY STOP** | Attempt to halt organism execution + training immediately | **Ack accounting:** the result screen lists every registered component with `confirmed stopped` / `unreachable` / `provably gone`. The app never displays "everything stopped" unless every component acknowledged or is provably gone. Never fakes successful cancellation |

---

## 14. Audit trail

- **Operational audit entry** (Control Plane-side, surfaced in-app): `who` (authenticated identity), `device`, `command`, `target`, `timestamp`, `previous_state`, `requested_state`, `result`, `error` (if any). Retained locally and on the Control Plane; exportable.
- **Relationship to the Engineer's Ledger:** the app audit is **not** the canonical ledger and never masquerades as it. No garbage events per UI tap — the ledger records *meaningful project turns*, not operations.
- **Ledger writes happen via the repo append script** (`scripts/append_engineers_ledger_event.py`, schema `axon-engineers-ledger-event-v1`, single-line JSON, fsync, uniqueness-checked), **not from the app.** Transport constraint (from recon): there is no remote ledger-ingestion endpoint; the canonical JSONL (~2.1 MB) exceeds the mission's push transport, and concurrent appends are hazardous ("always re-read the tail immediately before writing"). The app therefore surfaces ledger *read* views (rolling summary `roundtable/ENGINEERS_LEDGER.md`, canonical tail) and flags significant operational turns for a human/engineering agent to record through the governed script. A future Control Plane ledger-ingress endpoint is a designed-not-built item requiring single-writer discipline.

---

## 15. Data schema index

Every `axon-*-vN` schema the app consumes, with repo source and consumer. Strings verified against the snapshot (see Appendix B).

| Schema string | Source file (repo path) | Consumer (screen/module) |
|---|---|---|
| `shared-field-v4` | `runtime/field/schema.py` | B Field Explorer, C Tick Inspector, `:core` domain |
| `shared-field-delta-v1` | `runtime/field/delta.py` | B, C, event 9 |
| `axon-canonical-state-branch-head-v1` | `runtime/field/state_branch.py` | A Home, B, `/v1/field/head` |
| `axon-canonical-state-branch-event-v1` | `runtime/field/state_branch.py` | C Tick Inspector, audit provenance |
| `axon-field-compiler-d64-v3` | `runtime/field/compiler_d64.py` | E Rail Observatory |
| `axon-d64-semantic-surface-v2` | `runtime/field/semantic_d64.py` | E, G |
| `axon-heart-tick-identity-v1` | `runtime/heart/tick.py` | events 1/8, A |
| `axon-heart-frozen-tick-image-v2` | `runtime/heart/tick.py` | C, E, event 8 |
| `axon-heart-derived-view-v1` | `runtime/heart/tick.py` | F Masks |
| `axon-heart-commit-v1` | `runtime/heart/transaction.py` | C, event 9 |
| `axon-reasoning-circulation-v3` | `runtime/heart/circulation.py` | C Tick Inspector |
| `axon-english-proposal-v1` | `runtime/heart/english_reasoning.py` | B, C, events 2/4 |
| `axon-tagged-final-verdict-v2` | `runtime/heart/english_reasoning.py` | B, C, event 7 |
| `axon-reasoning-text-frame-v1` | `runtime/heart/reasoning_output.py` | C, K |
| `axon-heart-english-proposal-workspace-v2` | `runtime/heart/proposal_workspace.py` | C, events 3/5 |
| `axon-heart-turn-finalization-v1` / `axon-readable-turn-frame-v1` | `runtime/heart/turns.py` | B conversation view |
| `axon-heart-health-v2` | `runtime/heart/health.py` | A Home, `/v1/health` |
| `axon-heart-identity-v2` | `runtime/heart/identity.py` | A Home |
| `axon-heart-region-masks-v3` | `runtime/heart/masks.py` | F Masks, `/v1/masks` |
| `axon-remote-attended-d64-v1` | `runtime/heart/remote_rail.py` | future remote runtime adapter |
| `axon-rail-auth-envelope-v1` | `runtime/heart/rail_auth.py` | future remote runtime adapter |
| `axon-runtime-reasoning-episode-v3` | Dormant imports (`runtime/heart/autobiography.py` producer) | C Time Machine, I Dormant |
| `axon-private-soul-layer-v1`, `axon-private-soul-snapshot-v1`, `axon-private-soul-transition-v1`, `axon-private-soul-promotion-v1`, `axon-private-soul-commit-receipt-v1`, `axon-private-soul-head-v1`, `axon-private-soul-branch-v1`, `axon-private-soul-branch-event-v1`, `axon-private-soul-prepared-v1` | `runtime/soul/contracts.py`, `runtime/soul/store.py` | H Soul Observatory, `/v1/souls/{coreId}`, event 10 |
| `axon-d64-recurrent-soul-codec-v1` (media `application/x-axon-d64-recurrent-state`) | `training/living_reasoning_d64.py` | H (opacity labeling), capsule verify |
| `axon-parameter-mutation-plan-v2` | `runtime/trainer/contracts.py` | J, O capsules |
| `axon-trainer-learning-policy-v1` | `runtime/trainer/learning.py` | J, O |
| `axon-trainer-resource-tranche-v1`, `axon-trainer-tranche-continuation-v1` | `runtime/trainer/tranche.py` | J, K, O |
| `axon-accepted-reasoning-training-step-v1`, `axon-accepted-training-step-pointer-v1`, `axon-accepted-training-step-sentinel-v1`, `axon-accepted-step-pointer-recovery-evidence-v1` | `runtime/trainer/step_bundle.py` | J, K, O, event 11 |
| `axon-trainer-candidate-checkpoint-v2` (+ payload `axon-trainer-candidate-checkpoint-payload-v2`) | `runtime/trainer/lifecycle.py`, store | J, O, event 12 |
| `axon-trainer-optimization-step-v2`, `axon-trainer-learning-microstep-v1` | `runtime/trainer/lifecycle.py` | K Live View |
| `axon-parameter-telemetry-frame-v1` | `runtime/trainer/telemetry.py` | K |
| `axon-trainer-candidate-lifecycle-event-v1` | `runtime/trainer/lifecycle.py` | J |
| `axon-trainer-inspection-v1`, `axon-trainer-organ-status-summary-v1` | `runtime/trainer/inspection.py`, `organ.py` | J, `/v1/trainer/status` |
| `axon-trainer-worker-evidence-v1` | `runtime/trainer/supervisory_gates.py` | J gate evidence view |
| `axon-training-progress-event-v1` | `runtime/trainer/progress.py` | K, event derivation |
| `axon-english-reasoning-smoke-trainer-v1` | `scripts/train_living_reasoning_smoke.py` (report) | J reports |
| `axon-cloud-bundle-manifest-v1`, `axon-cloud-bundle-verification-v1` | `runtime/trainer/cloud_bundle.py` | N, O, event 16 |
| `axon-dormant-evidence-index-v1`, `axon-dormant-evidence-index-manifest-v1` | `runtime/dormant/evidence_bridge.py` / derived index | I Dormant |
| `axon-engineers-ledger-event-v1` | `scripts/append_engineers_ledger_event.py` | Q Roundtable (read-only), §14 |
| `axon-kimi-roundtable-lock-v1`, `axon-kimi-roundtable-job-v1` | `scripts/run_kimi_roundtable.py` | Q agent status (read-only) |
| `axon-source-of-truth-capacity-policy-v1` | `configs/source_of_truth/capacity_policy.json` | T Configuration (budgets view) |
| `axon-home-event-v1` (new) | this document §6 | `:controlplane`, all streamed screens |

---

## 16. Build / CI / release

- **Gradle (Kotlin DSL)** multi-module build (`:core`, `:controlplane`, `:app` per §4.2). Reproducible: locked dependency versions (`gradle/libs.versions.toml` + verification metadata), deterministic SDK/AGP pins, `org.gradle.caching` enabled.
- **Versioning in-app:** version name/code + git commit sha shown in Settings → About; commit sha stamped via build config from CI (`git rev-parse HEAD`).
- **GitHub Actions workflow `.github/workflows/axon-home.yml`** (note: the Axon repo currently has **no** `.github` directory — this is its first CI):
  - Job 1: JVM unit tests for `:core` and `:controlplane` (fast, no emulator).
  - Job 2: `assembleDebug`, lint, APK artifact upload (`app-debug.apk`) attached to the workflow run.
  - Instrumented/Compose UI tests on emulator are a later phase (runner cost), not Phase 1 CI.
- **Debug APK installable** directly from CI artifacts on Jeff's phone — no desktop required.
- **Release signing:** instructions in-repo (keystore generated locally, stored *outside* git, passwords in CI secrets; `assembleRelease` signed via `signingConfigs`; never commit keystores — consistent with the repo's credential gitignore block).
- **gradle-wrapper.jar note:** the mission's push transport cannot carry the binary wrapper jar; `gradle-wrapper.jar` is therefore **generated in CI** (`gradle wrapper` bootstrap step or a checked-in wrapper *script* + documented one-command local bootstrap). The bootstrap procedure is documented in the project README so any fresh checkout reproduces the build.
- The repo's `tests/test_day_zero_hygiene.py` pins the active Python surface; the Android tree (`android/`) is not a scanned Python package and `docs/` additions are not test-constrained — verified against recon A §6.

---

## 17. Risks & honest gaps

| # | Risk / gap | Consequence | Mitigation |
|---|---|---|---|
| 1 | **No network API exists today.** Heart/Trainer are in-process Python; nothing serves HTTP/WS | App v1 cannot talk to a live organism over the network | **App v1 = simulation + file-import mode** (import verified State snapshots/journals for read-only inspection). Real connection arrives with the Control Plane (Phase 2) |
| 2 | Colab has no legitimate control API; Kaggle API covers kernels/datasets but not arbitrary lifecycle; OAuth flows differ per provider | Some fleet controls will be deep-links/status-only | Capability advertisements (§7): UI shows only what is real |
| 3 | **Ledger append transport:** canonical JSONL ~2.1 MB exceeds the MCP push limit; no remote ingestion endpoint; concurrent-append hazard | App cannot write ledger events | §14: app reads ledger, flags turns for governed append by an engineering agent; future single-writer ingress endpoint |
| 4 | **Mid-run sync never proven** (`AXON_KAGGLE_SYNC` unattached); end-of-run bundle fetch is the only working telemetry return | Live cloud-training view may degrade to post-hoc | Design K to degrade honestly (STALE labels); prove sync before promising live cloud telemetry |
| 5 | **Job reconciliation loop open:** 34 local job records, 10 stuck `submitted`, no reconciler, no scheduler | Fleet/job status can drift from provider truth | Control Plane machine registry + periodic provider reconciliation (Phase 4); surface "last verified" timestamps |
| 6 | Broken `.bat` pointers and stale Kaggle docs describe archived tooling | Operator confusion on the Windows host | App supersedes them (§10); recommend a small repo hygiene commit flagging the stale docs |
| 7 | `tests/test_day_zero_hygiene.py` pins Python package file sets | Adding `runtime/controlplane/` Python modules will trip hygiene tests | Land Control Plane code with a deliberate, reviewed hygiene-test update in the same commit |
| 8 | Live State content (what exists at `D:\Axon\State` right now) is unknowable from the repo — `State/*` is gitignored by design | Screen designs assume record shapes, not live presence | `:controlplane` contract tests use repo fixtures; first connection does capability/content discovery |
| 9 | Soul payloads are doctrine-opaque | Temptation to over-visualize | Hard rule (E/H): structural/causal facts only; opaque payloads labeled opaque |

---

## 18. Phase plan

Mission §18 sequence, with this mission's delivered slice marked:

| Phase | Scope | Status |
|---|---|---|
| **Phase 0 — Architecture** | Inspect Axon; write `docs/AXON_HOME_ARCHITECTURE.md` (this document): trust boundaries, mobile architecture, control plane, provider interfaces, security, recovery, schemas, screens, event stream, offline model | **◀ THIS DELIVERABLE** |
| **Phase 1 — Executable shell** | Real APK: secure onboarding, endpoint profiles, Home, Shared Field, Cores, Trainer, Compute, Storage, Engineers, Settings, simulation backend | In scope for this mission's vertical slice (sibling deliverables: Android project, buildable APK, vault foundation, simulation mode, CI, contract tests) |
| **Phase 2 — Real Axon connection** | Control Plane adapter against current runtime/Trainer: real health, Core descriptors, Shared Field, ticks, Trainer status | Designed here (§5); not built |
| **Phase 3 — Recovery** | Google Drive-backed recovery capsules + restore validation | Designed here (§9); not built |
| **Phase 4 — Remote infrastructure** | SSH worker bootstrap, provider abstractions live | Contract designed (§7); not built |
| **Phase 5 — Collaboration** | Agent Studio / Roundtable interfaces | Screens mapped (§10 Q/R); not built |
| **Phase 6 — Deep observability** | Rails, masks, Cortex, detailed tick time-machine | Screens mapped (§10 E/F/G/C); not built |

---

## Appendix A — Recon sources

This architecture is grounded in four recon briefs produced from `Axepapag/Axon` `origin/main` @ **`843bbedd25347d367da1f979488a6ecb77128ab7`** (2026-09-20T11:45:28Z, "Ledger: audit Stage-0A step-24 behavior"):

1. `recon/A_governance_ledger.md` — governance docs, ledger protocol, commit conventions, repo-hygiene constraints.
2. `recon/B_cognition_runtime.md` — runtime tree, tick pipeline, Shared Field model, rails/substrate, masks, Cores, streamable outputs.
3. `recon/C_trainer_soul_dormant.md` — trainer control surface, gates, lineage model, checkpoint/capsule formats, Soul model, Dormant, gaps.
4. `recon/D_roundtable_ops.md` — Roundtable architecture, scripts catalog, operational topology, secrets handling, packaging, integration gaps.

Mission brief: `Axon_Home_Mission_for_Kimmy_K3.docx` (Jeff, 2026-09). Snapshot spot-verified at `axon-main/` (note: the snapshot carries no `.git` metadata; the HEAD sha is attested by recon A).

## Appendix B — Snapshot spot-checks performed

All checks run against the local snapshot; all recon claims verified accurate. **No contradictions found.**

| # | Claim checked | Result |
|---|---|---|
| 1 | 13 canonical regions in exact order (conversation_history → … → training_responses), `schema_version="shared-field-v4"`, `LOGICAL_REGION_IDS` ordinal map | ✅ `runtime/field/schema.py` lines 27–77 |
| 2 | `TrainerCommandKind` = exactly 13 kinds (status…rollback); fail-closed `UNAVAILABLE` status | ✅ `runtime/trainer/organ.py` lines 26–46 |
| 3 | Capacity policy values (global_items_per_beat=256, target_chars=32768, idle=30 s, recall_items=8, emission slice=512, rolling_checkpoint_count=3) | ✅ `configs/source_of_truth/capacity_policy.json` |
| 4 | FastAPI/uvicorn/websockets declared but unused; package set `curator*, runtime*, substrate*, training*` | ✅ `pyproject.toml` |
| 5 | Rail codec `axon-remote-attended-d64-v1` + HMAC envelope `axon-rail-auth-envelope-v1` | ✅ `remote_rail.py:27`, `rail_auth.py:29` |
| 6 | 20-slot valve plane: 4 CAPPED primitives (user_ingress, tool_ingress, advisor_ingress, dormant_recall) + 16 CLOSED | ✅ `runtime/heart/valve.py` lines 588–618 |
| 7 | Broken `.bat` pointers: `AXON_KAGGLE.bat` → `scripts\axon_kaggle_control.ps1`, absent from root `scripts/` (no kaggle/watch/tournament scripts there) | ✅ root `.bat` files read; `scripts/` listing |
| 8 | Soul schemas (`axon-private-soul-layer-v1`, `-snapshot-v1`) and temperature order HOT→WARM→COLD→DEEP_COLD | ✅ `runtime/soul/contracts.py` lines 18–40 |
| 9 | Accepted-step triple: `axon-accepted-reasoning-training-step-v1`, `-pointer-v1`, `-sentinel-v1`; checkpoint `axon-trainer-candidate-checkpoint-v2` | ✅ `step_bundle.py:28–30`, `lifecycle.py:15` |
| 10 | Branch/health/identity/mask schemas (`-branch-head-v1`, `-branch-event-v1`, `axon-heart-health-v2`, `axon-heart-identity-v2`, `axon-heart-region-masks-v3`) | ✅ `state_branch.py:41–42`, `health.py:14`, `identity.py:18`, `masks.py:32` |
| 11 | `AXON_PROGRESS` emitter + `axon-training-progress-event-v1` | ✅ `runtime/trainer/progress.py:20,128` |
| 12 | Living core anatomy locked: d_model=64, n_heads=1, n_layers=2, ffn_dim=131072, state_tokens=4, page_size=32 | ✅ `training/living_reasoning_d64.py` lines 86–107 |
| 13 | Cloud bundle manifest `axon-cloud-bundle-manifest-v1`; plan `axon-parameter-mutation-plan-v2`; policy `axon-trainer-learning-policy-v1`; tranche pair; telemetry `axon-parameter-telemetry-frame-v1`; inspection `axon-trainer-inspection-v1`; ledger `axon-engineers-ledger-event-v1` | ✅ `cloud_bundle.py:34`, `contracts.py:23`, `learning.py:11`, `tranche.py:26–27`, `telemetry.py:12`, `inspection.py:11`, `append_engineers_ledger_event.py:11` |

*End of document.*
