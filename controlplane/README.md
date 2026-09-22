# Axon Control Plane — API Contract (v1)

**Status: DESIGNED, NOT BUILT.** This directory is the language-neutral contract for
the Axon Control Plane described in `docs/AXON_HOME_ARCHITECTURE.md` §5 (endpoint map),
§6 (event stream), §7 (provider contracts), §13 (stop semantics), and §14 (audit).
There is **no server implementation today**: Heart and Trainer are in-process Python
objects, and nothing in the Axon repo serves HTTP/WebSocket. This contract pins the
surface that a future implementation (Phase 2+) must satisfy, so the Android client
(`android/controlplane/`) and the server can be built independently and verified
against the same documents.

Grounded against `Axepapag/Axon` `origin/main` @
`843bbedd25347d367da1f979488a6ecb77128ab7` (2026-09-20). All `axon-*-vN` schema strings
referenced here are the exact strings emitted by the repo (see arch doc §15 /
Appendix B).

---

## 1. What the Control Plane is

The **Axon Control Plane** is a headless, provider-neutral **operations service** that
the Axon Home APK talks to. It governs operations and infrastructure: machine
registry, worker heartbeats, trainer command dispatch, artifact and capsule catalogs,
streaming, and audit.

**Authority boundaries (non-negotiable, arch §2/§5.1):**

- The Control Plane is a **client of the governed seams, never a second authority.**
  It reads State-root JSON files, dispatches `TrainerOrganCommand`s to
  `TrainerOrgan`, and calls `HeartHost` methods. It never commits to the Shared Field
  itself — the only canonical commit path is `HeartTransactionBoundary`
  (`axon-heart-commit-v1`) via `HeartHost`.
- **Heart governs organism state. The Control Plane governs operations/infrastructure.**
  Those are different authority domains.
- Governed failures pass through **verbatim**: a Trainer `UNAVAILABLE` ("no governed
  handler; no action was taken") is returned as `409` with the status intact — the
  app renders "Not implemented", never success.

**Deployment (arch §5.3):**

- Runs beside the Heart host on the machine that owns the State root.
- **Binds localhost (127.0.0.1) by default.** Remote access via SSH tunnel /
  Tailscale-style overlay, or explicit bind config. NEVER publicly exposed without
  explicit config + auth.

**Authentication (arch §5.3/§8):**

- Bearer token minted during secure device enrolment (onboarding); all mutating
  endpoints require it. TLS is required; certificate errors are visible and never
  silently accepted. Revoking a device session invalidates its bearer token
  server-side.

## 2. Layout

```
controlplane/
  README.md                          — this file
  openapi/axon-control-plane-v1.yaml — OpenAPI 3.0.3 document for the v1 REST surface
  schemas/*.json                     — JSON Schema (draft 2020-12), one file per message group
  examples/*.json                    — example messages conforming to the schemas
```

The same schema files are copied (byte-identical) into
`android/controlplane/src/main/resources/schemas/` so the Kotlin `SchemaCatalog`
can load them from the classpath. The Kotlin module's contract tests deserialize
every example in `examples/` into the client models, keeping all three artifacts
(contract, models, examples) in lockstep.

## 3. Schema versioning

- Every request/response payload carries an explicit `schema` string. New,
  incompatible shapes get a new `-vN` suffix; additive changes may stay within a
  version.
- The server rejects unknown schema versions with a typed error
  (`UNSUPPORTED_SCHEMA_VERSION`, HTTP 426) listing supported versions (arch §5.6).
- Where a payload mirrors an existing repo record, the `schema` property carries the
  **exact repo schema string** (e.g. `axon-heart-health-v2`,
  `shared-field-v4`, `axon-heart-region-masks-v3`,
  `axon-cloud-bundle-manifest-v1`). Control Plane-native payloads use
  `axon-controlplane-*-v1` strings; the event envelope is `axon-home-event-v1`.
- Every JSON Schema file carries `$id` of the form
  `https://axon.gliksbot.com/schemas/controlplane/v1/<name>.json` and uses
  `$defs`/`$ref` for shared shapes.

## 4. Error model

Every error response (arch §5.6):

```json
{"schema": "axon-controlplane-error-v1",
 "error": {"code": "TRAINER_COMMAND_UNAVAILABLE", "message": "...", "retryable": false, "upstream_status": "UNAVAILABLE"}}
```

Governed failures pass through verbatim (Trainer `UNAVAILABLE`/`REJECTED`,
stale-delta, sealed-region errors) so the operator sees the organism's real answer.

**Idempotency:** all mutating endpoints accept an `Idempotency-Key` header; retries
never double-apply. Commands are audited before and after execution (arch §14).

**Freshness:** every stateful response carries `freshness` ∈
`LIVE | CACHED | STALE | DISCONNECTED` (arch §4.3/§12). Cached state is never
presented as current. (`SIMULATION` is a UI-side fifth state added by the app's
simulator; it is not produced by this API.)

## 5. Stop semantics (arch §13)

`POST /v1/stop` takes `scope` ∈ `pause_axon | stop_training | stop_compute |
emergency` plus an optional `target`. The response is a **per-component ack map**:
each registered component reports `confirmed_stopped`, `unreachable`, or
`provably_gone`, plus detail. `all_stopped` is true **only if every component
acknowledged or is provably gone** — the contract makes it structurally impossible
to claim "everything stopped" when any component is unreachable. A mid-segment
interactive training pause is a future `TrainerOrgan` handler; until wired, the
organ returns `UNAVAILABLE` and the ack says exactly that.

## 6. Mapping to a future Python FastAPI implementation

The intended implementation (arch §5.1) is a FastAPI service living in-repo at
`runtime/controlplane/` (landing there requires a deliberate, reviewed
`tests/test_day_zero_hygiene.py` update in the same commit — arch §17 risk 7).
All dependencies (`fastapi`, `starlette`, `uvicorn[standard]`, `websockets`) are
already declared in `pyproject.toml`; no new runtime dependency is required.

| Endpoint group | Implementation source (State root paths relative to `--state-root`) |
|---|---|
| `GET /v1/health` | `active/heart/health_latest.json` (`axon-heart-health-v2`) distilled |
| `GET /v1/field/head` | `active/branches/active/HEAD.json` (`axon-canonical-state-branch-head-v1`) + health |
| `GET /v1/field/snapshot/{fieldId}` | `active/branches/active/snapshots/<field_id>.json` (`shared-field-v4`) |
| `GET /v1/field/ticks/{n}` | tail of `active/branches/active/journal.jsonl` (`axon-canonical-state-branch-event-v1`) + deltas |
| `GET/PUT /v1/masks` | `active/heart/region_masks.json` (`axon-heart-region-masks-v3`); PUT applies via `HeartHost.set_region_mask_policy` — two-step confirm, never mutates on first touch |
| `GET /v1/cores` | `CoreRegistry` descriptors + bindings via `HeartHost` |
| `GET /v1/souls/{coreId}` | `active/souls/<core_id>/branches/live/` HEAD/journal/receipts (`axon-private-soul-*` metadata only; payloads opaque) |
| `GET /v1/trainer/status` | `inspect_trainer_state` (`axon-trainer-inspection-v1`) |
| `POST /v1/trainer/commands` | `TrainerOrganCommand(kind, args)` → `runtime/trainer/organ.py` (fail-closed; 13 kinds) |
| `/v1/compute/workers` | Control Plane machine registry (operational, non-canonical — arch §2.4) |
| `/v1/storage/artifacts` | Storage profiles + artifact catalog (operational) |
| `/v1/capsules` | Capsule catalog; manifests follow `axon-cloud-bundle-manifest-v1` philosophy (per-member sha256, manifest written last); restore verification per arch §9.3 fail-closed order |
| `GET /v1/audit` | Control Plane operational audit log (arch §14) — **not** the Engineer's Ledger |
| `/v1/agents` | Agent registry + routed messages (provider adapters, arch §7); never carries secrets |
| `POST /v1/stop` | Arch §13: host boundary / tranche boundary / provider stop / emergency fan-out with ack accounting |
| `/v1/events` (WS/SSE) | Derived from State-root journals: `journal.jsonl`, `health.jsonl`, `AXON_PROGRESS` events, ledger metadata; envelope `axon-home-event-v1`, seq-gapped replay |

**Ledger discipline (arch §14):** the audit log is operational, not the canonical
Engineer's Ledger. Ledger writes happen only via
`scripts/append_engineers_ledger_event.py` by a participating agent — never from the
app or the Control Plane. A future single-writer ledger-ingress endpoint is a
designed-not-built item.

## 7. Files

- `openapi/axon-control-plane-v1.yaml` — REST surface v1 (OpenAPI 3.0.3). The
  `/v1/events` stream is documented but is WebSocket/SSE, outside OpenAPI's scope;
  its schema is `schemas/events.json`.
- `schemas/` — `common.json`, `field.json`, `masks.json`, `cores.json`,
  `souls.json`, `trainer.json`, `compute.json`, `storage.json`, `capsules.json`,
  `audit.json`, `agents.json`, `events.json`, `stop.json`.
- `examples/` — runnable fixtures, mirrored by the Kotlin contract tests.
