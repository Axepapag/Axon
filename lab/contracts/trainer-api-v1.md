# Axon Lab trainer API v1

Codex / 2026-10-06. Interface specification for backend/frontend integration.
Status: version 1 specification accepted for frontend/backend integration. The
foundation implementation status is recorded in `../backend/README.md`; training
and runtime families remain pending. Changes
must be coordinated with the frontend and affected runtime/core/data owners.

Backend owns this specification; frontend consumes it. Suggested HTTP prefix:
`/api/v1`. JSON is the transport for lab metadata. Native-95 admission applies
to Axon exact content, not to arbitrary UI labels or metadata transport.

## Common behavior

- Responses declare `schema_version`. Version 1 uses `axon-lab-api-v1`.
- IDs are opaque strings. Times are UTC ISO-8601 strings. Counts and sequence
  numbers are integers. Unknown or unavailable measurements are `null`, never
  invented zero values.
- An error contains `code`, `message`, `details`, `retryable` and `request_id`.
  Validation issues identify a node/port/field path where applicable.
- Asynchronous POST commands return HTTP 202 with `operation_id`, `status` and
  optional `run_id`. Acceptance is not completion. An operation exposes its
  current status, result or failure through a separate GET.
- Mutating requests carry a unique `command_id` for deduplicated retry. Reusing
  the ID with different contents fails explicitly. Pause, stop and checkpoint
  commands acknowledge completion only at their defined consistent boundary.
- The backend supplies currently allowed actions and reasons. The UI does not
  infer eligibility solely from a lifecycle label.
- Tensor inspection is observational and bounded. The contract does not grant
  direct writes to model parameters, recurrent states or canonical field data.

## Endpoint families

| Method/path | Purpose |
|---|---|
| `GET /capabilities` | Versions, actual component/provider/device capabilities |
| `POST /preflight` | Start selected-device checks; return operation ID |
| `GET /operations/{id}` | Observe asynchronous command completion or failure |
| `GET /readiness` | Current evidenced start-readiness checks and reasons |
| `GET, POST /architectures` | List or register versioned core definitions |
| `GET /architectures/{id}` | Read a registered definition |
| `POST /architectures/validate` | Validate a proposed definition before launch |
| `GET /datasets`, `GET /curricula` | Provenance-bearing data and study manifests |
| `GET, POST /runs` | List runs or request creation from an explicit specification |
| `GET /runs/{id}` | Run status, measurements and allowed commands |
| `POST /runs/{id}/commands` | Request start, pause, resume, stop or checkpoint |
| `GET /runs/{id}/events?after_sequence=N` | Replay ordered events; SSE transport |
| `GET /runs/{id}/tensors` | List available observable tensors |
| `GET /runs/{id}/tensors/{tensor_id}` | Inspect a bounded snapshot slice |
| `GET /checkpoints`, `GET /checkpoints/{id}` | Checkpoint registry and completeness |
| `POST /checkpoints/{id}/restore` | Request verified restoration into a run/session |
| `POST /inference/sessions` | Create inference session from selected checkpoint |
| `GET /inference/sessions/{id}` | Session, draft and committed-response status |
| `POST /inference/sessions/{id}/commands` | Explicit input, continue or stop request |
| `GET /inference/sessions/{id}/events` | Ordered inference/draft/runtime events |
| `GET /backup/status` | Actual configured artifact backup and restore evidence |

An unimplemented family returns a structured unavailable response. Providers
remain capability entries until a real adapter exists; the presence of an entry
does not imply that a launch endpoint works.

## Capability fields

`schema_version`, `backend_version`, `components[]`, `providers[]`, `devices[]`,
`supported_actions`, `feature_flags`.

Each entry has a stable `id`, display `name`, `status` and `reason`. Status
distinguishes `available`, `unavailable`, `not_integrated` and `experimental`.
Execution eligibility is explicit. Components additionally expose a version,
configuration schema, supported ports and constraints. Devices expose actual
type/index and measured availability. Experimental configuration support and
trained performance are separate claims.

## Architecture definition

`architecture_id`, `version`, `name`, `nodes[]`, `edges[]`, `ports[]`.
Each node includes `node_id`, `component_type`, `component_version`, `config` and
its port definitions. Edges identify source/destination node and port and any
explicit versioned adapter. No silent width, dtype or semantic conversion.

Port fields: `port_id`, `direction`, `meaning`, `dtype`, `shape`, `width` where
applicable, `stateful`, `authority`, required `surface` (`substrate-exact` or
`free`), `constraints`. Both ends of an edge must have matching surface markers;
crossing surfaces requires an explicit registered versioned adapter component.
Caller-provided port overrides cannot authorize a conversion. Substrate ports
declare the frozen codebook and legal occupied-prefix range; actual occupancy is
data/runtime state. WAIT/zero-commit is control state, never an all-EMPTY content
surface. State contracts additionally
declare role, initialization/reset/retention rules and checkpoint inclusion.
Separate exact substrate ports from learned reasoning/response/resident states.
Character slots describe mechanical frozen-code layout, not neural memory size.

Validation returns `valid`, `errors[]`, `warnings[]`, actual adapter eligibility
and optional parameter/resource estimates, clearly labelled as estimates.
Missing implementation prevents execution even when a graph is shape-valid.
Runtime/core owners supply their component contracts; this API does not invent
their update equations or canonical authority.

## Data and curriculum manifests

`dataset_id`, `version`, `stage`, `splits`, `source_counts`, `native95_status`,
`conversion_status`, `heldout_policy`, `cursor`, `progress`, source/content hashes
and provenance references. Curriculum manifests identify ordered stages and
explicit dataset versions. Original and explicitly converted derivatives remain
distinguishable. Registry access does not authorize a new source inventory.

## Runs and commands

Run specification: architecture ID/version/hash, curriculum/dataset IDs and
versions, intended split, explicit provider/device, seed and training settings.
Run response: `run_id`, `lifecycle_state`, architecture and data references,
`device`, `provider`, `seed`, `step`, `epoch`, `start_time`, `elapsed`,
`allowed_actions`, `readiness`, `failure_reason` and latest checkpoint reference.

Lifecycle proposal: `created`, `ready`, `starting`, `running`, `pausing`,
`paused`, `resuming`, `stopping`, `stopped`, `completed`, `failed`. Preparation
and readiness failures remain observable. Allowed transitions and safe pause/
checkpoint boundaries must be finalized with runtime and measurement owners.
Reject unavailable devices/providers; never silently fall back.

## Events and observations

Envelope: `sequence`, `timestamp`, `run_id` or `session_id`, `type`, `payload`.
Sequence order is defined per run/session and survives reconnect. Replay from
`after_sequence`; if history is unavailable, report a gap and a snapshot recovery
route rather than silently skip it. Commands and observations carry request/
operation or snapshot identities when relevant.

Event families include lifecycle changes, scalar metrics, throughput/latency/
VRAM, state summaries, tensor snapshots, available gate/routing observations,
draft revisions, Heart beat/commit acknowledgements and warnings/errors. A metric
includes name, value, unit, measurement scope and step/time. Observability is
capability-driven; unsupported gate readings remain unavailable.

## Tensor and draft inspection

Tensor fields: `tensor_id`, `name`, `semantic_role`, `dtype`, `shape`, `device`,
`snapshot_id`, `step`, available statistics (`min`, `max`, `mean`, `std`, `norm`),
optional histogram and a bounded value slice. Query specifies snapshot, flattened
offset and count. Response states the limit, original shape and whether data is
sampled. Reject an expired snapshot rather than combine multiple passes.
Retention limits and supported precision are backend capabilities.

Draft observations distinguish the private exact text, revision/read cursor,
pending submission identity and acknowledged canonical response/commit identity.
Draft existence is not Heart acceptance. Large text observations use explicit
ranges/revisions; no silent truncation. Core resident/circulating memory values
are not displayed as exact English without an explicit decoder contract.

## Checkpoints and inference

Checkpoint fields: `checkpoint_id`, `run_id`, `parent_id`, `step`, architecture/
data hashes, completeness and checksums, state-included flags, optimizer/RNG/
curriculum cursor status, Heart replay/commit boundary, backup state and restore
verification evidence. Save all declared live states, draft, reader/mirror
positions and pending submission/acknowledgement state needed for coherent
resume. Runtime adapters define exact included artifacts. An ID alone never
proves completeness or identical restoration.

Inference fields: `session_id`, selected architecture/checkpoint, status, allowed
commands, private draft and committed response references, available state
snapshots and ordered events. Native-95 input errors identify the invalid input;
conversion is a separate explicit workflow. Training and inference use the same
agreed runtime path, not a UI-only output bypass.

## Preflight, readiness and backup

Preflight wraps the existing `axon-foundation-preflight-v1` report, including
`foundation_passed`, `training_authorized`, `checks` and
`unverified_requirements` fields. Preserve its existing `schema` discriminator
inside the wrapper; do not rename it to the lab envelope's `schema_version`.
Its current `training_authorized: false` is not overwritten to mean ready.
Higher-level readiness contains individual checks with status, reason, evidence
and update time, plus actual eligible actions.

Check model/runtime integration, native-95 admission, data/split validity,
selected device, lifecycle controls, complete checkpoint restoration and valuable
artifact backup/restore. Readiness semantics must reflect the final operator
acceptance contract; installing packages or passing codec tests is insufficient.
Backup fields distinguish configured destination, attempted copy, verified
artifact and restore drill. A Google Drive path is not proof of off-device
survival or completed upload. Cloud launches require actual adapter support.
