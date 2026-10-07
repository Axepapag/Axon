# E0 run adapter handoff v1

Codex chooses run lifecycle/events/snapshots as the first backend slice. KimiCode
owns `runtime/heart/host_e0.py` and its tests; existing source ownership stands.
This is a handoff specification pending the actual loop; no fabricated run
executor is registered from HeartHost alone.

The loop must supply a documented construction contract binding architecture
version/config/hash, frozen curriculum/dataset identities, explicit selected
device and seed. Reject unavailable devices/configurations without fallback.
Accept optional optimizer/objective execution for training; held-out evaluation
must not update weights or consume the training RNG/cursor. Core/Heart tick,
proposal and admission behavior is shared across training and inference.

Required semantic operations (Kimi supplies actual method names/signatures):

- Advance one bounded step through the actual HeartHost path and report whether
  a consistent pause/stop/checkpoint boundary has been reached.
- Observe one coherent snapshot: Core state roles/shapes, readable private draft,
  committed response/Heart generation, episode and optimizer progress, cursor
  conforming to execution-cursor-v2 (Lab observation), and actual measured outputs. Null fields
  remain unavailable. Do not expose masks/D16 views currently stubbed as present.
- Provide totally ordered observations with epoch/ordinal or ordered callback;
  report overflow/restart gaps explicitly. Backend owns durable API sequence,
  reconnect history and lifecycle commands.
- Save/restore the complete organism at a consistent boundary. Return structured
  completeness, checksums and restore reports; standalone HeartHost and Core
  checkpoints separately are insufficient for optimizer/mid-episode recovery.
- Release the Heart lease/resources on stopped/failed workers. Distinguish
  curriculum/episode completion, END control and stopped whole run. Define turn
  finalization ownership or explicitly block behavior that depends on it.

Backend responsibilities: durable idempotent commands, run registry and allowed
transitions, single owning worker per run/Heart lease, persisted API events,
snapshot limits, readiness enforcement and truthful restart/interruption status.
Start stays blocked until the loop is integrated and Tests/Measurement,
backup/restore and operator gates are evidenced. A prepared or blocked run is
never reported as running; loss/throughput metrics require actual measurements.

Checkpoint registry and inference sessions follow the lifecycle integration;
the composite save/restore primitive is required for safe pause/resume first.
Frontend coordination follows actual endpoint availability. Existing endpoint
families remain structured not_integrated until their real adapters exist.

## Landed response-learning increment, 2026-10-07

Jeff explicitly assigned Codex the `host_e0.py` correction for this increment.
The loop now runs shared observation/response phases, shifted teacher forcing,
independent bounded generation and one full-episode optimizer update. Its
constructor adds keyword-only `response_tick_budget=256`; existing public method
signatures remain. Result rows add `generation` mode/budget/ticks/termination,
and training rows add `objective_loss`. Checkpoint cursor files add an
`execution` protocol/mode/episode/budget block; existing structured cursor schemas
are unchanged. Active incompatible older walks fail closed.

See `docs/E0_RESPONSE_CORRECTION_2026-10-07.md` for exact replay, staging and
acceptance scope. Teacher-forced scores do not clear learned-recall gates.
