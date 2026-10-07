# First E0 Lab run adapter

The Lab now binds run lifecycle to the actual E0Loop/HeartHost path. Verified
frozen e0-first dataset/curriculum manifests are exposed, including hashes and
120/45/45 split counts. Preparation pins a current registered D512 graph,
dataset/curriculum identity, explicit local device, seed and bounded settings.
Canvas node IDs are opaque; topology, component versions and configurations
determine compatibility. Stored older component graphs are not silently upgraded.

Implemented endpoints: create/read/list runs; start/pause/resume/stop/checkpoint
commands; durable ordered SSE and JSON event replay; coherent episode snapshots;
bounded tensor inspection; checkpoint list/read. Duplicate command IDs replay
their operation and altered specifications conflict. Concurrent boundary
commands cannot overwrite an outstanding operation. One worker owns execution.

Public Start remains closed through a server-owned acceptance gate. Client JSON
cannot override it. Production readiness/backup still report false/not_verified.
The gate was opened only by an in-process fixture in isolated temporary tests.
No production/valuable training was launched. Tests exercised real E0 optimizer
updates and Heart commits at CPU scope, not learned-recall acceptance.

Pause/Stop/Checkpoint finish **after the current episode**. A bundle includes
Core weights/both states, Heart state, loop CPU/Python/NumPy RNG cursor, Adam
state, CUDA RNG when selected, and a hash-bound Lab run/curriculum cursor.
Unique bundles are published to the registry only after checksums are written.
Resume verifies the bundle before safe deserialization and refuses mismatched
identity, cursor or a Heart HEAD advanced beyond the checkpoint. Paused runs
can resume after backend restart at the saved episode boundary. In-flight runs
are marked failed/interrupted on restart; no automatic mid-episode restoration
is claimed. Generic restore-into-another-run and inference remain unavailable.

Observations are actual optimizer-call counts, curriculum result rows, readable
private draft versus committed response, and two recurrent tensor snapshots.
Measured train-phase loss and bounded samples are now passed through from the
loop, labelled `teacher_forced_training`; these are not held-out recall scores.
API events have durable per-run sequence, source epoch/ordinal metadata and
explicit source gaps. Snapshot IDs bind inspection to one completed boundary;
tensor values are fetched separately in at most256-value slices.

Verification: full suite480 passed and both substrate checks exited0. Focused
run tests also passed after snapshot/resume refinements. Coverage includes
real-loop pause, process-lifecycle restart/resume, stop, optimizer progression,
tamper rejection before deserialization, idempotent commands, blocked execution,
ordered reconnect history, frozen-artifact mutation rejection and snapshot
limits. Whitespace check scoped to modified source passes.
One later hosted-transport check returned httpx.ReadError; the clean isolated
rerun of both hosted tests passed. No legacy-host routing source was changed.

Deployment uses the existing managed8080 child and unchanged Cloudflare tunnel.
Core catalog is refreshed to v0.1.1 with occupancy32. A real E0 starter graph/run
was prepared without execution:
architecture a4df20d6-4708-4909-b2b9-290373db5c03,
run6993495d-b7d2-4172-ab42-f11bdda43fc9, deviceCUDA0.
Managed loopback preparation/retry and blocked Start409 were verified; public
GET of that run verified created/allowed_actions=[stop]. Public Python urllib
requests received403; no Cloudflare security or account settings were changed.
Public PowerShell GETs succeeded. Browser-owner preparation acceptance is pending.

Kimi owner follow-ups: the current loop supervises only the first expected
response character and may END before reaching a target; full response-learning
acceptance remains pending. Its standalone checkpoint lacks optimizer/CUDA RNG
and structured cursor/hash linkage; the backend supplements these at episode
boundaries. Do not equate standalone mid-episode forced-control tests with full
training resume proof. Broad native95 coverage, counterfactual recall, independent
current backup and complete operator acceptance remain open.

## Training-fix integration correction

Pause state, command completion and lifecycle event now publish in one SQLite
transaction. A held-commit concurrency regression proves observers see both
old values or both completed values, preventing the paused/queued race.

The backend registers the loop's optimizer and binds its frozen dataset. The
loop's `optimizer.pt` is preserved; Lab supplemental state is `lab_optimizer.pt`.
The Lab checksum manifest now covers nested RNG files before any loading.
Legacy bundles requiring migration are explicitly refused. Lab observations
use `axon-lab-execution-cursor-v2`; raw loop checkpoints retain their separate
v1 block, and older Lab v1 schema files remain historical artifacts.

Further source review found target/input identity in the response phase:
the current expected character is passed to the model and used as that same
step's target. This is an answer-copy shortcut. Inference currently lacks the
post-query response walk used during training, and observation states are
detached before response supervision. These findings were sent to Kimi/Tests;
shifted autoregressive targets, bounded free-running inference and a tested
memory-learning gradient strategy remain acceptance requirements. Public
Start remains blocked.

## Autoregressive correction (Jeff authorized Codex)

The target/input shortcut, missing generation phase and detached observation
credit are corrected by the E0 autoregressive-v2 protocol. See
`docs/E0_RESPONSE_CORRECTION_2026-10-07.md` for training/generation/recovery and
staging semantics. Optimizer progress counts one actual update per exercise;
main loss now reports the mean `objective_loss`. Teacher-forced text remains
labelled. This does not clear learned-recall or backup gates.
