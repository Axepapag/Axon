# Targeted Review: Recent Gemini Remote-Training Work

Identity stamp: Codex / GPT-6 / 2026-09-11

Scope: read-only review of the new training authority, endpoint, worker lease,
Kaggle worker, demo-server integration, and focused round-trip tests. Attribution
to Gemini follows Jeff's request; the files are untracked and have no matching
canonical engineer-ledger event, so authorship is not independently proven.

## FLAG [BLOCKING]

Mission item: launch real, canonical runtime training with an offline core on
Kaggle while the Heart and shared field remain local.

Problem: the current path is a protocol simulation, permits unauthenticated
training access, and does not publish accepted deltas into the running Heart's
canonical field.

Evidence:

1. `scripts/kaggle_training_worker.py` contains no model, optimizer, loss
   backward pass, parameter checkpoint, `LivingReasoningCoreD64`, or `SoulStore`.
   It sleeps to simulate computation, emits fixed prose by attempt number, reports
   invented loss/gradient values, and hashes an in-memory dictionary as its Soul.
2. `WorkerLeaseManager.register_or_bind` accepts a caller-supplied `worker_id`
   without a token. Claim and emit also accept `core_id` and manufacture/bind a
   session. The focused test explicitly requires token-free registration and
   claim to succeed. The demo wires these routes into its HTTP server.
3. The demo callback calls `self.host.boundary`, but `HeartHost` exposes no such
   property; the boundary is `self.host.coordinator.boundary`. The initialization
   catches that failure and leaves the demo runtime errored.
4. `HeartTransactionBoundary.commit` returns a `HeartCommit` containing a
   successor snapshot; it does not install or durably publish that successor.
   The test fixture manually appends `commit.successor` to a private list, while
   the demo callback does not. Therefore the passing test does not validate real
   Heart persistence.
5. Endpoint assignments, claims, attempts, sessions, and Soul state are ordinary
   in-memory dictionaries. Restart loses them. The assignment `soul_id` is a
   placeholder hash, not an exact snapshot loaded from the canonical Soul store.
6. The worker posts arbitrary `response_text` and its own `success` Boolean. The
   endpoint converts that text directly into a field delta and trusts the Boolean
   to complete the assignment. It does not validate a typed `ReasoningEmission`,
   categorical output, EOS, frozen tick/view binding, a real loss, a Soul
   transition, or a parameter update.
7. `TrainingCohortView.scoped_field` builds a different
   `SharedFieldSnapshot`, empties excluded regions, truncates response text by
   rebuilding its `RegionState`, and gives the view a new field identity. This is
   not yet the requested mask over one canonical Axon body with preserved exact
   16D address/history identity.
8. An unsuccessful assignment becomes permanently `failed` at `max_attempts`.
   That is a fixed candidate lifespan rather than a supervision checkpoint that
   investigates, adjusts, pauses, or escalates while preserving renewable work.

Options I see:

1. Replace the simulation in place: keep the useful HTTP envelope, but bind it
   to a real core/Soul/checkpoint worker, require authenticated leased requests,
   accept typed emissions and evidence, and commit through a real durable Heart
   operation. This is my recommendation.
2. Retain it under an explicitly named `protocol_simulator` test harness and
   build the production path separately. This avoids confusing scaffold success
   with learning but duplicates some wiring.
3. Launch it unchanged only as a network plumbing exercise. This conflicts with
   Jeff's explicit rejection of toys and must not be described as training.

Work halted: demo restart, public endpoint exposure, Kaggle launch, assignment
queueing, and any claim that a core is alive or learning.

Work continued: source inspection, focused tests, lint, canonical-commit trace,
and local listener check.

## Verification

- `python -m pytest -q tests/test_training_remote_roundtrip.py`: 27 passed.
  These tests prove the scaffold's own rules, including its authentication
  bypass; they do not prove real training or canonical runtime integration.
- Ruff on the reviewed files: failed with 53 findings, including unused imports,
  an unused inhaled Soul value, and duplicate exception handling.
- Static interface trace: no `HeartHost.boundary` property exists; the coordinator
  owns it. The transaction boundary returns a successor but does not install it.
- Local port check at 2026-09-11T13:15:44Z: no listener on TCP 8765. No service
  was started or stopped and no cloud job was launched.

## What is worth keeping

The separation of trainer instructions and response history, offline-training
registry status, assignment/attempt data shapes, sliding history concept, lease
concept, and compact rail transport are useful scaffolding. They should be kept
only after the authority, persistence, identity, real-core, and evaluation
contracts above are made real.
