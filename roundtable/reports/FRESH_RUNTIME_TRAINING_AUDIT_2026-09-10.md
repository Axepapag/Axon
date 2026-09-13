# Fresh runtime training inspection

Codex / GPT-6 / 2026-09-10

## Mission and status

Jeff authorized inspection of D:\axon, architecture corrections toward training
inside the real organism, Soul-centered persistent learning, sparse but adequate
checkpoints, and eventual Kaggle training. Cortex and reasoning rails remain
separate organs, with separate cortical, reasoning, Trainer, and Heart cadences.

Status: partial. A checkpoint-retention defect is corrected. The larger runtime
training architecture is not implemented or launched. One deployment question
has been put to Jeff: keep the canonical Heart local with Kaggle workers, move
the whole runtime for a run, or explicitly allow an isolated runtime proof.
Recommendation: keep canonical authority local and add a remote training worker.

## VERIFIED by source inspection

1. `runtime/field/schema.py` has eleven regions, through `identity`; neither
   trainer instructions nor training responses exists. Current masks are a
   Heart-owned global regional view, not a training-cohort view. The user's
   requirement that only training cores attend the two training regions needs
   an explicit extension to the shared-view contract, without a second body.
2. `runtime/heart/registry.py` describes ACTIVE, OFFLINE_TRAINING, and DISABLED,
   but selects only ACTIVE participants. It has no mode-transition method or
   separate offline-training circulation. `runtime/heart/host.py` owns the real
   canonical transaction boundary and reasoning circulation.
3. `training/living_reasoning_curriculum.py::living_episode_objective` feeds
   `episode.first_workspace_text` and `episode.refined_workspace_text` into
   `unroll_runtime_phases`. The current smoke trainer performs governed
   optimization and causal Soul transitions, but those proposal boards come
   from curriculum. This does not implement the learner's own answer returning
   through a heartbeat and becoming its next attempt.
4. `LivingReasoningCoreD64.emit` prepares a Soul transition from reader state
   before choosing/decoding the emitted action. That Soul snapshot therefore
   does not directly encode the subsequently selected output or its outcome.
   Later proposal attendance can supply information, but is not an equivalent
   proof of post-action experience capture.
5. `D64SoulCodec.encode` detaches to exact FP32 bytes. Every subsequent phase
   reloads those bytes. HOT changes; colder-layer promotion has validation
   contracts in `runtime/soul/contracts.py`, but the inspected D64 exhale path
   does not implement automatic validated consolidation or distillation.
6. Candidate parameter/optimizer checkpoints and candidate Soul can already be
   accepted as one durable bundle. Candidate Soul branching preserves private
   layers and detects intervening live-Soul advancement. Parameter activation
   in `runtime/trainer/host.py` does not itself atomically publish a matching
   runtime core descriptor, Soul HEAD, and decoder execution state. Seamless
   whole-core mode switching is not yet demonstrated.
7. The Kaggle adapter exports a selected, committed packet and runs it remotely;
   there is no live Heart connection in that path. Its internet option serves
   checkpoint synchronization, not remote reasoning circulation. Authenticating
   with Kaggle is not evidence that remote organism transport exists.
8. The demo has a separate Cortex auto-ticker and grounded retrieval paths.
   The learned autonomous Semantic Cortex is still documented as shelved. A
   deterministic retrieval tick must not be advertised as a trained semantic
   core. The prior ledger reports no learned reasoning core serving and a
   copy/EOS failure in the latest D64 experiment; that experiment was not rerun
   during this inspection.

## VERIFIED by execution

- Focused pre-change suite: 32 tests passed across private Soul, accepted step
  bundles, reasoning circulation, Trainer activation, and living D64 anatomy.
- `State/training/audits/fresh_runtime_20260911/causal_soul.json` records an
  isolated diagnostic (no optimizer steps): Soul generations 0, 1, 2, 3;
  consolidated state changed by L2 0.7557021975517273 under Soul ablation;
  the later loss had no autograd path to the first exhaled activation.
  This proves forward dependence and the gradient boundary, not useful memory,
  delayed-credit performance, consciousness, or a training failure.
- Kaggle doctor: authenticated account `axongliksbot`, 26.22 GPU hours remaining
  at inspection. Accelerator entitlement remains unproven until actual device
  computation. Credentials were not read. Local PyTorch reports CUDA available.

## Implemented correction

Previously, `CandidateTrainingSession.checkpoint()` pruned by newest optimizer
checkpoint records. Several newer unaccepted checkpoints could evict payloads
still referenced by the accepted parameter/Soul recovery pointer.

Retention now additionally protects all bundles in the accepted rolling pointer.
Evidence-linked landmarks can retain an older accepted bundle independently of
the rolling set. Landmark creation checks published ancestry, exact checkpoint
payload, and the private Soul snapshot. It is idempotent and does not advance a
competency or serving gate. Corrupted retention identities stop pruning.
The maintenance dry-run uses the same selection as actual pruning.

The default rolling count remains three. Landmark retention is additional and
explicit; it does not create a checkpoint every tick. Automatic milestone
selection belongs to the assignment controller still to be implemented.
No production checkpoint pruning or live-state migration was performed.

## Concrete proposed runtime contract

These are proposed implementation details, not claims about current code.

1. Append `trainer_instructions` and `training_responses` through an immutable
   schema successor. Preserve every historical schema, ID, character, and
   provenance record. Changing region embeddings requires an explicit new
   architecture/migration identity; an old checkpoint must not masquerade as a
   direct resume into a changed model shape.
2. Heart issues an explicit cohort-scoped attended view over the same canonical
   field ID. Operational cores cannot see training regions. Training cores see
   their assignment, permitted response history, identity, and explicitly
   selected contextual regions. Full coverage means the entire authorized view.
   Evaluation answers and later feedback must not leak into an earlier attempt.
3. Assignments have durable IDs, success criteria, supervisor state, and exact
   attempts. Each attempt binds core, Soul, parameters, field/view, assignment,
   and rail identity. Real emitted proposals enter the Heart path; no supplied
   gold proposal board substitutes for those emissions.
4. Heart commits the accepted exact response and attempt metadata atomically,
   then acknowledges rail consumption. Invalid, blank, incomplete, and rejected
   attempts retain their actual evidence and distinct status. History is
   append-only; a current draft is a selection over history, not a destructive
   overwrite. Last-N attendance never renumbers old source positions.
5. Soul experiences must include actual action/outcome information on a causal
   post-action boundary. Preserve exact runtime forward values while comparing
   bounded cross-breath gradient credit against the current detached baseline.
   Serialization at inference does not inherently require detaching every
   training gradient. Change learning policy only under an explicit versioned
   experiment, with exact-forward and checkpoint-resume proofs.
6. Use gated work assignments. Resource slices may checkpoint and pause; they
   never declare competence. A configured repeated-failure threshold triggers
   investigation, prerequisite teaching, or a preserved pause. It is a renewable
   supervision policy, not a permanent lifespan or attempt ceiling.
7. Mode transitions stop new dispatch at a complete boundary and bind parameters,
   optimizer, all private Soul layers, assignment cursor, pending rail/decoder
   state, and generation identity. Return to service requires a jointly verified
   publication; a failed candidate retains its evidence while the prior accepted
   serving generation remains recoverable.

## Assignment and launch acceptance

- Exact transport: native and Unicode output, EOS, rejection, mask boundaries,
  and crash/replay tests through actual Heart transactions.
- Actual recursion: an observed initial error must reappear as that core's own
  next attended attempt. No teacher-supplied response may stand in for it.
- Delayed memory: reveal a cue, remove it from the attended field and proposal
  history, then evaluate the later assignment with intact, zeroed, and swapped
  Soul. Require task benefit, not merely a nonzero hidden-state difference.
- Generalization: hold out entire assignments and sequences; after feedback on
  ABC -> D, test new sequences rather than count copying D as learning.
- Supervision: reproduce plateau, malformed transport, unresponsive worker,
  nonfinite gradients, and curriculum defects; each must have a diagnosed,
  preserved outcome rather than endless steps or a false completion.
- Resume: interrupted versus uninterrupted next-attempt parity must include
  parameters, optimizer, Soul, assignment cursor, decoder, and exact draft.
- Start with a local runtime smoke, then a small Kaggle integration smoke and
  its first evaluation. Do not launch a long campaign on falling loss alone.

## FLAG: deployment decision

Mission item: one real Axon body while using Kaggle compute.
Problem: the existing adapter executes an isolated exported copy. Treating that
as the local live organism would be a false architectural claim.
Options: local Heart with authenticated remote workers (recommended); governed
temporary transfer of runtime ownership; explicitly isolated first proof.
Work awaiting this answer: canonical deployment/connection and Kaggle launch.
Work continued: source audit, local mechanism checks, checkpoint repair, and the
concrete contract above. No cloud job, new public endpoint, or serving promotion
was launched. This question concerns deployment intent, not a request to repeat
the authorization Jeff already gave for architecture corrections and training.

## Limits and machine state

This is not a full security audit or full-repository test run. No existing
external workload was stopped. Only local tests and a no-optimizer diagnostic
were started; their temporary workspaces are under State/training. The existing
untracked legal directory and two route-diagnostic files were left untouched.
No GPU hours on Kaggle were consumed by this work. No commit or push was made.

## Verification closeout

Final checkpoint/bundle regression: 8 passed. Ruff and git diff --check passed;
both doctrine copies are byte-identical. During
development one test run caught a missing pytest import (corrected), and a
long Windows test-workspace path hit the known path-length limitation. The
final run uses a shorter State/training workspace; the failure evidence remains.
