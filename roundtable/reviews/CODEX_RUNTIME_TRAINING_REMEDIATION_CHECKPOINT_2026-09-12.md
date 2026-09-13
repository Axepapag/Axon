# Runtime Training Remediation Checkpoint — 2026-09-12

Identity stamp: Codex / GPT-6 / 2026-09-12

Status: **verified corrective checkpoint; local conformance loop is materially stronger, but Kaggle launch remains blocked.**

## Corrected in this checkpoint

- The authenticated gate submission binds the exact incoming Soul payload, codec metadata, emission reference, and Soul lineage. Gate recomputation and counterfactual probes use the same recurrent state the worker inhaled.
- Teacher-forced logits are supervision evidence only. The typed rail proposal now uses `decode_greedy`, feeds back its own predictions, and never receives the expected answer prefix. A regression proves that changing `D` to `XYZ` cannot change an otherwise identical proposal.
- Each Heart-committed response advances the live training view and the assignment's field/view/tick binding. Attempt 2 is verified to attend the successor of attempt 1 under the configured latest-eight history window.
- `TrainingAssignment.assignment_id` is stable across supervisor curriculum/view revisions. The previous derived-ID behavior silently changed the work-object identity when `adjust_curriculum` ran.
- An injected `CanonicalStateBranch` persists the Heart-computed response successor as durable HEAD and verifies the resulting field identity.
- Every completed attempt, pass or fail, writes an atomic hash-verified model/optimizer recovery workspace. The newest three are retained. A rejected learning attempt now survives a process-style restart with exact parameter, optimizer, and Soul lineage and no accepted checkpoint.
- Accepted Soul publication stamps the attempted field/tick, not the already-advanced successor view. Resume probes now include the live Soul state.
- `runtime.trainer.__all__` exports the actual `ATTEMPT_SCHEMA` and `CRITIQUE_SCHEMA` names.
- The follow-up D64 tournament now contains parameter-matched and parameter-rich depth candidates. Exact measured sizes are recorded in `roundtable/proposals/CODEX_D64_CORE_MUSCLE_RUNTIME_TRAINING_DECISION_2026-09-12.md`.

## Verified boundary

- The 253-test broad focused run reached 252 passes and one obsolete assertion after Soul binding became exact. That assertion compared post-attempt Soul evaluation with the gate's pre-attempt Soul evaluation; it was corrected and passed independently.
- The complete affected runtime group then passed after the WIP recovery integration, including assignments, local worker, supervisory gates, remediation, and session end-to-end tests.
- Four lineage-specific tests passed after final stamp/probe corrections: mode round-trip, accepted preemption recovery, rejected WIP recovery, and recursive Heart successor attendance.
- Attempt-workspace retention and corruption tests pass.
- `git diff --check` passes. Ruff passes on the newly added/modified runtime remediation modules and their focused tests; the package initializer retains its documented import ordering to avoid a known circular import.

## Remaining blocking flags

### FLAG BLOCKING — living core adapter

The session still instantiates the 114,145-parameter `CompleteField64D` conformance motor. It does not yet optimize the 33,982,137-parameter `LivingReasoningCoreD64` with its architecture-owned Soul codec and typed runtime heads. No competence conclusion or Kaggle learning launch is authorized from the conformance loop.

### FLAG BLOCKING — cross-store crash transaction

The attempt journal, Soul transition, attempt workspace, assignment outcome, canonical field commit, and accepted landmark are durable individually but do not yet share a prepare/finalize/recover transaction. A crash between the worker's attempt-journal write and workspace save remains a narrow unrecoverable window. Add transaction intents and crash injection at every boundary.

### FLAG BLOCKING — assignment completion semantics

A gate-accepted optimizer step currently means the submitted step is admissible. It does not yet mean the homework assignment is solved. The trainer needs a separate assignment-completion decision that can issue critique/retry or complete the assignment and author the next assignment.

### FLAG BLOCKING — live Kaggle transport

The remote rail primitives exist, but no authenticated Kaggle worker continuously leases work and returns exact living-core parameter/Soul artifacts to the local Heart. After the three blockers above, run a no-learning interruption/replay smoke before any learning tranche.

## Honest architecture assessment

Exact D16 character cells packed four per D64 row remove tokenizer ambiguity and preserve canonical character addresses. Words, sentences, and paragraphs remain exact sequences of rows; they are not each a single exact semantic vector. Learned attention and recurrent reasoning still communicate through D64 continuous state. Candidate A's 131,072-unit FFN already provides substantial private capacity, while depth supplies additional sequential transformations. The next clean experiment is 2x131,072 versus 4x65,536 at nearly equal parameter count, followed only if warranted by the 51M and 68M candidates.
