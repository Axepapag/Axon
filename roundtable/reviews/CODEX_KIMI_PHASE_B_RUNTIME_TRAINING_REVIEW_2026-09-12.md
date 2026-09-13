# Review: Kimi Phase-B Runtime Training

Identity stamp: Codex / GPT-6 / 2026-09-12

Status: **real and valuable foundation work, but not correct enough to commit or
launch.** This review supersedes the integration conclusions in the earlier
point-in-time Kimi review. It does not accuse Kimi of simulation: the worker runs
real PyTorch computation and writes real state. The blocking defects are joins
between those real mechanisms and overbroad end-to-end claims.

## BLOCKING FLAGS

### FLAG BLOCKING — the supervisory gate grades a different recurrent computation

Mission item: independently verify the Soul-conditioned worker result.

Problem: the worker decodes the persisted HOT Soul payload and passes it as
`initial_state`, but `TrainingSession._post_step_metrics()` and
`GateEngine._forward_metrics()` omit that state. The gate therefore recomputes
from the model's default recurrent state. Its Soul counterfactual compares that
default state with zeros; it does not bind or ablate the persisted Soul payload.

Evidence: `State/training/audits/kimi_phase_b_review_20260912/diagnostic.json`.
On the second real attempt, the gate output digest exactly matched the default-
state output (`e3fc3835...`) and did not match the persisted-Soul output
(`a63b9bf7...`). Losses also differed: 4.2849788666 versus 4.3108296394.

Options I see:

1. Put the exact inhaled Soul payload/codec identity in the authenticated worker
   evidence and make the gate recompute with it.
2. Give the gate a content-addressed local Soul reference and require it to load
   and hash-verify the exact payload. Remote verification still needs the bytes
   or an authenticated retrievable object.

My recommendation: use a sealed payload digest plus exact Soul bytes in the
worker submission; independently decode and verify them at the trainer gate.

Work halted: commit, Kaggle plumbing smoke, and every Soul-conditioned acceptance
claim.

Work continued: read-only inspection, focused tests, and diagnostic evidence.

### FLAG BLOCKING — recursive attempts do not attend their preceding responses

Mission item: make each heartbeat/tick attend the assignment and the core's own
committed response history.

Problem: `TrainingSession.run()` keeps using the original `self._view` for every
attempt. Heart commits advance only `self._canonical_head`. `successor_view()`
can be called manually after the run, but the loop never installs it. The test
named `test_heart_commits_typed_response_and_successor_view_drives_next_attendance`
only constructs a successor view after two attempts have already used the old
view. Updating the view also requires an explicit durable attempt-view binding;
the assignment currently binds one original field/view/tick and the worker
rejects a different one.

Evidence: `runtime/trainer/training_session.py` lines 452-474 and 606-676 keep
the issued view fixed; the regression test explicitly states that it stays fixed
for the assignment and calls `successor_view()` only after `run()` returns.

Options I see:

1. Keep a stable assignment identity and add durable, monotonically issued
   attempt-view records under it.
2. Create a successor assignment per response. This is simpler but fragments one
   conceptual homework assignment and complicates failure-series supervision.

My recommendation: keep one assignment and add exact attempt-view issuance, with
the trainer deciding whether the next view carries critique, a retry, or a new
assignment.

Work halted: claims of recursive self-revision through field history.

Work continued: the derived view compiler itself was verified and should be kept.

### FLAG BLOCKING — rejected learning cannot resume across preemption

Mission item: exact stop/restart/reclaim continuity.

Problem: every learning attempt advances the in-process model, optimizer, and live
Soul, including a rejected attempt. Only accepted steps receive a durable
parameter/optimizer checkpoint. A restart after rejection therefore reloads the
latest accepted checkpoint (or a fresh seed when none exists) while the Soul and
attempt journal remain ahead. This is especially serious because retaining failed
attempt experience is part of the requested design.

Evidence: the diagnostic forced two honest rejected learning attempts around a
process-style restart. `assert_resume_continuity()` then failed with
`ResumeContinuityError: attempt 1 carries no prior parameter identity`; the second
attempt's prior generation was null instead of the first attempt's
`521bd8f6...` generation.

Options I see:

1. Persist a recoverable work-in-progress parameter/optimizer/Soul bundle for
   every attempt; only gate-passed bundles become accepted landmarks.
2. Roll rejected optimizer and Soul state back to the last accepted bundle. This
   discards the recursive learning path Jeff explicitly wants.

My recommendation: persist every attempt's candidate WIP bundle, retain only the
rolling recovery window plus immutable receipts, and reserve acceptance for
assignment completion/promotion authority.

Work halted: preemption-safe Kaggle worker claims.

Work continued: accepted-step checkpoint and landmark mechanics passed their
focused tests.

### FLAG BLOCKING — the session is not connected to the real living reasoning core

Mission item: train a 64D reasoning core that can attend the field, use Soul, and
emit runtime responses.

Problem: the new local session instantiates `CompleteField64D` with the small
default `ReaderConfig` and labels it `complete-field-64d-v1`. That model has
114,145 trainable parameters and a teacher-forced character decoder. Axon's
actual first living reasoning tissue is `LivingReasoningCoreD64`, with typed
decision/operation/region/pointer heads, its architecture-owned Soul codec, and
33,982,137 trainable parameters under the canonical Candidate-A configuration.
The current session proves plumbing on a real neural model, but it does not train
the organ whose competence is being asked about.

Evidence: direct instantiation/counting of both checked-in model classes during
this review; source at `training/living_reasoning_d64.py`.

Options I see:

1. Generalize the worker/gate/session contracts to an architecture adapter and
   bind `LivingReasoningCoreD64` first.
2. Ratify the small `CompleteField64D` model as a separate plumbing-only test
   core and forbid competence or runtime-core claims from its results.

My recommendation: do both explicitly: preserve the small core as a fast local
conformance fixture, then make the first production adapter the existing living
reasoning core.

Work halted: competence estimates based on the current session's falling loss.

Work continued: model identities and actual parameter counts were verified.

### FLAG BLOCKING — typed attempts leak target prefixes and Heart state is not durable

Mission item: commit the core's actual attempt through the real local Heart and
canonical branch.

Problem: the emitted characters are argmax values from `decode_teacher()`. For a
multi-character answer, later positions receive the correct earlier target
characters, so this is training evidence rather than an independent attempted
response. Separately, `HeartTransactionBoundary.commit()` returns a successor,
but the session only stores it in `_canonical_head`; it never commits that delta
to `CanonicalStateBranch`. Finally, the assignment outcome and Heart response are
written before accepted checkpoint/Soul publication, without a recovery protocol
joining all of them. A failure during publication can leave a passed attempt and
advanced in-memory field without an accepted bundle.

Evidence: `runtime/trainer/local_worker.py` lines 538-565;
`runtime/trainer/training_session.py` lines 679-738 and 769-848; Kimi's own ledger
flag says canonical branch persistence remains future work.

Options I see:

1. Separate the free-running proposal emission from the teacher-forced loss
   artifact, then coordinate attempt, candidate WIP/accepted bundle, Soul, and
   canonical branch through prepare/finalize/recover records.
2. Keep the current order only as a noncanonical lab harness, with names and tests
   narrowed so it cannot be mistaken for runtime circulation.

My recommendation: implement the first option before any remote worker. Add crash
injection at every prepare/finalize boundary.

Work halted: canonical-runtime and actual-response claims.

Work continued: typed delta authorization and in-memory transaction behavior were
verified as useful components.

## VERIFIED

- 130/130 focused schema-v4 migration, exact D64 compilation, training-view,
  trainer authority/binding, HMAC/replay, remote-rail, assignment, accepted-step,
  and checkpoint-retention tests passed.
- 58/58 focused local-worker, supervisory-gate, remediation, and session tests
  passed. They demonstrate many real components but do not cover the blocking
  joins above.
- `git diff --check` passed.
- An explicit import probe found `from runtime.trainer import *` fails because
  `__all__` advertises nonexistent `ASSIGNMENT_ATTEMPT_SCHEMA` and
  `ASSIGNMENT_CRITIQUE_SCHEMA`; the imported names are `ATTEMPT_SCHEMA` and
  `CRITIQUE_SCHEMA`.
- Ruff reported 42 findings over the broad selected runtime/test surface,
  including the broken trainer exports and multiple pre-existing style findings.
- No credential value was found in the new runtime-training sources. Test-only
  secrets are fixed dummy values.
- No Kaggle job, service, listener, canonical live migration, promotion, or Git
  commit was started by this review.

## ATTEMPTED

- No full repository test run was attempted after the blocking diagnostics. The
  188 focused passing tests are the verification boundary for this turn.
- No failure-injection transaction test exists for a crash between Heart response
  commit and accepted-bundle publication; source ordering is the evidence.

## ASSUMED

- Kaggle entitlement/quota and credentials were not inspected; they are
  time-varying and irrelevant until the local gates pass.
- No claim is made that the 33.98M-parameter living core can become generally
  conversational merely from its present architecture or curricula. That remains
  an empirical research question.

## Commit decision

No commit was made. The working tree combines valuable schema, authority,
transport, worker, and checkpoint foundations with false end-to-end invariants.
Committing it as a completed runtime-training layer would make the misleading
claims harder to see. Preserve the tree and correct it in governed increments.
The unrelated pre-existing `legal/`, `scripts/diagnose_d64_routes.py`, and
`tests/test_d64_route_diagnostic.py` remain untouched.

## Recommended sequence

1. Ratify attempt semantics: free-running response versus teacher-forced learning
   evidence; candidate WIP retention across failed attempts; stable assignment
   plus successive issued attempt views.
2. Add one `LivingReasoningCoreD64` worker/gate adapter. Seal exact parameter,
   optimizer, incoming Soul, rail, objective, emission, and lineage identities.
3. Make gate recomputation consume the exact sealed incoming Soul and prove the
   persisted-Soul counterfactual.
4. Persist every attempt's WIP bundle and make rejection/preemption/restart
   byte-exact. Keep the locked rolling-three recovery rule plus sparse landmarks.
5. Coordinate WIP/accepted bundle, Soul, outcome, typed free-running response, and
   `CanonicalStateBranch` using durable prepare/finalize/recover records.
6. After every response commit, let the trainer inspect it and issue the next
   attempt view, critique, retry, or successor assignment. Prove the next forward
   actually attended its own preceding response.
7. Pass local crash, replay, wrong-view, target-leakage, failure-series,
   mode-round-trip, free-running task, and delayed-Soul causal gates.
8. Then run a no-learning Kaggle transport smoke, followed by a tiny learning and
   preemption/reclaim smoke. Only afterward begin the ABC curriculum.

## Distance to conversational competence

The transport foundation is close to a credible local conformance loop, but the
requested living loop is not one Kaggle launch away. The next honest milestone is
one `LivingReasoningCoreD64` assignment that emits a free-running answer, receives
a trainer decision, attends its own committed history, and survives restart with
exact WIP state. After that, Kaggle plumbing is a bounded engineering step.

The existing ABC curriculum has 104 verified cases, and the language L0-L4
curriculum has 349 cases. Those are mechanism and foundation curricula. A 180- or
360-step tranche can establish that learning happens; it cannot establish
conversation or reasoning competence. Current repository evidence says no learned
reasoning core is serving, and prior Candidate-A runs did not pass exact-output
gates.

A 33.98M-parameter model with a 64D residual stream may learn constrained
sequence tasks and structured short responses. General conversational reasoning
from scratch is uncertain and likely requires much more high-quality lived-
experience data, multi-stage evaluation, and possibly a new architecture or
distillation strategy. Do not promise it from the current 349-case language
curriculum. Measure the first real milestones before estimating a date.
