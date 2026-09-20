# English-native runtime and substrate review

Codex / GPT-6 / 2026-09-19 America/Chicago (2026-09-20 UTC)

Scope: Jeff requested a review of the massive recent changes. Inspected HEAD
`f88a49d`, recent migration history, current uncommitted curriculum/EOS edits,
runtime contracts, training path, ledgers, and local run reports. No implementation
edits, commit, promotion, training launch, or cloud operation in this review.

## Assessment

The English proposal migration is a meaningful architectural improvement. FIRST
and REFINED can express observations and uncertainty without having to encode a
field mutation. The consolidator names desired region bodies; Heart binds the
transaction metadata and performs deterministic parsing, validation, and commit.
Exact width-neutral text transport remains distinct from private latent Soul.
This is substantially closer to Jeff's intended deliberation process.

It is still an early developmental system. Exact representation preserves input
information; it does not demonstrate understanding. Persistent Soul receipts
demonstrate continuity; they do not by themselves demonstrate useful memory.

## VERIFIED: training evidence newer than the rolling handoff

`State/training/reasoning/english/english-candidate-53f4ad04ba58c4d27348/report.json`
records a CUDA run of 64 optimizer steps, eight experiences per step, and 64
accepted bundles. Its heldout mastery gate **failed**:

| Measurement | Recorded result |
| --- | ---: |
| Exact responses | 0 / 24, over 12 heldout episodes |
| Teacher-forced content accuracy | 8 / 118 = 6.78% |
| Teacher-forced EOS accuracy | 11 / 24 = 45.83% |
| Complete-field coverage | 100% |

The first and last training-batch losses were 9.7724 and 1.8501. Those are different
batches, not a controlled before/after evaluation. The failed heldout result is
the relevant competency evidence. `status=completed` describes execution, and
`mastery_gate.passed=false` correctly discloses failure. No promotion was attempted.

The current uncommitted v2 curriculum has 267 training and 32 heldout episodes,
uniform copy instructions, disjoint exact target texts, and mixed symbol/sequence
ordering. The launcher now explicitly weights text EOS at 4.0 and includes that
in objective identity. The newer `english-candidate-87370c321fa7a0824316/report.json`
is **preflight_only**, not evidence that those revisions learned successfully.
The older 64-step failure must not be attributed to this newer configuration.

## Review findings

1. **Launcher restart omits existing transaction recovery.**
   `scripts/train_living_reasoning_smoke.py:287` reads `latest_bundle` before
   acquiring the Trainer lease, and never calls `recover_pending`. Later it
   requires candidate Soul HEAD to equal the accepted bundle. An interruption
   during Soul publication or pointer/sentinel publication can therefore leave
   recoverable state that ordinary `--resume` rejects. The coordinator already
   implements recovery; its four tests pass. Wire recovery under the writer
   lease before selecting the resume boundary, then test the actual launcher.

2. **Refinement evaluation supplies the expected prior proposal.**
   `training/substrate_literacy_curriculum.py:76` stores the target text as both
   workspace texts. `training/living_reasoning_curriculum.py:487` evaluates an
   unroll using those authored texts. The inherited unroll never feeds its decoded
   FIRST attempt into REFINED. This is useful supervised scaffolding for copying,
   but it is not a measurement of autonomous revision after a mistaken attempt.
   FIRST still has a legitimate source-copy task; this does not invalidate every
   metric. Add a separately reported rollout using actual emitted proposals.

3. **The fail-closed mastery gate accepts NaN.**
   I directly called `decide_substrate_literacy_mastery` with NaN for every required
   metric and a constant floor of 0.25. It returned `passed=True`, `failures=[]`.
   The `<` and `<=` comparisons at `training/substrate_literacy_curriculum.py:269`
   do not reject NaN. Validate finite, bounded metrics and valid denominators.
   This reproduction does not mean the recorded failed run contained NaNs.

4. **Soul continuity is ahead of demonstrated Soul usefulness.**
   The training path threads Soul across experiences and persists transitions with
   checkpoints. However, phase boundaries serialize/detach state; future losses
   do not directly backpropagate through earlier writes. Active exhale stores
   reader state before output decoding (`training/living_reasoning_d64.py:433`),
   so it does not directly store the subsequent generated answer. Later board
   attendance can bring an answer back in, but training currently supplies gold
   boards. Cold layers remain preserved rather than educated by this Stage 0.
   None of this proves Soul cannot learn; it limits what current evidence shows.

5. **Longer training needs earlier observation and durable behavioral landmarks.**
   The launcher accepts recoverable optimizer/Soul bundles every step, evaluates
   only after the tranche, and overwrites one default report on each continuation.
   It does not use the existing landmark facility. Accepting an optimizer step is
   appropriately different from passing homework; temporary metric regression
   need not reject every update. Preserve a measured competent landmark separately
   from rolling recovery and publish periodic, immutable evaluations. The current
   smoke path does not yet implement the complete assignment/grading/retry loop.

## FLAG ADVISORY

Mission item: assess readiness for continued developmental training.
Problem: suspected implementation mistakes and a mismatch between autonomous
refinement language and authored-workspace evaluation.
Evidence: findings 1-3 and the failed 64-step report above.
Options: repair and prove the bounded loop first; or continue diagnostic training
with these limitations explicitly accepted and without competency claims.
Recommendation: repair restart/gate handling and add actual-attempt evaluation
before extending the run.
Work halted: no new launch was part of this review; no existing process stopped.
Work continued: review, focused tests, run-evidence inspection, and ledger updates.

## Verification and limits

VERIFIED: 39 tests passed across English contracts, runtime circulation, living
reasoning core, and substrate curriculum. Four additional training-step bundle
tests passed, including partial-Soul and damaged-pointer recovery. `git diff
--check` passed before documentation edits. PyTorch emitted its existing
norm-first/nested-tensor performance warning; tests did not fail.

ATTEMPTED: all selected verification completed. No fresh training experiment or
full repository test suite was attempted. The 64-step metrics were read from
existing artifacts, not reproduced by retraining.

ASSUMED / UNPROVEN: the revised curriculum will improve learning; Soul will support
useful delayed recall; a trained core will revise its own mistakes. These require
behavioral evidence. A useful next memory probe presents an unpredictable fact,
removes it from the field, then compares later recall with intact, reset, and
swapped Soul under identical parameters and visible input.

Machine: no Axon training process was observed; GTX 1650 showed 0% GPU utilization
at inspection. Existing unrelated Python services were left alone. Review used
local CPU tests; no paid cloud work or new background job.

Recommended sequence: fix gate/restart/reporting issues; prove heldout copying
with actual-attempt rollout; add delayed recall and error-correction assignments;
then expand English reasoning. Additional layers or ticks should follow measured
bottlenecks. This review provides no basis to declare D64 incapable or sufficient
for broad conversational competence.
