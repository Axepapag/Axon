# Codex review of GRU Dormant language and EOS experiments

Identity: Codex / GPT-6 / 2026-09-26 America/Chicago

Scope: review of the report supplied by Jeff, current trainer source, immutable
run summaries, protected milestone, and independent read-only checkpoint
evaluations on the local GTX. No optimizer steps, promotion, architecture edits,
trainer edits, commits, or cloud jobs were performed.

## Finding: observed stopping improved; correct stopping did not

The reported free-running character scores and breath retention reproduce.
However, `evaluate_language` counts any emitted EOS within its 72-decision
budget as termination. It does not require EOS at the target boundary. The
independent replay separates those outcomes:

| Checkpoint | Early EOS | EOS after exactly 64 characters | Late EOS | No EOS within budget | Median emitted characters | Free character accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Protected `dec45989...` | 0 | 0 | 0 | 64 | 72 | 13.9893% |
| EOS8 `93ee6102...` | 41 | 0 | 0 | 23 | 8 | 10.8398% |
| EOS12 + sampling `226e7dfe...` | 55 | 0 | 0 | 9 | 3 | 6.2500% |
| EOS4 follow-up `758d56b8...` | 22 | 0 | 1 | 41 | 72 | 13.3545% |
| EOS8 follow-up `093edcde...` | 42 | 0 | 0 | 22 | 8 | 10.4492% |

Each row covers the same 64 held-out windows and 4,096 target characters.
The EOS8 64.0625% termination result consists entirely of premature stops.
The aggressive 85.9375% result also consists entirely of premature stops.
This is evidence that EOS pressure changes output behavior, not that the
correct termination task has been solved.

Free-character scoring counts missing target positions as incorrect. Therefore
the fall in that score partly reflects shorter output; it does not by itself
measure how much language knowledge was forgotten. Accuracy on the surviving
overlap is length-selected and is not an alternative mastery score.

For every checkpoint, zero of 64 outputs matched the full 64-character
reference even when the EOS requirement was removed. The original zero exact
score is therefore not explained solely by missing EOS. Open-ended language can
have multiple reasonable continuations, so reference exactness is also not a
complete measure of language quality.

## Finding: the target confuses a data slice with a completed utterance

The curriculum concatenates eligible recovered messages, takes 128 source
characters followed by 64 target characters, and appends a private EOS target.
It does not align those target endpoints to word, sentence, or message ends.
This teaches stopping at a fixed data-slice boundary rather than completion of
a thought. A fixed-length diagnostic is possible, but its result cannot be
interpreted as learned conversational completion.

The 64-content-to-one-EOS ratio is a plausible optimization consideration, not
a demonstrated sufficient root cause. Increasing EOS weight before resolving
the boundary semantics has produced a measurable premature-stopping tradeoff.

Relevant source: `scripts/train_continuous_core_d512_dormant_language.py`,
window construction at lines 266-278, loss at 382-405, evaluation at 409-467.

## Finding: a required preflight claim lacks executable evidence

FLAG [BLOCKING for further parameter mutation on this launcher]
Mission item: readiness for the next language tranche.
Problem: the COUNTERFACTUAL_DEPENDENCE preflight payload sets `passed: True`
and supplies a prose claim without running a counterfactual test
(`scripts/train_continuous_core_d512_dormant_language.py:547`).
Evidence: source inspection; SOURCE_OF_TRUTH requires executable preflight
evidence for Trainer mutations, including for the B4 D512 baseline.
Options: implement the required measured checks; or seek an explicit contract
amendment if the intended experiment genuinely calls for different evidence.
Recommendation: implement real checks and fail closed when evidence is absent.
Work halted: no further parameter mutation was launched during this review.
Work continued: checkpoint verification, inference-only diagnosis, regressions,
and documentation.

This gap does not erase the measured learning results. It means the launch
receipt currently overstates what was checked. Existing immutable events and
artifacts remain evidence; corrections must be additive.

## What is verified

- Original and protected milestone files both match SHA256
  `566d33cdcb31ad92c0a785b87578c43ad9d28197112daa5f83556a0945397107`
  and size 21,219,008 bytes. Checkpoint identity:
  `dec45989bd4b872a90fb7b1f4d9ce9a0ce448b5ded70d04bacff56bf3bca754f`.
- Reconstructed the exact first 64 held-out windows from the selected Dormant
  source and checked every window identity against curriculum
  `47cb93018581e0221db34814a864942398653808e0d9723165a1280df19bc73c`.
- Strictly loaded all five checkpoint payloads after their record size/hash
  checks and ran normal greedy inference with the existing 72-decision budget.
- Independently reran the existing breath evaluator on 120 held-out episodes
  per checkpoint: all five achieved 120/120 complete episodes and 120/120
  silent third breaths. These are the compact synthetic relation tasks.
- Eight existing Core/breath tests passed with the pytest cache disabled.
  They do not cover the new language objective or sampling implementation.
- A separate numerical spot check confirmed that an all-false sampling mask
  produces exactly the same logits and state as normal teacher forcing.
- `git diff --check` passed during inspection.
- The first diagnostic attempt encountered the milestone JSON's UTF-8 BOM;
  the read-only reader was corrected to use BOM-aware decoding and the replay
  then completed successfully. No checkpoint or metadata repair was needed.

## Next experiment recommendation

Preserve the milestone and keep the GRU architecture. Before another optimizer
step, replace asserted preflight success with measured evidence and add focused
tests for EOS weighting, self-feedback, and premature-stop accounting.

Keep slice boundaries separate from genuine completion. For raw language
continuation, a compute slice ending does not mean EOS. For completion practice,
use provenance-bound examples ending at an actual authored message boundary,
or explicitly specified short completion tasks, with variable lengths and
proper loss masks. Keep breath rehearsal in the experiment.

Report early, correct-boundary, late, and absent EOS separately, together with
output lengths, content accuracy, and breath retention. Do not change locked
mastery rules silently; version the proposed curriculum/evaluation explicitly.

Then compare a small controlled repair against the same protected parent,
holding seed, data order, and learning rate fixed. Establish completion and
content together before layering on self-feedback. The current runs vary
multiple factors and do not establish that scheduled sampling improved results.

The breath result supports retained performance on the tested three-breath
tasks. It does not yet prove causal private-memory/Soul use, general autonomous
breathing, or live Heart integration. Those require separate counterfactual and
runtime tests.

## Machine and publication state

The five replay evaluations took about 137 seconds in aggregate after startup
and dataset reconstruction. No cloud spend or training occurred. The diagnostic
and regression processes completed. Existing unrelated services were untouched.
ChatGPT's implementation remains uncommitted; this review does not publish or
promote it. This report and both engineer ledgers are the only intentional
persistent changes by this review.

