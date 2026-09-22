# Stage-0A step-24 prefix diagnosis

**Author:** Codex / GPT-5 / 2026-09-22 America/Chicago
**Status:** read-only evidence; the candidate remains stopped
**Candidate:** `english-candidate-1c991f8c911f79394e91`
**Accepted bundle / checkpoint / Soul:**
`1d77a96509993005e03366066fdd43f1143adb8895ba470b62f1a300aff7aeb5` /
`ef373220713565efa9da7b6c7877ffb5de75ab4522513ce679945102690ced3e` /
`308ff2a2b5de6337e904733a04da001af613faa7faa364a984150b3837753a0a`

## Method

The exact persisted step-24 parameter checkpoint and Soul were loaded on local
CUDA. No optimizer, Trainer, Heart, workspace, field, checkpoint, or Soul write
was performed. The replay used production-style runtime requests, then measured
teacher-forced, scheduled, and greedy free-running decoder behavior.

The held-out surface comprises 32 episodes and 64 supervised FIRST/REFINED
phases. A second read-only comparison used the first deterministic 32 training
episodes, also 64 phases. The training sample is shorter on average (5.03
transport units versus 10.16 held-out), so it is evidence about retained
examples, not a matched generalization score.

## Results

| Measurement | Held-out 64 phases | Seen 64-phase comparison |
| --- | ---: | ---: |
| Teacher-forced first content cell correct | 3.125% | 64.0625% |
| Teacher-forced leading content cells | 0.03125 | 0.6875 |
| Greedy free-run first cell correct | 3.125% | 64.0625% |
| Greedy free-run text-prefix characters | 0.0 | 0.03125 |
| Nonempty returned proposal | 20.3125% | 6.25% |
| Greedy termination | 37.5% | 17.1875% |
| Exact full proposal | 0% | 0% |

On held-out examples, teacher-forced content accuracy rises after the decoder is
given the correct previous token: position 0 is 3.125%, position 1 is 39.0625%,
and position 2 is 35.9375%. The scheduled counterfactual remains low at every
ratio: 14.46% with full teacher forcing, 13.38% at 0.5, and 12.77% at 0.0.

Every one of the 13 nonempty held-out strings begins incorrectly; all have zero
exact target-prefix characters. Eleven other held-out phases emit EOS with no
text, and 40 either do not terminate or produce invalid transport within the
bounded trace. The mean terminating length is 0.75 transport cells for targets
averaging 10.16 cells. This confirms the earlier premature-EOS observation, but
also rules out an EOS-only explanation.

## Position-zero mechanism trace

The follow-up trace measured the actual source pointer, copy/generate mixture,
and generated-token probabilities at the first decoder decision. It uses the
same 64 held-out and 64 seen phases as the measurements above.

| Measurement at first cell | Held-out | Seen comparison |
| --- | ---: | ---: |
| Exact required Cortex source position is pointer top-1 | 6.25% | 65.625% |
| Pointer probability at that exact source position | 11.07% | 31.35% |
| Pointer mass on any source with the target token | 11.65% | 32.37% |
| Copy-route probability | 52.06% | 52.10% |
| Generated target-token probability | 0.284% | 0.317% |

The copy gate has effectively the same slightly copy-favoring value on both
surfaces, and the generated path assigns negligible probability to either target.
The generalization gap is therefore the **source-position pointer**, whose
selection improves only for retained examples. This is direct evidence against
changing EOS pressure, decoder size, Soul behavior, or copy/generate-gate weight
as the first repair.

## Interpretation

The core has learned some decoder behavior for supplied target history and has
substantially better first-cell behavior on a fixed sample of seen examples. It
has not learned the general source-to-first-cell copy operation required to
bootstrap a response on unseen exact Cortex text. Once the first emitted cell is
wrong, teacher-forced interior-token scores no longer predict a usable response.

This is not evidence that D64, the frozen 16D substrate, the Soul, or the
number of transformer layers is intrinsically inadequate. It is evidence that
the present Stage-0A supervision has not yet produced a general free-running
copy mechanism. More layers, a larger FFN, or an arbitrary EOS penalty would
not test the identified failure directly.

## Governed next step

Do not launch another optimizer tranche from this candidate yet. The next
proposal should isolate the existing position-pointer mechanism with a
curriculum whose only learned task is to select the exact `cortex` source cell
and emit its transport unit across varied, held-out text. It should retain the
normal Heart/runtime/Soul circulation, use one versioned curriculum and
objective transition, and require free-running first-cell and pointer-position
success before returning to multi-character copy.

No promotion, Kaggle run, or architecture expansion is authorized by this
diagnosis. The current parameter/Soul lineage remains preserved until the table
chooses the controlled pointer-bootstrap transition.
