# D64 mixer copy-alignment smoke — 60-step results

Author: Grok 4.6 / GitHub Copilot CLI / 2026-09-06
Status: EVIDENCE. Non-serving. No promotion. No Source of Truth change.

Jeff ratified invert-shape D64 and asked for Kaggle on different shapes.
Two 60-step motor-v2 `copy_alignment` smokes completed on real Kaggle CUDA
from commit `999df55`. Outputs fetched and hash-bundled.

## Jobs

| Shape | Candidate | Job | Packet | Result |
|---|---|---|---|---|
| 1h / 4L / FFN256 | `axon-d64-mixer-4l-ffn256-h1` `r64v2-9d4df4d17517d7eb` | `387a52ebc78e51087b11f461ebc68153069a84fdd46039f812edd43705f3d9b8` | `d5eacf003e2769699b15300ebb0c2ff4279d8e8d459556e42d66d88486fe723f` | completed, returncode 0 |
| 1h / 4L / FFN512 | `axon-d64-mixer-4l-ffn512-h1` `r64v2-a7dffef9feae5bb5` | `26302a3d6f93fa5e93438b92381266832e1c962205c640027e3a2b6704795747` | `72a2b0a199fd78e9d3f7e5b49a63ac2ecab6d91ad939e405cf2e836f41782710` | completed, returncode 0 |

Curriculum: `a872278fd0e8ef926370e1712d01dcf0a277672c4a0088af44aef483d8417740`.
Generate-gate bias 0.0. Coverage 1.0 both runs. Serving claimed: false.

## What moved

Heldout, both shapes, after 60 steps:

| Metric | Start | FFN256 | FFN512 |
|---|---|---|---|
| copy-gate accuracy | 0.0 | **1.0** | **1.0** |
| position accuracy | 0.0 | **1.0** | **1.0** |
| pair exact copy-gate | — | 1.0 | 1.0 |
| pair exact position | — | 1.0 | 1.0 |
| teacher-forced content | 0.0 | 1.0 | 1.0 |
| EOS-gate | 1.0 | 0.0 | 0.0 |
| heldout mean loss | ~2.4 | 0.635 | 0.571 |

Copy-gate locked to 1.0 on the train stream from step 2. Fat Candidate A
never did this in 120 steps.

## What did not pass

`copy_alignment` stage gate failed both shapes for the same reason:

- regression position accuracy **0.667** (need 0.95)
- regression changed-source position pair rate **0.5**

Regression copy-gate is already 1.0. EOS was not this stage's job
(copy_alignment weights copy-gate, not EOS). Free-running payload exact
remains 0; that belongs to `transport_eos` / later stages.

FFN256 (331,319 params) and FFN512 (463,415) are tied. Extra FFN did not
buy regression position.

## Verdict

64D can attend the rail and learn copy vs generate, plus exact source
position, once the tissue is a mixer instead of a 131072-wide MLP.
This is not motor mastery and not communication. It is the first honest
copy-alignment signal.

Recommended next shot: one renewable `copy_alignment` tranche on the
FFN256 parent (step-60 bundle
`8501e20f2ae016c4010050a6905de38b2eb85cde548edbb1cd76d7eb06b4397e`),
not a wider rail, not a third shape, not joint Stage-0-v2. If regression
position clears 0.95, advance to `transport_eos`.
