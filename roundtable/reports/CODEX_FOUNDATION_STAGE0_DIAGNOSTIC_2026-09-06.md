# Foundations Stage 0 — Two-Tranche Diagnostic

Author: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-06

Status: VERIFIED FAILURE OF THIS TRAINING RECIPE; CANDIDATE PAUSED,
NON-SERVING, AND PRESERVED. This is a curriculum/objective diagnosis, not an
architectural falsification of D64 or Axon's organism design.

## Bottom line

The new exact-alignment path proved that the existing D64 decoder can begin to
select exact source addresses. It did **not** teach the decoder to switch from
generation to copying, and the joint curriculum encouraged constant control
predictions. After 120 accepted steps the core still emitted the same
`DELTA` / `REPLACE` / `response_draft` / empty-payload shell for every sampled
heldout instruction.

Loss decline alone would have hidden that failure. The complete Stage-0 gate
caught it. No third tranche is authorized on this objective.

## Exact run lineage

### Segment 1 — steps 1 through 60

- Kaggle job:
  `c8ba53cb9f7044341fd7a9d072ea460700dbdbf457aed509acf8e2ecf3e58e72`
- source commit: `4b3bbba009b55535e71a2520779dae3f674ef332`
- packet SHA256:
  `2cc4ef394656ffc0d71e2757c70e22bc31967638e75fd0868b1085b8cfed4b3d`
- candidate generation: `r64v2-d39ec38a0f38f523`
- report ID:
  `4bec3e1fed151c13f2a7e7668a7eacb72ca1511b285a7a378e781a9b24c149de`
- final accepted bundle:
  `a3b6a5f33209cb397ab46ceeaa59a8c92714fbcf757ca1d0adb0e8b261497619`
- final checkpoint:
  `971dd55f2c865ed15a508cf41f0be4bcf1993ccd50c271093d2e23689435c5dd`

Verified metrics:

| Metric | Initial | Step 60 |
|---|---:|---:|
| Heldout mean loss | 12.9976677448 | 6.5498658717 |
| Teacher-forced payload-token accuracy | 0.00000 | 0.53125 |
| Strongest constant token floor | 0.50000 | 0.50000 |
| Free-running payload exact rate | 0.00000 | 0.31250 |
| Free-running typed-emission exact rate | 0.00000 | 0.00000 |
| Complete-field coverage | 1.00000 | 1.00000 |

The narrow task-signal gate passed because loss fell, token accuracy barely
exceeded its floor, all cases were evaluated, and the counterfactual signatures
were nonzero. The stricter Stage-0 gate correctly failed every action and every
changed-source pair.

### Segment 2 — exact continuation, steps 61 through 120

- Kaggle job:
  `3769a4caf10231a19e3bab940fdb700b2bf3eff8c208274d5232ad2070959137`
- source commit: `f9533b8de79717a71598ae5c472c048623e95c6d`
- packet SHA256:
  `dd6737311a3b84284f4093f4463099348f4ba7a94a83cbb40a13bf05561c3041`
- exact parent bundle: the segment-1 bundle `a3b6a5f...`
- report ID:
  `40fa9b4a9f3205b5801cbb91b4c6950f378b16f01b7a8b0783d0dea1b0c2a924`
- final accepted bundle:
  `17074c786ae305da67904009b5fce1b1cce70f5cdb68bf7652f641cd2c887147`
- final checkpoint:
  `baeceed582b46d0b1b4654bdb49bcbfa9c267e35fe85b8a8ce98a2b11aaa315b`
- final checkpoint artifact:
  `94fe9b730ac05b49c2f556927f88e05afff36ebc2b0d5d16f1811936957e78e4`

The initial step-61 evaluation exactly reproduced the step-60 metrics, proving
resume fidelity. Verified metrics then moved as follows:

| Metric | Step 60 parent | Step 120 |
|---|---:|---:|
| Heldout mean loss | 6.5498658717 | 5.7030190378 |
| Teacher-forced payload-token accuracy | 0.53125 | 0.50000 |
| Strongest constant token floor | 0.50000 | 0.50000 |
| Free-running payload exact rate | 0.31250 | 0.25000 |
| Free-running typed-emission exact rate | 0.00000 | 0.00000 |
| Complete-field coverage | 1.00000 | 1.00000 |

The task gate returned false. Heldout and regression free-running case exact,
changed-source-pair exact, payload-pair exact, and every per-action exact rate
were all 0.0. The 0.25 payload-exact rate is not copied content: it is the four
empty-payload delete cases terminating with EOS while the operation head still
predicted `REPLACE`.

## Mechanistic diagnosis

This result exposes two separable problems.

1. **The source-position mechanism is trainable.** On later payload-bearing
   training steps, exact source-position accuracy repeatedly reached 1.0 for
   one-cell targets and became nonzero for multi-cell Unicode targets. This is
   evidence against immediately redesigning the 64D core or exact substrate.
2. **The copy/generate gate never crossed over.** Its observed accuracy was
   exactly what is produced by classifying EOS as generated while classifying
   every payload cell as generated: 0.50 for one copied cell plus EOS, 0.33 for
   two plus EOS, 0.25 for three plus EOS. The generated branch therefore kept
   winning even when the position head knew the source address.
3. **The joint action targets have strong constant priors.** Four of the six
   equally sampled actions are DELTA, and two of those four use REPLACE. The
   heldout behavior converged to those majority choices. Decision loss stayed
   low on DELTA cases and high on no-op/abstain; operation loss stayed low on
   REPLACE and high on INSERT/DELETE.
4. **The aggregate token metric masks the distinction we need.** It combines
   payload content with EOS and includes empty-payload DELTAs. A model can
   approach its 0.50 constant floor without copying a source cell. Future
   reports must isolate content-token, EOS, copy-gate and source-position
   accuracy.

## Required next attempt

Preserve the step-120 lineage as immutable failed evidence. Start a fresh
Stage-0-v2 candidate after all of the following are implemented and tested:

1. report content-token accuracy separately from EOS and empty payloads;
2. report heldout copy/generate-gate and exact-position accuracy, including
   changed-source Unicode pairs;
3. balance DELTA / NO_OP / ABSTAIN teaching mass and INSERT / REPLACE / DELETE
   teaching mass so constant class priors cannot win the lesson;
4. make the early curriculum staged: first exact source position plus an
   explicitly copy-preferring gate, then EOS, then one control axis at a time,
   then the joint typed delta;
5. predeclare component-loss weights and gates in the content-addressed
   curriculum/policy rather than silently tuning after results; and
6. authorize a renewed tranche only when content-copy and at least one
   non-majority action show heldout changed-source signal above their declared
   constant floors.

This is decoder exposure and curriculum shaping for existing tissue. The
observed position learning does not justify an architecture change yet. If a
balanced, staged candidate still cannot move the copy gate or non-majority
actions, then compare decoder/gate architectural alternatives under the same
exact evidence surface.

## Transport and verification

Both jobs completed on real Kaggle CUDA and were imported with the new
hash-manifested bundle path. The first 1.126 GB bundle initially stopped safely
on a Windows legacy path-length error before acceptance. Commit `ec5772c`
changed local extraction I/O to extended Windows paths without shortening any
canonical Trainer/Soul name; a real >260-character regression and the complete
22-test cloud-bundle suite passed. Both production bundles then imported as
`fetch_mode=bundle`, and the accepted candidate state was copied additively
into the local non-serving Trainer store. Local latest accepted state is exact
step 120 bundle `17074c...`.

Verification in this turn: final repository suite before the extraction-only
fix, 578 passed / zero failures; post-fix cloud-bundle suite, 22 passed; Ruff,
compileall, diff checks, Source-of-Truth mirror equality, CUDA preflight and
both provider result identities passed. Kaggle reported approximately 0.41 of
30 free GPU hours used after both jobs and 29.59 hours remaining, resetting
2026-09-12. Paid spend: zero.
