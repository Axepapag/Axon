# D64 Receipt-Continuation Kaggle Ablation — Result Adjudication

Date: 2026-09-09
Identity stamp: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-09

## Disposition

**The ratified receipt mechanism passed its intended architectural ablation, but
the learned D64 candidate did not pass its curriculum stage or task gate.**

This is meaningful progress. The historical multi-cell Unicode position/content
failure is gone on both complete heldout and regression examinations. The
remaining learned failure moved to the copy/generate route decision and payload
termination. The candidate remains paused, renewable, and non-serving.

## Immutable evidence

- Cloud job:
  `4b228ccd6dcf2bfdddb09eed59d293c162b18486bd20f42ab70d52721c37f669`
- Candidate: `r64v3-075d61127ec9f3b8`
- Architecture: `living-d64-receipt-823973aed39c1fe14276d2c3`
- Geometry: D64 / 1 head / 4 layers / FFN256 / 331,319 parameters
- Exact checkpoint:
  `7a94d1934c2dd1999d3de1c8b33965b06d2eae6831529a1943dc7f63f46bf45c`
- Report: `State/training/cloud/jobs/4b228ccd6dcf2bfdddb09eed59d293c162b18486bd20f42ab70d52721c37f669/outputs/axon_job/State/training/reasoning/r64v3-075d61127ec9f3b8/segment_000000001_000000120.json`
- Report ID:
  `f19c30f3a47d8a1e519450a12c59b454843ed4e1d505cc4edad5aa256ce81c2d`
- Report file SHA256:
  `dafbeecb4764669591113aeec992fa0a6b9abee3f33f68164408c0c75f8d6280`
- Fetched output archive SHA256:
  `422767f2c7bf340f67a05e85ac0e659e20d9fbfc355e74fac1ce727a70a559f8`
- Fetch mode: bundle-first, full rehash, completed return code 0

## What changed

| Metric | Previous D64 Unicode-walk v3 | Receipt D64 | Meaning |
|---|---:|---:|---|
| Heldout source-position accuracy | 0.645 | **1.000** | Exact learned anchor selection now generalizes. |
| Regression source-position accuracy | 0.657 | **1.000** | The result is not confined to the training-like split. |
| Heldout payload-content accuracy | 0.645 | **1.000** | Under the supervised decoder path, complete Unicode content is exact. |
| Regression payload-content accuracy | 0.657 | **1.000** | Exact content survives changed-source regression. |
| Heldout copy-route gate | 1.000 | **0.000** | The candidate no longer chooses the copy route. |
| Regression copy-route gate | 1.000 | **0.000** | The route failure generalizes too. |
| Heldout EOS-route gate | 0.000 | **1.000** | EOS is now classified as a generate event rather than a copy event. |
| Regression EOS-route gate | 0.000 | **1.000** | The EOS-route correction generalizes. |
| Heldout emitted EOS token | 0.000 | **0.125** | Still far below the 0.95 gate. |
| Regression emitted EOS token | 0.000 | **0.208** | Still far below the 0.95 gate. |

Heldout mean loss fell from `2.382640` to `0.732245`, a 69.27% reduction. This
supports that optimization occurred; it is not a promotion criterion.

## Plain-language interpretation

Before receipt continuation, the core could often find the first 16D transport
cell of a multi-cell character but lost the exact intra-character path. It
revisited the first cell instead of walking the remaining cells. The new
content-addressed receipt gives a validated, scalar-local route through those
remaining cells. The candidate now demonstrates exact anchor position and exact
supervised content across both examinations.

The current core nevertheless does not autonomously use that route. Its learned
gate selects generate instead of copy on every supervised copy anchor. The QA
transcripts consequently show the core mostly abstaining and emitting an empty
payload. In analogy: Axon now has a trustworthy road and can identify the right
address, but this checkpoint does not choose to drive onto the road.

The current training overlay placed effective weights of `4.0` on source
position, `0.25` on the copy-route gate, `1.0` on payload, and `1.0` on the EOS
route. The result is consistent with that imbalance: position saturated while
the shared route logit learned the EOS-positive side and failed the
content-negative/COPY side. This is evidence for a targeted objective rebalance,
not evidence that D64 or the receipt architecture lacks capacity.

## Gate truth

- Complete heldout evaluation: **true**
- Complete regression evaluation: **true**
- Copy-alignment curriculum stage gate: **false**
- General task gate: **false**
- Curriculum stage complete: **false**
- Campaign complete: **false**
- Serving promotion claimed: **false**

The report's `exact_serving_gate_passed: true` must not be read as serving
readiness. That gate currently asks only whether free-running typed-emission and
payload exactness exceed constant zero floors. Both aggregate rates were 0.333,
partly benefiting from empty/abstain-like cases, while every non-abstain action
remained inexact. The name is stronger than the policy. The authoritative
curriculum and task gates correctly rejected the candidate.

## Recommended next shot

1. Preserve this exact step-120 checkpoint and all result evidence.
2. Correct the misleading exact-serving gate name or policy so a nonzero
   aggregate cannot look like serving approval.
3. Define a new content-addressed objective identity that restores strong
   copy-route pressure while retaining EOS and exact receipt continuation.
4. Run a bounded renewable continuation from the exact checkpoint and require
   heldout and regression copy-route, position, content, and emitted-EOS metrics
   to clear their 0.95 gates together.
5. Do not advance to later motor stages, promote, serve, or infer that FFN256 won
   the architecture tournament until those gates pass.
