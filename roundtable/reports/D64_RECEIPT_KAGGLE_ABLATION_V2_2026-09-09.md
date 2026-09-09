# D64 Receipt-Continuation Kaggle Ablation v2 — Result Adjudication

Date: 2026-09-09
Identity stamp: Antigravity / Gemini 2.5 Pro / 2026-09-09
Reviewed and corrected: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-09

## Disposition

**The balanced objective profile (`route_eos_balanced_v2`) recovered copy-route
alignment from 0.000 to 1.000 while retaining 1.000 source-position and
teacher-forced content-only accuracy. It did not recover complete output:
EOS-route accuracy fell from 1.000 to 0.000, teacher-forced token accuracy
remained 0.475, and exact terminated payload transport fell from 0.333 to
0.000.**

The final gate parameters are consistent with an undifferentiated route logit,
but the proposed scalar-bias explanation is a hypothesis until a direct
counterfactual or per-position logit probe demonstrates causality.

The candidate remains paused, renewable, and non-serving.

## Immutable Evidence

- Cloud job: `2f5687a63bd691a5c1b3af6823fabaebbae927ad1b74f43271fb5745edc94a5a`
- Candidate: `r64v3-a80504169350a5ce`
- Architecture: `living-d64-receipt-823973aed39c1fe14276d2c3`
- Geometry: D64 / 1 head / 4 layers / FFN256 / 331,319 parameters
- Git revision: `95e20482138f64e4af2bcd9ed5825d61184f3aed`
- Final checkpoint: `a91fcd16b5870661a82940c8ea11f27fd90502cf6ec2b5caf78d9faedf07e487`
- Inspected step-120 checkpoint artifact:
  `State/training/cloud/jobs/2f5687a63bd691a5c1b3af6823fabaebbae927ad1b74f43271fb5745edc94a5a/outputs/axon_job/State/training/trainer/candidates/reasoning-d64-axon-d64-mixer-4l-ffn256-h1-receipt-v1/r64v3-a80504169350a5ce/checkpoints/22af77fa0bcedb396f1b1aea189863a2e4299ee33a19ab198cfa6d4abfa7088d.pt`
- Report: `State/training/cloud/jobs/2f5687a63bd691a5c1b3af6823fabaebbae927ad1b74f43271fb5745edc94a5a/outputs/axon_job/State/training/reasoning/r64v3-a80504169350a5ce/segment_000000001_000000120.json`
- Report ID: `685024aeda64fe8fc71db334471403fff42f56187bd0d84066886448a5a46b09`
- Report file SHA256: `28184f7950dbfddf3efeda8cd28aa5edd4846bcdbac874902b6f1e97180b4607`
- Fetched output archive SHA256: `c48c6740877f46c05761ce14a25965956ad7926acceeb1a3d665c2531b718b7c`
- Fetch mode: bundle-first, full rehash, completed return code 0

## Comparison Across Runs

| Metric | Unicode-walk v3 (`050a3a97...`) | Receipt Ablation v1 (`4b228ccd...`) | Receipt Ablation v2 (`2f5687a6...`) | Meaning |
|---|---:|---:|---:|---|
| Objective Profile | Standard | `continuation_v1` (Copy 0.25, Pos 4.0, EOS 1.0) | `route_eos_balanced_v2` (Copy 4.0, Pos 1.0, EOS 2.0) | Rebalanced copy/position/EOS weights |
| Heldout source-position accuracy | 0.645 | **1.000** | **1.000** | Exact source pointer held at 100% |
| Regression source-position accuracy | 0.657 | **1.000** | **1.000** | Generalizes to changed-source regression |
| Heldout teacher-forced content-only accuracy | 0.645 | **1.000** | **1.000** | Content cells are exact when the correct decoder prefix is supplied; EOS is excluded |
| Regression content-only probe | 0.657 | **1.000** | **1.000** | Content-only alignment survives changed-source regression |
| Heldout copy-route gate | 1.000 | 0.000 | **1.000** | **Recovered from 0.000 to 1.000 (100%)** |
| Regression copy-route gate | 1.000 | 0.000 | **1.000** | **Recovered from 0.000 to 1.000 (100%)** |
| Heldout EOS-route gate | 0.000 | **1.000** | 0.000 | Inverted: gate predicts COPY at EOS position |
| Regression EOS-route gate | 0.000 | **1.000** | 0.000 | Inverted: gate predicts COPY at EOS position |
| Heldout emitted EOS token | 0.000 | 0.125 | 0.125 | Nonzero, but below 0.95 gate |
| Regression emitted EOS token | 0.000 | 0.208 | **0.333** | Modest improvement |
| Heldout teacher-forced token accuracy | 0.364 | 0.475 | 0.475 | Content plus EOS; no v1-to-v2 gain |
| Heldout exact terminated payload transport | 0.000 | **0.333** | 0.000 | v2 regressed on complete free-running payload transport |
| Heldout typed-emission exact rate | 0.000 | **0.333** | **0.333** | Exact supervised phases; flat from v1 to v2, not complete payload success |
| Heldout mean loss | 0.5760 | 0.7322 | **0.4461** | v2 is 22.55% below the walk-v3 final loss and 81.28% below its own 2.3826 step-zero loss |

## Plain-Language Analysis

1. **Observed Route Flip and Current Hypothesis**:
   In Ablation v1, the objective placed weight 0.25 on copy and 1.0 on EOS, while position was 4.0. The gate logit was pushed positive, causing the candidate to predict `GENERATE` on every step (EOS=1.000, Copy=0.000).
   In Ablation v2, the objective placed weight 4.0 on copy and 2.0 on EOS.
   Multi-cell training can contribute multiple copy positions for one EOS
   position, so aggregate copy pressure is a plausible cause of the flip. The
   run did not record the gradient decomposition needed to prove that cause.
   Parameter inspection of the final checkpoint confirms:
   - `copy_gate.weight` norm is only `0.09427` (after 120 steps at lr=1e-4, linear projection weights have barely moved from zero).
   - `copy_gate.bias` shifted to `-0.00978`.
   Ablation v1's step-120 gate bias was positive (`0.01111`); v2's was
   negative (`-0.00978`). The observed all-GENERATE versus all-COPY behavior is
   consistent with bias-dominated routing, but weight norm alone does not prove
   that the bias dominates every position. A bias-clamp or per-position-logit
   counterfactual is required before treating this as the root cause.

2. **What the Evidence Supports**:
   - Receipt continuation retained 1.000 source-position and teacher-forced
     content-only accuracy across complete heldout and regression probes.
   - The D64 mixer can reach 1.000 copy-route alignment under this objective.
   - Neither result is exact emitted transport: v2 teacher-forced token accuracy
     is 0.475, EOS accuracy is 0.125 heldout / 0.333 regression, exact terminated
     payload transport is 0.000, and QA payload cases remain empty/unterminated.
   - v2 heldout loss fell from its own step-zero value 2.3826 to 0.4461; the
     comparable walk-v3 final loss was 0.5760.

## Gate Truth

- Complete heldout evaluation: **true**
- Complete regression evaluation: **true**
- Copy-alignment curriculum stage gate: **false** (EOS gate failed)
- General task gate: **false**
- Nonzero exact output observed: **false**
- Exact serving gate passed: **false**
- Serving promotion claimed: **false**

`typed_emission_exact_rate = 0.333` counts exact supervised phases, including
empty/abstain-like phases. It is not evidence of an exact non-empty terminated
payload. Therefore it is consistent with
`nonzero_exact_output_observed = false` and
`payload_transport_exact_rate = 0.000`.

## Engineering Recommendation

1. **Do not abandon D64 or declare an architecture failure**:
   The core demonstrated 1.000 position and teacher-forced content-only
   accuracy under receipt continuation. Complete terminated transport remains
   failed, and v2 regressed on that metric relative to v1.
2. **Structural Fix for Copy vs. EOS Gate**:
   First run a bounded diagnostic that records per-position route logits and
   compares the untouched checkpoint with a zero-bias clamp. If that falsifies
   bias dominance, inspect contextual separability before changing architecture.
   If it confirms bias dominance, compare a longer tranche and non-zero gate
   initialization under new, content-addressed objective/candidate identities.
3. **Immediate Priority for Google Cloud Meeting**:
   Do not attempt an unverified architecture migration before the Google meeting. The deterministic components (Heart safety, autobiographical memory, spot tranche recovery) are production-ready for demonstration.
