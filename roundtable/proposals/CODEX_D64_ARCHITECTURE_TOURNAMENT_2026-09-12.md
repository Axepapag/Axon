# D64 Living-Core Architecture Tournament

**Author:** Codex / GPT-6

**Date:** 2026-09-12

**Status:** Implemented and packetized; Kaggle launch gated by authenticated mid-run recovery

## Decision

Axon will search the D64 Living-core family in stages. The declared space is
48 legal shapes:

- physical layers: 2, 5, 10;
- attention heads: 1, 2, 4, 8;
- private FFN width: 4,096, 16,384, 65,536, 131,072;
- D64, four Soul state tokens, 32-character pages, exact receipt-aware Unicode
  transport, and zero dropout held constant.

Ten heads is excluded because standard multi-head attention requires 64 to be
divisible by the head count. Eight heads supplies the intended high-head test
with eight dimensions per head.

The first learning stage is a balanced 16-shape screen. It covers every factor
level and adds four boundary anchors. Candidate sizes range from 1,222,329 to
169,384,121 trainable parameters; the mean is 35,130,229. This makes the screen
roughly comparable in total parameter-steps to training sixteen copies of the
current 33,982,137-parameter reference for the same number of steps.

## Stages

1. **Balanced causal screen:** 16 candidates, one seed, 32 optimizer steps,
   retain at most eight. This is a diagnostic opening, not competence or
   homework completion.
2. **Multi-seed learning:** the eight evidence-ranked survivors, three seeds,
   256 optimizer steps, retain at most three.
3. **Runtime assignment final:** three survivors, three seeds, renewable
   1,024-step resource tranches. A winner must pass complete homework, causal
   emission, Soul dependence, current-field override, proposal refinement,
   regression, checkpoint replay, and interruption recovery gates.

No shape wins because it is larger, faster, or lucky on one seed. Promotion
requires comparable evidence on the same immutable curriculum. Consuming a
step allowance never means that an assignment was completed.

## Local evidence

The new campaign identity is
`c4ebb873afb8a33c6c3f4e4e7cd3d7c23951f33a35ce6ef7470609fb4dcdbeab`.
The balanced screening tournament identity is
`cea217a0025fc9fa2e42a0d0c83b50eb77bb923e72114209684f52b0bb83394e`.

The 1.22M boundary and a 169.38M boundary both passed the real
LivingReasoningCoreD64 canonical-State preflight on the same foundation motor
manifest. The resulting observation is
`d2bf59af35fd432d66ead4cb5da27b1efcc77feb1d79adf94a94accb02cddbd6`.
It explicitly reports no tournament metrics and makes no promotion claim.

## Kaggle recovery boundary

Jeff removed mid-run sync from the stage-one launch gate on 2026-09-13. The
User Secret is not required to train, and the bounded 32-step candidate screen
does not justify blocking the campaign on kernel-to-workstation checkpoint
uploads. Kaggle's completed private output bundle is the recovery boundary for
this stage. Every fetched bundle is hash-verified before inspection or use.

An interrupted in-flight candidate can lose its current 32-step tranche. It
must be rerun from the same immutable recipe and seed; partial observation bytes
never authorize continuation or promotion. Mid-run sync remains an optional
facility for later long-duration runs where its recovery value exceeds its
credential and operational cost.

The older prepared job
`35c5c22b2e4e14c1db36b61306ec507bc79efbe72997cdcc43561a3dfab7a741`
predates the KGAT correction and this operator decision. It must not launch. A
fresh packet must be prepared from the current committed revision.

The later monolithic 16-candidate run filled Kaggle's notebook disk during its
tenth candidate and could not publish a detached output manifest. Stage one is
therefore executed as 16 single-candidate recipes under
`configs/kaggle/d64_architecture_screen_stage1_shards/`, with two checkpoint
landmarks per shard. Candidate results are aggregated only after every shard's
completed bundle is fetched and hash-verified locally.
