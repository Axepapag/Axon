# Living D64 Candidate A — Engineering Report

Date: 2026-08-28
Identity stamp: Codex / GPT-5 lineage / 2026-08-28
Status: mechanism-functional, trained for one bounded step, rejected for serving

## What is now real

Axon has one implemented neural reasoning candidate whose training anatomy
matches the permanent Heart circulation surface:

- canonical Shared Field schema v3 has the append-only `identity` region at
  region ID 10;
- every core owns a content-addressed private `HOT` / `WARM` / `COLD` /
  `DEEP_COLD` Soul;
- FIRST, REFINED, and CONSOLIDATED are causal Soul boundaries;
- the same frozen canonical field and derived proposal workspaces are swept in
  ordered pages, with pages acting only as bounded compute units;
- the core predicts a typed decision, operation, region, exact dynamic
  boundary addresses, 351 Unicode transport categories, EMPTY, and EOS;
- candidate parameters and candidate Soul state are isolated from live state.

The implementation does **not** register an untrained candidate with the live
Heart and does not claim useful reasoning, conversation, identity, or serving
capability.

## Candidate A anatomy

| Property | Value |
|---|---:|
| `d_model` | 64 |
| attention heads | 1 x 64D |
| Transformer layers | 2 |
| FFN dimension | 131,072 |
| persistent Soul-state tokens | 4 |
| default physical page | 32 characters/transport positions |
| trainable parameters | 33,981,879 |
| FP32 parameter bytes | 135,927,516 |
| FP16 parameter bytes | 67,963,758 |

The page size is not a context ceiling. The recurrent state, already seeded by
the core's Soul, traverses every eligible canonical page and then every
proposal-workspace page.

## Exact Soul contract

Soul storage is opaque and private. The Candidate-A codec uses a strict custom
little-endian float32 tensor frame rather than pickle. The frame binds its
architecture and tensor layout. Runtime phases must update HOT. Colder changes
require adjacent, evidence-vetted promotions, and DEEP_COLD additionally
requires validation evidence before it can be considered for future governed
LoRA distillation.

The runtime commits FIRST and REFINED Soul transitions before closing their
barriers. The consolidator transition is prepared before the canonical field
commit and finalized against the exact successor field. Restart recovery uses
the canonical journal to complete the prepared Soul transaction without
double-applying it.

## Trainer and curriculum surface

The first deterministic curriculum is intentionally a mechanism curriculum. It
covers addressed head/middle/tail edits, multi-page sources, Unicode payloads,
no-op/abstain, proposal refinement, conflicts, and the rule that current
canonical evidence outranks stale proposals. It is not a substitute for
high-quality lived-experience targets.

The launch preflight produces the six evidence classes required by Trainer
authority:

1. static capacity-poison scan;
2. architecture-capacity evidence;
3. train/heldout curriculum distributions;
4. page/boundary/full-coverage evidence;
5. field and private-Soul counterfactual dependence;
6. strict checkpoint compatibility and rejection.

The architecture tournament holds layers, FFN, curriculum, optimizer policy,
and gates fixed while comparing 1x64, 2x32, and 4x16 attention. Follow-up depth
and FFN ablations are predeclared rather than improvised after seeing results.

## Verified full-size smoke

The full exact Candidate-A architecture passed preflight and completed one
governed CPU optimizer step in an isolated temporary workspace. No live state
or serving pointer changed.

- workspace: `C:\Temp\axon-full-smoke-01`
- candidate: `r64a-smoke-fdc697719d4b258f`
- preflight receipt:
  `ea9ca34f4ea2c22003416140841d424512e33f0d456c6012fbf479265430a51c`
- optimizer receipt:
  `1570ce75bd843901df0411a625815b3391fd1c7bcf51a57be760b66e929f8258`
- checkpoint:
  `25986ceebbeb07d1a00204f64a334de4fbe122a5bb43dbd1391d1310919203c1`
- causal Soul receipts:
  `21293f2cda2b0f71043e00d650536a8b453b1be74617030d7bedf383fbf0bda9`,
  `be50a90cce38186880a83c92e42f22a2a936045586d650c3caaa84e4093d8a79`,
  `d37e8cd98f3065a9b00a6fe59acc4bd996ecf491d8a5de4bb9316a58b7db6c42`
- final candidate Soul:
  `f96f3959c696698d9f609e37dd3ae2e44b19928a5d59b2ff8bbfa3a96a398c49`
- training loss: `7.445512294769287`
- heldout mean loss: `7.443814754486084`
- report identity:
  `00b15867278f9865d947cbfbae98a6ad7a99a74433ac83d13da3b37e8f3c35f0`

The nearly unchanged heldout loss is expected from one step and is not treated
as a capability gain. The proof is that exact full anatomy can instantiate,
execute the three phases with causal Soul state, backpropagate, checkpoint, and
produce governed evidence on the current CPU.

An earlier reduced diagnostic failed before optimization because a verbose
candidate-generation name exceeded Windows path length. The launcher now uses
a short content-derived candidate identity; the rerun succeeded. The failed
workspace was preserved as evidence.

## Existing-checkpoint compatibility

Adding Identity creates an eleventh region-embedding row. Old ten-region D64
checkpoints are not silently loaded or discarded. The explicit migration
accepts only the exact 10xD to 11xD change, preserves all old rows byte-for-byte,
appends a zero Identity row, extends matching optimizer moments with zeros, and
emits a migration receipt before strict loading. Any other mismatch fails.

## Current blockers before long or serving training

1. The synthetic mechanism curriculum does not provide high-quality lived
   reasoning targets. Runtime outcomes need deterministic adjudication and
   counterfactual construction.
2. Candidate parameter checkpoints and candidate-Soul HEADs need one durable
   accepted-step bundle and resume protocol before a long campaign. Current
   failure cannot corrupt live state, but can orphan the latest candidate work.
3. Learned emissions must be rejected if their addressed range is outside the
   exact attended interval or authority of the proposing core. The final Heart
   validator remains the authority, but this invariant needs explicit serving
   tests for masked-address proposals.
4. The runtime reasoning-commit/autobiography deposit window and automatic
   tool/Trainer outcome hooks still need crash reconciliation and complete
   wiring.
5. One successful optimizer step proves executable anatomy, not learned use of
   the Soul, field, proposals, Unicode decoder, or exact addresses. Those need
   heldout and intervention gates.

## Recommended next shot

Freeze this implementation as the first living Candidate-A baseline. Next,
build the durable parameter+Soul step bundle and a small runtime-faithful
lived-episode curriculum with explicit success/correction evidence. Then run
the predeclared 1x64/2x32/4x16 tournament for enough bounded steps to observe
heldout exact-delta, source-use, Soul-use, proposal-use, Unicode, no-op, replay,
and CPU-latency metrics. Promote nothing until all serving gates pass.
