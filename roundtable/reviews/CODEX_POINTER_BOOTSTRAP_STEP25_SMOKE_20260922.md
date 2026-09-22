# Review: pointer-bootstrap step-25 local CUDA smoke

**Author:** Codex / GPT-5 / 2026-09-22 America/Chicago  
**Result:** failed smoke — no further optimizer step authorized by this review

## Lineage

The smoke resumed the real candidate rather than creating a substitute body:

- candidate: `english-candidate-1c991f8c911f79394e91`;
- exact parent bundle: `1d77a96509993005e03366066fdd43f1143adb8895ba470b62f1a300aff7aeb5`;
- parent checkpoint: `ef373220713565efa9da7b6c7877ffb5de75ab4522513ce679945102690ced3e`;
- parent Soul: `308ff2a2b5de6337e904733a04da001af613faa7faa364a984150b3837753a0a`;
- resumed architecture: D64, one head, two layers, FFN 16,384, page size 32;
- unchanged learning-policy identity:
  `db330c4d5dbd4bb6a549be0367eb8a84341ca24b6cd053089528d2b98f88d7a9`.

The immutable curriculum-transition receipt is
`8880eef5893bbb75f9471d40b592be52545fa3997e992547faca9d196d52eacd`.
All six preflight checks passed before the optimizer existed.

## One accepted step

Local CUDA accepted exactly one optimizer step, to global step 25:

- optimization receipt:
  `d51ac028b8288b531ddec56506b0a0dc2a1bdef5eca57ffcd5d621f47981591e`;
- checkpoint:
  `1b0282b026794109a5c574ed211f7c835ab05451e3785b2b7bf71d44d723cfe7`;
- candidate Soul successor:
  `13cf886a5a083200816c4487b12edbeb434500993dc8816ce6a374f0abe8f19e`;
- mean loss: `14.224648475646973`.

The heldout gate failed. Across 24 heldout episodes / 48 supervised
FIRST+REFINED phases:

| Measure | Result |
|---|---:|
| exact-source pointer top-1 | 0 / 48 (0.0%) |
| exact-source pointer probability mean | 0.02317% |
| free-running first transport unit | 2 / 48 (4.167%) |
| nonempty valid-Unicode output | 3 / 48 (6.25%) |
| complete-field coverage | 72 / 72 (100%) |

The gate correctly remained false. Its result is not a promotion or
competency claim.

## Parent comparison

A read-only evaluation of the exact step-24 checkpoint/Soul on the first two
deterministic heldout exercises found zero pointer top-1, zero free-running
first-unit accuracy, and zero nonempty valid-Unicode output in both cases.
After the one update, those same two cases remained zero on those three
measures. Exact-source probability changed from `0.00445% -> 0.00628%` for
the first and `0.01248% -> 0.02384%` for the second. Two samples with no
top-1 or emitted-output improvement are not evidence of a repair.

## Decision

Keep the step-25 checkpoint and Soul as recoverable evidence, but stop this
tranche. Do not launch Kaggle or a longer local run. The next work is a
read-only diagnostic of the pointer objective/curriculum geometry and a new
proposal only if that evidence identifies a narrower correction.
