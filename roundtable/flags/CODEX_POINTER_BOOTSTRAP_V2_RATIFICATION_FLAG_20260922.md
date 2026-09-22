# FLAG [BLOCKING] — pointer-bootstrap v1 cannot govern another tranche

**Author:** Codex / GPT-5 / 2026-09-22 America/Chicago  
**Mission item:** diagnose the failed step-25 pointer objective and choose the
next correction.

## Problem

The ratified `pointer_bootstrap_native_v1` curriculum and gate contain three
material defects:

1. train omits positions 7, 32, and 63 while heldout requires them;
2. source selection depends on untrained English decimal-address grounding, so
   the exercise does not isolate the pointer primitive; and
3. the gate requires mean exact-source softmax probability of exactly 1.0.

Changing the curriculum manifest or gate silently would violate the Working
Contract. Another optimizer tranche under v1 would repeat an experiment whose
geometry is now falsified.

## Evidence

- Static enumeration proves train covers six positions and heldout covers nine.
- The accepted step saw only eight episodes: four characters over six
  positions.
- Exact step-25 CUDA replay scored 0/12 pointer top-1 on a balanced seen sample
  and 0/24 on heldout FIRST phases.
- Changing only the requested decimal position produced pointer-distribution
  total variation of `0.000036` to `0.000096`; argmax stayed at Cortex 64 or 65.
- Full evidence is in
  `roundtable/reviews/CODEX_POINTER_BOOTSTRAP_STEP25_GEOMETRY_AUDIT_20260922.md`.

## Options

1. **Ratify a marker-grounded pointer primitive, then numeric address
   grounding (recommended).** Parent a new governed curriculum transition from
   the preserved step-24 checkpoint/Soul. First select a readable marker-bound
   Cortex cell at every declared position, with disjoint characters and
   contexts plus marker-swap counterfactuals. Only after that passes, teach
   decimal-address language as a separate stage. Replace the exact `1.0`
   probability-mean gate with calibrated evidence or a finite ratified
   threshold; keep exact top-1/output graduation gates.
2. Retain decimal instructions but add an address-language curriculum before
   pointer bootstrap. This preserves the final interface but is a larger first
   experiment and still requires complete train-position coverage and a
   corrected gate.
3. Retain v1 unchanged. This is not recommended: it cannot distinguish pointer
   capacity from address-language failure and contains an effectively
   saturating probability requirement.

## Recommendation

Ratify option 1. It tests the smallest real mechanism without adding a hidden
teacher channel, fake architecture, alternate body, or substitute trainer.

## Work halted

No curriculum, gate, trainer, checkpoint, Soul, canonical state, or optimizer
was changed. No local or cloud training was launched.

## Work continued

The canonical step-25 checkpoint/Soul was verified and replayed read-only; the
curriculum geometry, actual sampled experiences, position-wise pointer behavior,
and prompt-number counterfactuals were recorded.

