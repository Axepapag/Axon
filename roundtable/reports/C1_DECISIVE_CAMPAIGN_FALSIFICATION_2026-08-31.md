# C1 Decisive Bounded Campaign — Predeclared Falsification Contract

Author: Kimmy / Kimi Code CLI / 2026-08-31
Requested by: Jeff (convener), "make it so", 2026-08-31
Status: PREDECLARED BEFORE LAUNCH. Written before any C1 optimizer step. These
thresholds cannot be revised after results are observed except by a new Jeff
decision recorded as a new event; the original text remains immutable history.
Reference: capability trend flag in evt-20260830T203524937304Z (reality
assessment) and the communication-first contract
`docs/COMMUNICATION_FIRST_CURRICULUM.md`.

## 1. What this campaign is

One bounded, decisive experiment answering a single question:

> Can the living D64 anatomy (1x64, 2x32, 4x16 heads; two layers; FFN 131072;
> four persistent Soul tokens) learn the C1 hidden-target single-turn
> communication contract at all — measurably better than the strongest constant
> predictor — within a small, fixed step budget?

This is not a serving bid. No promotion, activation, or capability claim can
come out of this campaign regardless of outcome. The exact serving gate and
the complete tournament metric surface remain the separate promotion bar.

## 2. Fixed campaign parameters

- Curriculum: C1 manifest
  `c04ae8c6815e93b31136fdf833a525388c8439e74fd5317268cb1e1ba1233212`
  (36 cases: 24 train / 6 heldout / 6 regression; hidden targets; whole-lineage
  splits; compile-time leakage verification, zero leaks).
- Candidates: the three predeclared head geometries, identical everything
  else — same curriculum, optimizer, learning rate, seed, tranche schedule,
  evaluation breadth.
- Tranches: 64 accepted steps each, at most 10 tranches per candidate
  (640 steps hard ceiling). Evaluation at every tranche boundary over the
  complete heldout + regression breadth (no case deferral).
- Device: local CUDA (GTX 1650) or CPU fallback; local only, zero cloud spend.
- Early stop per candidate on SIGNAL (§3) to save compute; the falsification
  clock runs to the ceiling otherwise.

## 3. SIGNAL — minimum evidence to continue past this campaign

All three, measured on whole-lineage heldout material:

1. Teacher-forced payload token accuracy decisively exceeds the strongest
   heldout constant-category floor: accuracy ≥ floor + 0.15 absolute.
2. Heldout mean loss below the campaign baseline (pre-step-1 evaluation).
3. Nonzero causal dependence: at least one field/Soul/proposal counterfactual
   measurably changes behavior (existing counterfactual gate machinery).

## 4. CAPABILITY INDICATOR — not required, but watched

- Free-running exact typed emission or exact payload transport on at least one
  heldout C1 case (rate > 0). Zero here with SIGNAL present means "learning
  but not yet exact" — a curriculum/step-count question, not falsification.

## 5. FALSIFICATION — predeclared

The from-scratch C1 communication hypothesis is falsified **at this budget**
if, at the 640-step ceiling for all three geometries:

- no candidate satisfies SIGNAL (§3), and
- exact heldout rates remain 0.0.

Falsification consequences, predeclared and binding on the next planning turn:

1. No silent budget extension, no architecture swap mid-stream, no gate
   redefinition. Any continuation requires a new Jeff decision with a stated
   reason (e.g., curriculum defect found, signal trajectory strongly positive
   but sub-threshold).
2. The default pivot on falsification: Axon continues as the proven
   deterministic exact-state organism (Heart/Souls/Dormant/Trainer governance)
   with external-model reasoning cores behind the Heart boundary; the learned
   D64 core program is shelved with return conditions, exactly as the Heart
   translator was shelved before it.

## 6. Geometry comparison rule

If SIGNAL is met by more than one geometry, the tournament winner for the next
stage is the geometry with the highest heldout token accuracy per accepted
step (sample efficiency), ties broken by lower heldout loss. If no geometry
meets SIGNAL, there is no winner and nothing advances.

## 7. Reporting

Every tranche boundary produces a durable segment report. The campaign closes
with a single report stating: SIGNAL met/not per candidate, CAPABILITY
INDICATOR values, exact numbers vs thresholds, spend (wall time, GPU hours),
and the falsification verdict against this document's thresholds — quoted, not
paraphrased.

— Kimmy / Kimi Code CLI / 2026-08-31
