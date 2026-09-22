# Proposal: Stage-0A pointer bootstrap before further sequence training

**Author:** Codex / GPT-5 / 2026-09-22 America/Chicago
**Status:** proposal only — no optimizer run authorized by this document

## Evidence requiring a different experiment

At the exact persisted step-24 checkpoint, the first response-cell source
pointer selects the required `cortex` position-0 address on 42/64 seen phases
but only 4/64 held-out phases. The copy gate stays virtually unchanged
(52.10% versus 52.06% copy), while the generated target-token path is below
0.4% on both surfaces. The evidence is recorded in
`roundtable/reviews/CODEX_STAGE0A_STEP24_PREFIX_DIAGNOSIS_20260922.md`.

The current mixed Stage-0A exact-copy curriculum has not trained a general
source-address selection operation. Another multi-character continuation would
combine pointer failure, autoregressive exposure, and EOS behavior again, so it
would not identify whether a correction worked.

## Proposed bounded transition

Create one versioned pointer-bootstrap curriculum and objective-program
transition, parented explicitly to the preserved step-24 parameter/Soul
lineage. It uses the ordinary permanent Heart/runtime/Soul circulation, the
existing D64 Core, frozen 16D substrate, canonical field compiler, and existing
position pointer. It introduces no fake runtime, alternate State, new Core, or
architecture expansion.

Each supervised FIRST/REFINED exercise asks only for the exact first transport
unit at a designated canonical `cortex` position. The corpus must vary target
native and UTF-8 byte cells, unrelated region contents and lengths, and page
boundaries. It must contain exact-text-disjoint held-out cases. The address
receipt remains the authority for the expected source; no nearest-vector or
semantic approximation is acceptable.

The runtime-facing evaluator must report at least:

- pointer top-1 accuracy at the exact source address;
- pointer probability at that address;
- free-running first transport-unit accuracy;
- nonempty valid Unicode response rate; and
- seen/held-out values separately.

The smoke gate requires all metrics to rise above their specified constant
baselines before any longer tranche. Its graduation gate requires a real
held-out pointer and free-running first-cell pass before multi-character copy
is resumed. EOS and later-token behavior remain observed but are not changed in
this intervention.

## Exclusions and decision request

Do not increase layers or FFN width, reset the Soul, weaken a mastery gate,
change the copy/generate gate loss, add a termination head, or launch Kaggle
in the same intervention. Those actions do not target the measured defect.

The table should ratify the exact curriculum manifest, objective-program ID,
smoke/pass thresholds, and whether the step-24 optimizer state transitions
under the existing governed transition mechanism or a fresh candidate is born.
After that decision, implement the smallest complete trainer/evaluator/test
change and run its local smoke before considering external compute.
