# FLAG [BLOCKING] — pointer-bootstrap manifest and gate ratification

**Author:** Codex / GPT-5 / 2026-09-22 America/Chicago  
**Mission item:** implement and locally prove the Stage-0A pointer-bootstrap
repair proposed in `CODEX_STAGE0A_POINTER_BOOTSTRAP_PROPOSAL_20260922.md`.

## Problem

The proposal asks a supervised FIRST/REFINED exercise to return one exact
transport unit, including UTF-8 byte cells. The active public target contract
requires every emitted target to be nonempty valid Unicode. A non-ASCII UTF-8
byte is not, by itself, valid Unicode text. Treating such a byte as an English
proposal would violate the exact-Unicode contract; silently substituting an
escaped spelling would stop testing the real frozen transport unit.

The prior proposal also deliberately reserved the new curriculum manifest,
objective identity, smoke thresholds, and graduation gate for table
ratification. The working contract forbids an engineer from changing a gate in
a working session. No resolution ratifying those details is present in the
Roundtable material read for this turn.

## Evidence

- `training/living_reasoning_curriculum.py` rejects nonempty targets that are
  not exact Unicode text.
- `docs/SOURCE_OF_TRUTH.md` requires non-native scalars to use their exact
  one-to-four UTF-8 transport cells and strict decoding.
- `docs/WORKING_CONTRACT.md` sections 1 and 4 require a HALT AND FLAG when
  work would redefine a gate or contract without table authority.

## Options

1. **Ratify a native one-cell pointer bootstrap first (recommended).** Train
   only non-whitespace native one-cell scalar outputs at varied exact Cortex
   addresses, including positions around logical page boundaries. Hold out
   target characters entirely, use the normal Heart/Soul/field path, and
   require heldout source-pointer top-1, source probability, and free-running
   first-scalar evidence before resuming sequence copy. A follow-on stage then
   copies complete Unicode scalars so all UTF-8 cells are trained while the
   public response remains valid text.
2. Ratify an internal raw transport-unit auxiliary objective. This would need
   a versioned objective design and dedicated evidence showing it cannot be
   confused with a public response; it is more invasive and should not be
   invented while repairing Stage-0A.
3. Keep the current mixed full-copy curriculum. This repeats the experiment
   whose heldout source-pointer defect is already measured and does not
   isolate the mechanism.

## Recommended ratification text

> Approve `pointer_bootstrap_native_v1` as a curriculum transition from the
> accepted step-24 candidate/Soul lineage. It may use the existing
> `terminal-route-balanced-v1` objective without changing component weights.
> Its supervised outputs are non-whitespace native one-cell Unicode scalars at
> designated exact Cortex positions; normal FIRST/REFINED/Soul circulation,
> exact address alignment, and a heldout target-character split are required.
> Its smoke report must separately expose seen and heldout first-cell
> source-pointer top-1, exact-source probability, free-running first-scalar
> accuracy, and nonempty valid-Unicode rate. The table will ratify numerical
> pass thresholds in the implementing resolution before an optimizer step.

## Work halted

No trainer, objective, curriculum, evaluator, gate, optimizer, checkpoint,
Soul, Heart, or Kaggle state was changed. In particular, no invalid raw byte
was represented as a public English response.

## Work continued

The exact step-24 pointer diagnosis and the active target contract were
re-read. Codex's personal operating policy was also updated outside `D:\Axon`
to preserve the user's instruction to prefer verified system-wide tool
installations.
