# Resolution — D16 Core Bus B3 Live Circulation

Date: 2026-09-24
Authority: Jeff, project convener
Status: RATIFIED / IMPLEMENTED / VERIFIED

## Decision

B3 is accepted as the live serving transition for the continuous reasoning-Core family.

The existing Heart reasoning cadence remains intact:

`FIELD_DELTA -> FIRST barrier -> PROPOSAL_SET -> REFINED barrier -> FINAL -> Heart commit -> CANONICAL_SYNC`

What changes is the transport seam. A continuous Core no longer requires a width-specific packed rail matching its private `d_model` in order to participate in reasoning. Heart may synchronize that Core through the exact D16 Core Bus, prove mirror coherence, and issue a `ReasoningPassRequest` bound to a `D16RuntimeBinding`. Sibling proposal workspaces are transported to D16-capable Cores as exact `D16TextFrame` values rather than being repacked to the Core's hidden width.

Legacy rail-bound ports remain supported during the migration. Their D64 path, compiler, codecs, semantic surfaces, tests, and historical checkpoints remain preserved and independently testable.

## Verified B3 behavior

The implementation now supports mixed transport modes in one circulation layer:

- legacy ports receive their existing `RailRuntimeView` and width-specific proposal rail;
- D16-capable resident ports expose `apply_field_event(...)`, maintain an exact local mirror, and receive a Heart-issued D16 coherence binding instead of a home rail;
- Heart sends a full `FIELD_SNAPSHOT` when no provable predecessor exists and otherwise may send a contiguous `FIELD_DELTA`;
- Heart accepts only a matching `MIRROR_ACK` and excludes an unproven/stale D16 Core from synchronized participation;
- FIRST/REFINED proposal workspaces always carry one exact D16 text frame, while legacy rendered rails may coexist for old ports;
- after FINAL is materialized and committed, Heart emits `CANONICAL_SYNC` to D16 resident Cores without recursively starting another FIRST round;
- durable reasoning episodes can record either the legacy D64 attention view or the D16 attention-view representation and the Trainer loader verifies both;
- lived-English curriculum compilation recognizes circulation schemas v3 and v4 as English proposal lineages.

## Integrity repair discovered during independent audit

Before B3 completion, Codex performed a read-only sweep and found two interrupted-integration defects plus one identity-integrity defect:

1. `runtime.heart` could not import because `circulation.py` referenced a D16 recovery helper that did not yet exist.
2. `ProposalWorkspace` called a nonexistent `D16TextFrame.from_text()` helper.
3. `D16RegionView` could alias a caller-owned writable NumPy array. Mutating that external buffer after construction could change the view's cells without changing the stored hash/identity.

All three were repaired before the live circulation transition was accepted. `D16RegionView` now owns a detached C-contiguous float32 copy before hashing and marks that owned copy read-only. A regression test mutates the caller's original array and proves the D16 cells, hash, and identity remain unchanged.

## Evidence

Verified local tests on the canonical Axon checkout:

- D16 deterministic bus/integrity suite: 12/12 passed.
- Focused B3 circulation + lived-Trainer gate: 18/18 passed.
- D16/legacy mixed regression covering Unicode, D64 compiler, masks, circulation, and lived Trainer: 73/73 passed.
- Heart coordinator/control-plane/durable-ingress/host/mask chunk: 66/66 passed.
- D64 codec/intelligence/lease chunk: 13/13 passed.
- Heart↔Trainer authority binding: 18/18 passed.
- Heart training preflight: 4/4 passed.
- `python -m compileall runtime/heart runtime/field runtime/trainer training -q`: passed.
- `git diff --check`: passed after normalizing accidental PowerShell CRLF rewrites back to repository LF convention.

The decisive migration proof is `tests/test_d16_circulation_b3.py`: a fake Core registered at `d_model=512` completes the live FIRST -> REFINED -> consolidator -> FINAL cycle while the frozen tick contains no D512 rail. The tick retains the legacy D64 surface, but the D512 Core does not consume or require it. It receives D16 snapshot/coherence state, exact D16 proposal frames, commits FINAL, then receives `CANONICAL_SYNC`.

## Preserved boundaries

- Heart remains the sole canonical owner/writer.
- Exact Shared Field truth remains D16 substrate transport plus canonical metadata; private neural state is non-authoritative cognition.
- The old D64 implementation is preserved, not deleted or rewritten as if it never existed.
- B3 does not redefine Soul durability. The current durable Soul contracts remain in force until a separate explicit Soul decision reconciles the newer volatile-resident proposal.
- No `D:\ContinuousCoreLab` checkpoint or synthetic codebook is imported into Axon.
- No cloud training is required or authorized by this resolution.

## Next build boundary — B4

B4 may now begin with a fresh local D512 Core using Axon's real registered D16 substrate:

`exact Axon D16 -> learned 16→512 projection -> one persistent recurrent D512 chamber -> categorical exact-symbol output`

The first learned gate should remain deliberately small and falsifiable. Reproduce substrate copy/literacy first, then delayed recall and correction under live delta continuity. Variable-length output and explicit termination must be introduced before interpreting the Core as conversationally ready. Wider chambers remain future experiments; D512 is the first development baseline, not doctrine or a permanent ceiling.
