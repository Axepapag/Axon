# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-07T22:42:02Z
Current through event:
`evt-20260907T224202643848Z-codex-unicode-walk-shot`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
Identity stamp: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-07

## Current mission and honest status

The immediate objective remains **exact motor writing on physical D64**. The
corrected-identity fresh shot `2c7e912b…` completed and was correctly rejected:
heldout copy/position was 1.0/1.0, but regression position remained 0.667.
Immutable checkpoint replay proved the exact defect: all first transport cells
pass while every UTF-8 continuation cell returns to the scalar's first cell.

Codex added a separate content-addressed Unicode-walk curriculum (`12df4547…`)
with split-disjoint 2/3/4-cell train, heldout, and regression surfaces while
preserving the historical `a872278f…` exam. The fresh non-serving Kaggle shot
`050a3a97…` is now running against both exams. No reasoning core or learned
Heart tissue is serving.

Grok's smaller 1-head / 4-layer D64 mixers made real progress: FFN256 and
FFN512 both learned the heldout copy gate and one-cell position at 1.0 within
60 steps. Extra FFN did not help. The FFN256 renewal and multi-cell teaching
overlay did not solve Unicode continuation: regression position remained
0.667 through step 180. The gate correctly rejected every candidate.

## Mixer lineage (non-serving)

| Segment | Job | Result | Heldout copy / pos | Regression pos |
|---|---|---|---|---|
| 1–60 FFN256 | `387a52eb…` | completed | 1.0 / 1.0 | 0.667 |
| 1–60 FFN512 | `26302a3d…` | completed | 1.0 / 1.0 | 0.667 |
| 61–120 FFN256 renewal | `c726a825…` | completed | 1.0 / 1.0 | 0.667 |
| 121–180 multi-cell teach | `287787f8…` | fetched, paused, gate fail | 1.0 / 1.0 | 0.667 |
| 1–60 multicell v2 fresh (corrected identity) | `2c7e912b…` | fetched, paused, gate fail | 1.0 / 1.0 | 0.667 |
| 1–120 Unicode pointer walk v3 | `050a3a97…` | RUNNING (2026-09-07 22:37Z) | pending | pending |

Active job: `050a3a97336b8645fb13bb6a6a307fd884fa6450ca64c3f0fa29066640d68d59`

## Blocking audit findings (RESOLVED in `7b1857b`, verified by Hermes)

1. ~~Overlay changed sampling/position-reduction/weights while retaining
   objective program `00d5d384…`~~ — **resolved**: the multi-cell overlay is
   now content-addressed (`COPY_ALIGNMENT_MULTICELL_TEACH_ID` /
   `FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID`, effective `eedeb511…`), with a
   regression test proving any change alters identity; the fresh `2c7e912b…`
   shot runs under the corrected identity. The old step-180 checkpoint
   `3baf69d5…` remains rejected and unimported.
2. ~~Auto-sync catches every `CloudPacketError` as "waiting"~~ — **resolved**:
   `SyncDatasetUnavailable` now distinguishes proved not-yet-published from
   hash/incomplete/wrong-job/conflict failures; the dashboard fails closed.

## Advisory findings (RESOLVED in `7b1857b`, verified by Hermes)

- Commit `64c0b1a` converted 159 pre-existing canonical ledger lines from LF
  to CRLF, breaching physical append-only law (content survived; IDs unique).
  **Governed correction**: `.gitattributes` now marks
  `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` as `-text` so Git can never
  normalize it again; no history rewrite, old event content untouched.
- `generate_gate_bias` was outside topology architecture identity but absent
  from candidate identity. **Resolved**: bound into a v3 candidate-generation
  manifest (`r64v3-…`) alongside heads/layers/ffn/seed, with legacy `r64v2-…`
  fallback for existing bundles; topology architecture IDs unchanged.
- ~~Ruff 12 findings + trailing whitespace + manifest/program mislabel~~ —
  **resolved**: Ruff passes clean on every changed Python surface;
  `git diff --check` is empty; the fresh recipe's note states the true
  overlay/effective-program IDs.

## Verified state (Codex audit, superseded numbers kept for provenance)

- Codex audit: full repository suite **592 passed**; focused Grok-area suite
  **75 passed**; `compileall` passed; Kaggle job `287787f8…` COMPLETE/fetched,
  paused at step 180, task/serving gates false; the clean step-120 checkpoint
  `2f9f4032…` remained the authoritative latest; quota at audit GPU 28.58/30h.
- Hermes recovery verification (2026-09-07 21:20Z): full repository suite
  **594 passed**, 27:17, exit code 0 (delta +2 = the new objective-identity
  regression tests); focused hardening suites 143 collected, exit 0; Ruff
  clean on all changed Python surfaces; `git diff --check` empty.
- Recipe hashes cross-checked: overlay `e4bfc686…` and effective program
  `eedeb511…` equal live `canonical_sha256` values; include paths exist; all
  entrypoint flags exist in the smoke script.
- Git: recovery commit `7b1857b` (16 files, +938/−133, Co-Authored-By Codex)
  pushed `2a63c9a..7b1857b` — all 17 local commits now on `origin/main`.
- Kaggle: fresh-shot job `2c7e912b…` submitted and verified
  `KernelWorkerStatus.COMPLETE`; fetched bundle returncode 0; step 60 remained
  rejected/non-serving with regression position 0.667.
- Codex Unicode-walk pass: exact checkpoint replay showed 8/12 regression
  cells correct and all four continuation offsets wrong; manifest `12df4547…`
  verifies 144 cases with disjoint 2/3/4-cell exams; 43 focused tests passed,
  Ruff/diff-check clean; commit `ddbef59` pushed.
- Kaggle: Unicode-walk job `050a3a97…` launched privately from `ddbef59` and
  verified `KernelWorkerStatus.RUNNING`; quota at launch GPU 28.32/30.00 hours.
  Optional mid-run sync has no verified local window; one direct pull returned
  a visible 403. End-run bundle fetch remains authoritative.

## Binding continuity

- Physical D64 remains useful; do not bump width in response to this failure.
- Do not start `transport_eos` until multi-cell copy alignment passes its exact
  regression bar.
- Do not promote from falling loss or heldout-only success.
- Preserve `D:/00`, teammate state, exact 16D substrate, and all rejected
  evidence.
- Corrected-identity fresh candidates never resume the old base-objective
  checkpoint; mid-run sync is observation-only and never continuation
  authority.

## Next recommended shot (2026-09-07)

1. Monitor `050a3a97…`; fetch and hash-verify its end-run bundle when COMPLETE.
2. Grade continuation-cell position on both the preserved historical exam and
   the new split-disjoint 2/3/4-cell surfaces. Falling loss is not mastery.
3. Require exact copy-alignment gates before `transport_eos`; do not serve.
4. If diverse Unicode teaching still stalls on the first cell, add a
   content-addressed pointer-transition mechanism to D64 rather than repeating
   blind tranches or widening the rail.
5. Diagnose the optional mid-run sync 403 separately; it does not authorize or
   invalidate continuation, and end-run bundle evidence remains authoritative.
