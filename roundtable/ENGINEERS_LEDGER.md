# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-07T21:43:10Z
Current through event:
`evt-20260907T214310657835Z-hermes-codex-recovery-launch`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
Identity stamp: Hermes / glm-5.3-flash (cloud) / 2026-09-07

## Current mission and honest status

The immediate objective remains **exact motor writing on physical D64**. The
Codex audit's bounded hardening pass is complete: Codex implemented all
blocking corrections before his usage expired, Hermes verified the orphaned
work (594 passed, Ruff clean, hashes cross-checked), committed it as
`7b1857b`, and pushed everything to `origin/main`. The corrected-identity
fresh multicell shot `2c7e912b…` is in flight (non-serving). No reasoning
core or learned Heart tissue is serving.

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
| 1–60 multicell v2 fresh (corrected identity) | `2c7e912b…` | RUNNING (2026-09-07 21:33Z) | — | — |

Job: `287787f863df48f36cde85e4d8f8db84e0b75f8a0469716aa28c6f8304191033`

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
  `KernelWorkerStatus.RUNNING` as axongliksbot; quota at launch GPU
  28.54/30.00 hours, TPU 20/20 hours remaining.

## Binding continuity

- Physical D64 remains useful; do not bump width in response to this failure.
- Do not start `transport_eos` until multi-cell copy alignment passes its exact
  regression bar.
- Do not promote from falling loss or heldout-only success.
- Preserve `D:/00`, teammate state, exact 16D substrate, and all rejected
  evidence.
- The v2 fresh-shot candidate never resumes the old base-objective
  checkpoint; mid-run sync is observation-only and never continuation
  authority.

## Next recommended shot (status after Hermes recovery, 2026-09-07)

1. ~~Content-address the teaching overlay + regression test~~ — **done by
   Codex, verified by Hermes** (`7b1857b`).
2. ~~Fail-closed sync integrity~~ — **done by Codex, verified by Hermes**
   (`SyncDatasetUnavailable`).
3. ~~Bind generate-gate initialization to candidate identity~~ — **done by
   Codex, verified by Hermes** (v3 `r64v3-` manifest, legacy `r64v2-`
   fallback); Ruff/whitespace clean.
4. ~~Govern the canonical line-ending breach~~ — **done**: `.gitattributes`
   marks the canonical ledger `-text` (committed in `7b1857b`); no history
   rewrite.
5. **In flight**: fresh non-serving multicell v2 smoke `2c7e912b…` RUNNING —
   grade regression position on the 12-cell surface when COMPLETE; require
   1.0 before any `transport_eos`. Do not serve.
