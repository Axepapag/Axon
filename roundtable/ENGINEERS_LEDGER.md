# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-07T19:15:42Z
Current through event:
`evt-20260907T191542780619Z-codex-grok-audit`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
Identity stamp: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-07

## Current mission and honest status

The immediate objective remains **exact motor writing on physical D64**, but
the next training/import/continuation shot is halted for a bounded hardening
pass. No reasoning core or learned Heart tissue is serving.

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

Job: `287787f863df48f36cde85e4d8f8db84e0b75f8a0469716aa28c6f8304191033`

## Blocking audit findings

1. The step-121–180 multi-cell overlay changed sampling, position reduction,
   and component weights while retaining objective program `00d5d384…` and
   learning policy `e16f58ae…`, identical to steps 1–120. Trainer lineage does
   not identify the objective actually used. Do not import or continue from
   step 180 until governed resolution; the cheapest clean path is to correct
   the identity and rerun from the exact step-120 parent.
2. Uncommitted auto-sync catches every `CloudPacketError` as ordinary
   "waiting." That can hide incomplete, hash-failed, wrong-job, or conflicting
   bundles. It must distinguish not-yet-published from integrity failure before
   commit.

## Advisory findings

- Commit `64c0b1a` converted 159 of 162 pre-existing canonical ledger lines
  from LF to CRLF. All JSON content survived and all 180 current event IDs are
  unique, but physical append-only law was breached. Record/govern a correction;
  do not rewrite history silently.
- `generate_gate_bias` properly stays outside topology architecture identity,
  but it is also absent from candidate-generation identity. Bind it through an
  initialization/candidate manifest so different initial weights cannot share
  one candidate ID.
- Ruff reports 12 findings in Grok-touched Python surfaces. `git diff --check`
  also reports trailing whitespace in the uncommitted teach recipe, whose note
  mislabels curriculum manifest `a872278f…` as the program ID; the real program
  ID is `00d5d384…`.

## Verified state

- Full repository suite: **592 passed**, exit code 0.
- Focused Grok-area suite: **75 passed**, exit code 0.
- Python `compileall`: passed.
- Kaggle job `287787f8…`: private, COMPLETE, outputs fetched, return code 0,
  paused at step 180, task/serving gates false.
- The local candidate's authoritative latest checkpoint remains the clean
  step-120 checkpoint `2f9f4032…`; fetched step 180 was not imported.
- Kaggle quota at audit: GPU 28.58/30.00 hours remaining; TPU 20/20 hours.
- Shared machine sweep: no training, Kaggle, or pytest process remains.
- Test-owned temporary evidence remains under
  `State/training/pytest_codex_grok_{audit,full}_20260907` (2,361 files,
  about 39.2 MB total); it was not deleted under the preservation contract.
- Git: `main` is 16 commits ahead of `origin/main`; those commits are unpushed.
  Ten files also carry uncommitted Grok auto-sync/monitor/ledger work.

## Binding continuity

- Physical D64 remains useful; do not bump width in response to this failure.
- Do not start `transport_eos` until multi-cell copy alignment passes its exact
  regression bar.
- Do not promote from falling loss or heldout-only success.
- Preserve `D:/00`, teammate state, exact 16D substrate, and all rejected
  evidence.

## Next recommended shot

1. Content-address the complete teaching overlay in Trainer policy lineage and
   add an identity-change regression test.
2. Make sync integrity errors fail visibly; only a proved missing/unpublished
   dataset may display as waiting.
3. Bind initialization settings to candidate identity, clean Ruff/whitespace,
   and correct manifest/program labeling.
4. Govern the canonical ledger line-ending breach without a history rewrite.
5. Rerun the inexpensive multi-cell lesson from the clean step-120 parent.
   Require regression position 1.0 before advancing. Do not serve.
