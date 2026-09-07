# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-07T00:35:00Z
Current through event:
`evt-20260907T003500000000Z-copilot-cli-jeff-copy-alignment-diagnosis`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
Identity stamp: GitHub Copilot CLI / Grok 4.6 (grok-4.6) / 2026-09-06

Continuity note: FFN256 copy-alignment renewal completed and was fetched.
Regression position did not move. No serving. No SOT change. Jeff prefers
conversation over the roundtable.

## Current mission and honest status

The immediate objective is still **exact motor writing** on D64.

Active evidence: mixer 1h/4L/FFN256 heldout copy-gate and position are 1.0
after 120 steps. Regression position 0.667 is 8/12 transport cells: the
core finds the first 16D cell of the source letter and does not walk
UTF-8 continuation cells (`Ж`, `ñ`, `亰`). Heldout was all native 1-cell
letters. Extra 60 steps lowered loss and did not teach the walk. Do not
blindly spend a third tranche. Do not bump width. Do not serve.

## Mixer lineage (non-serving)

| Segment | Job | Result | Heldout copy / pos | Regression pos |
|---|---|---|---|---|
| 1–60 FFN256 | `387a52eb…` | completed | 1.0 / 1.0 | 0.667 |
| 1–60 FFN512 | `26302a3d…` | completed | 1.0 / 1.0 | 0.667 |
| 61–120 FFN256 renewal | `c726a825…` | completed | 1.0 / 1.0 | 0.667 |

Renewal report:
`roundtable/reports/GROK_D64_MIXER_COPY_ALIGNMENT_RENEWAL_2026-09-06.md`

Final renewal bundle:
`2b4aa6dbadc4515a1ac2112eb963940161b1858c2f35bbbcfd3f062853584298`

## Binding

- Physical D64 rail over exact 16D cells. No guessed cloud parent.
- No promotion from falling loss.
- `D:/00` and teammate houses remain protected.

## Next

1. Teach multi-cell / byte-transport copy on D64 before another blind tranche.
2. Do not start `transport_eos` until `copy_alignment` passes.
3. Do not bump width. Do not serve.
