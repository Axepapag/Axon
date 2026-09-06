# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-06T22:10:00Z
Current through event:
`evt-20260906T221000000000Z-copilot-cli-d64-mixer-kaggle-launch`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
Identity stamp: GitHub Copilot CLI / Grok 4.6 (grok-4.6) / 2026-09-06

Continuity note: Jeff ratified invert-shape D64 in conversation. Two
non-serving mixer smokes are RUNNING on Kaggle. Stage-0 fat-MLP candidate
remains failed/paused evidence. No serving. No SOT change. Personal
first-mission opinion stays off-table at D:\Grok\Briefs.

## Current mission and honest status

Axon is building permanent, additive organism tissue around an exact 16D
canonical substrate, private layered core Souls, width-specific reasoning
rails, Heart translation, governed proposal/consolidation, and a Trainer
that can teach offline cores from authored curricula.

The immediate objective is still **exact motor writing** on D64: attend the
rail and produce meaningful copy/delta, not language theater.

Active shot: inverted mixer tissue on motor v2 `copy_alignment`, not a
third fat-MLP Stage-0 tranche and not a width bump.

## Active mixer comparison (non-serving)

Source commit: `999df55906703bac7728cb4f0f2e2112ec15235b`

Motor v2 curriculum: `a872278fd0e8ef926370e1712d01dcf0a277672c4a0088af44aef483d8417740`
(144 cases; 72/36/36). Trainer starts at `copy_alignment`. Generate-gate
bias 0.0 is initialization, not architecture identity.

| Shape | Candidate | Job | Packet | Provider |
|---|---|---|---|---|
| 1 head, 4 layers, FFN 256 | `axon-d64-mixer-4l-ffn256-h1` | `387a52ebc78e51087b11f461ebc68153069a84fdd46039f812edd43705f3d9b8` | `d5eacf003e2769699b15300ebb0c2ff4279d8e8d459556e42d66d88486fe723f` | RUNNING `axongliksbot/axon-job-387a52ebc78e5108` |
| 1 head, 4 layers, FFN 512 | `axon-d64-mixer-4l-ffn512-h1` | `26302a3d6f93fa5e93438b92381266832e1c962205c640027e3a2b6704795747` | `72a2b0a199fd78e9d3f7e5b49a63ac2ecab6d91ad939e405cf2e836f41782710` | RUNNING `axongliksbot/axon-job-26302a3d6f93fa5e` |

Local 1-step CUDA probe (GTX 1650, distinct label
`axon-d64-mixer-4l-ffn256-h1-local-probe`):

- architecture `living-d64-1d35977970d59137a91674d1`
- 331,319 params (~1.3 MiB fp32) vs Candidate A 33,981,879
- peak CUDA 39,048,192 bytes; train-step wall 11.3s
- heldout copy-gate accuracy 0.0 -> 1.0 after one step
- heldout position accuracy 0.0 -> 0.0
- EOS-gate 1.0 -> 0.0 (copy_alignment weights copy-gate, not EOS)
- serving promotion claimed: false; stage gate failed as expected
- report: `State/training/reasoning/r64v2-9115d62260f82e2d/segment_000000001_000000001.json`

This is a one-step probe, not mastery. Kaggle 60-step smokes are the
comparison. Do not promote from the probe.

## Preserved Stage 0 evidence (failed, paused)

The F0 fat-MLP candidate
`axon-foundation-motor-stage0-aligned-1x64` / `r64v2-d39ec38a0f38f523`
took 120 accepted steps on 1 head / 2 layers / FFN 131072. Address
learning appeared; copy-gate and typed exact did not. Collapse:
`DELTA / REPLACE / response_draft / empty payload`. Strict gate held.
Lineage is immutable failed evidence, not a parent to continue.

Diagnostic:
`roundtable/reports/CODEX_FOUNDATION_STAGE0_DIAGNOSTIC_2026-09-06.md`.

## Binding architecture and governance

- Canonical Shared Field cells are exact 16D substrate.
- Living core implementation remains locked to the physical D64 rail.
  A 256D/512D/1024D brother is new rail/Soul codec work, not `--d-model`.
- `--generate-gate-bias` defaults to Candidate A's 1.5 and is excluded
  from architecture identity.
- Heart translates; cores propose; Trainer candidates are isolated.
- Limits are governed work/resource controls, not tissue ceilings.
- No silent truncation, gate weakening, guessed cloud parent, automatic
  promotion, or learned-serving claim.
- `D:/00` and teammate houses remain protected.

## Verification this turn

- Targeted tests: generate-gate init identity, frozen architecture IDs
  `living-d64-675b5ec0f0053cd54c0fbda6` / `40b4ad19...` / `e0a0ad21...`,
  motor v2 compile script. **5 passed**.
- Kaggle doctor before launch: READY, account `axongliksbot`,
  29.59/30 GPU hours remaining (refresh 2026-09-12). Paid spend: zero.
- Both mixer jobs submitted; Kaggle reported RUNNING for each.

## Next

1. Wait for both Kaggle jobs. Fetch hash-verified outputs.
2. Compare copy-gate, position, and coverage. Do not promote.
3. Only then decide a third mixer shape, Stage-0-v2 joint, or a 256D
   brother. Width is still the later question.

Operator entry points:

```powershell
python scripts/axon_kaggle.py doctor
python scripts/axon_kaggle.py status 387a52ebc78e51087b11f461ebc68153069a84fdd46039f812edd43705f3d9b8
python scripts/axon_kaggle.py status 26302a3d6f93fa5e93438b92381266832e1c962205c640027e3a2b6704795747
python scripts/axon_kaggle.py fetch <job-id>
```
