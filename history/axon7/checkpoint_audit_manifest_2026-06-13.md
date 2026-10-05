# Axon7 Checkpoint Quarantine Manifest

Date: 2026-06-13

Source audit: `D:\axon7\AXON_CORE_CURRICULUM_AND_CHECKPOINT_AUDIT.md`

Action: moved non-candidate checkpoint `.pt` files out of `D:\axon7\checkpoints` and into this dated quarantine folder. Nothing was deleted. Logs under `D:\axon7\logs` were left untouched.

## Selection Rule

The active checkpoint folder is now reserved for direct soul-first continuation candidates plus small model-creation config JSONs.

The only active checkpoint kept is:

- `D:\axon7\checkpoints\cv2copy_128_stage9_schema_mix_acc4_001200.pt`

Why: it is the best current soul-alive salvage candidate from the audit and verification. It is 128D, `act_reflect_v2`, has strong old copy/schema metrics, and its soul pathway still has measurable influence (`soul_cross_gate` about `0.0357` and `0.0698`, `soul_reflect_gate` about `0.2191`).

Active config JSONs kept in `D:\axon7\checkpoints`:

- `_custom_config.json`
- `_cv2_64_local_smoke_config.json`
- `_state256_2h2l_ffn32768_config.json`
- `_v2_128_act_reflect_v2_smoke_config.json`

## Verification Notes

- All checkpoint files listed here loaded successfully with `torch.load(..., map_location="cpu")`; none appeared physically corrupt.
- No local Axon7 trainer process was running before the move.
- `state256_2h2l_ffn32768_all_018000.pt` was moved despite being valuable evidence because its soul cross gates are effectively muted, so it should not remain in the active folder for soul-first continuation by accident.
- The 32D lineage was moved because Jeff said he was okay getting rid of the little 32, and these files are not current soul-first candidates.

## Moved Checkpoints

| File | Size bytes | Category | Reason |
| --- | ---: | --- | --- |
| `128_001200.pt` | 166101067 | Reference | Older 128D overfit lineage; soul gates exist but weaker/less balanced than the selected stage9 candidate. |
| `32_001200.pt` | 38778635 | Quarantine | Tiny 32D lineage; not a selected soul-first continuation candidate. |
| `act_reflect_v2_128_stage128_005500.pt` | 166108427 | Reference | Intact 128D act-reflect checkpoint, but weak draft/change metrics relative to later schema checkpoints. |
| `cv2copy_128_stage2_full48_acc4_002000.pt` | 166109131 | Reference | Strong older copy/delta metrics and soul influence, but superseded by stage9 for active soul-first work. |
| `cv2copy_128_stage6_schema_acc4_001600.pt` | 166109131 | Reference | Strong schema/soul-diversity reference, but superseded by stage9 as active candidate. |
| `state256_2h2l_ffn32768_all_018000.pt` | 664011595 | Reference | Large state/delta body with strong preservation, but muted soul path and failed runtime gate; keep only as evidence/reference. |
| `statecopy64_stage1_003000.pt` | 22734091 | Reference | Kindergarten copy proof, not a responder and not a soul-first continuation candidate. |
| `v7_000100.pt` | 38778635 | Quarantine | Early 32D intermediate; superseded and not selected. |
| `v7_000500.pt` | 79350923 | Quarantine | Old 64D intermediate from pre-soul-first path; superseded. |
| `v7_001000.pt` | 79350923 | Quarantine | Old 64D intermediate from pre-soul-first path; superseded. |
| `v7_001200.pt` | 38778123 | Quarantine | Early 32D/v7 lineage reference; not selected for active folder. |
| `v7_core64_fullcurr_gpu_resume_006000.pt` | 79360843 | Quarantine | Old 64D full-curriculum result with failed runtime eval and collapsed/near-zero soul gates. |

## Caveats

The selected stage9 checkpoint is a salvage candidate, not a deployable runtime core. It still needs placement exams through the ordered curriculum, especially Pre-K through Grade 4 soul-causality gates.
