# Legacy curricula: character audit against the frozen 95

Generated 2026-10-05 by `python tools/audit_characters.py curricula/legacy_raw` (read-only; nothing was
rewritten). A record is "fully clean" when every text value in it uses only the 95 native characters.

How to read it:

- **CLEAN** files can go into the new trainer's validator as they are. **MOSTLY CLEAN / NEEDS CONVERSION** files
  hold characters outside the 95 and must not be trained on until a separate, explicit conversion step writes a new,
  versioned file plus a report. Nothing converts silently.
- The most common outsider by far is the **backtick** (`U+0060`). Other offenders: tab (the pair separator in
  `soul_pairs.txt`), em/en dash, curly apostrophe, arrows, check marks and emoji.
- "Clean" means the characters fit, not that the lesson is ready. The `axon7` episodes use the old v7 ten-region state
  format built for the retired transformer core; the new trainer will define its own lesson format.
- `axon7_abcs_v1.jsonl` and `axon7_curriculum_v2_all.jsonl` are combined supersets of their stage files. Two exact
  duplicates (`stage4_delayed_letters.jsonl`, `stage5_delayed_words.jsonl`) were not copied.
- Every file here is a raw copy of a source in `G:\My Drive\New folder (2)\axon7\datasets` or
  `...\ashes_v6_history\Datasets`. `curricula/legacy_raw/` is git-ignored because it holds personal material
  (journals, old conversations).

| File | KB | Records | Fully clean records | Outside-95 chars | Verdict | Most common outsiders |
|---|---:|---:|---:|---:|---|---|
| `curricula/legacy_raw/ashes_v6/dialogue/soul_pairs.txt` | 20 | 81 | 0 (0%) | 0.48% | MOSTLY CLEAN | U+0009x81, `U+0060x15 |
| `curricula/legacy_raw/ashes_v6/grammar_clean.jsonl` | 2,464 | 1,914 | 1,914 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/ashes_v6/kimi_wisdom_batch_001.jsonl` | 12,064 | 5,100 | 3,073 (60%) | 0.08% | MOSTLY CLEAN | →U+2192x2831, —U+2014x2275 |
| `curricula/legacy_raw/ashes_v6/layer1_batch_01.txt` | 255 | 7,000 | 6,997 (100%) | 0.00% | MOSTLY CLEAN | éU+00E9x3 |
| `curricula/legacy_raw/ashes_v6/soul/soul_axonm.txt` | 267 | 3,209 | 2,843 (89%) | 0.38% | MOSTLY CLEAN | `U+0060x678, ✓U+2713x85, ✅U+2705x79, —U+2014x64, →U+2192x30, ❌U+274Cx11 |
| `curricula/legacy_raw/ashes_v6/soul/soul_journal.txt` | 25 | 453 | 406 (90%) | 0.21% | MOSTLY CLEAN | —U+2014x38, `U+0060x6, 🟡U+1F7E1x4, 🔴U+1F534x2, ≠U+2260x1, →U+2192x1 |
| `curricula/legacy_raw/ashes_v6/soul/soul_summaries.txt` | 4,329 | 23,858 | 17,096 (72%) | 0.28% | MOSTLY CLEAN | `U+0060x3868, —U+2014x3235, →U+2192x1267, ✅U+2705x751, ’U+2019x631, –U+2013x609 |
| `curricula/legacy_raw/ashes_v6/soul/soul_turns.txt` | 224 | 2,673 | 2,285 (85%) | 0.29% | MOSTLY CLEAN | –U+2013x205, `U+0060x170, →U+2192x103, ’U+2019x48, —U+2014x42, ✅U+2705x40 |
| `curricula/legacy_raw/axon7/axon7_abcs_v1.jsonl` | 1,403 | 1,144 | 1,114 (97%) | 0.01% | MOSTLY CLEAN | `U+0060x30 |
| `curricula/legacy_raw/axon7/axon7_abcs_v1_stage0.jsonl` | 134 | 128 | 128 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_abcs_v1_stage1.jsonl` | 218 | 248 | 248 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_abcs_v1_stage2.jsonl` | 150 | 160 | 160 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_abcs_v1_stage3.jsonl` | 143 | 160 | 133 (83%) | 0.05% | MOSTLY CLEAN | `U+0060x27 |
| `curricula/legacy_raw/axon7/axon7_abcs_v1_stage4.jsonl` | 430 | 256 | 253 (99%) | 0.00% | MOSTLY CLEAN | `U+0060x3 |
| `curricula/legacy_raw/axon7/axon7_abcs_v1_stage5.jsonl` | 328 | 192 | 192 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all.jsonl` | 373 | 384 | 384 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage1.jsonl` | 32 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage2.jsonl` | 34 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage3.jsonl` | 34 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage4.jsonl` | 73 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage5.jsonl` | 38 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage6.jsonl` | 67 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage7.jsonl` | 38 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_all_stage8.jsonl` | 58 | 48 | 48 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_stage10_sigil_drill.jsonl` | 321 | 276 | 276 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_stage9_schema_mix.jsonl` | 201 | 144 | 144 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_curriculum_v2_stage9_symbol_idempotence.jsonl` | 125 | 72 | 72 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1.jsonl` | 2,457 | 1,216 | 652 (54%) | 0.50% | MOSTLY CLEAN | `U+0060x8684 |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1_stage1.jsonl` | 420 | 256 | 153 (60%) | 0.45% | MOSTLY CLEAN | `U+0060x1266 |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1_stage2.jsonl` | 633 | 256 | 111 (43%) | 0.46% | MOSTLY CLEAN | `U+0060x2298 |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1_stage3.jsonl` | 258 | 160 | 97 (61%) | 0.37% | MOSTLY CLEAN | `U+0060x629 |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1_stage4.jsonl` | 209 | 128 | 68 (53%) | 0.46% | MOSTLY CLEAN | `U+0060x649 |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1_stage5.jsonl` | 232 | 128 | 70 (55%) | 0.42% | MOSTLY CLEAN | `U+0060x680 |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1_stage6.jsonl` | 420 | 128 | 61 (48%) | 0.66% | MOSTLY CLEAN | `U+0060x1942 |
| `curricula/legacy_raw/axon7/axon7_state_delta_d00_v1_stage7.jsonl` | 285 | 160 | 92 (58%) | 0.61% | MOSTLY CLEAN | `U+0060x1220 |
| `curricula/legacy_raw/axon7/axon7_state_delta_smoke.jsonl` | 55 | 23 | 23 (100%) | 0.00% | CLEAN |  |
| `curricula/legacy_raw/axon7/factory_d00_act_reflect_v2_stellar.jsonl` | 613 | 383 | 347 (91%) | 0.11% | MOSTLY CLEAN | `U+0060x311 |
| `history/ashes_v6/commandments.txt` | 21 | 10 | 10 (100%) | 0.00% | CLEAN |  |
