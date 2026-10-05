# Axon v7

A token-free, always-ticking, stateful agent. Every character is a frozen
16D vector; a small transformer rewrites its own readable state each tick.
See `FIELD_CONTRACT.md` for the full architecture.

## Run anywhere (Windows / Lightning AI / Colab / any box)

Everything resolves relative to this folder — **no path editing needed.**
Zip the whole `axon7/` folder, upload it, and run.

### Train
- **Windows:** double-click `train.bat`
- **Linux / Lightning AI / Colab:**
  ```bash
  bash train.sh          # or: chmod +x train.sh && ./train.sh
  ```
- **Direct (any OS):**
  ```bash
  python trainer_v2.py --episodes datasets --steps 50000 --device cuda
  ```

The menu auto-detects the GPU and suggests good defaults (soul size, FFN).

### Run him
- **Windows:** `run.bat`
- **Linux:** `bash run.sh cuda`  (or `cpu`)
- **Isolated smoke, no State/soul writes:**
  ```bash
  python runtime.py --device cpu --core core_v2_128_act_reflect_task_smoke --smoke "hello axon" --ticks 1
  ```

### Exact runtime gate
Trainer metrics can pass while rendered text is still wrong. Use the
runtime gate before deploying a checkpoint:

```bash
python runtime_eval.py --core checkpoints/cv2copy_128_stage9_schema_mix_acc4_001200.pt --suite core
python runtime_eval.py --core checkpoints/cv2copy_128_stage9_schema_mix_acc4_001200.pt --suite sigils
python runtime_eval.py --core checkpoints/cv2copy_128_stage9_schema_mix_acc4_001200.pt --suite repeat
```

`core` checks ordinary learned answers, `sigils` checks exact `++`, `@@`,
`$$`, and `##` rendering, and `repeat` checks whether a clean answer stays
clean across ticks. The gate exits nonzero on failures.

### State/delta curriculum
To teach Axon to preserve state before talking, generate full-state copy
and small-delta lessons from any folder/file:

```bash
python state_curriculum_factory.py --source D:/00 --out datasets/axon7_state_delta_d00_v1.jsonl
python validate_dataset.py datasets/axon7_state_delta_d00_v1.jsonl
```

This writes a combined dataset plus stage files:

- `stage1`: exact full-state copy
- `stage2`: dense/weird exact full-state copy
- `stage3`: response_draft deltas
- `stage4`: region eviction
- `stage5`: same-width non-draft rewrites
- `stage6`: two-tick `++` growth
- `stage7`: answer from structured state

For the first school run, train copy-only and watch all-region preservation:

```bash
python trainer_v2.py --episodes datasets/axon7_state_delta_d00_v1_stage1.jsonl \
  --config checkpoints/_cv2_64_local_smoke_config.json --core-id core_statecopy \
  --steps 1200 --accum 4 --device cuda --no-deploy \
  --require-visible-acc 0.50 --require-preserve-acc 0.50
```

The trainer log now includes `pres_vis`, the accuracy on visible rows that
should be preserved unchanged.

## Folder layout
```
substrate.py        16D frozen alphabet + geometry gates
field_contract.py   LAW 0: the one field-builder (trainer + runtime share it)
heads.py            letter16 + region16 projection heads
core.py             the transformer core
trainer_v2.py       the trainer (episodes -> trained core)
state_curriculum_factory.py full-state copy + delta curriculum generator
runtime.py          the always-ticking agent
runtime_eval.py     exact runtime pass/fail gate for checkpoints
train_menu.py       friendly training menu
validate_dataset.py check a curriculum batch
datasets/           episode JSONL (drop Kimi batches here)
Core/               trained cores the runtime loads
checkpoints/        training snapshots
tools/              ROOT TOOLS: drop a .py here, it becomes a `##` tool
State/              his live, human-readable state mirror
```

## First-time on a fresh machine
```bash
pip install torch numpy
bash train.sh        # gates run automatically; menu opens
```

## Picking a soul size
The soul is the core's private memory carried between ticks — a set of
"thought rows," each `d_model` wide. In `act_reflect_v2`, shared rows can
privately cross-attend to individual soul rows instead of receiving one
mean-pooled soul summary. The menu suggests: 32 for a 128D core, 64 for
256D, 128 for 512D, and 256 for larger cores. More rows = more private
carry across ticks, with extra cross-attention compute.

Soul modes:
- `act_reflect_v2`: row-addressable private soul cross-attention.
- `act_reflect`: older pooled soul modulation.
- `concat`: legacy mode, soul rows appended to the public field sequence.

## Training notes
- The trainer uses **contrastive** cosine-logit cross-entropy over the
  frozen alphabet. There is still no learned vocab and no token path.
- Nonempty write rows carry the main loss. Empty buffer/slack rows are
  light cleanup only, so blank output cannot look like success.
- Watch `nonempty_acc`, `draft_nonempty_acc`, and `pred_empty_frac`.
  If the target draft has text and the model writes blank, the trainer
  prints an explicit warning.
- LR uses warmup + cosine decay automatically.
- Gradient checkpointing is opt-in with `--grad-checkpoint` for large
  runs that would otherwise OOM.
- If you OOM: the launchers set `expandable_segments`; if it persists,
  try `--grad-checkpoint`, then lower `--accum`, then a smaller FFN.

## Portable proof run
Before spending GPU on a large core, make the trainer prove visible
writing on a tiny repeat:

```bash
python trainer_v2.py --episodes datasets/smoketest_episodes.jsonl \
  --config checkpoints/_smoke_config.json --core-id core_v2_smoke \
  --steps 150 --accum 1 --lr 0.01 --weight-decay 0 --warmup-steps 0 \
  --overfit-first 1 --require-visible \
  --render-every 5 --checkpoint-every 0 --no-deploy
```
