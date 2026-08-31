# Axon Trainer Operator Guide

Status: current bounded interface, 2026-08-31

## What is safe and available now

Double-click `TRAIN_AXON.bat` for a concise read-only snapshot, or run:

```powershell
python scripts/axon_trainer.py status
python scripts/axon_trainer.py status --detail full
```

The summary reports lease state, artifact counts, current candidate
checkpoints, latest lifecycle, and latest optimizer step. Full mode includes
per-tensor telemetry. Neither command creates an optimizer, changes a Soul,
starts training, promotes tissue, or spends cloud money.

The transport-neutral organ declares commands for inventory, preflight,
configure, start, pause, resume, evaluate, cloud export/import, comparison,
promotion, and rollback. The generic local status client connects only
`status`. The Kaggle control surface registers governed cloud-export, start,
and result-import handlers against this same organ for the lifetime of that
client. Every other unwired mutation fails closed with `unavailable`; there is
no hidden shell fallback.

## Renewable training semantics

New tissue uses `ParameterMutationPlanV2`. Architecture, seed, curriculum,
exact mutation scope, and learning policy remain governed. A resource tranche
grants one bounded run segment but is not part of candidate, plan, curriculum,
or learning-policy identity.

- A fresh v2 tranche starts at global step zero.
- A continuation tranche requires `--resume` and the exact accepted parent
  checkpoint, optimizer receipt, and private-Soul HEAD.
- Tranche exhaustion checkpoints, evaluates, and pauses the curriculum stage.
- Only a complete competency gate marks a stage complete.
- Evaluation-only restores accepted tissue but performs no optimizer step or
  Soul transition.
- Historical v1 candidates remain immutable and are addressed explicitly with
  `--legacy-plan-v1`; a tranche may adopt them beyond the old envelope without
  changing the original plan or generation.

## Checkpoint retention

Checkpoint payload artifacts are retained per candidate generation under a
keep-3 policy (`CHECKPOINT_RETENTION` in `runtime/trainer/store.py`). Every
accepted session checkpoint automatically prunes older payloads; the artifact
referenced by `latest_checkpoint.json` is always retained. Immutable
checkpoint records are never deleted, so a pruned payload remains detectable
by its recorded hash and any stale load fails closed. To reclaim space by
hand, run `python scripts/prune_trainer_checkpoints.py` (add `--dry-run` for
a projection); it also removes orphaned staging temp files that were never
promoted to content-addressed artifacts.

The current research harness exposes these mechanics for engineers. A fresh
bounded V2 invocation has this shape:

```powershell
python scripts/train_living_reasoning_smoke.py `
  --state-root D:\Axon\State `
  --curriculum-manifest <immutable-manifest.json> `
  --candidate-label <stable-lane> `
  --device cpu `
  --tranche-steps <bounded-allowance>
```

Continuation repeats the exact architecture, seed, curriculum, and learning
policy and adds `--resume --tranche-steps <allowance>`. These commands are an
engineering surface, not yet the promised clickable Trainer UI.

## Current curriculum warning

Recovered user/assistant adjacency is observable evidence, not automatic
answer-quality supervision. C1 is now a governed communication-first manifest
and has supported bounded tournament diagnostics, but no learned reasoning
candidate has passed its serving gates. Large campaigns still require explicit
target-quality review and exact held-out gates; repetition alone is not
evidence of intelligence.

The private Kaggle adapter and independent observable launchers are now
implemented. Double-click `AXON_KAGGLE.bat`; see
`docs/KAGGLE_TRAINING_GUIDE.md`. Colab/SimplePod adapters, mid-segment
interactive pause, and runtime-native clickable core/checkpoint/curriculum
selection remain future handlers on the same organ boundary. Packet
preparation spends nothing; Kaggle submission remains an explicit operator
action.
