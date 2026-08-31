# Axon Trainer Operator Guide

Status: current bounded interface, 2026-08-30

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

The transport-neutral organ declares future commands for inventory, preflight,
configure, start, pause, resume, evaluate, cloud export/import, comparison,
promotion, and rollback. Only `status` is connected today. Every unwired
mutation fails closed with `unavailable`; there is no hidden shell fallback.

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

## Why no overnight communication run is launched yet

The real Dormant State currently has 59,876 exact records but no recorded
runtime reasoning episodes and no explicit episode outcomes. Recovered
user/assistant adjacency is observable evidence, not automatic answer-quality
supervision. The communication-first contract therefore requires an endorsed,
corrected, or otherwise evidence-qualified C1 target manifest before sustained
training. Repeating the existing FFCS mechanism/copy schoolhouse for tens of
thousands of steps would train the wrong objective.

Cloud packets, Kaggle/Colab/SimplePod adapters, mid-segment interactive pause,
and clickable core/checkpoint/curriculum selection remain future handlers on
the same organ boundary. No provider credential or cloud spend is required by
the present local interface.
