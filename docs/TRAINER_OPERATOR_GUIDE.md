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

## Enforced motor-v2 termination route

Motor-v2 training must select the ratified termination objective. The trainer
refuses to start any other motor-v2 route before it does any compute:

```
python scripts/train_living_reasoning_smoke.py \
  --receipt-continuation \
  --receipt-teaching-profile termination_head_balanced_v6 \
  --termination-head-route
```

The rejected routes are the pre-receipt-continuation objectives, where the stop
decision shares one softmax with content: content-logit growth mechanically
depresses EOS probability, stop=1 is supervised twice, and stop=0 is never
supervised directly. The two pressures cancel, so EOS wins every argmax and
free-running transport emits nothing. That fixed point is already measured, so a
tranche that runs it cannot produce information; refusing to start is the only
outcome that does not spend allowance.

The refusal is fail-closed and derived, not a hand-maintained list of profile
names: membership of the ratified set is computed from each objective's own
`termination_continue_supervision` declaration, so a newly added profile that
does not declare explicit symmetric stop supervision is refused by default. The
same gate runs in `scripts/run_d64_tournament.py` before a candidate is spawned
and in `scripts/axon_kaggle.py` before a packet is built or uploaded. Read-only
`--evaluate-only` reproduction of an existing bundle is always permitted.

### Reading the termination objective from the dashboard

The rejected route reported `termination_continue_accuracy` as a vacuous `1.0`
for all 600 steps of the emission rung because it supervised zero anchors. The
metric now fails closed instead: with no supervised anchor the rate is
unavailable (`null`), not a perfect score, and a route that *declares* explicit
continuation supervision while supervising no content anchor raises rather than
reporting numbers for an objective it never exercised.

`scripts/axon_training_watch.py` shows the objective per step:

```
 termination: continue positions 3 (min 3 of 12 steps)  cont-loss 0.551 ▆▄▃  train eos-gate 62% ▂▃▅
```

- `continue positions` — stop=0 anchors supervised this step; the ratified route
  requires more than zero from its first applicable step, and any step that
  supervised none is called out as `UNSUPERVISED STEP(S)` in red.
- `cont-loss` — the mean of the explicit stop=0 BCE terms that are already part
  of the objective. Instrumentation only: the trainer optimizes exactly the same
  tensors it did before the metric existed.
- `train eos-gate` — teacher-forced alignment EOS-gate accuracy for the step.
  The graded rate is the heldout probe's `eos-gate`, printed on the
  `motor v2` lines (the rejected route never moved it off `0.3125`).

The private Kaggle adapter and independent observable launchers are now
implemented. Double-click `AXON_KAGGLE.bat`; see
`docs/KAGGLE_TRAINING_GUIDE.md`. Colab/SimplePod adapters, mid-segment
interactive pause, and runtime-native clickable core/checkpoint/curriculum
selection remain future handlers on the same organ boundary. Packet
preparation spends nothing; Kaggle submission remains an explicit operator
action.
