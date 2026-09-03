# Axon Kaggle Training Guide

Status: operational private-launch surface, 2026-09-03

## The easy path

Double-click `AXON_KAGGLE.bat`. It opens the Axon Kaggle Control Center in its
own PowerShell window. The menu can check quota, prepare a packet, launch it,
open a separate live monitor, inspect status, and download outputs.

The first prepared recipe is
`configs/kaggle/d64_first_cloud_tranche.json`. It creates a fresh private D64
tournament lane with the current C1 curriculum and a renewable 64-step
resource tranche. Sixty-four is not a lifetime ceiling: a later exact-resume
packet may grant another tranche to the same candidate lineage.

The first objective language recipe is
`configs/kaggle/d64_language_l0_l4_first_tranche.json`. It uses the published
349-case L0-L4 Language Foundations manifest with the permanent 1x64-head,
two-layer, 131072-FFN Candidate-A tissue. Its 360-step tranche is derived from
the six-lane family scheduler: it reaches every L0-L4/mechanism lane and gives
the largest 60-case language lane one complete exposure cycle. It is renewable
work authorization, not a maximum lifespan. The job evaluates only the
`VERIFIED_TARGET` standard-curriculum surface; synthetic mechanism heldout and
`PROCESS_EVIDENCE` prose cannot contaminate the language capability gate.

The smallest learned-sequence diagnostic is
`configs/kaggle/d64_abc_sequence_first_tranche.json`. It carries 104 objective
ABC/native-bank sequence cases and grants a renewable 180-step first tranche.
The recipe requires CUDA; it must fail rather than silently train the permanent
33.98-million-parameter tissue on a cloud CPU.

For GPU jobs the adapter explicitly requests Kaggle's `NvidiaTeslaT4` machine
shape in both kernel metadata and the CLI submission. Do not use Kaggle's
generic/default GPU selection: it may resolve to an unsuitable image or P100,
while the current default PyTorch cu128 build does not support P100 compute.

Preparation and launch are intentionally separate:

1. **Prepare** hashes committed executable source plus every explicitly named
   State input. It writes a local packet and uploads nothing.
2. **Launch** requires typing `LAUNCH`. It creates a private Kaggle dataset,
   submits a private Kaggle script, and may begin consuming accelerator quota.
3. The control center opens a separate monitor terminal. Closing Codex, the
   control center, or the monitor does not terminate the Kaggle job.
4. **Fetch** downloads the durable result and observability artifacts into the
   job's local `State/training/cloud/jobs/<job-id>/outputs` directory. Download
   is evidence import, not serving activation or promotion.

To reopen monitoring without the menu, double-click
`MONITOR_AXON_KAGGLE.bat` and paste the job ID.

## What is observable

The training harness now publishes one flushed `AXON_PROGRESS` JSON event for
startup, every accepted optimizer step, evaluation, pause, and completion.
Each event contains the global step, loss, curriculum lane, wall time, and
checkpoint/accepted-bundle identities when present. Kaggle preserves those
lines in native kernel logs.

The same events are also durable job outputs:

- `axon_observability/trainer/events.jsonl` — append-only progress history;
- `axon_observability/trainer/current.json` — atomic latest Trainer event;
- `axon_observability/runner_events.jsonl` — packet/runner lifecycle;
- `axon_observability/runner_current.json` — atomic latest runner event;
- `axon_job_result.json` — terminal return code, job identity, and Git revision.

The official CLI provides live status, followed logs, and output retrieval.
The monitor uses those provider-native facilities; it does not depend on an
engineer's model session or subprocess lifetime.

## Credentials and privacy

This machine is authenticated with Kaggle's official OAuth flow. The official
CLI owns credentials in its user credential store. Axon does not read, copy,
print, package, or commit them.

Safety is fail-closed:

- a cloud packet requires a clean committed Git revision;
- only committed executable directories are selected automatically;
- operational State must be named explicitly;
- any State upload requires `allow_sensitive_state_upload: true` in the exact
  job recipe;
- credential filenames, key material, symlinks, and paths outside `D:\Axon`
  are rejected;
- the generated dataset and kernel are private, public upload flags are never
  used, and kernel internet access is disabled;
- launch requires explicit operator confirmation;
- cloud results cannot activate a candidate or rewrite canonical State merely
  by being downloaded.

The current first-cloud recipe includes Axon's canonical active branch and
therefore identity-bearing State. Jeff explicitly authorized private Kaggle
training for Axon; this acknowledgement is recorded in that job recipe. A new
recipe must make its own exact State selection and acknowledgement.

## Command-line equivalents

```powershell
python scripts/axon_kaggle.py doctor
python scripts/axon_kaggle.py jobs
python scripts/axon_kaggle.py prepare configs/kaggle/d64_first_cloud_tranche.json
python scripts/axon_kaggle.py launch <job-id> --yes
python scripts/axon_kaggle.py monitor <job-id> --follow
python scripts/axon_kaggle.py status <job-id>
python scripts/axon_kaggle.py fetch <job-id>
```

`doctor` makes an authenticated live quota request. On 2026-08-31 it verified
30 GPU hours and 20 TPU hours remaining, with the provider reporting a refresh
at 2026-09-05T00:00:00. Always run it again before planning spend; Kaggle's
quota is external and can change.

## Continuation packets

Do not call a job a continuation merely because it uses the same label. Exact
continuation requires the accepted parent bundle, its parameter checkpoint,
optimizer state, private-Soul HEAD, governing plan/policy/authorization, prior
tranche evidence, curriculum, and report lineage. The generic packet builder
accepts explicit include paths so this closure can be transported, but the
first recipe intentionally starts fresh. A dedicated closure resolver should
be used before launching a C1 continuation; omitting any parent evidence must
fail rather than silently restart.
