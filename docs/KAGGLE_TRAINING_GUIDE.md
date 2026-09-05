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
shape in kernel metadata. Do not use Kaggle's
generic/default GPU selection: it may resolve to an unsuitable image or P100,
while the current default PyTorch cu128 build does not support P100 compute.
The CLI accelerator override is intentionally not used because Kaggle CLI 2.2.4
can drop `dataset_sources` when that override is supplied. The generated runner
also accepts Kaggle's two legitimate dataset presentations: the original ZIP
or an automatically unpacked input directory.

Transport datasets remain private and use Kaggle's `other` license category
with an explicit all-rights-reserved/non-redistribution description. The live
backend currently rejects the documented `copyright-authors` slug. Kaggle CLI
2.2.4 can print a dataset-creation error while returning exit code zero, so the
adapter also parses that semantic failure and refuses to submit a kernel.
After creation, Axon waits for Kaggle's authenticated dataset status to become
`ready` before submitting the kernel. The runner prefers the declared dataset
slug but can discover the single hash-verified packet manifest anywhere under
`/kaggle/input` if the provider rewrites the private mount name.

Kaggle script kernels may start under `/usr/bin/python3` even on a T4; that
interpreter can carry a CPU-only PyTorch build. Before training, the generated
runner probes the available Kaggle Python environments and selects one only
after `torch.cuda.is_available()` and a real CUDA tensor operation both pass.
Every probe result is emitted to the runner log. A GPU recipe fails closed when
no interpreter can execute CUDA; it never falls back to CPU.

The controller is submitted as a standard one-cell Kaggle notebook rather than
a script kernel. Kaggle's current script-kernel path can report a T4 machine
shape while still selecting its CPU Docker image; the notebook path is the
provider-native GPU runtime. The cell contains the same generated, auditable
runner source and performs the same packet hash verification before training.

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
  used, and kernel internet access is disabled unless a recipe explicitly opts
  in to mid-run sync (`sync_mid_run`), which is the only feature that uses it;
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

Running `launch <job-id> --yes` again after an errored or completed/fetched
version submits a new private version from the same immutable packet and input
dataset. It first checks provider status and refuses to duplicate a running,
queued, or pending job. The double-click control center exposes this as
"Launch or retry"; no raw Kaggle command is required.

Retrying a packet reruns that packet's original inputs; it is NOT a learning
continuation from outputs it subsequently produced. For continued learning,
prepare a new `--resume` packet with the latest accepted parameter/optimizer/Soul
bundle and its lineage evidence. Preserve completed outputs before any deliberate
provider-version rerun. Never interpret another submission as another accepted
tranche without checking its exact parent.

Dataset-upload success is durably recorded before waiting for Kaggle's indexing
to become ready. A temporary status 403 or readiness timeout therefore leaves
the job at `dataset_uploaded`, and the next confirmed launch continues from
that point without creating a duplicate dataset. Unknown worker statuses are
not treated as permission to resubmit.

The notebook wrapper preserves the training subprocess's exit receipt. Normal
exit code zero is success, not an exception; a failed learning gate is reported
separately from process failure. Historical outputs from the first Organism
tranche contain a wrapper-generated `SystemExit: 0` failure receipt even though
Kaggle completed all 360 steps. Preserve that artifact as historical evidence;
use its immutable segment report and accepted bundles for the learning result.

The Training Watch dashboard deduplicates replayed Kaggle event IDs and reports
both progress within the current tranche and the lifetime global optimizer step.
Its Q/A samples are teacher-forced payload diagnostics, not evidence of fluent
autonomous conversation. Provider and runner failures remain visible even if no
training step is reached.

Cloud-mode regression coverage must enable `--progress-dir` through both a real
first tranche and an exact resume. The `evaluated` status carries completed
metrics; accepting only `evaluating` is insufficient. Console JSON escapes
non-ASCII characters losslessly for legacy Windows pipes, while durable journals
and reports remain UTF-8. A console encoding must never determine what Axon can
learn or represent.

`doctor` makes an authenticated live quota request. It proves that the CLI and
account session work and reports the provider's current quota. It does **not**
prove accelerator entitlement: Kaggle may accept GPU metadata and expose quota
while withholding the GPU/TPU choices until account phone or identity
verification is complete. Every submitted GPU job therefore performs a real
CUDA allocation and compute probe before optimizer step one and fails closed if
Kaggle supplied a CPU image. Complete any verification request shown under the
notebook's Session options before retrying; never weaken the probe or silently
fall back to CPU.

On 2026-09-03 the Axon account reported 30 GPU hours and 20 TPU hours remaining,
but the live Session options panel explicitly requested phone verification for
GPU/TPU access. That external account action blocked the first ABC launch even
though the private dataset, T4 metadata, and packet mount were valid. Always run
`doctor` again before planning spend; Kaggle's quota and entitlement can change
independently.

## Bundled fetch and mid-run sync (ratified 2026-09-04)

Every new packet now ends its run by writing two files into the kernel output
root: `axon_outputs_<job-id>.tar.gz` and `axon_outputs_<job-id>.sha256.json`
(a detached manifest mapping every member path to its SHA256, plus the
archive-level hash). This happens at normal completion and at a governed
failure receipt, so even a crashed run leaves one verifiable archive.

`fetch` is bundle-first by default:

```powershell
python scripts/axon_kaggle.py fetch <job-id>          # bundle-first, automatic
python scripts/axon_kaggle.py fetch <job-id> --no-bundle   # force legacy per-file
```

The bundle path downloads exactly two files, verifies the archive hash,
extracts into a short temp staging dir, rehashes every member against the
manifest, and only then moves the tree into
`State/training/cloud/jobs/<job-id>/outputs` (robocopy, long-path aware). Any
mismatch quarantines the bundle under `jobs/<job-id>/quarantine/` and stops
loudly — nothing half-verified reaches canonical outputs. The verified
manifest is kept in the outputs directory as the transfer contract. Jobs
launched before this feature have no bundle; `fetch` detects that and falls
back to the legacy per-file download automatically. The job record notes which
path was used (`fetch_mode`).

### One-time Kaggle UI setup for mid-run sync (Jeff only)

Mid-run sync is opt-in per recipe (`"sync_mid_run": true` in the recipe JSON).
It needs two one-time manual steps in the Kaggle web UI; engineers never
handle the token:

1. In **Settings → User Secrets** (or the kernel's **Add-ons → Secrets**),
   create a secret named `AXON_KAGGLE_SYNC`. Either attach the account's API
   token so the kernel environment carries `KAGGLE_USERNAME`/`KAGGLE_KEY`, or
   make the secret value a JSON payload `{"username": "...", "key": "..."}`
   holding an API token for the `axongliksbot` account.
2. Attach that secret to the job kernel and make sure the session runs with
   **internet enabled** (the adapter sets `enable_internet` in kernel metadata
   automatically for sync recipes; Kaggle still requires the account to allow
   it).

The packet treats the secret as write-only: it is never printed, logged,
receipted, or persisted. If the secret is absent or internet is unavailable,
training runs correctly with sync disabled — one journal note, no failure.

When enabled, after every accepted checkpoint boundary (every 30 steps) the
trainer bundles the artifacts produced since the last boundary into
`sync_<job-id>_steps_<a>_<b>.tar.gz` plus manifest and pushes it as a new
version of the private dataset `axongliksbot/axon-job-<short>-sync`
(`<short>` = first 8 characters of the job id). Uploads run on a daemon
thread; GPU compute never waits on the network. A failed upload is receipted
and retried at the next boundary with an extended step range — it is never a
training failure. Sync receipts also land in
`axon_observability/trainer/sync_receipts.jsonl` inside the run outputs.

### Watching synced artifacts locally

```powershell
python scripts/axon_kaggle.py sync-pull <job-id>     # download + verify + extract latest sync version
python scripts/axon_kaggle.py sync-status <job-id>   # which step ranges are locally verified
```

`sync-pull` verifies and rehashes everything into
`State/training/cloud/jobs/<job-id>/sync/` (`bundles/`, `members/`,
`receipts/`); `sync-status` rehashes again by default (`--no-rehash` is a
faster, weaker view) and reports contiguous verified coverage. A crash at
step 700 still leaves verified local artifacts through the last pulled
boundary. Kaggle dataset downloads always serve the **latest** version, so
run `sync-pull` regularly during a watched run; anything missed remains fully
recoverable from the end-of-run bundle.

**Synced mid-run artifacts are observation-only.** They are evidence under
the same provenance law, but they never write into canonical training State,
never authorize a continuation, and never relax the exact-parent
tranche-renewal rule. The continuation decision stays with Jeff after the
immutable segment and final report arrive and are verified.

## Continuation packets

Do not call a job a continuation merely because it uses the same label. Exact
continuation requires the accepted parent bundle, its parameter checkpoint,
optimizer state, private-Soul HEAD, governing plan/policy/authorization, prior
tranche evidence, curriculum, and report lineage. The generic packet builder
accepts explicit include paths so this closure can be transported, but the
first recipe intentionally starts fresh. A dedicated closure resolver should
be used before launching a C1 continuation; omitting any parent evidence must
fail rather than silently restart.
