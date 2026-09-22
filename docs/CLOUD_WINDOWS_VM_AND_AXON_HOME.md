# Cloud Windows VM + Axon Home

> **Deployment status (2026-09-21):** This is a tested optional-host design,
> not the selected authority target. `PHONE_SOVEREIGN_AXON.md` now defines the
> intended topology. The control server, secure enrollment, bootstrap lessons,
> and disposable-worker separation remain reusable. No cloud VM is presently
> authoritative, and the verified organism remains at `D:\Axon`.

**Status:** deployment plan and implementation scaffold for the next Axon host. This does not amend locked Axon doctrine.

## Target topology

If explicitly promoted after recovery and single-writer checks, a cloud Windows
VM can serve as an authoritative runtime host. It runs the existing Python organism:

- `HeartHost` and the one canonical `State` root;
- exact Shared Field / D16 substrate / D64 rail compiler;
- Semantic Cortex and Dormant machinery;
- reasoning Cores and private Soul stores;
- Trainer authority and durable training telemetry.

Axon Home is a control center and monitoring surface. It does not own a second canonical body. The phone enrolls once with the VM's HTTPS URL plus a bearer token stored under Android Keystore, then reconnects automatically.

Training accelerators (later: Google Cloud TPU/GPU workers, including interruptible/cheap capacity) are disposable workers. Training is already tranche-bounded, so provider loss should cost at most the unaccepted work after the last accepted bundle.

## What this branch adds

- `scripts/axon_control_server.py`: authenticated read-mostly FastAPI surface for Heart, branch HEAD, Trainer state and durable training progress.
- `android/app/src/local/.../RemoteHome.kt`: VM dashboard with automatic reconnect and Heart/Trainer monitoring.
- `RemoteConnectionStore.kt`: non-exportable Android Keystore encryption for the persistent VM control token.
- `ops/windows_vm/bootstrap_axon_vm.ps1`: repeatable VM bootstrap, Python venv, startup tasks for Heart/control plane and optional Caddy HTTPS reverse proxy.
- CI for the Android APK and Windows/Python VM readiness.

## VM creation assumptions

When the Google Cloud credit is available, create a **Windows Server VM** with a persistent boot disk and reserve a static external IP. CPU/RAM can be minimal for the body/control plane and increased later if Cortex/runtime pressure requires it. Training acceleration is deliberately separate.

Install these prerequisites on the VM:

1. Git.
2. Python 3.12 x64.
3. Caddy if the phone will connect directly over the public Internet.

Point a DNS name (for example `axon.example.com`) at the VM's static IP. Caddy can then obtain/renew the public TLS certificate automatically. This uses ordinary DNS + HTTPS; there is no tunnel dependency.

The Google Cloud VPC firewall must allow TCP 443 to the VM. Do **not** expose Uvicorn's internal port publicly; the bootstrap binds it to `127.0.0.1` only.

## Bootstrap

From elevated PowerShell after cloning or downloading this repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\ops\windows_vm\bootstrap_axon_vm.ps1 `
  -PublicHost "axon.example.com"
```

The script:

- clones/updates `cloud-vm-control-20260921` into `C:\Axon`;
- creates `.venv` and installs Axon;
- uses `C:\Axon\State` as the canonical state root;
- generates one high-entropy control token under `%ProgramData%\Axon` with restricted ACLs;
- registers `Axon-Heart` and `Axon-ControlPlane` as SYSTEM startup tasks with automatic restart;
- if Caddy is installed and `-PublicHost` is supplied, registers `Axon-HTTPS` and opens the Windows 443 firewall rule;
- prints the one-time phone enrollment URL/token.

## Phone enrollment

Open the local Axon Home APK and enter:

- `https://<your-hostname>`
- the control token printed by the VM bootstrap.

The phone encrypts that token using Android Keystore and keeps retrying the same VM automatically. No SSH key is required for normal Axon monitoring, and the operator does not need to paste the token on each launch.

Administrative RDP/SSH is separate from Axon Home. It should be needed only for VM maintenance, not for normal organism operation.

## Monitoring surfaces

The current control server exposes:

- `GET /v1/health` — control-plane and latest Heart health;
- `GET /v1/field/head` — canonical active branch HEAD;
- `GET /v1/trainer/status` — authoritative `TrainerOrgan` status summary;
- `GET /v1/training/progress` — durable `axon-training-progress-event-v1` current event when present;
- `GET /v1/runtime/summary` — one compact phone polling surface combining all of the above.

The control service deliberately has no shell-command fallback. Trainer mutations remain unavailable unless a governed handler is explicitly registered in `TrainerOrgan`.

## Trainer / legacy state

The active English-native path is `training/living_reasoning_d64.py` plus `training/living_reasoning_curriculum.py` and Stage-0A `training/substrate_literacy_curriculum.py`.

The pre-2026-09-19 typed decision/operation/address curriculum is retired historical evidence under `archive/legacy_typed_reasoning_20260919/`. The remaining `training/legacy_typed_reasoning_d64.py` file is a quarantined migration donor for compatibility tests, not the active trainer. Existing hygiene tests explicitly exclude that donor from active-source scans.

Do not resume an old candidate merely because its checkpoint exists. The next training run is intended to be a **fresh English-native Core generation**. The last Stage-0A gate diagnosis remains evidence for the new run but is not patched in this infrastructure pass.

## Next training pass

After VM deployment and runtime smoke tests:

1. verify Heart restart continuity and canonical branch HEAD;
2. verify Axon Home sees heartbeat/tick/field IDs and Trainer status over HTTPS;
3. create a brand-new English-native Core candidate and genesis Soul;
4. run Stage-0A preflight only;
5. then revisit the last failed gate using the existing EOS/transport evidence before authorizing the first new tranche;
6. add a Google Cloud accelerator adapter only after local/VM preflight is deterministic.

This ordering keeps infrastructure changes separate from the scientific question at the failed gate.
