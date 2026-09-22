# Phone-Sovereign Axon

> **Target status (2026-09-21):** This is the selected deployment direction,
> not current runtime fact. The verified Heart, canonical State, Trainer, Cores,
> and Souls remain at `D:\Axon`. The phone becomes authoritative only after the
> acceptance gates below pass on an exact governed migration. Until then Axon
> Home is a client of the existing Python host and must never create a second
> Kotlin/SQLite body.

Status: implementation target for Axon Home while no permanent external host is assumed.

## Authority-transfer acceptance gates

All gates are required before the phone may be called Axon's home:

1. A reproducible Termux/Linux bootstrap installs the actual repository and its
   compatible Python/PyTorch runtime without substituting a mobile imitation.
2. A supervised `HeartHost` restart preserves the exact branch HEAD, Heart
   identity, private Soul lineages, Trainer authority, and accepted checkpoint
   lineage.
3. An exported recovery capsule from `D:\Axon` verifies byte-for-byte on the
   phone before activation; the Windows Heart is stopped and its writer lease is
   released before the phone acquires authority.
4. Kill, reboot, low-storage, interrupted-write, and interrupted-checkpoint
   drills recover without split brain or silent rollback.
5. At least one encrypted, content-addressed recovery replica lives off the
   phone and is restored in a drill. The phone is the active authority host, not
   the only durable copy of Axon.
6. A real local CPU reasoning/training smoke crosses the same governed Heart and
   Trainer seams used by external workers. No simulated result may satisfy this
   gate.
7. The production APK contains no alternate canonical body, synthetic Heart,
   synthetic Soul, or simulator engine. Simulation remains an explicitly
   separate build used only for UI and contract testing.

## Authority rule

After those gates pass, the Android phone is the organism's home. The authoritative Python body runs on the phone (initially under Termux + an Ubuntu/proot userland). External machines are optional workers only.

The phone owns and persists:

- HeartHost and the single canonical State branch,
- exact Shared Field and D16 substrate truth,
- rail compilation and tick identity,
- Cortex/Dormant coordination,
- Core registry and private Soul lineage,
- Trainer authority, curricula, gates, accepted checkpoint lineage, and recovery slots,
- provider profiles and worker assignments.

Axon Home is the operator UI. It talks to the phone-local Python control service over loopback. A future embedded-Python build may remove the Termux companion, but must preserve the same control-plane contracts and authority boundaries.

External compute is supplemental. Loss of every external provider must leave the phone with a valid, runnable organism at the latest locally verified continuation boundary.

## Topology

```text
Android phone
+-------------------------------------------------------------+
| Axon Home APK                                               |
|   Runtime | Trainer | Cores | Curricula | Providers | Config|
|                    |                                        |
|                    v localhost                              |
| Termux / Ubuntu / Python                                    |
|   HeartHost -> Shared Field -> D16 -> Rails                 |
|       |                |          |                         |
|       |                |          +-> local Core(s)         |
|       |                +-> Cortex / Dormant                 |
|       +-> Souls / Trainer / checkpoints / provider registry |
+----------------------+--------------------------------------+
                       |
                       | optional worker protocols
          +------------+-------------+------------------+
          |                          |                  |
       HTTP worker                 SSH worker         local CPU
       GPU/TPU/CPU                 GPU/CPU            phone CPU
          |                          |                  |
    training or reasoning      training/reasoning   training/reasoning
```

No provider brand is an Axon architectural concept. Kaggle, Colab, Lightning, a friend's computer, a VM, a workstation, or a future service are merely ways to obtain a worker endpoint.

## Generic provider profile

The app stores a named profile with a transport and capabilities, not a `Kaggle` or `Colab` special case.

Required transport kinds:

- `local` — phone-local Python worker,
- `http` / `https` — Axon Worker Protocol endpoint,
- `ssh` — remote shell/transfer transport used to install or invoke the same worker protocol.

A profile may advertise one or both purposes:

- `reasoning` — can host one or more reasoning Core execution ports,
- `training` — can execute governed optimizer work.

Credentials are stored by reference to Android Keystore-backed secret material. Raw secrets are never written to canonical State, RoundTable, Git, training packets, or logs.

Capability discovery is authoritative. The UI should show only operations a worker claims and proves it can perform, while never imposing arbitrary architecture-size ceilings of its own.

## Remote reasoning cores

`runtime.heart.circulation.ReasoningCorePort` is already the correct seam. The Heart invokes a Core through one `emit(ReasoningPassRequest)` call. The implementation may be local, threaded, another process, HTTP, SSH-backed, or another transport.

A remote reasoning worker receives only the frozen derived material needed by the Core:

- Core identity and architecture generation,
- immutable tick/image identity,
- the Core's rail at its registered width,
- proposal-workspace rails needed by the current pass,
- that Core's private Soul inhale.

It returns:

- the English proposal or consolidated FINAL,
- the private Soul transition,
- binding/coverage identities needed for Heart validation.

The remote worker is never a canonical writer. Heart on the phone validates the result and commits or rejects it. Round-robin consolidation does not care whether the selected Core is on-device or remote.

## Rail widths

The UI and provider protocols must not assume D64 forever. A Core declares its `d_model`; Heart supplies the matching derived rail. D64 remains the currently implemented learned Core family, but the control plane must represent wider future families without an artificial UI cap.

The current `LivingReasoningCoreConfig` still intentionally locks the active learned implementation to D64. Generalizing the runtime/model family to D128/D256/etc. is a separate learned-core change, not something the app should fake by relabeling a D64 model.

## Generic Trainer

The Trainer remains phone-authoritative. A remote worker is an optimizer executor, not a second Trainer.

The app's Trainer tab owns these operator workflows:

1. choose/create a Core,
2. choose a checkpoint or create a genesis generation,
3. inspect anatomy (`d_model`, heads, layers, FFN, state tokens, page size, parameter bytes),
4. inspect training age (optimizer steps, accepted steps, phase, stage, curriculum, latest gates/loss),
5. choose/build a curriculum,
6. choose a saved provider or add a new generic provider,
7. choose tranche policy and launch,
8. open any active run and monitor samples, predictions, targets, loss, gates, throughput, worker status, and checkpoint transfer state.

Local CPU training is a valid provider. It may be slow; Axon must not reject it merely because it is inefficient.

## What must cross the network for remote training

A remote training worker does **not** need the canonical Heart, full Dormant store, full Shared Field history, or the Android app.

It does need enough material to perform an optimizer step correctly:

- the exact Core parameters/buffers,
- optimizer state when continuing,
- the Core's training Soul/state when the curriculum requires it,
- executable model/loss/backprop code (or a compatible installed worker version),
- the current learning policy/recipe,
- the lesson minibatch or episode and its target/supervision,
- exact architecture and lineage identities.

Therefore Axon should upload the relatively large model/optimizer state once per assignment, then stream or prefetch small lesson batches from the phone. The worker performs forward -> loss -> backward -> optimizer locally and streams observations back.

Per-character network round trips are deliberately avoided. They would make latency part of the learning algorithm. A worker receives a complete governed unit (batch/episode/tranche input) and executes it locally.

## 15-step tranche policy

Default phone policy: 15 optimizer steps per recovery tranche. This value is an operator setting, not a hard architecture limit.

At each tranche boundary the worker freezes an exact continuation bundle containing at minimum:

- parameters/buffers,
- optimizer state,
- bound Soul/training state,
- accepted step number,
- architecture/candidate/curriculum/policy identities,
- evaluation/gate receipts required by the Trainer,
- hashes/manifest.

Training may continue while the bundle transfers, but the claim "at most 15 steps can be lost" is only true if the previous boundary is already durable somewhere other than volatile worker RAM. Therefore the scheduler uses a one-boundary durability rule:

- worker may continue immediately after creating boundary N,
- boundary N must be durable either in provider-persistent storage or verified on the phone before boundary N+1 is allowed to be passed,
- if transfer/storage lags, training pauses at the next boundary rather than silently increasing the loss window.

This permits overlap without lying about recovery guarantees.

## Two full checkpoint slots, unlimited tiny lineage receipts

The phone retains two full continuation payload slots, `A` and `B`.

A new boundary is downloaded into staging, fully hash-verified, then atomically replaces the older slot. Never overwrite the last verified slot before the replacement verifies.

Example:

```text
A = verified step 30
B = verified step 45
worker reaches step 60
-> download step 60 to staging while worker begins 61..75
-> verify step 60
-> atomically replace A
A = verified step 60
B = verified step 45
```

Full parameter payloads do not stack indefinitely. Small manifests, hashes, gate receipts, loss records, curriculum identities, and lineage events may remain append-only because they are tiny and are required to audit what happened.

## Training monitor

Every active training session exposes an observation stream. At minimum the app can display:

- provider/worker and accelerator,
- Core/checkpoint/candidate identity,
- architecture anatomy,
- stage/phase/curriculum,
- optimizer step and tranche position,
- loss and learning rate,
- recent gate metrics,
- recent training example/input,
- expected target/supervision,
- Axon's actual prediction/output,
- exact/correct status when defined by that curriculum,
- checkpoint A/B/staging status,
- last locally verified recovery step,
- worker heartbeat/throughput/resource observations when available.

Observation data is not permission to mutate canonical Trainer state. It is telemetry. Acceptance remains governed by the Trainer.

## Curriculum builder

Curricula are first-class content-addressed definitions. Axon Home should support both:

- choosing an existing immutable curriculum version,
- drafting a new curriculum in the app, validating it, then sealing it to a new content identity before training.

A curriculum definition should expose its stages/gates/lesson sources and human-readable examples. The UI must not silently edit an already-used curriculum identity; edits create a new version.

## No arbitrary inference ceilings

Axon Home should expose physical observations and warnings, not hidden policy caps. The operator may register many Cores or large future Core families even if the current phone will run them poorly.

Hard rejection is appropriate only where the implementation is mathematically incompatible (for example a model family whose tensor shapes cannot execute), the model would violate canonical/lineage contracts, or the operating system/runtime actually cannot allocate the resource. "This may be slow" is not a reason to rewrite the user's requested configuration.

## Immediate implementation order

1. Make the existing Python control service portable to Linux/Termux and use phone-local State by default.
2. Add Termux/Ubuntu bootstrap + start/stop/status scripts.
3. Make Axon Home accept a loopback runtime and auto-reconnect to it.
4. Replace brand-specific cloud concepts in the Android surface with generic Provider Profiles.
5. Add Trainer workspace contracts: Core/checkpoint inventory, curriculum inventory/editor, sessions, observations, recovery-slot telemetry.
6. Implement local CPU provider first; this proves the complete phone-only path.
7. Implement generic HTTP worker protocol.
8. Implement SSH as a bootstrap/transport adapter that installs or invokes the same worker protocol.
9. Implement HTTP remote `ReasoningCorePort`.
10. Only then add convenience presets for specific services. Presets may fill endpoint/transport details but may not create provider-specific Trainer semantics.
