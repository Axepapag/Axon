# Axon Home: phone foundation

Codex / model ID unavailable / 2026-09-21

The `local` flavor is a **body-storage foundation**, not a completed Axon runtime.
It hosts a transaction boundary for exact user ingress, the 13-region v4 field,
canonical Identity v1, persistent masks, pause/resume, and body backup/restore.
SQLite atomically commits body plus local audit history. It never increments a
reasoning tick, emits a proposal, or pretends that a Core or cloud worker exists.

`simulation` preserves Kimmy's original operator-console prototype in a separate
application ID. Its UI, vault and automatic tick driver are excluded from the
local flavor. Never distribute its APK as the living Axon runtime.

## Build

JDK 17, Android SDK 35, Gradle 8.10.2:

```
gradle :core:test :controlplane:test :app:testLocalDebugUnitTest :app:testSimulationDebugUnitTest
gradle :app:lintLocalDebug :app:assembleLocalDebug
```

GitHub Actions produces `axon-home-local-debug-apk`. Debug signing is for internal
installation only. Updates require the same signing certificate; do not uninstall
an existing app containing data without exporting a verified backup first.

## Phone use

1. Install the local APK and open Axon Home.
2. Enter text and choose **Save input to Heart**. It persists locally; no AI reply
   is expected before an accepted inference backend is installed.
3. Select a Shared Field region; the mask slider changes attendance without
   deleting text. Identity always remains attended.
4. **Export body backup** opens Android's document picker. Choose Google Drive
   if its document provider is available. This is manual file exchange, not
   unattended OAuth sync. Readback confirms provider bytes, not cloud replication.
5. **Restore body backup** verifies the checksum, schema and exact Identity, then
   asks before changing the active body. Earlier states remain in SQLite history.

Backups are explicitly **body-only**, not full organism recovery: no weights,
Soul, Dormant corpus, optimizer state or model lineage are included. They are
not accepted by the existing full recovery-capsule interface.

## Remaining work and authority boundary

- No mobile learned backend, exact 16D/D64 rail implementation, FIRST/REFINED/
  rotating FINAL loop, Soul transitions, or full coverage receipt exists here.
- `CoreInferenceBackend` is an inactive interface, not an installed model loader.
- No trained/accepted mobile checkpoint was supplied. Existing ledger evidence
  does not establish model mastery. Do not relabel old failed checkpoints.
- Cross-language canonical hash compatibility is restricted to locally generated
  spans with confidence 1.0. Imported floating-point confidence forms are rejected;
  general Python/JVM float and supplementary-key ordering parity remains work.
- `training/living_reasoning_d64.py` uses a recurrent reader, addressable copy
  memory, decoder state and layered Soul inhale. Export must explicitly preserve
  those tensors and ordinary EOS. ONNX vs ExecuTorch is not yet selected or proven.
- The new local body is a fresh phone body; it does not recover the offline
  Windows server's state or claim its cardiac identity/lease continuity.
- No remote training command is dispatched. `notebooks/Axon_Cloud_Preflight.ipynb`
  prepares a Colab worker and preserves zero-step preflight evidence in Drive.
  A new training tranche and accepted mobile export still need engineering.

The earlier `docs/AXON_HOME_ARCHITECTURE.md` is preserved as source evidence.
Its "phone is not a compute node" premise is superseded by Jeff's phone-host
mission in the 2026-09-21 handoff. Locked doctrine is unchanged.
