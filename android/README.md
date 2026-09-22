# Axon Home

Codex / GPT-5 / 2026-09-21

The `local` flavor is a secure client of the real Python Axon control service.
It stores an enrolled control endpoint and bearer token under Android Keystore,
then reads live Heart, canonical branch, Cortex/ingress, Trainer, and training
telemetry. It contains no Kotlin Heart, canonical Shared Field, Soul store, or
alternate phone body.

The `simulation` flavor is an explicitly separate UI and contract-test fixture
with a separate application ID and permanent `SIMULATION` labeling. Only that
flavor depends on the Kotlin `:core` simulator module. Never distribute its APK
as the living Axon runtime or use its outputs as training/runtime evidence.

## Build

JDK 17, Android SDK 35, Gradle 8.10.2:

```
gradle :core:test :controlplane:test :app:testLocalDebugUnitTest :app:testSimulationDebugUnitTest
gradle :app:lintLocalDebug :app:assembleLocalDebug
```

GitHub Actions produces `axon-home-local-debug-apk`. Debug signing is for
internal installation only.

## Enrollment

1. Start `scripts/axon_control_server.py` beside the real Python organism.
2. For a phone-local service, bind only to loopback and enroll
   `http://localhost:<port>`. Cleartext traffic is restricted to exact
   `localhost` by Android Network Security Config.
3. For any remote host, terminate TLS and enroll an `https://` URL.
4. Enter the bearer token once. Axon Home encrypts it with a non-exportable
   Android Keystore key and reconnects automatically.

The current verified authority remains `D:\Axon`. The phone-sovereign target is
defined in `docs/PHONE_SOVEREIGN_AXON.md`; it is not active until every listed
authority-transfer gate passes. Installing the APK alone never moves Axon's
Heart, State, Trainer, Cores, or Souls.
