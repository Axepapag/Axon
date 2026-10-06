# Foundation validation, 2026-10-06

Verified in the current `G:\My Drive\Projects\Axon` checkout:

- Full suite: **165 passed**, including 14 new backend tests. Latest run after
  registration deduplication: `python -m pytest -o addopts='' -q`, 15.32 seconds.
- Both D16 and D1024 self-tests exited 0.
- Live HTTP health returned 200 on loopback port 8184. Async preflight returned
  202 while the UI remained available. Persisted operation
  `06e86432-8512-46d1-aa29-2bb2e6e58b64` completed with all five checks passing.
  The actual GPU tensor operation passed on NVIDIA GeForce GTX 1650, CUDA 0,
  compute capability 7.5, 4,294,508,544 VRAM bytes, Torch 2.13.0+cu126.
  `training_performed` and `training_authorized` remain false.
- Kimi Browser Extension read the real local frontend: **Backend connected**,
  actual API schema/version, unavailable model adapters and readiness blockers.
  A screenshot at 390x844 showed the connected Backup view with truthful
  unavailable destination/restore evidence. This is browser emulation, not
  physical phone acceptance or full training-control acceptance.
- The actual public browser at `https://axon.gliksbot.com` loads Axon Lab but
  still shows **Disconnected / Request failed (404)**. No public API proxy or
  shared launcher change was made in this slice.
- Scoped `git diff --check` exited 0. Backend files are untracked, so pytest
  provides their executable validation. Repository-wide diff checking also
  reported existing whitespace in immutable historical ledger lines; those
  bytes were preserved.

The service was restarted after the last source change. It is a manually
started hidden loopback process, not yet managed by the Cloudflare launcher.
Local registry persistence is not a tested off-device backup. Model construction,
training, inference, runtime event streaming and checkpoint recovery are pending.

Bus notifications: `c3bb5f39-1acb-4d98-b37f-4812c00efc2a` and
`d11661fe-d595-4cf0-a77a-c1fce1e2a146`. Frontend owner follow-ups: explicit device
preflight and polling, registration command IDs, native-alphabet hints, and
component configuration/edge editing after real contracts arrive.
