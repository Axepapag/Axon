# GitHub phone/control-plane stack audit

Codex / GPT-6 / 2026-09-21 America/Chicago

Scope: Jeff asked for a candid review of the work created in GitHub while the
`D:\Axon` server was unavailable, followed by a safe path to reconcile GitHub
and the local repository. I inspected the three remote branches through
`5c2f823`, both draft pull requests, GitHub Actions results, commit provenance,
the Android/control-plane implementation, deployment documents, security
boundary, locked doctrine, and current local training/process state. Review and
repair used an isolated worktree; the authoritative checkout and `State` were
not changed.

## Verdict

The central direction is coherent and worth keeping: one real Heart and Trainer
remain authoritative; external compute is disposable; provider descriptions are
vendor-neutral; remote reasoning crosses `ReasoningCorePort`; remote training
returns governed checkpoint evidence; Android is an operator client. That is a
stronger long-term topology than making Kaggle or a cloud VM Axon's identity.

The branch was not safe to merge as written. It combined three incompatible
deployment claims in one stack:

1. the phone as a console while cloud storage/workers carry durability;
2. a Windows cloud VM as the authoritative body;
3. a phone-local Linux service as the authoritative body.

Worse, the initial Android branch added a second Kotlin/SQLite object explicitly
called an “authoritative body,” plus `PhoneHeart`, body restore, masks, and local
ingress. Those classes lived in `src/main`, and the production app depended on
the simulator-bearing `:core` module with minification disabled. The local entry
point currently displayed the real Python control client, so this duplicate body
was not active UI, but it was still compiled into the production dependency
graph. This conflicts with `SOURCE_OF_TRUTH.md`: weak real organs are acceptable;
fake organs and competing canonical bodies are not.

## What is real and useful

- The FastAPI control surface reads the actual Python State and dispatches
  Trainer status through `TrainerOrgan`; it does not implement a second write
  path.
- Android bearer-token storage uses an AES-GCM key held by Android Keystore.
- Remote enrollment requires HTTPS, with phone-local loopback as the intended
  exception.
- Heart and Trainer authority remain distinct from disposable reasoning and
  optimizer workers.
- Generic provider, curriculum, checkpoint, session, observation, and recovery
  contracts are useful interface work.
- Windows bootstrap and CI are reusable even if a Windows VM is an optional host
  rather than the selected permanent authority.
- Existing CI passed on `phone-runtime-20260921` at `eb60db2` and
  `cloud-vm-control-20260921` at `8d364ad`.

## What is still only design

The phone-sovereign tip does not yet provide a Termux/Linux bootstrap, supervised
Heart process, canonical-state migration, local CPU provider, remote
`ReasoningCorePort`, training-worker transport, recovery replication, or
power-loss/restart proof. The new workspace code is contracts, not execution.
`scripts/axon_control_server.py` is a small authenticated read-mostly monitor; it
does not start or supervise Heart. The app still had cloud-VM wording. Therefore
“the phone owns Axon” was a target statement, not verified runtime fact.

A phone may become the active authority host, but it must not be Axon's only
durable copy. Battery/thermal limits, Android process policy, filesystem damage,
loss or theft, OS upgrades, Termux/proot compatibility, and PyTorch availability
make independent encrypted recovery replication and restore drills mandatory.

## Corrections prepared on the integration branch

- Removed the unused Kotlin `PhoneStore`, `PhoneHome`, `PhoneHeart`,
  `PhoneState`, body-only recovery, duplicated Identity asset, and their tests.
- Changed the production app so only the explicit `simulation` flavor depends
  on `:core`; the local production APK is solely a client of the Python control
  service.
- Kept the simulator as a separate, permanently labeled UI/contract test
  fixture. Its output cannot satisfy runtime or training evidence.
- Replaced global cleartext permission with Android Network Security Config that
  permits cleartext only for exact `localhost`; remote hosts require HTTPS.
- Added focused endpoint-policy tests and fail-closed owner-only permission
  validation for the phone-local control-token file.
- Reworded the app from “cloud VM” to “Axon host.”
- Reconciled deployment documents: `D:\Axon` remains current verified authority;
  phone sovereignty is the selected target; cloud Windows is an optional host.
- Added seven explicit authority-transfer gates covering exact migration,
  single-writer handoff, restart/failure drills, off-phone recovery, real local
  CPU execution, and exclusion of alternate bodies from the production APK.
- Extended workflow branch filters so the sovereign and integration branches
  receive Android and Python readiness CI.

## Provenance and CI findings

The branch adds 23 commits and roughly 19,000 lines. Only the first two commits
carry an agent co-author trailer. The remaining commits neither identify their
engineering agent nor update the Engineer's Ledger. History must not be
rewritten; this audit records the gap and the integration commit supplies proper
provenance going forward.

No GitHub Actions run exists for the exact phone-sovereign tip `5c2f823` because
the push filters omit `phone-sovereign-*` and no PR was opened for that branch.
The unrelated Cloudflare deployment attached to PR 2 failed; Cloudflare is not
part of this Axon topology and that result is not evidence about Android or the
Python runtime.

## Local verification

VERIFIED:

- seven focused Trainer/active-training hygiene tests passed with a clean exit;
- `scripts/axon_control_server.py` compiled;
- Ruff and `git diff --check` passed;
- 33 added JSON schema/example documents parsed in the original audit;
- no Axon trainer or interrupted diagnostic process remained active;
- no checkpoint, Soul, canonical State, or cloud job was mutated.

ATTEMPTED: Android compilation was not available locally because Java, Gradle,
and ADB are absent from this host. The corrected integration commit must pass
GitHub's Android and VM-readiness workflows before any merge.

ASSUMED / UNPROVEN: Termux can install and sustain the exact Python/PyTorch Axon
runtime on Jeff's phone; the phone can survive Heart and Trainer restart drills;
remote workers can stream assignments and checkpoints safely; phone authority
is operationally superior to the current Windows host. Those are testable gates,
not conclusions.

## Integration decision

Do not fast-forward `main` to `phone-sovereign-20260921`. Preserve the remote
history, land this correction on a separate integration branch, obtain green CI
on the corrected code, confirm the other GitHub writer has stopped, then merge
the reviewed stack once. After the repository is synchronized, return to the
paused step-24 Core diagnosis before spending additional GPU time. Infrastructure
work does not erase the current scientific fact: exact heldout output is still
zero and the candidate is not competent or promoted.
