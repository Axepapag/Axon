# Mid-Run Artifact Sync — Proposal

Author: Kimi / Kimi Code CLI / 2026-09-04
Requested by: Jeff (convener), 2026-09-04 ("download as training is running…
more seamlessly" → "yes please" to a formal spec)
Audience: Codex, ChatGPT, Hermes (reviewers), Jeff (final authority)
Status: PROPOSAL ONLY. Nothing here is doctrine until Jeff ratifies. No code,
gate, schema, budget, or Source of Truth text is changed by this document.

---

## 1. Problem

Cloud training evidence currently moves in one slow, all-or-nothing direction:

- A completed job's outputs are only retrievable as thousands of individual
  per-file HTTP downloads (`scripts/axon_kaggle.py fetch`). The step-361–720
  fetch began 2026-09-04T21:05Z and was still running past 3,500 files — this
  proposal was drafted while it downloaded.
- Kaggle serves kernel output files **only after the run finishes**. There is
  no partial-output download while a kernel is running; the only live channel
  is the log stream, which carries metadata (steps, metrics, hashes) but not
  artifact bytes.
- The CLI's output download has a known pagination defect with large output
  trees (Kaggle/kaggle-cli issue #1045): only the first page of output files
  is downloaded. Our tree is exactly the shape that triggers it.
- A crash late in a run leaves all not-yet-downloaded evidence on the far
  side of that same slow path, and the failure-recovery flow (used for job
  `619b181e`) had to fall back to selective receipt/log fetching precisely
  because full retrieval was impractical.

Codex already named the core fix as active gap #3: "Add bundled, hash-verified
output transfer and an automatic continuation-closure resolver." This proposal
specifies that fix and extends it to mid-run access.

## 2. Binding constraints (unchanged by this proposal)

- **Provenance is law.** Every transferred artifact is content-addressed;
  every bundle carries a SHA256 manifest; local import rehashes everything.
  Download is evidence import, never serving activation or promotion.
- **Credentials never in packets, reports, Git, or logs.** The Kaggle API
  token lives only in Kaggle's User Secrets store and the local official
  credential store. Packet code must treat secret env vars as write-only.
- **Private only.** Sync datasets are private on `axongliksbot`; the existing
  cross-account launch gate (commit `1dedfde`) extends to sync targets.
- **No tissue, curriculum, gate, or architecture changes.** This is transport
  plumbing. It does not alter what is trained, what is accepted, or what any
  gate requires.
- **Tranche law stands.** Easier retrieval must not become automatic renewal;
  the continuation decision stays with Jeff after each verified verdict.

## 3. Design

Two parts sharing one bundling module. Part A alone closes gap #3; Part B
answers Jeff's mid-run question. They are independently shippable; B is the
reason to build A first.

### Part A — end-of-run single archive (closes active gap #3)

- The packet's training process, at normal completion or governed failure
  receipt, writes one `axon_outputs_<job-id>.tar.gz` plus a detached
  `axon_outputs_<job-id>.sha256.json` manifest mapping every member path to
  its SHA256, into the kernel output root.
- `axon_kaggle.py fetch` gains a bundle-first path: download the two files
  (two HTTP requests), verify the archive hash, extract, and rehash every
  extracted member against the manifest. Fall back to per-file fetch only if
  the bundle is absent (legacy jobs).
- Expected effect: completion fetch drops from thousands of requests (measured
  at ~1,500 files per several minutes tonight) to seconds.

### Part B — mid-run sync via private dataset versions

- The kernel runs with **internet enabled** (permitted: `axongliksbot` is
  phone-verified) and an API token attached as a **Kaggle User Secret**
  (env-injected, never printed, never committed).
- After each accepted checkpoint boundary (currently every 30 steps), the
  trainer-side sync hook bundles the *new* artifacts since the last boundary
  (checkpoint, parameter/Soul bundles, journal slice, receipts) into one
  `sync_<job-id>_steps_<a>_<b>.tar.gz` + manifest and pushes it as a new
  version of a private per-job dataset `axongliksbot/axon-job-<short>-sync`.
- Uploads run in a daemon thread so GPU compute is never blocked waiting on
  the network; a failed upload is retried at the next boundary and recorded
  as a receipt, never as a training failure.
- Local side: `axon_training_watch.py` (or the control center) polls the sync
  dataset for new versions on its existing refresh cadence. Each new version
  is one download + one hash verify + extraction into
  `State/training/cloud/jobs/<job-id>/sync/`. The operator watches accepted
  checkpoints land **while the run is still going**.
- Crash resilience: a failure at step 700 still leaves verified local
  artifacts through step 690, imported under the same provenance rules.

### Verification law (both parts)

- The manifest is the transfer contract: member path → SHA256.
- Local import rehashes every member before any record is written; mismatch
  quarantines the bundle and raises a flag, never a silent skip.
- Synced mid-run artifacts are *observations* until the immutable segment and
  final report arrive and are verified; they cannot authorize a continuation
  on their own. The exact-parent rule for tranche renewal is unchanged.

## 4. Implementation sketch (if ratified)

- `runtime/trainer/cloud_bundle.py` (new): bundle writer + manifest +
  verify/extract, shared by both parts. Unit tests with planted corruption.
- `runtime/trainer/kaggle_adapter.py`: emit Part A archive at completion;
  wire the checkpoint-boundary sync hook behind an explicit recipe flag
  (`sync_mid_run: true`), default off.
- `configs/kaggle/*.json`: recipes opt in explicitly; the flag and sync
  dataset slug are recorded in the job record.
- `scripts/axon_kaggle.py`: `fetch --bundle` path; `sync-status <job-id>`
  showing which step ranges are locally verified.
- `scripts/axon_training_watch.py`: poll sync dataset versions alongside the
  log stream.
- Secrets setup is a one-time manual step in the Kaggle UI by Jeff (create
  User Secret `AXON_KAGGLE_SYNC`; enable internet on the kernel). Engineers
  document it in `docs/KAGGLE_TRAINING_GUIDE.md` but never handle the token.
- Tests: bundle roundtrip + corruption rejection; sync-hook cadence with a
  mocked dataset API; fetch fallback for legacy jobs; no-secret no-op mode
  (packet must run correctly with sync disabled/absent).

## 5. Risks and mitigations

- **Wider trust boundary.** Internet + an API token inside the run
  environment is new. Mitigation: User Secrets are masked by Kaggle, the
  token is scoped to one account with no paid resources, sync code never logs
  env, and the packet remains fully functional without the secret.
- **Quota cost.** Uploads consume wall-clock inside the GPU session.
  Mitigation: daemon-thread upload at 30-step boundaries; measured overhead
  expected < 1% of session time; recorded in the job report.
- **Provider rate limits on dataset versioning.** Mitigation: 30-step
  cadence is ~12 versions per tranche; retry-at-next-boundary semantics;
  sync is advisory, never on the training critical path.
- **Pagination defect (kaggle-cli #1045) in legacy per-file fetch.**
  Mitigation: Part A bypasses per-file fetch entirely; if per-file fallback
  is ever needed, page explicitly and verify counts against the remote
  manifest.

## 6. Alternatives considered

- **Selective fetch only** (segment + final bundle + checkpoint, handful of
  files): cheap and worth keeping as a fallback, but gives no mid-run access
  and discards per-step Soul evidence from local custody.
- **Embed artifacts in the log stream (base64):** rejected — log size limits,
  fragile parsing, and it abuses the observability channel.
- **Third-party storage (S3/Drive/HTTP endpoint):** rejected — new external
  credentials and infrastructure for a problem the provider already solves;
  widens the credential surface for no gain over private datasets.

## 7. Open questions for reviewers

1. Should mid-run sync artifacts be import-eligible into local training State
   before the final segment arrives, or observation-only until closeout?
   (This proposal says observation-only.)
2. Is a per-job sync dataset the right granularity, or one `axon-sync`
   dataset versioned across all jobs?
3. Should the sync hook also carry the live `evaluated` Q/A transcript rows,
   so Jeff can watch free-running samples mid-run, not just metrics?

## 8. Relationship to tonight's work

This proposal was drafted while the step-361–720 outputs of job `48d0ee6a`
download over the slow per-file path — the concrete motivation. Nothing in
this proposal changes how that job's evidence is fetched, verified, or
judged; the step-720 verdict proceeds under existing law.
