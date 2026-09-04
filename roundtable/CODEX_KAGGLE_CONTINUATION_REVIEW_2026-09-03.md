# Kaggle continuation review — 2026-09-03

Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-03

## Verdict

Hermes delivered real cloud training and useful operator tooling. Learned
communication remains unproven. Several launch/telemetry defects were real and
are corrected; they were not reasons to abandon the accepted core or Soul.

The current corrected continuation is private Kaggle job
`48d0ee6a7ecce6b9f52095d2388676497c62a4c75b9141461c218260acdb2e03`,
kernel `axongliksbot/axon-job-48d0ee6a7ecce6b9`, packet revision `4732fe9`.
It requests the next renewable 360 steps (global 361–720), not an indefinite
campaign or a changed lifetime capacity. It passed a real Tesla T4 CUDA compute
probe, restored step 360, and completed all 56 initial heldout cases with loss
2.7382284147398814 (within 4.3e-9 of the accepted parent's final evaluation).
At 00:17:06 UTC on September 4, its live progress journal reported the accepted
step-390 bundle `07d0e2886dc1a18b4f03752a873b07fdce7b1c877db09381214400559539115e`
and checkpoint `533a0acc4d52664aceaa61400498dc258b9ee7bf4d73ae4d615d49fb523d4a11`.
This is verified live-provider evidence; the new cloud artifacts are not yet
downloaded or independently rehashed. A separate visible terminal follows the
run and is not its owner. Closing the engineer harness cannot stop Kaggle.

Closeout observation: global step **512** at 00:30:18 UTC, with accepted
checkpoints/bundles at 390, 420, 450, 480, and 510. The step-510 bundle is
`2862ca0b55b432e302a09aa14b68b8fda856ad0323544b9700ca09bf86635608`;
checkpoint `bfc3a6e9e8420b983666eab28782fe3f21c91865feff42100a700b54a8e1b46e`.
The final step-720 learning result is not yet known. Local canonical training
imports still end at step 360; do not confuse live cloud observations with an
already imported, reverified continuation parent.

## Verified first-tranche result

Evidence: `State/training/reasoning/r64v2-b05b6dbf67efdd8f/segment_000000001_000000360.json`.

- Same 33,981,879-parameter D64, one 64D head, two layers, FFN 131072.
- Exactly 360 optimizer steps and **12** accepted parameter/Soul bundles.
  Hermes's dashboard counted 24 because Kaggle replayed identical event IDs.
  Replaying the downloaded provider log through the corrected dashboard
  independently reproduced 360 steps, 12 checkpoints, and 12 bundles.
- Sixty updates each to mechanism, L0, L1, L2, L3, and L4.
- Heldout loss: 10.8857747402 → 2.7382284105.
- Teacher-forced token accuracy: 0.0819935691 → 0.0900321543;
  the final strongest constant-category baseline is also 0.0900321543.
- No individual L0–L4 family exceeds its own constant baseline.
- Free-running typed and complete-payload exact rates are zero. Coverage is
  complete; reported heldout and regression evaluation are complete.
- No learning gate, promotion, activation, or communication claim is earned.

Falling loss is an optimization observation, not a demonstration of intelligence.
These results neither establish architectural success nor prove impossibility.

## Exact continuation evidence

- Generation: `r64v2-b05b6dbf67efdd8f`.
- Accepted step-360 parent bundle:
  `17005f0e4abed6ea5e04ac54544e03c2339b256a659a65a10c709b0f089340be`.
- Checkpoint record:
  `823ac86ecff9e25b7d3beb9824d5d35b6d42cd29d143d616fb8d79df0c824d07`.
- Checkpoint artifact SHA256:
  `3b13f2b3e155632580c1606a871eff70f981fd3b8fdec00814064f5c2ed8249b`.
  The artifact was hash-verified and loaded with optimizer state present.
- Private Soul HEAD:
  `e439656fbf512ef7620c9668815a905f598bb44da86b22bb2adcdd7f4c3edbbe`,
  generation 1080, verified against the accepted bundle.
- Recipe: `configs/kaggle/axon_organism_l0_l4_second_tranche_resume.json`.
- Corrected packet SHA256:
  `c34702301b8c7f0790cf92a2c8bbc2102106a5d2781a783970c7b94fe3ce911a`.

The training changes are telemetry support and lossless console serialization.
No architecture, learning policy, canonical Identity, live Soul, or curriculum
was changed. The current account is `axongliksbot`; prior account-entitlement
blockers are not the current condition.

## Repairs and failed attempt retained

1. Dataset success is recorded before readiness polling. Transient provider
   status failures no longer leave a successful upload looking uncreated.
   Unknown kernel states cannot authorize a resubmission.
2. The notebook no longer catches its own successful `SystemExit(0)` and
   overwrites a success receipt with failure. The first-tranche false-failure
   artifact remains unchanged; its segment and accepted bundles establish
   the real outcome. Commit `54ee17e`.
3. Dashboard replay deduplication, tranche-relative versus global step counts,
   visible provider failures, and honest teacher-forced sample labeling.
   These samples are not autonomous conversations. Commit `3395392`.
4. Hermes's producer emitted `evaluated`, but the recorder rejected that
   status. The original continuation packet (`b57aee0`, job `619b181e...`)
   restored step 360 and then failed before any new optimizer step.
5. Enabling progress in a real local train/resume test exposed a second bug:
   Unicode transcript JSON could crash Windows cp1252 stdout. The recorder now
   accepts completed evaluation events; console JSON is losslessly escaped,
   while durable UTF-8 data remains unchanged. Commit `4732fe9`.

The old upload was verified rather than duplicated: its downloaded remote
manifest was byte-identical to the local manifest, and all 4,613 packet members
and the archive hash were checked before recovering its mutable job record.
Failure receipts/logs are preserved under
`State/training/cloud/recovery/619b181e/failure/`. Its local job record explicitly
marks the failure and warns against relaunching the unchanged broken packet.

An unnecessary full-tree failure download was stopped after retaining its
partial files at `C:/Users/axema/AppData/Local/Temp/axon_fetch/axon_out_opgp050d`;
the relevant receipts and logs were then fetched selectively. Full per-file
Kaggle downloads are slow for thousands of Soul artifacts. Bundled, verified
result transfer is recommended future work, not implemented in this turn.

## Verification and next decision

The 44-test targeted core/curriculum/cloud suite passed before the additional
telemetry discoveries. Three dashboard tests passed. The repaired progress
tests plus a real CPU first-tranche/exact-resume integration passed (4 tests,
286.53 seconds), including initial/final evaluation events and exact parent,
plan, generation, and learning-policy continuity. The final full repository
suite passed: **546 tests**, 52 known PyTorch nested-tensor warnings, 1408.03
seconds (23m28s), exit code zero. Changed-file Ruff, compileall, and
`git diff --check` passed. The earlier full-suite attempt was stopped after a
failure while the Unicode repair was in progress; it is not counted as a pass.

At global step 720, fetch and inspect the immutable segment, accepted parent,
optimizer checkpoint and candidate Soul. Compare isolated L0–L4 metrics and
regression, not just combined training loss. If outputs remain constant, inspect
per-loss gradients, predicted-category/EOS distributions, source ablations and
copy exposure before another long commitment. Do not silently loosen gates,
change tissue, restart the lineage, or auto-renew a failing experiment forever.

The runtime's durable Heart-host decoder continuation remains an independent
serving prerequisite. No candidate from this work is serving.

Closeout machine/resource sweep: only the separate user-visible dashboard is
left from this work (PowerShell 16796 / Python 16556); no local Trainer/test
writer remains. Kaggle still reports the job RUNNING, with 28.79 GPU hours and
20.00 TPU hours remaining. Approximately 0.42 GPU-quota hours elapsed during
this recovery, including the failed attempt and current run so far; no paid
resource was purchased. D: has 320,640,577,536 free bytes. The cloud run continues
independently and its final metrics are still pending.
