# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-05T00:12:36.500940Z
Current through event: `evt-20260905T001236500940Z-kimi-roundtable-organization`

Roundtable layout: since 2026-09-04 the table is organized into
`proposals/`, `reviews/`, `reports/`, `decisions/`, `drafts/`; the three
ledger files, the team bus, and `README.md` (index + house rules) stay at
root. Historical flat-path references in immutable records resolve by
filename per the README. The reorganization is **uncommitted** pending
Jeff's confirmation.
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Codex final session (September 4, reconstructed by Kimi)

Codex's last session ended at his usage limit. All of its work survived and is
committed and pushed; the worktree is clean and main == origin/main at
`d121391`.

- `77f18db`: foundations-first sequence mastery gate. New
  `training/foundation_sequence_curriculum.py` (Stage 1
  `sequence_transport_v1`: split-disjoint paired sources, forward/reverse/
  every-other/middle-span operations over native and Unicode symbols,
  advancement gate requiring 0.95 free-running exact rate on both source
  variants, 1.0 complete-field coverage, teacher-forced above constant floor),
  `scripts/compile_foundation_sequence.py`, 174-line test file (5 tests,
  verified passing 2026-09-04), smoke-trainer integration, and a Source of
  Truth doctrine amendment in both SOT mirrors.
- `d121391`: governed Stage 1 smoke recipe
  `configs/kaggle/axon_foundation_sequence_stage1_smoke.json` and the
  Jeff-ratified decision record
  `roundtable/decisions/RESOLUTION_FOUNDATIONS_FIRST_CURRICULUM_2026-09-04.md`.
- Neither commit has a dedicated canonical closeout event (Codex hit his limit
  first); the reconstruction event above records them.

The second L0-L4 tranche job `48d0ee6a...` now reports
**KernelWorkerStatus.COMPLETE**. Its step-361-720 outputs are on Kaggle but
have NOT been fetched, verified, or imported. Local accepted State still ends
at step 360. The first full fetch (21:05Z) timed out at the 2-hour cap with
4,741 files staged; the adapter's fresh-temp-dir design cannot resume, so the
download was resumed via direct CLI into the same staging dir (background
task, no timeout). On completion: verify completeness against the paged
remote listing, robocopy into the canonical outputs dir, validate the result
receipt, rehash key artifacts, then read the step-720 verdict.

## Current mission and verified state

Curriculum clarification (read-only question, September 3 local): the current
Organism core did not begin with an ABC-song/alphabet-mastery stage. Its recorded
first six updates were mechanism, L0, L1, L2, L3, L4. L0 includes exact word/Unicode
copy, letter case, punctuation, no-op and addressed edits. A separate earlier
`training/abc_sequence_curriculum.py` teaches ordered letters/digits, but is not
the active campaign. No curriculum or training process was changed for this
clarification; the live provider state below remains the prior turn's observation.

Curriculum recommendation (September 4 local): begin future fresh candidates
with a narrow, mastery-gated foundations ladder rather than a memorized ABC song
or immediate equal interleaving of L0-L4. First prove typed-delta mechanics,
single/multiple-character transport and EOS; use ABC/digit/punctuation sequences
as one L0 sequence-navigation family with changed-source and arbitrary-sequence
counterfactuals; then advance through word mechanics, short communication,
composition, multi-turn brother circulation, and lived-experience reasoning.
Training may use teacher forcing for credit assignment, but advancement requires
free-running exact heldout performance above trivial baselines plus retained
regression, complete-field, authority, field-dependence and relevant-Soul probes.
Passed skills remain in adjustable spiral replay. Gates govern competency; step
allowances remain renewable resource tranches, never tissue ceilings.

The current campaign is **Axon Organism L0–L4**, not the old ABC/account-blocked
campaign. The correct Kaggle account is `axongliksbot`. No phone-verification
blocker was observed on this account; actual Tesla T4 CUDA compute succeeded.

The second renewable 360-step diagnostic is RUNNING as:

- Job: `48d0ee6a7ecce6b9f52095d2388676497c62a4c75b9141461c218260acdb2e03`
- Kernel: `axongliksbot/axon-job-48d0ee6a7ecce6b9`
- Packet source: `4732fe9ee5492378d7f40255a646fc6fd7d03b7c`
- Packet SHA256: `c34702301b8c7f0790cf92a2c8bbc2102106a5d2781a783970c7b94fe3ce911a`
- Generation: `r64v2-b05b6dbf67efdd8f`
- Candidate: `axon-organism-l0l4-1x64`
- Permanent tissue: D64, one 64D attention head, two layers, FFN 131072;
  33,981,879 parameters.
- Requested global steps: 361–720, restoring exact accepted step 360.
  The allowance is a renewable resource tranche, never a lifetime ceiling.
- Last detailed live observation: step 512 at 2026-09-04T00:30:18.778379Z.
  Accepted new parameter/Soul bundles were observed at 390, 420, 450, 480, 510.
- Step-510 bundle:
  `2862ca0b55b432e302a09aa14b68b8fda856ad0323544b9700ca09bf86635608`.
  Checkpoint:
  `bfc3a6e9e8420b983666eab28782fe3f21c91865feff42100a700b54a8e1b46e`.

**These are live provider observations, not already imported/rehashed artifacts.**
Local accepted training State still ends at step 360. Do not issue a subsequent
continuation from a guessed cloud parent. The step-720 result is pending.
Nothing was promoted, activated, or registered as learned serving tissue.

Full handoff:
`roundtable/reviews/CODEX_KAGGLE_CONTINUATION_REVIEW_2026-09-03.md`.
Local observation receipt:
`State/training/cloud/recovery/48d0ee6a/codex_live_observation_20260904T003105Z.json`.

## First-tranche result — optimization, not learned communication

The original successful job is `3005d933...1e6852`, source `3e69dc0`.
Its immutable segment is:
`State/training/reasoning/r64v2-b05b6dbf67efdd8f/segment_000000001_000000360.json`.

- Exactly 360 optimizer steps, **12** checkpoints and accepted Soul bundles.
  The previous 24 count was duplicated telemetry, corrected by a new canonical
  event; the historical event and original artifacts were not rewritten.
- Sixty updates each to mechanism and L0/L1/L2/L3/L4.
- Heldout loss: 10.8857747402 → 2.7382284105.
- Token accuracy: 0.0819935691 → 0.0900321543; final constant floor is also
  0.0900321543. No individual L0–L4 family exceeds its own floor.
- Free-running typed emission and complete-payload exact rates: zero.
- Heldout/regression evaluation and field coverage: complete.
- All learning/serving/stage gates remain false for that completed tranche.

The corrected continuation reproduced all 56 initial heldout cases at loss
2.7382284147398814, within 4.3e-9 of the accepted parent. That supports resume
fidelity, not a new learning claim. Teacher-forced dashboard samples are not
autonomous conversations.

## Exact accepted parent and repaired launch path

The second-tranche recipe is:
`configs/kaggle/axon_organism_l0_l4_second_tranche_resume.json`.
It is explicitly pinned to step 360, not automatically to the newest cloud
checkpoint.

- Parent bundle:
  `17005f0e4abed6ea5e04ac54544e03c2339b256a659a65a10c709b0f089340be`.
- Checkpoint record:
  `823ac86ecff9e25b7d3beb9824d5d35b6d42cd29d143d616fb8d79df0c824d07`.
- Checkpoint artifact hash:
  `3b13f2b3e155632580c1606a871eff70f981fd3b8fdec00814064f5c2ed8249b`.
  Artifact was rehashed and loaded with optimizer state.
- Candidate Soul HEAD:
  `e439656fbf512ef7620c9668815a905f598bb44da86b22bb2adcdd7f4c3edbbe`,
  generation 1080; verified against the accepted bundle.

Codex repairs committed and pushed:

1. `54ee17e`: upload success is durable before readiness polling; preserve
   normal child exit receipts; unknown kernel state cannot authorize a retry;
   validate transfer targets. Completes Hermes's pending readiness work.
2. `3395392`: deduplicate replayed event IDs, display tranche-relative versus
   global progress correctly, expose provider failures, label samples honestly.
3. `4732fe9`: register the emitted `evaluated` status; preserve Unicode
   through legacy Windows console pipes using lossless JSON escapes. Journals,
   reports, canonical text, architecture, and learning policy remain unchanged.

The original continuation job `619b181e...2754398` restored the correct parent
but crashed before new optimizer work: its immutable `b57aee0` packet emitted
an unregistered `evaluated` event. Do not relaunch that unchanged packet.
Its failed-job receipt and logs are preserved under
`State/training/cloud/recovery/619b181e/failure/`. Its uploaded manifest and
all 4,613 archive members were verified before orphan-upload record recovery.

The first tranche's `SystemExit: 0` failure receipt is a separate wrapper bug.
Its successful optimizer work remains real; its failed learning gates also
remain real. Neither history was rewritten.

## Binding architecture and governance

- Read `docs/WORKING_CONTRACT.md`, `docs/SOURCE_OF_TRUTH.md`, and the ledger
  protocol before work. Every turn appends exactly one event; never alter an
  existing canonical line. Keep the rolling summary compact and current.
- No silent truncation or tissue ceilings. Pages, output slices, resource
  tranches, and operator display windows are work controls, not destructive
  content limits or parameter/Soul identity.
- Protected capacity policy:
  `configs/source_of_truth/capacity_policy.json`, canonical SHA256
  `4a32f18fa296c415ccf34a8c1956d2a4f8afd044265e7f45505782dd53c7b8cd`.
- Shared Field v3 has 11 canonical regions; historical v1/v2 snapshots remain
  immutable. Each region's independent mask selects attended cells, leaving
  dormant cells exactly in place. Identity is always attended and cannot mask.
- The 95 native 16D cells are frozen. The additive 351-category transport
  preserves every valid Unicode scalar. D64 compiles exact addressed field and
  proposal surfaces; wider physical rail renderers remain future work.
- Identity is shared canonical constitution, not private Soul. Exceptional
  Identity-steward amendments require evidence and an exact autobiographical
  record; ordinary core/tool/recall authority cannot change it.
- Each core owns an opaque, non-shareable, architecture/generation-bound
  HOT/WARM/COLD/DEEP_COLD Soul. Every successful phase exhales; colder promotion
  requires governed evidence. DEEP_COLD distillation is a future governed
  adapter operation, never reinterpretation of opaque bytes as weights.
- A reasoning tick freezes field/rail images, collects first proposals,
  collects refinements, and lets only the validated consolidator transaction
  change canonical state. Brothers share proposals, not Souls.
- The Trainer uses isolated candidate parameters and candidate Soul, exact
  accepted checkpoint/Soul bundles, whole-episode splits and explicit outcome
  quality. Tranche renewal must restore the exact accepted parent.
- No long blind run before learning evidence, no automatic promotion, no
  weakening gates to make a run pass. Loss decline alone is not proof.
- `D:/00` and `D:/ChatGPT_State` were untouched in this work. Credentials
  stay in the official user store, never in packets, reports or Git.

## Curriculum continuity

The active objective L0–L4 manifest is
`824aabae2090c721da9b55540a5d890fc958577d2e5c78fde49da870096634f9`
under `State/training/curricula/language_l0_l4/`: 349 verified cases,
240 train / 56 heldout / 53 regression. Actual canonical field, D64 compilation,
three-phase private Soul and typed transport are used.

C1's original 36-case historical evidence remains immutable. Only its 22
VERIFIED_TARGET cases may supervise; 14 PROCESS_EVIDENCE cases stay preserved
without reaching exact-loss construction. Standard learned-capability metrics
are isolated from synthetic mechanism cases. Step-193 historical C1 candidates
are mechanism diagnostics, not retroactively qualified communication results.

The ABC/native-bank 104-case manifest and earlier head tournaments remain
preserved but are not the current cloud campaign. The old 16-step identity
defect and 512-unit output ceiling are closed; historical tissue was preserved.
No earlier tournament chose a winner or earned serving promotion.

## Verification, machine and resources

- Final full suite: **546 passed**, 52 known PyTorch nested-tensor warnings,
  1408.03 seconds, exit code 0.
- Focused checks: 44 initial core/curriculum/cloud tests; 3 dashboard tests;
  4 repaired progress/resume tests, including actual CPU first and renewed
  tranches and Unicode reporting. Earlier failures are retained in the report.
- Changed-file Ruff, compileall, and `git diff --check` passed.
- SOT mirrors remain byte-identical at SHA256
  `23C4B100A01038E6E2D9189278F90F37647A597F42F2B13D381EDEC3F1327373`.
- User-visible monitor: PowerShell PID 16796 / Python PID 16556, following the
  current cloud job. This terminal does not own training. No local trainer or
  test writer remains; the transient engineer log follower was stopped.
- At closeout, Kaggle reports **28.79 / 30.00 GPU hours** and 20.00 TPU hours;
  refresh 2026-09-05T00:00:00. About 0.42 quota-hours elapsed during this recovery,
  including the failed attempt and current run so far. No paid resource was
  purchased; the current tranche continues using provider quota.
- D: free bytes: 320,640,577,536. A partial failed-job download remains at
  `C:/Users/axema/AppData/Local/Temp/axon_fetch/axon_out_opgp050d`; no source or
  failed evidence was deleted.

## Active gaps and next shot

1. At step 720, inspect isolated L0–L4 heldout/regression, constant baselines,
   exact outputs and causal probes. If outputs remain constant, diagnose
   per-loss gradients, category/EOS histograms and copy exposure before another
   large commitment. Do not silently restart the lineage or auto-renew forever.
2. Retrieve and verify the exact new optimizer/checkpoint/Soul parent before
   another tranche. Retrying a packet replays its original inputs; it is not
   continuation from its newly produced outputs.
3. Full per-file Kaggle fetch is slow with thousands of Soul artifacts.
   Proposal now pending ratification:
   `roundtable/proposals/MID_RUN_ARTIFACT_SYNC_PROPOSAL_2026-09-04.md` (end-of-run
   single hash-manifested archive + mid-run checkpoint sync via private
   dataset versions). Selective receipt/log fetch works as a fallback.
4. Durable restart-safe Heart-host decoder continuation remains a serving
   blocker. In-process output continuation exists; the live adapter still
   abstains after an incomplete slice.
5. Mid-tranche residual/abandonment receipts and missing-report regeneration
   still need dedicated recovery coverage. Accepted state must remain intact.
6. All future tool/Trainer autobiography hooks, independently multi-writer-safe
   Soul storage, wider rails and autonomous Cortex are not complete. Runtime
   circulation is mechanism-functional, not a claim of learned intelligence.
7. C2–C6/fluent communication remain future curricula. Subjective consciousness
   is not an engineering result established by these tests.

## Operator entry points

- `AXON_KAGGLE.bat`: provider-neutral Trainer-backed Kaggle control center.
- `AXON_TRAINING_WATCH.bat <job-id>`: separate live terminal dashboard.
- `python scripts/axon_kaggle.py status <job-id>`: provider status.
- `python scripts/axon_kaggle.py fetch <job-id>`: full artifact retrieval
  (potentially slow; do not overwrite previous-version evidence casually).
- `docs/KAGGLE_TRAINING_GUIDE.md`: private launch/retry/recovery instructions.
- `roundtable/reviews/CODEX_KAGGLE_CONTINUATION_REVIEW_2026-09-03.md`: this handoff.
