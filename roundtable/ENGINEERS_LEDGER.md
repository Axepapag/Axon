# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-14T16:43:46+00:00
Current through event:
`evt-20260914T164346Z-codex-tournament-pause-diagnosis`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`

Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

Identity stamp: Codex / GPT-6 / 2026-09-14

## Current mission and honest status

Jeff removed optional mid-run Kaggle checkpoint sync from the bounded stage-one
screen. `AXON_KAGGLE_SYNC` is not core or rail security and no longer blocks
this tournament. The accepted recovery boundary is Kaggle's completed private,
locally fetched, hash-verified output bundle; an interrupted 32-step candidate
may be rerun from its immutable recipe and seed.

The monolithic 16-candidate job `b4b9a380...` is terminal with provider status
`ERROR`. Candidates 1 through 9 completed 32 optimizer steps and wrote
content-address-correct reports. Candidate 10 reached step 31, then accumulated
model/optimizer checkpoints filled Kaggle's notebook disk. Candidate 11 and
papermill output saving failed with `OSError: [Errno 28] No space left on
device`. No detached archive/manifest was published, so those reports remain
diagnostic evidence rather than accepted tournament results.

Commit `0c78b53` replaces the monolithic recipe with 16 single-candidate shard
recipes and makes the parent prefer durable progress receipts over stdout that
contains progress prefixes. The largest shard, `d64-l10-h2-f131072`, completed
on a Tesla T4 as job `ff09dd152c2cb657b696f509725092e84efebbb2d6915e0a15503d67a492a1d2`.
Its locally fetched 3.49 GiB detached archive passed the bundle verifier, and
its report ID independently recomputed exactly.

The architecture tournament is now paused. Ten completed 32-step openings,
from 1.22M through 169.38M parameters, all reduced loss and reached
teacher-forced payload-content accuracy 1.0, but all retained copy-gate
accuracy 0.0 and exact free-running payload transport 0.0. The common failure
surface means the current opening diagnoses the motor objective more strongly
than architecture capacity. No candidate is promoted.

Operational handoff:
`roundtable/HANDOFF_D64_KAGGLE_TOURNAMENT_2026-09-13.md`.

## Completed KGAT and smoke evidence

- Commit `3492182` accepts current `KGAT_` access tokens from either the
  environment or `AXON_KAGGLE_SYNC` and mirrors them to
  `KAGGLE_API_TOKEN`; legacy key handling remains supported.
- Codex independently reviewed the patch. The cloud bundle/job suites pass
  51/51.
- Private Kaggle smoke job
  `0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425`
  ran 60 real optimizer steps on a Tesla T4 from Git `3492182` and returned 0.
- The fetched detached output manifest covers 1,322 members and 30,487,504
  bytes. Extended-length-path rehashing found zero missing, size-mismatched, or
  hash-mismatched members. Archive SHA256:
  `ef5dac88c511b8387ca0936a7700577befbffa00f771b793db3ec2b66955e078`.
- Final held-out mean loss was `0.5992539127667745`, but the task gate failed,
  nonzero exact output was false, free-running payload exact rate was 0.0, and
  typed emission exact rate was 1/3. This is learning-signal evidence only.
- Final accepted checkpoint:
  `700c2e74f74ca27a9e6799970a5cc4259379a4ed3ebfb16007a2d66897c37d33`.
- The only sync receipt is `disabled` with reason
  `sync credentials unavailable: SyncCredentialsMissing`.

## D64 architecture tournament

The implemented tournament path uses the real `LivingReasoningCoreD64`,
`living_episode_objective`, typed categorical heads, causal free-running
receipt-aware transport, complete-field coverage, and recurrent private Soul
unroll.

The declared search space contains 48 legal geometries:

- layers: 2, 5, 10;
- heads: 1, 2, 4, 8;
- FFN widths: 4,096, 16,384, 65,536, 131,072.

Ten standard heads is illegal for D64 because 64 is not divisible by ten. The
balanced opening screens 16 candidates and later stages retain at most eight,
three, and one across increasing budgets and multiple seeds. Candidate sizes
range from about 1.22M to 169.38M parameters.

Campaign ID:
`c4ebb873afb8a33c6c3f4e4e7cd3d7c23951f33a35ce6ef7470609fb4dcdbeab`

Tournament ID:
`cea217a0025fc9fa2e42a0d0c83b50eb77bb923e72114209684f52b0bb83394e`

Recipes: `configs/kaggle/d64_architecture_screen_stage1_shards/`

Proposal:
`roundtable/proposals/CODEX_D64_ARCHITECTURE_TOURNAMENT_2026-09-12.md`

The opening evaluation is deliberately incomplete. It can screen learning
signal and runtime cost; it cannot promote or serve a winner. The older
prepared packet `35c5c22b...` predates the KGAT correction and must not launch.

The first no-sync launch, job `c4e38226...`, ran candidate
`d64-l2-h1-f4096` for all 32 optimizer steps on a T4 but then stopped at the
parent boundary because Kaggle supplied empty captured stdout. The candidate's
durable report was intact: held-out mean loss moved from `7.38446` to `3.44517`
and free-running payload exact rate stayed `0.0`. Commit `9d177bd` makes the
parent recover that report through its content-addressed progress receipt and
reject path escapes, hash mismatches, or receipt/report disagreement.

The second monolithic launch proved all nine completed segment reports and
their progress receipts were internally hash-consistent, but stdout progress
prefixes still caused `ReportContractError`. It then proved the shared-job disk
shape was unsound. Stage one now uses one candidate and two checkpoints per
private Kaggle job. The removed monolithic config must not be recreated.

## Runtime Trainer truth boundary

The standalone tournament and smoke train a real Living core, but the newer
Heart-owned runtime `TrainingSession` still uses the approximately 114K
`CompleteField64D` conformance motor. Do not conflate these paths.

Completed runtime foundations include authenticated worker evidence, exact
incoming Soul binding, free-running target-blind proposal evidence, recursive
successor attendance, rolling attempt workspaces, recovery of accepted and
rejected work, landmarks, and a separate homework-completion verdict.

Two material runtime blockers remain:

1. Replace the conformance motor with a sealed `LivingReasoningCoreD64`
   worker/gate adapter carrying exact parameter, optimizer, rail, Soul,
   objective, emission, and lineage identities.
2. Finish one recoverable transaction across attempt, candidate Soul,
   workspace, canonical field, and landmark writes.

Only after those pass may Axon claim that the real local Heart is supervising a
remote Living core inside the canonical runtime training loop.

## Binding decisions and invariants

- Cortex and reasoning rails are separate organs.
- Canonical exact text is D16. D64 rails pack four exact D16 cells per row with
  receipts; words and paragraphs remain sequences, not single opaque vectors.
- Larger core lanes may lift the D16 substrate but never replace canonical text.
- Soul is private recurrent experiential state. Durable learning requires
  retained Soul state and/or parameter/optimizer updates whose later effects
  are tested causally.
- A valid optimizer step and a completed homework assignment are independent.
- More recurrent ticks reuse a learned transition; more physical layers add
  distinct stored transformations. The tournament must measure both rather
  than assume infinite ticks repair insufficient transition capacity.
- Mid-run sync is observation-only. It never becomes continuation authority.
- Checkpoint continuation requires the exact accepted parent and full model,
  optimizer, Soul, curriculum, and objective identity.
- Every long or expensive run waits for a real smoke gate and current mission
  freshness.
- No partial screen, declining loss, or teacher-forced score authorizes serving
  or promotion.

## Current blockers and risks

- **BLOCKING:** the runtime `TrainingSession` still lacks the real Living-core
  adapter and complete cross-store recovery transaction.
- **BLOCKING:** breadth-first architecture screening is paused until one
  reference core crosses the exact free-running motor gate. The current
  receipt-continuation objective emphasizes source position (4.0) while giving
  the copy/generate route only 0.25 weight; the route loss remained effectively
  flat across the completed largest shard.
- **ADVISORY:** D:/extension source is restored and the Browser Hub recovered
  to two WebSocket clients at the final machine sweep. Duplicate WebSocket/HTTP
  command delivery can still double-toggle Kaggle MUI menus.
- **ADVISORY:** the fetched smoke learned its narrow surface but produced no
  exact free-running payload. Architecture and objective selection remain open.
- Pre-existing day-zero hygiene failure: `attempt_workspace.py` is absent from
  the trainer active-surface allow-list.
- `legal/`, `scripts/diagnose_d64_routes.py`, and
  `tests/test_d64_route_diagnostic.py` are unrelated untracked work and remain
  untouched.

## Next actions

1. Diagnose one small reference core on a tiny exact-copy assignment, including
   copy-gate logit, gradient, route choice, EOS, and free-running payload traces.
2. Align the staged objective with the advancement gate and renew assignments
   until pass, bounded supervisory stop, or a localized failure.
3. Resume the breadth tournament only after the reference core establishes
   nonzero exact free-running output; rank time/steps-to-gate, retention, causal
   Soul use, recurrence, degeneration, and cost across multiple seeds.
4. Implement and gate the real Living-core runtime adapter plus the cross-store
   transaction before making a canonical Heart-owned remote-training claim.

## Useful commands

```powershell
python scripts/axon_kaggle.py doctor
python scripts/axon_kaggle.py status ff09dd152c2cb657b696f509725092e84efebbb2d6915e0a15503d67a492a1d2
python scripts/axon_kaggle.py monitor ff09dd152c2cb657b696f509725092e84efebbb2d6915e0a15503d67a492a1d2
python scripts/axon_kaggle.py fetch ff09dd152c2cb657b696f509725092e84efebbb2d6915e0a15503d67a492a1d2
python scripts/axon_kaggle.py jobs
```

## Continuity health

- Canonical ledger: 232 valid unique events through
  `evt-20260914T164346Z-codex-tournament-pause-diagnosis`.
- Latest architecture-screen commit: `0c78b53` (single-candidate shards and
  receipt-first report loading).
- Kaggle account: `axongliksbot`; no newly launched tournament kernel remains
  active. The completed largest-shard proof is preserved locally.
- No production Heart or serving service was started or stopped.
- The Windows clipboard was cleared of the earlier secret payload.
- No billed API spend was incurred. Kaggle reported 27.46 of 30 GPU hours after
  the completed shard proof and before the brief, cancelled launches.
