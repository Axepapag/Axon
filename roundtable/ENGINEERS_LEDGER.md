# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-13T08:01:00+00:00
Current through event:
`evt-20260913T074500Z-kimi-sync-smoke-first-run`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`

Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

Identity stamp: Codex / GPT-6 / 2026-09-13

## Current mission and honest status

The immediate mission is to prove authenticated, interruption-safe Kaggle
transport and then launch the real D64 architecture tournament. The first
private KGAT-patched smoke is complete and fetched, but it failed the sync gate:
the kernel emitted `SyncCredentialsMissing`, created no sync dataset, and
continued training under the intended fail-open observation policy. The
tournament remains held.

Jeff's already-authorized Kaggle token was recovered from the local Codex
session record and the complete `AXON_KAGGLE_SYNC` JSON value was placed on the
Windows clipboard without printing or persisting a new copy. Jeff must manually
create/attach/enable that User Secret in the correct Kaggle notebook. A relaunch of the same immutable smoke
must then produce a locally hash-verified checkpoint sync, survive a real
interruption, and continue from its exact accepted parent before the tournament
may launch.

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

Recipe: `configs/kaggle/d64_architecture_screen_stage1.json`

Proposal:
`roundtable/proposals/CODEX_D64_ARCHITECTURE_TOURNAMENT_2026-09-12.md`

The opening evaluation is deliberately incomplete. It can screen learning
signal and runtime cost; it cannot promote or serve a winner. The older
prepared packet `35c5c22b...` predates the KGAT correction and must not launch.

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

- **BLOCKING:** `AXON_KAGGLE_SYNC` is not attached/proven in the Kaggle
  notebook. The first KGAT-patched smoke recorded `SyncCredentialsMissing`.
- **BLOCKING:** authenticated checkpoint interruption/replay has not passed.
- **BLOCKING:** the runtime `TrainingSession` still lacks the real Living-core
  adapter and complete cross-store recovery transaction.
- **ADVISORY:** D:/extension source is restored after a deduplication
  experiment, but its Browser Hub feed may require Jeff to click **Turn Feed
  ON**. Duplicate WebSocket/HTTP command delivery can double-toggle Kaggle MUI
  menus.
- **ADVISORY:** the fetched smoke learned its narrow surface but produced no
  exact free-running payload. Architecture and objective selection remain open.
- Pre-existing day-zero hygiene failure: `attempt_workspace.py` is absent from
  the trainer active-surface allow-list.
- `legal/`, `scripts/diagnose_d64_routes.py`, and
  `tests/test_d64_route_diagnostic.py` are unrelated untracked work and remain
  untouched.

## Next actions

1. In the `axongliksbot` notebook, Jeff creates/enables Kaggle User Secret
   `AXON_KAGGLE_SYNC` using the payload currently on the clipboard.
2. Relaunch job `0cd589c4...` from the same immutable packet and pull/hash its
   first released checkpoint range.
3. Interrupt the run after that verified release, then continue from the exact
   accepted parent and prove gap-free model/optimizer/Soul lineage.
4. Prepare a fresh packet from the current clean commit and launch the
   16-candidate D64 opening screen.
5. Monitor complete candidate evidence; stop collapsed or unhealthy runs and
   preserve their receipts.
6. Implement and gate the real Living-core runtime adapter plus the cross-store
   transaction before making a canonical Heart-owned remote-training claim.

## Useful commands

```powershell
python scripts/axon_kaggle.py doctor
python scripts/axon_kaggle.py status 0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425
python scripts/axon_kaggle.py sync-status 0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425
python scripts/axon_kaggle.py jobs
python scripts/axon_kaggle.py prepare configs/kaggle/d64_architecture_screen_stage1.json
```

## Continuity health

- Canonical ledger: 229 valid unique events through
  `evt-20260913T074500Z-kimi-sync-smoke-first-run`.
- Latest code commit accepted: `3492182` (KGAT compatibility).
- Latest documentation commit: `8c994c8` (D64 Kaggle handoff).
- Kaggle account: `axongliksbot`; first private smoke is complete and fetched.
- No production Heart or serving service was started or stopped.
- Approximate observed T4 consumption for the completed smoke: 0.56 GPU hours.
