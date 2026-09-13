# Handoff: D64 Kaggle architecture tournament

Status: **no-sync tournament active; first candidate boundary passed**

Identity stamp: Codex / GPT-6 / 2026-09-13

## Read first

1. `docs/WORKING_CONTRACT.md`
2. `docs/SOURCE_OF_TRUTH.md`
3. `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
4. `roundtable/ENGINEERS_LEDGER.md`
5. The tail of `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`

The canonical ledger is authoritative. This handoff is an operational map, not
a replacement for it.

## Current truth

The real D64 tournament path exists and uses `LivingReasoningCoreD64`,
`living_episode_objective`, typed categorical heads, free-running receipt-aware
transport, and candidate-private Soul unroll. The balanced opening screen has
16 candidates selected from 48 legal D64 geometries:

- layers: 2, 5, 10;
- attention heads: 1, 2, 4, 8 (10 is illegal because 64 is not divisible by 10);
- FFN widths: 4,096, 16,384, 65,536, 131,072.

Campaign ID:
`c4ebb873afb8a33c6c3f4e4e7cd3d7c23951f33a35ce6ef7470609fb4dcdbeab`

Tournament ID:
`cea217a0025fc9fa2e42a0d0c83b50eb77bb923e72114209684f52b0bb83394e`

The opening recipe is
`configs/kaggle/d64_architecture_screen_stage1.json`. It is deliberately an
incomplete screen. It may diagnose learning signal and runtime cost; it may not
promote or serve a winner.

The newer runtime `TrainingSession` is a separate path and still uses the
`CompleteField64D` conformance motor. Do not report the standalone tournament
as proof that the Heart-owned runtime trainer is already driving a real Living
core. Replacing that motor with a sealed `LivingReasoningCoreD64` worker/gate
adapter remains required. The cross-store transaction spanning attempt, Soul,
workspace, field, and landmark writes also remains required.

## Kimi's KGAT correction

Commit `349218294d00e15e2e59ca47ddc783ebb0adbe3c` changes
`KaggleDatasetUploader` so a current `KGAT_` access token supplied through the
`AXON_KAGGLE_SYNC` JSON secret is mirrored to `KAGGLE_API_TOKEN`, while legacy
username/key authentication stays supported. Codex independently reviewed the
diff and ran:

```powershell
python -m pytest tests/test_trainer_cloud_bundle.py tests/test_trainer_cloud_jobs.py -q
```

Result: 51 passed. No token value is stored in Axon, this handoff, the ledger,
or a cloud packet.

## Completed Kaggle smoke

Kimi prepared and launched a private 60-step Living-core smoke from commit
`3492182`:

- job ID:
  `0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425`
- kernel: `axongliksbot/axon-job-0cd589c4325ed48d`
- private input dataset: `axongliksbot/axon-job-0cd589c4325ed48d-input`
- intended private sync dataset: `axongliksbot/axon-job-0cd589c4-sync`
- model: 4 layers, 1 head, FFN 256;
- checkpoints: steps 15, 30, 45, 60;
- packet SHA256:
  `98a0b7d6490911e9fb6d1ba537aee3b688819fb85689962c723f65f78ac7b1f6`.

The job completed with return code 0 and its output bundle was fetched into
canonical local cloud-job State. The bundle manifest covers 1,322 members and
30,487,504 uncompressed bytes; Codex rehashed every member through Windows
extended-length paths with zero missing files, size mismatches, or hash
mismatches. Archive SHA256 is
`ef5dac88c511b8387ca0936a7700577befbffa00f771b793db3ec2b66955e078`.

The live and fetched receipts proved that the secret was unavailable. At the first checkpoint
the kernel emitted an `axon-mid-run-sync-receipt-v1` with status `disabled` and
reason `sync credentials unavailable: SyncCredentialsMissing`. The intended
sync dataset was never created and local `sync-status` had zero verified
members. Training correctly continued rather than corrupting the run, but this
smoke **failed the sync/recovery gate**.

The 60 optimizer steps did show a real learning signal: training loss began at
31.5408, and final held-out mean loss was 0.5993. The final narrow foundation
held-out probe reached 1.0 for its copy-gate and position measures. Those
teacher-forced/narrow results do not establish usable output: final free-running
payload transport exact rate was 0.0, typed emission exact rate was 0.3333, and
the sampled proposals were unterminated empty payloads. Treat the result as a
transport/training smoke only, never as conversation or reasoning competence.

Useful commands:

```powershell
python scripts/axon_kaggle.py doctor
python scripts/axon_kaggle.py status 0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425
python scripts/axon_kaggle.py sync-pull 0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425
python scripts/axon_kaggle.py sync-status 0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425
python scripts/axon_kaggle.py fetch 0cd589c4325ed48d19f8f5bd4838be3c3d864ed1accaad5efa4c784a26790425
```

The fetched `sync_receipts.jsonl`, runner events, segment report, checkpoint
sentinels, and bundle hashes are the evidence. A missing-secret receipt is a
failed sync gate even when training itself completes.

## Operator decision on mid-run sync

Jeff removed `AXON_KAGGLE_SYNC` from the stage-one launch path on 2026-09-13.
It was an optional checkpoint-upload credential, not core or rail security. The
feature remains available for future long runs, but this bounded screen does
not require a User Secret or interruption/replay smoke.

The accepted tradeoff is explicit: if Kaggle interrupts a candidate before the
job publishes its final output bundle, its current 32-step tranche may be lost
and must be rerun from the same immutable recipe and seed. Kaggle never writes
the local canonical body directly. Only completed, locally fetched,
hash-verified output bundles can be inspected or admitted by later governance.

## Tournament launch and repair

The first no-sync packet was prepared from commit `87b5a52` and launched as
job `c4e382262164f86be39da9dea5d53a9a8b1a98837419b7de7aea8aee0908871d`.
Its T4 probe passed and candidate `d64-l2-h1-f4096` completed all 32 optimizer
steps with checkpoints at steps 8, 16, 24, and 32. Held-out mean loss moved
from `7.38446044921875` to `3.4451667070388794`; exact free-running payload
transport remained `0.0`, so this is screening evidence only.

The candidate wrote a valid content-addressed report and progress receipt, but
Kaggle delivered an empty captured stdout stream to the tournament parent. The
parent attempted `json.loads("")` and stopped the job after candidate 1. The
fetched output bundle preserves the full failure and candidate evidence.

Commit `9d177bd` repairs that orchestration boundary. The parent now falls back
to the candidate's durable progress receipt, requires the report path to stay
inside governed State, verifies the report's content address, and checks that
the receipt names the same report. Three focused tests passed, including the
existing multi-candidate launcher integration test.

The active corrected job is:

- job ID: `b4b9a3802abf24d9ebf493fc15b717dcc4e067f2e76ab405806f3980353911c5`;
- Git revision: `9d177bddc0dfa4fd6dbf23ddab9e5ff8317976b6`;
- packet SHA256: `12429739096a604eba19ab1614d8a3ef392bdd10bde9e942a47e02097343be0a`;
- kernel: `axongliksbot/axon-job-b4b9a3802abf24d9`;
- private input dataset: `axongliksbot/axon-job-b4b9a3802abf24d9-input`.

Live Kaggle evidence proves the corrected parent accepted candidate 1 and
started candidate 2, `d64-l2-h2-f16384`. The job remains private and running.
Do not infer final ranking, competence, or promotion from this partial screen.

## Tournament commands

```powershell
python scripts/axon_kaggle.py prepare configs/kaggle/d64_architecture_screen_stage1.json
python scripts/axon_kaggle.py status b4b9a3802abf24d9ebf493fc15b717dcc4e067f2e76ab405806f3980353911c5
python scripts/axon_kaggle.py monitor b4b9a3802abf24d9ebf493fc15b717dcc4e067f2e76ab405806f3980353911c5
python scripts/axon_kaggle.py fetch b4b9a3802abf24d9ebf493fc15b717dcc4e067f2e76ab405806f3980353911c5
```

Prepare a new packet after all required executable changes are committed. The
older packet `35c5c22b...` predates the KGAT correction and must not be launched.

## Protected and unrelated worktree state

Leave these pre-existing untracked paths untouched:

- `legal/`
- `scripts/diagnose_d64_routes.py`
- `tests/test_d64_route_diagnostic.py`

## Next engineering order

1. Monitor the active 16-candidate D64 opening screen without restarting it.
2. After completion, fetch and independently inspect the hash-verified output
   bundle before accepting any candidate metrics.
3. In parallel only when repository/machine ownership permits, replace the
   runtime conformance motor with the real Living core adapter and finish the
   cross-store recovery transaction.
4. Run complete held-out, causal Soul, recurrence, degeneration, cost, and
   multi-seed gates before any architectural promotion.
