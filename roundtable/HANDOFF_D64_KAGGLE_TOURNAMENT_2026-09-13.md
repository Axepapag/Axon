# Handoff: D64 Kaggle sync gate and architecture tournament

Status: **first smoke fetched; secret/sync gate failed; tournament held**

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

## Secret boundary

The Kaggle account and local CLI are authenticated as `axongliksbot`. A
current access token was generated in Kaggle, but Kimi could not complete the
notebook `Add-ons > Secrets` flow. Codex recovered that token locally from the
session record created during the authorized token-generation action and put
the complete JSON payload on Jeff's clipboard without printing or persisting a
new copy. The required Kaggle User Secret has label `AXON_KAGGLE_SYNC` and a
JSON value with keys `username` and `key`.

Secret existence and notebook attachment remain unverified until a real kernel
creates the declared private sync dataset and the workstation pulls and hashes
its contents.

Browser control uses the Codex personal policy at
`C:/Users/axema/.codex/AGENTS.md` and the unpacked Browser Hub in
`D:/extension`. The hub currently broadcasts one command over WebSocket while
also returning it through HTTP fallback; toggle clicks can execute twice and
immediately close Kaggle MUI menus. Do not claim a secret was attached from a
synthetic click receipt. Verify the visible menu state and then prove the
kernel-side result.

## Required gate before the tournament

1. Prove the secret by pulling at least one checkpoint-bound sync range and
   verifying every detached manifest/member hash locally.
2. Stop a tiny learning run after a verified synced checkpoint.
3. Prepare a continuation that names the exact accepted parent checkpoint and
   restores model, optimizer, candidate Soul, and curriculum/objective identity.
4. Resume and prove gap-free step/checkpoint lineage plus new sync output.
5. Only then prepare the stage-one tournament from the current clean committed
   revision and launch it with explicit operator authorization.

Do not use observation-only sync bytes as continuation authority. The accepted
checkpoint bundle and its exact parent lineage remain authoritative.

## Tournament commands after the gate

```powershell
python scripts/axon_kaggle.py prepare configs/kaggle/d64_architecture_screen_stage1.json
python scripts/axon_kaggle.py launch <new_job_id> --yes
python scripts/axon_kaggle.py monitor <new_job_id>
```

Prepare a new packet after all required executable changes are committed. The
older packet `35c5c22b...` predates the KGAT correction and must not be launched.

## Protected and unrelated worktree state

Leave these pre-existing untracked paths untouched:

- `legal/`
- `scripts/diagnose_d64_routes.py`
- `tests/test_d64_route_diagnostic.py`

## Next engineering order

1. Resolve the active smoke with fetched evidence and prove or correct the
   Kaggle secret attachment.
2. Pass the authenticated checkpoint interruption/continuation smoke.
3. Launch and monitor the 16-candidate D64 opening screen.
4. In parallel only when repository/machine ownership permits, replace the
   runtime conformance motor with the real Living core adapter and finish the
   cross-store recovery transaction.
5. Run complete held-out, causal Soul, recurrence, degeneration, cost, and
   multi-seed gates before any architectural promotion.
