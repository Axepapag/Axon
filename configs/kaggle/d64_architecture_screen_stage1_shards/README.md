# D64 architecture screen stage 1 shards

Run exactly one JSON recipe per Kaggle job. Each shard trains one declared
candidate for the same 32-step, single-seed diagnostic and writes checkpoints
at steps 16 and 32. This prevents one candidate's model and optimizer lineage
from consuming disk needed by another candidate.

Prepare and launch one shard:

```powershell
python scripts/axon_kaggle.py prepare configs/kaggle/d64_architecture_screen_stage1_shards/d64-l10-h2-f131072.json
python scripts/axon_kaggle.py launch <prepared_job_id> --yes
python scripts/axon_kaggle.py monitor <prepared_job_id>
```

After Kaggle reports completion:

```powershell
python scripts/axon_kaggle.py fetch <prepared_job_id>
```

Only a completed, locally fetched, hash-verified output bundle is admissible
screen evidence. Compare all 16 shards only after every required candidate has
a complete bundle. This opening is not a promotion or serving gate.

The former monolithic recipe was removed after job `b4b9a380...` accumulated
nine complete candidate lineages, filled Kaggle's notebook disk during
candidate 10, and could not publish a consolidated manifest.
