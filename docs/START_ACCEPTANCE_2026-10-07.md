# Start acceptance and preservation

Jeff authorized Codex to preserve current source/runtime state, run bounded
learning/recovery/operator checks, and connect verified evidence to Start.

The learning smoke is preregistered in `tools/acceptance_memory.py`: D512/512
two-state GRU, Adam .003, at most 540 updates/600 seconds, balanced A/B/C targets,
one- or three-tick delays and two-character distractors. Train, validation and
test distractor contexts are disjoint. The same last distractor/query occurs
with each target, so a reader without the original fact has a 1/3 baseline.
Validation selects a checkpoint at >=90%; test must reach >=80% and exceed
zero-state and changed-observation conditions by >=30 percentage points. The
report names `swapped` and `irrelevant` replace the originally observed fact;
they do not transplant hidden states between exercises. No expected
text/length reaches inference. This is a three-symbol memory-mechanism check,
not full-curriculum mastery or a claim that the core is fluent. Broader curriculum
scores remain learning outcomes to measure as Jeff trains.

`tools/acceptance_recovery.py` verifies actual CPU and CUDA interrupted-training
weights, optimizer, losses, output, all RNG and no duplicated Heart commits.
`tools/preserve_state.py` takes a consistent SQLite snapshot and preserves frozen
exercises plus the disposable checkpoint with a per-file SHA256 manifest.
Restore validates every digest, database integrity, curriculum and Core loading.
Source is cloned independently from GitHub; private runtime ZIPs remain private
in Jeff's Drive. Provider download checks and clean restore evidence are recorded.

`lab/backend/acceptance.py` reads only the server-owned `acceptance.json` beside
the AppData registry. There is no HTTP endpoint to publish approval. It checks
the production source/substrate fingerprint, frozen dataset hash and immutable
proof-file hashes. Missing, failed, modified or stale evidence blocks Start.
Only a named one-epoch run may execute in the bounded-validation phase. Normal
execution requires the recorded browser lifecycle proof and accepted phase.
CPU/CUDA remain explicit selections, with no device fallback.

Each real checkpoint and its canonical Heart branch/journal dependencies are
copied and hash-checked under the configured private Drive backup folder. OS
writer leases are recreated on restore. Completed lifecycle operations also preserve SQLite through
its backup API. These later copies are labelled `local_copy_verified_cloud_pending`
until cloud upload is independently verified; Drive sync location alone does
not establish an offsite copy. The initial verified restore remains explicit.
`tools/restore_checkpoint.py` verifies the complete backup and creates a new
restored directory. Heart checkpoint branch roots are explicitly relocated with
new hashes and a provenance report; original backup bytes remain unchanged.
Importing that restored organism into the Lab registry is separate from this
offline recovery tool.

UI loss is the measured mean training objective. Teacher-forced text/scores are
labelled and cannot stand in for independent recall scores. Pause/Stop/Checkpoint
remain episode-boundary actions. The staging policy is unchanged.

Codex / GPT-6 / 2026-10-07 UTC
