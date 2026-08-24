# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-24T14:58:38-05:00
Current through event: `evt-20260824T195838633077Z-codex-d00-heart-trainer-real-d64`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Executive state

Axon now has a functional permanent mechanism for exact recovered memory,
crash-safe accepted-ingress autobiography, real-D64 Heart input, governed
candidate training, deterministic lived-experience session compilation and
inspectable failure evidence. It does **not** yet have a serving-capable learned
Heart, Semantic Cortex or reasoning core. Every current Heart candidate is
rejected and no promotion proposal, activation receipt or active learned
generation pointer exists.

The project remains on the ratified Heart-first sequence. The next work is not a
wider model or unrelated organ: repair the D64 translator's free-running
decoder/curriculum, prove exact grounded conduction, complete autobiographical
event capture, then turn exact lived-experience sessions into organ loaders.

## Binding decisions and invariants

- Jeff is project convener and final authority.
- `docs/SOURCE_OF_TRUTH.md` and root `SOURCE_OF_TRUTH.md` are byte-identical at
  SHA256 `C590179ED0B37E65337D4F8CD4FB406B2D1CA013DE846384DF59A63CD552DA63`.
- No trained anatomy may contain a fixed character/context ceiling, finite
  learned position table, wrapping page index, destructive truncation or
  silent long-item exclusion. Pages/batches/steps are compute controls only.
- Developmental anatomy must have an honest permanent specialist or additive
  upgrade/composition path. D64 may remain useful when wider peers arrive.
- The Heart alone validates/materializes canonical Shared Field commits.
- The Trainer alone governs parameter mutation, candidate lineage and
  activation. Falling loss never authorizes serving.
- `D:\00` is protected read-only autobiographical source evidence. Never modify
  or delete its databases.
- Exact autobiographical evidence and derived semantic knowledge coexist in
  Dormant State; derivation never replaces exact source records.
- Observed assistant responses are evidence, not automatically correct targets.
- Reasoning and Semantic Cortex learning remain behind a functional Heart.

## Exact memory and autobiography

Seven durable `D:\00` sources totaling 3,529,335,677 bytes are preserved
byte-for-byte under `State/dormant/experience_v1/source_snapshots/`.

- Snapshot ID: `ed94773707e0b60876b75b8410adf34d1748d3eb4cf1c7e70c1095010f05394f`
- Exact import ID: `718f33bf470b90f3f1b2375de3aeb8bf4f47f48c5440b21395f9a2d0b3933ba6`
- Exact logical records: 59,875
- Counts: 28,410 messages; 248 missions; 1,452 objectives; 20,407
  episodes; 8,567 backlog jobs; 33 diary entries; 758 runtime conversation
  messages.

`runtime/heart/autobiography.py` now deposits every canonically accepted
external user/tool/advisor ingress event before the durable spool acknowledges
it. Commit-before-deposit and deposit-before-ack failures recover idempotently.
Future Axon responses, tool invocation requests, consolidator decisions and
Trainer outcomes still need automatic deposit hooks.

The old recovered container/edge JSONL corpus remains the exact authority used
by the current Dormant evidence bridge. `experience_v1` is a separate exact
authority and is not yet indexed/surfaced by Cortex.

## Trainer state

The Trainer's parameter authority, preflight firewall, isolated candidates,
telemetry, checkpoints, gates and rejection behavior remain healthy.

`runtime/trainer/sessions.py` now compiles deterministic content-addressed
sessions that reference exact experience IDs and complete context ranges:

- Heart grounding session
  `1d781c2768ca51a0203260f98611223dfe249b14c9eb20469da9f75ee7277e29`:
  59,858 examples.
- Observed conversation session
  `9ee2747132fa2a70aceacfa777023ebb2c51d116ecece7a62e7dc6bbfaa5733f`:
  14,205 examples; explicitly non-promotional.
- Split policy: stable SHA-based 80/10/10 train/heldout/regression.

The Heart training recipe now uses deterministic shuffled epochs: all 590
synthetic mechanism cases are visited once before reshuffling. Recipe ID
`a7f293f9762f86a68cecdd93cd9b0a96f304a4e4a62a0f14cf2dd18eafbfac25`
is bound into candidate generation and source lineage.

Session manifests are not yet organ-specific dataset loaders. Outcome-derived
target quality, replay, counterfactual construction and Cortex/reasoning loader
integration remain open.

## Heart state and capability truth

Architecture v3 is permanent 64D tissue: two Transformer layers, four heads,
4096 FFN, grounded from frozen 16D character cells. A 256-character page is a
processing unit, not a context limit. Two ordered sweeps visit every active
character with unbounded deterministic positions.

`runtime/heart/d64_codec.py` compiles actual `SharedFieldSnapshot` instances
through exact and semantic D64, verifies exact roundtrip/grounding, and supplies
literal raw 16D lane cells plus canonical positions. Earlier masked text cannot
renumber later active text; substituted cells or stale identities fail closed.

The strongest governed diagnostic is the 512-step batch-8 v4 run:

- Run ID: `9257422e04d5f23b80ccfb0550742212d47f1fa84f9c0278696ed69784010fc6`
- Loss: 4.9441 -> 2.6819
- Held-out termination: 0.8182
- Source-semantic exactness: 0.2857
- Referent-pointer exactness: 0.5844
- Grounding-pointer exactness: 0.6753
- Regression failures: 14
- Exact translation: 0.0
- Grounded roundtrip: 0.0
- Aggregate semantic fidelity: 0.0
- Outcome: rejected, not activated.

Evaluation v4 stores every generated string and its exact per-case decisions.
Strict checkpoint re-evaluation artifact
`State/training/heart/evaluations/f28e035c07c644c92b440d48d87835b0d200fef9be15d00032eca630ec52ea9f.json`
shows repetitive free-running text despite improving teacher-forced loss and
pointer accuracy. The next defect is decoder exposure/curriculum, not field
visibility. Do not spend another long campaign or increase width until this is
addressed.

## Verification and Git

- Final full suite: 354/354 passed.
- Known non-failing warnings: PyTorch nested-tensor warning and unwritable local
  `D:\Axon\.pytest_cache`.
- `D:\00` source sizes/mtimes remained unchanged; archive was not modified.
- D: retained about 140.3 GB free at closeout.
- Implementation commits pushed during this turn:
  `1904cd5`, `65cfe4d`, `ee861da`, `3934af0`, `e8b40c2`.
- Implementation authority before ledger-only closeout:
  `origin/main == e8b40c2b149b003a0e85be9372edc000413ec489`.

## Active flags

1. **CAPABILITY BLOCKER:** no learned Heart translator is usable; best candidate
   has zero exact translation and grounded roundtrip.
2. **AUTOBIOGRAPHY GAP:** accepted ingress is covered, but response/tool-request/
   consolidator/learning-outcome event classes need deposits.
3. **TRAINER INTEGRATION GAP:** lived sessions are governed artifacts, not yet
   Heart/Cortex/reasoning loaders.
4. **RETRIEVAL GAP:** `experience_v1` is not yet available through the Dormant
   evidence bridge/Cortex.
5. **RESOURCE NOTE:** exact snapshots plus logical records consume about 4.30 GB;
   first full session compilation is RAM-heavy but completed.

## Recommended next actions

1. Add explicit identity/copy/autoencoding curriculum stages and free-running or
   scheduled-sampling diagnostics for the current D64 Heart.
2. Add teacher-forced token/EOS accuracy and a tiny train-set overfit gate so
   optimization is distinguished from greedy generalization before GPU spend.
3. Turn v4 repetitive outputs into failure-driven decoder tests; run one small
   overfit diagnostic before another balanced campaign.
4. Deposit Heart responses, tool requests/results, consolidator commits and
   Trainer evaluation outcomes into exact autobiography.
5. Add verified `experience_v1` retrieval/Cortex projection without normalizing
   or replacing source evidence.
6. Convert lived-experience session manifests into provenance-aware loaders with
   target-quality, replay and counterfactual gates.
7. After decoder repair, run one bounded D64 candidate under unchanged serving
   floors. Only after exact grounded conduction should wider rails or
   Cortex/reasoning training begin.

## Important commands and paths

- SOT: `docs/SOURCE_OF_TRUTH.md`
- Exact autobiography: `State/dormant/experience_v1/`
- Import: `python curator/import_d00_memories.py`
- Sessions: `python scripts/compile_lived_experience_sessions.py`
- Heart smoke: `python scripts/train_heart_translation_smoke.py --steps N --batch-size N --device cuda`
- Trainer inspection: `python scripts/inspect_trainer.py --state-root State`
- Tests: `python -m pytest -q`
- Canonical ledger append helper: `scripts/append_engineers_ledger_event.py`
