# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-25T16:52:08-05:00
Current through event: `evt-20260825T215208751311Z-codex-heart-generalization`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Executive state

Axon has exact recovered autobiography, crash-safe accepted-ingress deposits,
complete-field Heart rails, a governed Trainer control plane, and a permanent
D64 Heart tissue whose decoder mechanism can copy a trained 374-character
multi-page sequence exactly. It still does **not** have a serving learned Heart,
Semantic Cortex, reasoning-core runner, or consolidator service. No learned
generation is active.

The first diverse Heart identity-generalization curriculum and strict evidence
gate now exist. Bounded local-GPU candidates improved unseen character and
long-position alignment metrics, exposed an EOS-route defect, and validated its
objective-v2 correction. Every candidate still failed the capability gate and
was rejected without activation. Current evidence says to target long-position
alignment/tail source dependence next; falling loss alone is not sufficient.

## Binding decisions and invariants

- Jeff is project convener and final authority.
- `docs/SOURCE_OF_TRUTH.md` and root `SOURCE_OF_TRUTH.md` are byte-identical at
  SHA256 `40B1CFC651355519DF1DAD106601B02510710FC2989B27BB80F2153FE672E9FE`.
- No trained anatomy may contain a fixed character/context ceiling, finite
  learned position table, wrapping page index, destructive truncation, or silent
  long-item exclusion. Pages, buckets, batches, and steps are compute controls.
- Developmental anatomy needs a permanent specialist or additive upgrade path.
  D64 remains useful tissue when wider peers arrive.
- Heart alone validates/materializes canonical Shared Field commits. Trainer
  alone governs parameter mutation, lineage, gates, promotion, and activation.
- Discrete content uses discrete loss. Counterfactual/source-use evidence and
  regression/replay floors outrank decreasing training loss.
- `D:\00` is protected read-only source evidence. Exact lived evidence is never
  replaced by a summary or semantic derivative.
- Reasoning and Cortex activation remain behind a functional grounded Heart.

## Memory, autobiography, and Trainer

Seven `D:\00` sources totaling 3,529,335,677 bytes are preserved under
`State/dormant/experience_v1/source_snapshots/` as 59,875 hash-bound logical
records. Accepted external user/tool/advisor ingress is deposited before spool
acknowledgement. Automatic response, tool request/result, consolidator, and
Trainer outcome hooks remain missing.

The Trainer provides leased parameter authority, preflight, isolated candidate
clones, exact mutation grants, telemetry, immutable checkpoints, deterministic
gates, activation, and rollback machinery. Current inspection snapshot
`0deb8220b9a82d51a1fbdd941e88d905b871d1328feaf3857d193780e9a8eaee`
reports 16 authorizations/plans/gate decisions, 17 evaluations, 15 preflight
receipts, zero promotion proposals, zero activation/rollback receipts, zero
active-generation pointers, and no writer lease. It is a working experiment
constitution, not yet an autonomous teacher.

## Heart identity-generalization evidence

Architecture v3 remains permanent 64D tissue: two Transformer layers, four
heads, 4096 FFN, exact frozen 16D cells, a frozen orthonormal per-character
16-to-64 lift, unbounded deterministic positions, and two ordered complete-field
page sweeps. A 256-character page is not an attention limit.

Generalization curriculum
`00d000ce720eb8be5f43bbbb30fd2a672dee11f807a369a6c71e43ab15113f88`
contains:

- 37 deterministic train cases and 21 content-disjoint held-out cases;
- all 95 supported substrate characters in training;
- lengths from 5 through 521 characters, including multi-page train/heldout and
  held-out length extrapolation;
- six exact replay cases preserving the prior mechanism proof; and
- paired single-character head, middle, and tail counterfactuals.

Length-bucketed scheduling visits every case before repeating and never pads
across buckets. Evaluation records unseen teacher-forced/free-running metrics,
diagonal alignment, length buckets, all source-change pairs, train-output
collisions, and replay. The strict gate now also requires replay character,
EOS, and termination fidelity so short exact cases cannot conceal long replay
damage.

Candidate history from this turn:

- Initial 8-step and 16-step smokes were rejected. They exposed forgetting and
  motivated 50% dedicated replay at learning rate `1e-4`.
- Candidate `h64g-1b219088d306` completed 128 steps, then a post-training
  control-flow bug made `_generalization_gate` return `None`. Its checkpoint
  remained immutable and its lifecycle correctly ended rejected. Recovery
  evaluation `99cbe965...` did not rewrite history or activate it: audit loss
  was 20.5432 -> 9.6707, held-out character accuracy 0.0729, diagonal mass
  0.0477, one of three counterfactual pairs, and five of six replay cases.
- Objective-v1 continuation `f7f29500...` improved character accuracy to
  0.1249 and diagonal top-one alignment to 0.1000 but reduced termination to
  0.6667. The loss lacked a separately balanced EOS generation-route target.
- Objective v2 keeps per-character NLL, raises identity diagonal weight, adds a
  separately balanced EOS route loss, and strengthens replay gates. A 64-step
  smoke restored held-out termination from 0.6667 to 1.0 and reduced EOS-route
  loss from 2.1079 to 0.4970.
- Final bounded v2 run
  `f05e47fcc2d16a747514eb2b7976159abb9f85777397e12996a47b07c6112d7d`
  reduced fixed audit loss 7.7983 -> 6.6192; raised held-out character accuracy
  0.1179 -> 0.1544, diagonal mass 0.0780 -> 0.1015, and diagonal top-one
  0.1076 -> 0.1510; termination remained 0.9524. It still produced only 0.0476
  greedy exactness, passed one of three counterfactual pairs, and preserved five
  of six exact replay cases; long replay character accuracy was 0.6013. It was
  rejected and not activated.

The D64 tissue is learning but is not a learned Heart. Do not resume blind step
doubling. Next curriculum work must directly exercise address/position alignment
at many long distances, both sides of page boundaries, and late-tail changes,
while protecting EOS and the complete 374-character replay.

## Reasoning and Cortex boundary

Reasoning cores may work in their native grounded home rails; they do not need
to emit raw substrate vectors. Heart must preserve each native emission,
translate it into registered discrete text plus canonical addresses and a typed
`FieldDelta`, and provide roundtrip/grounding evidence. A rotating consolidator
selects a final proposal, but only Heart validation and atomic commit make it
canonical. Frozen D64 tick binding, proposal barriers, and final transaction
mechanics exist. Core execution, native-proposal translation, refinement, and
consolidator services do not.

## Verification and Git

- Full repository suite: 363/363 passed.
- Fixed-character poison scanner: passed with zero violations.
- Python compileall and `git diff --check`: passed.
- Known non-failing warnings: PyTorch nested-tensor warning and unwritable local
  `.pytest_cache`.
- Implementation/SOT commit `8f782b595b6a688dffa520f6e53045eb006987b3`
  was pushed to `origin/main`.
- No Heart training/evaluation process or Trainer writer lease remained.
- D: retained 139,985,031,168 bytes free after artifacts. All training used the
  local GTX 1650; cloud spend was $0.

## Active flags

1. **CAPABILITY BLOCKER:** unseen complete-field identity conduction remains far
   below gate, especially middle/tail alignment and the long replay.
2. **AUTOBIOGRAPHY GAP:** response, tool request/result, consolidator, and
   learning-outcome deposits remain unwired.
3. **TRAINER INTEGRATION GAP:** lived-experience manifests are not organ loaders,
   and observed outputs are not automatically trustworthy targets.
4. **RETRIEVAL GAP:** `experience_v1` is not indexed through the Cortex/Dormant
   evidence bridge.
5. **REASONING CIRCULATION GAP:** no running core/refinement/consolidator service
   or learned native-proposal translator exists.
6. **TOOLING NOTE:** Ruff is unavailable; compileall, poison scan, diff hygiene,
   and all 363 tests pass.

## Recommended next actions

1. Build a deterministic long-position alignment curriculum with balanced
   probes around page boundaries and head/middle/tail positions across multiple
   lengths; retain complete-field coverage and no fixed ceilings.
2. Add targeted long-replay sampling and position/source-dependence diagnostics,
   then run a short objective-v2 smoke before any longer candidate.
3. Continue identity conduction until unseen exactness, all counterfactuals,
   EOS/termination, and full replay pass together. Only then blend semantic
   translation and its roundtrip/grounding gates.
4. Wire response/tool/consolidator/Trainer outcome autobiography deposits and
   provenance-aware lived-experience loaders.
5. Implement native proposal envelopes/translation receipts before connecting
   reasoning runners to the existing proposal board.

## Important paths and commands

- SOT: `docs/SOURCE_OF_TRUTH.md`
- Generalization curriculum/loss/evidence: `training/heart_translation.py`
- Governed run: `scripts/train_heart_decoder_generalization_smoke.py`
- Checkpoint recovery: `scripts/evaluate_heart_decoder_generalization_checkpoint.py`
- Trainer inspection: `python scripts/inspect_trainer.py --state-root State`
- Poison scanner: `python -m pytest -q tests/test_no_fixed_character_poison.py`
- Full tests: `python -m pytest -q`
- Canonical append helper: `scripts/append_engineers_ledger_event.py`
