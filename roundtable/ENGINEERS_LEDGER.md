# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-25T18:47:15-05:00
Current through event: `evt-20260825T234715350159Z-codex-heart-long-position`
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
was rejected without activation. A new additive long-position curriculum now
probes exact page-boundary and late-tail positions through held-out length 769.
Protective replay/EOS training improved all principal teacher-forced alignment
metrics, but free-running exactness remained fixed at 1/46 and the learning
slope flattened. Further identical step doubling is not evidence-supported.

## Binding decisions and invariants

- Jeff is project convener and final authority.
- `docs/SOURCE_OF_TRUTH.md` and root `SOURCE_OF_TRUTH.md` are byte-identical at
  SHA256 `F2FEA9E40433E191E9B2084E25AC120D1DF8063F4F123E8ECFF652D89E5680A7`.
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
`15d74aa6f4aea1eda93c3c21379fd3aaa4f6363bba1b6b82b12df38d20ada8e2`
reports 18 authorizations, 19 evaluations, 18 gate decisions, 17 preflight
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

Long-position curriculum
`ed83bb3669898e950fab44af43187506d8f1ee6a2fe041299e80e21acfc4387f`
is additive: 64 train cases, 46 held-out cases, all 37 earlier train cases as
replay, 14 source-change pairs, train lengths through 640 and held-out
extrapolation through 769. It probes positions 0, 254-257, 510-513, 699 and
768. These are complete-case evidence points, not limits; nothing is truncated.

The first 64-step run
`d51a6047c6a51401633cb38033d767eb150877a2c7c2a68068d9e04138c7fa11`
raised held-out character accuracy 0.0824 -> 0.1077 and diagonal top-one
0.0870 -> 0.1034, but replay EOS fell 0.2432 -> 0.1351. It was rejected.
Trainer controls were then extended with protective replay cadence,
configurable EOS weight, and memory-safe batching of independent complete
evaluation cases. The latter reduced observed VRAM from about 3.88 GiB plus an
allocator OOM warning to about 1.57 GiB without changing field coverage.

Protected 192-step continuation
`3e12b8a642fde98c3c6feaf39baf3dd4fdb058af903cf19eb599eedddbcb69f9`
used 64 novel and 128 replay steps with EOS-route weight 1.0. Audit loss fell
8.7739 -> 7.2936; held-out character rose 0.1077 -> 0.1479, diagonal mass
0.0586 -> 0.0786, diagonal top-one 0.1034 -> 0.1497, source-change pairs 2/14
-> 3/14, replay character 0.2533 -> 0.2992 and replay EOS 0.1351 -> 0.3243.
Free-running exactness stayed 1/46. Read-only step-64/128 evaluations measured
held-out character 0.1163/0.1404 and diagonal top-one 0.1233/0.1446; gains
flattened by step 192 and sequence exactness never moved. All checkpoints are
durable, the candidate failed twelve requirements, and it remains rejected.

The D64 tissue is learning but is not a learned Heart. The next bounded shot is
an ablation between staged whole-case exposure and an additive explicit
positional-copy facility that preserves content attention, semantic paths,
old parameter tissue and exact evidence. It must not hard-code a claim of
semantic Heart function.

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

- Full repository suite: 365/365 passed.
- Fixed-character poison scanner: passed with zero violations.
- Python compileall and `git diff --check`: passed.
- Known non-failing warnings: PyTorch nested-tensor warning and unwritable local
  `.pytest_cache`.
- Changed-file Ruff, compileall, poison scan, SOT mirror, and diff hygiene pass.
  Global Ruff 0.16.4 is installed; whole-active-repo baseline currently reports
  211 findings, intentionally not mass-fixed in this training turn.
- Implementation commits `25a620bb39b85f966b76c4d8397d9250da7d3e15`,
  `83b48fce8c943f409b741bc002684d79c86aac68`, and SOT commit
  `00b56b7a9cdac88c91896f926fd9213037c64410` were pushed to `origin/main`.
- No Heart training/evaluation process or Trainer writer lease remained.
- D: retained 139,902,599,168 bytes free after artifacts. All training used the
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
6. **TOOLING BACKLOG:** Ruff 0.16.4 is installed/configured and changed files
   pass; 211 pre-existing findings remain across the active repository.

## Recommended next actions

1. Specify an additive positional-copy facility and strict v3-checkpoint
   parameter migration; preserve content attention and require ablation evidence.
2. Add staged whole-case exposure bands as pedagogy only—never source slicing,
   truncation, model ceilings, or permanent exclusion—and compare against v3.
3. Continue identity conduction only when the new mechanism moves sequence-level
   exactness as well as characters; require unseen exactness, all counterfactuals,
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
- Latest long-position evidence:
  `State/training/heart/generalization_runs/3e12b8a642fde98c3c6feaf39baf3dd4fdb058af903cf19eb599eedddbcb69f9/summary.json`
- Checkpoint recovery: `scripts/evaluate_heart_decoder_generalization_checkpoint.py`
- Trainer inspection: `python scripts/inspect_trainer.py --state-root State`
- Poison scanner: `python -m pytest -q tests/test_no_fixed_character_poison.py`
- Full tests: `python -m pytest -q`
- Changed-file lint: `python -m ruff check <changed files>`
- Canonical append helper: `scripts/append_engineers_ledger_event.py`
