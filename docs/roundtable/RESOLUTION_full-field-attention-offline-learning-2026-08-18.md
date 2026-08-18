# Resolution: complete-field council attention, phased souls, and offline learning

Date: 2026-08-18
Authority: Jeff, convener
Recorded by: Codex / GPT-5 / 2026-08-18
Status: field, tick, and state-root rulings are binding; offline-learning
mechanism remains an implementation design requiring smoke and promotion gates

## Binding ruling

The canonical shared field is not an archive that merely preserves unseen
text. Every core must be influenced by every exact character in the shared
field during every logical attention pass. A fixed checkpoint window is a
physical processing page only. It is not the logical context boundary.

One logical tick is one atomic three-stage transaction:

1. **Proposal:** every core inhales its private soul, completes a proven sweep
   over the entire shared field, emits a complete proposed typed delta, and
   exhales.
2. **Refinement:** every core inhales again, completes a proven sweep over the
   entire base field and every brother's complete proposal, emits a refined
   typed delta, and exhales.
3. **Consolidation:** the rotating consolidator inhales again, completes a
   proven sweep over the entire base field and every complete refined delta,
   emits a typed delta addressed to the entire shared field, and exhales. The
   runtime validates and atomically commits that delta as the next canonical
   field.

No strongest-only proposal, 48-character proposal fragment, masked tail, or
unreported last-N-character slice satisfies this contract.

All Axon state lives beneath `D:\Axon\State`.

## Current implementation gap

The live 64D checkpoint has a learned 384-slot positional surface and the
council wrapper allocates only 128 history characters, 64 user characters,
and 64 response characters. The wrapper further reduces the council pool to
one strongest proposal truncated to 48 characters. It therefore does not
satisfy the ruling.

Increasing Python string length or position-embedding size would not repair
this. The reader and curriculum must be trained for complete-field sweeps and
for complete delta-set refinement.

## Feasible complete-field reader

Literal dense self-attention over 100,000 characters requires roughly ten
billion pairwise attention scores per head per layer. Even before model
activations, one float32 score matrix is about 40 GB. That is not viable on a
4 GB GTX 1650 and becomes quadratically worse as the field grows.

The viable interpretation of complete attention on this machine is a
**coverage-proven streaming reader**:

- preserve the full field as exact 16D character cells with absolute region
  and character coordinates;
- page through every character in canonical order using a trained recurrent
  read state or linear-attention state carried between pages;
- never replace the exact field with that state—the state is a reader's
  working digest, while exact characters and pointers remain authoritative;
- let any character influence the final proposal through the carried read
  state;
- emit per-pass evidence: base field hash, region lengths, page boundaries,
  page hashes, visited-character count, and a complete-coverage assertion;
- fail the tick rather than commit if coverage is incomplete or the field
  changes during a pass.

This changes the physical reader and requires matching multi-page training.
The old 384-slot checkpoint may seed local page processing, but it cannot be
claimed as a complete-field reasoner until counterfactual tests prove that
early, middle, and late characters all causally affect output.

## Delta-set representation

Every proposal and refined proposal remains exact, individually attributed,
and immutable for the duration of the tick. The next phase sweeps the ordered
set of all deltas. A typed delta references a base field hash and contains
sparse replace/append operations over any region; it does not rewrite or copy
the entire hundred-thousand-character field when only a small portion changes.

## Offline learner exploration

The proposed learning role is a rotating **sabbatical core**:

1. One brother leaves the active consolidator rotation at a tick boundary.
2. The learner receives an immutable, provenance-stamped batch from dormant
   episodic memory.
3. It reads and digests the entire batch through the same complete-sweep
   reader, writes a cited digestion record into dormant state, and trains a
   small private LoRA candidate.
4. The candidate is evaluated against identity recall, new-batch recall,
   unrelated capability retention, contradiction/unknown handling, and
   correct/zero/swapped adapter counterfactuals.
5. Only a passing candidate is copied to `State/adapters/<core>/` and attached
   at a tick boundary. The core then rejoins rotation. Failed candidates remain
   preserved under `runs/` and never become active state.
6. The offline role rotates to the next brother.

Base-parameter updates should come later than LoRA proof. Episodic facts remain
in exact dormant memory even after adapter training; parameters encode learned
habits and retrieval intuition, not the sole historical copy.

## State layout

```text
D:\Axon\State\
  active\                 canonical shared field and active transaction head
  dormant\                exact episodes, containers, provenance, digestions
  souls\council\          private live soul for each council brother
  adapters\<core>\        promoted adapters plus atomic active pointer
  learning\queue.jsonl    append-only offline batch queue
  learning\batches\       immutable source manifests and coverage evidence
  learning\promotions\    evaluation, attach, rollback, and lineage records
  archive\                processed raw material, append-only
```

Candidate checkpoints, optimizer state, and training logs live in `runs/`.
Only promoted, runtime-attached state crosses into `State/`.

## Required gates before activation

- complete character coverage on fields longer than one page;
- early/middle/late character counterfactual influence;
- every Phase-A delta visible to every Phase-B core;
- every refined delta visible to the consolidator;
- exact soul inhale/exhale ordering across all three phases;
- atomic base-hash-validated canonical commit;
- LoRA new-memory gain without held-out capability or identity regression;
- adapter correct/zero/swapped counterfactual proof;
- clean restart and rollback from `D:\Axon\State` only.

No long training or base-weight mutation is authorized by this document. The
next implementation step is a bounded reader/coverage prototype and an
offline-learning contract smoke.
