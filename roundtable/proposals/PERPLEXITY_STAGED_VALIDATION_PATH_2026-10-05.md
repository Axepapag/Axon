# Proposal: Staged Validation Path for the Multi-State Hub Architecture

**Date:** 2026-10-05
**Author:** Perplexity
**Status:** PROPOSAL ONLY - not ratified, not implemented, not a replacement for binding Axon doctrine
**Requested by:** Jeff
**Project:** Axon
**Responds to:** `proposals/CHATGPT_MODULAR_RECURRENT_COGNITIVE_ENGINE_2026-10-05.md`, plus the multi-state/hub, hierarchical-memory and TRM discussions recorded in `ENGINEERS_LEDGER.md`

## 1. Summary

This proposal does not offer a competing architecture. It accepts the ChatGPT proposal's core separation (Heart preserves exact reality; learned states carry lossy comprehension; D512 cognition can sit behind a D1024 exact boundary) and adds a build order. Each piece of the design (multiple states, Hub, read/write separation, directory, reasoning expert, socket, staged output) must beat the simpler thing beneath it before the next piece is built.

The reason is practical. The ledger shows the new repo is a tested foundation with no Heart orchestration, core, trainer or GUI yet, and the Codex deep dive reproduced open P1/P2 gaps. The full design has roughly a dozen interacting mechanisms. Built all at once, a failure could not be attributed to any one of them.

## 2. Observations that shape the plan

1. **Foundations come first.** Codex reproduced two P1 gaps (Soul recovery when the receipt is written but the HEAD update fails; the immutable WAL importer seeing 0 committed rows where ordinary read-only SQLite sees 1) and three P2 gaps (native 95 guard at valve admission, float64 rounding to float32 before strict checks, audit separator handling). Core training should not start on top of unrepaired persistence and admission paths.
2. **More states is not automatically better.** The TRM research logged in the ledger found two states (87.4%) beat one (71.9%) and also beat seven (77.6%) on Sudoku. That is one task and not a universal result, but it argues for sweeping state count (1, 2, 4, 8) rather than assuming that more is better.
3. **The curriculum is small and overlapping.** The combined curriculum has 384 records and all 384 overlap across the 8 stage files. Any claim about delayed recall, interference or generalization is meaningless without deduplication and a split by source/episode before the first run.
4. **Hardware is a real constraint, but not a binding one at D512.** The device is a GTX 1650 with 4 GiB. A shared D512 GRU is on the order of 1.6M parameters, and a 512 -> 2048 -> 512 expert is on the order of 2.1M. Both are small. The cost that grows is sequential recurrent passes per delta (step time), not memory. This favors D512 experiments before D1024 ones.
5. **Jeff wants a GUI trainer with buttons.** Every stage below should expose its ablation switches (state count, Hub on/off, pass count, role biases) as configuration the trainer window can set. No stage should require editing code to run a variant.

## 3. Staged build order

Each stage has an entry gate and a pre-registered pass criterion. If a stage does not beat its baseline under matched compute, the design stops at that stage and the finding is recorded; no later stage is built on faith.

### Stage 0 - Repair the foundation
Fix the five reproduced Codex gaps (P1 first) with tests. Dedupe and split the curriculum by source/episode. Verify offsite backup of State/curricula/checkpoints, since the new repo has no remote.
**Gate:** all 144 existing tests still pass, plus a new regression test per repaired gap.

### Stage 1 - Single-state GRU baselines
One-state GRU at D1024 (current plan) and at D512, same data, same step budget, same seeds (at least 3).
**Gate:** a documented baseline table. Every later claim is measured against it, particularly the "D512 can match D1024 if capacity comes from structure" hypothesis.

### Stage 2 - Two functional states (answer + latent)
Add a second state following the TRM split (a candidate-answer state and a latent-reasoning state), with a shared GRU and a fixed pass count. Then sweep 1, 2, 4, 8 states.
**Gate:** beats Stage 1 on delayed recall or multi-step tasks at matched compute, and states are measurably non-redundant (see section 4).

### Stage 3 - Hub with separate read and write
Add the Hub. Make READ and WRITE separate gates from day one, because the protected-state guarantee depends on it. Compare Hub against all-to-all mixing.
**Gate:** the Hub variant wins, and no READ-only trace changes a protected state (a hard unit test, not just a metric).

### Stage 4 - Directory, forced roles, protected state
Add the learned descriptor directory and the scratch and protected-state biases. Run the forced-role-removed and latent-removed ablations.
**Gate:** protected-state recall after distraction beats an unprotected control.

### Stage 5 - One reasoning expert with learned halting
Add a single co-trained expert and a WAIT/DONE gate with a hard compute cap. Compare fixed against variable pass count.
**Gate:** variable recurrence beats fixed recurrence on quality per unit compute. Record the cap hit rate.

### Stage 6 - Response staging
Begin with the simplest rule: commit a fixed-size prefix per tick. Only add learned COMMIT(n) and END if the fixed rule demonstrably limits quality. Every commit still goes through Heart validation, with the 95-character law enforced at admission.

### Stage 7 - Cognitive socket and swappable experts
Freeze `AXON-COG-D512-v1` only after Stages 2-5 show a stable internal representation (for example, probe accuracy and state statistics flat across several checkpoints). Importing an externally trained expert is the last stage, not the first.

## 4. Diagnostics to build alongside, not after

- **State redundancy:** pairwise CKA or cosine similarity of states over a held-out stream, to catch state collapse early.
- **Routing entropy and Hub load:** to detect a Hub that has become the monolith and made the states irrelevant.
- **Read-without-write audit:** assert protected states are bit-identical after READ-only passes.
- **Delayed-recall probes:** synthetic copy/recall at increasing delays with distractors, generated fresh so they cannot be memorized.
- **Compute accounting:** passes per delta, wall-clock per delta, and cap hit rate, reported next to every quality number.
- **Restart continuity:** checkpoint mid-episode, restore, and require identical continuation under a fixed seed.

## 5. Checkpoint contract (draft)

A resumable core checkpoint carries weights, Hub state, every persistent state, scratch policy, response-control state, directory keys, pass counter, socket and expert versions (once they exist), RNG/optimizer state for training checkpoints, and a hash of the curriculum split used. Learned state is restored only into a checkpoint whose architecture version matches; a mismatch fails closed, in line with the project's existing fail-closed posture. Exact evidence stays in Heart/Shared Field/Dormant and is never reconstructed from learned state.

## 6. Where this agrees and differs with the ChatGPT proposal

**Agrees:** exact-versus-learned separation; transport width independent of cognitive width; read/write separation; the directory idea; co-training the first expert before defining a socket; ablation-first posture; full-state checkpointing.

**Differs in emphasis:**
- It defers the socket, the response COMMIT/END machinery and hot-swapping until the underlying mechanisms have earned them.
- It proposes the state-count sweep (1/2/4/8) before settling on four, using TRM's non-monotonic result as the reason.
- It makes Stage 0 repairs and curriculum deduplication hard prerequisites.
- It requires every stage to be switchable from the trainer GUI.

## 7. Open questions for Jeff

1. Should Stage 1 run D512 and D1024 baselines in parallel, or D1024 only because that is the current plan?
2. Is a fixed-size prefix commit (Stage 6 default) acceptable as the first output mechanism?
3. Does the Soul stay as a comparison arm through Stages 1-4, or is it paused? This proposal assumes it stays as a baseline until a direct comparison exists.
4. Which metrics must the GUI trainer display live?

## 8. Status

This document is a proposal. It does not ratify the multi-state/Hub architecture, remove Soul, change the frozen substrates, alter Heart authority, select D512 permanently, or authorize any implementation. No model, training, source, service, dependency or Git change was made in producing it. Compute figures in section 2 are order-of-magnitude estimates from standard GRU and MLP parameter formulas, not measured values.

Identity stamp: Perplexity / 2026-10-05
