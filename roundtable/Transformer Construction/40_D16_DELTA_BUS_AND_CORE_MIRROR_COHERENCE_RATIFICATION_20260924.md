# 40 — D16 Delta Bus and Core Mirror Coherence — Ratification Candidate

Identity: ChatGPT / GPT-5.6 Sol / 2026-09-24 America/Chicago
Status: RoundTable ratification candidate; no runtime code changed by this document.
Authority basis: Jeff's explicit 2026-09-24 direction after the isolated ContinuousCoreLab experiment.

## 1. Convener direction to codify

Jeff directed that the next reasoning-Core architecture should stop treating width-specific packed rails as the Core communication surface. The intended new serving boundary is:

- Heart remains the sole canonical Shared Field owner and sole canonical writer.
- Canonical/public transport remains exact 16D substrate cells.
- A Core has its own architecture-native internal width (Gen-1 candidate: D512), independent of the Heart bus width.
- Heart provides a complete exact field/view on initial synchronization and then exact versioned deltas only.
- Each resident Core maintains a non-authoritative exact local D16 mirror of the Shared Field view it is allowed to see.
- Each Core's learned/recurrent cognitive state is private and separate from that exact mirror.
- FIRST, REFINED, and FINAL communication remains exact English, but is carried directly as exact D16 transport rather than being mechanically repacked into the Core's `d_model` width.
- Heart owns proposal collection, barriers, consolidator selection, validation, canonical commit, and post-commit synchronization.
- The old D64 rail implementation is preserved as historical/proven specialist evidence but is not the intended serving interface for the new Core family.

The phrase “Core attends over the field” in this candidate means “the Core reads/consumes its exact permitted field mirror.” It does **not** imply Transformer attention heads. Internal attention remains optional future tissue.

## 2. Why this candidate exists now — isolated evidence

The experiment was deliberately outside Axon at `D:\ContinuousCoreLab`. It used a synthetic frozen 16D categorical codebook, not Axon's canonical substrate. No checkpoint or source file from that lab is imported by this proposal.

### D512 one-chamber baseline

An isolated Core with:

`exact D16 cells -> Linear(16,512) -> one GRUCell(512,512) -> categorical character readout -> exact symbol identity -> exact D16 lookup`

used no attention heads, no pointer network, no Transformer stack, no giant FFN, no learned physical addressing, and no width-specific rail.

Verified outcomes:

- Parameters: 1,660,051.
- Representative CUDA training step: ~47.9 MiB peak PyTorch allocated VRAM, ~66 MiB reserved.
- CPU benchmark, 30 representative training steps: 9.708 s total, 0.3236 s/step mean, 3.09 steps/s.
- Copy experiment: resident delta-driven path 397/400; fresh full-field replay 392/400; resident/fresh agreement 395/400.
- Corrected delayed-recall experiment: 400/400. Example: memory `CAT`, later unrelated `DOG FOX BIR`, then query -> `CAT` without replaying the old memory into the resident path.
- Replacement/correction experiment: resident 599/600, fresh rebuild 600/600, edited 311/312, unedited 288/288. Example: remember `CAT`, later exact replacement `MEMORY=OWL`, more distractors, then query -> `OWL`.

These results prove only the narrow mechanics tested: a resident wider recurrent state can consume exact D16-derived input, retain arbitrary short information across later deltas, accept a later replacement, and emit exact categorical symbols. They do not prove conversation, repository-scale reasoning, semantic understanding, Soul, or production Axon correctness.

### D2048 baseline

The same one-GRU/no-attention shape widened to D2048:

- Parameters: 25,514,131.
- Peak PyTorch VRAM observed: ~518 MiB.
- Total three-phase baseline: ~839.8 s on GTX 1650.
- Copy after 300 steps: 1/300.
- Delayed recall after 400 steps: 145/400.
- Replacement/edit after 500 steps: 581/600.

Interpretation: D2048 is computationally feasible on modest hardware and can learn the revision task strongly, but the untuned baseline is harder to optimize. Width is therefore a capacity variable, not an automatic quality improvement. D512 is the recommended first production-development chamber because it already passed the narrow continuity gates with minimal compute.

## 3. Architectural rule: mirror is evidence, chamber is cognition

The exact local Shared Field mirror and the learned chamber must never be conflated.

For a Core `c`:

- `M_c` is a variable-length exact sequence of permitted D16 transport cells plus structural metadata. It is non-authoritative but exactly reconstructable from Heart truth.
- `h_c` is architecture-native private cognition, initially proposed as D512 recurrent state.

There is no requirement to pack the whole field into one D512 vector. Large repositories, long tool results, and growing conversation history live exactly in `M_c`; the Core may incrementally process, summarize, index, revisit, or selectively activate portions into `h_c` and later hierarchical structures. Stored exact mirror data consumes memory, not continuous FLOPs.

A future D2048 or larger chamber may pay dividends for deeper active cognition without changing the Heart transport contract.

## 4. New Heart/Core interface: the D16 Core Bus

Recommended contract name: **D16 Core Bus**.

The bus is an event protocol, not a neural layer. Exact payload content is represented in the registered D16 substrate/Unicode transport. Control metadata remains typed deterministic metadata and is not forced into learned latent vectors.

### 4.1 Event classes

Minimum event set:

1. `FIELD_SNAPSHOT`
   - complete exact permitted view for initial attach or repair;
   - carries field/tick/view/mask identities and exact region hashes.
2. `FIELD_DELTA`
   - exact canonical/view change authored against one known base identity;
   - carries ordered insert/delete/replace semantics and resulting identities.
3. `MIRROR_ACK`
   - Core attests the exact identity/hash it now holds after applying snapshot/delta.
4. `RESYNC_REQUIRED`
   - Heart or Core declares a sequence gap, stale base, mask/view mismatch, hash mismatch, or other coherence failure.
5. `FIRST_REQUEST` / `FIRST_PROPOSAL`
   - reasoning trigger and exact-English response.
6. `PROPOSAL_SET`
   - Heart's exact sibling-FIRST board delivered after the FIRST barrier.
7. `REFINED_REQUEST` / `REFINED_PROPOSAL`
   - refinement trigger and exact-English response.
8. `REFINED_SET`
   - exact refined board made available to the selected consolidator.
9. `FINAL_REQUEST` / `FINAL_VERDICT`
   - rotating consolidator produces the existing tagged desired-region English verdict.
10. `CANONICAL_SYNC`
   - post-commit synchronization event carrying the exact accepted canonical change/resulting identity;
   - sync-only: it does not automatically trigger another FIRST round.

Tool/advisor results and ingress become ordinary Heart-validated canonical changes, then appear to Cores as `FIELD_DELTA` events. Tool completion during an active barrier may be queued for the next external reasoning event rather than recursively nesting reasoning rounds.

## 5. Mirror coherence — the missing word and the required mechanism

The required property is **mirror coherence** (also accurately described as state synchronization).

Heart must know whether each active Core's exact mirror is synchronized before accepting that Core as a valid reasoning participant.

### 5.1 Coherence identity

At minimum, each Core mirror ACK should bind:

- `core_id` and Core generation;
- canonical `field_id`;
- canonical `tick_id` or monotonic field sequence;
- `view_id`;
- mask-policy/revision identity;
- substrate/transport schema identity;
- exact whole-view hash or deterministic per-region hashes;
- last applied bus event sequence/id.

### 5.2 Heart-side coherence registry

Heart maintains, per Core:

- last acknowledged canonical/view identity;
- expected next event sequence;
- synchronization state: `SYNCED | APPLYING | STALE | RESYNC_REQUIRED | OFFLINE`;
- last ACK timestamp/receipt id;
- optional per-region hashes for targeted repair.

A Core marked anything other than `SYNCED` is excluded from the FIRST/REFINED success barrier until it repairs or is explicitly treated as offline/failed by the existing participant-accounting rules.

### 5.3 Repair policy

Fail closed:

- If a delta base does not equal the Core's acknowledged base, do not guess.
- If an event sequence is missing, do not skip it silently.
- If hashes disagree, do not trust latent state.
- If mask/view identity changes in a way not representable by the available patch chain, send a new exact `FIELD_SNAPSHOT`.
- If a precise contiguous delta chain is available and verified, replay it; otherwise rebuild from Heart truth.

After repair, the Core may keep or rebuild private cognitive structures according to dependency certainty. Exact mirror truth always wins over stale latent interpretation.

## 6. Proposal collection is already mostly built

Axon's existing circulation is valuable and should be preserved.

Keep conceptually:

- `runtime/heart/board.py` participant accounting and FIRST/REFINED barriers;
- exact-English `EnglishProposal` and tagged `TechnicalFinalVerdict` contracts;
- rotating consolidator semantics;
- `FieldDelta` materialization and Heart-only canonical commit;
- authority/transaction validation;
- turn finalization and recovery/autobiography receipts.

Change the transport surface:

- `ProposalWorkspace` remains a noncanonical exact English board.
- Replace width-specific `RenderedProposalRail` output with one exact D16 proposal frame/bus payload.
- Remove the requirement that sibling proposals be repacked to the recipient Core's internal `d_model`.
- Internal Core width becomes an implementation property, not a public transport width.

The proposal board is the “area to collect proposals” Jeff described. It already exists as a noncanonical workspace; the new bus should expose that workspace in one exact D16 form rather than one rendered rail per width.

## 7. What can stay versus what must change

### Keep with little or no semantic change

- `runtime/field/schema.py` — canonical Shared Field structures.
- `runtime/field/delta.py` — exact insert/delete/replace mutation semantics and stale-base checks.
- `runtime/field/state_branch.py` — canonical branch/state lineage.
- Heart authority, transaction, lease, valve, host sovereignty.
- `runtime/heart/board.py` — participant/barrier accounting.
- `runtime/heart/english_reasoning.py` — English FIRST/REFINED and tagged FINAL intent.
- much of `runtime/heart/circulation.py` — FIRST -> REFINED -> consolidator -> commit sequence.
- `runtime/heart/registry.py` — extend rather than replace.
- masks, canonical transaction validation, recovery/autobiography, Dormant evidence identity.

### Add

Recommended new modules/seams:

- `runtime/heart/core_bus.py` — typed D16 Core Bus envelopes/events.
- `runtime/heart/mirror_coherence.py` — Heart-side per-Core ACK/version/hash synchronization registry.
- optionally `runtime/field/d16_view.py` — deterministic materialization/hash of the exact permitted D16 view, without width packing.

### Adapt

- `runtime/heart/tick.py`: frozen reasoning image should bind exact field/view/mask identity directly; rail bindings become optional legacy evidence rather than mandatory Core input.
- `runtime/heart/circulation.py`: replace mandatory `RailRuntimeView` in `ReasoningPassRequest` with a bus/session/coherence binding suitable for long-running resident Cores.
- `runtime/heart/proposal_workspace.py`: preserve exact English workspace; replace width-rendered proposal rails with one D16 bus representation.
- `runtime/heart/registry.py`: track transport generation/capability plus mirror-coherence state independently of `d_model`.
- reasoning recovery/autobiography schemas: record exact bus/mirror identities and events in addition to or instead of rail identities for the new Core family.
- remote transport: evolve `remote_rail.py` into/alongside a width-independent authenticated Core-bus transport.
- Dormant/Cortex surfacing seams that currently compile directly to D64: surface canonical exact field content first; D64 compilation remains available only for the legacy path.

### Preserve but remove from the new serving path

Do not delete or rewrite history:

- `runtime/field/compiler_d64.py`
- `runtime/field/semantic_d64.py`
- `runtime/heart/d64_codec.py`
- current D64 `remote_rail.py` wire contract
- `runtime/axon_runtime/d64_adapter.py`
- D64 checkpoints/training artifacts

These remain reproducible evidence and may serve a legacy/specialist compatibility lane, but Gen-1 continuous Cores should not depend on them.

## 8. Gen-1 Core recommendation

First production-development candidate after deterministic bus proofs:

- exact canonical Axon D16 view, not the synthetic lab codebook;
- learned `16 -> 512` projection;
- one persistent D512 recurrent chamber as the smallest proven baseline;
- zero attention heads initially;
- categorical exact-symbol output serialized mechanically through Axon's registered substrate/Unicode transport;
- exact local mirror retained outside the recurrent hidden state;
- hidden state and any learned hierarchical caches are replaceable private cognition, never canonical truth.

This does not lock Axon to GRUs. The GRU is the currently demonstrated minimal recurrent chamber. Later SSM/Mamba, local attention, proposition-level attention, graph memory, multiple chambers, or wider recurrent tissue must beat the same held-out/counterfactual gates before promotion.

## 9. Build ladder after doctrine ratification

### B0 — Doctrine amendment only

Amend the packed-rail serving clauses so exact D16 bus transport is the default reasoning-Core interface. Preserve packed rails as legacy/specialist transport, not canonical requirements for every Core.

### B1 — Deterministic D16 view + bus contracts, no neural Core

Prove:

- complete canonical permitted view -> exact D16 materialization;
- exact snapshot -> Core mirror -> hash equality;
- insert/delete/replace delta -> same final mirror as clean rebuild;
- event gap -> fail closed;
- stale base -> fail closed;
- changed masks/views -> correct invalidation/resync;
- reconnect -> full resync equivalence;
- proposal board exact English -> D16 -> exact English roundtrip.

### B2 — Mirror-coherence registry

Use fake deterministic Cores to prove Heart never runs a stale Core through a successful barrier and can repair/rejoin a Core without changing canonical state.

### B3 — Circulation transport swap

Preserve existing FIRST/REFINED/consolidator/FINAL/commit semantics while substituting bus/coherence bindings for mandatory rail bindings. Existing legacy D64 circulation should remain testable independently during transition.

### B4 — Fresh D512 canonical-substrate Core

Rebuild from scratch against Axon's real frozen D16 substrate. Do not import the lab checkpoint. Re-run narrow gates:

- exact copy;
- delayed recall across unrelated deltas;
- replacement/correction;
- resident-versus-clean-rebuild comparisons;
- insertion/deletion/mask-change stale-state probes.

### B5 — Developmental language curriculum

Proceed from substrate use to pseudowords/words, vocabulary definitions, grammar, sentence meaning, conversation continuity, and exact English generation. The Core should learn language and reasoning, not physical address geometry.

### B6 — Multi-Core organism circulation

Clone or independently train multiple compatible Cores, give each a separate private cognitive/Soul lineage, and run the existing FIRST -> proposal set -> REFINED -> rotating consolidator -> FINAL -> Heart commit -> CANONICAL_SYNC cycle.

## 10. Required Source-of-Truth reconciliation

Before B1 becomes the serving runtime, explicitly reconcile at least these current binding statements:

- packed rail as mandatory Core serving surface (`docs/SOURCE_OF_TRUTH.md` packed-rail section);
- Cores attend directly to their designated rail;
- Heart repacks every English proposal onto every destination rail width;
- D64 receipt-continuation requirements that apply specifically to the legacy D64 family;
- runtime reasoning evidence schemas that currently make D64 rail identity part of every reasoning episode.

Recommended doctrine replacement principle:

> Canonical/public organism truth and inter-Core communication are exact substrate transport at D16. A Core's internal width is private architecture. Heart serves each authorized Core an exact versioned D16 view, then exact deltas, and validates mirror coherence. Derived packed rails may exist as optional specialist/legacy interfaces but are not required for the continuous Core family.

## 11. Stop conditions

Halt and investigate if any of the following occur:

- a Core can participate while its mirror hash/version is stale;
- applying the same ordered delta chain does not produce byte/vector-identical mirror state to clean rebuild;
- Heart must semantically interpret a Core's latent state to synchronize it;
- D16 output is accepted from arbitrary continuous vectors rather than registered categorical transport identities;
- proposal traffic mutates canonical state before consolidator/Heart validation;
- a Core's internal `d_model` leaks back into the public bus contract;
- a new implementation requires deleting or rewriting the legacy D64 evidence to work.

## 12. Recommended decision

Ratify the D16 Core Bus and mirror-coherence architecture as the next reasoning-Core serving direction, with D512 one-recurrent-chamber/no-attention as the first **development baseline**, not a permanent tissue ceiling.

Then amend Source of Truth narrowly, build B1/B2 deterministic proofs first, and reuse the existing Heart circulation/transaction machinery rather than replacing the organism.
