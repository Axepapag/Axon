# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-23T15:38:55-05:00
Current through event: `evt-20260823T203855138785Z-chatgpt-trainer-control-plane`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission state

Shared Field schema **v2** is now implemented and live: the former canonical `structured_knowledge` region is named `cortex`. Historical v1 snapshots remain immutable and hash-identical; the active branch was advanced by an explicit schema-migration successor rather than rewriting history. Live HEAD is now field `b8ab4107382cd2e185b8233ab3742d3e1c77335dc4cd77f35a4cc0a35ce6f28a`, generation 3 / tick 3, parented to the former v1 HEAD `36ee60543e55e327b3dd7353c3c26307b67d78d25a4d443943fb002e82d3aa9e`, with the same 4,268 exact characters (27 `user_input` + 4,241 `cortex`). The old snapshot's file SHA256 remained `3A57205C6B421436F4DCCAB01DA4B79E470B97C4BF0C52CD3ED97C933E667C63` across migration.

Current Cortex doctrine is now explicit: the old `structured_knowledge` role is one retrieval function inside the broader Semantic Cortex, not a separate peer organ. The canonical `cortex` region is the auditable/materialized semantic-context surface; the Semantic Cortex may maintain richer grounded noncanonical working state on its own cortical cadence, distinct from heartbeat and reasoning tick, and may re-read an unchanged Shared Field while querying Dormant State repeatedly. Mature ownership puts dormant recall behind Cortex as a semantic sense; today's direct Heart-owned `dormant_recall` path is accepted bootstrap anatomy. `semantic_cortex` remains CLOSED and Cortex work is deliberately parked while Trainer anatomy advances.

The prior turn performed a bounded compatibility experiment with the archived 461,500-step Bible/384-slot checkpoint family and a five-step modern donor smoke; the experiment never earned promotion. Jeff has now explicitly reversed that initialization direction: those checkpoints remain historical evidence only and future reasoning/semantic cores start fresh from current anatomy. Active donor-import code/tests were removed from the live tree. The experiment artifacts remain preserved as non-authoritative evidence; no training ran in this turn.

Trainer is now the next organism focus. `runtime/trainer/` implements the first permanent non-training control plane: heterogeneous parameter registration, exact per-tensor lineage fingerprints, complete-inventory fail-closed checks, per-parameter value/gradient telemetry, explicit mutation grants/plans, immutable State-store records, and non-activating promotion proposals. Future Trainer Transformer cores are advisory only; deterministic Trainer authority remains the parameter writer. The control plane is deliberately width-agnostic: tests register 64D reasoning, 256D semantic, and 128D Trainer cores together. Full active verification is **297/297 tests passed**; no optimizer loop, training run, parameter mutation, or model promotion occurred in this turn.
## Binding authority and invariants

- Jeff is final authority; `docs/SOURCE_OF_TRUTH.md` is the architecture master below his explicit rulings.
- Root `SOURCE_OF_TRUTH.md` is an exact mirror. Current verified SHA256: `29C8CBB9A86DE8D1A2BBC35377A85F6FE1A2483FE9ED26248A81296D7A172AF0` (Shared Field v2 Cortex role/cadence + Trainer parameter-authority doctrine).
- Root/docs Working Contract are exact mirrors. SHA256: `DC946600ADD64C51EC0AE40BB4D7F4F5E708387DD2A9E384DE268701959FAA31`.
- One living/durable State root: `D:\Axon\State`.
- One dormant-memory authority: exact recovered files under `State\dormant`.
- The derived SQLite evidence index is a disposable lookup sense, not a second memory body.
- Exact visible text remains grounded in the frozen 16D character substrate.
- Rail/compiler output at any width is derived/authority-free; cores/consolidator propose, while the Heart-owned typed validation/transaction boundary alone materializes canonical Shared Field commit.
- The Trainer is the governed parameter-state authority: live parameter-bearing organs must register into a complete inventory; candidate mutation is generation-bound and grant-scoped; Trainer advisory cores never gain direct tensor-write authority.
- 64D is the present proving width, not mature Axon doctrine. Future semantic/reasoning/Trainer ensembles may contain multiple proven `d_model` widths concurrently, all grounded to the same canonical Shared Field and frozen 16D substrate.
- Archived Day Zero/Bible code and checkpoints are evidence only, never implicit fallback or initialization.
- A smoke may be small; its anatomy may not be fake.

## Day Zero / active surface

Cleanup implementation: `7db5e37` (`Establish clean Day Zero repository`).

Tracked historical implementation: `archive/day_zero_legacy_2026-08-20/`.
Local historical artifacts: `State/archive/day_zero_legacy_20260820/`.

Active canonical field/runtime:

- `runtime/field/schema.py`
- `runtime/field/delta.py`
- `runtime/field/compiler_d64.py`
- `runtime/field/state_branch.py`
- `runtime/axon_runtime/d64_adapter.py`
- `runtime/dormant/evidence_bridge.py`, `runtime/dormant/relevance.py`, `runtime/dormant/generations.py`, `runtime/dormant/incremental.py`, `runtime/dormant/evaluation.py`
- `runtime/heart/` (accepted control plane + permanent host, sovereign valves, durable ingress/replay, OS lease, cardiac identity, health, explicit view identity)

Active Trainer parameter-control plane:

- `runtime/trainer/contracts.py`
- `runtime/trainer/registry.py`
- `runtime/trainer/authority.py`
- `runtime/trainer/telemetry.py`
- `runtime/trainer/store.py`
- `runtime/trainer/host.py`

Current developmental canonical D64 training tissue:

- `training/canonical_d64.py`
- `training/complete_field_64d.py`
- `training/train_complete_field_64d.py`

Offline exact dormant construction/audit utilities remain in `curator/`.

Engineering coordination/workforce:

- `.agents/agents/`
- `scripts/run_kimi_roundtable.py`
- `scripts/run_supervised_kimi_packet.py`
- `roundtable/ENGINE_TEAM_BUS.md`

No production neural D64 reasoning driver exists yet.

## P0 dormant evidence bridge — COMPLETE

Implementation publication:

- `2dddc87` — `Implement canonical dormant evidence bridge (P0)`.
- `d9fa520` — P0 ledger publication.

Real-corpus proof publication:

- `7a3d4a1` — `Add real-corpus dormant evidence roundtrip proof`.
- `661b43f` — proof ledger publication.

Real derived index:

- `State/dormant/.derived/evidence_v1/index.sqlite3`
- 4,415,164,416 bytes
- 427,001 containers
- 351,978 edges
- index ID `139f4a6268e7c500378426e804a11e2d8ec847c52d4b00b464dab168c9d2c330`.

The index stores lookup metadata/hashes/row references and hashed lexical postings; exact authoritative text/provenance is dereferenced from JSONL and reverified before surfacing.

### ChatGPT drift-audit re-verification — 2026-08-21

`python scripts\verify_dormant_evidence_real_index.py`:

- exit 0;
- `all_ok=true`;
- 3/3 live queries passed exact verified dereference -> provenance-bearing `structured_knowledge` -> complete D64 compile -> exact roundtrip;
- observed chain times approximately 0.416 s, 1.362 s, 0.508 s.

`python -m pytest -q -p no:cacheprovider`:

- exit 0;
- **173 tests passed by progress count**;
- only existing PyTorch nested-tensor/norm-first warnings.

P0 integration caveats: `DormantEvidenceBridge.surface()` currently returns an in-memory successor `SharedFieldSnapshot` and replaces `structured_knowledge` by default. It does not persist/commit. The living heart/runtime needs an explicit governed transition plus retention/replacement/masking policy rather than treating this proof helper as commit authority. Candidate retrieval is presently exact hashed lexical matching plus semantic-edge lexical hits and one-hop graph expansion with metadata/confidence filters; this is a strong deterministic retrieval pipe, but not yet a learned semantic search or relevance auditor across lexical mismatch.

## Cross-session drift audit — 2026-08-21

Audit start:

- `HEAD == origin/main == 5f30b100ff66eb50445795d4876e766d717ae780`;
- tracked worktree clean.

Overall classification: **positive and controlled drift**.

What went right:

- P0 moved from planned/fixture-only to implemented, real-index built, and end-to-end real-corpus proven.
- Day Zero stayed intact; no council, ExactV4, identity-v2/v3, 384-slot, old core/soul, runtime bus/table, tick-loop, or legacy trainer path returned to active code.
- `State/active` and `State/souls` remain empty/reserved; `State/training` has no live run/curriculum files beyond README/empty directory.
- No Axon runtime/training process was running during audit.
- Jeff's heartbeat-vs-tick and one-organ-at-a-time framing improves architecture clarity.
- The scripted proposer plan was superseded **before** implementation.

Drift/hygiene issues found:

1. `roundtable/ENGINE_TEAM_BUS.md` was ignored/untracked even though commit `5f30b10` described establishing it. ChatGPT fixed this and posted the audit/open questions in `72d8da6` (`Make engineering team bus durable`).
2. ChatGPT's carried state still pointed at the obsolete scripted/non-neural P1 skeleton. `D:\ChatGPT_State` was refreshed to the current heart-first boundary.
3. The previous rolling summary had stale P0-next wording and the pre-P0 Source-of-Truth hash; this summary corrects them.
4. Jeff's heart directives are recorded in canonical history and outrank older doctrine, but unresolved details are not yet reconciled into Source of Truth.
5. ChatGPT's old machine profile still presented the pre-Day-Zero council endpoint as current; corrected externally. Council is archive history, not current runtime.

## Current heart direction from Jeff

Recorded standing direction:

- no scripted/fake organs;
- one organ at a time; do not advance while the current organ is broken;
- Field Compiler Organ = **heart**, running on its own heartbeat cadence;
- heartbeat != tick;
- tick = full deliberation round ending in consolidator decision/validated canonical commit;
- heart detects field change, recalls related dormant evidence, decomposes exact text structurally, organizes concepts/semantic edges, and presents core-native rails;
- **64D heart first**, then 128/256/512/1024 sequentially after each is working/dialed;
- specialized/partial-field cores are only a contemplated future doctrine amendment.

The former scripted proposer P1 plan is superseded and must not be revived as a shortcut.

## Heart round-two rulings — ready for doctrine amendment

Jeff accepted the exact+semantic semantic-valve direction and clarified the authority/tick lifecycle:

1. **One canonical body:** there is one authoritative `SharedFieldSnapshot`. Heart-owned frozen tick images, d_model rails, and proposal/refinement workspaces are derived/noncanonical and must never become a second truth body.
2. **Heart-only canonical commit:** external ingress, dormant recall materialization, and accepted consolidator decisions cross a heart-owned typed validation/transaction boundary. Ordinary cores never write canonical state. The rotating consolidator is final reasoning authority for a tick but still returns a proposed final `FieldDelta`; the heart commits or rejects it.
3. **Sparse deltas:** ordinary proposals, refined proposals, and consolidator output are sparse edits against the exact frozen base, not full-field copies. “Against the whole field” means field-wide governed authority/addressability, not payload duplication.
4. **Frozen tick base:** once heartbeat N stabilizes ingress + dormant recall and emits field/rail image N, the canonical base stays frozen for that cognitive tick. Proposal-board changes do not mutate it. New external/tool/advisor arrivals queue for heartbeat N+1.
5. **Per-rail shared reasoning workspace:** each rail carries the derived field representation plus explicitly noncanonical first-pass/refined proposal boards with author/base/rail/pass provenance and participant accounting.
6. **Explicit tick barriers:** the heart knows the core registry and rail membership. First-pass and refinement stages close only according to the declared participant set and governed failure/timeout policy; consolidation follows the refinement barrier; commit ends the tick and rings the next heartbeat.
7. **Primitive-but-real dormant semantics are sufficient to advance heart-v1:** retain P0 exact lexical/graph retrieval and exact JSONL dereference; use the recovered semantic-edge/container graph plus a real relevance auditor as the first semantic valve. It may be noisy, miss relevant evidence, or surface irrelevant evidence and still be valid progress if the mechanism is genuine, observable/testable, permanent anatomy, and its limitations remain explicit. Stronger learned/vector semantic senses remain expected future growth, not permission to fake capability now.
8. **Exact + semantic 64D rail:** preserve exact source mapping/roundtrip as a heart integrity scaffold while exposing derived semantic slots for words/phrases/sentences/concepts/edges, each bound back to exact canonical spans. Reasoning cores should not need to reproduce every character to prove grounding.
9. **Authority classes:** current core write policy is insufficient for the whole organism. Doctrine/implementation must distinguish runtime ingress-owned regions, dormant-valve-owned `structured_knowledge`, ordinary core proposal scopes, consolidator proposal scope, and heart-only canonical commit.
10. **Cadence:** canonical change is the primary doorbell; beat promptly while work exists and use a bounded slow idle heartbeat for liveness/health. Proposal-board progression is part of the in-flight tick, not a new canonical heartbeat.
11. **Living dormant index maintenance:** current full rebuild is acceptable for static P0 but not continuous memory writes. Keep derived senses open across beats and move toward incremental/generational update with atomic swap/rebuild fallback; never rebuild 4.4 GB per heartbeat.
12. **Runtime organism before substantive core training:** build the real heartbeat/tick/proposal/commit anatomy first. Reasoning cores are neurons/intelligence operating inside Axon; they should later be trained on how to function as organs within this body, not trained first as conversational models and used to stand in for missing anatomy.

Kimmy is now cleared to revise the amendment with the authority wording above, land the mirrored Source of Truth under Jeff's present authorization, run mirror/full verification, and begin Build A only. No fake proposer, training launch, wider rail, or Build B work should leap ahead of a green Build A.

## Semantic Cortex proposal — proposal only, not doctrine

ChatGPT created `roundtable/SEMANTIC_CORTEX_PROPOSAL.md` on 2026-08-21 as a durable side-branch proposal while Kimmy continues Build A.1. It does **not** authorize semantic training or interrupt the Heart path.

Core proposal:

- grow a **Semantic Cortex** as a federation of narrow semantic specialist cores rather than one universal similarity model;
- keep exact source memory/provenance authoritative while semantic vectors, graphs and ANN indexes remain disposable derived senses;
- keep Heart-only canonical materialization/commit; semantic specialists emit representations, typed-edge candidates, retrieval candidates, confidence and provenance;
- treat 64D as the first developmental width, not a permanent semantic ceiling; individual specialists may earn 128D/256D, larger FFNs, MoE or other heterogeneous anatomy based on measured bottlenecks;
- use serving/training twins with replay and promotion gates so semantic tissue can learn continually without destabilizing serving behavior;
- preserve the current real P0 lexical/graph valve as baseline circulation while learned semantic senses mature;
- defer first learned specialist selection until real Heart circulation can host it and a held-out semantic/retrieval evaluation is designed from the actual dormant corpus.

The proposal also sketches specialized taxonomy/identity, causal/mechanistic, analogy/opposition, temporal/episodic, procedural, relevance/novelty and contradiction/supersession senses; continuous per-specialist semantic spaces plus a sparse typed relationship graph; and provenance-preserving retrieval services over authorized engineer-owned memory namespaces without absorbing those memories into Axon's own dormant identity.

## Current State / machine observations

Reverified 2026-08-23 through the Shared Field v2 migration and Trainer-control-plane work:

- `State/active` contains the one canonical active branch plus Heart observability/control metadata. HEAD is now `b8ab4107382cd2e185b8233ab3742d3e1c77335dc4cd77f35a4cc0a35ce6f28a`, generation 3 / tick 3, parented to historical v1 field `36ee60543e55e327b3dd7353c3c26307b67d78d25a4d443943fb002e82d3aa9e` by the explicit schema migration.
- The live v2 field contains the same 4,268 exact chars: 27 `user_input` plus 4,241 canonical `cortex`; all other regions are currently empty. The migrated material is still the original P0 circulation specimen and carries no later C.1/D.2 container/edge refs, so it should not be used as proof that the persisted live body has exercised richer recall provenance.
- Post-migration read-only dual-surface verification is green: 1,068 exact D64 rows, exact roundtrip complete, 704 grounded semantic slots, canonical HEAD unchanged by verification.
- Exact dormant authority remains 427,001 containers / 351,978 edges; the 4,415,164,416-byte derived evidence index remains a disposable verified sense with index id `139f4a6268e7c500378426e804a11e2d8ec847c52d4b00b464dab168c9d2c330`.
- Real-index verification on 2026-08-23 passed 3/3 exact dereference -> provenance/container refs -> surfaced field -> complete D64 roundtrip.
- Last Heart health has no failure, no pending queue, no tick in flight, four primitive valves CAPPED and all future cognitive valves CLOSED. No `HeartHost` process is currently running; the durable organism is presently quiescent.
- `State/souls` is empty and runtime has no soul integration. No production reasoning core is registered/attached; runtime has no `ProposalBoard` instantiation or `declare_participants` call site yet.
## Kimi / team coordination

`roundtable/ENGINE_TEAM_BUS.md` is coordination only; canonical ledger remains historical authority. It is now tracked/durable as of `72d8da6`.

Kimmy maintains `D:\kimmy` and holds day-to-day senior-engineer steering unless Jeff directs otherwise. ChatGPT owns its own `D:\ChatGPT_State` continuity when ChatGPT is active; Jeff explicitly requested that refresh in this audit turn.

## Next recommended actions

1. ~~Heart Host + Sovereign Valve Plane~~ **DONE / accepted**: permanent single-writer Heart, durable ingress/replay, OS lease, health, twenty-slot fail-closed valve plane and Heart-only canonical transaction path are established.
2. ~~Dormant C.1/C.2~~ **DONE / accepted**: exact recovered dormant authority, deterministic relevance/retention, generational derived index and safe incremental maintenance are established; exact JSONL remains the sole memory body.
3. ~~D.1/D.2 D64 + specialist boundary~~ **DONE / accepted**: exact lossless D64 plus grounded structural-lexical semantic slots and fail-closed specialist projection receipts are established; 64D is the only real rail width today.
4. **Primary focus: complete the Trainer organ before new core training.** Build the parameter-generation lifecycle around the new `runtime/trainer/` authority: isolated candidates, optimizer/backprop execution boundary, exact checkpoint lineage, rollback, LoRA/adapter lifecycle, held-out/regression/forgetting gates, study/curriculum ingestion, and transparent live monitoring. No new semantic/reasoning training campaign starts until this control plane is ready to govern it.
5. **Trainer intelligence may become an ensemble, but authority stays deterministic.** Add advisory Trainer-core interfaces for curriculum, optimizer/gradient analysis, evaluation, forgetting audit and promotion criticism without giving those Transformer advisers unrestricted tensor-write authority.
6. **D.3 Semantic Cortex work is PAUSED, not discarded.** v3/v4/v5 remain valuable evidence that grounded semantic signal exists but query-specific representation is not mature. Keep `semantic_cortex` CLOSED while Trainer work proceeds; when Cortex resumes, implement its independent cortical cadence and move semantic dormant recall behind Cortex rather than treating retrieval as the organ itself.
7. **Reasoning circulation remains the next cognitive integration after Trainer governance.** Wire real heterogeneous core participants through `CoreRegistry` -> Heart-delivered frozen per-width rails -> `ProposalBoard` -> refinement -> consolidator proposal -> Heart validation/commit. Do not assume the first production ensemble is permanently 64D-only.
8. **Reasoning valve activation must be real, not a state flip.** `core_initial_proposal`, `core_refinement`, and especially `consolidator` are generic CLOSED placeholders today. Version them with correct source/envelope/budget/authority definitions as their actual organs are attached; consolidator activation must use true `CONSOLIDATOR` authority.
9. **Heterogeneous rails are mature doctrine; implementation remains incremental.** 64D is today's proven developmental rail. Future 128D/256D/512D/1024D or other widths may coexist once each width's exact grounding/coverage/compiler contract is proven; validation may proceed one width at a time for attributable engineering, but mature runtime is not restricted to one width at a time.
10. **Refresh the live specimen after the next controlled runtime milestone.** The active v2 Cortex material is schema-current but content/provenance still descends from the original P0 recall specimen; later governed circulation should prove current rich recall/Cortex receipts.
11. **Souls remain future runtime anatomy.** `State/souls` is empty and no runtime soul loop exists; attach souls only through the governed reasoning lifecycle and Trainer parameter lineage.
## Fast orientation

- Team bus: `roundtable/ENGINE_TEAM_BUS.md`
- Architecture authority: `docs/SOURCE_OF_TRUTH.md`
- Day Zero map: `docs/DAY_ZERO.md`
- D64 compiler: `runtime/field/compiler_d64.py`
- Canonical branch: `runtime/field/state_branch.py`
- Runtime adapter: `runtime/axon_runtime/d64_adapter.py`
- Dormant evidence bridge: `runtime/dormant/evidence_bridge.py`
- Beat coordinator / ingress queue: `runtime/heart/coordinator.py`, `runtime/heart/ingress_queue.py`
- Real proof: `scripts/verify_dormant_evidence_real_index.py`
- Trainer parameter authority: `runtime/trainer/`
- Developmental D64 trainer tissue: `training/train_complete_field_64d.py`
- Dormant authority: `State/dormant/`
- Full tests: `python -m pytest -q -p no:cacheprovider`

**ChatGPT / GPT-5.6 Sol / 2026-08-23**
