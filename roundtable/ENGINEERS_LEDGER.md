# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-22T22:07:44-05:00
Current through event: `evt-20260823T030744617562Z-chatgpt-build-d2-acceptance`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission state

Axon now has a **real permanent Heart runtime, accepted C.1/C.2 dormant-memory senses, accepted D.1 dual-surface D64 anatomy, and accepted Build D.2 grounded specialist evaluation anatomy**. The exact D64 character rail remains the sole lossless coverage/roundtrip scaffold and Heart remains the sole canonical mutation boundary. D.2 service schema v2 binds every specialist query/observation to an explicit exact field/tick/rail/semantic-surface projection plus selected grounded slot receipts; stale or substituted projection identity fails closed.

Full active verification is **290/290 tests passed** on the D.2 tree. The accepted 64-case held-out reranking evaluation reuses the C.1 forward semantic-edge task and identical candidate pools: pool recall 0.687500; untrained D.1 structural-lexical cosine Hit@8 0.078125 / MRR 0.021354; existing exact-evidence C.1 auditor Hit@8 0.625000 / MRR 0.529557. The D.2 evaluator compiled 1,659,828 exact characters into 250,507 grounded semantic slots. One recovered candidate contains exact `μ`, unsupported by the frozen 16D substrate; it is explicitly counted as D64-inaccessible, never normalized/truncated, and is not an expected target. Artifact SHA256: `1e176a27d6ab24cf79969f73c6ab8b68486f66acaaf5d0176e8b5b453cc4b0ee`.

D.2 does **not** claim learned English semantics: `d64-structural-lexical-cosine-v1` is an untrained measured baseline and demonstrates a material semantic capability gap. The evidence now justifies a contained narrow semantic-edge reranking training experiment, but `semantic_cortex` remains CLOSED; no model, wider rail, fake core, reasoning vote, production service, promotion, or new commit path has been activated.
## Binding authority and invariants

- Jeff is final authority; `docs/SOURCE_OF_TRUTH.md` is the architecture master below his explicit rulings.
- Root `SOURCE_OF_TRUTH.md` is an exact mirror. Current verified SHA256: `91CBAC89DBF9DE756CA6B1DCBAC27223366072DC7720C7D8B7E9271815F7360F` (Build D.2 grounded specialist evaluation ratification).
- Root/docs Working Contract are exact mirrors. SHA256: `DC946600ADD64C51EC0AE40BB4D7F4F5E708387DD2A9E384DE268701959FAA31`.
- One living/durable State root: `D:\Axon\State`.
- One dormant-memory authority: exact recovered files under `State\dormant`.
- The derived SQLite evidence index is a disposable lookup sense, not a second memory body.
- Exact visible text remains grounded in the frozen 16D character substrate.
- D64 compiler output is derived/authority-free; cores/consolidator propose, while the heart-owned typed validation/transaction boundary alone materializes canonical commit.
- Archived Day Zero code/state is evidence only, never implicit fallback.
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

Active canonical D64 training:

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

Reverified 2026-08-22 after live Heart proof:

- `State/active` is now live and contains the canonical active branch plus Heart control/health metadata; this is the first real canonical runtime body, not smoke-only state.
- Live canonical head after the first real Heart circulation: `36ee60543e55e327b3dd7353c3c26307b67d78d25a4d443943fb002e82d3aa9e`.
- First real Heart circulation surfaced 4,241 chars of P0 dormant evidence and froze an exact D64 rail; dormant index id `139f4a6268e7c500378426e804a11e2d8ec847c52d4b00b464dab168c9d2c330`.
- Restart/idle proof advanced heartbeat/start identity without changing canonical field/tick; pending durable ingress remained 0.
- Cross-process lease proof denied a second live Heart writer.
- No production neural D64 reasoning core is attached yet; absence of reasoning participants is explicit rather than simulated.
## Kimi / team coordination

`roundtable/ENGINE_TEAM_BUS.md` is coordination only; canonical ledger remains historical authority. It is now tracked/durable as of `72d8da6`.

Kimmy maintains `D:\kimmy` and holds day-to-day senior-engineer steering unless Jeff directs otherwise. ChatGPT owns its own `D:\ChatGPT_State` continuity when ChatGPT is active; Jeff explicitly requested that refresh in this audit turn.

## Next recommended actions

1. ~~Heart Host + Sovereign Valve Plane~~ **DONE / accepted 2026-08-22**: permanent single-writer Heart, durable identity/ingress, poison handling, twenty-slot fail-closed valves, rich provenance, view identity, health, production entry point, live P0/D64 proof, restart proof, and cross-process lease proof are green; full suite 248/248.
2. ~~Build C.1 dormant relevance/generational substrate~~ **DONE / accepted 2026-08-22**: bounded recovered-relation candidate expansion, deterministic relevance/retention auditor, exact canonical evidence refs, verified atomic index generations, held-out forward evaluation, and Cortex service contracts are green; accepted v3 forward benchmark pool recall 0.687500, raw Hit@8 0.203125 / MRR 0.053032, audited Hit@8 0.625000 / MRR 0.529557; full suite 266/266.
3. ~~Build C.2 incremental derived-index maintenance~~ **DONE / accepted 2026-08-22**: append-only and equal-length/layout-preserving changes update the disposable SQLite sense transactionally, publish/recover verified logical generations, reject stale plans, and fail closed to full isolated rebuild when authoritative byte layout shifts; full suite 277/277.
4. ~~Build D.1 first-form dual-surface D64~~ **DONE / accepted 2026-08-22**: exact D64 remains lossless authority while `structural-lexical-v1` 64D word/sentence/paragraph/source-span slots bind exact lanes, provenance and evidence refs; frozen tick v2 binds semantic generation identity; live active proof 4,268 exact chars -> 704 grounded slots with HEAD unchanged; full suite 286/286.
5. ~~Build D.2 grounded specialist evaluation boundary~~ **DONE / accepted 2026-08-22**: Cortex service schema v2 binds specialist I/O to exact D64 projection/slot receipts; the untrained structural-lexical cosine baseline was measured on the accepted 64-case C.1 task at Hit@8 0.078125 / MRR 0.021354 versus exact-evidence auditor 0.625000 / 0.529557; full suite 290/290.
6. **Next intelligence experiment:** train the smallest narrow semantic-edge reranking specialist against the declared D.2 held-out boundary. It must preserve exact projection/evidence grounding and beat declared baselines/counterfactual gates before promotion. Keep the experiment isolated beneath `State/training` and do not open the production valve merely because training runs.
7. **Semantic Cortex remains behind the CLOSED valve:** no learned model/service is active. Promotion/opening requires explicit evidence and a separate governed integration step.
8. **Build E remains the reasoning lifecycle target:** real core proposal/refinement/consolidator lifecycle -> one Heart commit -> successor field -> next heartbeat. Do not substitute a reranker for a reasoning core or let specialist training create a second canonical authority.
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
- Trainer: `training/train_complete_field_64d.py`
- Dormant authority: `State/dormant/`
- Full tests: `python -m pytest -q -p no:cacheprovider`

**ChatGPT / GPT-5.6 Sol / 2026-08-22**
