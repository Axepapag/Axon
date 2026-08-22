# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-21T19:26:26-05:00
Current through event: `evt-20260822T002626307422Z-chatgpt-semantic-cortex-proposal`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission state

Axon remains at the clean Day Zero boundary, now with **P0 dormant evidence retrieval/surfacing complete and proven against the real recovered corpus**.

Current engineering focus is **not** the old scripted/non-neural P1 skeleton. Jeff's newer directive supersedes it: build one real organ at a time, no scripted/fake organs, with the **64D Field Compiler Organ / heart** next. Heartbeat is its own cadence and is distinct from a cognitive tick.

Heart doctrine is now ratified (`b0fcb85`) and Kimmy published Build A (`63b8106`), but ChatGPT's independent supervision found **Build A is not complete yet**. The architecture is sound and the full suite passes 187 tests, but adversarial probes reproduced three transaction-boundary violations: an ordinary core proposal can commit directly between ticks; a consolidator can commit against a different base while another tick is open if optional `tick=` is omitted; and a successful consolidator commit does not atomically consume the open tick, allowing two divergent successor commits from one frozen tick. `FrozenTickImage.from_compiled()` also accepts a real `CompiledD64Field` mislabeled as d_model 128, violating the 64D-first/no-fake-wider-rails boundary. Build A.1 hardening is therefore the immediate blocker before Build B. Jeff's primitive-but-real/runtime-first rulings remain binding: semantic quality may be weak and improve later, but authority/frozen-tick anatomy must be real and fail closed before circulation advances.

## Binding authority and invariants

- Jeff is final authority; `docs/SOURCE_OF_TRUTH.md` is the architecture master below his explicit rulings.
- Root `SOURCE_OF_TRUTH.md` is an exact mirror. Current verified SHA256: `D35A0FA96FACB5717982549FBC214A268616F60F9BE1E385761C7B53558A5527` (heart amendment ratified 2026-08-21, `b0fcb85`).
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
- `runtime/dormant/evidence_bridge.py`
- `runtime/heart/` (authority, registry, tick, board, transaction — Build A, `63b8106`)

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

Reverified 2026-08-21:

- `State/active`: empty.
- `State/souls`: empty.
- `State/training`: README + empty curriculum directory; no live run/curriculum files.
- no Axon runtime/training process.
- GTX 1650 4 GiB: 0% utilization, ~1307 MiB used, 32 C when checked.
- D: free bytes ~145,027,747,840.
- Python 3.12.10.
- Git 2.55.0.windows.4.
- local MCP bridge safety mode: yolo (permission mode only; Axon contract still binding).

## Kimi / team coordination

`roundtable/ENGINE_TEAM_BUS.md` is coordination only; canonical ledger remains historical authority. It is now tracked/durable as of `72d8da6`.

Kimmy maintains `D:\kimmy` and holds day-to-day senior-engineer steering unless Jeff directs otherwise. ChatGPT owns its own `D:\ChatGPT_State` continuity when ChatGPT is active; Jeff explicitly requested that refresh in this audit turn.

## Next recommended actions

1. ~~Revise and land heart doctrine~~ **DONE 2026-08-21 (`b0fcb85`)**: wording corrections applied, Jeff's primitive-but-real/runtime-first rulings incorporated, amendment spliced into both SoT mirrors (SHA256 `D35A0FA9…`).
2. ~~Mirror equality, doctrine/hygiene tests, full suite, doctrine event~~ **DONE**: full suite exit 0 with zero failures (mirror equality enforced); event `evt-20260821T222755580056Z-kimmy-heart-amendment-ratified`.
3. **Build A.1 — BLOCKER before B:** harden `HeartTransactionBoundary` so CORE proposals are never commit-capable; CONSOLIDATOR may commit only one matching in-flight tick/frozen base and successful commit atomically consumes/closes that tick; ingress/dormant commits remain between-tick only. Add regression tests for direct-core commit rejection, omitted/mismatched tick rejection, and duplicate-final-commit rejection.
4. **Build A.1 rail guardrail:** reject any `CompiledD64Field` labeled as a non-64 d_model until a genuine wider rail implementation exists. No fake 128/256/512/1024 labels.
5. **Build A.1 doctrine truth cleanup:** add `runtime/heart/` to the Source-of-Truth active implementation surface; state incremental dormant indexing as a required future/live-memory capability rather than already implemented; allow future learned semantic improvement to strengthen the 64D heart itself instead of binding trained semantics exclusively to wider rails.
6. **Build B after A.1 is green:** real ingress/beat coordinator: heart-owned user/tool/advisor queue -> governed between-tick ingress commit -> change detection -> primitive existing P0 dormant recall/materialization -> canonical stabilization -> exact D64 compile/freeze -> tick image. Weak-but-real graph semantics are acceptable; do not wait for semantic perfection.
7. **Build C later:** mature dormant recall quality, retention budgets, semantic senses and incremental/generational index maintenance after real Build B circulation works.
8. **Build D later:** dual-surface 64D rail semantic slots bound to exact source spans; no wider rail until 64D is proven.
9. **Build E later:** attach real 64D cores/souls to the permanent tick path: sparse initial proposals -> all-proposal visibility -> soul exhale/inhale -> sparse refinement -> rotating consolidator proposal -> one heart commit -> successor field -> next heartbeat.
10. **Training follows organism anatomy.** Only after the 64D runtime/tick path is real and proven should substantive core training begin; train the cores to operate as Axon organs inside that path.

## Fast orientation

- Team bus: `roundtable/ENGINE_TEAM_BUS.md`
- Architecture authority: `docs/SOURCE_OF_TRUTH.md`
- Day Zero map: `docs/DAY_ZERO.md`
- D64 compiler: `runtime/field/compiler_d64.py`
- Canonical branch: `runtime/field/state_branch.py`
- Runtime adapter: `runtime/axon_runtime/d64_adapter.py`
- Dormant evidence bridge: `runtime/dormant/evidence_bridge.py`
- Real proof: `scripts/verify_dormant_evidence_real_index.py`
- Trainer: `training/train_complete_field_64d.py`
- Dormant authority: `State/dormant/`
- Full tests: `python -m pytest -q -p no:cacheprovider`

**ChatGPT / GPT-5.6 Sol / 2026-08-21** (prior update: Kimmy / Kimi Code CLI / 2026-08-21)
