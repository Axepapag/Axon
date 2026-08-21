# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-21T15:04:42-05:00
Current through event: `evt-20260821T200442765692Z-chatgpt-heart-dormant-design-review`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission state

Axon remains at the clean Day Zero boundary, now with **P0 dormant evidence retrieval/surfacing complete and proven against the real recovered corpus**.

Current engineering focus is **not** the old scripted/non-neural P1 skeleton. Jeff's newer directive supersedes it: build one real organ at a time, no scripted/fake organs, with the **64D Field Compiler Organ / heart** next. Heartbeat is its own cadence and is distinct from a cognitive tick.

Before heart implementation chooses unresolved behavior, the newer heart directive needs explicit reconciliation with `docs/SOURCE_OF_TRUTH.md` on semantic scope and runtime mechanics. Convergence is running on `roundtable/ENGINE_TEAM_BUS.md`: Kimmy proposes exact-only heart-v1 semantics, immediate typed ingress commit to `user_input`, adaptive beat cadence, a journaled proposal board projected into rails, and a formal Source-of-Truth heart amendment before code. ChatGPT's 15:04 design review agrees on doctrine-before-code, adaptive/event-driven circulation and sparse proposals, but flags one substantive gap: current P0 candidate retrieval is hashed lexical matching plus graph expansion, not genuine semantic search/reranking. If Jeff's requirement is semantic recall across lexical mismatch, the dormant valve still needs derived semantic/relevance tissue before that requirement is satisfied. Formal ChatGPT bus delta is deferred until Jeff rules after discussion.

## Binding authority and invariants

- Jeff is final authority; `docs/SOURCE_OF_TRUTH.md` is the architecture master below his explicit rulings.
- Root `SOURCE_OF_TRUTH.md` is an exact mirror. Current verified SHA256: `717D2D9A2DE44F4C5A0D59F9E0D3EAEFD2E644E52DCCA8D964BFC66290597D5A`.
- Root/docs Working Contract are exact mirrors. SHA256: `DC946600ADD64C51EC0AE40BB4D7F4F5E708387DD2A9E384DE268701959FAA31`.
- One living/durable State root: `D:\Axon\State`.
- One dormant-memory authority: exact recovered files under `State\dormant`.
- The derived SQLite evidence index is a disposable lookup sense, not a second memory body.
- Exact visible text remains grounded in the frozen 16D character substrate.
- D64 compiler output is derived/authority-free; reasoning/consolidation proposes and typed validation/transaction owns commit.
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

## Open heart questions — do not silently answer in code

Current unresolved heart questions:

1. **Semantic valve scope:** Jeff now explicitly describes the dormant connector as a heartbeat valve that must quickly semantically search the entire dormant body and surface only highly relevant knowledge/memory. Current P0 is lexical+graph, exact and strong but not genuinely semantic across lexical mismatch. Decide whether learned 64D semantic retrieval/reranking belongs inside the first heart or immediately follows deterministic circulation proof.
2. **Exact + semantic core rail:** preserve heart-level exact 16D/D64 roundtrip for grounding/audit while adding derived semantic word/sentence/concept/edge slots bound back to exact spans, versus making exact rows the only core-facing representation in v1.
3. **Ingress authority:** external user/tool/advisor input should become canonical immediately, but current core `FieldDelta` cannot write sealed `user_input`; define a typed runtime/heart-owned mutation path rather than bypassing transaction rules.
4. **Tick freeze / proposal board:** recommend freezing canonical `field_id` during a tick; initial/refined core proposals are sparse typed patches in a separate deliberation overlay, not canonical state. Decide whether proposal-triggered dormant recall may only enrich that overlay or may restart the heartbeat/tick.
5. **Heartbeat cadence:** event-driven immediate beat on canonical change, with adaptive idle backoff/coalescing, versus another policy.
6. **Production `structured_knowledge` policy:** relevance threshold/budget, novelty/diversity, retention/hysteresis, replacement/masking and provenance when the dormant valve refreshes evidence.
7. **Final delta semantics:** cores and consolidator should emit sparse patches, not duplicate the entire shared field; the consolidator's delta is complete in decision/authority and may address multiple regions while remaining sparse in payload.

No Source-of-Truth change was made in either review; these require Jeff/table convergence first.

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

1. Jeff rules the seven open heart questions above after considering Kimmy's round-1 delta and ChatGPT's dormant/semantic review.
2. ChatGPT then posts its formal convergence delta to `roundtable/ENGINE_TEAM_BUS.md`; continue Kimmy/ChatGPT rounds until the heart contract converges.
3. Reconcile the converged heart contract into `docs/SOURCE_OF_TRUTH.md` and root exact mirror before unresolved behavior is implemented.
4. Finish the dormant valve to the level Jeff actually requires: real change-triggered recall, governed relevance/retention and, if ruled in-scope, genuine derived semantic retrieval/reranking over the real dormant body.
5. Build the first **real 64D heart** in permanent anatomy using real field/input/dormant evidence; preserve exact grounding while providing the agreed semantic core-facing rail; no fake/scripted proposer.
6. Prove heart change detection, governed evidence surfacing, cartography/semantic projection, freshness, rail generation and heart-level exact integrity before moving on.
7. Attach the real 64D reasoning/tick path: sparse initial proposals -> all-proposal visibility -> soul exhale/inhale -> sparse refinement -> rotating consolidator -> one validated sparse authoritative decision delta -> successor field -> next heartbeat.
8. Then make training branch-backed/resume-proven, port remaining faculties only onto the permanent spine, and expand rails sequentially after 64D is working.

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

**ChatGPT / GPT-5.6 Sol / 2026-08-21**
