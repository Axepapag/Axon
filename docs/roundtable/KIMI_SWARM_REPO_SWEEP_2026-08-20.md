# First Kimi AgentSwarm Repository Sweep — Consolidated Report

Date: 2026-08-20
Supervisor/consolidator: ChatGPT / GPT-5.6 Sol
Published orchestration baseline: `7d11781` (`Add supervised Kimi sub-agent orchestration`)
Swarm job: `State/kimi_orchestrator/jobs/20260820t2102z-full-repo-swarm/`

## Mission and evidence status

Jeffrey asked for a read-only Kimi Swarm to sweep the Axon repository from independent perspectives, with each worker returning a delta and ChatGPT acting as consolidator.

The mission used Kimi Code's real `AgentSwarm` tool exactly once with seven `axon-explorer` items and a concurrency cap of four. Kimi persisted seven distinct worker contexts (`agent-0` through `agent-6`). The intended secondary-model routing did not take effect for this run: persisted worker wires show the started workers bound to `kimi-code/k3`. Kimi exhausted the billing-cycle usage limit after the swarm had returned four completed deltas and three failed workers. The K3 parent then received the same 403 and could not write its own synthesis.

The repository remained read-only and clean throughout the sweep. Git HEAD before and after the swarm was `7d117819f0dd2ec1757e07f9f83ab948f8861579`.

The complete raw swarm tool result is preserved at:

- `State/kimi_orchestrator/jobs/20260820t2102z-full-repo-swarm/agent_swarm_raw.txt`

The four completed Kimi deltas are separately preserved under the same job directory. Lenses 5-7 failed with the provider 403, so their Kimi deltas do not exist. ChatGPT performed independent read-only supplemental sweeps for those three lenses and labels them accordingly below.

## DELTA 1 — Kimi: doctrine and State integrity

### Verified strengths

- `runtime/field/compiler_d64.py` closely implements the current exact-D64 doctrine: one immutable snapshot, exact frozen 16D lanes, provenance-bearing addresses, complete coverage, exact roundtrip checks, stale-rail rejection, and no reasoning/commit authority.
- `runtime/field/schema.py` has the ten canonical logical regions in doctrinal order.
- `substrate/substrate.py` retains the frozen 16D substrate and fail-closed supported-character boundary.
- Canonical D64 training enforces the real `D:\Axon\State` root; state-branch tests enforce branch containment and stale/tamper rejection.
- ExactV4 production materialization is fail-closed by default.

### Verified gaps / risks

- There are **two tracked Source of Truth files**: `SOURCE_OF_TRUTH.md` and `docs/SOURCE_OF_TRUTH.md`. The root copy is dated 2026-07-18 and materially differs from the authoritative 2026-08-18 document. ChatGPT independently verified the diff.
- `runtime/council/CONTRACT.md` still points at the stale root `D:\Axon\SOURCE_OF_TRUTH.md`.
- Some legacy paths remain easy to mistake for current anatomy despite being preserved/bootstrap-only.

### Recommended move

After the core integration work below, perform one doctrine-hygiene packet: turn root `SOURCE_OF_TRUTH.md` into an explicit shim to `docs/SOURCE_OF_TRUTH.md`, repoint current contracts, and add a test preventing a second non-shim doctrine authority.

## DELTA 2 — Kimi: runtime / D64 / delta / council integration

### Verified strengths

- D64 compile, freshness, coverage, typed-delta validation, branch persistence, and replay primitives are strong and independently guarded.
- `runtime/field/delta.py` rejects stale, sealed-region, out-of-bounds, and overlapping operations.
- `runtime/field/state_branch.py` uses immutable snapshots/deltas, append-only journal events, atomic HEAD writes, and tamper checks.
- `runtime/axon_runtime/d64_adapter.py` does not acquire reasoning authority.
- Existing ExactV4 crash-safety/private-state machinery is substantial reusable engineering evidence even though that reader is not canonical.

### Verified gaps / risks

- **No production D64 neural runtime driver exists.** Runtime D64 consumers are currently the authority-free adapter/compiler plus tests; the neural runtime proposer still sits on the legacy 384x16 view path.
- `runtime/council/engine.py` maintains a parallel `dict[str, str]` canonical state, mutates it directly, and sanitizes unsupported characters instead of using the exact canonical snapshot/delta contract.
- `runtime/multi_tick_refiner.py` defaults to `compile_field_view`, the legacy 384-slot view.
- No runtime driver-shaped boundary currently says: proposal finalization is impossible unless the D64 rail is complete and fresh to the exact current `field_id`.

### Recommended move

Build a **non-neural D64 driver contract skeleton** first: load canonical HEAD -> compile D64 rail -> assert coverage/freshness -> call a scripted proposer callback -> accept only typed `FieldDelta` -> validate/commit -> recompile/roundtrip the successor. Test it before attaching a neural core.

## DELTA 3 — Kimi: training / evaluation / resume alignment

### Verified strengths

- Canonical D64 is the trainer default; detached JSON anatomy is explicit legacy-only.
- Training materializes `SharedFieldSnapshot`, uses the same `D64FieldCompiler`, performs scratch via typed delta, rematerializes/recompiles, then performs response.
- Counterfactual scratch interventions use canonical snapshots/deltas rather than direct field-dictionary mutation.
- Coverage/substrate gates, exact-position/copy supervision, causal probes, deterministic settings, anatomy fingerprints, dataset fingerprints, rolling checkpoints, and sentinels are present.

### Verified gaps / risks

- `training/verify_complete_field_v6_resume.py` is still explicitly legacy-only and uses `forward_transaction`; there is **no canonical D64 split/resume proof**.
- Anatomy-mismatch refusal exists but lacks direct negative tests.
- The canonical trainer does **not actually persist its episode through `CanonicalStateBranch`**. `training_branch()` is exported but unused by the trainer; the trainer currently creates a `State/training/runs/<name>/anatomy.json` workspace while snapshot/delta transactions remain in memory.
- Some run-gate facts are hardcoded rather than derived from the persisted coverage evidence.
- Significant legacy 384-slot training/soul machinery remains unported.

### Recommended move

After the D64 driver contract exists, add one canonical branch-backed split/resume proof: canonical transaction before save, exact resume into the same anatomy, negative legacy/compiler-schema mismatch cases, and exact successor identity verification.

## DELTA 4 — Kimi: dormant memory / retrieval / provenance

### Verified strengths

- `State/dormant` is a rich exact recovered body: 427,001 containers, 351,978 semantic edges, explicit readable relationships, provenance, confidence, and a corpus manifest.
- `runtime/axon_runtime/dormant.py` contains strong append-only exact-provenance mechanics (`SourceRecord`, `ExactProvenance`, content hashes, lifecycle transitions, immutable surfacing records, explicit omissions).
- `runtime/axon_runtime/projection.py` and `curator/typed_surfacer.py` contain useful provenance-bearing surfacing concepts into `structured_knowledge`.
- Dormant/retrieval/schema code has substantial test coverage.

### Verified gaps / risks

- **The recovered `State/dormant` body is not connected to the canonical D64 path.** No D64 compiler/runtime/training adapter reads it through a canonical evidence-surfacing bridge.
- `kg_cache_50k.jsonl` covers only 50,000 of 427,001 containers and strips important provenance/identity fields; it is not an adequate canonical retrieval layer.
- Recovered-corpus provenance and `DormantStore` exact-source provenance are different schemas. This is a real architecture boundary, not a naming cleanup.
- The old runtime dormant SQLite body was empty and is archived; recreating another full memory authority would revive the multi-State failure mode.

### ChatGPT consolidator correction to the proposed move

Do **not** copy the 1.02 GiB recovered corpus into a second canonical dormant database merely to reuse `DormantStore`.

Instead, preserve `State/dormant/*.jsonl` as the single dormant memory authority and build a **derived, rebuildable index + read-only corpus adapter**:

1. Bind the index to the current dormant corpus manifest/source hashes.
2. Store derived lookup metadata only (container ID, byte offset/length, exact text hash, lexical postings, graph adjacency, optional later semantic vectors).
3. Retrieve candidate IDs from the derived index.
4. Dereference the exact container/edge record from the authoritative JSONL body.
5. Verify hash/provenance at dereference time.
6. Convert exact evidence into provenance-bearing `FieldSpan`s in canonical `structured_knowledge`.
7. Compile that resulting `SharedFieldSnapshot` through the existing D64 compiler.

The index is a sense organ. It is not a second memory body.

## DELTA 5 — ChatGPT supplemental: tests / ops / performance

Kimi lens 5 failed from quota; this section is ChatGPT evidence.

### Verified strengths

- The tracked repository has 78 `tests/*.py` files and strong focused coverage around D64 compiler/branch/training contracts.
- Existing launch guards prevent accidental canonical-training bypass.
- Current V6 input does not silently truncate: targets beyond configured output bounds fail rather than clipping.

### Verified gaps / risks

- No tracked tests currently cover `scripts/run_kimi_roundtable.py` supervisor behavior; its lock/dirty-tree/model routing evidence is from live supervised probes rather than unit tests.
- No `.github` CI workflow surfaced in the tracked ops inventory, so the full suite is not visibly enforced by hosted CI.
- V6's exact reader retains encoded state for every attended character (`memory_states` then `torch.cat(..., dim=1)`), so exactness is preserved but memory/decoder-attention cost still grows with active character count.
- `D64FieldCompiler.compile()` also builds full row/address arrays, canonical address serialization/hash material, roundtrip text, and a full vector roundtrip. Correctness is excellent; realistic-size performance has not yet been characterized.

### Recommended move

Do not optimize these mechanisms blindly. Add a bounded performance/coverage benchmark after the dormant bridge + D64 driver skeleton are present, using realistic canonical field sizes and recording compiler time, rail bytes, retained decoder memory, and complete-coverage proof cost.

## DELTA 6 — ChatGPT supplemental: stale paths / security / maintainability

Kimi lens 6 failed from quota; this section is ChatGPT evidence.

### Verified strengths

- No tracked `ssh.py`, `ssh_helper.py`, `.pem`, `.key`, or root `.env` credential-bearing files were found.
- Current authority references generally point to `docs/SOURCE_OF_TRUTH.md`.
- Several old paths are clearly preserved as archive/freeze/rejection evidence rather than live runtime configuration.

### Verified gaps / risks

- The stale duplicate root `SOURCE_OF_TRUTH.md` is the clearest live drift hazard.
- Historical `ops/kimi_packets/*.md` contain many references to retired `State/axon_runtime`, `D:\Kimmy`, and old `C:\Users\Jeffg` continuity paths. They are historical packets, but broad grep/search can still feed stale context to an agent unless clearly treated as history.
- `runtime/table/wrappers/hermes_glm.py` contains hard-coded old-user paths and is a real portability/maintainability smell if that wrapper is used again.
- `runtime/council/CONTRACT.md` references the stale root Source of Truth.

### Recommended move

After the core integration packet, do one narrow hygiene pass: Source-of-Truth shim + current contract redirect + explicit historical banners/namespace rules for old Kimi packets + portability cleanup for any still-live wrapper. Do not delete historical acceptance evidence.

## DELTA 7 — ChatGPT supplemental: souls / multi-core / product seams

Kimi lens 7 failed from quota; this section is ChatGPT evidence.

### Verified strengths

- Axon already has serious private-soul machinery: `runtime/axon_runtime/soul_store.py`, checkpoint soul validation, content-addressed soul blobs, generation/freshness checks, crash-safe private-state preparation/install, and `runtime/axon_runtime/core_backend.py` candidate-soul validation.
- `training/soul_load_bearing.py` provides controlled correct/zero/swapped/shuffled causal tests rather than treating lower loss as proof of soul use.
- `training/differentiable_soul_writer.py` is explicitly pilot-only and preserves warm/cold tiers while differentiably writing hot rows.
- `runtime/multi_tick_refiner.py` already provides typed-delta commit/rematerialize/replay structure.
- Advisor/tool/situation/diary regions and advisor provider abstractions exist as useful product-facing organs.

### Verified gaps / risks

- The strong soul/private-runtime backend remains coupled to the old core/view geometry: `core_backend.prepare_action` calls the legacy `compile_next_read_page` path rather than the new D64 rail.
- `runtime/multi_tick_refiner.py` still defaults to the legacy `compile_field_view`.
- The current council path is not the canonical D64 organism, so advisor/tool/diary/situation regions are not yet being maintained by a production-compliant D64 council tick.
- Training the soul writer further before the D64 runtime spine exists would improve a faculty attached to the wrong sensory pathway.

### Recommended move

Do not add new soul complexity yet. Once the non-neural D64 driver contract is proven, adapt the existing private-soul candidate/commit lifecycle to the D64 complete-field action interface, then port first-pass/refinement/consolidator sequencing on top of that one canonical driver.

## Cross-delta convergence

The strongest agreement across the four Kimi deltas and three ChatGPT supplements is:

> **Axon's primitives are ahead of its integration.**

The repository already contains unusually strong exactness, provenance, replay, soul-state, delta-validation, coverage, and causal-evaluation components. The major risk is not lack of mechanisms; it is that several good mechanisms still terminate in different historical anatomies.

The most important remaining integration gaps are:

1. the real dormant body is not surfaced into the canonical D64 field;
2. no production D64 driver makes complete/fresh rail coverage a proposal precondition;
3. canonical training transactions are not yet branch-backed/resume-proven;
4. old council/384-slot paths remain searchable/executable enough to confuse future engineers;
5. exact current mechanisms need realistic-size performance characterization only after the canonical spine is connected.

## ChatGPT consolidated next canonical engineering state

This report does **not** silently change `docs/SOURCE_OF_TRUTH.md`. It establishes the next implementation order in the project continuity/ledger, subject to Jeffrey's authority.

### P0 — Canonical dormant evidence bridge (next packet)

Build a read-only adapter + rebuildable derived index over the **existing** `State/dormant` body. Do not create a second canonical memory corpus.

Acceptance boundary:

- authoritative body remains `State/dormant` recovered JSONL + its manifest;
- index is disposable/rebuildable and manifest-bound;
- candidate lookup dereferences exact authoritative records;
- returned evidence retains container/edge/provenance/hash identity;
- evidence surfaces into a real `SharedFieldSnapshot.structured_knowledge` span;
- the resulting snapshot compiles through `D64FieldCompiler` with complete/fresh coverage;
- one bounded test proves query -> exact dormant dereference -> structured_knowledge -> D64 rail roundtrip;
- no runtime authority, training, service, checkpoint, or corpus rewrite.

### P1 — Non-neural production D64 driver contract

Prove the complete/fresh rail barrier and typed-delta commit transaction with a scripted proposer before attaching a model.

### P2 — Canonical State-training branch + split/resume proof

Make training actually journal its canonical episode branch, then prove exact checkpoint/resume and anatomy mismatch refusal.

### P3 — Doctrine/continuity hygiene

Retire the duplicate root Source of Truth as a shim, repoint the council contract, and harden tests against duplicate live doctrine.

### P4 — Attach existing faculties

Port the existing soul lifecycle, refinement/consolidation, advisor/tool, diary/situation behavior onto the canonical D64 driver. Benchmark realistic-size compiler/reader memory and time before scaling.

## Swarm/model-routing lesson

The first swarm was intentionally ambitious and used more K3 quota than intended. The local Kimi configuration now declares ordinary K2.7 Coding as the default secondary/workhorse model, with K3 reserved for hard architecture/synthesis and Highspeed explicitly non-default. The supervisor enables Kimi's documented secondary-model experiment for child processes. `kimi doctor` validates the configuration.

Behavioral verification of the new secondary-model routing is **pending** because the provider currently rejects new Kimi calls with the billing-cycle usage-limit 403. Future jobs must inspect each worker's persisted `profile.bind.modelAlias` before claiming which model actually ran.

## Final status

- Kimi AgentSwarm: VERIFIED real, one call, seven worker contexts.
- Completed Kimi deltas: 4/7.
- Quota-failed Kimi deltas: 3/7, explicitly not fabricated.
- ChatGPT supplemental deltas: 3/3 missing lenses covered read-only.
- Repository mutation by swarm: none.
- Consolidator recommendation: P0 canonical dormant evidence bridge over the existing memory body; no duplicate memory authority.

**ChatGPT / GPT-5.6 Sol / 2026-08-20**
