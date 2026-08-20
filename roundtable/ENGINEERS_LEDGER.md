# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-20T15:00:03.1666939-05:00
Current through event: `evt-20260820T200003166693Z-chatgpt-canonical-d64-compiler`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission

Build Axon D2 as one stateful organism around exact canonical character state, heterogeneous internal cores, private per-core souls, auditable dormant memory and validated atomic deltas.

The deterministic **D64 canonical Field Compiler/state adapter is now implemented and published**. The immediate next work is not undirected model training: connect the real dormant memory body through exact-evidence retrieval, then wire a production D64 neural runtime driver to this canonical compiler.

## Binding architecture and convener boundaries

- `docs/SOURCE_OF_TRUTH.md` is architecture authority.
- Exact visible text remains grounded in the frozen 16D character substrate.
- Every logical core pass must cover all currently attended canonical characters; physical paging/packing is compute geometry, never permission to silently omit context.
- One living/durable State root: `D:\Axon\State`. Training isolation belongs beneath `State\training`; runtime and trainer do not own competing State universes.
- A smoke may be small in content/compute; its core-facing anatomy cannot be fake, truncated or disposable.
- Cores/consolidation reason and propose typed deltas; the Field Compiler has no reasoning vote or commit authority. Canonical validation/transaction code remains the commit boundary.
- Dormant memory remains exact/auditable. Derived retrieval indexes may locate evidence but must dereference exact source text/provenance before it becomes reasoning evidence.

## Canonical D64 compiler — implemented 2026-08-20

Published code/docs commit:

- `3e8838d` — `Implement canonical D64 field compiler adapter`

At turn start the worktree already contained an uncommitted, unledgered compiler implementation. ChatGPT treated it as orphaned work rather than discarding it, audited it, completed the policy integration, tested it, and published it.

Core implementation:

- `runtime/field/compiler_d64.py`
  - consumes one immutable `SharedFieldSnapshot`;
  - binds each rail to exact source `field_id` and `tick_id`;
  - packs up to four literal frozen 16D character cells into each 64D physical row;
  - never crosses logical-region boundaries inside a row;
  - represents unused lanes as explicit padding;
  - retains exact row/lane, region position, global active-field position, span ID/position, source and provenance for every valid lane;
  - visits all ten logical regions;
  - fails closed on unsupported attended characters;
  - accepts output only after complete coverage and exact 16D vector/text roundtrip verification;
  - rejects stale rails after canonical field/tick changes;
  - produces deterministic word/sentence/paragraph cartography as structural metadata only.
- `runtime/field/state_branch.py`
  - persists the same `SharedFieldSnapshot` and `FieldDelta` objects used by the core architecture;
  - provides authorized runtime branches under `State\active\branches` and training branches under `State\training\branches`;
  - stores immutable snapshots/deltas, append-only journal events and an atomic HEAD pointer;
  - detects stale commits and serialized-state tampering.
- `runtime/axon_runtime/d64_adapter.py`
  - exposes the canonical compiler/delta/branch contract to runtime orchestration without granting the adapter reasoning authority.
- `training/canonical_d64.py`
  - materializes curriculum source records as real canonical `SharedFieldSnapshot` objects before a core reads them;
  - uses the same `D64FieldCompiler` as runtime;
  - exposes canonical training branches and exact teacher deltas.
- `training/complete_field_64d.py`
  - can consume compiled D64 rails directly;
  - deterministically unpacks exact 16D lanes before the existing V6 per-character neural lift, preserving checkpoint input geometry;
  - performs scratch teacher/runtime update through typed `FieldDelta`, rematerializes the canonical snapshot, recompiles, rereads, then produces the response-draft transaction;
  - counterfactual scratch interventions also pass through canonical snapshots/deltas instead of mutating detached dictionaries.

Important physical interpretation: one packed D64 row is lossless storage for four exact 16D cells. Current V6 deliberately unpacks those lanes before neural attention; the implementation does **not** pretend one Transformer token gives four independently addressable character tokens.

## Trainer execution policy

`training/train_complete_field_64d.py` now defaults to canonical D64 anatomy.

- With no anatomy flag, it requires the real `D:\Axon\State` root and uses `SharedFieldSnapshot -> D64FieldCompiler -> typed FieldDelta` core-facing anatomy.
- `--canonical-d64` remains an optional explicit assertion of the default.
- `--legacy-record-direct` is the only way to use the preserved detached-record V6 mechanism path.
- Canonical trainer workspaces live beneath `State\training\runs`.
- Checkpoints record anatomy mode/compiler schema and refuse incompatible resume.
- `TRAIN_COMPLETE_FIELD_64D_R0.bat` remains a no-training guard; this compiler implementation did not automatically launch GPU work.

The preserved ExactV4 production runtime remains fail-closed because its neural driver can propose after one 384x16 physical view. The next runtime task is a D64 neural driver built on this compiler, not reactivation of ExactV4.

## Canonical State body

One-State reconciliation remains in force:

- `State\dormant` is the real recovered dormant-memory organ and was not rewritten;
- former `State\axon_runtime` and `State\private_curriculum` remain archived beneath `State\archive\pre_canonical_reconciliation_20260820`;
- verified curriculum source copy lives under `State\training\curriculum`;
- regression-only runtime/identity roots live beneath `State\training\regression`;
- root `codex-turn-state.md` is a deprecation shim; July content is archived rather than active continuity.

`State\dormant` remains approximately 1.02 GiB: 427,001 recovered containers and 351,978 readable semantic edges, with no vector index in its manifest. Connecting this exact body to the canonical field is the next major organ integration.

## Learned semantic Field Compiler status

The deterministic compiler is implemented. The proposed learned English-semantic tissue is **not** implemented by this commit.

Still future work:

- learned understanding/ranking of words, sentences, paragraphs, entities and semantic relevance;
- semantic compression views for different D-model widths;
- learned semantic rewrite/decompilation;
- dormant semantic-vector ranking.

These future representations must remain derived/rebuildable. Exact canonical text, dormant records and provenance remain authority; semantic reconstruction error must remain detectable.

## Verification

Final evidence for `3e8838d`:

- initial new compiler/branch/training adapter slice: 18/18 passed;
- final focused compiler + canonical branch + canonical training + V6 suite: 33/33 passed;
- final full repository pytest on the final code: exit code 0, one expected skip;
- touched Python modules compiled with `py_compile`;
- `git diff --check` and staged diff check passed;
- no-flag trainer invocation with an invalid `D:\not-axon-state` root failed before data load with the canonical-State-root error, proving canonical D64 is the default rather than a fallback;
- deterministic echo-round fixture churn produced by full pytest was restored to tracked bytes;
- no model training, checkpoint promotion, service start or dormant-corpus mutation occurred.

## Current Git / collaboration state

- Working branch: `agent/fortify-axon`, publishing to private `origin/main`.
- Previous one-State reconciliation publication: `278b996`.
- Previous reconciliation ledger publication: `a5455d1`.
- D64 compiler implementation: `3e8838d` pushed to private `origin/main`.
- This rolling summary/canonical event will be published as a separate ledger-only commit.
- Connector Git may require per-command `-c safe.directory=D:/Axon`; do not add a global exception.

## MCP bridge / continuity

The local ChatGPT MCP bridge supports `normal|strict|yolo`; live mode was changed to YOLO during the prior reconciliation. Recheck after any bridge restart. YOLO removes bridge click-through approval friction but does not waive Axon contracts or evidence/ledger discipline.

`D:\ChatGPT_State` is ChatGPT's carried external continuity and must be refreshed on every substantive turn. Live repository/ledger evidence remains operational authority over personal continuity.

## Next recommended actions

1. Build a rebuildable local lexical/graph retrieval index over `State\dormant` that returns exact container/span/provenance evidence and surfaces that evidence into canonical `structured_knowledge` before reasoning.
2. Wire a production D64 neural runtime driver to `D64FieldCompiler` with a hard freshness/coverage barrier before proposal finalization.
3. Add a canonical-adapter checkpoint/resume proof before any resumed training campaign.
4. Exercise a bounded canonical D64 mechanism smoke only after the runtime/training adapter path is fully observed; do not resume the old detached training lineage by default.
5. Then design/train learned English semantic compiler tissue over deterministic word/sentence/paragraph cartography without replacing exact character state.
6. Continue updating both the canonical engineer ledger and `D:\ChatGPT_State` every substantive turn.

## Fast orientation

- Architecture authority: `docs/SOURCE_OF_TRUTH.md`
- Reconciliation record: `docs/CANONICAL_STATE_RECONCILIATION.md`
- D64 compiler: `runtime/field/compiler_d64.py`
- Canonical State branch: `runtime/field/state_branch.py`
- Runtime D64 adapter: `runtime/axon_runtime/d64_adapter.py`
- Training adapter: `training/canonical_d64.py`
- V6 reader/model: `training/complete_field_64d.py`
- Trainer: `training/train_complete_field_64d.py`
- Dormant authority: `State/dormant/`
- Full tests: `python -m pytest -q -p no:cacheprovider`
- Ledger protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

**ChatGPT / GPT-5.6 Sol / 2026-08-20**
