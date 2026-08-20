# Canonical State Reconciliation

Status: active engineering boundary
Date: 2026-08-20
Convener: Jeff

## Purpose

Axon has one living durable state root: `D:\Axon\State`.

A small experiment may use less content and less compute, but it may not invent a substitute anatomy. Runtime and training must converge on the same canonical field, dormant-memory, compiler, typed-delta, and commit interfaces.

This document records implementation status. It does not promote the still-discussed semantic Field Compiler behavior into locked doctrine.

## Canonical layout

The intended top-level layout is:

- `State\active\` — the current canonical shared-field transaction state.
- `State\dormant\` — the exact dormant memory body and rebuildable retrieval indexes.
- `State\souls\` — current private per-core soul state only.
- `State\training\` — copy-on-write/isolation branches for training through the same canonical interfaces.
- `State\archive\` — immutable prior state lineages retained as evidence, never treated as current.
- `State\transfer\` — explicit transfer/import staging only.

Training curricula belong beneath `State\training\curriculum\`; a curriculum is not a second canonical runtime state.

## Verified discrepancies found before reconciliation

1. `State\dormant` contains the real recovered dormant body: 427,001 containers and 351,978 readable semantic edges, about 1.02 GiB total.
2. `State\axon_runtime\dormant.sqlite3` was a separate runtime dormant database with zero substantive records.
3. `State\axon_runtime\runtime.sqlite3` held a separate generation-4/tick-4 runtime head whose ten regions contained zero spans.
4. `State\active\council_field.json` represented an older council lineage, independent of that runtime database.
5. The ExactV4 runtime bootstrap installed a 4,096-character budget projector before core observation.
6. More seriously, the ExactV4 driver executed one 384x16 physical view per core action and could propose before a complete logical field sweep.
7. The newer V6 D64 trainer does complete coverage-proven paging, but its CLI still trained directly from detached JSON examples rather than a canonical State branch.
8. The active V6 curriculum builder still read historical D00 sources directly, and curriculum artifacts lived at the top-level `State\private_curriculum` path.

## Immediate cleanup boundary

Until the D64 canonical compiler/runtime adapter is implemented:

- Production `runtime.axon_runtime.bootstrap.materialize_runtime()` fails closed on the preserved ExactV4 runtime path.
- Focused injected-dependency tests may continue exercising that preserved runtime foundation as regression evidence.
- `training/train_complete_field_64d.py` refuses detached-record training unless `--legacy-record-direct` is explicitly supplied.
- `training/verify_complete_field_v6_resume.py` uses the same explicit legacy opt-in for optimizer-step verification.
- `training/build_complete_field_r0_curriculum.py` refuses direct D00 rebuilding unless `--legacy-d00-source` is explicitly supplied.
- `TRAIN_COMPLETE_FIELD_64D_R0.bat` is a non-training guard launcher. It cannot start a GPU run.

These guards are intentional. They prevent a future session from spending compute on an interface already known to be temporary.

## Completed physical reconciliation — 2026-08-20

The live State surface now reflects the one-State rule:

- the former `State\axon_runtime` tree was moved intact to `State\archive\pre_canonical_reconciliation_20260820\axon_runtime`;
- the former `State\private_curriculum` tree was moved intact to `State\archive\pre_canonical_reconciliation_20260820\private_curriculum` after its 36 files / 39,030,916 bytes were copied to `State\training\curriculum` and verified byte-for-byte with zero differences;
- `State\dormant` remained in place and was not rewritten;
- the preserved ExactV4 CPU descriptor now writes only to `State\training\regression\exact_v4_runtime_smoke`, so running regression tooling cannot recreate a top-level competing `State\axon_runtime` tree;
- the identity-v2 sample contracts now declare only `State\training\regression\identity_v2*` roots, are rebound to the relocated smoke-config ID, and the parser rejects any in-repository identity-v2 root outside `State\training\regression`;
- the stale root `codex-turn-state.md` is now a deprecation shim pointing to current authority, while its complete July contents are preserved at `archive\codex-turn-state-2026-07-27.md`;
- `State\README.md` now documents the canonical top-level anatomy and archive boundary.

The repository-wide test suite passed after these guards and path changes with exit code 0 and one expected skip. Known echo-round fixture churn produced by the suite was restored to tracked bytes before publication.

## D64 next implementation target

The next active runtime/training work is one D64 canonical branch/compiler interface that:

1. reads a `SharedFieldSnapshot` bound to one exact `field_id`;
2. covers every currently active canonical character before a D64 core may finalize a proposal;
3. surfaces retrieved dormant evidence as exact provenance-bearing field spans before reasoning;
4. gives training a copy-on-write State branch rather than a detached field dictionary as its authority;
5. commits only through typed validated deltas/atomic canonical transactions;
6. leaves the Field Compiler without reasoning or commit authority.

The later learned semantic compiler remains a design question. Deterministic exact structure, source spans, hashes, region identity, and reversible 16D character access should not be learned merely to prove that they can be learned.

## Semantic compiler discussion boundary

A future learned component may understand words/sentences/paragraphs and present a semantic D64-friendly representation, and may help decompile a proposed semantic edit back to exact characters. However:

- learned semantics may not become the sole copy of canonical text;
- a generated word such as `dog` ultimately has to resolve to exact canonical `d`, `o`, `g` character cells;
- semantic reconstruction errors must remain detectable and auditable;
- the consolidator remains the reasoning authority;
- the runtime validator remains the atomic commit authority.

No semantic-lossy compiler is authorized by this cleanup alone.
