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

## Current execution boundary

The deterministic D64 canonical compiler/adapter is now implemented. The old
ExactV4 neural runtime is still not the active organism:

- Production `runtime.axon_runtime.bootstrap.materialize_runtime()` continues to
  fail closed on the preserved ExactV4 driver because it can propose before a
  complete logical-field sweep.
- `runtime.field.D64FieldCompiler` and
  `runtime.axon_runtime.CanonicalD64RuntimeAdapter` are the runtime-facing exact
  D64 compiler boundary; the production D64 neural/soul/round-table driver is
  still to be wired.
- `training/train_complete_field_64d.py` defaults to the canonical anatomy.
  Curriculum source records are materialized as `SharedFieldSnapshot` state and
  read through the shared compiler; `--canonical-d64` is an optional explicit
  assertion, while `--legacy-record-direct` is retained only for preserved
  mechanism work.
- Canonical trainer launches are bound to the real `D:\Axon\State` and create a
  run workspace below `State\training\runs`.
- `training/verify_complete_field_v6_resume.py` remains a legacy V6 optimizer
  mechanism verifier until a canonical-adapter resume proof is added.
- `training/build_complete_field_r0_curriculum.py` still requires
  `--legacy-d00-source` for direct D00 rebuilding; curriculum files are source
  material, not core-facing state.
- `TRAIN_COMPLETE_FIELD_64D_R0.bat` remains a non-training guard launcher. No
  GPU training is automatically authorized by this adapter implementation.

These boundaries prevent compute from silently falling back to obsolete anatomy.

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

## D64 deterministic adapter — implemented 2026-08-20

The exact shared adapter now exists:

1. `runtime/field/compiler_d64.py` compiles one immutable `SharedFieldSnapshot`
   into a source-field-bound D64 rail.
2. Each physical row carries up to four literal frozen 16D cells from exactly
   one logical region; padding lanes are explicit and every valid lane retains
   exact row/lane/source-span/region/global provenance.
3. All ten regions are visited and compilation fails closed unless complete
   active-character coverage and exact 16D vector roundtrip both succeed.
4. Rails are stale as soon as the canonical `field_id`/`tick_id` changes.
5. `runtime/field/state_branch.py` persists the same canonical snapshots/deltas
   with append-only journaling and atomic HEAD updates for authorized runtime or
   training branches.
6. `runtime/axon_runtime/d64_adapter.py` exposes this compiler to runtime without
   granting it decision or commit authority.
7. `training/canonical_d64.py`, `training/complete_field_64d.py`, and the active
   trainer path materialize curriculum source material into canonical snapshots,
   read exact characters from the D64 rail, and perform scratch/response teacher
   commits and interventions through ordinary typed `FieldDelta` objects.
8. The current V6 neural reader still reasons at one exact character token per
   position: it deterministically unpacks rail lanes and applies its existing
   frozen lift. Literal D64 packing is therefore exact storage/addressability,
   not a false claim that one packed row is already four independent attention
   tokens.

The next active architecture work is to connect the real `State\dormant` body
through a rebuildable local retrieval index that surfaces exact provenance-
bearing evidence into canonical active state, then wire a production D64 neural
runtime driver to this compiler. The later learned English-semantic compiler is
still a separate design/training question.

## Semantic compiler discussion boundary

A future learned component may understand words/sentences/paragraphs and present a semantic D64-friendly representation, and may help decompile a proposed semantic edit back to exact characters. However:

- learned semantics may not become the sole copy of canonical text;
- a generated word such as `dog` ultimately has to resolve to exact canonical `d`, `o`, `g` character cells;
- semantic reconstruction errors must remain detectable and auditable;
- the consolidator remains the reasoning authority;
- the runtime validator remains the atomic commit authority.

No semantic-lossy compiler is authorized by this cleanup alone.
