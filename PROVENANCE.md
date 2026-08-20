# Axon Provenance

## Archive Sources

- `D:\AxonGliksbot` and `D:\axon7` are protected historical source archives.
- `D:\Axon` is the clean active repository.
- Historical material may inform design, but active implementation must match `docs/SOURCE_OF_TRUTH.md` and the current Engineer's Ledger.

## Day Zero Active Lines

| Area | Current status |
|---|---|
| `substrate/` | Frozen exact 16D character alphabet remains bedrock. |
| `runtime/field/` | Canonical ten-region snapshot, typed delta, deterministic D64 compiler, and canonical State branch contracts. |
| `runtime/axon_runtime/d64_adapter.py` | Runtime-facing authority-free D64 adapter. No production neural runtime driver exists yet. |
| `training/canonical_d64.py` | Canonical curriculum-to-`SharedFieldSnapshot` materialization and training branch adapter. |
| `training/complete_field_64d.py` | Current 64D neural mechanism reading only canonical compiled D64 rails. |
| `training/train_complete_field_64d.py` | Canonical-only trainer; run/checkpoint workspace must live under `State/training/runs`. |
| `curator/` | Offline exact dormant-corpus schema/materialization/building utilities retained for the canonical memory body. |
| `scripts/run_kimi_roundtable.py` | Supervised engineering orchestration only; not Axon's cognitive runtime. |

## Day Zero Historical Namespace

Superseded tracked implementations are preserved under:

`archive/day_zero_legacy_2026-08-20/`

Superseded local State/run/dataset/checkpoint/distribution artifacts are preserved under:

`State/archive/day_zero_legacy_20260820/`

Those archives are evidence, not active anatomy. They may not be imported, launched, resumed, or promoted without an explicit convener decision and Source-of-Truth reconciliation.

## Policy

Do not reintroduce a parallel runtime, council State body, detached training anatomy, 384-slot core-facing view, or duplicate doctrine authority. Exact canonical text remains visible through the frozen character substrate, derived D64 rails remain reversible/provenance-bound, and the canonical validator/transaction boundary remains the only path to committed field state.
