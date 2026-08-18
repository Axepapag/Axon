# Phase B Packet 001: isolated v3 contracts and strict serde

Implement the first neural-free v3 protocol slice in new source/test files.
Do not integrate it into the current store, engine, bootstrap, configuration,
package exports, runtime state, or training code in this packet.

## Required source files

- `runtime/axon_runtime/v3_contracts.py`
- `runtime/axon_runtime/v3_serde.py`

## Required test file

- `tests/test_axon_runtime_v3_contracts.py`

## Protocol decisions that are already resolved

1. Protocol version is the exact integer `3` and the canonical protocol name is
   `axon-runtime-protocol-v3`.
2. Every record is an immutable frozen dataclass, Torch-free, canonical-JSON
   safe, content-addressed, and validated fail-closed at construction and
   deserialization.
3. Preserve author order only where the ordered online ring is semantic.
   Content-addressed sets and mappings must serialize deterministically.
4. A working tick has an ordered online population and exactly one distinct
   offline core. The consolidator must be one of the online cores.
5. Phases are `INITIAL`, `REFINE`, and `FINAL`.
6. INITIAL and REFINE each require exactly one pass per online core. FINAL is
   authored only by the consolidator.
7. All pass deltas are authored against the same working field W. Boards are
   sealed sets; they never imply sequential application of competing deltas.
8. A read cycle records ordered page/view hashes and must explicitly say
   whether selected projection coverage is complete.
9. A soul transition is one immutable artifact. Its later outcome is a
   separate `SoulTransitionDispositionRecord`; do not create a duplicate
   rejected-transition subtype.
10. Supported dispositions are `COMMITTED`, `ACCEPTED_NOOP`,
    `REJECTED_NOT_INSTALLED`, `QUARANTINED`, and `SUPERSEDED`.
11. Structurally invalid raw material is described by an
    `ArtifactRejectionRecord`; malformed material must not become a valid pass,
    board, transition, or commit.
12. V2 types remain untouched.

## Required record families

Define and strictly validate:

- `TickPhasePlan`
- `ReadCycleManifest`
- `SoulTransitionRecord`
- `SoulTransitionDispositionRecord`
- `CorePassRecord`
- `ProposalBoardManifest` with a phase field supporting INITIAL and REFINE
- `ConsolidationRecord`
- `TickCommitRecordV3`
- `ArtifactRejectionRecord`

Use schema strings that include `v3` and derived IDs with stable descriptive
prefixes. Reuse the repository's canonical hashing functions and validation
style where appropriate, without importing private helpers from
`contracts.py`.

## Minimum invariants

- Exact non-negative integer validation rejects booleans.
- SHA-256 fields require exactly 64 hexadecimal characters and normalize case.
- IDs and core names are non-empty.
- Populations contain no duplicates.
- Offline core is not online.
- Consolidator is online.
- Pass phase cannot be FINAL.
- Board phase cannot be FINAL.
- Board required authors exactly match its pass-author mapping.
- Board pass IDs and author mapping are deterministic and complete.
- REFINE passes require their initial-board parent.
- INITIAL passes cannot claim a board parent.
- Soul transition phase/substep and optional board parent are mutually valid.
- FINAL consolidation references the complete REFINE board and is authored by
  the plan's consolidator.
- Tick commit references the plan, both boards, all initial/refine passes, the
  consolidation, final per-online-core state leaves, field transaction audit,
  and output field.
- Only the final consolidation/commit representation may carry invocation
  request IDs. Passes and boards have no invocation field.
- Deserializers reject unknown/missing/extra keys, wrong schema strings,
  derived-ID mismatches, noncanonical population order where order is
  semantic, and every construction invariant.

If a proposed field is ambiguous, choose the smallest representation that can
prove these invariants and document that choice in the module docstring. Do not
invent model, trainer, scheduler, database, or effect behavior.

## Required tests

Test at least:

- deterministic canonical dictionaries and IDs
- strict round-trip for every record family
- hash/ID tampering rejection
- missing/extra key rejection
- duplicate online core rejection
- offline-core participation rejection
- wrong consolidator rejection
- missing/duplicate board author rejection
- INITIAL/REFINE parent-rule rejection
- incomplete read-cycle representation remains representable but cannot be
  used by a valid pass/commit
- wrong soul-transition parent/substep rejection
- non-final invocation authority is structurally impossible
- semantic rejection is represented through a disposition record
- structural invalidity uses an artifact rejection record

## Validation commands

Run:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_v3_contracts.py
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_contracts.py tests/test_axon_runtime_transaction_serde.py
python -m compileall -q runtime/axon_runtime/v3_contracts.py runtime/axon_runtime/v3_serde.py
```

Stop after this packet. Do not begin store integration.
