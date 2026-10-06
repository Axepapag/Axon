# Axon

An exact-character, stateful AI. This is the **new** repository, started 2026-10-05. The old one is untouched
at `G:\My Drive\Projects\Axon_old\Axon_old`; this repo carries only what was good in it.

If you are Jeff: nothing here needs Python editing. If you are an agent: read `AGENTS.md` first.

## What Axon is, in five lines

- **Exact text only.** Every character is one of 95 frozen native characters. Anything else (emoji, tabs, accents)
  fails closed. There is no byte transport and no silent conversion.
- **Frozen substrate.** Each character has one fixed 16D code. A fixed 1024D substrate carries 64 characters per
  vector (64 lanes x 16 floats). Neither is ever learned or tuned.
- **The Heart is the only writer.** It guards the Shared Field (the canonical, exact text body). Cores propose;
  the Heart validates and commits.
- **Cores are recurrent (GRU first)**, with a private layered Soul, and read an exact mirror of the field.
- **No rails, no packing, no D64.** The Heart serves exact 16D cells; wider vectors are built from them mechanically.

The full picture of the new direction, and what is still undecided, is in `docs/DIRECTION_2026-10-05.md`.

## What works today (verified 2026-10-05)

| Piece | Where | State |
|---|---|---|
| 16D substrate, exactly 95 characters, fail-closed | `substrate/substrate.py`, `substrate/native.py` | working, frozen vectors bit-identical to the old repo |
| 1024D substrate (64 lanes x 16), sealed by SHA-256 | `substrate/substrate_1024.py` | working |
| Shared Field: regions, masks, deltas, branches (95-only) | `runtime/field/` | working |
| Heart: authority, valve, ingress, lease, masks, turns, autobiography | `runtime/heart/` | working (no heartbeat/coordinator yet) |
| Soul: layered, evidence-gated, content-addressed store | `runtime/soul/` | working |
| Dormant: exact evidence index and retrieval | `runtime/dormant/` | working |
| Character audit for curricula | `tools/audit_characters.py` | working |

**Not built yet:** the heartbeat/tick and core registry, the multi-core output region, the wiring of the GRU core into the
Heart path, and the trainer engine and window. (The E0 GRU core exists as component manifests plus a PyTorch adapter
skeleton under `core/`, tested but not yet integrated.)

## Check that everything is healthy

From this folder:

    python -m pytest
    python substrate/substrate.py --quiet
    python substrate/substrate_1024.py --quiet

All three should finish without errors (pytest prints the number passed; the other two print nothing and exit 0).

## Folder map

| Folder | What it is |
|---|---|
| `substrate/` | The frozen 16D and 1024D substrates and their sealed reference files |
| `core/` | Core-owned E0 component manifests and the two-state GRU adapter skeleton (unintegrated) |
| `runtime/` | `field/`, `heart/`, `soul/`, `dormant/` and the capacity policy loader |
| `curator/` | Importer that loads recovered legacy memories into Dormant |
| `tools/` | Small standalone tools (character audit) |
| `tests/` | The test suite |
| `curricula/` | Legacy curricula (raw, git-ignored) and `CURRICULA_AUDIT.md` |
| `reference/` | Old trainer and v6 code, **read-only reference, never imported** |
| `history/` | Records of the old repo, axon7 and v6 (ledger, old doctrine, audits) |
| `docs/` | Direction, carry manifest, working contract |
| `roundtable/` | Engineer's Ledger (continuity record) and its protocol |

`docs/CARRY_MANIFEST.md` lists exactly what was carried from where, what was edited, and what was left behind.
