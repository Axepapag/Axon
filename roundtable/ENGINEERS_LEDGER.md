# Axon Engineer's Ledger - Rolling Summary

Updated: 2026-10-05T14:22:08.149418+00:00
current_through_event_id: `evt-20261005T142208149418Z-copilot-architecture-choice-advice`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` (this repo, starts at the genesis event above).
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`. The old repo's ledger (470 events, last
`evt-20261005T114814882948Z-copilot-substrate-1024-lane-law`) is preserved in `history/old_axon/` and by path and SHA-256
in the genesis event.

Identity stamp: GitHub Copilot / Claude Sonnet 5.5 / 2026-10-05 UTC

**Start here next session: `docs/HANDOFF_2026-10-05.md`** (state, honest assessment, open decisions, milestones, database safety).

## Mission and state (2026-10-05)

New Axon repository started after the D: drive loss. Jeff (not a programmer; wants a GUI trainer with buttons) is moving to:
exactly 95 native characters (nothing else, fail closed), a frozen 16D and a frozen 1024D substrate (64 lanes x 16), no
D64 rails / packing / continuous D16 port, the Heart as sole writer serving exact 16D cells, GRU cores (1024 first) with a
mirror of the field, a layered Soul and an FFN, a new multi-core output region and a round-robin consolidator. Full
statement and open decisions: `docs/DIRECTION_2026-10-05.md`.

## Binding invariants

Heart is the sole writer; cores propose. The 95-character law is enforced in the canonical body (`FieldSpan`). Frozen
substrates and their sealed reference files are never retuned. The old repo, axon7 and ashes_v6_history are read-only
sources. Never delete; archive or leave in place and report.

## Done this turn

Carried the clean parts of the old repo (see `docs/CARRY_MANIFEST.md`): substrate (16D fail-closed + native API + 1024D),
Shared Field (schema, deltas, branches, native D16 view), Heart (authority, valve, ingress, durable ingress, lease,
identity, health, masks, turns, autobiography), Soul, Dormant, the legacy-memory importer and ledger tooling. Left behind
the rail/D64/transport code and the rail-shaped Heart modules (rewrite later). Copied history, reference trainer/v6 code
and 37 raw curricula with a character audit.

## Verified

`python -m pytest`: 144 passed. `python substrate/substrate.py --quiet` and `python substrate/substrate_1024.py --quiet`
exit 0. All 96 frozen 16D vectors are bit-identical to the old repo. Carried files compared by SHA-256 (34 identical, 21
edited, 2 new).

## Open flags

1. 1024D lane layout reuses the 16D codes: ASSUMED, unconfirmed. 2. Source of truth not carried as live doctrine; needs a
Jeff-approved restatement. 3. Dormant keeps non-95 originals byte-exactly (decision pending). 4. No heartbeat, registry,
transaction layer, multi-core region (field v5), GRU core or trainer yet. 5. Recovered memory databases in
`New folder (2)` not opened; the 59,875-record corpus is absent. 6. GTX 1650 (4 GiB); official Mamba needs Ampere.

## Paths and commands

Repo `G:\My Drive\Projects\Axon`; old repo `G:\My Drive\Projects\Axon_old\Axon_old`; axon7 and v6 under
`G:\My Drive\New folder (2)\`. Health check: `python -m pytest`. Append a ledger event:
`python scripts/append_engineers_ledger_event.py <pending-event.json>`. Character audit: `python tools/audit_characters.py <path>`.

## Next

Jeff answers `docs/DIRECTION_2026-10-05.md` section 3 (items 1-4); then the new heartbeat/registry/transaction layer and the
multi-core output region; then the GRU core; then the trainer engine and window.