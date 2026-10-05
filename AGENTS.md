# Axon Agent Instructions

These apply to every human or agent working in this repository.

**Jeff (the convener) is not a programmer.** Ordinary operation must never require editing Python. Offer clear
menus or buttons, understandable defaults, and plain-language reports. Say what you verified and what you assumed.

## Read first, in this order

1. `README.md` and `docs/DIRECTION_2026-10-05.md` (current direction and open decisions).
2. `docs/WORKING_CONTRACT.md` (inherited from the old repo; Jeff's word overrides it, and the HALT AND FLAG rule applies).
3. `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`, then `roundtable/ENGINEERS_LEDGER.md` (rolling summary) and the tail of
   `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`.

## Standing laws (Jeff, 2026-10-05)

- The substrate is exactly the 95 native characters. Nothing outside them is ever admitted, converted, or escaped
  inside the system. Conversion of outside text is an explicit tool that writes a new file and a report.
- The frozen 16D and 1024D substrates and their sealed reference files are never retuned or regenerated.
- No D64 rails, no packing of cells into model-width rows, no learned projection of the substrate, no continuous
  D16 port.
- The Heart is the sole writer of canonical state. Cores propose; they never write the Shared Field directly.
- The trainer must use the same Heart / field / Dormant / Soul path as runtime.
- Never delete: move, archive, or leave in place and report. The old repo (`Axon_old`), axon7 and ashes_v6 are
  read-only sources.

## Every turn

Follow the ledger protocol: append exactly one canonical event (`scripts/append_engineers_ledger_event.py`), never
edit an existing ledger line, and update the rolling summary before ending the turn. Read-only work is still work.

## Before reporting success

Run `python -m pytest` and the two substrate self-tests listed in `README.md`.
