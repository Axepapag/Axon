# Axon Agent Instructions

These instructions apply to every human or agent working anywhere in this
repository.

Before doing project work:

1. Read `docs/WORKING_CONTRACT.md`.
2. Read `docs/SOURCE_OF_TRUTH.md` for architectural doctrine.
3. Read `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`.
4. Read `roundtable/ENGINEERS_LEDGER.md` to regain current context.
5. Inspect the tail of `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` for the
   latest immutable events.

During and after every turn:

- Follow the ledger protocol.
- Record every material action in one canonical turn event.
- Update the rolling summary before ending the turn.
- Never edit, reorder, compact, redact, or delete an existing canonical
  ledger line. Corrections are new events.
- A read-only investigation is still work and must be recorded.
- If a turn aborts or fails, record the attempted actions and failure.

The canonical historical ledger is append-only. Git conflicts in that file
must be resolved by preserving every valid event, never by choosing one side
or rewriting existing events.
