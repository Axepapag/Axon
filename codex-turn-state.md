# Codex Turn State — DEPRECATED CONTINUITY SHIM

Last current as an active handoff: 2026-07-27.
Deprecated: 2026-08-20.

This root path is retained only because older Kimi/Codex packet scripts reference it.
It is **not current architecture authority** and must not be used to reconstruct present Axon state.

Current project authority is:

1. `AGENTS.md`
2. `docs/WORKING_CONTRACT.md`
3. `docs/SOURCE_OF_TRUTH.md`
4. `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
5. `roundtable/ENGINEERS_LEDGER.md`
6. the tail of `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`

The complete historical contents formerly stored here are preserved verbatim at:

`archive/codex-turn-state-2026-07-27.md`

Important 2026-08-20 boundary: Axon has one living State tree at `D:\Axon\State`. Runtime and training may use isolated branches inside it, but must not create competing top-level runtime/training state roots or disposable truncated core-facing anatomy.

If an older packet conflicts with current Source of Truth or the current engineer ledger, the current authority wins.
