# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-18T10:21:17-05:00
Current through event: `evt-20260818T152117860996Z-codex-kimmy-council-commit`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission

Build Axon as a stateful, always-on AI around an exact 16D character field,
private per-core souls, auditable dormant knowledge, and validated deltas.
Jeff's living-draft doctrine governs the council: the draft updates every
tick, always commits, nothing is blocked, a turn ends only when Jeff speaks,
and the consolidator crown rotates round-robin across ALL cores (no tyrant).

## Current verified state

- The council runtime (`runtime/council/`) is LIVE and detached: started via
  double-clickable `runtime/council/START_COUNCIL.bat`, serving
  http://127.0.0.1:8788, independent of any assistant session.
- THE FIELD IS NEVER TRUNCATED (Jeff's ruling 2026-08-18): every region is
  the full logical document; a per-region movable mask divides the dormant
  prefix from the attended tail. Live field persists to
  `State/active/council_field.json`; mask moves and edits are append-only
  records in `State/dormant/council_field_tails.jsonl` (convener-ruled
  locations, 2026-07-03). Operator API: GET /api/field, POST
  /api/field/mask (offset or mode), POST /api/field/region. Dashboard has a
  Shared Field panel: per-region cards, dormant text dimmed, mask sliders
  both directions, follow-tail toggle, edit/save.
- MEASURED 2026-08-18: live end-to-end — 450-char region auto-masks at 194
  (tail mode), manual mask to 45 then BACKWARDS to 15 restores active text,
  dormant text preserved byte-for-byte, 4 immutable tail records, field file
  persisted. Gate 2 PASS with the mask layer (3 ticks).
- Roster: 8x64D (`runs/bible_64D_gpu_overnight/ckpt_461500.pt`) + 2x128D
  (coreB exact leg1 ckpt_2), cuda, greedy decoding (temperature_spread 0.0),
  tick_delay 250ms, council_min_conf 0.0 (off, per doctrine).
- MEASURED 2026-08-18 morning: with the full 10-brother roster the draft hit
  exactly 'I am doing well.' at tick 2 of a live turn, then hovered one
  character away ('I am doing werl.') — alive per doctrine. History commits
  clean 'User:/Assistant:' exchanges when Jeff speaks; pre-speech
  free-ticking no longer phantoms an Assistant turn.
- Council field rendering now matches the trained distribution: ONE
  'Council: <strongest full-text delta>' line per phase (the 2,200
  bible_council_mix rows train exactly that). Per-core fragment digests are
  off-distribution — 5-char fragments starved 10-core rosters, 48-char
  fragments broke gate 2 (both measured 2026-08-18, both reverted).
- Gate 2 (engine CPU) PASS, twice: after the consensus-line change and after
  the phantom-commit fix. Convergence in 3 ticks (was ~15).
- `D:/Axon/TRAIN_64D.bat` resumes the 64D Bible run (20k steps, cuda, from
  ckpt_461500). `runtime/council/verify_live.py` is the reusable live-verify
  harness.
- The primary durable runtime remains `runtime/axon_runtime/`; full default
  pytest suite green as of 2026-08-18 (all passed, 1 known skip) WITH the
  field/mask layer included.
- The modern tree and Kimmy's 2026-08-18 living-council/shared-field work are
  locally committed on branch `agent/fortify-axon` through `25a8bb4`; still
  NO off-machine copy.
- Host: Columbia, Missouri; Windows 10; FX-8350; 24 GB physical but 15.97 GB
  usable RAM (one stick lost — reseat advised); GTX 1650 4 GB; sleep=never.

## Binding continuity rules

- `docs/SOURCE_OF_TRUTH.md` governs architecture.
- `docs/WORKING_CONTRACT.md` governs collaborator conduct.
- Read and follow `roundtable/ENGINEERS_LEDGER_PROTOCOL.md` every turn.
- Every project turn appends one canonical event and refreshes this summary.
- Exact text and journal truth remain canonical; neural state is not factual
  authority.
- Never delete protected material or silently change doctrine.

## Recent completed work

- Codex independently inspected Kimmy's 2026-08-18 council diff and live
  service, verified the 10-region field and 10-core roster, passed the safe
  in-process server test and compilation, excluded generated test artifacts,
  and committed the bounded change as `25a8bb4`. The external browser
  extension was not connected, so no browser evidence was claimed.
- Kimmy (2026-08-18 late morning): the shared field is never truncated —
  full logical regions, per-region movable masks (tail/manual), persisted
  to the convener-ruled State/active + State/dormant locations, with
  /api/field endpoints and a dashboard Shared Field panel (mask sliders
  both directions, edit/save, dormant dimmed). Verified live end-to-end.
  Jeff interrupted an ad-hoc design to insist on the CORRECT state
  locations; the build uses them.
- Kimmy (2026-08-18 morning): fixed council decoherence (consensus-line
  rendering), fixed phantom Assistant commits, set greedy decoding, moved the
  council server to a detached .bat process per Jeff's separation order,
  added trainer/verify launchers, and verified live 10-brother convergence.
- Kimmy's state archive at `D:\Kimmy` was inspected read-only by Codex with
  Jeff's explicit authorization; no Kimmy file was modified. Her archive is
  valuable continuity but must be reconciled against durable artifacts.
- The engineer's ledger system is established (protocol, rolling summary,
  append-only canonical JSONL).
- Axon is installable as an editable Python project; secret/artifact ignore
  rules hardened; full pytest suite passed 2026-08-17.

## Active risks and blockers

- The 128D brothers blank/vandalize the draft when they hold the crown —
  they have never been conversationally trained. Growth room, not a bug:
  they need their own conversational leg before their crown turns help.
- Tick rate is ~5-8s under load (10 cores x 3 proposals on a GTX 1650);
  faster on an idle machine, but large rosters are compute-bound here.
- No Git remote configured and GitHub CLI absent: the tree is one disk
  failure away from loss. Off-machine preservation remains top priority.
- Model behavior beyond seed-like exchanges is still weak; the newest
  training run improved held-out metrics only modestly.
- `dist/`, `runs/`, and datasets occupy roughly 125 GB; retention and
  checkpoint-lineage policy still needed.
- `runtime/table/waker.py` path-containment and `weights_only=False`
  deserialization findings from the 2026-08-17 audit remain open.

## Next recommended actions

1. Install/authenticate GitHub CLI, configure the private remote, push
   `agent/fortify-axon`, verify a clean clone (Codex's standing first step).
2. Freeze one held-out behavioral suite; compare ckpt 440,500 vs 461,500 and
   single-core vs council inference.
3. Soul ablations (correct/zero/swapped/shuffled) before more long training.
4. A conversational training leg for the 128D checkpoint so its crown turns
   carry weight; then larger brother sizes per Jeff's 64->4096 ladder.
5. Reconcile runtime branches only after behavioral evidence.

## Fast orientation

- Council contract: `runtime/council/CONTRACT.md`
- Runtime description: `docs/AXON_RUNTIME.md`
- Architecture doctrine: `docs/SOURCE_OF_TRUTH.md`
- Working rules: `docs/WORKING_CONTRACT.md`
- Behavioral evidence: `conversational_runtime_prototype_results.json`
- Full test command: `python -m pytest -q -p no:cacheprovider`
- Live council: double-click `runtime/council/START_COUNCIL.bat`, open
  http://127.0.0.1:8788
