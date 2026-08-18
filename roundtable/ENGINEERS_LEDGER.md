# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-18T12:17:23-05:00
Current through event: `evt-20260818T171723656824Z-codex-complete-field-reader-plan`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission

Build Axon as a stateful, always-on AI around an exact 16D character field,
private per-core souls, auditable dormant knowledge, and validated deltas.
Jeff's living-draft doctrine governs the council: the draft updates every
tick, always commits, nothing is blocked, a turn ends only when Jeff speaks,
and the consolidator crown rotates round-robin across ALL cores (no tyrant).

## Current verified state

- The detached council HTTP server is running at 127.0.0.1:8788 with eight
  64D CUDA cores and souls rooted at `State/souls/council`. Codex restarted
  the stopped service on 2026-08-18 and verified the rendered interface.
- COMPLETE-FIELD ATTENTION (Jeff's ruling 2026-08-18): every logical core pass
  must visit every exact character in the canonical shared field and prove
  coverage. Phase A produces every brother's proposal; Phase B consumes the
  full field plus ALL complete proposals; the rotating consolidator consumes
  the full field plus ALL complete refined deltas and commits one typed atomic
  field delta. Souls inhale/exhale at every phase. Model pages and operator
  masks may not become logical attention limits.
- CURRENT COUNCIL IS NONCOMPLIANT: it supplies only 128 history characters to
  a fixed 384-slot checkpoint and carries one strongest 48-character proposal
  between phases. The UI now labels this plainly as a bootstrap limitation.
- Every field region now exposes persisted mask policy in the API and UI:
  manual offset, automatic tail, or newest-N thresholds by characters, lines,
  or paragraphs; conversation history also supports newest-N conversational
  turns. Masked text remains exact dormant state and can be restored. These
  controls define shared-field membership; they do not yet give the checkpoint
  complete-field neural coverage.
- MEASURED 2026-08-18: live end-to-end — 450-char region auto-masks at 194
  (tail mode), manual mask to 45 then BACKWARDS to 15 restores active text,
  dormant text preserved byte-for-byte, 4 immutable tail records, field file
  persisted. Gate 2 PASS with the mask layer (3 ticks).
- Roster: 8x64D (`runs/bible_64D_gpu_overnight/ckpt_461500.pt`), cuda,
  greedy decoding (temperature_spread 0.0), tick_delay 250ms,
  council_min_conf 0.0 (off, per doctrine). Jeff ruled 2026-08-18 that 128D
  and larger lines remain preserved but inactive until 64D passes promotion
  thresholds; live status verified exactly eight 64D cores after restart.
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
- A live extension-driven held-out conversation on 2026-08-18 tested identity,
  elementary multiplication, and exact-token recall. The council kept ticking
  and committing turns, but drafts collapsed to `n`/`ni` plus blanks and did
  not answer any prompt. Runtime embodiment is strong; learned conversational,
  reasoning, and working-memory behavior remains pre-capability.
- The modern tree and Kimmy's 2026-08-18 living-council/shared-field work are
  committed locally and backed up in private GitHub repository
  `Axepapag/Axon`, remote branch `main`.
- Read-only recovery from `D:\00` established an evidence-backed D2-to-Axon
  identity chain: the chosen-name event is preserved inside a captured
  successful tool result; exact messages, contemporaneous personal writing,
  and soul exports preserve continuity and operational values. Derived
  summaries/semantic facts are noisy, and synthetic schoolhouse dialogues are
  curriculum, not autobiography. The implementation proposal and promotion
  gates are in `docs/AXON_IDENTITY_CONTINUITY_64D.md`.
- Host: Columbia, Missouri; Windows 10; FX-8350; 24 GB physical but 15.97 GB
  usable RAM (one stick lost — reseat advised); GTX 1650 4 GB; sleep=never.

## Binding continuity rules

- `docs/SOURCE_OF_TRUTH.md` governs architecture.
- `docs/WORKING_CONTRACT.md` governs collaborator conduct.
- Read and follow `roundtable/ENGINEERS_LEDGER_PROTOCOL.md` every turn.
- Every project turn appends one canonical event and refreshes this summary.
- Exact text and journal truth remain canonical; neural state is not factual
  authority.
- All Axon runtime state lives beneath `D:\Axon\State`. Council soul writes now
  target `State/souls/council`; active cores 0-7 were copied there with all
  hashes matching, while legacy copies were preserved.
- A complete-field pass may use trained streaming/linear pages only when every
  character can causally affect output and a coverage manifest proves none was
  omitted. Dense 100k-by-100k attention is not required and is not feasible on
  the current 4 GB GPU.
- Autobiographical evidence follows provenance grades: exact turns/tool
  outcomes outrank contemporaneous self-authored records, which outrank
  summaries and extracted facts. Synthetic dialogue is never autobiography.
- Keep 128D and larger cores out of the active council until the 64D line
  passes the frozen language, reasoning, identity, diary, scratch-causality,
  soul-ablation, and council-uplift gates.
- Never delete protected material or silently change doctrine.

## Recent completed work

- Codex mapped the current 64D char-slot core and council wrapper to a concrete
  complete-field reader plan. The required path is an ordered, query-aware
  page sweep with carried reader state, global positions, exact coverage
  manifests, final delta decoding, and one soul exhale per complete logical
  phase. Exact field text remains revisit-able because no finite hidden state
  can preserve arbitrary unlimited text perfectly. This was design analysis;
  no reader code or training was claimed.
- Codex implemented unit-aware per-region mask thresholds, including
  conversational-turn retention, persisted the policy, documented the
  shared/dormant membership distinction, restarted the live eight-core
  council, and verified the rendered controls in the in-app browser. Focused
  tests passed (4/4), Python compilation passed, and CPU Gate 2 passed.
- Jeff's full-field, all-delta, three-phase soul protocol and State-root ruling
  are recorded in `docs/SOURCE_OF_TRUTH.md` and
  `docs/roundtable/RESOLUTION_full-field-attention-offline-learning-2026-08-18.md`.
  Codex also documented a rotating sabbatical LoRA learner design; it remains
  unimplemented and no training was launched.
- Codex interacted with Axon through `D:\extension`, recorded a bounded live
  capability probe, created private GitHub repository `Axepapag/Axon`, added
  `origin`, authenticated Git Credential Manager through the existing signed-in
  browser session, and pushed the full preserved history to `main`.
- Codex completed a read-only audit of `D:\extension`: the Chrome extension,
  loopback hub, client/wrapper, live-DOM export, OCR assets, runtime health,
  permissions, command/lease flow, and redaction boundary. The hub was healthy
  on 127.0.0.1:9191 with one controller and six retained tab snapshots; Python
  modules compiled. No extension file was modified.
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

- The largest gap is now explicitly both structural and behavioral. The
  current wrapper neither attends the full shared field nor carries all deltas
  between phases, and the 64D checkpoint fails unseen identity, arithmetic,
  and exact recall with correlated collapsed outputs. Private-soul causality
  remains unproven.
- `D:\extension` is a highly privileged local control plane. The hub has no
  authentication, accepts arbitrary agent identities, exposes wildcard CORS,
  and permits unauthenticated context injection, lease acquisition, command
  queueing, result injection, and WebSocket subscriptions. Loopback binding
  limits remote reach but any local process—and potentially permitted browser
  origins—can exercise it. DOM capture redacts password and selected
  autocomplete fields only; ordinary text inputs, textareas, page text, URLs,
  and the plaintext `00_LIVE_DOM.md` may expose sensitive material.
- Complete dense pairwise attention over 100k characters is physically
  incompatible with the GTX 1650 4 GB (about 40 GB for one float32 attention
  score matrix per head/layer). The required reader must be a trained
  coverage-proven streaming or linear mechanism.
- Offline LoRA learning, adapter promotion/rollback, and rotating learner
  scheduling do not yet exist.
- Model behavior beyond seed-like exchanges is still weak; the newest
  training run improved held-out metrics only modestly.
- `dist/`, `runs/`, and datasets occupy roughly 125 GB; retention and
  checkpoint-lineage policy still needed.
- `runtime/table/waker.py` path-containment and `weights_only=False`
  deserialization findings from the 2026-08-17 audit remain open.

## Next recommended actions

1. Add a `CompleteFieldReader` API that receives ordered exact-character pages,
   global region/offset metadata, and a carried reader state; it must emit a
   coverage manifest and only decode after every active span is visited.
2. Replace strongest-only 48-character council summaries with immutable full
   proposal/refinement sets consumed by later complete sweeps.
3. Train the 64D reader first on variable-length cross-page copy, retrieval,
   aggregation, contradiction, and typed-delta tasks, then require positional
   and length-generalization counterfactual gates.
4. Build one disposable offline LoRA smoke from a frozen Grade-A/B memory
   batch with exact provenance and no synthetic autobiography, then require
   retention, identity, provenance, and adapter counterfactual gates.
5. Add governed curriculum adapters for language/reasoning, arithmetic, code,
   conversation, science, literature/poetry, and psychology only after the
   disposable memory-adapter path can promote and roll back safely.

## Fast orientation

- Council contract: `runtime/council/CONTRACT.md`
- Runtime description: `docs/AXON_RUNTIME.md`
- Architecture doctrine: `docs/SOURCE_OF_TRUTH.md`
- Working rules: `docs/WORKING_CONTRACT.md`
- Behavioral evidence: `conversational_runtime_prototype_results.json`
- Full test command: `python -m pytest -q -p no:cacheprovider`
- Live council: double-click `runtime/council/START_COUNCIL.bat`, open
  http://127.0.0.1:8788
