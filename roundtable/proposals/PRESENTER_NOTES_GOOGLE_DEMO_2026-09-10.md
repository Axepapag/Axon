# Google Demo Presenter Briefing — Jeff — 2026-09-10

Identity stamp: Hermes / glm-5.3:cloud / 2026-09-09
Companion to: `roundtable/proposals/KIMI_GOOGLE_ORGAN_DEMO_RUNBOOK_2026-09-10.md` (commands, fallbacks, evidence paths)
Deck: `roundtable/proposals/GOOGLE_DEMO_DECK_2026-09-10.html` (14 slides, 1920×1080)
Live runtime: `scripts/demo_organ_server.py` → http://127.0.0.1:9201

---

## LIVE ORGAN DEMO (added 2026-09-09 evening — the runtime in action)

A browser UI driving the production organs on an isolated demo state root
(`State/tmp/organ_demo`, own lease, own small demo-authored dormant corpus —
no private memory displayed, real `State/` untouched). Verified end-to-end
2026-09-09: valve ingress → heartbeat (3 commits) → dormant recall surfacing
6 records into CORTEX with provenance → fixture-core scripted answer committed
through the real two-barrier circulation ("what color is the sky?" → "The sky
is blue.") → D64 rail 118–133 rows, coverage complete → canonical roundtrip
exact (466+ valid lanes) → mask slider 40% showing view≠body with the body
untouched.

### Boot (30 seconds, before preflight)

```bash
cd /d/Axon
PYTHONUTF8=1 python scripts/demo_organ_server.py --port 9201
# open http://127.0.0.1:9201  (rebuild state: add --rebuild)
```

### The 3-minute live story (between deck scenes 2 and 3, or as its own beat)

1. **Type "what color is the sky?" → valve → heart.** Watch: beat appears with
   commits count, "The sky is blue." lands in RESPONSE_DRAFT, dormant recall
   card shows the surfaced facts WITH provenance (organ-demo containers).
   Say: "The question went through a valve, the Heart ticked, dormant memory
   surfaced, and the answer was committed transactionally — no learned model
   anywhere in that loop."
2. **Click any character cell in the D64 rail.** Its real 16D vector draws as
   a bar chart. Say: "One character = one frozen 16D cell; four cells pack
   into one 64-wide rail row. This is the 16→64 packing, live."
3. **Press "roundtrip check."** Chip turns green: canonical body roundtrips
   exactly. Say: "64 back to 16 — exact, verified by hash against the frozen
   substrate bank."
4. **Slide conversation_history to ~40%.** The rail view shrinks; the amber
   warning shows "view differs from canonical body — body untouched." Slide
   back to 100%. Say: "Attention moves; memory never gets destroyed. Masked
   characters go dormant in place."

### Honest framing (say this once, on screen anyway)

- The header chip says it: "fixture cores · not learned · not serving". The
  answers are scripted fixture cores committing through the REAL Heart
  boundary — the demo shows mechanism, not learned intelligence.
- Unscripted questions get "[fixture core] no scripted answer is registered
  for this input" — that honesty is a feature; let Google see it.
- Demo corpus is demo-authored facts; provenance strings say organ-demo.

### Fallback ladder for the live demo

Server won't boot → run the Kimi runbook Scene 2 mask heredoc + Scene 1
roundtrip (both deterministic, verified). Live UI errors → refresh the page
(state persists on disk). Nothing works → deck slides 5–6 carry the same
content with verified outputs quoted.

## How to drive the deck

- Open `GOOGLE_DEMO_DECK_2026-09-10.html` in Chrome (double-click works).
- `←` / `→` navigate · `N` toggles presenter notes (rehearsal cues per slide) · `F` fullscreen · click left/right half also navigates.
- Slide position persists across refresh (localStorage) — if the deck opens mid-deck before the meeting, press `Home`.
- Print → PDF exports all 14 slides one per page if you want a paper backup.
- The deck does NOT replace the live terminal scenes — it is the frame you project between and during commands.

## The four craft rules (what makes this demo good)

1. **Bad numbers before good ones.** Truth-boundary slide is slide 2, before any win. Scene 7's spoken line leads with 0.000 exact transport, then 1.000 alignment. Nobody can "catch" us hiding anything — that is the credibility engine.
2. **Every number on screen has an evidence path.** If asked "where does that number come from", the answer is always a file on disk or a report in `roundtable/reports/`. Never quote a number that is not in the runbook.
3. **Say what each scene does NOT prove.** Deck slides carry a "Does not prove" box for every scene; say it out loud for Scenes 3, 4, 5, 7.
4. **Mechanism vs learned, every time.** "The Heart committed" / "the field masked" / "the candidate learned". Never "Axon thinks". If a live command fails on stage, use that scene's deterministic fallback — never improvise.

## Scene-by-scene rehearsal cues (2 passes, ~25 min)

| # | Slide | While it runs / you show | The one sentence that matters |
|---|---|---|---|
| 1 | Title | — | "I'm going to show you the organs, and the one that's still learning — with its exact numbers." |
| 2 | Truth boundary | read verbatim | This slide is why everything after it is believable. |
| 3 | Story | — | Build the body first, then the learning organ. |
| 4 | Scene map | — | "Every scene has a deterministic fallback." |
| 5 | Scene 1 | `python substrate/substrate.py` → v7 PASS; roundtrip heredoc | Malformed input is rejected, not repaired. |
| 6 | Scene 2 | mask heredoc, 3 lines all True | The view moves; the body never does. |
| 7 | Scene 3 | pytest → 19 passed; narrate test names | Cores propose; only the Heart commits. |
| 8 | Scene 4 | pytest → 5 passed (~40 s, torch import — start it, keep talking) | This is the machinery; the learned cores are not serving. |
| 9 | Scene 5 | manifest one-liner + timed seek script | 0.40–1.30 ms measured, hash-verified; full walk ~103 s, stated as such. |
| 10 | Scene 6 | 2 cats + `--json jobs` (offline) | Pause at an exact accepted checkpoint is a designed state. |
| 11 | Scene 7 table | slide only, no command | Walk columns: v1 flipped GENERATE, v2 flipped COPY. |
| 12 | Read | — | The honest sentence (verbatim, below). |
| 13 | Ask | — | "Not a chatbot — measured mechanism." |
| 14 | Q&A | — | Keep the same honesty level as the demos. |

## The honest sentence (Scene 7, verbatim)

"The receipt mechanism fixed what it was designed to fix — exact anchor position and exact supervised content at 1.000 across heldout and regression. The learned gate is now the bottleneck: v2 always picks COPY, never EOS, so exact terminated payload transport is 0.000 and teacher-forced token accuracy is 0.475. We have a hypothesis — scalar-bias dominance; checkpoint inspection shows copy_gate.weight norm 0.094, bias −0.0098 — and a ratified, observation-only diagnostic plan. No architecture change before this meeting."

## The ask (Slide 13, verbatim ending)

"We are not asking you to fund a chatbot; we are asking you to fund measured mechanism."

## Preflight addition (append to runbook §1)

- [ ] T+20a (2 min): open the deck in Chrome, press `F`, flip through all 14 slides once, press `Home` to reset to slide 1.
- [ ] T+20b (1 min): `export PYTHONUTF8=1` in every terminal you will show (deck slides 5's café 🙂 中文 needs it).
- [ ] Note: run the pytest scenes with the Python 3.12 full install on PATH (`C:\Users\axema\AppData\Local\Programs\Python\Python312\python.exe`), NOT a 3.11 venv — `python -m pytest` from a bare 3.11 env has no pytest (observed 2026-09-09).

## Timing safety valve

If you are behind at the Scene 4 checkpoint (>7.5 min elapsed): skip Scene 5's live timed-seek script and quote the measured 0.40–1.30 ms from the runbook — the fallback is pre-authorized in the runbook for exactly this. Scene 7 (the credibility scene) is never the one to cut; cut from Scenes 5–6 instead.

## If something live fails

Every scene's fallback is in the runbook §2 (verified commands or quoted evidence). Rule: one retry max, then fallback, then move on. A clean fallback beats a live miracle — the story is discipline, not luck.