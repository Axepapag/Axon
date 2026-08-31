# Patent Roadmap — Decision Aid

Version 1.0 — 2026-08-31
Status: decision framework for Jeff + input for an attorney. This file
deliberately does NOT recommend filing everything; it lays out cost, value,
and risk per disclosure so Jeff can choose with an attorney.

## The three protection modes (pick per mechanism)

| Mode | What it costs | What it buys | Risk if skipped |
|---|---|---|---|
| **Trade secret** (default today) | $0 — just keep the repo private | Protection lasts while secret | Vanishes if the idea is independently invented or leaks |
| **Provisional patent** | ~$1.5–3k each (self-filed from attorney-prepared docs) or ~$3–6k attorney-handled | 12 months of U.S. priority; "patent pending" | Money spent if never converted; public after 18 months if converted |
| **Defensive publication** | ~$0–500 | No one can ever patent it (prior art worldwide) | You also give up YOUR chance to patent it |

**The clock rule (critical):** in the U.S. you have 12 months after
*public disclosure* to file. In most foreign countries: **zero** months.
Once a mechanism is publicly described, foreign patent rights are gone
immediately, and U.S. rights start a 12-month countdown. This is why
`CONFIDENTIALITY_AND_IP_POLICY.md` §6 exists.

## Per-disclosure assessment (engineer's honest view)

| # | Disclosure | Patent-plausible? | Independent-invention risk | Verdict sketch |
|---|---|---|---|---|
| 1 | Heart / transaction-governed canonical state | Medium — combination claim; 101 subject-matter question | **High** (big labs all need agent-state governance; this is where they're heading) | Best single provisional candidate. Or: publish as paper + defensive publication |
| 2 | Private layered Souls, causal phase commits | Medium | **High** (personalization + continuity is the hottest research area) | Second provisional candidate |
| 3 | Provenance-bound memory + eligibility gating | Medium-high; also the most safety-relevant | Medium | Strong defensive-publication candidate — the idea *should* be prior art |
| 4 | No-silent-truncation complete-field law | Lower (technique is close to art) | Low-medium | Defensive publication or skip |
| 5 | Renewable resource tranches | Medium-high; freshest; least art seen | Medium (it's an obvious pain point) | Provisional candidate; also fine as trade secret while private |
| 6 | Falsification-contract governance | Low (method/governance) | Low | Defensive publication / practice paper |

## The realistic strategic picture

**What actually protects Jeff today (in order):**
1. **Copyright on the code** — automatic, his, regardless of anything else.
2. **The evidence trail** — 188 hash-chained commits proving he built it
   first, which supports both inventor-ship arguments and any "they copied
   us" posture (access + substantial similarity).
3. **Trade secrecy** — while the repo is private, competitors must
   independently invent, and the evidence trail proves if they instead
   copied.
4. **Speed of publication** — the MemGPT/Zep playbook: a good paper +
   Apache repo *creates* the association "this idea belongs to Axon's
   author" faster than a patent does.

**What patents add:** the ability to *stop funded competitors* who
independently build the same mechanisms. Without one, being first gives
you reputation but not exclusivity.

**The honest counterweight for a solo self-funded builder:**
- Provisionals are cheap; **conversions are not** (utility filing +
  prosecution: $15–30k+ each over 2–4 years, per patent, per country).
- Software patents face real 101 hurdles and heavy invalidation risk post-
  Alice; examiner's "abstract idea" rejections are the norm for anything
  that reads like "do X... with a computer."
- An issued patent you can't afford to enforce is a trophy, not a shield.
  Enforcement runs $1–5M through trial.

**Therefore the sensible ladder for Jeff:**
1. **Now, $0:** keep everything private; evidence trail keeps growing
   automatically.
2. **Before any public description of Disclosures 1/2/5:** either file a
   provisional on each ($1.5–3k self-prepared from
   `INVENTION_DISCLOSURES.md` + attorney hour) or record a decision not to.
   This is the only hard deadline, and only Jeff's publication choices
   trigger it.
3. **Consider one "flagship" provisional** (Disclosure 1, maybe +2/+5 in the
   same filing) if budget allows, mainly to buy 12 months of option value
   while the C1 verdict and early feedback clarify commercial direction.
4. **Convert to full utility applications only if** a funded business
   exists to justify $15–30k+ and enforcement.

## If Jeff files: how to file cheaply-but-correctly

1. Take `ATTORNEY_BRIEF.md` + `INVENTION_DISCLOSURES.md` to a **registered
   patent attorney or agent** (agents are cheaper; find one via AIPLA).
   Ask for: 60-minute consult → provisional drafting estimate.
2. Say the magic words: **"I want a provisional application on my
   inventor-disclosed mechanism; I have written disclosures with evidence
   anchors; please advise on claim strategy and 101 risk."**
3. Self-represented filing (pro se) is legal but risky — an attorney's
   1–2 hours reviewing a self-drafted provisional is the best $500 Jeff
   can spend on protection.
4. **Never** describe the invention to anyone outside the attorney-client
   privilege bubble before filing.