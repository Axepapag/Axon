# Axon Legal Protection Folder

Created: 2026-08-31
Maintained by: the engineering table on Jeff's behalf (this folder is a legal
support artifact, not a legal opinion; Jeff's licensed attorney has final say)

## What this folder is

Everything Jeff needs to protect his ownership, priority, and options for
Axon — organized so a non-coder can navigate it and a lawyer can act from it.

## Contents

| # | File | What it does | Priority |
|---|---|---|---|
| 0 | `README.md` | This index — start here | — |
| 1 | `CONFIDENTIALITY_AND_IP_POLICY.md` | The one-page rules that protect everything else. Read first. | **NOW** |
| 2 | `INVENTION_DISCLOSURES.md` | Formal invention disclosures for the load-bearing mechanisms, each with evidence anchors (commit hashes) that prove when they existed | **NOW** |
| 3 | `EVIDENCE_REGISTRAR.md` | How to timestamp proof (GitHub commits, hashes, this folder) so priority is uncontestable | **NOW** |
| 4 | `OPEN_SOURCE_READINESS_CHECKLIST.md` | The pre-publication hygiene backlog (license, email scrub, paths, names) — execute when Jeff decides to share | WHEN SHARING |
| 5 | `TRADEMARK_NOTES.md` | Name protection basics for "Axon" in software/AI classes | EARLY |
| 6 | `PATENT_ROADMAP.md` | Which mechanisms are plausibly patentable, what each protects, costs, and the honest counter-arguments — a decision aid, not a recommendation to file | WHEN BUDGETED |
| 7 | `ATTORNEY_BRIEF.md` | A one-page handoff brief for a licensed IP attorney, so Jeff does not pay for discovery the table already did | ON RETAINER |

## The five rules (summary of file 1)

1. **The repository stays private** until Jeff explicitly decides otherwise.
2. **Nothing here is legal advice.** The table are engineers, not lawyers; a
   licensed attorney must confirm everything before it matters.
3. **Describe mechanisms publicly only after the priority trail exists** —
   and even then, only what Jeff chooses to share.
4. **No secrets, credentials, or personal data in any tracked file.** The
   credential screen (roundtable harvest proposal) governs any future
   transcript imports too.
5. **This folder is itself evidence.** Commit it; never delete it.

## Current protection posture (as of 2026-08-31)

- **Priority evidence:** 188 commits, 2026-07-03 (cad4bb8) through today,
  each hash-chained and dated; key mechanisms anchored in
  `INVENTION_DISCLOSURES.md`.
- **Confidentiality:** GitHub repo `Axepapag/Axon` — believed private
  (verify in file 3); no public disclosures of the core mechanisms found
  beyond this repo and the private roundtable.
- **Third-party exposure (acknowledged):** development was performed via AI
  agent CLIs (Anthropic, OpenAI, Moonshot/Kimi, local/Ollama-class models).
  Jeff assumes their providers have potentially observed the work. This is
  the practical reality of agent-assisted development and is why the
  internal evidence trail (hashes, dates, commits) is the primary protection.
- **License:** none yet — all rights reserved by default (see file 4 for the
  decision framework).

## How Jeff uses this folder

1. Read `CONFIDENTIALITY_AND_IP_POLICY.md` (5 minutes).
2. When ready to spend on protection: hand `ATTORNEY_BRIEF.md` +
   `INVENTION_DISCLOSURES.md` to a registered patent attorney and ask for a
   60-minute consult on provisional filing strategy.
3. When ready to share anything publicly: execute
   `OPEN_SOURCE_READINESS_CHECKLIST.md` first, then `TRADEMARK_NOTES.md`
   filings if the name matters commercially.
4. Update `EVIDENCE_REGISTRAR.md` whenever a milestone lands (the engineering
   table's ledger already does this automatically for code; this file covers
   the manual steps).