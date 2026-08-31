# Open-Source / Public-Release Readiness Checklist

Version 1.0 — 2026-08-31
Status: BACKLOG. Execute only when Jeff decides to share. Nothing here is
done yet; this is the plan of record for when it is.

## Stage 0 — Jeff's decisions (required before any execution)

| Decision | Options | Recommended default |
|---|---|---|
| What gets shared | (a) nothing, (b) architecture write-up only, (c) organism core code, (d) everything incl. ledgers | (b) first, then (c) after hygiene |
| License | Apache-2.0 / MIT / FSL (source-available) / none-just-copyright | Apache-2.0 (matches Mem0/Letta/Zep; field-standard) |
| Project name public? | "Axon" is a common neuro/AI term (see TRADEMARK_NOTES.md) | Decide name strategy before public repo |
| Jeff's identity | personal name vs. handle vs. company entity | Handle or entity; decide with attorney (cost: $0-150 to form nothing, more for LLC) |
| Patent-first? | file provisionals on Disclosures 1-3 before any public text | Yes for 1 and 2 if budgeting any protection at all |

## Stage 1 — Repo hygiene (the technical scrub — the table can execute all of this)

- [ ] 1.1 **LICENSE file added** (chosen license). Note: this changes the
      repo's legal state for every future commit; do not do this until the
      license decision is final.
- [ ] 1.2 **Personal email scrubbed from published history.** Current git
      author on all 188 commits: Jeff <axemanjeffro@gmail.com>. Options:
      (a) publish a fresh squashed "v1.0" orphan branch (cleanest, loses
      per-commit history *in the public repo* while the private full
      history remains Jeff's evidence), or (b) `git filter-repo` rewrite of
      a public mirror branch. The evidence trail in EVIDENCE_REGISTRAR.md
      stays intact locally either way. **The public repo does not need the
      188-commit history; a clean initial commit is normal and fine.**
- [ ] 1.3 **Local username paths scrubbed.** Known: `docs/KIMI_SUBAGENT_ORCHESTRATION.md`
      references `C:\Users\axema\...`. Sweep for `axema`, `C:\Users`, `D:\Axon`
      strings in all tracked files at publish time.
- [ ] 1.4 **Personal names decision.** "Jeff"/"Jeffrey" appears in docs and
      ledgers (Working Contract, SOT, roundtable). For a public release:
      either accept (it's your project) or replace with a handle in the
      published branch. The word "Jeff" is fine to keep if identity is
      public anyway.
- [ ] 1.5 **Secret re-screen of everything published.** Re-run the
      credential screen (per the roundtable harvest proposal standard)
      across the published set. Current quick-scan is clean; re-verify at
      publish time since content changes daily.
- [ ] 1.6 **Exclude from the public set:** `State/` (already gitignored),
      `archive/` (contains day-zero legacy with unknown history),
      roundtable ledgers (personal work patterns), `legal/` itself (this
      folder stays private forever — publishing your IP strategy is
      self-defeating), `docs/AXON_IDENTITY_V1.md` unless Jeff chooses.
- [ ] 1.7 **README written for outsiders:** what Axon is, what works
      (organism, 100% test-verified), what doesn't yet (learned cores —
      honest framing), how to run the test suite. The current README
      assumes insider context.
- [ ] 1.8 **SOT/AGENTS governance docs:** decide whether the working
      contract/ledger protocol ships publicly (recommendation: yes — it's
      a differentiator and a method paper in itself) — but the *content*
      of the historical ledgers stays private.

## Stage 2 — Legal steps before public

- [ ] 2.1 Provisional patent applications filed on chosen disclosures
      (attorney; ~$1.5-3k each self-filed via attorney-prepared docs, or
      attorney-handled ~$3-6k) — OR recorded decision not to.
- [ ] 2.2 Defensive publication decisions recorded in the ledger for any
      mechanism that will be described publicly without a filing.
- [ ] 2.3 If a company entity is desired pre-launch: LLC formation
      (Wyoming/Delaware, ~$100-500 + registered agent) — attorney call.
- [ ] 2.4 Contributor framework if accepting outside PRs: DCO
      (Developer Certificate of Origin, `Signed-off-by` trailer — matches
      Axon's provenance culture, zero administration burden) over a CLA.
- [ ] 2.5 Trademark: decide and record (see TRADEMARK_NOTES.md); if the
      name matters commercially, file before public launch.

## Stage 3 — Publication mechanics (in order)

- [ ] 3.1 Architecture write-up / blog post / arXiv paper (the MemGPT and
      Zep playbook — the paper *is* the adoption engine).
- [ ] 3.2 Public repo (clean branch, post-hygiene) under chosen license.
- [ ] 3.3 Announce where the agent-memory community lives
      (Hacker News, r/LocalLLaMA, the Letta/Mem0/Zep adjacent communities).
- [ ] 3.4 Hosted offering decision — per the monetization ladder in the
      strategy chat: managed memory-core service → enterprise governance
      tier. This is a business decision with infrastructure costs; not
      part of repo hygiene.

## Standing rule

Until Stage 0 decisions are made and recorded in the engineers' ledger,
**the Confidentiality and IP Policy governs**: repo stays private,
mechanisms stay undescribed externally, and this checklist is not
authorization to publish anything.