# Attorney Brief — One-Page Handoff

Version 1.0 — 2026-08-31
For: any registered patent/IP attorney Jeff engages.
From: the Axon engineering table, at Jeff's direction.
Purpose: give you full context in one page so Jeff's paid hours go to
counsel, not discovery.

## The client and the project

Jeff (a non-coder directing AI agents) is sole owner of "Axon," a private
GitHub repository (Axepapag/Axon) containing an AI system architecture and
working implementation: 188 commits of hash-chained development history
from 2026-07-03 (first commit cad4bb8) to date, built primarily through
AI-agent CLIs (OpenAI Codex, Anthropic Claude, Moonshot Kimi, Nous Hermes,
and local models) under Jeff's direction. Jeff seeks to understand and
secure his IP position before deciding what, if anything, to share
publicly or commercialize.

## What the project contains

A stateful-AI "organism" architecture with a working, fully unit-tested
implementation (~483 automated tests passing), centered on:

1. **Canonical exact-character state under a single transaction authority**
   ("the Heart"): all AI/model edits are untrusted proposals validated and
   atomically committed against frozen base-state hashes with per-source
   authority classes; fail-closed. See `INVENTION_DISCLOSURES.md` §1.
2. **Per-core private state ("Souls")** with causal phase-boundary
   serialization/commits. §2.
3. **Provenance-bound memory** where recovered history is "observation"
   that cannot supervise training without explicit outcome evidence. §3.
4. **No-silent-truncation processing** guarantees (complete-field law with
   machine-enforced launch gates). §4.
5. **Renewable resource tranches** decoupling training budgets from lineage
   identity. §5.
6. **Predeclared falsification-contract governance** (an engineering
   practice). §6.

Each disclosure in `INVENTION_DISCLOSURES.md` includes what it does, why we
believe it was novel when built, evidence anchors (commit hashes/dates),
closest known art from our own landscape research, and honest weaknesses.

## The exposure reality (client already understands)

Development used third-party AI services; client assumes major AI
providers have potentially observed the work. Client wants protection
strategy that accounts for this rather than depending on provider
confidentiality.

## What the client needs from you

1. **Validity check on the disclosures:** are any of §§1–5 plausibly
   patentable (post-Alice), and which? Our internal ranking: 1 and 2
   strongest candidates, 5 fresh with least art seen; 4 weakest.
2. **Provisional vs. defensive-publication vs. trade-secret** advice per
   disclosure, considering cost (client is self-funded) and the fact that
   conversion to issued, enforceable patents is likely out of current
   budget.
3. **Priority-confirmation strategy:** what to file/record now to lock the
   2026-07→09 development timeline as defensible inventorship evidence.
4. **Ownership hygiene for AI-assisted invention:** any documentation Jeff
   should maintain regarding his direction of AI-agent authorship (work
   made for hire, contemporaneous direction records — the repo's ledgers
   already record this in detail; tell us if a stronger form is needed).
5. **Pre-publication protocol:** a checklist for the moment Jeff wants to
   publish code or papers, so no rights are lost by accident (U.S. 12-month
   grace, foreign absolute novelty).

## Evidence available to you (all verifiable)

- Private GitHub repo with server-side push timestamps.
- Append-only JSONL engineering ledger (141 events, protocol-enforced
  never-edited).
- Content-addressed artifacts (SHA-256-named manifests/checkpoints).
- Monthly `git bundle` snapshots planned per `EVIDENCE_REGISTRAR.md`.
- This folder (committed, hash-chained).

## Immediate questions for the consult

1. Given AI-assisted development, is inventorship attribution at risk?
2. Should Jeff file one flagship provisional now on §1 (Heart) to open the
   12-month option window, or defer all spending?
3. Is there anything in the repo that, if published, would be worse than
   useless to patent (i.e., should never be published)?
4. Does the third-party AI exposure change the patent-eligibility calculus?
5. What is the minimum credible monthly evidence ritual for a self-funded
   inventor (our EVIDENCE_REGISTRAR.md proposes git bundles + RFC 3161
   timestamping)?

## Constraints

- Client budget-conscious: prioritize advice that maximizes protection per
  dollar.
- All communication with Jeff should assume non-technical reader; the
  engineering agents can translate into implementation detail on request.
- Everything in this folder is confidential attorney-client communication
  material once an engagement begins.