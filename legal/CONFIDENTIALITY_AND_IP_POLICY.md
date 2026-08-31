# Axon Confidentiality and IP Policy

Version 1.0 — 2026-08-31
Authority: Jeff (convener). This policy is binding on every human and AI
agent working anywhere in this repository. It extends
`docs/WORKING_CONTRACT.md` and the ledger protocol; it does not replace them.

## 1. The default

**Everything in this repository is confidential and proprietary.** No
license has been granted. All rights reserved by Jeff. Nothing here may be
copied, published, described externally, or used to build a competing system
without Jeff's explicit written direction.

## 2. Repository posture

- The GitHub repository (`Axepapag/Axon`) must remain **private**. Any change
  to its visibility is a Jeff-level decision, like Source-of-Truth doctrine.
- The `legal/` folder, the engineers' ledgers, and the roundtable documents
  are **internal evidence**, not publication material.
- If the repo is ever found public, treat it as an incident: snapshot
  everything, record the exposure window, and ask Jeff whether to
  re-private and/or file defensive publications immediately.

## 3. What may never leave this repository

1. The recovered `D:\00` corpus and `State/dormant/` evidence — it contains
   third-party personal data and is protected read-only source material.
2. Canonical Identity text for Axon (`docs/AXON_IDENTITY_V1.md`) — publish
   only by explicit Jeff decision.
3. Any credential, key, token, or secret — permanent ban (already enforced
   by the Working Contract §4 and the roundtable credential-screen flag).
4. Training checkpoints, candidate weights, and private Soul branches —
   until and unless Jeff decides a released model is part of the plan.
5. Personal names, emails, and local filesystem paths of the humans and
   machines involved (see the readiness checklist for the scrub list).

## 4. External communication rules

- **Describe mechanisms externally only when Jeff says so**, and then only
  the specific description he approved. The safe default is: internal
   documentation is written so it *could* be published later, but isn't.
- Engineering agents (Codex, Claude, Kimmy, ChatGPT, Hermes, and any future
  participant) must not post about Axon mechanisms on public channels,
  blogs, social media, or in other projects' contexts.
- Press, papers, blog posts, and conference talks about Axon require Jeff's
  sign-off on the final text.

## 5. The third-party exposure reality (acknowledged, not waived)

Development is performed through AI agent CLIs operated by third-party
providers (Anthropic, OpenAI, Moonshot/Kimi, and local/Ollama-class
serving). Jeff assumes those providers have potentially observed the work;
this cannot be prevented and is the industry reality. Consequences:

- **The primary protection is the internal evidence trail**: dated, signed,
  hash-chained commits and ledgers that prove Jeff's team built this, when.
- Jeff does not rely on provider confidentiality for protection.
- Nothing in this policy should be read as permission to publish *because*
  "it's already been seen." Exposure through a service provider does not
  create a public disclosure that starts patent clocks in most regimes, but
  only an attorney can advise on specifics — do not assume either way.

## 6. Patent-clock discipline

If a mechanism is ever **publicly disclosed** (published, open-sourced, demoed
to outsiders, or described in a public talk), the U.S. provides a 12-month
grace window to file a patent application on it; many foreign jurisdictions
provide **none**. Therefore:

- Before any public disclosure of a mechanism listed in
  `INVENTION_DISCLOSURES.md`, Jeff decides: file a provisional first, file
  a defensive publication, or accept the disclosure.
- This decision is recorded in the engineers' ledger like every other
  material decision.

## 7. Agent-authored work

- All code, documents, and designs in this repository are work product of
  Jeff's project, authored by his direction (humans directing AI agents).
  Commits carry `Co-Authored-By` agent trailers for provenance, which do not
  convey ownership of any kind.
- Engineers' ledger events and identity stamps are provenance records, not
  ownership records. Ownership questions (especially any future human
  collaborator or contractor) go through Jeff and an attorney, in writing,
  before work begins.

## 8. If someone wants to use, fund, buy, or join Axon

All inbound interest (collaboration, investment, acquisition, license
requests) routes to Jeff personally. No agent may grant anything, sign
anything, or accept terms. The standard response for an agent receiving
such an inquiry is: "I'll pass this to the project owner; he'll respond
directly."