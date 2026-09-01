# Evidence Registrar — How to Timestamp Proof

Version 1.0 — 2026-08-31

Jeff does not need to do anything in this file day-to-day. The engineering
table's normal discipline (commit everything, append-only ledgers,
content-addressed artifacts) already generates 90% of the evidence
automatically. This file explains what that evidence is, so Jeff can point
an attorney — or a court — at it, and lists the few manual strengthening
steps worth doing.

## The evidence that already exists (verify any of these yourself)

1. **The git history.** 188 commits, first commit `cad4bb8`
   (2026-07-03), each with a cryptographic hash chained to its parent.
   Anyone with repo access can verify authorship dates, content, and order.
   `git log --format="%H %cI %an" | head` shows it.
2. **The append-only engineers' ledger.**
   `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` — 141+ events, one per work
   turn, never edited (protocol-enforced), each with timestamp, agent
   identity, actions, and evidence. This is a contemporaneous lab notebook.
3. **Content-addressed artifacts.** Every curriculum manifest, checkpoint,
   tranche, and Soul snapshot is named by the SHA-256 of its own content.
   Their existence at paths under `State/` with file dates + the
   repo-recorded hashes is mutually corroborating.
4. **The Source-of-Truth mirrors.** `docs/SOURCE_OF_TRUTH.md` and root
   mirror are byte-identical by an automated test; their git history shows
   each doctrine decision and its date.
5. **This legal folder.** Committed to the same history.

## What GitHub adds (because origin is a private GitHub repo)

- GitHub's server-side records (commit timestamps as received) are
  independent corroboration of the local dates. They are viewable in the
  repo's web UI and via the API even for private repos.
- **Important caveat:** git commit dates are author-controlled. For
  anything that might ever be litigated, the GitHub *push* record (server
  timestamp) is stronger than the local commit date. Since Jeff pushes
  regularly (origin/main is current), the push trail exists already.
- **Visibility confirmed private on 2026-09-01** by Hermes via the Axon
  Browser Hub extension (live GitHub settings page, "This repository is
  currently private." verified from the Danger Zone). Re-verify monthly.

## Manual strengthening steps (recommended, in order of value)

### 1. Confirm the repo is private (2 minutes, do first)

GitHub → Axepapag/Axon → Settings → General → Danger Zone → "Change
visibility" should say **"This repository is currently private."** If it
says anything else, make it private immediately and record the incident in
the ledger.

### 2. Monthly: create a dated evidence snapshot (5 minutes)

Once a month (or before any big decision):

```
git bundle create axon-evidence-YYYY-MM-DD.bundle --all
sha256sum axon-evidence-YYYY-MM-DD.bundle
```

Store the bundle + its SHA-256 on two different media (e.g., an external
drive AND a cloud storage Jeff controls, not a provider that also runs an
AI lab). Write the bundle filename + hash into the engineers' ledger. A
git bundle is a single file containing the entire history; combined with
its hash recorded in a second place, it is a tamper-evident snapshot.

### 3. Optional: independent timestamping (small cost, real value)

For the strongest date proof — beyond even GitHub — use an RFC 3161
timestamping authority on the monthly bundle:

```
# free options exist (e.g. digicert, sectigo via freetsa or similar)
sha256sum axon-evidence-YYYY-MM-DD.bundle
# submit the hash to a timestamping service; store the .tsr token
```

The returned token cryptographically proves the hash existed on that
date. An attorney can walk Jeff through this; it costs nothing to
minutes-level.

### 4. Optional: register with the U.S. Copyright Office

Code is copyrighted automatically on creation, but **registration**
($45–65, fully online, no lawyer needed) is required before suing and
enables statutory damages. A repo snapshot can be registered as an
unpublished work. Jeff can do this himself at copyright.gov — deposit
requirements for code allow the first and last 25 lines of each file
(or the whole thing if he prefers).

### 5. When publishing someday: defensive publication

If Jeff decides *against* patenting a mechanism but wants to stop others
from patenting it: publish a permanent public description (IP.com
defensive publication, or simply a public repo/commit). This creates
prior art worldwide. The decision — patent vs. defensive publication vs.
trade secret — is made per mechanism, with the attorney, using
`PATENT_ROADMAP.md`.

## Who has seen what (the honest exposure ledger)

| Channel | Exposure | Notes |
|---|---|---|
| This repo (private GitHub) | GitHub/Microsoft infrastructure | Standard cloud custody |
| AI agent CLIs used to develop (Anthropic, OpenAI, Moonshot/Kimi, local/Ollama-class) | Providers' systems | Potentially observed; cannot be prevented; assume known |
| This `legal/` folder + all commits | GitHub | Same as repo |
| Local machines | Jeff's control | Source machines |

**The practical stance:** assume the ideas may be known to major AI labs.
What cannot be contested is *who built it and when* — the 188-commit
hash-chained trail — and *ownership of this specific implementation*,
which is Jeff's by copyright regardless. Priority evidence protects the
inventor credit; copyright protects the expression; trade secrecy protects
whatever stays unpublished.

## If a dispute ever happens

1. Do not engage directly. Route everything through the attorney
   (`ATTORNEY_BRIEF.md` is their starting brief).
2. Preserve: make a fresh `git bundle` snapshot immediately, plus copies of
   the ledger, and record the dispute's first contact date in the ledger.
3. The evidence package for any attorney: repo bundle + ledger file + this
   folder + the exposure table above.