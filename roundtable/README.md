# Roundtable — Index and House Rules

The roundtable is the engineering team's shared table: proposals, reviews,
reports, decisions, and drafts. The three ledger files and the team bus stay
at this root because `AGENTS.md`, `docs/WORKING_CONTRACT.md`, and every
engineer's session-start ritual point at them here.

## Layout (organized 2026-09-04)

| Folder | Holds |
|--------|-------|
| `proposals/` | Design proposals awaiting or holding ratification |
| `reviews/` | Reviews of another engineer's work; campaign recoveries; implementation handoffs |
| `reports/` | Audits, campaign evidence, tournament/candidate result reports |
| `decisions/` | Jeff-ratified resolutions |
| `drafts/` | Unratified doctrine amendment drafts |

Root (never move these):

- `ENGINEERS_LEDGER_PROTOCOL.md` — the ledger law
- `ENGINEERS_LEDGER.md` — rolling continuity summary
- `ENGINEERS_LEDGER_CANONICAL.jsonl` — append-only canonical history
- `ENGINE_TEAM_BUS.md` — coordination channel (not canonical history)

## Rules

1. New documents go into the folder that matches their kind, named
   `KIND_TOPIC_YYYY-MM-DD.md` following existing convention. Proposals carry
   a `Status: PROPOSAL ONLY` header until Jeff ratifies.
2. Everything inside the five folders is Git-tracked (see `.gitignore`);
   documents are meant to travel with the repo.
3. **Historical references may point at old flat paths.** The canonical
   ledger, the team bus history, Source of Truth, and legal evidence docs
   cite files as `roundtable/<NAME>.md` from before the 2026-09-04
   organization. Those records are immutable by ledger law and were not
   rewritten. Resolve any such reference by filename in the folder map
   above — filenames are unique across the table.
4. Moving a document does not change its ratification status. Check the
   document's own Status header and the canonical ledger.
5. The bus remains the place for assignments, questions, and steering;
   the canonical ledger remains the place of record. A new proposal worth
   the team's attention should also get a one-line bus post.

## Current hot items (2026-09-06)

- FFN256 copy-alignment renewal COMPLETE: heldout copy/position 1.0,
  regression position still 0.667. Do not spend a third tranche until
  the 12/36 misses are diagnosed. No serving.
  `reports/GROK_D64_MIXER_COPY_ALIGNMENT_RENEWAL_2026-09-06.md`
- `proposals/MID_RUN_ARTIFACT_SYNC_PROPOSAL_2026-09-04.md` — ratified and
  implemented in `418a9d9`; end-of-run bundle transfer is the default and
  mid-run sync is opt-in. Codex subsequently hardened checkpoint-boundary
  provenance against slow or failed uploads.
- `decisions/RESOLUTION_FOUNDATIONS_FIRST_CURRICULUM_2026-09-04.md` —
  ratified; Stage 1 implementation is committed (`77f18db`, `d121391`) and its
  first smoke falsified premature sequence training. Stage 0 exact typed-delta
  motor control is the active prerequisite.
- `reviews/CODEX_KAGGLE_CONTINUATION_REVIEW_2026-09-03.md` — historical
  continuation contract; the step-720 output is now locally fetched and
  remains non-serving comparison evidence.
