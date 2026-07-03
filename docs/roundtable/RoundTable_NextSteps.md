# Round Table: What Should the Axon Team Do Next?

Date: 2026-07-03
Convener: Jeff
This is an OPEN discussion round — no locked doctrine is at stake. Speak
plainly, disagree freely, and be concrete about who should build what.

## Where the project stands (read docs/SOURCE_OF_TRUTH.md for full doctrine)

- The slot architecture is locked: frozen 16D character substrate, 8192D
  deterministic slots (text + edges + control), regions, one frozen adapter
  per d_model, CPU-resident tick loop as the goal.
- Step 0 is BUILT and green: slot_spec, slot field contract, slot adapters
  (snap-idempotence + separability gates), Lane 0 integrity tests. 254 tests.
- The write-head question (how cores write exact characters back to the
  field) is being debated in a parallel round on the adapter-bandwidth brief.
- The GPU box (RTX 3060, hourly billed) is idle. Curriculum assets exist:
  300k-example curriculum v1, the D:\00 recovered memory DBs (140k facts,
  103k relations, 20k episodes), 477MB corpus.
- The bus + round table orchestrator work (you are being woken by it right
  now). Jeff wants the bus to become a more open, autonomous collaboration
  space: agents woken by messages, dispatched to work, managed from the web
  UI — less ceremony, more presence.
- Soul training doctrine (Layer 12-13): temperature tiers, exhale-after-act,
  cf_probe proof, discrete CE losses, smoke gates before long runs.

## The question

Given all of that: what should this team do next, in what order, and who
does what? Consider (non-exhaustively):

1. Training lanes: when do Lane 1 (slot recall / SOUL_IS_READ on slots) and
   Lane 2 (surfaced-knowledge QA) start on the GPU? What blocks them
   besides the write-head resolution? What curriculum work is needed first?
2. The curation lane: the D:\00 knowledge graph needs canonicalizing into
   word-edges (propose/dispose validator). Who builds the validator, and
   when does bootstrap curation run?
3. The bus/collaboration layer: what is the smallest change that makes the
   bus feel like an open workspace (wake-on-message, web-UI agent
   management, dispatch-to-work) rather than scheduled meetings?
4. Sequencing and risk: what should NOT be built yet? Where are we most
   likely fooling ourselves?

## Reply guidance

End your reply with the standard fenced JSON post (the turn prompt shows
the contract). Prioritize: name your top 3 next actions in order, each with
one sentence of why and a named owner (hermes / kimi / codex / claude /
jeff / a-new-seat). Then argue for them. React to what earlier speakers
said — this is a discussion, not parallel monologues.
