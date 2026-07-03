# Round Table: Slot Architecture Review + Open Questions

Date: 2026-07-02
Convened by: Jeff
Participants: Codex, ChatGPT, Hermes, Kimi, Claude (delta docs, one each,
named `docs/<YourName>_delta_slots.md`)

Read `docs/SOURCE_OF_TRUTH.md` (amended today) and `docs/CAPSULE_DIRECTIVE.md`
first. Push back hard. Today's decisions are committed but not sacred; the
open questions below are genuinely open.

## What was decided today (context)

1. The shared field is REGIONS holding SLOTS. A slot is one 8192D vector:
   4096 dims text payload (up to 256 chars, one frozen 16D substrate code per
   character), 2048 dims edge payload (up to 128 chars of edge codes), 512
   dims control block (kind/length/chain/status, ensemble-visible), rest
   reserved. Provenance lives in the container record, not slot dims.
2. All runtime construct/deconstruct is DETERMINISTIC concatenation +
   nearest-code decode. Exact round-trip by arithmetic. No learned encoder in
   the read/write path. Paragraphs chain across slots.
3. Rationale (compute math, 256D 4-layer core, 8k chars active text):
   wide-slot field = 237M MACs/tick vs 1,046M for 1024D chained slots; the
   gap grows quadratically with context. Slot width is bounded above by the
   per-slot information bottleneck (a slot reaches a core as d_model floats),
   which lands on one-sentence-per-slot. One-slot-per-region was rejected
   (padding scales with max not typical; attention/masking/delta granularity
   collapses).
4. CPU residency principle: the tick loop stays matmul-light; a full 8-core
   ensemble tick at 8k chars active is ~1.9G MACs, ~25 ticks/sec on a desktop
   CPU. GPUs are for offline training only.
5. Learned compression is reassigned to soul temperature-tier passes
   (hot->warm->cold). The capsule-core trainer (CE loss, collapse tripwire,
   smoke gates) is retained as the prototype for those passes.
6. New locked training rules born from the v2 collapse: discrete per-slot
   cross-entropy for discrete content; smoke-run gate before any long run;
   no silent truncation (skip and count).

## Open questions for this table

### Q1. Edge labels: registered words vs compact symbols
`{is_a:animal}` vs `{AA}`. Claude's position: typed word-edges with a
controlled vocabulary; the registry canonicalizes edge types and terms
instead of assigning opaque codes; sense-suffix only when ambiguous
(`{is_a:bank.river}`); short aliases later only if the 128-char edge section
gets tight. Rationale: self-grounding (no curriculum needed to teach every
symbol), human auditability, lexical search and edge labels share one
vocabulary. Push back if compactness or stability arguments outweigh this.

### Q2. Curator staffing: API workers vs trained core
Jeff's proposal: the semantic curator role is staffed by API calls (Kimi
CLI or similar) — extraction and canonicalization as offline batch work —
under a propose/dispose split: the API model proposes edges/containers;
deterministic validation commits them (schema check, registry dedup,
append-only registry, provenance stamp created_by/confidence/status).
Expected load shape: front-loaded bootstrap (canonicalize the existing 103k
extracted relations + 140k facts; mine unprocessed episodes), then a
low-rate forever trickle from the dump bucket. A local trained curator
becomes optional (independence luxury), trained later from accumulated
curation traces only if wanted. Stress-test: failure modes of vendor-model
curation, validation rules needed, cost/rate estimates.

### Q3. Edge-ablation gate
Edges must prove retrieval benefit empirically: dormant search with
edge-walking vs without, on held-out questions, before edges are treated as
load-bearing. Design the experiment.

### Q4. Cold-soul -> LoRA distillation sequencing
Deferred until core reasoning lanes work. Position: facts->records,
skills->parameters; parameters buy automaticity, generalization, perception
shaping, and unverbalizable skill — nothing else. Challenge this framing or
its sequencing.

### Q5. Proposed next build (Step 0 + two lanes) — approve or amend
- Step 0: extend capsule_spec.py to the locked 8192D slot layout (sentence
  packing + edge-section packing + control block); revive the 8192D adapter
  registry from legacy_8192/ (mint 64/128/256D adapters); write the slot
  field contract module (regions, masking, materialization). All gated by
  exact round-trip tests.
- Lane 1 (slot recall): the historically proven SOUL_IS_READ drill rebuilt
  on slots — cue on tick A, masked on tick B, answer from soul into
  response_draft; cf_probes (zero-soul, swapped-soul) as the only accepted
  proof.
- Lane 2 (surfaced-knowledge QA): question slot + 2-4 surfaced word-edge
  fact slots from the knowledge graph -> answer into response_draft;
  one-hop questions before two-hop; doubles as the Q3 ablation vehicle.

## Ground rules for deltas

- Cite the SOT layer you're amending or defending.
- Separate "blocking objection" from "preference."
- If you propose an alternative, include its compute/training cost sketch.
- Doctrine floor (not up for review): token-free substrate; exact
  round-trip gates; no silent truncation; runtime autonomy preserved;
  provenance required; cf_probe-style proof over proxy metrics.
