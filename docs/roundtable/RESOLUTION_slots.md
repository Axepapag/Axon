# Round Table Resolution: Slot Architecture

Date: 2026-07-03
Participants: Jeff (convener), Hermes (glm-5.2), Codex, Claude (synthesis)
Inputs: `Hermes_delta_slots.md`, `docs/Codex_delta_slots.md`,
`RoundTable_SlotArchitecture.md`
Status: CONSENSUS — no further round required. This document is the input to
the next SOURCE_OF_TRUTH amendment and the fresh-repo bootstrap.

## R1. Edge labels (Q1) — MERGED POSITION

1. Every edge is BORN as a typed word-edge: `{is_a:animal}`. The registry's
   canonical form is always the full human-readable record (edge_type, target,
   sense, directionality, provenance, status).
2. The registry MAY promote a short substrate-safe alias (2-4 chars) for an
   edge type+target pair appearing in >= N containers (start N=10). Aliases
   are registered abbreviations, bidirectionally mapped; never new meanings;
   never opaque-from-birth.
3. Slot edge-payload pack policy (deterministic): prefer full word-edges;
   fall back to aliases for the densest edges when the 128-char payload would
   overflow; the control block records which form each edge uses.
4. EDGE OVERFLOW CONTRACT (Codex B1, blocking, adopted): the slot edge payload
   is a bounded VIEW of the container's full edge record, which lives whole in
   the dormant/container record. If edges exceed the payload even in alias
   form, the packer must (a) chain an edge-continuation slot, or (b) surface a
   prioritized subset by explicit, recorded policy. Either path is
   deterministic, tested, and counted. No semantic edge is ever dropped
   silently.
5. Rare-alias expansion is owned by the context-annotation lane (R6).

## R2. Curation contract (Q2) — MERGED POSITION

API workers (Kimi/Hermes/cloud) staff the curator role. Propose/dispose split:

- Workers PROPOSE: containers, edge candidates, canonical labels, aliases,
  confidence, provenance, source pointers.
- A deterministic VALIDATOR commits or rejects. Checklist (Codex P2 + Hermes
  Q2A merged): schema validity; SOURCE POINTER REQUIRED (claims without a
  source record pointer are committed as status=candidate, never as facts);
  registry dedup; alias collision check; edge direction/type check;
  provenance hash; confidence/status; payload capacity check;
  redaction/sensitivity status; append-only registry writes; CONTRADICTION
  GATE (a new edge contradicting a higher-confidence existing edge on the
  same (source, edge_type) is flagged for review, not auto-committed).
- Drift detection (Hermes): every batch stamped `created_by: model:version:date`;
  periodic re-probe of a fixed 100-container held-out sample through the
  current vendor model; Jaccard similarity below threshold flags review.
- All accepted AND rejected proposals accumulate as training traces for an
  optional future local curator (train only if API costs ever warrant).
- Cost expectation: bootstrap ~243k items ~= $50-250 API; ongoing trickle
  negligible; the real budget item is validator engineering.

## R3. Edge-ablation gate (Q3) — MERGED DESIGN

Run AFTER Lane 2 exists (never on proxies). Held-out QA set from recovered
DBs: one-hop, two-hop, procedure, project-memory, and negative-control
questions. Frozen answer core, slot budget, corpus, and context window.
Conditions: (1) exact text only; (2) + registry lookup; (3) + compact aliases;
(4) edge-walk 1 hop; (5) edge-walk 2 hops; (6) + context-annotation capsules.
Metrics: answer accuracy, recall@{1,3,5}, precision@k, active-slot count,
latency, SIBLING-NOISE RATE (dog must not drag in every animal).
Pass: edges improve accuracy or recall on held-out questions; improvement
survives negative controls; active-field noise stays under a fixed budget;
result recorded with corpus hash, retriever config, checkpoint ids.

## R4. Cold-soul -> LoRA distillation (Q4) — DEFERRED, GATES HARDENED

Sequencing unchanged: after core reasoning lanes work. Design notes recorded
for when it goes live (Hermes + Codex merged):
- Classify cold slots procedural vs episodic via compression lineage +
  cross-slot similarity; only procedural candidates distill; unique episodes
  stay in soul/records.
- Bake gates: held-out PROCEDURAL TASK improvement with source cold slots
  masked (generalization, not just recall); must not increase false factual
  recall; manifest links training traces, cold-slot lineage, evals, redaction.
- Target: "behave better because I have done this class of thing before,"
  never "remember this fact."

## R5. Deltas are typed records (Codex B2, adopted — resolves the SOT Open Maybe)

Cores may produce proposed 8192D field snapshots (training simplicity). The
runtime derives explicit typed delta records from each proposal:
`update_slot`, `append_slot`, `clear_slot`, `update_response_draft`,
`attach_edge`, `request_surface_memory`, `request_tool_call`, ...
The consolidator attends over proposals and commits TYPED deltas. Deterministic
CPU diffing; auditable state mutation; no added transformer compute.

## R6. Context-annotation lane (Codex P5, adopted as Lane 3)

A `context_annotations` region owned by an async worker (API first, trained
later, outside the tick gate): compact-alias expansions, why-this-surfaced
notes, source confidence, conflict notes, request-for-more-context markers.
Never overwrites source slots; never blocks the tick.

## R7. Build plan (Q5) — APPROVED AS AMENDED

Step 0 (foundations, in the fresh repo):
1. `slots/slot_spec.py` — NEW module (not a capsule_spec extension): 8192D
   layout (4096 text / 2048 edge / 512 control / reserved), pack/unpack for
   slot and region, chain semantics for text AND edges, edge-overflow
   contract, control-block grammar, response-draft commit-marker placeholder,
   exact round-trip self-test.
2. `slots/slot_field_contract.py` — nine regions, masking, materialization
   from container records.
3. `adapters/slot_adapter.py` — NEW adapters on the 16D-packed slot basis
   (legacy design as template only; verified: legacy imports wide_substrate).
   Mint 64/128/256D. Gate: text-payload dims must round-trip exactly through
   project-down/project-up; edge/control sections exact-gated once stable.
4. Tests for all of the above. Nothing under a legacy path is current
   (Codex B3).

Lanes (one trainer, staged drills — Hermes A3):
- Lane 0 (slot integrity, blocking): pack/unpack exactness, edge payload,
  chain reconstruction, overlength skip-and-count, invalid-symbol rejection,
  control decode. No trainer runs until green.
- Lane 1 (soul recall): SOUL_IS_READ rebuilt on slots. Tick A cue + exhale;
  tick B masked continuation from soul. Probes: correct/zero/swapped/
  irrelevant soul + field-leak. Not solved without probes.
- Lane 2 (surfaced-knowledge QA): question slot + 2-4 fact slots ->
  response_draft; one-hop then two-hop; hosts the R3 ablation.
- Lane 3 (context annotation): after Lane 2 baseline; must improve accuracy
  without breaking the noise budget.
- soul_v2 wiring into core.py: new `soul_mode="v2_tiered"` option, config-
  gated, existing checkpoints unaffected; exhale is a new code path.

## R8. Fresh repository — APPROVED

The current repo carries three architectural generations and is hard to read.
A fresh repo is bootstrapped with:
- Layout: shallow folders by role — docs/, substrate/, slots/, adapters/,
  cores/, training/, curator/, runtime/, tests/ (+ gitignored State/,
  datasets/, checkpoints/, runs/).
- SOURCE_OF_TRUTH v2 written from the amended SOT + this resolution.
- Promote-by-review: only modules the SOT names as current (substrate.py,
  core.py, soul_v2.py, cf_probe.py, kg_search.py, container_schema.py,
  semantic_layout_machine.py, axon_bus/, curated docs). Everything else stays
  behind in the archive repo.
- PROVENANCE.md records the archive repo path and source commit hash of every
  promoted file.
- AxonGliksbot becomes the second archive ring (alongside axon7). Archives
  are never deleted.

## Doctrine floor check

Token-free substrate: preserved. Exact round-trip gates: extended (slot,
adapter, chain, edge view). No silent truncation: strengthened (edge overflow
contract). Runtime autonomy: no new limits. Provenance: strengthened (source
pointers, drift stamps, PROVENANCE.md). Proof over proxy: preserved (probes,
ablation, negative controls).
