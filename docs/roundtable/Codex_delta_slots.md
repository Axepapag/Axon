# Codex Delta: Slot Architecture Review

Date: 2026-07-03
Author: Codex
Source docs read:
- `roundtable/RoundTable_SlotArchitecture.md`
- `docs/SOURCE_OF_TRUTH.md`
- `docs/CAPSULE_DIRECTIVE.md`

## Scope

This delta treats the current `SOURCE_OF_TRUTH.md` and roundtable brief as the
active architecture: 16D frozen alphabet substrate, deterministic 8192D slots,
regions holding slots, shared-state adapters per d_model, private souls, and
deltas in common 8192D slot field space.

The older capsule directive is useful history and should still influence
training discipline, especially exact reconstruction gates, collapse tripwires,
smoke runs, and counterfactual probes. It should not override the current
runtime field decision. The active runtime field is deterministic 8192D slots,
not learned 1024D sentence capsules.

## Blocking Objections

### B1. The edge payload needs an overflow contract before implementation

SOT layer amended: Layer 0 slot layout, Layer 2 semantic edges, Layer 13 locked
training rules.

The current 8192D slot layout gives 2048 dims to edge payload, which means 128
characters of packed 16D substrate text. That is enough for compact aliases, but
it is not enough for unrestricted typed word-edges such as:

`{is_a:animal}{has_property:furry}{risk:uncertain_source}{source:...}`

This becomes a silent-truncation risk unless the slot spec defines what happens
when the edge payload is full.

Required amendment before code:
- Edge payload is a bounded view, not the whole edge record.
- Full edge records live in the container/dormant record with provenance.
- The visible edge payload carries a prioritized compact edge capsule.
- If edges overflow 128 chars, the system must either chain an edge-continuation
  slot, skip-and-count for that view, or surface fewer edges by explicit policy.
- The policy must be deterministic and measured in tests.

No semantic edge can be dropped silently.

Cost sketch: no extra model compute if compact aliases are used. Edge-continuation
slots add attention tokens only for dense cases. Validation cost is CPU string
packing and record lookup.

### B2. `delta = full field rewrite` is too coarse for production

SOT layer amended: Layer 9 delta protocol and Open Maybe on typed operation tags.

The core may internally produce a proposed 8192D field snapshot, but the committed
delta format should not be only "replace the whole active field." Full-field
replacement makes audit, conflict resolution, round-robin consolidation, and
multi-core merge behavior much harder.

Recommendation:
- A core can emit a proposed field snapshot for training simplicity.
- The runtime should derive an explicit delta record from that proposal:
  `update_slot`, `append_slot`, `clear_slot`, `request_surface_memory`,
  `update_response_draft`, `attach_edge`, `request_tool_call`, etc.
- The consolidator attends over the proposals and commits typed deltas.

This preserves the common 8192D language while making state mutation auditable.

Cost sketch: adds deterministic diffing and validation on CPU. It reduces
debugging cost and does not add transformer MACs.

### B3. The current SOT references promoted `legacy_8192` modules in a confusing way

SOT layer amended: Layer 6 adapter ownership and Layer 20 file map.

The text says `legacy_8192/state_adapter.py` is "promoted back to current." That
is understandable historically, but it is brittle as a source of truth. A module
under `legacy_8192/` should not be described as current production ownership.

Recommendation:
- Keep old 8192 wide-page code under `legacy_8192/`.
- Put the active slot adapter registry in a non-legacy module name, for example
  `slot_adapter.py` or `state_adapter.py`.
- If a legacy implementation is reused, copy or move it into the active module
  path and leave a historical note in `legacy_8192/`.

Cost sketch: small repo cleanup. No architecture cost.

## Preferences And Amendments

### P1. Use a hybrid edge-label system

Question addressed: Q1, registered words vs compact symbols.

I do not recommend choosing only `{is_a:animal}` or only `{AA}`.

Use both, with clear ownership:
- The registry source of truth stores human-readable canonical labels:
  `edge_type=is_a`, `target=animal`, `sense=animal.general`, description,
  directionality, provenance, status.
- The slot edge payload stores compact aliases: `{AA}`, `{K9}`, `{FR}`, etc.
- The context/annotation region can expand compact aliases into short readable
  capsules when the reasoning cores need it.

Why:
- Compact aliases fit the 128-character edge payload.
- Human-readable labels keep the registry auditable and searchable.
- The system can train and evaluate on the alias while humans inspect the label.
- New symbols do not require every core to memorize the world. The semantic
  search layer and context core surface the relevant expansion when needed.

Training implication:
- Cores should learn the common high-frequency aliases through repeated use.
- Rare aliases should be expanded into active context capsules by the context
  core or surfacing worker.
- The dormant registry remains the authority.

### P2. API curator workers are acceptable for bootstrap, but only behind a validator

Question addressed: Q2, API workers vs trained core.

Use Kimi/Hermes/cloud workers for front-loaded curation. Do not let them directly
commit canonical knowledge.

The split should be:
- API worker proposes containers, edge candidates, labels, aliases, confidence,
  and provenance.
- Deterministic validator commits or rejects.
- Accepted and rejected proposals become the training trace for a future local
  semantic curator.

Validator requirements:
- schema validity
- source span or source record pointer
- registry deduplication
- alias collision check
- edge direction/type check
- provenance hash
- confidence/status
- payload capacity check
- redaction/sensitivity status
- append-only registry writes

Failure mode to avoid: a vendor model invents a neat edge that is semantically
plausible but unsupported by the source record. The validator should require a
source pointer and mark unsupported claims as candidates, not facts.

Cost sketch: front-loaded API cost, then low-rate trickle from the dump bucket.
The local deterministic validator is cheap. A trained local curator should wait
until there are enough accepted/rejected traces to supervise it.

### P3. Edge-ablation must measure retrieval quality and active-field noise

Question addressed: Q3, edge-ablation gate.

Experiment design:
- Build a held-out QA set from recovered DB records.
- Split into one-hop, two-hop, procedure, project-memory, and negative-control
  questions.
- Freeze the answer core, slot budget, dormant corpus, and context window.
- Run retrieval modes:
  1. exact text only
  2. exact text plus registry lookup
  3. exact text plus compact edge aliases
  4. edge-walk one hop
  5. edge-walk two hops
  6. edge-walk plus context-core expansion capsules
- Compare answer accuracy, retrieval recall@k, precision@k, active-slot count,
  latency, and sibling-noise rate.

Sibling-noise rate matters. If `dog` brings in every animal, the edge system is
hurting reasoning. The expected win is surfacing the right fact bundle for the
current question, not flooding active state with neighbors.

Pass condition:
- Edges must improve answer accuracy or retrieval recall on held-out questions.
- The improvement must survive negative controls.
- Active-field noise must stay under a fixed budget.
- The result must be recorded with corpus hash, retriever config, and checkpoint
  identifiers.

Cost sketch: offline CPU retrieval evaluation plus optional GPU/CPU answer-core
passes. No new model architecture required.

### P4. Cold-soul to LoRA distillation should stay late

Question addressed: Q4, LoRA sequencing.

I agree with the framing:
- facts go to dormant records
- skills and repeated procedural experience can go to parameters

Do not start LoRA distillation until the core reasoning lanes are working.
Otherwise the system will bake confusion into parameters and lose auditability.

Add one more gate:
- A LoRA candidate must improve held-out procedural tasks with the source cold
  slots masked.
- It must not increase false factual recall.
- It must write a manifest linking training traces, cold-slot lineage, evals,
  and redaction status.

The target is not "remember this fact." The target is "behave better because I
have done this class of thing before."

### P5. Add a context-core annotation lane, but keep it outside the tick gate

SOT layer amended: Layer 4 semantic search and surfacing.

The context core should own a `context_annotations` or `slot_context` region. It
can run async over active slots and write:
- compact aliases relevant to the slot
- human-readable edge expansions
- why this memory surfaced
- source confidence
- conflict notes
- request-for-more-context markers

It should not overwrite source slots. It should not block the tick loop. It is a
producer of better active context, not a runtime gate.

This resolves the symbol problem: reasoning cores do not need to permanently
know every rare alias. They see compact symbols when useful and readable
expansions when the context core decides the slot needs them.

Cost sketch: can start as API/offline worker. Later it can become a trained
local core. Runtime impact is bounded by annotation slots surfaced into active
state.

## Q5 Next Build Recommendation

I approve Step 0 plus Lane 1 and Lane 2, with these amendments.

### Step 0 Amendments

Build the slot contract first:
- `SlotSpec`: exact dimensions, text capacity, edge capacity, control grammar.
- deterministic `pack_slot` and `unpack_slot`
- paragraph/sentence chain semantics
- edge overflow behavior
- control block schema
- region names and masks
- response-draft commit marker placeholder
- exact round-trip tests

Do not start another long trainer until this exists and passes.

### Lane 0: slot integrity

Add a small lane before soul recall:
- pack/unpack exact text
- pack/unpack exact edge payload
- chain reconstruction
- overlength skip-and-count
- invalid symbol rejection
- control-block decode

This is not optional. It prevents another architecture drift where the trainer
runs before the field contract is real.

### Lane 1: slot recall

Rebuild `SOUL_IS_READ` on slots:
- Tick A: cue in shared field, response/delta supervised, then exhale writes hot
  soul trace.
- Tick B: original cue masked or removed, continuation requires the soul trace.
- Probes: correct soul, zero soul, swapped soul, irrelevant soul, field leak.

Do not count it as solved without the probes.

### Lane 2: surfaced-knowledge QA

Question slot plus 2-4 surfaced fact/edge slots goes to response draft. Start
with one-hop questions. Then run the edge-ablation experiment on the same lane.

This should be the first proof that semantic edges are worth carrying in active
slots.

### Lane 3: context annotation

After Lane 2 has a baseline, add the context-core annotation region:
- same question
- same surfaced facts
- plus context annotations

Measure whether annotations improve answer accuracy without increasing noise.

## Decision Summary

My delta is:

1. Keep deterministic 8192D slots as the active field.
2. Keep the 16D alphabet as the bottom substrate.
3. Treat the capsule directive as historical training discipline, not current
   runtime field design.
4. Use compact edge aliases in slots and readable canonical labels in the
   registry.
5. Add a hard edge-overflow contract before implementation.
6. Use API workers for curation only through propose/validate/commit.
7. Make edge ablation a required gate before edges become load-bearing.
8. Keep LoRA distillation late and procedural.
9. Add a context annotation lane so rare symbols can be expanded when useful.
10. Start with slot integrity tests before any new long trainer.

