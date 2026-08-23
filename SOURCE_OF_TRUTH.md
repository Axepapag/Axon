# Axon Source Of Truth

Last updated: 2026-08-23 (Trainer parameter-control plane + Cortex cadence/role doctrine ratified)

## Core Doctrine

Axon is a stateful, always-on AI built around a canonical shared field and private reasoning cores.

The field is Jeff's bridge into Axon's world. It must preserve exact English characters, provenance, and structured context without hiding meaning behind opaque semantic symbols.

## Substrate

- The frozen alphabet substrate is 16D.
- Every visible character in active text maps to one frozen 16D vector.
- Character identity and order are source-of-truth data, not a lossy summary.
- A larger `d_model` core may lift those 16D cells into its own lane, but it does not replace the canonical character field.

## Active Shared Field

The shared field is organized into regions. Regions can include:

- `conversation_history`
- `user_input`
- `response_draft`
- `cortex`
- `situation_awareness`
- `scratch`
- `tool_results`
- `advisor_input`
- `task_state`
- `diary`

As of Shared Field schema `shared-field-v2`, the canonical semantic-context region is named **`cortex`**. The old `structured_knowledge` concept is a subset of Cortex function: exact dormant material recovered by semantic relevance belongs inside the Cortex picture, but Cortex is not merely a retrieval bucket. The region is the Heart-governed, auditable textual/materialized surface of Axon's broader Semantic Cortex organ: relevant dormant evidence, grounded semantic relationships, concepts, entity/relation context, and other semantic interpretation that reasoning cores should be able to inspect alongside the exact rest of the Shared Field.

The canonical `cortex` region is not the whole Semantic Cortex organ. The organ may maintain richer derived/noncanonical working state and semantic-core outputs between canonical materializations. It never becomes a second canonical body and never bypasses Heart authority. The separate `semantic_cortex` Heart valve is the governed organ-to-Heart boundary and remains CLOSED until a real autonomous Cortex service earns activation.

Persisted `shared-field-v1` history is immutable. Historical v1 snapshots keep their original serialized region name `structured_knowledge` and therefore keep their original `field_id`/hash. `CanonicalStateBranch.migrate_to_current_schema()` advances a branch by creating a new v2 successor with the exact same spans, provenance, manifests, and text, parented to the v1 HEAD; it never rewrites the historical snapshot. New v2 snapshots serialize the region as `cortex`.

Each region contains ordered character cells plus metadata spans. Words, sentences, paragraphs, and semantic edges are represented as spans over exact characters with auditable metadata.

No active exact-text path may collapse a paragraph into one opaque vector and then ask a small core to recover exact text from that vector.

Every core pass attends the entire currently unmasked shared field. A physical
model window may be used as one page in a complete ordered sweep, but it is not
an attention limit and may not silently omit unmasked field characters. Every
logical pass must produce an auditable coverage record proving that each exact
shared-field character was visited.

Each persisted region may contain an unmasked shared-field portion and a masked
dormant portion. Per-region policies may retain exact characters, lines,
paragraphs, containers, or conversational turns. Changing a threshold moves
the boundary only: masked text is preserved exactly, and moving the boundary
back immediately restores that material to the shared field.

Attention masks are **derived compile-time views**, not part of the canonical
shared-field identity. The canonical `SharedFieldSnapshot` is the ordered spans;
mask policies are resolved to attended intervals when the heart compiles a rail
or forms a recall query. Changing a mask therefore does not create a new
canonical body and does not allow the heart to hold a second canonical field
that is not persisted through the branch.

The initial implementation may use one movable boundary per region. The
versioned future mask schema may additionally select multiple ordered,
non-overlapping active intervals, such as a pinned older passage plus the
newest turns. This is an additive feature, not a prerequisite for the first
complete-field reader; in every form, masked characters remain exact and
restorable.

## Dormant State

Dormant state is structured memory outside the current canonical shared field.
It stores containers, edges, facts, procedures, episodes, diary entries,
source chunks, and provenance.

Dormant memory is not attended directly. Search and surfacing copy relevant readable material into active regions.

Masking is not truncation. It is an explicit, auditable shared-to-dormant
membership transition governed independently per region. A model window may
never move this boundary implicitly.

## Semantic Cortex Organ

The Semantic Cortex gives Axon a continuously refreshed semantic understanding
of the universe represented by the current Shared Field and the knowledge and
experience stored in Dormant State. It is not equivalent to vector search,
embeddings retrieval, or the canonical `cortex` region alone. Dormant retrieval
is one Cortex sense; semantic interpretation is the organ's larger job.

The Cortex runs on its own **cortical cadence**. A Cortex tick is distinct from
both a heartbeat and a reasoning tick. The Cortex may inspect the exact current
Shared Field repeatedly even when `field_id` has not changed. On each cortical
tick it may inspect its prior derived semantic state, query Dormant State,
follow semantic edges, reevaluate concepts/entities/relationships and confidence,
and refresh its grounded semantic picture. A later pass may discover useful
relationships that an earlier pass did not, even against an unchanged Shared
Field.

The intended mature circulation is:

1. Heart exposes the current canonical Shared Field and exact grounded rail views.
2. Semantic cores of one or more `d_model` widths operate as a Cortex ensemble on
   their own cadence.
3. The Cortex queries Dormant State through a governed dormant-memory sense as
   often as required by its semantic work; Dormant State remains the durable
   store of Axon's experience and knowledge.
4. The Cortex maintains derived/noncanonical semantic working state and may submit
   a grounded semantic materialization proposal through the `semantic_cortex`
   Heart valve.
5. Only the Heart may materialize accepted Cortex output into the canonical
   `cortex` region or bind a frozen cortical semantic projection into a reasoning
   tick image.

The current implementation is a bootstrap predecessor to that mature circulation:
`dormant_recall` is presently a real Heart-owned CAPPED valve and the Heart itself
runs recall/materialization into `cortex`; `semantic_cortex` remains CLOSED. This
direct Dormant-to-Heart recall path is acceptable temporary anatomy, not the final
ownership model. As the Cortex comes alive, dormant recall becomes a Cortex-owned
semantic sense while Heart remains the sole canonical state writer.

Semantic Cortex cores are not reasoning cores. They may share substrate, rail
contracts, parameter governance, and Transformer mechanisms, but their purpose,
cadence, curricula, evaluations, and promotion gates are distinct. The Cortex
may ultimately contain an ensemble of semantic cores of different widths and
specialties, all grounded back to the same exact Shared Field and Dormant State.

## Semantic Edges

Semantic edges are spelled out in English:

```text
dog is a animal
FastAPI exposes endpoint
checkpoint produced by trainer
```

Opaque semantic-edge symbols are not part of the reasoning surface. Layout IDs or grouping codes may exist as internal metadata only if they never replace the English edge meaning shown to cores.

## Containers

Containers are structured records over text and spans. A container may represent a word, phrase, fact, procedure, event, diary entry, tool result, or recovered memory.

Required properties:

- exact surface text or letter sequence,
- normalized text,
- readable semantic edges,
- provenance,
- confidence,
- lifecycle status,
- metadata needed for retrieval and audit.

## The Heart (Field Compiler Organ)

The Field Compiler Organ is Axon's heart. It runs on its own cadence — the
heartbeat — which is distinct from a cognitive tick. The heart pumps exact
information: external input (users, tools, advisors) inward to the organs,
and organ output outward, roundtrip. Only the heart may validate,
materialize, and commit canonical shared-field state. Other organs and
ingress paths may originate and submit proposed mutations; they never mutate
canonical state directly. Cores, consolidators, ingress paths, and the
dormant valve all cross the heart's typed validation/transaction boundary.

A heartbeat is event-driven: durable ingress or other governed work is the
primary doorbell. While input or commit work exists, the heart beats promptly.
While idle, the permanent host keeps a slow bounded liveness beat. Idle
heartbeats advance durable cardiac liveness identity but do not create a new
canonical field or cognitive tick. Proposal-board activity advances the
in-flight tick workspace and does not by itself create a new canonical field.

The permanent runtime is `runtime/heart/host.py`. Exactly one host may own the
active canonical branch at a time: `runtime/heart/lease.py` holds an operating-
system file lock beneath `State/active/heart`, so a second process fails closed
before it can mutate canonical state. `runtime/heart/identity.py` persists the
Heart epoch, process-start sequence, heartbeat sequence, and cognitive-tick
sequence with atomic replace + fsync. Sequence identities are reserved durably
before use; a crash may leave a gap but may not cause identity reuse after
restart.

Every external or organ-facing mutation crosses the sovereign valve plane in
`runtime/heart/valve.py`. The plane has twenty permanent valve slots. The
primitive real `user_ingress`, `tool_ingress`, `advisor_ingress`, and
`dormant_recall` valves begin CAPPED under explicit item/queue/character
budgets; the sixteen future-organ slots begin CLOSED. CLOSED is fail-closed,
and even OPEN valves remain subject to global cardiac intake budgets. An
external envelope carries source identity, payload, type, and provenance only;
it cannot supply an `AuthorityGrant`. The Heart resolves valve identity to the
permitted authority class and exact canonical region itself, then revalidates
valve version, source class, payload type/size, replay identity, budgets, and
typed-delta invariants at the final gate before commit.

Each beat:

1. Drains queued external arrivals. Between ticks, intake commits immediately
   as a heart-governed typed delta into its runtime-owned region. Arrivals
   during an in-flight tick queue for the next beat and never mutate the
   frozen base.
2. Resolves derived per-region attention masks for the rail and recall query.
   Masked text remains canonical and restorable; only attended intervals enter
   the compiled rails. Policies are resolved from reusable mask policies such
   as `all`, `none`, or `last_n_spans` at compile/recall time and do not alter
   the canonical `SharedFieldSnapshot` identity.
3. Detects change via canonical field identity/freshness. No change means no
   recompilation.
4. In the current bootstrap runtime, runs the Heart-owned `dormant_recall` valve when change warrants recall. Mature Cortex anatomy moves semantic recall ownership behind the independent cortical cadence; Heart continues to validate any canonical materialization.
5. Recompiles the affected rail(s), proving complete coverage and exact
   roundtrip against the fresh canonical field.
6. Services the tick workspace: collecting proposals, enforcing barriers, and
   committing the validated consolidator decision.

Build B established the first real circulation path. The permanent Heart-host
increment now makes that path restartable and continuously runnable:
`runtime/heart/durable_ingress.py` provides a durable global FIFO with explicit
ack cursor, rejection/quarantine evidence, and crash recovery;
`runtime/heart/coordinator.py` remains the one circulation/transaction engine
that commits through the existing `HeartTransactionBoundary` and canonical
branch. Durable ingress is acknowledged only after canonical persistence. If a
process dies after commit but before ack, the event id embedded in canonical
span provenance lets the restarted Heart prove the item already became real
and acknowledge it without duplicating text.

Canonical branch HEAD is the authority during partial failures; the branch
journal is audit evidence. Heart commit receipts and branch events retain valve
id/version, source id, item id, authority class, governed regions, provenance,
base/successor field identity, and tick binding when applicable. Health and
lease files beneath `State/active/heart` are durable observability/control
metadata, not a second canonical body.

Attention-mask choices remain derived. Each frozen tick image and rail now also
carry an explicit derived `view_id` computed from the mask policy set. Two
masked views of the same canonical `field_id` are therefore distinguishable
without making masks canonical. A changed field still must pass the exact D64
coverage/roundtrip proof before the tick image is frozen. With no real reasoning
cores registered yet, the host explicitly closes the empty developmental tick
after a successful freeze so circulation can continue; it does not invent a
participant or proposal. Proposal/refinement/consolidation barriers attach to
these real frozen images in later builds.

Primitive but real organs are acceptable progress; fake organs are not. An
organ may be noisy or weak in its first form provided it is real permanent
anatomy, actually functions, is observable and testable, and its shortcomings
remain explicit improvement work. A weak first form must never be declared
the final target. The living organism (heartbeat, ingress, recall, freeze,
rails, barriers, consolidator proposal, heart commit) is established before
substantive reasoning-core training; cores are later trained to operate
correctly inside this anatomy.

## Authority Classes

- External ingress (user/tool/advisor) may submit heart-governed mutations
  targeting only its runtime-owned regions, and only between ticks.
- The current bootstrap dormant valve may submit Heart-governed materialization of governed `cortex`; it never independently writes truth. In mature anatomy, dormant recall is a Cortex-owned semantic sense and the Semantic Cortex submits any canonical Cortex materialization through its governed Heart boundary.
- Core proposals may target only the scopes their authority class permits.
- The consolidator's proposal may address every canonical region as governed.
- Only the heart's transaction layer converts any proposal into canonical
  state. Validation must detect and reject conflicting or overlapping sparse
  edits and any proposal not bound to the frozen base.

Bootstrap write restrictions (such as a scratch/response-only validator) are
implementation restrictions, not doctrine; widening happens only through this
authority model.

## Cores

Cores are transformer reasoners with private souls. Axon's mature reasoning ensemble is heterogeneous: cores may have different architectures, specialties, parameter counts, and `d_model` widths while still reasoning against the same frozen canonical Shared Field. Each active width receives its own derived rail/lens from the Heart, bound to the same exact field/tick identity and provenance. No core's larger or smaller rail becomes a competing truth body.

The present 64D core is a developmental proving width, not a final architecture limit. Development may prove new widths deliberately and independently, but mature Axon may run 64D, 128D, 256D, 512D, 1024D, or other explicitly governed widths together in one ensemble once each rail/compiler/core contract is proven. A tick is one full
deliberation round against a frozen canonical base:

1. The heart stabilizes intake and dormant recall, freezes canonical field
   F_N, and emits for each active d_model rail a derived, immutable tick
   image R_N. Rails and tick images are projections, never second canonical
   state.
2. The heart declares the tick's participant set from the core registry
   (active / offline-training / disabled, rail membership).
3. Each participating core inhales its private soul, attends the complete
   tick image with a coverage proof, emits a sparse proposed delta (only the
   edits it proposes, each bound to F_N with author/rail/pass provenance),
   and exhales the experience into its soul.
4. The first pass closes when every required participant has returned,
   failed, or timed out under governed policy. The heart then exposes the
   complete first-pass proposal board in the per-rail workspace.
5. Each core inhales its updated soul, re-attends the tick image plus the
   complete proposal board, emits one refined sparse delta, and exhales.
6. After the refinement barrier, the rotating consolidator attends the tick
   image plus every refined delta and emits one proposed authoritative delta.
   The consolidator is the final reasoning authority of the tick; it still
   only proposes.
7. The heart validates the consolidator's proposal (typing, authority class,
   base freshness, provenance) and atomically commits it, producing the
   successor canonical field; the consolidator exhales its experience. The
   tick ends at that commit — and only there.

"Against the entire shared field" means authored against the exact frozen
base with field-wide addressability as permitted by authority class; it never
means reproducing unchanged content.

The validated consolidator delta may address every canonical shared-field
region. Axon's cores ultimately maintain Axon's conversation, knowledge,
situation awareness, task state, scratch, response, diary, and other canonical
regions. Runtime validation, immutable provenance, base-field identity, and
atomic replay remain mandatory; field-wide authority is not permission for
unattributed or partial writes. Any narrower validator in the bootstrap
runtime is a temporary implementation restriction rather than final doctrine.

Input does not enter the soul first. The shared field is the input interface.

## Souls

Soul state is private per core. It is not the canonical knowledge store.

Soul writes move hot to warm to cold over time. The soul should learn experience, habits, and intuition from repeated episodes. Auditable knowledge belongs in dormant state.

## Trainer Organ (Parameter Guardian)

The Trainer is a permanent organism subsystem, not merely a command-line script used to create seed models. Its sovereignty is over **parameter state and learning lineage** in the same way that the Heart's sovereignty is over canonical Shared Field state.

The Heart remains the sole canonical Shared Field writer. The Trainer becomes the sole governed authority that may create or mutate candidate model parameters, optimizer state, LoRA/adapters, or other learned parameter-bearing generations. Reasoning cores, semantic cores, Cortex, tools, and Trainer advisory cores may request or recommend learning; they do not directly own unrestricted backpropagation or parameter writes.

Every live parameter-bearing organ must eventually register with the Trainer. The Trainer maintains a complete parameter inventory containing module identity, organ role, architecture, `d_model` width, generation identity, parameter names/shapes/dtypes/trainability, and exact lineage fingerprints at governed boundaries. A declared organism inventory that is missing an expected parameter-bearing module is incomplete and may not authorize a training mutation.

The Trainer's learning state is durable beneath `State/training/trainer`. It records immutable inventories, candidate-generation lineage, mutation plans and authorization receipts, source/curriculum manifests, holdouts, optimizer and schedule configuration, telemetry, checkpoints, evaluations, rejected generations, promotion proposals, active adapter ancestry, and rollback points. Parameter history must be inspectable rather than mystical.

The Trainer must expose transparent telemetry for every parameter under its authority. At minimum, each governed training step/cadence must be able to report per-tensor value and gradient health, norms/RMS/extrema, finite/zero fractions, trainability, optimizer/schedule state, active grants, candidate generation, data provenance, evaluation state, and parameter/update budgets. Exact full tensor hashes are required at lineage/checkpoint/promotion boundaries; continuous telemetry may use bounded numeric summaries while still enumerating every parameter.

Training is branch-like. A live accepted generation is never silently edited in place. Learning creates an isolated candidate generation from an explicit base inventory, with declared writable tensors and budgets. Promotion requires held-out and counterfactual evidence, lineage receipts, regression/forgetting checks, and a separately governed activation step. Rejection preserves evidence; it does not erase the failed generation from learning history.

LoRA/adapters are first-class governed parameter generations, not a loophole around parameter authority. The Trainer may issue adapter-only grants that fail closed if a plan attempts to touch base parameters. Online/inference-time adaptation must be more tightly budgeted than offline learning and is never allowed to bypass source provenance, telemetry, holdouts, rollback, or promotion rules.

The Trainer may eventually contain its own ensemble of Transformer cores. Candidate advisory roles include curriculum construction, optimizer/gradient control, evaluation, catastrophic-forgetting audit, and promotion criticism. These Trainer cores may have different `d_model` widths and specialties, but their outputs are advisory proposals. A deterministic Trainer authority layer validates the exact parameter inventory and grant before any optimizer/backpropagation path is allowed to mutate tensors.

Trainer cadence is distinct from heartbeat, Cortex tick, and reasoning tick. The Trainer may run sustained offline learning, bounded online adaptation, continuous parameter-health observation, or study campaigns such as "learn philosophy". Study acquisition enters Dormant State with provenance first; Cortex may semantically organize it; Trainer then constructs governed curricula/candidates and evaluates them before any learned generation can become active.

`runtime/trainer/` implements the first non-training control plane for this organ: complete heterogeneous parameter registration/inventory, exact parameter fingerprints, per-parameter value/gradient telemetry, fail-closed mutation grants/plans, immutable durable control records, and promotion proposals that themselves have no activation authority. No optimizer loop is activated by this control-plane milestone.

## Training Contract

Training must match runtime:

- curriculum enters through the shared field,
- unused regions are masked,
- core inhales,
- core attends,
- core emits deltas,
- loss is applied to the delta/response target,
- core exhales after action.

Full-field training must reproduce the same complete ordered sweep, all-delta
refinement, consolidator pass, soul boundaries, and typed canonical commit used
at runtime. A short physical page may not be trained or reported as though it
were the complete field.

Current Day Zero D64 trainer:

- `training/train_complete_field_64d.py` is canonical-only; there is no active detached-record or legacy-anatomy switch,
- curriculum records are source material only and are materialized as `SharedFieldSnapshot` before core access,
- every neural read enters through the deterministic D64 compiler and its complete/fresh coverage proof,
- the current D64 reader deterministically unpacks exact 16D lanes before its per-character neural lift,
- scratch changes are ordinary typed deltas followed by canonical rematerialization and a second complete read,
- response-draft learning remains observable and exact-position/copy-gate evaluation remains available,
- training workspaces live beneath `State/training`; branch-backed episode journaling and canonical split/resume proof remain required before a new training campaign is authorized.

The archived 461,500-step Bible-trained 64D checkpoint family remains historical evidence only. A bounded compatibility/donor experiment was performed during development, then explicitly rejected as the future initialization path. Fresh reasoning-core and semantic-core training begins from clean current anatomy; legacy 384-slot checkpoints are not imported, resumed, or used as seed weights.

Pre-Day-Zero 384-slot readers, ExactV4 runtime/trainer paths, multi-tick prototypes, soul pilots, detached curriculum builders, and their dedicated tests are historical evidence only under `archive/day_zero_legacy_2026-08-20/`. They are not active fallback or initialization interfaces.

## Deterministic D64 Field Compiler

The shared exact D64 compiler is implemented in `runtime/field/compiler_d64.py`
and is the canonical D64 core-input boundary for both runtime-facing and
training-facing adapters.

Binding invariants:

- the compiler consumes one immutable `SharedFieldSnapshot` and binds every
  rail to that snapshot's exact `field_id` and `tick_id`;
- every attended canonical character is represented by its literal frozen 16D
  substrate cell; unsupported attended characters fail closed rather than
  being omitted;
- one D64 physical row contains at most four exact 16D cells; rows never cross
  logical-region boundaries and unused lanes are explicit padding;
- every valid lane retains exact region position, global active-field position,
  source span, span position, source, provenance, row, and lane identity;
- all ten logical regions are visited on every compile, including empty or
  explicitly masked regions; masked text remains canonical state but is not an
  attended rail character; attended intervals are sorted, non-overlapping,
  half-open ranges over the region's full span text and are compiled exactly;
  masks may be supplied at compile time as derived views and do not change the
  canonical field identity;
- compilation is accepted only after complete coverage and exact 16D roundtrip
  verification; a rail from an older `field_id` is stale and must not be used;
- a D64 row is lossless storage, not four magically independent Transformer
  tokens. Current V6 consumers deterministically unpack exact lanes before the
  existing per-character neural lift;
- compiler output is derived and rebuildable. It has no reasoning vote and no
  commit authority. Cores/consolidation propose ordinary typed `FieldDelta`
  objects and canonical validation/transaction code decides whether they may
  become the next field.

The deterministic compiler may mark exact structural spans such as words,
sentences, and paragraphs. Learned English semantics, semantic compression,
semantic rewrite/decompilation, and salience ranking remain separate future
work and are not made canonical by this section.

Each d_model rail is a dual surface over the frozen tick image. The exact
scaffold is the lossless, region-preserving, provenance-complete packing of
exact 16D character cells defined above, and it alone carries the coverage
and roundtrip guarantees. Alongside it, the heart may derive semantic slots
for words, phrases, sentences, paragraphs, concepts, and edges; every
semantic slot carries source-span references back to exact canonical
characters. Semantic slots are derived and rebuildable; they never become the
only copy of anything, and they hold no reasoning vote and no commit
authority. First-form semantic slots are deterministic derivations from exact
structure. During development, new rail widths may be proven deliberately one
at a time so failures stay attributable, but this is a validation sequence rather
than a mature-runtime exclusivity rule. Axon's eventual Cortex and reasoning
ensembles may contain 64D, 128D, 256D, 512D, 1024D, or other explicitly governed
widths concurrently. Each width receives its own derived lens from the same
canonical Shared Field and must preserve exact-character grounding, provenance,
freshness, and coverage/roundtrip guarantees. Improvements at larger widths may
also feed better 64D specialists; no width supersedes the canonical 16D substrate.
A reasoning core is never required to reproduce every character to prove grounding.

Build D.1 implements the first permanent dual-surface D64 anatomy in
`runtime/field/semantic_d64.py`. The exact `CompiledD64Field` remains unchanged
as the sole coverage/roundtrip scaffold. Alongside it, a
`D64SemanticSurfaceCompiler` derives deterministic D64 slots for exact words,
sentences, paragraphs, and canonical `FieldSpan` boundaries. The current
feature generation is explicitly `structural-lexical-v1`: hashed exact lexical
and structural features only, not a trained English embedding model and not a
simulation of Semantic Cortex intelligence.

Every D.1 semantic slot is bound to the same `field_id`, `tick_id`, and exact
`rail_id`; carries its exact D64 lane references, canonical source-span IDs,
container/edge refs, region range, and exact text hash; and stores a read-only
64D feature vector whose deterministic value is recomputed during grounding
verification. A slot is omitted from a masked derived view if its complete
structural source span is not attended; semantic slots may not bridge hidden
characters. The semantic surface is derived/rebuildable and cannot mutate
canonical state.

Frozen tick images use `axon-heart-frozen-tick-image-v2` for this dual surface.
Each 64D `RailBinding` may now carry the exact semantic-surface ID, feature
generation, and slot count. The Heart coordinator freezes and retains the
noncanonical `CompiledD64DualSurface` for the lifetime of the tick, and clears
it when the tick closes. A semantic surface from another field, exact rail, or
mask view fails closed. Exact-only callers remain supported, but their frozen
image identity records that no semantic surface is bound.

Build D.1 is first-form semantic *anatomy*, not mature learned semantics. The
`semantic_cortex` Heart valve remains CLOSED, no Cortex model is trained or
activated by D.1, and learned specialists must later earn their own explicit
model/surface generation while preserving the same exact grounding contract.

Build D.2 establishes the grounded specialist evaluation boundary without
activating a Cortex organ. `Cortext/contracts.py` service schema v2 binds every
specialist query/observation to an explicit D64 semantic projection: exact
`field_id`, `tick_id`, `rail_id`, semantic-surface ID, feature generation, and
selected semantic-slot receipts containing exact lane references, canonical
source-span IDs, and exact text hashes. A stale, substituted, or tampered
projection fails closed. Specialist observations remain noncanonical and carry
no Heart authority.

`Cortext/evaluation.py` provides the first real consumer of that boundary: the
untrained `d64-structural-lexical-cosine-v1` reranking baseline over the same
held-out recovered semantic-edge task used for Build C.1. On the accepted
64-case seed and pool limit 64, candidate-pool recall remains 0.687500. The D.1
`structural-lexical-v1` surface alone achieves Hit@8 0.078125 / MRR 0.021354,
while the existing exact-evidence C.1 relevance auditor on the identical pools
achieves Hit@8 0.625000 / MRR 0.529557. The evaluation compiled 1,659,828 exact
characters into 250,507 grounded semantic slots across the cases. One recovered
candidate contained an exact character unsupported by the frozen 16D substrate;
it was counted explicitly as D64-inaccessible rather than normalized or
truncated, and it was not an expected target. Artifact:
`State/dormant/.derived/evidence_v1/evaluations/build_d2_d64_specialist_baseline_64_v1.json`,
SHA256 `1e176a27d6ab24cf79969f73c6ab8b68486f66acaaf5d0176e8b5b453cc4b0ee`.

D.2 therefore proves a material semantic capability gap and justifies designing
and training a narrow semantic-edge reranking specialist as the next measured
intelligence experiment. It does not prove any learned specialist, does not
open the `semantic_cortex` valve, and does not authorize canonical writes or a
wider rail. Any trained successor must beat declared held-out baselines while
preserving the same exact projection/evidence grounding before promotion.

## Canonical state root

All living or durable Axon state resides beneath `D:\Axon\State`,
including canonical field state, dormant memory, private souls, active adapter
pointers and promoted adapters, cursors, and offline-learning control records.
Runtime and training do not own separate competing state roots.

Training may create isolated copy-on-write branches, run workspaces, and curriculum material beneath `State\training`,
but core-facing state must use the same `SharedFieldSnapshot`, dormant-memory,
exact D64 compiler rail, typed-delta, validation, and commit contracts as
runtime. Curriculum JSON may remain reproducible source material, but it is
materialized as canonical state before a D64 core reads it. A smoke or
curriculum may be small in content or compute; it may not substitute a
truncated/fake core-facing anatomy that production later discards.

Candidate checkpoints and reproducible run logs belong beneath
`State/training/runs/` while non-authoritative. Promotion moves or copies an
accepted state-bearing artifact into its governed canonical State location with
explicit provenance.

## Dormant Evidence Bridge

The canonical read-only retrieval/surfacing organ is implemented in
`runtime/dormant/evidence_bridge.py`. It treats the recovered JSONL files under
`State/dormant` as the only dormant-memory authority.

Its lexical/graph index is a disposable sense, not a memory body. The derived
SQLite file may contain only lookup metadata needed for retrieval: stable IDs,
byte offsets/lengths, hashes, lexical postings, metadata filters, and graph
neighbor references. Lexical postings should use derived term hashes and local
integer row references rather than repeating source text or stable IDs per
posting. It must not copy authoritative container text, semantic-edge text,
source strings, or provenance strings into index record tables.

Every active index generation is bound to SHA256 identities for the authoritative
dormant files plus the recovered source hashes recorded by `corpus_manifest.json`.
Candidate retrieval returns IDs. Before evidence can enter the shared field,
the bridge seeks back into the authoritative JSONL, rereads the exact bytes,
and verifies raw-record hash, record identity, exact text hash, source hash,
and provenance hash. A changed corpus makes the previously bound derived index
stale until verified maintenance publishes a generation bound to the new exact
corpus; readers never silently treat stale lookup metadata as current memory.

Selected exact container text is surfaced as provenance-bearing `FieldSpan`
material in canonical `cortex`; source container IDs and verified
semantic-edge IDs remain attached as references. Generated separators are
explicit runtime spans rather than alterations to dormant text. The resulting
`SharedFieldSnapshot` must compile through the deterministic D64 compiler with
complete coverage and exact roundtrip before it is accepted as core-facing
state. Retrieval/indexing has no reasoning vote and no commit authority.

The dormant valve is part of the heart. Candidate ranking combines lexical
support, recovered graph topology, query-matched semantic-edge support,
confidence, type/task relevance, and novelty versus the active field under a
governed budget. First-form semantic relevance is recovered-edge graph
semantics already owned by the corpus; no trained encoder exists and none may
be simulated. Relation propagation is bounded and fail-closed: generic graph
hops and query-matched relations are distinct signals, one strongly matched
edge may contribute bounded best-edge support, and duplicate/weak edges may
not accumulate into synthetic certainty. Exact bytes are always dereferenced
from authoritative JSONL and hash/provenance-verified before surfacing. If the
semantic/relevance path yields nothing eligible, fallback is limited to exact
grounded lexical candidates under the same full-item and total budgets.

Build C.1 adds a deterministic relevance/retention auditor in
`runtime/dormant/relevance.py`. It ranks only already verified exact evidence;
it does not truncate source containers, invent vectors, or gain commit
authority. Heart materialization preserves the selected container and semantic-
edge references on the canonical `cortex` span itself as well as
in the Heart commit receipt, together with dormant index generation identity.

The derived evidence index has verified generations and atomic active-generation
promotion in `runtime/dormant/generations.py`. The existing
`evidence_v1/index.sqlite3` may be adopted zero-copy as a generation after full
binding verification. Candidate full generations are opened and verified before
an atomic pointer swap; a failed candidate cannot evict a healthy reader.

Build C.2 adds real transactional append/update maintenance in
`runtime/dormant/incremental.py`. Ordinary append-only growth and equal-byte-
length/layout-preserving updates are detected by sequential exact-byte scan,
reindexed inside one SQLite transaction, rebound to the complete current corpus,
then exact-opened and binding-verified before a new logical generation pointer is
published. The logical generation may reuse the same derived SQLite file; this
is not a second memory body and does not copy 4.4 GB merely to record a small
memory change. Updated postings and graph links are limited to affected rows and
existing lookup indexes rather than scanning/rebuilding the entire derived graph.

Incremental maintenance is deliberately narrower than arbitrary file mutation.
Truncation, deletion, insertion into the indexed prefix, stable container-ID
replacement, or any variable-length edit that shifts authoritative record offsets
fails closed to the isolated full-generation rebuild path. A prepared generation
manifest is fsynced before the SQLite transaction, so a crash after derived-index
commit but before active-pointer publication can be recovered only from durable
generation evidence that matches the current verified index and exact corpus.
Stale maintenance plans cannot replay after generation identity advances.

Index maintenance is not heartbeat work and has no canonical-state write
authority. The Heart only notices the verified active-generation token change and
reopens its disposable reader. On the accepted real 427,001-container / 351,978-
edge corpus, a C.2 dry-run verified an exact no-op in about 22 seconds without
modifying the 4.4 GB index. An isolated 10,000-container / 9,999-edge benchmark
then appended 250 containers and 250 edges in-place in about 2.01 seconds and
converged on the same final corpus/index identity as a clean full rebuild. These
figures are machine-specific operational evidence, not latency guarantees.

Build C.1 also establishes deterministic held-out evaluation in
`runtime/dormant/evaluation.py` and `scripts/evaluate_dormant_relevance.py`.
The accepted 64-case forward recovered-semantic-edge benchmark measures
source+relation -> exact target without target-text leakage and reports both
candidate-pool and post-auditor quality. On the accepted generation it produced
pool recall 0.687500, raw Hit@8 0.203125 / MRR 0.053032, and audited Hit@8
0.625000 / MRR 0.529557. Final v4 publication verification reproduced the same
evaluation ID and byte-identical report after a lookup-plan optimization reduced
the representative 36-term real candidate query from 61.5 seconds to 2.87
seconds and the full 64-case evaluation from about one hour to 2m50s. These
metrics are evidence of a primitive real semantic sense, not a claim of mature
semantic recall.

`Cortext/contracts.py` is now permanent interface anatomy only: grounded
semantic queries, exact evidence references, semantic observations, and a
width/architecture-independent specialist protocol. No production Semantic
Cortex implementation or training is authorized by this contract alone, and
the Heart's `semantic_cortex` valve remains CLOSED until a real specialist is
evaluated and explicitly promoted.

## Day Zero active surface

The active implementation surface is intentionally narrow:

- `runtime/field/schema.py` ? canonical ten-region exact field schema,
- `runtime/field/delta.py` ? typed canonical deltas and validation/apply/replay,
- `runtime/field/compiler_d64.py` and `runtime/field/semantic_d64.py` ? exact deterministic D64 compiler plus grounded deterministic first-form semantic-slot surface,
- `runtime/field/state_branch.py` ? canonical branch persistence,
- `runtime/axon_runtime/d64_adapter.py` ? runtime-facing exact and dual-surface D64 adapter,
- `runtime/dormant/evidence_bridge.py`, `runtime/dormant/relevance.py`, `runtime/dormant/generations.py`, `runtime/dormant/incremental.py`, and `runtime/dormant/evaluation.py` ? read-only manifest/hash-bound dormant retrieval, exact dereference, bounded graph/relation relevance, verified derived-index generations, transactional append/layout-preserving update maintenance, and held-out evaluation,
- `Cortext/contracts.py` ? grounded Semantic Cortex service contract only; no active specialist/training authority and the `semantic_cortex` valve remains CLOSED,
- `runtime/heart/` ? heart anatomy: authority/core control plane, canonical transaction boundary, beat coordinator, sovereign 20-slot valve plane, OS single-writer lease, restart-safe cardiac identity, durable ingress/replay/quarantine spool, health observability, explicit derived-view identity, relevance-gated dormant recall, and the permanent Heart host;
- `runtime/trainer/` ? first permanent Trainer control-plane anatomy: heterogeneous parameter registration, exact inventory/fingerprints, mutation authority, per-parameter telemetry, immutable control-state store, and non-activating promotion proposals; no optimizer loop is activated by this package;
- `scripts/run_axon_heart.py`, `scripts/evaluate_dormant_relevance.py`, `scripts/maintain_dormant_index.py`, and `scripts/verify_d64_dual_surface.py` ? permanent Heart runtime, deterministic dormant semantic/relevance evaluation, explicit derived-index maintenance/recovery, and read-only live D64 dual-surface verification entry points;
- `training/canonical_d64.py`, `training/complete_field_64d.py`, and `training/train_complete_field_64d.py` ? current developmental canonical D64 training path only; future campaigns must execute behind Trainer parameter authority,
- `curator/` recovered-corpus schema/materialization/building utilities ? offline exact dormant-memory tooling,

The former council, old core/soul implementation, ExactV4/identity-v2 runtime stack, 384-slot views/schedules, legacy trainers/curricula, launchers, policies, and dedicated tests are archived beneath `archive/day_zero_legacy_2026-08-20/`. Local historical runs, datasets, checkpoint bundles, and generated distributions are preserved beneath `State/archive/day_zero_legacy_20260820/local_artifacts/`. They may be inspected for provenance or mechanism recovery but may not be imported, launched, resumed, or presented as current Axon without a new explicit convener decision.

There is one Source of Truth text. `docs/SOURCE_OF_TRUTH.md` is the master path and root `SOURCE_OF_TRUTH.md` is a byte-for-byte compatibility mirror. Any doctrine update must update both in the same change; repository tests enforce equality. `docs/WORKING_CONTRACT.md` and root `WORKING_CONTRACT.md` follow the same exact-mirror rule.

## Deleted Architecture

The wide-slot adapter lineage is deleted from this repo. It is not stale, experimental, archived, or fallback code inside `D:\Axon`.

Do not rebuild that path silently. Any future representation change must preserve exact character visibility and must be approved in this source of truth first.
