# Axon Source Of Truth

Last updated: 2026-08-25 (Heart identity-generalization evidence and objective v2 recorded)

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

### Autobiographical continuity

Axon's lived history is append-only at the evidence layer. Accepted user turns, Axon responses, tool invocations and results, advisor inputs, canonical state transitions, and governed learning/evaluation outcomes are not silently deleted when they leave active attention. They may become masked/cold, but their exact content, ordering, source identity, outcome, and provenance remain durably recoverable through Dormant State. Rejected or quarantined ingress is also preserved as evidence, but remains explicitly distinguished from accepted lived experience.

The current bootstrap mask implementation may retain cold exact spans inside persisted Shared Field snapshots while deriving a narrower attended view. Mature storage may externalize sufficiently cold spans into exact Dormant containers and rematerialize them by stable identity when an older mask interval is reopened. That storage choice may change; the invariant does not: sliding the governed history mask backward must recover the same exact prior material rather than a summary or regenerated approximation.

Dormant State therefore carries two inseparable forms of memory under one authority: exact episodic/autobiographical evidence and structured semantic knowledge derived from that evidence. Derived facts, procedures, summaries, and semantic edges must keep provenance back to the exact episodes that support them; semantic processing never replaces or erases the lived source record.

`State/dormant/experience_v1` is the exact autobiographical evidence layer. The
seven durable recovered sources from `D:\00` are preserved there as one
byte-exact, content-addressed 3,529,335,677-byte source snapshot and 59,875
individually hash-bound logical records. Record text is not normalized,
summarized, clipped, or substituted for its source bytes. The original
`D:\00` databases remain protected read-only source evidence.

The permanent Heart also deposits every canonically accepted external
user/tool/advisor ingress event into this layer before acknowledging its
durable spool item. Each deposit freezes the exact envelope bytes, lifecycle
`accepted`, canonical region, source/provenance, and ordering identity in an
immutable content-addressed batch. Commit-before-deposit or deposit-before-ack
failure is restart-safe and idempotent: canonical span provenance detects the
accepted event, the same Dormant identities are verified or rebuilt, and only
then is the spool advanced. Automatic deposits for future Axon response
commits, tool invocation requests, consolidator decisions, and governed
learning/evaluation outcomes remain an implementation gap; therefore the
system does not yet claim complete lifelong capture of every event class.

## Semantic Cortex Organ

The Semantic Cortex gives Axon a continuously refreshed semantic understanding
of the universe represented by the current Shared Field and the knowledge and
experience stored in Dormant State. It is not equivalent to vector search,
embeddings retrieval, or the canonical `cortex` region alone. Dormant retrieval
is one Cortex sense; semantic interpretation is the organ's larger job.

The Cortex is also the semantic digestion layer for lived experience. Raw episodes remain exact in Dormant State; Cortex may continuously derive grounded entities, relationships, procedures, causal links, recurring patterns, confidence, and semantic edges from them. Those structures remain provenance-bound to their source episodes and may themselves be stored in Dormant State as structured knowledge. Cortex does not turn an episode into a lossy replacement and does not decide what becomes parametric memory; it supplies semantic organization that both reasoning and Trainer curriculum construction can consume.

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
and organ output outward, roundtrip. The **canonical Shared Field is the truth
body**; the Heart is its sovereign guardian, compiler, translator, and sole
canonical writer. Learned Heart tissue may become extremely capable, but no
neural Heart model is itself canonical truth and no learned output gains
unchecked commit authority. Other organs and ingress paths may originate and
submit proposed mutations; they never mutate canonical state directly. Cores,
consolidators, ingress paths, and the dormant valve all cross the heart's typed
validation/transaction boundary.

### Heart intelligence and semantic conduction

The mature Heart is not merely a deterministic router. It may contain a
Trainer-governed ensemble of learned translation/conduction cores that help it
preserve meaning across many human languages, tool/advisor dialects, Cortex
representations, and heterogeneous neural rail dialects. The deterministic
Heart authority remains wrapped around those learned senses: learned tissue
proposes interpretation/translation; deterministic code binds it to the exact
field, checks freshness/provenance/semantic-fidelity receipts, exposes
uncertainty or disagreement, and alone decides whether a proposed canonical
mutation is admissible.

Heart translation is hub-and-spoke rather than every dialect translating
directly into every other dialect. Each source dialect is decoded toward one
grounded Heart semantic interlingua and each destination is encoded from that
interlingua. The original native message is always preserved beside the Heart's
interpretation. Translation may be classified as exact, semantically equivalent,
approximate, ambiguous, or rejected; approximate/ambiguous meaning is never
silently relabeled exact. Material distinctions such as referent identity,
polarity/negation, modality, quantification, temporal relation, causal relation,
speech act, confidence/ambiguity, and grounding/provenance remain first-class
translation obligations.

Every learned Heart translation is expected to be roundtrip-testable: source
meaning -> destination dialect -> recovered meaning. Semantic roundtrip is not
accepted merely because embeddings are close. Critical semantic distinctions
must survive explicitly, and every transported claim remains anchored to exact
canonical characters/spans or provenance-bearing dormant/Cortex evidence. If a
destination dialect cannot preserve a distinction, the Heart carries the loss
or ambiguity explicitly rather than inventing equivalence.

Mature Heart intelligence uses **at least three accepted serving translation
cores plus at least one isolated candidate-learning lane**. Serving cores may
have heterogeneous architecture, `d_model`, depth, head structure, FFN width,
and training history. Their native interpretations are independent evidence;
disagreement is observable and deterministic Heart policy arbitrates/fails
closed rather than averaging away a material contradiction. Candidate Heart
tissue may train continuously, but it cannot participate in live translation or
canonical commit until it independently passes Trainer promotion and activation.

Heart promotion is intentionally much stricter than a coarse 60/40 capability
score. The current first-form policy contract requires at least 0.9999 grounded
roundtrip, at least 0.999 aggregate held-out semantic fidelity, **1.0 on every
critical semantic class**, explicit counterfactual input-use proof, and zero
regression failures before a candidate can pass the Heart-specific promotion
floor. These are policy floors, not a claim that the present runtime already
possesses such a model. The policy may become stricter as evaluation matures.

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

### Additive developmental anatomy law

This permanence rule applies to every Axon organ, suborgan, core, rail, sense,
compiler, adapter, and learned surface. Before material implementation or
training begins, a developmental part must have either an honest enduring role
in the organism or an explicit composition and versioned-upgrade path into
larger anatomy. A small part may remain as a specialist, join an ensemble,
receive governed adapters, operate on a narrower cadence, or be accompanied by
wider peers. Increased scale is additive growth; it is not permission to
silently relabel, overwrite, or discard previously accepted tissue.

"Upgradeable" does not mean that every experimental candidate must be promoted
or serve forever. Failed and rejected candidates remain immutable learning
evidence. An accepted part may later be retired or moved offline when measured
evidence and an explicit lifecycle decision justify it. What is forbidden is a
campaign knowingly aimed at dead-end anatomy whose learned capacity must be
thrown away merely because the next width, context length, or organ generation
arrives. Canonical State, provenance, interfaces, and unrelated learned tissues
must survive scale-up. Incompleteness and weakness are admissible first forms;
fakeness, concealed replacement, and planned disposability are not.

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

Cores are transformer reasoners with private souls. Axon's mature reasoning ensemble is heterogeneous: cores may have different architectures, specialties, parameter counts, and `d_model` widths while still reasoning against the same frozen canonical Shared Field. Each active width has a **home rail** supplied by the Heart, bound to the same exact field/tick identity and provenance. A core inhales from, attends through, and exhales its native proposal back onto its own home rail; it is not required to speak another width's tensor dialect or directly author canonical serialization. The Heart translates/grounds native rail proposals into the shared proposal/interlingua surface and re-renders other organs' proposals back into each destination home-rail dialect. No core's larger or smaller rail becomes a competing truth body.

A home rail must be roundtrip-capable with respect to its exact canonical scaffold. Different widths may require different physical slot counts for the same story: smaller `d_model` rails can use more slots while larger rails can carry more exact substrate cells and richer derived semantic structure per slot. Larger width is therefore additional representational bandwidth, not permission to discard characters or provenance. A 4096D core may bind broad entity/event/context relationships more compactly than a 64D core while both remain grounded to the same exact field evidence.

Native rail proposals are preserved exactly as emitted. Heart translation may render a proposal very closely into another rail's dialect, but any semantic looseness is receipted rather than erased. Thus cores may genuinely develop different dialects while communication still converges on one grounded meaning space and one canonical body.

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

Soul writes move hot to warm to cold over time. The soul carries private per-core experiential state and short-to-medium-horizon habits/intuition. Slower generalized procedural intuition may also be distilled into governed core parameters by the Trainer from repeated lived episodes. Auditable factual/episodic knowledge belongs in Dormant State; neither soul state nor weights are allowed to become the only copy of evidence that should be recoverable exactly.

## Trainer Organ (Parameter Guardian)

The Trainer is a permanent organism subsystem, not merely a command-line script used to create seed models. Its sovereignty is over **parameter state and learning lineage** in the same way that the Heart's sovereignty is over canonical Shared Field state.

The Heart remains the sole canonical Shared Field writer. The Trainer becomes the sole governed authority that may create or mutate candidate model parameters, optimizer state, LoRA/adapters, or other learned parameter-bearing generations. Reasoning cores, semantic cores, Cortex, tools, and Trainer advisory cores may request or recommend learning; they do not directly own unrestricted backpropagation or parameter writes.

Exactly one active Trainer writer may govern `State/training/trainer` at a time. `runtime/trainer/lease.py` supplies an OS-backed exclusive writer lock plus human-readable ownership metadata; stale metadata cannot create a second writer, and corrupt lease evidence fails closed. Read-only Trainer inspection does not require the writer lease.

Every live parameter-bearing organ must eventually register with the Trainer. The Trainer maintains a complete model-state inventory containing module identity, organ role, architecture, `d_model` width, generation identity, parameter names/shapes/dtypes/trainability, persistent buffers, and exact lineage fingerprints at governed boundaries. Persistent buffers are part of learned model state even though they are not optimizer parameters; an ungranted buffer mutation therefore fails closed. A declared organism inventory that is missing an expected parameter-bearing module is incomplete and may not authorize a training mutation.

The Trainer's learning state is durable beneath `State/training/trainer`. It records immutable inventories, candidate-generation lineage, mutation plans and authorization receipts, source/curriculum manifests, holdouts, optimizer and schedule configuration, telemetry, checkpoints, evaluations, rejected generations, promotion proposals, active-generation pointers, exact generation snapshots, activation receipts, active adapter ancestry, and rollback receipts/targets. Parameter history must be inspectable rather than mystical.

The Trainer must expose transparent telemetry for every parameter under its authority. At minimum, each governed training step/cadence must be able to report per-tensor value and gradient health, norms/RMS/extrema, finite/zero fractions, trainability, immutable learning-policy identity, optimizer/schedule state, microstep/accumulation state, active grants, candidate generation, data provenance, evaluation state, precision mode, and parameter/update budgets. Exact full tensor hashes are required at lineage/checkpoint/promotion boundaries; continuous telemetry may use bounded numeric summaries while still enumerating every parameter.

Training is branch-like. A live accepted generation is never silently edited in place. Learning creates an isolated candidate generation from an explicit base inventory, with declared writable tensors and budgets. Promotion requires held-out and counterfactual evidence, lineage receipts, regression/forgetting checks, and a separately governed activation step. The atomic parameter-state commit is the Trainer-owned active-generation pointer: before it advances, the current live generation is snapshotted as an exact rollback target, the candidate is verified off to the side, the live module is loaded and re-hashed, and a normalized active-generation snapshot is persisted. Any failure before pointer publication restores the previous live state. Rejection preserves evidence; it does not erase the failed generation from learning history.

LoRA/adapters are first-class governed parameter generations, not a loophole around parameter authority. The Trainer may issue adapter-only grants that fail closed if a plan attempts to touch base parameters. Online/inference-time adaptation must be more tightly budgeted than offline learning and is never allowed to bypass source provenance, telemetry, holdouts, rollback, or promotion rules.

The Trainer may eventually contain its own ensemble of Transformer cores. Candidate advisory roles include curriculum construction, optimizer/gradient control, evaluation, catastrophic-forgetting audit, and promotion criticism. These Trainer cores may have different `d_model` widths and specialties, but their outputs are advisory proposals. A deterministic Trainer authority layer validates the exact parameter inventory and grant before any optimizer/backpropagation path is allowed to mutate tensors.

Trainer cadence is distinct from heartbeat, Cortex tick, and reasoning tick. The Trainer may run sustained offline learning, bounded online adaptation, continuous parameter-health observation, or study campaigns such as "learn philosophy". Study acquisition enters Dormant State with provenance first; Cortex may semantically organize it; Trainer then constructs governed curricula/candidates and evaluates them before any learned generation can become active.

The Trainer is also Axon's **lived-experience compiler**. It must not blindly stream the Dormant corpus into gradient descent. It selects provenance-complete episodes and constructs runtime-faithful curriculum examples from what Axon actually experienced: the pre-action Shared Field and Cortex context, available tools/advisors, proposals/deltas/actions, tool consequences, later corrections, test/evaluation evidence, and eventual outcome. Successful and failed episodes are both valuable. The Trainer may build paired corrections, replay tasks, counterfactuals, delayed-outcome examples, retrieval-use examples, and multi-episode curricula, but every derived training target remains linked to the exact source episodes that justify it.

The ratified developmental learning sequence is functional Heart circulation
first, followed by Semantic Cortex and reasoning-core learning from exact,
provenance-bound lived episodes in Dormant State. This sequencing does not make
the Heart disposable pretraining infrastructure: its translations,
disagreements, corrections, and observed consequences become part of the same
experience record and can improve later Heart generations as well as Cortex and
reasoning cores.

A Trainer-built learning session is a durable, inspectable organ product rather
than an ad hoc batch. The Trainer selects eligible episodes, reconstructs the
pre-action canonical context and actually available senses/actions, derives
targets only from outcomes, later corrections, and cited evidence, and marks
observation separately from inference. It then constructs balanced splits,
replay, counterfactual, delayed-outcome, retrieval-use, and forgetting probes;
chooses an eligible candidate organ and non-disposable capacity contract;
obtains the mandatory preflight receipt; trains only an isolated candidate;
evaluates and promotes or rejects it through Trainer authority; and deposits
the session manifest, lineage, evidence, and outcome back into durable history.
Trainer advisory cores may propose curricula and session plans, but deterministic
Trainer authority validates sources, permissions, capacity, splits, and
promotion. Neither a curriculum nor learned weights may replace the exact
Dormant episodes from which it was derived.

`runtime/trainer/sessions.py` implements the first deterministic durable
lived-experience session compiler. It verifies an exact experience import and
publishes content-addressed Trainer manifests whose examples reference source
record identities and complete context ranges rather than copying or clipping
text. Stable SHA-based 80/10/10 train/heldout/regression assignment prevents
split drift. The recovered `D:\00` import currently yields a 59,858-example
Heart grounding session and a 14,205-example observed conversation session.
Observed assistant responses are explicitly labeled as observations whose
correctness is not inferred, and the conversation session cannot authorize
serving promotion. These manifests are governed curriculum inputs; direct
dataset-loader/preflight integration and outcome-based target curation remain
future work.

The purpose of lived-experience training is primarily **procedural compression**: reasoning habits, tool-use instincts, error avoidance, planning patterns, semantic discrimination, confidence calibration, and other generalized intuition that should become easier because Axon has encountered similar situations before. Parameters are not required to memorize every factual detail. Exact facts, versions, identities, conversations, source material, and auditable outcomes remain in Dormant State and can be surfaced by Cortex when needed. In mature operation, weights should answer roughly "how have situations like this tended to work?" while Dormant State + Cortex answer "what exactly happened, what is known now, and what evidence supports it?"

Core diversity should emerge naturally from governed variation in lived-experience sampling, temporal windows, curriculum order, objectives, initialization, adapters, architecture/width, and replay/counterfactual emphasis. Multiple generalist reasoning or semantic cores may therefore learn overlapping life history through different lenses and acquire different useful intuitions without requiring every core to be narrowly labeled "coding", "math", or "science". Explicit specialist cores remain optional additions, not the only path to ensemble diversity.

Steady-state Axon should normally keep at least one **isolated non-live candidate learning lane** active on admissible lived-experience or study curriculum while other cores serve the organism. "Always learning" never means forcing meaningless gradient steps: if no curriculum passes provenance/quality gates, that lane remains occupied with curation, replay construction, evaluation, or forgetting analysis until admissible learning material exists. The live accepted cores remain immutable until a candidate independently passes Trainer gates and activation.

**Current first learned-organ priority is the Heart translation/conduction ensemble.** Reasoning and Semantic Cortex training remain behind it. The Trainer recognizes Heart translation cores/adapters as explicit parameter-bearing organ kinds. `runtime/heart/translation_core.py` is now the first permanent learned Heart tissue: a 64D, two-layer, four-head, 4096-FFN translator grounded from the frozen 16D character substrate, with explicit semantic, referent, and grounding heads; it has no canonical-write authority. Architecture v3 has no learned or validated source/target character ceiling. A configurable physical page is only a processing unit: two ordered recurrent sweeps visit every exact source character, the second sweep builds full addressable character memory from a query state that has already traversed the complete source, and a coverage record binds per-row source-index hashes, page spans, and visited counts. Source and decoder positions are deterministic sinusoidal functions rather than finite learned tables. `runtime/heart/d64_codec.py` freezes each actual `SharedFieldSnapshot` through the exact and semantic D64 compilers, verifies exact roundtrip and grounding, and supplies the Heart with the literal raw 16D lane cells plus monotonically increasing canonical character positions. Masking earlier spans therefore cannot renumber later active text. A substituted lane cell, stale field/rail/surface identity, or proposal not bound to the frozen frame fails closed. `training/heart_translation.py` materializes every provenance-labeled structured-proposition curriculum case as a real Shared Field and real D64 frame before model input; it provides disjoint heldout/regression/counterfactual suites, semantic/grounding evaluation, and an immutable content-addressed task-loss objective. Curriculum v3 includes train and held-out complete-field cases for every critical semantic class whose grounded spans begin beyond character 256. Its current training recipe uses deterministic shuffled epochs: every case is visited once before reshuffling, and the immutable recipe identity is bound into candidate generation and source lineage. A first real Trainer-governed 12-step CUDA smoke used the obsolete fixed-192 architecture v1; it lowered loss from 4.9502 to 4.3422 and moved some semantic submetrics, but grounded roundtrip and aggregate semantic fidelity remained 0.0, so the candidate was rejected and no activation/promotion proposal occurred. Its immutable artifacts remain historical evidence and are not compatible with v3.

The current real-D64 v4 diagnostic culminated in a governed 512-step, batch-8
CUDA candidate (`run_id`
`9257422e04d5f23b80ccfb0550742212d47f1fa84f9c0278696ed69784010fc6`).
Loss fell from 4.9441 to 2.6819; held-out termination reached 0.8182,
source-semantic exactness 0.2857, referent-pointer exactness 0.5844, and
grounding-pointer exactness 0.6753, while regression failures fell to 14.
Exact translation, grounded roundtrip, and aggregate semantic fidelity remained
0.0, so both independent promotion gates rejected the candidate and no active
generation pointer or promotion proposal exists. Evaluation v4 preserves every
generated held-out string and its per-case decisions rather than only a digest.
The verified re-evaluation artifact
`f28e035c07c644c92b440d48d87835b0d200fef9be15d00032eca630ec52ea9f`
shows repetitive free-running character sequences despite improving
teacher-forced loss and pointer accuracy. Checkpoint-bound decoder diagnostic
`b75409895073862643225e72e25c4c104be9b6bec87f77c8495842985edb473c`
then localized the failure: teacher-forced character accuracy was only 0.2289,
teacher-forced sequence exactness and greedy exactness were both 0.0, mean
correct greedy-prefix fraction was 0.0202, and target-character attention mass
was 0.0959. The problem therefore began before free-running exposure collapse;
the supervised decoder mapping and copy alignment themselves were weak.

The separate content-addressed decoder-mechanism curriculum now supplies exact
copy/alignment cases from five characters through multi-page strings, with
complete-field train and holdout referents beyond character 256. It is a
mechanism gate and never substitutes for the semantic/counterfactual Heart
curriculum. `scripts/train_heart_decoder_mechanism_smoke.py` resumes only from a
verified checkpoint, passes the exact curriculum through the complete-field
preflight firewall, trains only an isolated Trainer candidate, records
checkpoint-bound teacher-forced/greedy/copy-gate/alignment diagnostics, and can
never activate a Heart generation.

The staged governed campaign proved the permanent D64 anatomy can learn exact
copy across a physical-page boundary. One-case run
`e24ca23af9dc57e4ab500c61def327cd2c6cbdc7f0b5e3503c49e90387afc560`
and five-case run
`7843b66c315e5fd31b7094a952b8ba9e4e31bda9eebebf724571c3ec16ac39c2`
passed their strict mechanism gates. The first six-case run
`1b1964d0a47757af2ff0e5a4b3adf6267c84e2ba79e7312e18d5a7be7f341861`
was correctly rejected because the 374-character case missed one of 374
teacher-forced characters and therefore was not free-running exact. A bounded
64-step continuation,
`c6393e28539da07bc40a0b62488099ec6592400cd7fcd9e466017c9e01e74482`,
passed: all six training sequences were exact under teacher forcing and greedy
decoding, EOS and termination were perfect, mean copy-route probability was
0.9489, and mean matching-character attention mass was 0.9767. No serving
activation occurred. This is learned complete-field mechanism evidence, not a
claim of learned Heart function.

Generalization remains the current boundary. The content-addressed identity
curriculum now contains 37 deterministic training cases, 21 disjoint held-out
cases, six replay cases preserving the proven mechanism, all 95 registered
characters, lengths through 521 characters, multi-page train/held-out material,
and paired head/middle/tail single-character counterfactuals. Training is
length-bucketed without cross-bucket padding; every case is visited before an
epoch repeats. Its strict non-serving gate requires unseen character/EOS/free-
running/termination performance, diagonal alignment, all three source-change
counterfactuals, no train-output collision, and exact replay retention.

The first governed identity candidates were all correctly rejected and never
activated. The 128-step candidate `h64g-1b219088d306` completed optimization
but its post-training gate call failed due an orchestration control-flow defect;
the immutable step-128 checkpoint remained valid, its lifecycle remained
rejected, and recovery evaluation
`99cbe9653905b6b5ca7ad9bc99b157af4efbe1d968f6bf7aed938b58a5c7f931`
recorded 0.0729 held-out character accuracy, 0.0477 diagonal mass, one of three
counterfactual pairs and five of six replay cases. Recovery did not rewrite the
lifecycle or create an activation path.

A later 256-step objective-v1 continuation raised held-out character accuracy
to 0.1249 and diagonal top-one alignment to 0.1000, but termination fell to
0.6667. The cause was explicit in the loss: copyable characters were taught to
use the copy route, while EOS had no separately balanced route target. Objective
v2 therefore preserves discrete per-character NLL, strengthens identity-only
diagonal alignment, and separately teaches EOS to return to the generation
route. Replay gates now also require character accuracy, EOS and termination,
so one short exact case cannot hide damage to the 374-character replay.

The final bounded objective-v2 run
`f05e47fcc2d16a747514eb2b7976159abb9f85777397e12996a47b07c6112d7d`
reduced the fixed audit loss from 7.7983 to 6.6192, raised held-out character
accuracy from 0.1179 to 0.1544, diagonal mass from 0.0780 to 0.1015, diagonal
top-one rate from 0.1076 to 0.1510, and retained 0.9524 termination. It still
achieved only 0.0476 held-out greedy exactness, one of three counterfactual
pairs, and five of six exact replay cases; the long replay character accuracy
was only 0.6013. The candidate was rejected and not activated. Falling audit
loss and improving partial metrics are learning signal, not learned-Heart
function. The next experiment must target long-position alignment and tail
source dependence explicitly while protecting EOS and full long replay; blind
step doubling is not warranted. Only after identity conduction passes should
training return to semantic translations and their counterfactual/roundtrip
floors. These results still do not justify discarding the permanent D64 tissue.

`runtime/trainer/` now implements the first governed candidate-generation lifecycle behind that control plane. A live registered organ is never handed to an optimizer: the Trainer creates an isolated candidate clone, freezes tensors outside the exact mutation grant, enforces the authorized step/parameter budget, rejects non-finite loss or gradients before contamination, hashes live and unauthorized state around each step, and records full per-parameter telemetry. `runtime/trainer/learning.py` defines an immutable content-addressed learning policy bound to every candidate checkpoint/step. Current first-form execution governs AdamW or SGD, weight decay, Adam betas/epsilon, SGD momentum, constant or warmup-cosine scheduling, gradient accumulation, gradient clipping, hard gradient/update L2 budgets, and explicit FP32/BF16/FP16 precision. FP16 requires CUDA and uses a governed GradScaler; non-finite gradients still fail closed before optimizer mutation. Persistent buffers are inventoried and any buffer mutation fails closed until a future explicit buffer-state grant is designed.

Candidate checkpoints are atomic, SHA256-bound, generation/plan/authorization/learning-policy-bound, and exactly restorable inside the candidate branch. A mid-accumulation checkpoint preserves optimizer state, pending authorized gradients, AMP scaler state when present, optimizer/microstep counters, accumulation index, current learning rate, and accumulated loss sum so resume does not silently discard or misreport learning state. Scheduler state is deterministic from the immutable policy plus optimizer-step count rather than hidden mutable scheduler objects. Deterministic promotion gates require predeclared evaluation suites and metric thresholds; missing capability, counterfactual, regression, or forgetting evidence blocks promotion. A passed gate may create a promotion proposal, but only the leased deterministic Trainer authority may activate it. Activation requires an exact current inventory, passed gate/proposal/checkpoint lineage agreement, architecture-compatible generation transition, exact candidate tensor/buffer verification, a durable pre-activation rollback snapshot, and atomic active-pointer publication. Runtime trainability flags are preserved independently from temporary candidate freeze policy. Rollback restores the exact previous generation and snapshots the displaced generation so rollback itself remains reversible. `hydrate_active_generation()` rehydrates a freshly registered organ from the durable pointer after restart. `scripts/inspect_trainer.py` provides read-only visibility into current lease ownership, lifecycle, optimizer-step receipts, per-parameter telemetry, checkpoints, active-generation pointers, activation/rollback evidence, evaluations, gates, and promotion proposals. These mechanisms are proven with unit-scale synthetic candidates only; no semantic-core or reasoning-core training campaign is launched by this milestone.

`runtime/trainer/preflight.py` now makes the non-disposable capacity law part of
Trainer authority. `TrainerControlPlane.authorize()` and
`begin_candidate()` require a passed content-addressed preflight receipt; the
authorization identity itself binds that receipt, and the Trainer persists the
exact capacity contract and receipt before an optimizer candidate can exist.
`training/heart_preflight.py` is the first organ-specific implementation. It
performs semantic Python/config inspection for finite learned positions,
page/position aliasing, destructive text slicing, tokenizer truncation and
forgiving checkpoint loads; certifies Heart curriculum length/page/grounding
distributions; executes page-boundary complete-coverage and head/middle/tail
input-dependence probes; verifies strict incompatible-anatomy rejection; and
declares physical-page, batch and optimizer-step controls as source-preserving
work bounds. Passing this launch gate is anatomy evidence, not serving
capability or permission to activate the candidate.

Heart source-coverage receipts are computed from the page spans and character
counts actually visited by each ordered sweep, not inferred merely from input
length. Bounded greedy translation returns explicit per-item termination state;
an item that exhausts its caller-selected compute budget is incomplete, is not
used as a completed semantic roundtrip, and cannot satisfy Heart promotion.

## Training Contract

Training must match runtime:

- curriculum enters through the shared field,
- unused regions are masked,
- core inhales,
- core attends,
- core emits deltas,
- loss is applied to the delta/response target,
- core exhales after action.

### Non-disposable capacity law

No learned-organ campaign may spend material compute on anatomy that is known
to be incapable of Axon's intended runtime contract. Phase-zero development may
reduce model width, parameter count, dataset size, batch size, optimizer steps,
precision, concurrency, and execution speed. It may not introduce a fixed
character/context ceiling, finite learned position table, page-index alias,
silent tokenizer/collator truncation, long-item exclusion, or other capacity
limit that would require discarding the resulting checkpoint when the complete
field is exercised. A physical page is a processing unit, never a declaration
of how much canonical reality the trained organ can address.

Finite hardware requires finite work controls. Every such bound on a live or
training path must be declared in a machine-readable registry and classified as
a physical processing unit, compute/optimization budget, admission budget, or
result-count policy. The declaration must state what source is preserved, how
work continues or resumes, and how the path fails closed when completion is not
achieved. An undeclared bound, a partial result presented as complete, or a
bound baked into checkpoint capacity is a launch-blocking defect.

Before any Trainer-authorized parameter mutation, a deterministic
Complete-Field preflight must issue a content-addressed receipt bound to the
exact module/architecture configuration, compiler and positional schemes, base
inventory and candidate lineage, curriculum and holdout manifests, declared
bounds, and executable evidence. That evidence must include static source and
configuration inspection; curriculum length/page/grounding distributions;
boundary and beyond-page cases; exact coverage identities; head/middle/tail
counterfactual dependence; and checkpoint/resume compatibility. Trainer must
refuse mutation if the receipt is absent, stale, incomplete, or mismatched.

Checkpoint restoration is exact-lineage restoration, not architectural
adaptation. Missing-key, unexpected-key, shape-mismatch, positional-table, or
compiler-contract forgiveness may not turn an obsolete checkpoint into current
tissue. Incompatible checkpoints are preserved immutably as historical or
behavioral evidence and machine-marked non-resumable/non-activatable. They may
become donors only through a separately authorized, measured experiment that
creates a new lineage and never represents the donor as a valid resume.

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

The active developmental 64D reasoning reader follows the same permanence rule.
`training/complete_field_64d.py` uses deterministic unbounded local/page
positions and a complete ordered page sweep; page indexes never wrap or alias
through a finite embedding table. Teacher-forced targets have no configured
character ceiling. Greedy inference may use an explicit caller-selected compute
budget, but that budget is not parameter capacity, may be raised without
changing a checkpoint, reports nontermination, and an unfinished result is not
committed as a canonical delta.

Candidate checkpoints and reproducible run logs belong beneath
`State/training/runs/` while non-authoritative. Promotion moves or copies an
accepted state-bearing artifact into its governed canonical State location with
explicit provenance.

## Dormant Evidence Bridge

The canonical read-only retrieval/surfacing organ is implemented in
`runtime/dormant/evidence_bridge.py`. For its recovered semantic corpus it
treats the container/edge JSONL files under `State/dormant` as authority.
`State/dormant/experience_v1` is a second exact authority under the same
Dormant organ for byte-preserved source snapshots and autobiographical records;
the current evidence bridge does not yet index or surface that experience
layer. Neither authority is a disposable derived index.

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

Active recovered-corpus tooling preserves the full normalized source value for
messages, diary entries, episodes, goals, procedures, relations, metadata, and
surfaced Cortex examples; it does not slice source records to field-width or
training convenience limits. Derived lexical bucket keys and content-addressed
IDs may be short because they are indexes, never substitutes for source text.
Dormant lexical queries process every unique term in ordered SQL-safe pages;
the former 128-term rejection is removed. Recall item/result budgets remain
explicit governed selection policy: skipped items are identified and the
authoritative dormant record remains exact and retrievable; selected text is
never partially truncated.

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
- `runtime/dormant/experience.py`, `runtime/dormant/evidence_bridge.py`, `runtime/dormant/relevance.py`, `runtime/dormant/generations.py`, `runtime/dormant/incremental.py`, and `runtime/dormant/evaluation.py` ? immutable content-addressed exact experience/source snapshots plus read-only manifest/hash-bound dormant retrieval, exact dereference, bounded graph/relation relevance, verified derived-index generations, transactional append/layout-preserving update maintenance, and held-out evaluation,
- `Cortext/contracts.py` ? grounded Semantic Cortex service contract only; no active specialist/training authority and the `semantic_cortex` valve remains CLOSED,
- `runtime/heart/` ? heart anatomy: authority/core control plane, canonical transaction boundary, beat coordinator, sovereign 20-slot valve plane, OS single-writer lease, restart-safe cardiac identity, durable ingress/replay/quarantine spool, crash-safe exact autobiographical ingress deposit, health observability, explicit derived-view identity, relevance-gated dormant recall, permanent Heart host, `runtime/heart/intelligence.py` for learned Heart identity/fidelity/promotion contracts, `runtime/heart/d64_codec.py` for literal real-field D64 framing, and `runtime/heart/translation_core.py` for the first permanent non-authoritative 64D neural translator tissue; no learned Heart translator is serving/active yet;
- `runtime/trainer/` ? permanent Trainer parameter-authority anatomy: heterogeneous parameter+buffer inventory, OS single-writer lease, scoped mutation grants, immutable content-addressed learning policies, isolated candidate optimizer execution with governed accumulation/scheduling/precision/budgets, per-parameter telemetry, exact mid-accumulation checkpoint/restore, deterministic promotion gates, atomic active-generation pointers, exact activation/rollback snapshots and receipts, restart hydration, deterministic content-addressed lived-experience sessions, immutable lifecycle records, and read-only inspection; no model is trained or activated without an explicit governed plan/policy/gate/activation path;
- `scripts/run_axon_heart.py`, `scripts/evaluate_dormant_relevance.py`, `scripts/maintain_dormant_index.py`, and `scripts/verify_d64_dual_surface.py` ? permanent Heart runtime, deterministic dormant semantic/relevance evaluation, explicit derived-index maintenance/recovery, and read-only live D64 dual-surface verification entry points;
- `training/canonical_d64.py`, `training/complete_field_64d.py`, and `training/train_complete_field_64d.py` ? developmental canonical D64 reasoning path; `training/heart_translation.py` plus `scripts/train_heart_translation_smoke.py` ? real-field-D64 Heart translation curriculum/evaluation and Trainer-governed bounded candidate smoke path with content-addressed task objective and no activation,
- `curator/import_d00_memories.py` and the remaining `curator/` recovered-corpus utilities ? protected-source, byte-exact autobiographical import plus offline exact dormant-memory schema/materialization/building tooling; `scripts/compile_lived_experience_sessions.py` ? deterministic governed session compilation from exact Dormant experience,

The former council, old core/soul implementation, ExactV4/identity-v2 runtime stack, 384-slot views/schedules, legacy trainers/curricula, launchers, policies, and dedicated tests are archived beneath `archive/day_zero_legacy_2026-08-20/`. Local historical runs, datasets, checkpoint bundles, and generated distributions are preserved beneath `State/archive/day_zero_legacy_20260820/local_artifacts/`. They may be inspected for provenance or mechanism recovery but may not be imported, launched, resumed, or presented as current Axon without a new explicit convener decision.

There is one Source of Truth text. `docs/SOURCE_OF_TRUTH.md` is the master path and root `SOURCE_OF_TRUTH.md` is a byte-for-byte compatibility mirror. Any doctrine update must update both in the same change; repository tests enforce equality. `docs/WORKING_CONTRACT.md` and root `WORKING_CONTRACT.md` follow the same exact-mirror rule.

## Deleted Architecture

The wide-slot adapter lineage is deleted from this repo. It is not stale, experimental, archived, or fallback code inside `D:\Axon`.

Do not rebuild that path silently. Any future representation change must preserve exact character visibility and must be approved in this source of truth first.
