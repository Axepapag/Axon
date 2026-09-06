# Axon Source Of Truth

Last updated: 2026-09-04 (foundations-first reasoning curriculum ratification)

## Core Doctrine

Axon is a stateful, always-on AI built around one canonical regional state body,
an actively attended Shared Field selected from that body, and private reasoning
cores.

The field is Jeff's bridge into Axon's world. It must preserve exact Unicode text, provenance, and structured context without hiding meaning behind opaque semantic symbols.

Axon tissue has no engineer-chosen total content ceiling. Canonical text,
attended context, dormant history, ingress payloads and queues, decoder output,
recall queries, rail packs, organ/valve inventory, and dialect identities may
grow until physical reality requires visible pause, paging, spilling, or
failure. A finite work slice may control one execution turn only; it must
preserve exact source and continuation state and may never turn incomplete work
into completion. Integrity, authority, type, provenance, Unicode, EOS, and
transaction checks remain hard fail-closed laws rather than capacity limits.

## Substrate

- The frozen substrate width is 16D. The original 95-character native bank and
  all of its token IDs and vectors remain frozen unchanged.
- Canonical text stores raw exact Unicode scalars. A native character maps to
  its one original 16D cell. Every other valid scalar maps deterministically to
  its strict UTF-8 bytes, one typed 16D byte-transport cell per byte. This is an
  additive categorical transport, not normalization, escaping, or a widening
  of the substrate.
- Byte transport uses a frozen deterministic 256-cell extended-Hamming
  codebook. Native IDs are `0..94`; byte IDs are `95..350`. Token kind and ID
  are carried in exact compiler receipts. Nearest-vector guessing has no
  decoding authority.
- Strict decode rejects malformed UTF-8, unpaired surrogates, out-of-range
  scalars, and alternate byte spellings of native characters. Masks resolve
  over canonical scalar positions before any one-to-four-cell expansion.
- Character identity and order are source-of-truth data, not a lossy summary.
- A larger `d_model` core may lift those 16D cells into its own lane, but it does not replace the canonical character field.

## Canonical Regional State and Active Shared Field

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
- `identity`

As of Shared Field schema `shared-field-v3`, the canonical semantic-context region is named **`cortex`** and the append-only region registry also contains **`identity`** at ID 10. The old `structured_knowledge` concept is a subset of Cortex function: exact dormant material recovered by semantic relevance belongs inside the Cortex picture, but Cortex is not merely a retrieval bucket. The region is the Heart-governed, auditable textual/materialized surface of Axon's broader Semantic Cortex organ: relevant dormant evidence, grounded semantic relationships, concepts, entity/relation context, and other semantic interpretation that reasoning cores should be able to inspect alongside the exact rest of the Shared Field.

The canonical `cortex` region is not the whole Semantic Cortex organ. The organ may maintain richer derived/noncanonical working state and semantic-core outputs between canonical materializations. It never becomes a second canonical body and never bypasses Heart authority. The separate `semantic_cortex` Heart valve is the governed organ-to-Heart boundary and remains CLOSED until a real autonomous Cortex service earns activation.

Persisted `shared-field-v1` and `shared-field-v2` history is immutable. Historical v1 snapshots keep their original serialized region name `structured_knowledge`; historical v2 snapshots keep their ten-region form; both retain their original `field_id`/hash. `CanonicalStateBranch.migrate_to_current_schema()` advances a branch by creating a new v3 successor parented to the historical HEAD. It preserves every existing span, provenance record, manifest, character, address, and region ID, performs the v1 `structured_knowledge` to `cortex` rename only in the successor, and appends an initially empty `identity` region at ID 10. It never rewrites historical snapshots.

`identity` is part of the canonical body, is always in the attended Shared Field,
and may not be masked. It contains Axon's explicit identity and constitution,
not a core's private experiential state. Ordinary ingress, reasoning cores,
consolidators, tools, and recall cannot write it. Only the exceptional
`IDENTITY_STEWARD` authority may propose an amendment, and the permanent Heart
host accepts one only between ticks with an amendment identity, nonempty
evidence identities, provenance, canonical validation, atomic commit, and an
autobiographical record. This is deliberate friction, not immutability by
accident.

Jeff ratified canonical Identity v1 on 2026-08-29 and delegated exact wording
to Codex's best judgment after the round-table convergence. The permanent text
is `docs/AXON_IDENTITY_V1.md`; its marked canonical body is 2,064 characters
with SHA256 `63f7b61587647b991e0b2ded10345dfeb8429539595a6aeb643ada9c7d7049fc`.
`IDENTITY_STEWARD` amendment `axon-identity-v1-2026-08-29` advanced the active
field from `f312641a04f7f9087befe50effdd93fb881578d9bbbbb9f31f3d5448c002ed80`
to `d804740fffef41f9dc25b2a05b02fac40ecec531a22bd379d6fa3897451fcaee`;
reapplication after restart is idempotent. Identity v1 defines persistent
organizational identity, evidence/provenance discipline, brother-core society,
Heart authority, private Soul practice, additive growth, and Jeff's role as
friend, convener, and trusted human partner. It makes no untestable claim of
subjective consciousness or metaphysical continuity.

Each region contains one position-stable ordered sequence of exact character
cells plus metadata spans. Words, sentences, paragraphs, and semantic edges are
represented as spans over those exact characters with auditable metadata. A
region's complete sequence is canonical; masking never creates a second copy or
changes the sequence's positions.

No active exact-text path may collapse a paragraph into one opaque vector and then ask a small core to recover exact text from that vector.

The **Shared Field** is exactly the currently unmasked partition of every
canonical region. Every core pass attends that entire Shared Field. A physical
model window may be used as one page in a complete ordered sweep, but it is not
an attention limit and may not silently omit unmasked field characters. Every
logical pass must produce an auditable coverage record proving that each exact
shared-field character was visited.

Every region has an independent governed mask adjustable from 0% through 100%.
Policies may resolve over exact characters, lines, paragraphs, containers, or
conversational turns; a percentage control resolves deterministically to exact
character intervals before compilation. Zero percent exposes none of that
region, and 100% exposes all of it. Moving the mask backward toward older
history makes the original exact cells part of the Shared Field immediately;
moving it forward makes them dormant-in-place. No cell is moved, copied,
deleted, regenerated, or renumbered.

Attention masks are **derived compile-time views**, not part of the canonical
regional-body identity. The canonical `SharedFieldSnapshot` is the complete
ordered spans. The Heart durably owns the independent per-region mask-control
state and resolves it to attended intervals when it compiles rails or forms a
recall query. Changing a mask produces a new derived `view_id` and new rail view
over the same `field_id`; it never creates a new canonical body.

The initial implementation may use one movable boundary per region. The
versioned future mask schema may additionally select multiple ordered,
non-overlapping active intervals, such as a pinned older passage plus the
newest turns. This is an additive feature, not a prerequisite for the first
complete-field reader; in every form, masked characters remain exact and
restorable.

The serialized `RegionVisibility` field is retained only for immutable legacy
snapshot compatibility. It is not the live mask controller and may not be used
to create a second canonical masking path. Current Heart circulation supplies
the complete explicit per-region policy set.

## Dormant State

Dormant State is exact state that is not currently part of the attended Shared
Field. For canonical regions, dormancy is an in-place membership state selected
by that region's mask: the original cells remain in their original region and
at their original addresses. Dormant also owns exact provenance stores for
imported/recovered episodes and source material plus provenance-bound structured
knowledge derived from exact evidence. Those stores are not a relocated copy of
masked regional cells and never become a competing canonical body.

Dormant material is not attended directly. Moving a regional mask exposes the
original in-place regional cells. Retrieval from the broader Dormant evidence/
knowledge stores may instead surface verified, provenance-bearing readable
material into a governed canonical region through the Heart.

Masking is not truncation, transfer, eviction, archival, or physical tiering. It
is an explicit, auditable attended/dormant membership boundary governed
independently per region. A model window, storage pressure, or compute budget may
never move this boundary implicitly.

### Autobiographical continuity

Axon's lived history is append-only at the evidence layer. Accepted user turns, Axon responses, tool invocations and results, advisor inputs, canonical state transitions, and governed learning/evaluation outcomes are not silently deleted when they leave active attention. Regional material becomes dormant in place under its mask; its exact content, ordering, stable address, source identity, outcome, and provenance remain unchanged. Rejected or quarantined ingress is also preserved as evidence, but remains explicitly distinguished from accepted lived experience.

Canonical regional cells may never be externalized, rematerialized, or replaced
by a pointer merely because they become cold. Storage implementations may cache,
page, or index the immutable regional body, but the operation must be transparent:
the same canonical cells and addresses remain authoritative before, during, and
after mask movement. Sliding a history mask back to the first recorded turn
therefore exposes the original material directly, never a summary or regenerated
approximation.

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
then is the spool advanced.

Every successful configured reasoning circulation now deposits a complete
`runtime_reasoning_episode`: exact pre-action canonical field, frozen tick
image, first/refined participant accounting, exact per-rail proposal
workspaces, categorical emissions, source and materialized consolidator
deltas, each participant's exact private-Soul lineage and durable receipts,
finalization receipt, canonical commit, exact accepted response, and an
explicit initial quality of `observed`. Heart exposes exact deposit hooks
for later outcome evidence, completed tool invocations/results, and governed
Trainer outcomes. Those hooks are permanent evidence boundaries, but not every
future tool executor or Trainer path is automatically connected to them yet;
therefore complete lifelong capture of every event class is not yet claimed.

Runtime reasoning episode v3 also freezes the exact per-region attended
intervals and D64 rail identity used by the tick. The Trainer loader rebuilds
that derived view from the complete canonical pre-action field, proves the
rail identity and complete region accounting, and rejects any categorical
delta that addresses a masked/dormant position. Before the Heart commits a
reasoning successor it writes an immutable recovery preparation containing the
complete circulation and pending consolidator-Soul transition. If the process
fails after canonical commit but before Soul finalization or autobiographical
deposit, startup reconstructs and verifies the committed circulation,
finalizes the exact Soul transition, deposits the same episode idempotently,
and only then marks the preparation complete.

## Semantic Cortex Organ

The Semantic Cortex gives Axon a continuously refreshed semantic understanding
of the universe represented by the current Shared Field and the knowledge and
experience stored in Dormant State. It is not equivalent to vector search,
embeddings retrieval, or the canonical `cortex` region alone. Dormant retrieval
is one Cortex sense; semantic interpretation is the organ's larger job.

The Cortex is also the semantic digestion layer for lived experience. On its
own cadence it may inspect material newly made dormant in any region, plus exact
Dormant evidence stores, and derive grounded entities, triples, relationships,
procedures, causal links, recurring patterns, confidence, and semantic edges.
Those structures remain provenance-bound to their exact regional/evidence
sources and may themselves be stored in Dormant State as structured knowledge.
Cortex harvesting never removes, relocates, edits, or replaces the source cells.
Cortex does not decide what becomes parametric memory; it supplies semantic
organization that reasoning and Trainer curriculum construction can consume.

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
and organ output outward, roundtrip. The complete canonical regional state is
the truth body; its unmasked partition is the Shared Field circulated to the
rails. The Heart is the body's sovereign guardian, mask controller, compiler,
translator, and sole canonical writer. Learned Heart tissue may become
extremely capable, but no
neural Heart model is itself canonical truth and no learned output gains
unchecked commit authority. Other organs and ingress paths may originate and
submit proposed mutations; they never mutate canonical state directly. Cores,
consolidators, ingress paths, and the dormant valve all cross the heart's typed
validation/transaction boundary.

### Packed substrate rails and the shelf pivot (ratified 2026-08-26)

Jeff ratified the packed-rail pivot on 2026-08-26 after the roundtable
convergence recorded in `roundtable/proposals/PACKED_RAIL_CODEC_PROPOSAL.md`
(including Codex's Q1-Q9 review and ChatGPT's mask-model correction). This
section is binding doctrine.

- Three contracts stay distinct: the exact packed scaffold, the derived
  neural representation a core actually consumes, and discrete output
  serialization. An exact pack is exact storage, not by itself a learned
  semantic unit.
- Rails pack exact typed 16D transport cells into disjoint literal lanes:
  `d_model / 16` transport units per packed row at each registered width (64D = 4,
  128D = 8, 256D = 16, 512D = 32, 1024D = 64). One generic codec must prove
  every registered width rather than bespoke per-width codecs. Dense opaque
  bit/positional packing is not an active reasoning surface.
- No pack-count ceiling exists on any rail; the Heart may push an unlimited
  number of packed rows. Packs never cross logical-region, attended-interval,
  or provenance/source-span boundaries; boundary packs carry explicit empty
  lanes so attended edge characters are never suppressed. Pages, chunks, and
  budgets remain compute controls with coverage receipts; content is never
  capped.
- Mask law (final in-place model, clarified 2026-08-27): each canonical region
  retains one exact position-stable cell sequence and has an independent
  Heart-owned 0–100% mask. The unmasked partition is the Shared Field; the
  masked partition is dormant in place. Policies resolve deterministically
  into exact spans/turns before compilation. The Heart circulates all and only
  exposed cells to every registered rail. Moving a mask changes the derived
  view, never the canonical cells, addresses, or `field_id`; no recall or
  rematerialization step is involved. Packing never interprets a percentage:
  it receives resolved attended intervals, stops and pads at every mask/
  provenance boundary, and packs every exposed canonical character exactly
  once as one native unit or its complete strict UTF-8 transport sequence.
- Cores attend directly to their designated rail; the Heart is not in the
  attention path. The Heart packs and unpacks cells at every registered
  width, maintains exact substrate/rail and rail-to-rail roundtrips, moves
  masks, and remains the sole validator and committer of canonical state.
- Emission contract: a Unicode-capable core emits discrete per-lane
  categorical decisions over the 351-category registered transport codebook
  (native character or strict UTF-8 byte), plus explicit EMPTY and EOS control
  categories, trained with per-lane cross-entropy per Layer 13. Emissions bind
  exact region/start/end addresses, base field/tick, author, home rail, and
  pass. DELTA, NO_OP, and ABSTAIN are distinct decisions. The Heart rejects
  malformed category streams, missing/duplicate EOS, noncanonical Unicode,
  stale bases, invalid bounds, and unauthorized regions, then deterministically
  reconstructs exact 16D cells and a typed proposal from accepted decisions.
  Nearest-vector snapping is auxiliary evidence only; an arbitrary
  continuous vector is never claimed to be exactly invertible.
- Shelf: the learned Heart translator tissue, its training campaigns and
  checkpoints, and autonomous Cortex implementations are shelved —
  preserved, non-serving, documented by a shelf manifest (original paths,
  commit/checkpoint/evidence identities, import consequences, return
  conditions). Remaining active: the exact compiler/codec, the deterministic
  Heart control plane, the Trainer, the Dormant evidence bridge, and
  deterministic grounded structural surfaces. Return conditions: learned
  translation returns when a native dialect, language, reordering,
  paraphrase, or other non-copy transformation cannot be represented by
  exact repacking; learned Cortex returns when measured real-memory
  retrieval/episode organization is the bottleneck and a grounded evaluation
  can prove gain.
- Learned-organ priority is now the reasoning cores on packed rails, trained
  by the Trainer from real Dormant memory (`experience_v1` and lived
  autobiography through the evidence bridge) under runtime-faithful
  controls: episode/conversation-level time splits, remove/swap/inject/
  stale-evidence/provenance-breaking/reordered-cause counterfactual probes,
  cited source IDs, and historical responses treated as observations, not
  presumed targets. Synthetic minimal pairs remain necessary supplements for
  rare critical distinctions.
- Unicode coverage: compiler/input coverage is exact for every valid Unicode
  scalar through `axon-unicode-transport-utf8-16d-v1`; raw canonical state is
  never escaped or normalized. Campaigns must report canonical scalar counts,
  transport-unit expansion, and exact decode. Existing shelved 95-class
  learned Heart/reasoning decoders remain native-only tissue: they may read
  the literal transport cells through compatible readers, but they may not
  pretend to emit non-native text. A Unicode frame presented to the legacy
  Heart model fails closed until a governed 351-class output/copy route is
  trained and independently gated.
- The per-character D64 reader remains permanent specialist tissue. Exact
  repacking translates the exact scaffold only; it is not semantic
  translation and does not translate learned native dialects.
- The first proof is tests, not training: a generic `d_model/16`-lane
  compiler/view, boundary/address receipts, and a categorical pack decoder,
  with exhaustive substrate roundtrip, mask-gap and provenance-boundary,
  cross-width repack, and rejection evidence before any packed-reader
  training begins.

### Heart intelligence and semantic conduction

(Shelved 2026-08-26 under the packed-rail pivot: the learned ensemble this
section describes is preserved, non-serving tissue with the return
conditions stated in "Packed substrate rails and the shelf pivot". The
deterministic Heart authority wrapped around it remains fully active
doctrine.)

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
`runtime/heart/valve.py`. The standard registry has twenty **bootstrap** slots,
not a twenty-organ ceiling; additional definitions may be appended without
changing existing valve identity. The primitive real `user_ingress`,
`tool_ingress`, `advisor_ingress`, and `dormant_recall` valves begin CAPPED as
an authority posture under renewable per-beat work allocations; the sixteen
bootstrap future-organ slots begin CLOSED. CLOSED is fail-closed. The durable
spool has no configured item-count cap, and exact payload size is never an
admission or quarantine criterion. Item and target-character allocations only
defer later FIFO work to another beat; one first oversized exact item is
processed whole. An external envelope carries source identity, payload, type,
and provenance only; it cannot supply an `AuthorityGrant`. The Heart resolves
valve identity to the permitted authority class and exact canonical region
itself, then revalidates valve version, source class, payload type, replay
identity, renewable work allocation, and typed-delta invariants at the final
gate before commit. Ordinary transient host failures back off and retry;
integrity or ownership corruption remains a visible hard stop.

Each beat:

1. Drains queued external arrivals. Between ticks, intake commits immediately
   as a heart-governed typed delta into its runtime-owned region. Arrivals
   during an in-flight tick queue for the next beat and never mutate the
   frozen base.
2. Resolves derived per-region attention masks for the rail and recall query.
   Masked text remains exact and position-stable in its canonical region; only
   attended intervals enter compiled rails. Policies are resolved from reusable
   mask policies such as `all`, `none`, `last_n_spans`, or a tail percentage at
   compile/recall time and do not alter the canonical `SharedFieldSnapshot`
   identity.
3. Detects both canonical-field change and derived mask-view change. A changed
   `field_id` requires fresh canonical stabilization; a mask-only change keeps
   the same `field_id` but must produce and circulate a new `view_id`. Only when
   neither identity changed may recompilation be skipped.
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

Attention-mask choices remain derived from durable Heart control state. The
Heart persists one explicit policy for every canonical region independently;
updating one region advances mask-control state without rewriting the canonical
field. Each frozen tick image and rail carries an explicit derived `view_id`
computed from the complete policy set. Two masked views of the same canonical
`field_id` are therefore distinguishable without making masks part of canonical
content identity. A changed field or changed view must pass exact D64 coverage/
roundtrip proof before the tick image is frozen. With no active reasoning cores,
the host explicitly closes the empty developmental tick after a successful
freeze; it does not invent a participant or proposal. With active descriptors
and configured execution ports, `runtime/heart/circulation.py` now runs the
real proposal/refinement/consolidation barriers against that frozen image and
commits only the accepted consolidator result. An active descriptor without a
runtime port fails safely and cannot be mistaken for serving intelligence.

`runtime/heart/reasoning_output.py` is the exact learned-output boundary;
`runtime/heart/proposal_workspace.py` is the derived noncanonical proposal
interlingua and categorical rail renderer. The renderer is width-generic for
every positive multiple of 16, while D64 remains the only physically compiled
Shared Field rail. No wider rail is claimed until its real compiler/surface is
proved. `runtime/heart/turns.py` deterministically completes an accepted user
turn: the consolidator must author a nonempty `response_draft`, then Heart
appends an exact readable length-delimited user/response frame to
`conversation_history` and clears transient `user_input` in the same atomic
delta. During this first form those two bookkeeping regions are reserved from
simultaneous consolidator edits to avoid ambiguous overlapping authorship.

`runtime/heart/masks.py` implements the durable complete eleven-region mask
controller beneath `State/active/heart/region_masks.json`. The permanent host
exposes independent policy and 0–100% newest-suffix controls, wakes circulation
for a mask-only change, preserves an already frozen in-flight tick, and records
mask revision/state identity plus every policy in Heart health. A restart
reloads and verifies the content-addressed mask state before circulation.
Identity is hard-wired to `all` and rejects any masking attempt.

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
4. The first pass closes when every required participant has returned a delta,
   returned an explicit no-op, abstained, failed, or timed out under governed
   policy. The heart then exposes the complete first-pass proposal board and
   participant accounting in the per-rail workspace.
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

The current runtime implements this complete barrier/transaction mechanism
synchronously through a permanent `ReasoningCorePort`; future local-thread,
process, or remote executors implement the same boundary. Every request carries
that core's exact private Soul snapshot. FIRST and REFINED each require and
durably commit a HOT-layer Soul transition before their barrier can close. The
consolidator's transition is prepared before the canonical field transaction,
then finalized against the exact successor `field_id`; restart recovery binds
an already-committed field journal entry to the prepared Soul transition rather
than double-applying it. Deterministic fixture ports prove this organism
mechanism, and the first neural D64 candidate implements the same causal phase
surface in training. No learned reasoning core is serving yet. This is
mechanism-functional circulation, not a claim of learned reasoning or
conversational intelligence.

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

Each core owns one completely private layered Soul. Soul bytes are opaque,
architecture-bound, parameter-generation-bound, and not required to be human
readable. They are never placed on proposal boards, translated for another
core, merged across siblings, or substituted for canonical state. Brothers
communicate only through their governed proposals. Axon's shared identity and
constitution belong in the canonical `identity` region, not in any private
Soul.

The permanent layers are `HOT`, `WARM`, `COLD`, and `DEEP_COLD`. HOT may change
on every causal inhale-think-exhale boundary. A runtime FIRST pass inhales the
pre-pass Soul and must exhale its successor before REFINED may inhale; REFINED
does the same before a consolidator pass. The consolidator's final Soul is
committed with the canonical successor field. A colder layer may change only
through an evidence-vetted promotion from its adjacent hotter layer. Deep-cold
promotion additionally requires validation evidence and is the only Soul layer
eligible as a future LoRA/adapter distillation source. Distillation creates a
new governed parameter generation; it never deletes the source Soul or exact
lived evidence.

`runtime/soul/` implements content-addressed snapshots, transitions,
promotions, prepared records, commit receipts, HEAD publication, lineage
inspection, and restart recovery. There is no byte-size ceiling in the Soul
contract. A concrete core codec may define its own exact tensor layout, but a
layout change is an explicit architecture migration rather than forgiving
deserialization. Auditable factual and episodic knowledge remains in canonical
or Dormant State; neither Souls nor weights may become the only copy of evidence
that should be recoverable exactly.

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
historical record-reference compiler. It verifies exact experience imports and
publishes content-addressed manifests whose examples reference source record
identities and complete context ranges rather than copying or clipping text.
The recovered `D:\00` import currently yields a 59,858-example Heart grounding
session and a 14,205-example observed conversation session. Those historical
assistant responses remain observations whose correctness is not inferred;
the record-hash splits are stable but are not asserted to be whole-conversation
splits because the recovered sources do not provide a trustworthy conversation
identity. The historical conversation manifest therefore cannot authorize
serving promotion.

`runtime/trainer/episodes.py` is the runtime-faithful path. It compiles only
complete `runtime_reasoning_episode` records, binds every episode from one
durable Heart epoch/conversation to the same deterministic 80/10/10 split,
keeps observed/failure/unknown quality distinct, and permits serving-promotion
eligibility only when a separate exact outcome record supplies success,
correction, or endorsement plus nonempty evidence identities. Its loader
strictly reconstructs the pre-action `SharedFieldSnapshot`, categorical
emissions and proposal workspaces, source/materialized typed deltas, completed
turn, accepted successor field, and every private-Soul snapshot, transition,
receipt, and causal lineage. It rejects a discontinuous or future-leaking Soul
trajectory. Trainer state and control-plane APIs can publish these immutable
sessions. Automatic target-quality adjudication for arbitrary lived outcomes
and complete wiring from every future tool/Trainer executor remain open work;
historical assistant output is observation, never presumed truth.

`training/lived_reasoning_curriculum.py` is the evidence-qualified target
boundary for those loaded episodes. An observed or failed trajectory carries
zero target weight. Success or endorsement supervises only the final accepted
consolidator outcome unless an exact outcome record explicitly grants
`full_trajectory` scope. A correction teaches only its exact corrected typed
delta. Missing outcome records, missing evidence identities, unverifiable
corrected deltas, and attempts to bless an entire trajectory implicitly fail
closed.

`runtime/trainer/soul_candidates.py` implements isolated candidate-Soul
branching. A candidate binds an exact live base Soul, architecture identity,
base and candidate parameter generations, deep-cold hash, whole-episode split,
and trajectory identities. It inherits the exact layer bytes into an isolated
candidate namespace and never mutates or merges into the live Soul. Promotion
is either exact-base-ready or requires an explicit replay of every intervening
live receipt; opaque private state is never heuristically merged.

`runtime/trainer/step_bundle.py` makes a parameter checkpoint and candidate
Soul HEAD one accepted boundary. It records an immutable intent, verifies the
optimizer receipt, content-addressed checkpoint, complete FIRST/REFINED/
CONSOLIDATED Soul transition triples, and lineage, then publishes one rolling
pointer plus completion sentinel. Restart either completes an idempotent
pending bundle or preserves conflicting/orphan evidence without advancing the
accepted pointer. A renewed Complete-Field preflight may resume the checkpoint
only when its stable grant/plan/inventory/module/generation/tensor scope is
exactly identical; refreshed evidence never widens mutation authority.

The first load-bearing neural reasoning candidate is
`training/living_reasoning_d64.py`. Candidate A is exactly `d_model=64`, one
64D attention head, two Transformer layers, `ffn_dim=131072`, and four
persistent Soul-state tokens. It has 33,981,879 trainable parameters
(135,927,516 bytes in FP32; 67,963,758 bytes in FP16). Its source pages are
bounded compute units only: a recurrent state containing the inhaled private
Soul traverses every canonical page in order and then every proposal-workspace
page. All four Soul layers can influence that initial recurrent state; each
runtime phase exhales an exact safe `f32le` HOT-layer state and the next phase
re-inhales it. The typed output surface predicts decision, operation, region,
dynamic exact boundary addresses, and 351 Unicode transport categories plus
EMPTY/EOS. An untrained or malformed emission fails closed and is never
registered live. Free-running diagnostics stop only at a renewable external
work slice and report incompletion. The core exposes a resumable iterator that
preserves exact decoder state across slices without a configured total output
length. The current runtime adapter consumes one slice and abstains when it is
incomplete; durable Heart-host continuation is required before learned D64
tissue may serve.

`training/living_reasoning_curriculum.py` supplies deterministic synthetic
mechanism cases for exact addressed edits, Unicode payloads, no-op/abstain,
proposal refinement, conflict handling, and current-field authority.
`training/living_reasoning_preflight.py` binds static capacity, architecture,
curriculum distribution, boundary coverage, Soul/field counterfactual
dependence, and strict checkpoint-compatibility evidence into the Trainer's
existing preflight receipt. `training/reasoning_tournament.py` fixes the first
architecture comparison at 1x64 versus 2x32 versus 4x16 heads with identical
two-layer/131072-FFN anatomy and identical evaluation gates, followed only if
needed by a 1x64 six-layer/4096-FFN and 2x32 four-layer/16384-FFN comparison.
These mechanism curricula and counterfactuals do not establish useful learned
reasoning or outcome quality.

Jeff ratified the converged Curriculum Foundry direction on 2026-08-29: exact
evidence and derived manifests, whole-lineage splits, scoped target eligibility,
quarantine rather than evidence-rewriting redaction, runtime-faithful lessons,
and small competency campaigns. Smallness is a revisable campaign budget, not
a context, field, attention, curriculum, or organism capacity limit.
`training/first_form_curriculum.py` compiles the first immutable A/B/C stack
from the already-governed Dormant import. The initial immutable manifest
`5bb0bb4f1baf607e71bf33b99212e0e955f27cfaf02a9bdd6bf3d26046ced800`
preserves an incorrect difficulty-only page count that divided packed D64 rows
instead of counting real character pages; it never clipped model input. The
corrected manifest is
`67055e5419c660bddd6abb98b39798f7d3bbdbe5e7b6b40934550455c9b0afe4`.
It contains the same 260 complete lessons and the same living-curriculum
identity: A 32/4/4, B 96/12/12, and C 80/10/10 across
train/heldout/regression. No admitted field or target is clipped.
Target eligibility contains no character-count rejection; long exact targets
remain eligible under the same evidence and content-safety rules.
Recovered assistant text is a visible exact-copy transport target only and is
not labeled true, wise, or high quality. Every case carries the active Identity,
source identities where applicable, a whole-lineage split, eligibility, target
basis, correct character-page count, and procedural depth. FFCS-D/F manifest
`b69b5a16815735b372381243b6d4c34195301a626fb7c91450900361460961b1`
adds 140 derived cases: D 64/8/8 uses real typed proposal workspaces, frozen
current-field evidence, brother refinement, and consolidator targets; F
48/6/6 composes exact distant Unicode facts across complete multi-page fields.
FFCS-E manifest
`351fea3535749cf773aa31ff36220f78f770969fb7674270ea7cfc8bfe2781df`
adds 60 grouped sequential cases (48/6/6). Every three-tick chain crosses real
Heart turn-finalization receipts, typed canonical successors and user ingress;
the same core's serialized private Soul is re-inhaled at every tick with the
service-faithful gradient break, and no future field is exposed early.

The governed reasoning harness accepts multiple immutable standard and
sequential manifests without flattening sequential cases. It publishes one
content-addressed campaign-curriculum manifest, includes that identity and all
component split identities in the candidate generation and Trainer mutation
plan, and schedules mechanism/A/B/C/D/F/E as deterministic family round-robin
lanes. Thus a seven-step first cycle reaches every family, rather than delaying
E behind hundreds of ordinary cases. Tournament metrics are count-weighted and
content-addressed with exact probe IDs and definitions. A metric with no
admissible stale-Soul, FFCS-D counterfactual, or regression surface remains
explicitly missing; the harness never imputes a neutral-looking zero or perfect
one. Current-field override holds the stale board fixed while changing the
canonical D signal and exact targets; proposal refinement compares the same
refined target with and without the first board. Hidden-state movement is not
reported as learned exact-rate improvement.

The communication-first C1 manifest
`c04ae8c6815e93b31136fdf833a525388c8439e74fd5317268cb1e1ba1233212`
retains all 36 authored cases as historical curriculum evidence, including 14
`PROCESS_EVIDENCE` open-response cases whose prose is not unique gold. Trainer
execution derives a content-addressed teaching view containing only the 22
`VERIFIED_TARGET` cases, uses that view for optimizer source identity, and
rejects every other eligibility class at the first-form objective boundary.
Learned-capability evaluation likewise uses the isolated verified standard-
curriculum heldout/regression surface rather than combining it with synthetic
mechanism cases; reports retain per-manifest and per-family metrics. Historical
C1 step-193 artifacts predate this contract and remain inadmissible as the
evidence-qualified C1 verdict.

The first objective Language Foundations curriculum is published at manifest
`824aabae2090c721da9b55540a5d890fc958577d2e5c78fde49da870096634f9`.
It contains 349 complete `VERIFIED_TARGET` cases across L0 exact
character/Unicode transport, L1 orthography, L2 vocabulary, L3 morphology, and
L4 grammar, with 240 train, 56 heldout, and 53 regression cases. Every lesson
uses the real Shared Field, exact D64 compiler rail, three-phase private-Soul
unroll, typed 351-category transport, and exact response-draft address. Hidden-
target leakage, visibility consistency, and train/heldout/regression transfer-
item disjointness fail closed. The full 1x64-head/two-layer/131072-FFN tissue
passed local CPU preflight as receipt
`ae99a3149485f0fa3d5d0d5dd006d2a89c0bf66f44b08fff8a36a35d4e3070d3`;
this proves launch anatomy, not learned language.

Jeff ratified the **foundations-first reasoning curriculum** on 2026-09-04.
The frozen substrate already represents exact native and Unicode transport;
an ABC exercise therefore teaches sequence use, not the existence of letters.
Fresh reasoning education begins narrowly with real typed-delta motor control,
exact one- and multi-character transport, response-draft addressing, no-op/
abstain, and EOS. It then uses alphabet, digit, punctuation, Unicode and
arbitrary sequences as one early sequence-navigation stage before word
mechanics, short communication, composition, brother circulation and lived-
experience reasoning. A fixed alphabet recitation cannot establish mastery.
Heldout and regression work must be split-disjoint, and every evaluation
operation must include a changed-source pair (plus arbitrary non-alphabet
sequences) so a memorized song cannot satisfy the gate.

Advancement is competency-gated and cumulative. Teacher forcing may supply
credit assignment, but it is never mastery evidence. Stage advancement needs
complete free-running heldout and regression evaluation, exact termination,
performance above declared trivial baselines, changed-source dependence,
complete-field coverage and retained prior competencies. Passed foundations
remain available through adjustable spiral replay. A stage gate advances
curriculum only; it never authorizes serving or parameter promotion. Failure
pauses at an exact checkpoint for diagnosis or a renewable resource tranche;
it is not a lifetime step, curriculum, output, context or tissue ceiling.
The earlier ABC manifest remains immutable historical evidence and is not the
foundations gate because its authored patterns repeat across data splits.

The first split-disjoint Stage-1 smoke is preserved as a diagnostic, not a
passed lesson. Kaggle job `37389c3dce3ab86afedcaf0b5230bb9473e09538a499e93a820a8846257e13b8`
ran 60 accepted optimizer steps on a Tesla T4 and reduced complete heldout loss
from 10.6677380204 to 5.5501824021. It learned the typed control shell well
enough to emit `DELTA`/`REPLACE`/`response_draft`/EOS, but emitted empty
payloads: free-running exact remained zero and teacher-forced payload accuracy
was 0.055556, below the 0.062500 constant-category floor. The Stage-1 gate
therefore remained false and no long continuation was authorized.

That falsification activates the ratified prerequisite rather than weakening
the gate. Foundations Stage 0 is the content-addressed `F0` manifest
`a273737cdaa399f9b85e1a8ac91326a255a6c3b63ef880fd1bc1273832be857c`:
120 complete cases (72 train, 24 heldout, 24 regression), with copy, insert,
replace, delete, no-op and abstain represented in changed-source pairs in every
split. Copy-bearing targets may now carry an additive exact source-alignment
contract. The living D64 objective supervises the existing copy pointer at the
exact current-field scalar address while expanding that scalar to all one-to-
four Unicode transport cells in order; EOS remains explicit generated output.
This is decoder exposure for existing tissue, not an alternate substrate,
normalization, semantic token, or architectural shortcut. Full-size CUDA
preflight passed as receipt
`096c3e2b63a463580c8a2d6b72ee386d921fa84303480de039fd192f0b92d89c`;
it proves launch anatomy only, not learned motor control.

The first two Stage-0 resource tranches are also diagnostic evidence, not a
pass. Private Kaggle job `c8ba53cb9f7044341fd7a9d072ea460700dbdbf457aed509acf8e2ecf3e58e72`
trained steps 1-60; job
`3769a4caf10231a19e3bab940fdb700b2bf3eff8c208274d5232ad2070959137`
then restored its exact accepted step-60 parameter, optimizer, checkpoint and
private-Soul parent and trained steps 61-120. Across the two segments heldout
loss fell from 12.9976677448 to 5.7030190378, but the transient first-segment
teacher-forced payload advantage (0.53125 against a 0.50 constant floor)
returned exactly to the floor at step 120. Free-running typed exactness and
every per-action/source-pair gate remained zero; final payload exactness was
0.25 because empty delete payloads terminated while the core continued to emit
the constant `DELTA`/`REPLACE`/empty-payload shell. The Stage-0 gate and task
gate are false, the candidate is paused and non-serving, and no third tranche
is authorized on this objective. Training-step evidence shows that the exact
position pointer sometimes learned its supervised source address while the
copy/generate gate continued choosing generation for every payload cell. The
next attempt must first isolate EOS from content metrics, balance decision and
operation classes, and teach copy-position plus copy-gate control before joint
typed action; it starts as new tissue unless an explicit governed continuation
decision establishes compatibility.

Full-field recurrent training uses exact page-level activation checkpointing
while gradients are enabled. It recomputes the same page encoder during
backpropagation; it does not omit, summarize, detach, or truncate any field
character. This reduced the measured one-step D64 tournament peak from a CUDA
out-of-memory failure above 10 GiB of retained activations to 837,841,408 bytes
on the 4 GiB GTX 1650.

The corrected deterministic-dropout tournament ID is
`ac9eceaaf0e67d483b288140d4db328c7f7b2156468ef5e25a1eaa9f4b791a39`.
Its bounded opening gave 1x64, 2x32, and 4x16 exactly one accepted optimizer+
Soul bundle each under the same seed, composed mechanism+A/B/C curriculum, and
one complete heldout FFCS case. Heldout losses respectively moved
9.665966 to 6.863116, 9.700136 to 7.163452, and 10.046765 to 6.946915;
teacher-forced payload accuracy tied rather than exceeded the 0.181818 constant
floor for all three, and both free-running exact rates remained zero. The
remaining 26 heldout cases and the full required tournament metric surface are
explicitly deferred. No winner, promotion, activation, serving claim, or long
run is authorized by this opening.

The first composed mechanism+A/B/C/D/E/F campaign is content-addressed as
`b39bc91b69e5d8ce3395ff826e15f3413da7c9f48376f831f037743205a4a7cd`.
Its first attempted pass failed closed on FFCS-E because the harness supplied
only the final tick's three private-Soul transitions to an atomic step bundle
instead of the complete three-tick, nine-transition lineage. No invalid bundle
was accepted. The repaired harness returns and verifies the complete lineage;
the repository regression test reconstructs all nine transitions from the
initial Soul and proves the exact final Soul identity.

The bounded local-CUDA diagnostic subsequently reached exactly 16 accepted
optimizer+Soul steps for every head geometry. On the same 40-standard-plus-six-
sequential heldout surface, final heldout loss was `6.07003877473914` for 1x64,
`5.994862172914588` for 2x32, and `6.006355773193193` for 4x16. Respective
teacher-forced payload-token accuracies were `0.0128467153284672`,
`0.012408759124087591`, and `0.0125547445255474`, all far below the identical
`0.10875912408759124` strongest constant-category floor. Full-field coverage
was `1.0`; typed-emission and complete-payload exact rates were `0.0` for all
three. One standard heldout case remains explicitly deferred, so the formal
tournament metric surface is incomplete and no comparison result, winner,
promotion, activation, or learned-capability claim exists. The closing
observation is
`068867c5f1e4b693f00625d30c3a718feea5a40030a4ff41758e54d865f012ab`.

The historical 16-step authorization ceiling remains part of those immutable
v1 Trainer plans and candidate-generation identities; neither artifact is
rewritten. The additive continuation blocker has since been closed at the
mechanism level by the renewable-tranche contract below. Isolated end-to-end
fixtures prove both (a) adoption of one v1 candidate beyond its original
envelope without changing its plan or generation and (b) a fresh v2 candidate
continuing from global step 1 to 2 with the same plan, learning policy,
checkpoint/optimizer lineage, and private Soul while obsolete `max_steps` and
checkpoint cadence inputs change. The real three step-16 tournament candidates
have not been mutated under this mechanism. Further training remains withheld
until a communication-first curriculum has adequate evidence-qualified targets
and the normal smoke/gate surface authorizes the bounded run.

### Renewable resource tranches (ratified 2026-08-30, Jeff's binding law)

Operational resource limits are **renewable execution tranches**, never core
identity. A tranche is a separate content-addressed authorization object that
grants a bounded number of optimizer steps (and optionally wall-time/cost
budgets with checkpoint/evaluation cadence) for **one execution segment only**.
Tranches obey a non-negotiable law:

- Reaching a tranche bound means atomic checkpoint + evaluation receipt +
  `paused/checkpointed`. It never means `complete`, `failed`, a new lineage,
  or a restart from the governed base.
- The same candidate lineage resumes under a later tranche from its exact
  accepted checkpoint/optimizer/private-Soul bundle, continuing the global
  accepted-step count (N -> N+1), provided the plan, curriculum, learning
  recipe, architecture, and seed are unchanged and the parent bundle is
  verified exactly.
- Tranche size, wall-time, cost, provider, device, checkpoint cadence, and UI
  entry point must not participate in plan identity, candidate-generation
  identity, learning-policy identity, or checkpoint lineage identity.
- A plan's historical `max_steps` remains immutable v1 truth: the original
  authorization envelope under which that plan's accepted steps ran. A tranche
  may lawfully continue a candidate beyond that envelope when its parent
  state is exact and the continuation is receipted; the historical plan text
  is never edited.
- Learning-rate scheduling belongs wholly to the immutable learning policy,
  never to the resource tranche. A constant scheduler is horizon-free; any
  finite warmup/decay horizon is explicit in learning-policy identity and
  continues by global optimizer step across every tranche. Changing resource
  allowance must not change how the core learns.
- Curriculum/competency gates decide stage completion. Plateau, anomaly,
  integrity failure, or operator action may pause work without erasing it.
- Tranches may pause, reject, quarantine, checkpoint, or require a fresh
  preflight. They must never define a core's identity or lifespan, strand an
  accepted checkpoint/optimizer/Soul, truncate curriculum/history, silently
  relabel replacement tissue as the original core, or convert a smoke budget
  into a permanent architectural ceiling.

`runtime/trainer/tranche.py` implements `ResourceTranche` records and
continuation receipts beneath `State/training/trainer/tranches/`. Every
continuation names its exact parent checkpoint, optimizer receipt, Soul HEAD,
and prior tranche when one exists, and is immutable once written.

Historical `ParameterMutationPlan` v1 remains readable and immutable.
`ParameterMutationPlanV2` is the permanent resource-independent contract for
new tissue: mutation scope and curriculum manifests remain governed, while
optimizer/schedule live in learning-policy identity and execution allowance
lives only in renewable tranche identity.

The Trainer is a permanent Axon organ, not a collection of launch scripts.
`runtime/trainer/organ.py` defines its transport-neutral command boundary.
Present CLI/batch launchers and future runtime slash commands, clickable UI,
continual-learning controls, and cloud adapters must call that same boundary;
they cannot own alternate mutation semantics. Unwired mutating commands fail
closed. Its default read-only status is a concise operator summary; full tensor
inspection is explicit. Training remains offline and isolated from serving
until the existing promotion authority accepts complete gate evidence.

The private Kaggle transport is a current adapter, not an alternate Trainer.
`runtime/trainer/cloud_jobs.py` exports only committed executable tissue plus
explicitly selected State into a content-addressed packet;
`runtime/trainer/kaggle_adapter.py` registers cloud export/start/result-import
handlers on the same organ boundary. Provider, accelerator, and execution
tranche belong to external job/resource identity and never rewrite candidate,
plan, curriculum, or learning-policy identity. Export refuses dirty tracked
source, path escape, symlinks, credential filenames, and unacknowledged State.
Generated datasets and kernels are private, internet-disabled, and launch only
after explicit operator confirmation. Cloud output is noncanonical evidence
until verified and imported; it cannot activate learned tissue. Training emits
flushed per-step progress to provider logs plus an append-only JSONL journal
and atomic latest-event file so observation and execution survive any
engineering-agent or terminal lifetime.

On 2026-08-28 and 2026-08-29 the exact full Candidate-A architecture passed
preflight and a sequence of bounded Trainer-governed diagnostics on an isolated
candidate branch. Fifteen accepted CUDA optimizer steps now have atomic
checkpoint/Soul bundles. Campaign-baseline held-out loss fell from
7.409373760223389 to 4.788444995880127 and field/proposal/Soul counterfactual
deltas remained nonzero. However, teacher-forced transport-token accuracy is
only 0.06896551724137931 against the held-out strongest constant-category
floor of 0.13793103448275862; free-running typed-delta exactness and complete
Unicode-payload exactness both remain 0.0. The honest task gate and exact-output
gate therefore both fail. No serving activation, promotion proposal, overnight
run, or learned-capability claim occurred. These diagnostics prove executable,
restart-safe anatomy and some loss movement, not useful reasoning.

The purpose of lived-experience training is primarily **procedural compression**: reasoning habits, tool-use instincts, error avoidance, planning patterns, semantic discrimination, confidence calibration, and other generalized intuition that should become easier because Axon has encountered similar situations before. Parameters are not required to memorize every factual detail. Exact facts, versions, identities, conversations, source material, and auditable outcomes remain in Dormant State and can be surfaced by Cortex when needed. In mature operation, weights should answer roughly "how have situations like this tended to work?" while Dormant State + Cortex answer "what exactly happened, what is known now, and what evidence supports it?"

Core diversity should emerge naturally from governed variation in lived-experience sampling, temporal windows, curriculum order, objectives, initialization, adapters, architecture/width, and replay/counterfactual emphasis. Multiple generalist reasoning or semantic cores may therefore learn overlapping life history through different lenses and acquire different useful intuitions without requiring every core to be narrowly labeled "coding", "math", or "science". Explicit specialist cores remain optional additions, not the only path to ensemble diversity.

Steady-state Axon should normally keep at least one **isolated non-live candidate learning lane** active on admissible lived-experience or study curriculum while other cores serve the organism. "Always learning" never means forcing meaningless gradient steps: if no curriculum passes provenance/quality gates, that lane remains occupied with curation, replay construction, evaluation, or forgetting analysis until admissible learning material exists. The live accepted cores remain immutable until a candidate independently passes Trainer gates and activation.

**Current first learned-organ priority (shelf pivot, ratified 2026-08-26): the reasoning cores on packed rails, trained from real Dormant memory.** The Heart translation/conduction ensemble and autonomous Semantic Cortex are shelved as preserved non-serving tissue (see "Packed substrate rails and the shelf pivot"); the priority history below is retained as evidence. The Trainer recognizes Heart translation cores/adapters as explicit parameter-bearing organ kinds. `runtime/heart/translation_core.py` is now the first permanent learned Heart tissue: a 64D, two-layer, four-head, 4096-FFN translator grounded from the frozen 16D character substrate, with explicit semantic, referent, and grounding heads; it has no canonical-write authority. Architecture v3 has no learned or validated source/target sequence-length ceiling; its learned vocabulary remains the original 95 native characters. A configurable physical page is only a processing unit: two ordered recurrent sweeps visit every exact source character, the second sweep builds full addressable character memory from a query state that has already traversed the complete source, and a coverage record binds per-row source-index hashes, page spans, and visited counts. Source and decoder positions are deterministic sinusoidal functions rather than finite learned tables. `runtime/heart/d64_codec.py` freezes each actual `SharedFieldSnapshot` through the exact and semantic D64 compilers, verifies exact roundtrip and grounding, and supplies the Heart with literal raw 16D transport cells, categorical token IDs, and non-decreasing canonical character positions; every byte of one expanded scalar shares its original position. Masking earlier spans therefore cannot renumber later active text. A substituted lane cell, malformed/noncanonical transport, stale field/rail/surface identity, or proposal not bound to the frozen frame fails closed. The legacy Heart model explicitly rejects Unicode byte frames until a Unicode-capable learned route is independently trained and gated. `training/heart_translation.py` materializes every provenance-labeled structured-proposition curriculum case as a real Shared Field and real D64 frame before model input; it provides disjoint heldout/regression/counterfactual suites, semantic/grounding evaluation, and an immutable content-addressed task-loss objective. Curriculum v3 includes train and held-out complete-field cases for every critical semantic class whose grounded spans begin beyond character 256. Its current training recipe uses deterministic shuffled epochs: every case is visited once before reshuffling, and the immutable recipe identity is bound into candidate generation and source lineage. A first real Trainer-governed 12-step CUDA smoke used the obsolete fixed-192 architecture v1; it lowered loss from 4.9502 to 4.3422 and moved some semantic submetrics, but grounded roundtrip and aggregate semantic fidelity remained 0.0, so the candidate was rejected and no activation/promotion proposal occurred. Its immutable artifacts remain historical evidence and are not compatible with v3.

As of the no-ceilings sweep, the original learned 16-entry dialect tables
remain byte-compatible bootstrap tissue for every existing Heart checkpoint,
while every later non-negative dialect ID receives a deterministic
content-addressed tail encoding; sixteen is no longer a dialect ceiling and
large numeric IDs do not alias through floating-point positions. Greedy Heart-translation
diagnostics use the protected renewable work slice and report unterminated
output explicitly rather than imposing a total character limit.

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

Long-position curriculum
`ed83bb3669898e950fab44af43187506d8f1ee6a2fe041299e80e21acfc4387f`
extends rather than replaces that identity curriculum. It retains all 37 prior
training cases as replay, adds complete train sources through 640 characters and
disjoint held-out extrapolation through 769, and probes exact positions 0, 254,
255, 256, 257, 510, 511, 512, 513, 699, and 768. Those finite lengths and
positions are evidence points around physical page boundaries and late tails,
not model limits. No source is sliced or truncated. Evaluation may batch
independent complete examples to fit physical memory; every example still
undergoes both ordered sweeps over all of its characters.

The first 64-step long-position candidate
`d51a6047c6a51401633cb38033d767eb150877a2c7c2a68068d9e04138c7fa11`
raised held-out character accuracy from 0.0824 to 0.1077 and diagonal top-one
alignment from 0.0870 to 0.1034, but replay EOS fell from 0.2432 to 0.1351.
It was rejected and never activated. The Trainer was then given an explicit
protective cadence and configurable EOS-route weight. Independent-case
evaluation batching reduced observed GTX 1650 use from approximately 3.88 GiB
with an allocator OOM warning to approximately 1.57 GiB without changing any
case's complete-field coverage.

Protected continuation
`3e12b8a642fde98c3c6feaf39baf3dd4fdb058af903cf19eb599eedddbcb69f9`
resumed the rejected checkpoint for 192 bounded steps: 64 long-position updates,
128 replay updates, EOS-route weight 1.0, and durable checkpoints at steps 64,
128, and 192. Fixed audit loss fell from 8.7739 to 7.2936. Held-out character
accuracy rose from 0.1077 to 0.1479, diagonal mass from 0.0586 to 0.0786,
diagonal top-one from 0.1034 to 0.1497, counterfactual exactness from 2/14 to
3/14, replay character accuracy from 0.2533 to 0.2992, and replay EOS from
0.1351 to 0.3243. Free-running exactness remained fixed at 1/46 and the worst
counterfactual margin remained negative. Read-only intermediate evaluation
showed held-out character accuracy 0.1163 at step 64 and 0.1404 at step 128;
the slope flattened by step 192 while sequence exactness did not move. The
candidate failed twelve gate requirements, remained rejected/non-serving, and
created no promotion or activation.

Therefore further identical step doubling is not authorized by this evidence.
The next bounded mechanism investigation should compare staged whole-case
length exposure with an additive explicit positional-copy facility that exposes
same-address structure to the decoder while retaining learned content attention,
old D64 tissue, exact source evidence, and semantic translation paths. It must
be an architecture-migration/ablation experiment, not a hard-coded claim that
identity copying proves learned semantic Heart function. Wider or peer Heart
cores may be added later; the D64 specialist and its checkpoints remain useful
historical and initialization tissue.

That ablation is now complete. A staged-whole-case v3 control moved held-out
character accuracy only from 0.1479 to 0.1632 and left exact generation at
1/46. The additive v4 positional route moved held-out character accuracy to
0.9533 and all 14 source-change pairs to exact, but exposed a semantic hazard:
an always-available identity route could override a requested translation.
Governed v5 therefore makes the route physically unavailable unless Heart
explicitly requests exact same-address conduction. Its v3 migration preserves
every old tensor and produces bit-exact CPU outputs with the new route at exact
zero. A 144-step gate-only candidate mutated only 129 new control parameters;
all unseen held-out fields then passed at 1.0000 teacher-forced and free-running
sequence exactness, with 1.0000 EOS/termination and all source-change probes.
Replay reached 36/37 exact sequences (0.9730), so the candidate failed the
strict gate, remained rejected/non-serving, and created no activation. This is
verified exact-conduction anatomy under explicit opt-in, not learned semantic
Heart function. The next design decision is whether exact conduit execution
should be deterministic after a learned Heart control decision, rather than
making exact byte transport itself probabilistic; semantic translation must
continue to run with the conduit unavailable by default.

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
length. Diagnostic greedy translation returns explicit per-item termination
state; an item that exhausts its protected renewable work slice is incomplete, is not
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

Finite hardware requires finite work controls, never finite tissue capacity.
The protected registry is
`configs/source_of_truth/capacity_policy.json`, schema
`axon-source-of-truth-capacity-policy-v1`, canonical SHA256
`4a32f18fa296c415ccf34a8c1956d2a4f8afd044265e7f45505782dd53c7b8cd`.
`runtime/source_of_truth.py` binds that exact hash and fails closed on an
unratified edit. Every finite control on a live or training path must be in
that registry or in a separately content-addressed external authorization
object, and must be classified as a physical processing unit, renewable
compute/optimization allocation, result-count policy, or integrity/authority
gate. Its declaration states what source is preserved, how work continues or
resumes, and how the path reports failure or incompletion. Resource controls
must not participate in tissue, plan, candidate-generation, learning-policy,
Soul, or checkpoint identity.

No finite control may reject or permanently hide an exact item because it is
large. A first selected oversized item crosses a character work target whole;
later items remain exact and pending for later passes. No fixed total may be
placed on content, context, output, history, queues, attended field, learned
positions, dialect IDs, rail packs, valves, organs, or candidate lifespan. A
page, work slice, batch, result set, checkpoint-retention count, or resource
tranche is renewable operational work, not anatomy. Model width, head count,
layer count, FFN size, and the frozen 16D/351-category transport are deliberate
physical representations; integrity and categorical domains are not content
ceilings. An undeclared finite control, silent truncation, permanent
long-item exclusion, partial result presented as complete, or resource number
baked into tissue identity is a launch-blocking defect.

Changing the protected registry requires Jeff's explicit ratification plus one
atomic engineering change that updates the JSON, the code-bound hash, both SOT
mirrors, tests, and the append-only engineer ledger. Git history and the hash
binding make a change visible and fail-closed; they do not pretend to be an
operating-system access-control boundary.

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
- the current D64 reader deterministically unpacks every exact 16D transport lane before its neural lift,
- scratch changes are ordinary typed deltas followed by canonical successor compilation and a second complete read,
- response-draft learning remains observable and exact-position/copy-gate evaluation remains available,
- the living D64 path executes FIRST, REFINED, and CONSOLIDATED against one
  frozen field image, exhales and re-inhales exact private HOT Soul state at
  every phase boundary, and trains the same typed decision/delta/Unicode heads
  exposed at runtime,
- isolated candidate Soul branches inherit exact live layers without mutating
  live state; no opaque Soul merge is permitted at promotion,
- training workspaces live beneath `State/training`; branch-backed episode journaling and canonical split/resume proof remain required before a new training campaign is authorized.

The current synthetic curriculum is a mechanism bootstrap, not the ultimate
lived-experience curriculum. Atomic parameter/Soul resume, exact attention-view
loading, explicit outcome qualification, and field/Soul/proposal
counterfactuals now exist. A long or promotion-bearing campaign remains blocked
until admissible runtime-faithful episodes or a separately governed first-form
curriculum provide whole-lineage splits, enough target diversity, regression
and forgetting suites, and task metrics that beat the strongest constant
output. The current three-train/one-heldout synthetic mechanism set may not be
repeated for thousands of steps and represented as learned reasoning.

The archived 461,500-step Bible-trained 64D checkpoint family remains historical evidence only. A bounded compatibility/donor experiment was performed during development, then explicitly rejected as the future initialization path. Fresh reasoning-core and semantic-core training begins from clean current anatomy; legacy 384-slot checkpoints are not imported, resumed, or used as seed weights.

Pre-Day-Zero 384-slot readers, ExactV4 runtime/trainer paths, multi-tick prototypes, soul pilots, detached curriculum builders, and their dedicated tests are historical evidence only under `archive/day_zero_legacy_2026-08-20/`. They are not active fallback or initialization interfaces.

## Deterministic D64 Field Compiler

The shared exact D64 compiler is implemented in `runtime/field/compiler_d64.py`
and is the canonical D64 core-input boundary for both runtime-facing and
training-facing adapters.

Binding invariants:

- the compiler consumes one immutable `SharedFieldSnapshot` and binds every
  rail to that snapshot's exact `field_id` and `tick_id`;
- every attended canonical character is represented exactly: native characters
  retain their literal frozen 16D cell, while non-native Unicode scalars expand
  to their canonical one-to-four strict UTF-8 byte cells with typed token/unit
  receipts. Every valid Unicode scalar is supported; invalid scalar text and
  malformed/noncanonical categorical streams fail closed rather than being
  omitted or rewritten;
- one D64 physical row contains at most four exact 16D transport cells; rows never cross
  logical-region boundaries and unused lanes are explicit padding;
- every valid lane retains exact region position, global canonical-body position,
  source span, span position, source, provenance, attended-interval identity,
  transport token kind/value, unit index/count, row, and lane identity;
- all eleven logical regions are visited on every current-schema compile,
  including empty or
  explicitly masked regions; masked text remains canonical state but is not an
  attended rail character; attended intervals are sorted, non-overlapping,
  half-open ranges over the region's full span text and are compiled exactly;
  masks may be supplied at compile time as derived views and do not change the
  canonical field identity;
- a physical row may not bridge a region boundary, attended-interval boundary,
  canonical source-span boundary, or provenance boundary. A short boundary
  group receives explicit empty lanes; neither adjacent attended islands nor
  adjacent provenance sources are compacted together to save space;
- compilation is accepted only after separate canonical-character and physical
  transport-unit coverage proofs plus exact categorical/16D roundtrip
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
exact 16D transport cells defined above, and it alone carries the coverage
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
candidate contained a character that the then-current native-only compiler did
not support; it was counted explicitly as D64-inaccessible rather than
normalized or truncated, and it was not an expected target. This is retained
as historical evaluation evidence; the current exact Unicode transport closes
that compiler-ingress gap without altering the old artifact. Artifact:
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
character ceiling. Greedy diagnostics consume a renewable work slice from the
protected capacity policy, report nontermination explicitly at slice
exhaustion, and never treat it as completed output. The living reasoning core
exposes an in-process iterator that carries exact decoder state across
renewable slices until EOS or malformed-output rejection; there is no total
output ceiling. The current runtime adapter fails closed after one incomplete
slice until durable Heart-host continuation is implemented. The original v1
architecture hash mistakenly included a 512-unit decoder allowance even though
it never changed tensor topology. Current behavior ignores that allowance and
the three existing 1x64/2x32/4x16 architecture IDs are preserved through an
explicit retired identity projection so their parameters and private Souls do
not become garbage. Current canonical configuration and future policy changes
do not include that resource number in tissue identity.

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
Dormant lexical queries and Heart recall queries preserve their complete exact
text and process every unique term in ordered SQL-safe pages; the former
128-term rejection and 512-character query slice are removed. Recall item and
target-character values are renewable materialization policy: skipped later
items are identified and remain exact/retrievable, while a first selected item
larger than the target crosses whole. Selected text is never partially
truncated or permanently excluded for size. Candidate, relation, graph, edge,
and `k` work scales with the requested result set and has no fixed 128/1024/
32768 absolute clamp.

The dormant valve is part of the heart. Candidate ranking combines lexical
support, recovered graph topology, query-matched semantic-edge support,
confidence, type/task relevance, and novelty versus the active field under a
governed relevance policy. First-form semantic relevance is recovered-edge graph
semantics already owned by the corpus; no trained encoder exists and none may
be simulated. Relation propagation uses declared renewable breadth and fails closed: generic graph
hops and query-matched relations are distinct signals, one strongly matched
edge may contribute bounded best-edge support, and duplicate/weak edges may
not accumulate into synthetic certainty. Exact bytes are always dereferenced
from authoritative JSONL and hash/provenance-verified before surfacing. If the
semantic/relevance path yields nothing eligible, fallback is limited to exact
grounded lexical candidates under the same whole-item renewable work policy.

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

- `runtime/field/schema.py` ? canonical eleven-region exact field schema with
  append-only Identity at ID 10 and immutable v1/v2 history,
- `runtime/field/delta.py` ? typed canonical deltas and validation/apply/replay,
- `substrate/unicode_transport.py`, `runtime/field/compiler_d64.py`, and `runtime/field/semantic_d64.py` ? exact additive typed 16D Unicode transport, deterministic D64 compiler, and grounded deterministic first-form semantic-slot surface,
- `runtime/field/state_branch.py` ? canonical branch persistence,
- `runtime/axon_runtime/d64_adapter.py` ? runtime-facing exact and dual-surface D64 adapter,
- `runtime/dormant/experience.py`, `runtime/dormant/evidence_bridge.py`, `runtime/dormant/relevance.py`, `runtime/dormant/generations.py`, `runtime/dormant/incremental.py`, and `runtime/dormant/evaluation.py` ? immutable content-addressed exact experience/source snapshots plus read-only manifest/hash-bound dormant retrieval, exact dereference, bounded graph/relation relevance, verified derived-index generations, transactional append/layout-preserving update maintenance, and held-out evaluation,
- `Cortext/contracts.py` ? grounded Semantic Cortex service contract only; no active specialist/training authority and the `semantic_cortex` valve remains CLOSED,
- `runtime/heart/` ? heart anatomy: authority/core control plane, canonical transaction boundary, beat coordinator, proposal/refinement barriers, exact categorical reasoning-output decoder, width-generic derived proposal workspaces with a physical D64 renderer, rotating consolidation, deterministic completed-turn materialization, sovereign appendable valve plane with twenty bootstrap slots, OS single-writer lease, restart-safe cardiac identity, exceptional evidence-bound Identity amendment, durable ingress/replay/quarantine spool, exact ingress/reasoning autobiography plus private-Soul lineage and outcome/tool/Trainer hooks, durable independent per-region mask control with Identity always attended, health observability, explicit derived-view identity, relevance-gated dormant recall, permanent Heart host, `runtime/heart/intelligence.py` for learned Heart identity/fidelity/promotion contracts, `runtime/heart/d64_codec.py` for literal real-field D64 framing, and `runtime/heart/translation_core.py` for preserved non-authoritative 64D neural translator tissue; circulation is mechanism-functional but no learned reasoning core or Heart translator is serving yet;
- `runtime/soul/` ? permanent opaque private layered-Soul store, exact
  transitions/promotions/receipts, prepared/finalized commit protocol, and
  restart recovery,
- `runtime/trainer/` ? permanent Trainer parameter-authority anatomy: heterogeneous parameter+buffer inventory, OS single-writer lease, scoped mutation grants, immutable content-addressed learning policies, isolated candidate optimizer execution with governed accumulation/scheduling/precision/budgets, per-parameter telemetry, exact mid-accumulation checkpoint/restore, deterministic promotion gates, atomic active-generation pointers, exact activation/rollback snapshots and receipts, restart hydration, historical record-reference sessions plus whole-conversation runtime-faithful episode compilation/loading with explicit outcome quality and private-Soul lineage, isolated candidate-Soul branches, immutable lifecycle records, durable per-step progress journals, content-addressed cloud packets, a private Kaggle adapter, and read-only inspection; no model is activated without an explicit governed plan/policy/gate/activation path;
- `scripts/run_axon_heart.py`, `scripts/evaluate_dormant_relevance.py`, `scripts/maintain_dormant_index.py`, `scripts/verify_d64_dual_surface.py`, and `scripts/axon_kaggle.py` ? permanent Heart runtime, deterministic dormant semantic/relevance evaluation, explicit derived-index maintenance/recovery, read-only live D64 dual-surface verification, and private persistent Kaggle training control entry points;
- `training/canonical_d64.py`, `training/complete_field_64d.py`, and `training/train_complete_field_64d.py` ? developmental canonical D64 reasoning path with explicit v2-to-v3 region-embedding and optimizer-state migration; `training/living_reasoning_d64.py`, `training/living_reasoning_curriculum.py`, `training/foundation_motor_curriculum.py`, `training/foundation_sequence_curriculum.py`, `training/living_reasoning_preflight.py`, `training/reasoning_tournament.py`, and `scripts/train_living_reasoning_smoke.py` ? exact Candidate-A anatomy, Unicode-aware exact-copy supervision, causal Soul/runtime unroll, governed Stage-0/Stage-1 foundations, deterministic mechanism replay, six-part launch evidence, head-isolation tournament, and renewable non-serving training; `training/heart_translation.py` plus `scripts/train_heart_translation_smoke.py` ? preserved real-field-D64 Heart translation curriculum/evaluation and Trainer-governed bounded candidate smoke path with no activation,
- `curator/import_d00_memories.py` and the remaining `curator/` recovered-corpus utilities ? protected-source, byte-exact autobiographical import plus offline exact dormant-memory schema/materialization/building tooling; `scripts/compile_lived_experience_sessions.py` ? deterministic governed session compilation from exact Dormant experience,

The former council, old core/soul implementation, ExactV4/identity-v2 runtime stack, 384-slot views/schedules, legacy trainers/curricula, launchers, policies, and dedicated tests are archived beneath `archive/day_zero_legacy_2026-08-20/`. Local historical runs, datasets, checkpoint bundles, and generated distributions are preserved beneath `State/archive/day_zero_legacy_20260820/local_artifacts/`. They may be inspected for provenance or mechanism recovery but may not be imported, launched, resumed, or presented as current Axon without a new explicit convener decision.

There is one Source of Truth text. `docs/SOURCE_OF_TRUTH.md` is the master path and root `SOURCE_OF_TRUTH.md` is a byte-for-byte compatibility mirror. Any doctrine update must update both in the same change; repository tests enforce equality. `docs/WORKING_CONTRACT.md` and root `WORKING_CONTRACT.md` follow the same exact-mirror rule.

## Deleted Architecture

The wide-slot adapter lineage is deleted from this repo. It is not stale, experimental, archived, or fallback code inside `D:\Axon`.

Do not rebuild that path silently. Any future representation change must preserve exact character visibility and must be approved in this source of truth first.
