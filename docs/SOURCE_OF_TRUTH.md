# Axon Source Of Truth

Last updated: 2026-08-22 (Build B: beat coordinator, ingress queue, and per-region attention masking ratified)

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
- `structured_knowledge`
- `situation_awareness`
- `scratch`
- `tool_results`
- `advisor_input`
- `task_state`
- `diary`

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

A heartbeat is event-driven: a canonical field change is the primary
doorbell. While input or commit work exists, the heart beats promptly. While
idle, the heart may keep a slow bounded liveness beat. Proposal-board
activity advances the in-flight tick workspace and does not by itself create
a new canonical field.

Each beat:

1. Drains queued external arrivals. Between ticks, intake commits immediately
   as a heart-governed typed delta into its runtime-owned region. Arrivals
   during an in-flight tick queue for the next beat and never mutate the
   frozen base.
2. Applies per-region attention masks. Masked text remains canonical and
   restorable; only attended intervals enter the compiled rails. Policies are
   resolved from explicit attended intervals or reusable mask policies such as
   `all`, `none`, or `last_n_spans`.
3. Detects change via canonical field identity/freshness. No change means no
   recompilation.
4. Runs the dormant valve when change warrants recall.
5. Recompiles the affected rail(s), proving complete coverage and exact
   roundtrip against the fresh canonical field.
6. Services the tick workspace: collecting proposals, enforcing barriers, and
   committing the validated consolidator decision.

Build B realizes the first living circulation organ: `runtime/heart/ingress_queue.py`
holds external arrivals, and `runtime/heart/coordinator.py` drains the queue,
reapplies masks, runs primitive recall, freezes a D64 tick image, and guards
against commits during an in-flight tick. Build B stops at the frozen tick
image; proposal/refinement/consolidation barriers attach to that image in later
builds.

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
- The dormant valve may submit heart-governed materialization of governed
  `structured_knowledge`; it never independently writes truth.
- Core proposals may target only the scopes their authority class permits.
- The consolidator's proposal may address every canonical region as governed.
- Only the heart's transaction layer converts any proposal into canonical
  state. Validation must detect and reject conflicting or overlapping sparse
  edits and any proposal not bound to the frozen base.

Bootstrap write restrictions (such as a scratch/response-only validator) are
implementation restrictions, not doctrine; widening happens only through this
authority model.

## Cores

Cores are transformer reasoners with private souls. A tick is one full
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

Pre-Day-Zero 384-slot readers, ExactV4 runtime/trainer paths, multi-tick prototypes, soul pilots, detached curriculum builders, and their dedicated tests are historical evidence only under `archive/day_zero_legacy_2026-08-20/`. They are not active fallback interfaces.

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
structure; trained semantic richness enters in the 128/256/512/1024 dialing
sequence, one rail size at a time, each proven before the next begins, and may
also improve the 64D heart provided exact-character grounding and roundtrip
remain mandatory. A reasoning core is never required to reproduce every
character to prove grounding.

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

Every index build is bound to SHA256 identities for the authoritative dormant
files plus the recovered source hashes recorded by `corpus_manifest.json`.
Candidate retrieval returns IDs. Before evidence can enter the shared field,
the bridge seeks back into the authoritative JSONL, rereads the exact bytes,
and verifies raw-record hash, record identity, exact text hash, source hash,
and provenance hash. A changed corpus makes the derived index stale and it
fails closed until rebuilt.

Selected exact container text is surfaced as provenance-bearing `FieldSpan`
material in canonical `structured_knowledge`; source container IDs and verified
semantic-edge IDs remain attached as references. Generated separators are
explicit runtime spans rather than alterations to dormant text. The resulting
`SharedFieldSnapshot` must compile through the deterministic D64 compiler with
complete coverage and exact roundtrip before it is accepted as core-facing
state. Retrieval/indexing has no reasoning vote and no commit authority.

The dormant valve is part of the heart. Candidate ranking combines graph/edge
semantics, lexical support, confidence, type and task relevance, and novelty
versus the active field, under a governed budget. First-form semantic
relevance is recovered-edge graph semantics: the readable English semantic
edges and container graph are walked from changed text to find related
memories. This is real semantics already owned by the corpus; no trained
encoder exists and none may be simulated. Learned vector similarity, when
trained later, is a disposable derived sense only. The semantic path fails
closed to exact lexical retrieval, and exact bytes are always dereferenced
from the authoritative JSONL and hash/provenance-verified before surfacing.

The derived index must become incremental/generational: append/update with
binding verification and atomic swap. That capability does not exist yet;
until then, full rebuild is the accepted static cost, and it must never
become the per-beat cost of live memory.

## Day Zero active surface

The active implementation surface is intentionally narrow:

- `runtime/field/schema.py` ? canonical ten-region exact field schema,
- `runtime/field/delta.py` ? typed canonical deltas and validation/apply/replay,
- `runtime/field/compiler_d64.py` ? exact deterministic D64 compiler,
- `runtime/field/state_branch.py` ? canonical branch persistence,
- `runtime/axon_runtime/d64_adapter.py` ? runtime-facing D64 adapter,
- `runtime/dormant/evidence_bridge.py` ? read-only manifest/hash-bound dormant retrieval, exact dereference, and structured-knowledge surfacing,
- `runtime/heart/` ? heart control plane: authority classes, core registry, tick identities and frozen images, the noncanonical proposal board, the heart transaction boundary, the ingress queue, and the beat coordinator;
- `training/canonical_d64.py`, `training/complete_field_64d.py`, and `training/train_complete_field_64d.py` ? canonical D64 training path,
- `curator/` recovered-corpus schema/materialization/building utilities ? offline exact dormant-memory tooling,

The former council, old core/soul implementation, ExactV4/identity-v2 runtime stack, 384-slot views/schedules, legacy trainers/curricula, launchers, policies, and dedicated tests are archived beneath `archive/day_zero_legacy_2026-08-20/`. Local historical runs, datasets, checkpoint bundles, and generated distributions are preserved beneath `State/archive/day_zero_legacy_20260820/local_artifacts/`. They may be inspected for provenance or mechanism recovery but may not be imported, launched, resumed, or presented as current Axon without a new explicit convener decision.

There is one Source of Truth text. `docs/SOURCE_OF_TRUTH.md` is the master path and root `SOURCE_OF_TRUTH.md` is a byte-for-byte compatibility mirror. Any doctrine update must update both in the same change; repository tests enforce equality. `docs/WORKING_CONTRACT.md` and root `WORKING_CONTRACT.md` follow the same exact-mirror rule.

## Deleted Architecture

The wide-slot adapter lineage is deleted from this repo. It is not stale, experimental, archived, or fallback code inside `D:\Axon`.

Do not rebuild that path silently. Any future representation change must preserve exact character visibility and must be approved in this source of truth first.
