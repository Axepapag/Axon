# Axon Source Of Truth

Last updated: 2026-08-20

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

## Cores

Cores are transformer reasoners with private souls.

Per tick:

1. Runtime materializes the active field.
2. Each core inhales its private soul.
3. Each core attends every exact character of the active field through a
   complete, coverage-proven pass.
4. Each core emits a proposed delta.
5. Each core exhales experience into its soul.
6. Each core inhales its updated soul again, attends the full active field plus
   every complete first-pass core delta, emits a refined delta, and exhales.
7. The rotating consolidator inhales its updated soul, attends the full active
   field plus every complete refined delta, and emits one typed delta against
   the entire shared field.
8. Runtime validates and atomically commits that delta as the next canonical
   shared field; the consolidator exhales its experience.

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
  attended rail character;
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

## Day Zero active surface

The active implementation surface is intentionally narrow:

- `runtime/field/schema.py` ? canonical ten-region exact field schema,
- `runtime/field/delta.py` ? typed canonical deltas and validation/apply/replay,
- `runtime/field/compiler_d64.py` ? exact deterministic D64 compiler,
- `runtime/field/state_branch.py` ? canonical branch persistence,
- `runtime/axon_runtime/d64_adapter.py` ? runtime-facing D64 adapter,
- `training/canonical_d64.py`, `training/complete_field_64d.py`, and `training/train_complete_field_64d.py` ? canonical D64 training path,
- `curator/` recovered-corpus schema/materialization/building utilities ? offline exact dormant-memory tooling,

The former council, old core/soul implementation, ExactV4/identity-v2 runtime stack, 384-slot views/schedules, legacy trainers/curricula, launchers, policies, and dedicated tests are archived beneath `archive/day_zero_legacy_2026-08-20/`. Local historical runs, datasets, checkpoint bundles, and generated distributions are preserved beneath `State/archive/day_zero_legacy_20260820/local_artifacts/`. They may be inspected for provenance or mechanism recovery but may not be imported, launched, resumed, or presented as current Axon without a new explicit convener decision.

There is one Source of Truth text. `docs/SOURCE_OF_TRUTH.md` is the master path and root `SOURCE_OF_TRUTH.md` is a byte-for-byte compatibility mirror. Any doctrine update must update both in the same change; repository tests enforce equality. `docs/WORKING_CONTRACT.md` and root `WORKING_CONTRACT.md` follow the same exact-mirror rule.

## Deleted Architecture

The wide-slot adapter lineage is deleted from this repo. It is not stale, experimental, archived, or fallback code inside `D:\Axon`.

Do not rebuild that path silently. Any future representation change must preserve exact character visibility and must be approved in this source of truth first.
