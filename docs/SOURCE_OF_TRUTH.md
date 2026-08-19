# Axon Source Of Truth

Last updated: 2026-08-18

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

Current trainer:

- `training/trainer_slot.py`
- exact 16D charfield input,
- copy / partial / blank response draft modes,
- suffix-only partial metrics,
- story-aware continuation where available.

Additive full-field path:

- `runtime/field/`: immutable ten-region canonical snapshots, exact typed
  spans/provenance, checkpoint-compatible masked 384-character views, and
  validated/replayable deltas,
- `runtime/multi_tick_refiner.py`: commit/rematerialize/refine rollouts with
  explicit teacher-forced versus free-running modes,
- `training/build_multitick_curriculum.py`: source-lineage episodes from
  read-only recovered material,
- `training/soul_load_bearing.py`: per-core correct/zero/swapped/shuffled
  causal evaluation,
- `training/differentiable_soul_writer.py`: unpromoted 168-row
  write-delay-recall pilot that may write hot rows only.

The 384-character view and three learned type IDs are physical
checkpoint-compatibility roles, not the complete logical field. They do not
limit canonical field length or logical-region count. Existing checkpoints
that consume only one such view are bootstrap checkpoints and do not satisfy
the required full-field council protocol until a trained complete-sweep reader
has been added and passed coverage and behavioral gates.

## Canonical state root

All living or durable Axon runtime state resides beneath `D:\Axon\State`,
including canonical field state, dormant memory, private souls, active adapter
pointers and promoted adapters, cursors, and offline-learning control records.
Candidate training artifacts and reproducible run logs may remain under
`runs/`; promotion copies the active state-bearing artifact into `State/` with
provenance.

The differentiable writer is not yet production soul doctrine. It remains
pilot-only until real 64D and 128D write-delay-recall suites independently pass
the round-table causal, regression, optimizer, and state-integrity gates.

## Current Checkpoint Reality

Existing small charfield checkpoints are bootstraps, not finished reasoners. They have learned useful copy and partial-repair behavior, but blank generation is still early.

They are acceptable as starting weights only if their config is charfield-compatible.

## Deleted Architecture

The wide-slot adapter lineage is deleted from this repo. It is not stale, experimental, archived, or fallback code inside `D:\Axon`.

Do not rebuild that path silently. Any future representation change must preserve exact character visibility and must be approved in this source of truth first.
