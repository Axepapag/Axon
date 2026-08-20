# Axon Source Of Truth

Last updated: 2026-07-18

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

## Dormant State

Dormant state is masked structured memory. It stores containers, edges, facts, procedures, episodes, diary entries, source chunks, and provenance.

Dormant memory is not attended directly. Search and surfacing copy relevant readable material into active regions.

Masking is not truncation. Region histories may grow dormant while active masks expose only the current working window.

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
3. Each core attends the active field.
4. Each core emits a proposed delta.
5. Cores may refine against gathered deltas.
6. A consolidator selects or merges the committed delta.
7. Each core exhales experience into hot soul rows.

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

Current R0 trainer:

- `training/train_complete_field_64d.py`,
- exact frozen 16D character input lifted into one 64D core,
- complete ordered paging across all ten active regions before decoding,
- V6 exact-position copy supervision over one dedicated pointer head, with
  immutable region-local source positions retained through paging,
- scratch commit/rematerialize followed by a second complete sweep,
- variable-length response-draft decoding with explicit termination,
- diary, conversation history, and tool results sealed in R0.

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
limit canonical field length or logical-region count.

The differentiable writer is not yet production soul doctrine. It remains
pilot-only until real 64D and 128D write-delay-recall suites independently pass
the round-table causal, regression, optimizer, and state-integrity gates.

## Current Checkpoint Reality

Existing small charfield checkpoints are bootstraps, not finished reasoners. They have learned useful copy and partial-repair behavior, but blank generation is still early.

They are acceptable as starting weights only if their config is charfield-compatible.

## Deleted Architecture

The wide-slot adapter lineage is deleted from this repo. It is not stale, experimental, archived, or fallback code inside `D:\Axon`.

Do not rebuild that path silently. Any future representation change must preserve exact character visibility and must be approved in this source of truth first.
