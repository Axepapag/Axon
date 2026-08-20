# RESOLUTION full-field-multitick-soul-2026-07-18

Status: accepted for additive local implementation and verification.

Authority: Jeff's explicit 2026-07-18 direction to train the 64D and 128D
cores inside Axon's real shared-field and multi-tick environment. This
resolution incorporates the read-only Kimi architecture review and independent
Codex audits of the field, curriculum, dormant-memory, checkpoint, and soul
paths. It authorizes local schema, curriculum, evaluator, and CPU-smoke work.
It does not authorize a Kaggle launch or checkpoint promotion by itself.

## 1. Preserve the valuable lineages

- The 64D step-400000 and 128D step-250000 checkpoint lineages are preserved.
- Existing Phase0/Phase0b trainers remain the bootstrap/control path.
- Failed, quarantined, superseded, or unpromoted artifacts remain quarantined.
- No existing checkpoint, recovered source, or dirty-worktree file is deleted.
- 64D and 128D migrate and pass gates independently. Widths are never mixed.

## 2. Canonical field versus checkpoint view

The canonical shared field is an immutable, versioned snapshot with exact text,
typed spans, provenance, parent/tick lineage, and a deterministic hash. It is
not limited to 384 characters.

The inherited 384-position tensor is a checkpoint-compatible active view of
that field. It is not the field itself. Every projection records exact
character references, masks, and explicit omissions; materialization never
silently clips or replaces canonical text.

The canonical top-level regions are:

1. `conversation_history`
2. `user_input`
3. `response_draft`
4. `structured_knowledge`
5. `situation_awareness`
6. `scratch`
7. `tool_results`
8. `advisor_input`
9. `task_state`
10. `diary`

`situation_awareness` is promoted by this resolution to a separately tagged,
provenance-bearing top-level logical region. It is not opaque metadata and does
not add a learned checkpoint tensor in the compatibility phase: all ten
logical IDs remain model-view metadata over the three inherited physical slot
roles.

Dormant state is not an eleventh active region and is never attended directly.
Retrieval copies relevant facts, procedures, episodes, and semantic edges into
the appropriate readable active region with their provenance intact.

## 3. Checkpoint-compatible 384-position profile

The inherited learned positions and three learned type rows keep their exact
meaning as physical slot roles:

| Slice | Width | Learned role | Meaning |
|---|---:|---:|---|
| `0:256` | 256 | `0` | packed readable context spans |
| `256:320` | 64 | `1` | current `user_input` |
| `320:384` | 64 | `2` | one authorized proposal region |

All logical context regions are packed into `0:256` with visible ASCII region
tags and per-character logical-region IDs. The proposal window targets exactly
one authorized region per tick, initially `scratch` or `response_draft`.
Existing proposal text may also be surfaced in the context window so later
ticks can refine it. Core-written `task_state` changes require a later policy
resolution after scratch/response transaction safety is proven.

Blank proposal positions are active writable positions. Unused padding is
masked. Attention, pooling, loss, and exhale must all honor the active mask.

No inherited tensor is resized for this compatibility phase. Logical-region
embeddings, if later desired, require a distinct zero-initialized migration
artifact and tensor-by-tensor inheritance proof.

Because the canonical read field can exceed 320 physical read positions, one
immutable episode cursor advances sealed context-region and `user_input`
offsets across ticks, including across `scratch` to `response_draft` proposal
switches. Every tick records the cursor before/after, compiled view hash,
canonical character-reference hash/counts, cycle status, and the tag-aware
proposal-tail offset. Resetting offsets on every tick is forbidden.

## 4. Ownership and deltas

Runtime-owned/read-only evidence:

- `conversation_history`
- `user_input`
- `structured_knowledge`
- `situation_awareness`
- `tool_results`
- `advisor_input`
- `task_state`

Core-proposable, validator/consolidator-committed state:

- `scratch`
- `response_draft`

`diary` is writable only in an explicit, local, redacted, opt-in diary family.
It is excluded from Kaggle bundles by default.

Every proposal names its base field/tick, target region, operation, exact
character bounds, proposed text, and evidence references. Validation rejects
stale bases, overlaps, out-of-bounds edits, unauthorized regions, and malformed
operations. Canonical Unicode remains intact; unsupported 16D-view characters
and over-budget ranges produce exact omission records rather than silent
substitution. Accepted deltas commit atomically and must replay to the
identical canonical hash.

## 5. Multi-tick contract

Training and runtime share the same state transition:

1. Freeze canonical snapshot `F_t`.
2. Surface relevant dormant evidence into typed active spans.
3. Materialize the masked checkpoint-compatible view.
4. Inhale the core's private soul.
5. Attend and propose one typed region snapshot/delta.
6. Validate and commit or reject the proposal.
7. Materialize `F_t+1`, including the committed scratch/draft.
8. Run the next tick to critique and refine it.
9. Exhale exactly once after the real commit outcome.

Teacher-forced prior commits are allowed during early training. Evaluation must
also run free-running rollouts using the core's actual committed text.
Multi-core proposal boards and consolidation follow the same transaction after
the single-core transition is proven.

The checkpoint-compatible exact-v4 output transaction is fixed at four
64-position segments for any scratch/draft target of at most 256 characters.
Segment zero replaces the current region; segments one through three append at
its current end; empty suffix segments are supervised all-empty no-ops.
Teacher and free-running results are scored both per segment and as an
aggregate reconstructed transition in the 0-64, 65-128, 129-192, and
193-256 length bins. Free-running receives only structural region/phase/segment
schedule data; gold text, length, and hashes remain outside the rollout.

This transaction does not yet prove that all characters of a completed
256-character writable region were subsequently read back. Exact-v4 therefore
records `launch_gate_passed=false` even when its lossless output supervision
passes. A later review/readback transaction must prove complete post-commit
scratch/draft visibility before any GPU launch.

## 6. Curriculum sources and families

`D:\00` is the primary historical source and remains read-only. Derived
curricula live outside it and carry source path, record ID, lineage, source
hash, exact spans, and transformation provenance.

For the production grounded source contract, `axon_semantic_memory.db` is the
canonical target store and the same raw assertion must occur in
`axon_episodic_memory.db:episodes.extracted_json`. The bounded deterministic
mix is 8,192 rows: 3,687 facts, 3,686 relations, and 819 procedures. Queries
must omit their normalized answers and compile fully through the real tagged
view. Ungrounded semantic material and every row from the older residual
memory databases are quarantined. `axon_personal_log.json` is manual-only.
Every emitted grounded row is local-only and has
`cloud_export_allowed=false`.

Initial episode families:

- grammar draft repair
- conversation response refinement
- structured-evidence revision
- tool-result integration
- scratch-plan-then-response
- advisor-driven revision and irrelevant-advisor no-op
- task/situation interruption and resume
- no-op and forbidden-write rejection
- isolated diary continuity only when explicitly opted in

All ticks, corruptions, and derived forms from one source lineage remain in one
deterministic split. Builds use whole-lineage 80/10/10 allocation plus exact
and near-duplicate cross-split leakage audits.

Unsupported source characters remain in the canonical record. A 16D training
view must either reject the example or use a lossless, audited escape/projection
map. Silent replacement or dropping is forbidden.

## 7. Objectives and field gates

Primary objectives:

- per-character proposal cross-entropy
- authorized target-region selection
- changed/unchanged edit-mask prediction
- extra weight on changed positions
- exact committed-region and character accuracy
- edit precision/recall and unchanged-character preservation
- monotonic edit-distance reduction over refinement ticks
- state-transition hash equality
- collapse/diversity metrics

Before a GPU pilot, local evidence must prove:

- all 95 substrate characters round-trip exactly
- inherited copy/partial/blank behavior remains within locked regression bounds
- padding-mask invariance
- provenance and omission-map round-trip
- native scratch and response commits
- stale/overlap/unauthorized delta rejection
- commit-log replay to the identical field hash
- pass-two/free-running refinement improves over pass one without collapse

## 8. Load-bearing soul definition

Soul occupancy, norm, salience, persistence, or nonzero rows are diagnostics;
they are not evidence that the soul affects behavior.

Each core receives an immutable, width-compatible causal suite. The visible
tick-B field is byte-identical in every condition and omits the expected fact:

1. `correct`: the owner's exact soul snapshot
2. `zero`: tensor content zeroed while shape and all metadata stay identical
3. `swapped`: a deterministic compatible donor snapshot
4. `shuffled`: deterministic within-tier/category row permutation

Evaluation is pure: the pre/post soul bytes and metadata hashes must match.
64D and 128D pass separately; their scores are never averaged.

Read-bearing promotion gates:

- at least 256 identities per core
- correct exact accuracy at least `0.90`
- correct character accuracy at least `0.98`
- `CE_zero - CE_correct >= 0.50`
- `CE_swapped - CE_correct >= 0.30`
- paired bootstrap 95-percent lower bounds above zero
- swapped state follows its donor at least `0.80`
- zero and shuffled states produce the locked abstention at least `0.90` exact
- standard-suite character regression at most `0.01`
- standard-suite exact regression at most `0.02`
- finite nonzero intended soul gradients on at least 95 percent of batches
- 100-percent named optimizer coverage for every declared trainable parameter
- deterministic repeat reproduces metrics and output hashes

Writer promotion additionally requires a write-delay-probe curriculum:

- post-write delayed recall exact at least `0.80`
- pre-write and zero-state success at most `0.20`
- swapped state follows donor at least `0.80`
- intended tier/category delta is nonzero
- off-target state delta remains below a locked bound

The current core writer and external manager writer are not promoted. The
saved 168-row `SoulManager` state is authoritative for checkpoint-compatible
reads. The inherited core writer remains frozen. A separate additive
differentiable writer may be exercised only as a quarantined
write-delay-recall pilot over the authoritative 128-hot/32-warm/8-cold layout;
it writes hot rows while warm/cold rows remain byte-identical.

The strict pilot checkpoint permits only that external writer plus the
architecture-derived soul read path (`soul_ingest` and each layer's soul
cross-attention, norms, and gate) to change. All other inherited core tensors
and buffers are source-byte-locked. Source/current manifests and per-tensor
delta hashes are persisted, and load reconstructs exact `requires_grad` and
AdamW coverage. These mechanics remove the former row/optimizer ambiguity but
do not grant promotion standing; every behavioral writer gate still applies.

## 9. Launch order

1. Canonical schema, materializer, delta, commit, and replay tests.
2. Exact-character, mask, provenance, and source-leakage gates.
3. Multi-tick teacher-forced and free-running CPU smokes.
4. Causal soul evaluator and gradient/optimizer coverage smoke.
5. Byte-identical checkpoint inheritance audit.
6. Bounded 64D local/pilot run.
7. Independent evaluation and quarantine review.
8. Only after 64D mechanics pass, repeat independently for 128D.

No Kaggle launch occurs until the local contracts above pass and a fresh launch
manifest is audited. A passed pilot remains quarantined until every applicable
promotion gate is independently verified.
