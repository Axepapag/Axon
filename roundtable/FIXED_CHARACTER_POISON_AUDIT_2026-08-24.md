# Fixed-Character Poison Audit — 2026-08-24

## Verdict

The objection was correct. The active Heart translator was not merely using a
192-character page: it had finite learned source/target position tables and
explicit validation that rejected anything longer than 192 characters. That was
an architecture ceiling. Continuing to train it would have produced weights for
an anatomy that had to be replaced.

The audit found and repaired the same class of defect in other active paths.
Protected archives and existing State artifacts were inspected but not edited,
deleted, or relabeled.

## Active defects repaired

1. **Heart translation core v1 — architectural source/target ceiling**
   - Removed `max_source_chars=192` and `max_target_chars=192`.
   - Removed finite learned source and decoder position embeddings.
   - Replaced them with deterministic, length-unbounded sinusoidal positions.
   - Added an ordered two-sweep recurrent source reader. A page is now a
     physical compute unit only. Both sweeps visit every exact source character;
     the second produces full addressable memory after the recurrent query state
     has already traversed the complete source.
   - Added per-batch coverage evidence: exact source-index SHA256, source counts,
     page spans, visited counts per sweep, and completion state.
   - Kept full-source pointer/copy memory so referent and grounding positions
     beyond any page remain addressable.
   - Removed the target-training ceiling. Greedy inference accepts a caller
     compute budget; it is not encoded into model capacity.
   - Bumped the model identity to
     `heart-translation-64d-paged-complete-field-v2`.

2. **Heart curriculum v1 — no training pressure beyond the old window**
   - Bumped curriculum/case schemas to v2.
   - Added train and held-out complete-field cases for every critical semantic
     class. Their grounded referent/cue spans begin beyond character 256, forcing
     multi-page reading and long-range pointer behavior.
   - Materialized immutable curriculum v2 at
     `State/training/heart/curricula/ce7125f3ae19369787917d22161d6557dedc3d3b7d2fcf1a147886edeef95850.json`.

3. **Developmental complete-field 64D reader — finite page alias and target cap**
   - Removed `max_pages=4096` and the modulo-wrapped learned page embedding.
   - Removed the learned local-position table as another checkpoint-shaping
     length dependency.
   - Added deterministic unbounded local/page positions.
   - Removed the teacher-forced target ceiling formerly named
     `max_output_chars`.
   - Renamed the remaining greedy limit to `inference_budget_chars`: it is a
     caller-changeable compute budget and does not shape parameters.
   - An inference result that exhausts that budget without EOS is explicitly
     nonterminated and is no longer committed as a canonical delta.

4. **Recovered dormant corpus — destructive source slicing**
   - Removed fixed-character slicing of diary entries, messages, episode
     summaries, facts, relations, goals, objectives, procedures, payloads,
     metadata, and surfaced Cortex examples.
   - Recovered records now retain their full normalized text. Access/redaction
     policy governs sensitive material; loss of source text does not.

5. **Dormant evaluation/retrieval — fixed source/query rejection**
   - Removed evaluation exclusion based on 256 source characters / 64 terms.
   - Removed the 128-unique-term query rejection. All unique terms are processed
     in ordered SQL-safe pages and accumulated before result-count policy.
   - Removed the 32-item metadata inspection slice used by relevance scoring.

6. **Regression prevention**
   - Added `tests/test_no_fixed_character_poison.py`, which fails if active
     neural/corpus paths reintroduce the known architecture-limit identifiers or
     destructive text/content/summary slices.
   - Added executable tests at 257, 521, 1,024, and 5,000 characters and a
     retrieval query with more than 128 unique terms.

## Deliberate bounded mechanisms that remain

These are not model context ceilings and do not destroy source text:

- Shared Field masks: explicit, reversible, audited active/dormant policy.
- Heart valve queue/payload budgets: governed admission/backpressure; rejection
  is explicit and the sender may retry or page the payload.
- Dormant recall result/item budgets: exact whole records are selected or
  skipped with IDs and counts; authoritative dormant records remain intact.
- Retrieval `limit` / top-k and evaluation sample counts: result-set policy, not
  input-character capacity.
- Lexical bucket prefixes and abbreviated hashes: derived indexes/identifiers,
  never substitutes for authoritative text.
- Page sizes: physical compute units. No page count or source length ceiling is
  encoded in the active readers.
- Greedy generation budgets: caller-selected compute bounds. Teacher-forced
  target length is unbounded by configuration, and incomplete runtime output is
  not canonically committed.

## Historical artifact findings

- Repository `archive/`: 53 immutable source/report/test files contain legacy
  limit/truncation terminology or implementations. They remain read-only.
- `State/archive`: 184 total model-like artifacts occupy 67,699,727,421 bytes.
  This audit does **not** claim all 184 are invalid for the same reason.
- Verified fixed-limit subset: 30 archived complete-field run directories have
  configs containing `max_pages=4096` and/or `max_output_chars`; those
  directories contain 78 model artifacts totaling 359,704,706 bytes.
- Current non-archive State still contains one legacy D64 donor smoke config and
  baseline-comparison record with the same v1 settings. They are historical
  artifacts, not edited.
- The current Heart v1 candidate has three checkpoints and an inventory that
  includes `source_position_embedding`; its lifecycle ended `rejected` at step
  12 and no activation/promotion proposal exists. It must not be resumed into
  architecture v2.

## Checkpoint disposition

Do not delete the old checkpoints. Do not load them into v2 with missing-key or
shape-mismatch forgiveness. Preserve them as evidence and, where useful, as
behavioral donors for offline comparison only. New Heart training must start a
new base/candidate generation under the v2 architecture and v2 curriculum.

## Remaining engineering risk

Complete-field support means finite work scales with input size; it does not
promise constant latency or memory. The v2 Heart retains full encoded character
memory for pointer/copy grounding, so very large fields are linear in memory and
decoder cross-attention work. That is an honest scaling cost, not an omission.
Future optimization may add spillable/paged addressable memory, but it may not
change the exact-coverage or no-silent-omission contract.
