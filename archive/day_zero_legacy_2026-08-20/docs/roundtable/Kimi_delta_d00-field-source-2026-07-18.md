# Kimi delta: exact-grounded D:\00 full-field source

Date: 2026-07-18  
Scope: read-only source adapter and audit only  
Kimi session: `session_98d277fa-41b4-4693-b199-1e6b64a8f9b6`

## Verdict

**PASS. No remaining adapter blocker.**

Kimi performed an adversarial read-only review of
`training/d00_field_sources.py` and `tests/test_d00_field_sources.py`. The
initial review returned PASS with six non-blocking observations. The
tag-capacity and hardening follow-up also returned PASS with no remaining
blocker.

This verdict does not authorize cloud export or a GPU launch. Every emitted
record is private/local-only. The adapter writes no curriculum artifact.

## What the adapter uses

The adapter opens exactly two canonical stores:

- `D:\00\axon_semantic_memory.db`
- `D:\00\axon_episodic_memory.db`

Both SQLite connections use URI `mode=ro` plus `PRAGMA query_only=ON`.
SHA-256 is computed before and after extraction and must remain identical.

It does not open:

- `axon_memory.db`
- `axon_memory_backlog.db`
- `axon_runtime_state.db`
- `axon_personal_log.json`

The three old-memory databases are detected and quarantined by filename. Rows
inside the canonical stores are also rejected when their semantic or episodic
`source` marker names one of those residual stores. The personal-log file is
detected with `is_file()` only; its contents are manual-only and were not read.

## Exact grounding contract

Facts, relations, and procedures are eligible only when the raw semantic
components equal an object in an episodic row's `extracted_json` with no
normalization:

- fact: exact `(key, value)`
- relation: exact `(subject, predicate, object)`
- procedure: exact `(name, trigger, steps, outcome, applicability)`

Fact targets are the exact episodic `value`; relation targets are the exact
episodic `object`; procedure targets are an exact indexed step plus the exact
outcome. Every target is between 1 and 256 supported characters. Per-row
provenance contains semantic and episodic database hashes, row/item hashes,
table and numeric IDs, exact JSON paths, selection hash, and the indexed
procedure-step target pointer.

Free-form semantic and episodic `source` values are retained only as
presence-plus-SHA-256 metadata, not copied into the training row.

## Real field material

Each selected row populates:

- `structured_knowledge` with the raw grounded fact, relation, or procedure
  components;
- `conversation_history` with the stored supported episode summary, or a
  provenance-marked fallback derived only from episode ID/type;
- `user_input` with a generic task instruction that contains no normalized
  target answer;
- `situation_awareness` with the actual episode type;
- `task_state` with a non-personal adapter task contract.

No personal fact is invented. Facts and relations produce
`structured_evidence_revision_v4`; procedures produce
`scratch_plan_response_v4`.

## Tagged user-input capacity

The inherited user window has 64 physical slots, but the real rendered
`[user_input]` tag consumes 13. The adapter derives the 51-character payload
capacity by compiling a probe through `runtime.field.compile_field_view`.

Every candidate query is preflighted through that same tagged runtime view.
The emitted record stores canonical/visible hashes and omission counts.
Validation recompiles every selected query, requires exact visible equality,
requires zero omissions, and rejects preflight or capacity drift. Paging is
therefore not needed for these queries and silent clipping cannot pass.

Boundary tests prove 51 characters are fully visible and 52 are not.

## Deterministic bounded selection

The production bound is 8,192 rows with largest-remainder 45/45/10 quotas:

- facts: 3,687
- relations: 3,686
- procedures: 819

Selection uses a fixed salt and canonical SHA-256 priority. It scans all three
semantic tables for audit counts while retaining only the bounded winning set.

## Actual-store audit

The read-only production audit passed:

- selected rows: 8,192
- selected quotas: 3,687 / 3,686 / 819
- selected target count: 9,011
- target length range: 1 to 234
- tagged user-input preflights: 8,192 / 8,192
- tagged user-input capacity: 51
- maximum selected user-input length: 51
- tagged preflight failures: 0
- selected normalized query/answer overlaps: 0
- exact-grounding failures quarantined: 234
- unsupported-text candidates quarantined: 1,487
- over-256 targets quarantined: 254
- query-overlap candidates quarantined: 76
- empty/non-text candidates quarantined: 143
- old-memory residual rows included: 0
- personal-log rows read: 0
- personal-log rows included: 0

Source hashes were unchanged:

- semantic:
  `A94745101D5E48D79224ACDB763D706B65EC632F552599D97613130F36DBC90E`
- episodic:
  `A9A79D29A70B84C427E62B9CE70DB2940712DF53C67C0AD537EA6C50DE18B51C`

No real curriculum file was created by this audit.

## Kimi findings and disposition

1. The procedure scratch target originally pointed to the whole `steps`
   array. It now records the exact selected `steps[index]`.
2. Raw free-form `source` values originally appeared in provenance. They are
   now represented by presence plus SHA-256 only.
3. The CLI reports absolute canonical source paths. This remains intentional
   for a local read-only source audit; the entire result is classified
   local-only and cloud export is forbidden.
4. Separate residual filenames alone could not exclude marked residual rows
   inside canonical stores. Explicit source-marker quarantine is now applied
   before selection.
5. Summary fallback is derived only from episode ID/type and its origin is
   recorded; it contains no invented personal fact.
6. The tests now include the 51/52 tagged user boundary, and the actual
   8,192-row production audit exercised the full quota.

## Integration API and required downstream gate

Use:

```python
result = extract_d00_field_source_records(
    r"D:\00",
    total_limit=8192,
)
for kwargs in result.source_record_kwargs:
    record = SourceRecord(**kwargs)
```

`result.rows` exposes the row dictionaries when a caller does not need the
outer `SourceRecord` fields. `result.audit["passed"]` must be true before any
curriculum build.

The exact-v4 builder now exposes the direct path:

```python
manifest = build_curriculum_from_records(
    records,
    output_dir,
    source_audit=result.audit,
)
```

It accepts no simultaneous `SourceSpec` inputs, requires the exact adapter
identity and a passing source/quota/grounding/preflight audit, requires every
record to be local-only and non-exportable, and re-hashes both SQLite
dependencies before writing the artifact. No lossy intermediate source JSONL
is created.

The normal no-explicit-source CLI is now:

```text
python training\build_multitick_curriculum.py ^
  --source-root D:\00 ^
  --output-dir <new-output-directory> ^
  --grounded-d00-limit 8192
```

When neither `--jsonl` nor `--sqlite` is supplied, this path uses the grounded
adapter and does not discover the seven generic legacy JSONLs. Explicit
`--jsonl`/`--sqlite` inputs retain their existing generic path. Private grounded
D:\00 records cannot be mixed with those explicit sources.

The builder now preserves `row["local_only"] is True` and
`row["cloud_export_allowed"] is False` in every episode, artifact privacy
summary, diary/export summary, and embedded adapter audit. Fixture integration
proved this end to end.

The artifact remains launch-ineligible. The manifest records
`writable_readback_gate_passed=false`,
`read_coverage_launch_gate.passed=false`, and
`artifact_launch_eligible=false`; every episode also records that writable
post-commit readback is not guaranteed.

No real D:\00 curriculum artifact was built during this integration pass.

## Verification

```text
python -m pytest -q -p no:cacheprovider \
  tests\test_d00_field_sources.py tests\test_multitick_curriculum.py
20 passed

python -m pytest -q -p no:cacheprovider \
  tests\test_d00_field_sources.py tests\test_multitick_curriculum.py \
  tests\test_trainer_multitick.py tests\test_field_view_schedule.py \
  tests\test_shared_field.py tests\test_multi_tick_refiner.py \
  tests\test_core_proposer.py
69 passed

python -m pytest -q -p no:cacheprovider
457 passed, 1 skipped, 1 pre-existing Starlette deprecation warning
```
