# Kimi Audit: D00 Soul Production Adapter

Date: 2026-07-18  
Kimi session: `session_67fe9274-8b74-41d2-9edb-334bd68846fd`  
Disposition: **PASS; no blocking issues**

## Scope

Kimi performed a read-only audit of:

- `training/d00_soul_sources.py`
- `training/build_soul_write_delay_curriculum.py`
- `tests/test_d00_soul_sources.py`
- `tests/test_soul_write_delay_curriculum.py`

The audit explicitly prohibited editing files or building an artifact from
`D:\00`.

## Contract reviewed

- `axon_semantic_memory.db` is the sole canonical target store.
- Every emitted fact, relation, or procedure must have an exact canonical
  assertion match in `axon_episodic_memory.db:episodes.extracted_json`.
- Recall queries must exclude the normalized answer value, relation object, or
  first procedure step.
- Production selection is capped at 8192 and allocated deterministically at
  45/45/10 by family without quota redistribution.
- Ungrounded semantic rows are quarantined.
- Every row in every non-internal table of an old-memory database is excluded
  and counted as quarantined.
- `axon_personal_log.json` is manual-only and emits no rows.
- Every emitted source row and derived episode is local-only with cloud export
  disabled.
- The manifest records source hashes, exact-grounding results, family counts,
  quarantine counts, and read-only verification.

## Kimi findings

The initial audit found no correctness, privacy, determinism, or fail-closed
blocker. It gave three non-blocking advisories:

1. A source mtime-only change can conservatively trip the mutation gate.
2. Raw SQLite schema errors could be wrapped in the adapter-specific error.
3. The sensitive-key filter is intentionally broad and may over-quarantine.

The implementation retained the conservative mtime gate and broad sensitive-key
filter. It adopted the schema-error recommendation by wrapping SQLite
schema/read failures as `D00GroundingError`. It also strengthened residual
handling so every row in every non-`sqlite_%` table is counted and quarantined,
while structured and message counts remain separately visible.

Kimi then performed an initial delta check and returned:

> Final recommendation: PASS. Blocking issues: None.

Kimi confirmed that the final residual quarantine and error-wrapping changes
preserve and strengthen the complete contract above.

## Real-preflight capacity correction

A subsequent aggregate production attempt exposed a tag-aware capacity bug
that the fixtures and initial Kimi audit had missed:

```text
FieldViewContractError: user_input is not fully visible in the active
FieldView (expected=57 observed=51)
```

The original adapter treated a 64-slot physical window as 64 text characters.
Runtime tags occupy slots in every window. The correction now:

- derives payload capacities from `runtime.field.compile_field_view` and active
  `SlotKind.SPAN` references rather than encoding 64 as policy;
- filters obviously over-capacity semantic rows before template/rank sampling;
- compiles every row considered for quota admission and requires exact,
  omission-free visibility for the response target, recall query, and the
  actual wrapped Tick-A structured payload;
- quarantines a failed compiled-view row and continues deterministically until
  the family quota is filled, otherwise failing closed;
- records the exact per-row audit in source provenance and aggregate results in
  the manifest;
- uses `proposal_payload_capacity(LogicalRegion.RESPONSE_DRAFT)` as the generic
  builder's response limit;
- requires `unsupported_policy="reject"` for D00 builds before selection or
  output creation, preventing escape-mode expansion of supported literal `[`
  characters after the capacity audit.

Current compiler-derived observations are 51 recall-query characters, 47
response-target characters, and 118 characters for the structured-only Tick-A
context payload. These are regression observations, not hard-coded policy.

An aggregate-only actual-store preflight selected 20 rows with the expected
9/9/2 family split. Exact compiled-view mismatches were zero; grounding,
privacy, source hash/mtime immutability, and the complete adapter audit passed.
After all grounding, template, and compiled-capacity filters, production quota
headroom was:

- facts: 11,081 rows above the 3,687-row quota;
- relations: 17,119 rows above the 3,686-row quota;
- procedures: 1,887 rows above the 819-row quota.

No source content was printed, no curriculum artifact was created by this
preflight, and the failed artifact/logs were retained.

Kimi re-audited the capacity correction, identified the escape-mode `[` edge,
and then verified its fail-fast closure. Final result:

> PASS. No encoded-length bypass remains in a permitted D00 build.

## Direct verification

Command:

```text
python -m pytest -q tests/test_d00_soul_sources.py tests/test_soul_write_delay_curriculum.py -p no:cacheprovider
```

Result:

```text
.............                                                            [100%]
13 passed
```

`py_compile` also passed for the adapter, builder, and focused D00 test module.
The capacity-correction preflight created no curriculum artifact, retained the
earlier failed build and logs, and verified that `D:\00` hashes and mtimes were
unchanged.
