# Generator Specification v0.2.1

Stamp: ChatGPT / GPT-5 / 2026-08-18

## Interface

```text
python generate_curriculum.py
  --package-root PATH
  --output-root NEW_PATH
  --groups MULTIPLE_OF_20
  --page-size INTEGER_AT_LEAST_256
  --seed INTEGER
```

`--groups` is the number of complete lineages, not the number of rows. Every
lineage emits exactly five variants. The default 20 groups emit 100 rows.

`--build-review-fixtures` is a maintainer-only operation used to rebuild the
bundled starter and frozen fixtures. It does not run during ordinary shard
generation.

## Determinism

All text, rosters, phases, splits, page strata, evidence positions, operations,
and corruptions are arithmetic functions of the group index. The builder uses
canonical compact UTF-8 JSON for hashed identities and sorted compact JSONL
for rows. It does not use wall-clock time, random module state, network calls,
subprocesses, CUDA, or external services.

The supplied `seed` is recorded for future generator families. v0.2-compatible fixtures
remain index-derived so replay is byte-identical.

Every focused record contains nonempty active text in all ten canonical
regions. Its successful transaction contains three insert operations in the
fixed order `scratch`, `response_draft`, `diary`. The default page size is 256
characters so even one-page fixtures can represent every region meaningfully.

## Output safety

The builder accepts a missing or empty output directory and rejects a
non-empty output directory. It never resumes or overwrites stale output.

It writes:

- `train.jsonl`;
- `dev.jsonl`;
- `test.jsonl`;
- `manifest.json`.

Every counterfactual lineage remains in one split.

## Builder checks

The builder actually checks:

- alphabet manifest identity and length;
- exact runtime-style snapshot identity;
- selected target authority;
- complete ten-region active coverage for the focused profile;
- exact focused three-region delta membership;
- sibling roster construction;
- complete variant sets;
- exact 1/2/4/8 strata;
- non-empty eligible evidence bins;
- corruption oracle expected-versus-observed equality;
- non-empty output rejection.

The builder does not claim JSON Schema validation or two-run replay. Those
manifest fields remain `null`. External validation remains `null` until a
separate verifier emits a separate evidence report.

## Independent verifier

`verify_curriculum.py` does not import the builder. It independently
recomputes snapshots, envelopes, sibling sets, source streams, coverage
failures, policies, evidence pointers, soul bindings, and chunk assembly. It
launches two isolated clean builder processes and compares every output byte.

## Position scheduling

- 1 page: first;
- 2 pages: first or last;
- 4 pages: first, early, late, or last;
- 8 pages: first, early, middle, late, or last.

The default 20-group schedule includes five 8-page groups, guaranteeing all
five bins without lying about bins that cannot be distinct in shorter inputs.

## Failure codes

Namespace-prefixed failures distinguish base and sibling coverage. Examples:

```text
BASE_FIELD_MISSING_PAGE
BASE_FIELD_PAGE_INDEX_INVALID
SIBLING_DELTA_SET_DUPLICATE_PAGE
SIBLING_DELTA_SET_HASH_MISMATCH
```

The exact code is recomputed from the corrupted record. Negative records carry
no target delta or mask update.
