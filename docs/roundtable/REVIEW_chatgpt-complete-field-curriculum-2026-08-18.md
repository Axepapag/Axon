# Review: ChatGPT Complete-Field Curriculum Inspection Package

Reviewer: Codex / GPT-5 / 2026-08-18

## Verdict

**CONDITIONAL ACCEPTANCE AS A DESIGN DRAFT. DO NOT INTEGRATE OR TRAIN YET.**

The package is thoughtful, doctrine-aware, CPU-compatible in intent, private-
memory cautious, and materially useful. All twelve requested artifacts are
present. Its records parse and validate against its own schemas, and its
reference generator runs deterministically on CPU.

However, the current council schema cannot prove complete sibling-delta
coverage, its council episode models the three phases as different ticks, and
the reference generator does not meet its own position-balance, one-page, or
complete-counterfactual-group contract. Runtime schemas and the production
alphabet also remain unreconciled by the package's own explicit admission.

No package artifact was copied into Axon's production curriculum or dataset
tree, and no model training was launched.

## Reviewed source

- Original archive:
  `C:\Users\axema\Downloads\Axon_complete_field_curriculum_inspection_2026-08-18.zip`
- Archive SHA-256:
  `D5E5F77D4D01EBF5CD098374D1461D19209455BD765E5487575C8DB136181852`
- Archive size: 60,222 bytes
- Entries: 12 files plus one directory
- Expanded size: 271,296 bytes
- Archive safety inspection: no path traversal, duplicate names, encryption,
  or suspicious compression ratio found
- Original archive remained unchanged

The archive was extracted only into an isolated temporary review directory.
The static generator review found only Python standard-library imports and no
network, subprocess, shell, registry, or system-control path. After that
review, it was executed twice into two new isolated temporary output roots.

## Deliverable inventory

All requested artifacts were present:

1. `CURRICULUM_ARCHITECTURE.md`
2. `curriculum_families.json`
3. `complete_field_example.schema.json`
4. `curriculum_manifest.schema.json`
5. `starter_examples.jsonl`
6. `frozen_evals.jsonl`
7. `GENERATOR_SPEC.md`
8. `generate_curriculum.py`
9. `TRAINING_SCHEDULE.md`
10. `PRIVATE_MEMORY_INGESTION_SPEC.md`
11. `REVIEW_CHECKLIST.md`
12. `OPEN_QUESTIONS.md`

## Verified strengths

### Architecture

- Preserves all ten canonical field-region names in the ruled order.
- Treats a page as physical processing geometry, not a logical context limit.
- Requires CPU correctness and treats GPU as optional acceleration.
- Separates primary full coverage from bounded exact-span re-reads.
- Keeps exact source text authoritative instead of claiming finite reader
  state losslessly contains arbitrary text.
- Places coverage validation outside the model's self-report.
- Requires one soul inhale/exhale per logical phase rather than per page.
- Keeps curricula width-independent and recommends fresh 64D mechanism proof
  before 128D comparison.
- Separates synthetic mechanics, public foundations, private identity, and
  quarantine lanes.
- Preserves Grade A/B/C/D/S identity evidence boundaries and prohibits
  synthetic autobiography.

### Data contracts

- Both JSON Schema documents parse as draft 2020-12 schemas.
- `starter_examples.jsonl`: 18/18 rows parse and pass the bundled example
  schema; every family A-M is represented, plus linked scratch and council
  episode records.
- `frozen_evals.jsonl`: 15/15 rows parse and pass the bundled example schema.
- No duplicate example/episode IDs were found.
- No lineage split leak or starter/frozen-eval lineage overlap was found.
- Recomputed starter/eval snapshot hashes, page hashes, active-character
  counts, evidence spans, normal page coverage, operation terminators, and
  successful final snapshots matched their records.
- Starter and frozen-eval snapshots/targets had no exact cross-file duplicate.

### Reference generator

- Python AST parses successfully.
- Imports are standard library only.
- Two isolated 64-record CPU runs completed successfully.
- Every generated record passed `complete_field_example.schema.json`.
- The generated manifest passed `curriculum_manifest.schema.json`.
- All four output files from the two runs were byte-identical, proving the
  tested run deterministic externally.
- Generated output contained train, dev, and test shards with every lineage
  confined to one split.

## Blocking findings

### B1. Sibling deltas are not coverage-proven or machine-identifiable

**Evidence**

- `sibling_deltas` is an array of generic delta transactions.
- A transaction contains `transaction_id`, snapshot hash, atomic flag,
  chunk count, and operations, but no required `core_id`, source phase,
  source soul/checkpoint binding, or roster position.
- The main `page_plan` covers only the ten-region snapshot. It does not page or
  include coverage spans for serialized sibling proposals/refinements.
- The expected online roster and consolidator identity appear only as free
  text inside `task_state` in starter examples.

**Impact**

The schema cannot prove that every expected core contributed exactly once,
that no sibling was omitted or duplicated, or that every character of every
complete sibling delta was visited. This fails the binding Phase-B and
consolidation contract.

**Required correction**

- Add structured `online_core_ids`, `acting_core_id`,
  `consolidator_core_id`, and `expected_sibling_core_ids`.
- Replace the untyped array with a versioned sibling-delta set whose entries
  bind `core_id`, phase, input snapshot, transaction hash, exact serialized
  payload, and provenance.
- Page sibling payloads exactly like field text or place their exact lossless
  rendering in a ruled input stream.
- Extend coverage expectations/manifests across both the active field and the
  complete sibling-delta streams.
- Reject missing, duplicate, extra, truncated, wrong-phase, or wrong-snapshot
  sibling entries before finalization.

### B2. Proposal, refinement, and consolidation are modeled as separate ticks

**Evidence**

The starter council episode assigns:

- proposal: `tick_index = 0`
- refinement: `tick_index = 1`
- consolidation: `tick_index = 2`

Binding doctrine defines all three as phases inside one tick.

**Impact**

Training on these episode indices would teach the wrong temporal and soul
lifecycle boundary and could rotate the crown or commit state at the wrong
time.

**Required correction**

- Give all three records one `tick_seq`.
- Add `phase_seq` or a strict phase enum/order inside that tick.
- Bind each phase to its input/output soul hashes and enforce exactly one
  inhale/exhale for its complete sweep.
- Commit only the final consolidator transaction as the next canonical field.

### B3. The default generator fails evidence-position balance

**Evidence**

The isolated default 64-record run reported:

```text
first: 64
early: 0
middle: 0
late: 0
last: 0
```

`deterministic_text` places the marker near the first page boundary, and
`marker_page` therefore always resolves to the first page.

**Impact**

This directly permits the first-page shortcut the curriculum is supposed to
defeat. The generated smoke cannot establish complete-field use.

**Required correction**

- Schedule target evidence bins explicitly and deterministically.
- Place full and boundary-crossing needles across first, early, middle, late,
  and last bins.
- Require nonzero per-bin minimums in the builder and manifest validator.
- Fail the build if requested balance cannot be satisfied.

### B4. The default smoke omits one-page examples

**Evidence**

The 64-record default run produced zero records below two pages. `main()`
forces every requested page count to at least two whenever `reordered_page`
is among the selected variants, even for non-reordered variants. The package's
CPU smoke specification explicitly calls for 1, 2, 4, and 8 page cases.

**Required correction**

- Choose page count per counterfactual group independently of whether one
  variant needs two pages.
- Either represent the one-page reordered corruption using the already-coded
  invalid page index case, or omit only that impossible variant with an
  explicit counted reason while keeping other one-page variants.
- Make exact boundary page counts first-class strata rather than relying only
  on random sampling inside a range.

### B5. The default count emits an incomplete counterfactual lineage

**Evidence**

With five configured variants and `--count 64`, the generator emitted twelve
complete five-variant groups and one four-variant group. Lineage 12 lacks
`hash_corrupted`.

**Impact**

The manifest reports 64 accepted records but does not flag that a required
matched variant is missing. Paired causal evaluation can silently become
unbalanced.

**Required correction**

- Interpret count as complete lineage groups, or round only with an explicit
  incomplete-group rejection.
- Validate the exact required variant set for every counterfactual group.
- Report any intentionally incomplete group and prohibit it from paired gates.

### B6. Generator validation claims exceed validation actually performed

**Evidence**

- `validate_complete_pages()` exists but is never called.
- Non-positive variants are accepted by checking only that their expected mode
  is `failure`; the validator does not independently recompute that each
  declared failure code matches the supplied corruption.
- The manifest sets `round_trip_valid: true` and
  `deterministic_replay_valid: true` inside one run. The generator does not
  perform the documented second clean regeneration.
- The manifest truthfully leaves `schema_valid: null`, while listing semantic
  validators such as `page_hash` and `coverage_variant` that were not all
  independently applied.

External review did prove deterministic replay for the tested invocation, but
the emitted manifest must describe evidence produced by its own build or a
named external verifier.

**Required correction**

- Call an independent page-integrity oracle for every variant.
- Require actual versus expected failure-code equality.
- Set replay/schema fields to `null` until an external verifier supplies a
  signed result, or make the builder actually perform those checks.
- Distinguish `builder_validation` and `external_validation` with tool/version,
  run ID, and evidence hashes.

## Required runtime reconciliation before acceptance

These are not necessarily defects because the package labels them as
proposals, but they block production compatibility:

1. Replace `synthetic_ascii_smoke_v1` with the actual frozen Axon alphabet and
   16D table/version for production records.
2. Reconcile snapshot canonicalization and hashing against
   `runtime/field/`; do not invent a parallel authority.
3. Reconcile character offsets with Axon's actual character-table cell
   indexing and encoding boundaries.
4. Reconcile typed delta operations with the existing runtime parser,
   validator, replay, and atomic commit schemas.
5. Decide whether multiple disjoint active intervals are canonical state or a
   derived page representation. Current council masks are a persisted boundary
   with tail/manual/threshold policy; a broader mask model must not silently
   amend doctrine.
6. Add actual variable-delta chunk boundaries and ordering if `chunk_count`
   can exceed one; the current schema stores only a scalar count plus one
   operation array.
7. Bind soul fixtures and phase transitions to versioned before/after hashes
   if they will support causal production claims.
8. Require a new empty output directory or fail. The reference generator uses
   `mkdir(..., exist_ok=True)` and may overwrite named shards while leaving
   unrelated stale files in an existing directory.

## Non-blocking observations

- The inspection examples are intentionally tiny and synthetic. They are
  useful schema fixtures, not sufficient training volume or proof.
- The package does not provide public foundation datasets or licenses; it
  correctly leaves that as an open decision instead of inventing sources.
- The private-memory specification is conservative and compatible in spirit
  with the existing Grade A/B/C/D/S policy.
- Scratch, soul, and council causal gates are well described, but the frozen
  fixture count is far too small for statistical promotion claims. The
  training schedule correctly anticipates larger suites.
- The manifest payload self-hash convention should explicitly state whether
  the hash field is omitted, nulled, or replaced with a fixed placeholder
  during hashing.

## Acceptance matrix

| Area | Status |
| --- | --- |
| Twelve requested artifacts present | Accept |
| High-level architecture | Accept with runtime reconciliation |
| CPU inference requirement | Accept |
| 64D-first, width-independent curriculum | Accept |
| Evidence-grade private memory policy | Accept |
| JSON syntax and self-schema validation | Accept |
| Starter/frozen fixtures | Accept for inspection only |
| Reference generator determinism | Accept for tested invocation |
| Reference generator coverage balance | Reject pending B3/B4/B5 |
| Generator evidence claims | Reject pending B6 |
| Council tick semantics | Reject pending B2 |
| Complete sibling-delta coverage | Reject pending B1 |
| Production runtime compatibility | Not yet established |
| Authorization for training | Not granted |

## Recommended next action

Create a corrected **inspection package v0.2**, not production datasets yet.
The bounded correction mission should:

1. fix B1-B6;
2. reconcile the snapshot, offset, alphabet, delta, mask, and soul contracts
   against current Axon code;
3. add automated schema and semantic tests;
4. rerun deterministic generation twice;
5. prove balanced evidence positions, exact boundary page counts, complete
   counterfactual groups, correct council tick/phase structure, and complete
   sibling-stream coverage;
6. return the revised package for one more go/no-go review.

Only after that review should Axon implement the smallest CPU
`CompleteFieldReader` and generate an R0 training shard.

## Verification commands and results summary

- ZIP inventory/safety: PASS
- JSON parse: PASS, all JSON/JSONL files
- JSON Schema validation:
  - starter: 18/18 PASS
  - frozen eval: 15/15 PASS
  - generated examples: 64/64 PASS
  - generated manifest: PASS
- Python AST/static import inspection: PASS
- Isolated CPU generator run 1: PASS, 64 records
- Isolated CPU generator run 2: PASS, 64 records
- Byte-identical replay: PASS, all four generated files
- Evidence-position balance: FAIL, first=64 and all other bins=0
- One-page default smoke coverage: FAIL, zero one-page records
- Complete counterfactual groups: FAIL, final lineage has 4/5 variants
- Council tick/phase fidelity: FAIL, phases modeled as tick 0/1/2
- Complete sibling-delta paging/coverage: FAIL, absent from schema
- Model training: NOT ATTEMPTED

Identity stamp: Codex / GPT-5 / 2026-08-18
