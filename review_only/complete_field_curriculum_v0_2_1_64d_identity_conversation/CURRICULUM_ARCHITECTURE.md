# Axon Complete-Field Curriculum Architecture v0.2.1

Status: isolated inspection contract

Stamp: ChatGPT / GPT-5 / 2026-08-18

## 1. Purpose

The curriculum trains a width-independent Axon core to process every exact
character in its current active view through ordered physical pages, carry
temporary reader state, use exact span pointers, and emit a typed delta only
after externally verified coverage completes.

The first mechanism proof is 64D and CPU-correct. Larger cores use the same
record contract. A page is processing geometry, never a logical context limit.

The first behavioral proof is deliberately narrow: one 64D learner practices
identity-grounded conversation while every phase sees active text from all ten
regions. After coverage, its successful delta updates scratch, response draft,
and diary in one atomic transaction.

## 2. Two-layer contract

The semantic layer uses exact `shared-field-v1` snapshot bytes and canonical
`shared-field-delta-v1` payload bytes from `runtime/field/schema.py` and
`runtime/field/delta.py`.

The council layer adds the information the canonical delta does not carry:

- one `tick_seq` and strict `phase_seq`;
- persistent `core_id` and per-tick `roster_index`;
- checkpoint and soul-before/soul-after hashes;
- exact sibling roster and phase identity;
- canonical envelope and sibling-set identities;
- base-field and sibling-stream coverage;
- chunk transaction bindings;
- per-core content and attention authority profiles.

The live plain-dictionary council is a bootstrap compatibility surface. This
package does not make it import speculative schemas.

## 3. Independent content and attention switches

Every authorization decision is default deny and evaluates five coordinates:

```text
profile + core + action class + region + operation
```

There are two action classes.

### Content authority

Content operations are `insert`, `delete`, and `replace`. The fixture v2 roster
is intentionally regional:

- `core-alpha`: scratch and response draft;
- `core-beta`: structured knowledge, situation awareness, and task state;
- `core-gamma`: diary and response draft.

No fixture core can maintain the complete field.

The default R0 profile is `core-r0-64d-identity-conversation`. For each
fixture identity it grants content operations on exactly:

- `scratch` for compact visible work products;
- `response_draft` for the living conversational answer;
- `diary` for evidence-grounded exchange records.

It grants no other content region. The broader v2 profiles remain contract
fixtures and future semantic capacity; they are not the first training lane.

Conversation history, user input, tool results, and advisor input are
immutable to cores under every profile. Their trusted runtime ingress adapters
may append or materialize exact evidence, but a core delta may not rewrite it.
Conversation turn commitment is therefore a runtime-owned operation, not a
general core edit.

### Attention authority

Attention operations are separate from content operations. Under the future
v2 profile, a core may move its own one-boundary attention mask for any region,
including immutable tool results. The update changes a derived core view, not
canonical source text.

The update emitted after a phase takes effect on the next tick in these R0
fixtures. It cannot retroactively reduce the current pass's coverage
obligation. Each view binds the full source-text digest, exact boundary, and
active interval. Sliding backward restores exact characters immediately.

The v1 compatibility profile retains operator/runtime-owned canonical masks
and grants cores no mask authority.

`set_intervals` exists only as a denied extension point. Multiple disjoint
intervals are future work and are not required for R0.

## 4. One tick, three phases

One tick contains:

1. Proposal, `phase_seq=0`: every online core reads its complete active base
   field. Sibling input is empty. Outputs do not mutate canonical state.
2. Refinement, `phase_seq=1`: every online core reads its complete active base
   field plus exactly one immutable proposal envelope from every roster member.
   Outputs remain non-mutating.
3. Consolidation, `phase_seq=2`: the rotating crown reads its complete active
   base field plus exactly one refinement envelope from every roster member.
   Only its validated result may commit atomically as tick `t+1`.

Every acting-core phase has exactly one soul inhale and one exhale. Physical
pages never create additional soul transitions.

## 5. Sibling delta identity

`CouncilDeltaEnvelope` wraps, but does not alter, a canonical `FieldDelta`.
It binds core, roster, tick, phase, base snapshot, checkpoint, soul transition,
payload bytes, digest, and provenance. The envelope has its own canonical
identity.

Repeated identities inside the exact `FieldDelta` payload must equal envelope
bindings. A mismatch is a hard failure.

The sibling set is a separate exact namespace, not an eleventh field region.
Refinement and consolidation require an exact ordered envelope for every
expected core. Missing, duplicate, extra, reordered, wrong-phase, wrong-base,
or corrupt siblings prevent decoding.

## 6. Coverage

Every phase carries one `LogicalPassCoverageManifest` with two namespaces:

- `base_field`: the acting core's exact active character stream;
- `sibling_delta_set`: the canonical serialized sibling set when required.

Each namespace binds stream identity, source digest, total characters,
expected indices, ordered pages, page spans, and page digests. The independent
oracle rejects gaps, duplicates, reordering, invalid indices, corruption, or a
different source stream before the writer runs.

Coverage is view-relative but never implicit: a core may attend a smaller
future view only after an authorized, recorded mask update. Canonical content
is not deleted by masking.

## 7. Typed output and chunks

Append is encoded as runtime `InsertText` at the exact current region length.
No-change is a phase outcome without a `FieldDelta`. End-of-delta is transport
control, not a semantic operation.

Chunked output binds transaction ID, expected count, index, byte length,
per-chunk digest, and final assembled digest. All chunks assemble and
deserialize before validation. No chunk mutates state independently.

## 8. Counterfactual construction

The count unit is a complete five-record lineage:

- correct;
- missing page;
- duplicated page;
- reordered page;
- hash-corrupted page.

One-page reordered cases use an invalid page index, preserving all five
variants without falsely requiring two pages. Corruption targets alternate
between base field and sibling stream when siblings exist. The builder runs a
corruption oracle and requires exact expected-versus-observed failure codes.

Default generation schedules exact 1, 2, 4, and 8-page strata and places
evidence across every eligible first/early/middle/late/last bin.

Focused examples keep every canonical region nonempty and expose at least one
active interval from each. The tool region also carries an exact dormant prefix
to prove that complete-field attendance means every currently active character,
not silently unmasking dormant text.

## 9. Evidence boundaries

The exact field remains authoritative. Reader state and souls are not factual
stores. Dormant semantic search may later surface exact, cited source spans
into an active region; it may not replace exact evidence with an embedding or
summary.

Private identity material remains a separate provenance-graded lane. Synthetic
fixtures are never autobiographical.

## 10. Acceptance boundary

v0.2.1 is accepted only after an independent clean-directory review confirms:

- schema documents and records;
- exact snapshot and envelope identities;
- correct one-tick phase structure;
- complete sibling rosters and coverage;
- authority switches and immutable evidence;
- mask restoration without content mutation;
- page and evidence distributions;
- complete counterfactual groups;
- atomic chunk assembly;
- two byte-identical clean generations.
- all ten focused regions contribute active text to coverage;
- every successful focused delta contains exactly scratch, response-draft,
  and diary operations.

Acceptance authorizes planning the smallest CPU R0 reader. It does not itself
authorize training or production integration.
