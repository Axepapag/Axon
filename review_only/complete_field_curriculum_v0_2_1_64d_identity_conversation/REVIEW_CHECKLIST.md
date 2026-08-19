# Independent Review Checklist v0.2.1

Reviewer: Codex

## Package boundary

- [ ] Directory is review-only and outside runtime, State, datasets, and runs.
- [ ] Live council and live validator were not changed by this package.
- [ ] No training or production shard promotion occurred.
- [ ] Package file hashes match `PACKAGE_MANIFEST.json`.

## Contracts

- [ ] Snapshot bytes match `shared-field-v1` canonicalization.
- [ ] Delta payloads match `shared-field-delta-v1` canonicalization.
- [ ] Council envelopes repeat and bind core, tick, phase, base, checkpoint,
      soul, payload, and provenance identities.
- [ ] One tick contains phase sequences 0/1/2; only phase 2 commits.
- [ ] Every required sibling core appears exactly once in order.
- [ ] Base field and sibling set are separate covered namespaces.

## Authority and masks

- [ ] Default decision is deny.
- [ ] Per-core content switches operate independently per region.
- [ ] No supplied core has full-field content authority.
- [ ] Conversation history, user input, tool results, and advisor input are
      immutable to core content deltas.
- [ ] A v2 core may separately receive `set_boundary` authority over immutable
      evidence.
- [ ] v1 rejects content outside scratch/response and grants no core mask move.
- [ ] Mask movement preserves source bytes and takes effect only on a later
      declared view.
- [ ] Disjoint interval operations remain disabled.
- [ ] The R0 focus profile grants content writes to exactly scratch,
      response_draft, and diary.
- [ ] The R0 focus profile exposes all ten regional attention switches.
- [ ] Every focused fixture has nonempty active text in all ten regions.
- [ ] Every successful focused delta contains exactly three operations, one
      for each focused writable region.

## Generator corrections B1-B6

- [ ] Sibling envelopes are attributed and exact.
- [ ] Sibling serialized bytes receive no-gap/no-duplicate page coverage.
- [ ] Council phases share one tick.
- [ ] Evidence positions include every eligible bin.
- [ ] Exact 1/2/4/8 page strata are nonempty.
- [ ] Every group contains all five variants.
- [ ] One-page reordered cases use an invalid index fixture.
- [ ] Builder failure codes equal an independently recomputed corruption.
- [ ] Builder leaves schema/replay/external validation claims null.
- [ ] Non-empty output roots are rejected.
- [ ] Default generation uses 256-character pages and includes one-page
      ten-region examples.

## Independent evidence

- [ ] Compile all Python files.
- [ ] Run all bundled unit tests.
- [ ] Run verifier without importing generator logic.
- [ ] Run two clean generations and compare every file byte.
- [ ] Validate JSON Schema documents and instances with an independent schema
      implementation if available.
- [ ] Append exact commands, hashes, outcomes, and reviewer identity to the
      collaboration document and engineer ledgers.

## Acceptance

- [ ] Explicit ACCEPT or REJECT decision recorded.
- [ ] If accepted, scope is limited to planning/implementing CPU R0.
- [ ] Training remains separately gated.
