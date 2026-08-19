# v0.2.1 Contract Decision Matrix

| v0.2 field or rule | Source authority | Decision in this package | Production status |
| --- | --- | --- | --- |
| Ten region names/order | `runtime/field/schema.py`, Source of Truth | Exact canonical order | Existing semantic contract |
| Snapshot identity | `SharedFieldSnapshot` | Canonical compact UTF-8 JSON SHA-256 | Existing semantic contract |
| Delta payload | `runtime/field/delta.py` | Exact `shared-field-delta-v1` bytes | Existing payload contract |
| Council identity | Full-field resolution and Codex reply | Versioned wrapper; do not mutate `FieldDelta` | Review only |
| Tick/phase | Source of Truth | One tick; proposal/refinement/consolidation = 0/1/2 | Binding doctrine |
| Sibling input | Source of Truth and Codex reply | Separate exact namespace; complete expected roster | Review only |
| Coverage | Source of Truth | Base and sibling namespaces both externally verified | Review only |
| Stable core identity | Codex reply | Persistent `core_id`; roster/checkpoint/soul separate | Registry not implemented |
| Alphabet | `substrate.default_alphabet()` | `axon-char-alphabet-v8`; exported ordered manifest | Verified source identity |
| Hash transport | Runtime convention | Separate `algorithm: sha256` plus lowercase bare hex | Review only |
| Append | Runtime `InsertText` | Insert at exact current region length | Existing operation |
| No change | Runtime `FieldDelta` nonempty invariant | Phase outcome without a delta | Review only |
| End token | Codex reply | Decoder/wire control only | Decoder not implemented |
| Chunking | Codex reply | Ordered complete transaction; assemble then apply once | Review only |
| v1 content authority | Current runtime validator | Scratch and response draft only | Existing compatibility behavior |
| v2 content authority | Jeff clarification 2026-08-18 | Default-deny per-core region switches | Review only |
| First 64D profile | Jeff clarification 2026-08-18 | Read active text from all ten regions; write only scratch, response draft, diary | Review-only R0 target |
| Focused delta | Jeff clarification 2026-08-18 | One typed transaction contains operations for all three writable focus regions | Review only |
| Identity material | Evidence-grade doctrine | Synthetic Axon-name fixtures only; private biography requires approved Grade A/B evidence | Private lane not populated |
| Immutable evidence | Jeff clarification 2026-08-18 | Conversation, user input, tool/advisor results core-immutable | Binding direction; adapter pending |
| Attention authority | Jeff clarification 2026-08-18 | Separate per-core switches; tool evidence may be masked | Review only |
| Current mask model | Live council behavior | One movable suffix boundary | Existing bootstrap behavior |
| Future interval masks | Jeff approval 2026-08-18 | Schema extension point present but disabled | Unavailable in R0 |
| Mask effective time | Conservative engineering choice | Next tick; no retroactive coverage change | Open for later phase policy |
| Builder validation | Codex review B6 | Claims only checks run internally | Implemented here |
| External validation | Codex review B6 | Separate verifier evidence; manifest remains null | Implemented for review |
| Package placement | Codex reply | Isolated review-only repository directory | Implemented here |
| Runtime integration | Codex reply | Prohibited until second acceptance | Not authorized |
| Training | Working Contract and Codex reply | Prohibited until R0 gates | Not authorized |
