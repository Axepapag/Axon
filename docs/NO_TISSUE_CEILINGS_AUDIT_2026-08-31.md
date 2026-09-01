# No-Tissue-Ceilings Audit — 2026-08-31

## Executive finding

The reported `512` was not an isolated presentation problem. It was a real
total decoder allowance embedded in D64 model configuration and, for the
living reasoning cores, contaminated architecture identity. The sweep also
found fixed per-item Heart ingress limits, a 512-character recall-query slice,
permanent oversize-evidence exclusion, fixed 128/1024/32768 retrieval clamps,
a three-failure Heart shutdown, and a learned sixteen-dialect Heart table that
rejected any later dialect ID.

Those live ceilings have been removed. Finite numbers that remain in the
protected policy are renewable units of work, not declarations of total
capacity. Physical exhaustion may make Axon pause, page, spill, or fail
visibly. It may not silently discard, truncate, permanently ignore, or call
partial work complete.

## Binding law and protection

The machine-readable authority is
`configs/source_of_truth/capacity_policy.json`. Its canonical SHA256 is
`4a32f18fa296c415ccf34a8c1956d2a4f8afd044265e7f45505782dd53c7b8cd`.
`runtime/source_of_truth.py` embeds that hash and refuses an unratified change.
Both Source-of-Truth mirrors name the same path and hash. Automated source
inspection rejects known capacity-poison identifiers and destructive text
slices before training preflight can pass.

This is deliberately strong but not magical security. An authorized engineer
can edit code and its hash together. Such a change must also change both SOT
mirrors, tests, and the append-only ledger, making the decision reviewable in
Git rather than hidden in tissue.

## Corrections made

| Surface | Before | Now |
|---|---|---|
| Living D64 output | Total configured 512 transport units | External renewable work slices; an in-process iterator carries exact state, while the runtime adapter abstains until durable continuation exists |
| Legacy D64 R0 output | Total configured 512 characters | External diagnostic work slice with explicit incomplete result |
| Existing tournament tissue | Removing the field changed architecture IDs | Existing 1x64, 2x32, and 4x16 IDs preserved through a retired non-operative identity projection |
| First-form curriculum | Targets beyond 512 could be silently excluded | No target-length eligibility rejection |
| Heart ingress | Payload character/byte caps and queue cap | Exact payloads accepted whole; durable spool has no configured count cap |
| Heart beat work | Character target could reject an oversized item | First oversized item crosses whole; later FIFO items defer to later beats |
| Heart liveness | Host stopped after three ordinary failures | Ordinary failures retry after governed backoff; corruption/ownership failures still stop visibly |
| Recall query | Sliced to 512 characters | Complete attended ingress query is used |
| Dormant materialization | Records beyond per-item size were permanently skipped | First selected exact record crosses whole; later records defer |
| Dormant retrieval | Absolute 128/1024/32768 clamps | Work scales with requested results; physical SQL pages remain renewable units |
| Heart dialects | Learned table rejected ID 16 and above | Existing first sixteen embeddings are preserved; later IDs use deterministic content-addressed encoding without a fixed numeric ceiling |
| Valve inventory | Doctrine called twenty slots permanent | Twenty are bootstrap definitions; registries accept additional organs |
| Checkpoint retention | Rolling three hard-coded in execution | Materialized restart count is declared in protected policy; immutable lineage remains complete |

## Numbers that are not prohibited ceilings

- `16D`, rail widths, head counts, layer counts, and FFN sizes are additive
  physical anatomy. Axon may add wider/different tissue without discarding
  proven specialists.
- The 351 Unicode transport categories are an exact categorical encoding, not
  a limit on text length or Unicode scalar coverage.
- Page sizes, work slices, result counts, batches, resource tranches, and
  checkpoint materialization counts are renewable processing policy. They do
  not alter source, tissue identity, or the meaning of completion.
- Authority, integrity, Unicode validity, typed-address, provenance, EOS, and
  transaction checks remain hard. Removing them would make Axon corrupt, not
  free.

## Preserved history

Historical State, archive checkpoints, manifests, launch receipts, and ledgers
were not rewritten. They may still truthfully contain old fields such as a
512-unit decoder allowance or historical `max_steps`. The live code no longer
uses those values as total capacity. Existing reasoning tensor and Soul
identities remain resumable; old historical records remain immutable evidence.

## Remaining reality boundary

No software design creates infinite RAM, disk, time, or compute. Very large
fields and outputs can still exhaust the current machine before future
cross-process paging/spilling is built. The governing response is visible
failure or resumable continuation, never silent truncation. No learned
reasoning core or Heart translator is currently serving. Until-EOS serving
remains promotion-gated on demonstrated EOS behavior and a durable Heart-host
continuation adapter. The in-process core iterator already preserves decoder
state across renewable slices; the runtime correctly abstains instead of
truncating or blocking forever.
