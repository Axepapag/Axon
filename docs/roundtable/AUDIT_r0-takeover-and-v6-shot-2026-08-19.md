# Audit: R0 takeover and the v6 shot

Date: 2026-08-19
Auditor: Codex / GPT-5
Scope: changes and evidence after `evt-20260818T202220895739Z-codex-jeff-field-authority-ruling`
Result: implementation accepted as a bounded mechanism prototype; all trained
candidates remain rejected; curriculum integrity defects fixed; no new training
authorized

## Failures first

1. **Arbitrary binding has not been learned.** V5 intermittently emits familiar
   names but does not retrieve held-out random tool tokens. At step 2,000,
   exact cross-page retrieval remained 0 and exact copy was 0.3333.
2. **The old synthetic copy gate was not a held-out binding gate.** It repeatedly
   used only `Axon`, `Jeff`, `Council`, `scratch`, and `response` in every split.
   Exact fields overlapped between train/dev/test, so familiar-name output could
   not prove generalization.
3. **Conversation-foundation labels were contradictory.** The same three exact
   input fields each had all three response targets. A deterministic model
   could not reach the required 0.50 exact rate when labels were approximately
   uniform; this was a curriculum defect, not a model failure.
4. **The current scratch counterfactual contract confuses causality with
   obedience.** For copy and retrieval tasks, immutable `user_input` or
   `tool_results` resolves the answer. Empty or conflicting scratch should not
   force a generic meta-response. Scratch is working state, not factual
   authority. The existing unconditional response-change gate must be replaced
   before another pilot.
5. **The pointer objective is position-ambiguous.** Character likelihood credits
   every source occurrence of the same character. Common characters therefore
   reward diffuse frequency attention rather than one contiguous evidence span.
6. **The live bootstrap council is stopped.** `127.0.0.1:8788` actively refused
   the status request. It was not restarted because this audit concerns the
   training line and the previous handoff explicitly separated service recovery.

## Report claims compared with actual artifacts

| Claim | Audit result | Evidence |
|---|---|---|
| Complete ordered paging over all ten regions | VERIFIED | `CompleteFieldPager`, coverage manifests, focused tests |
| Decode blocked without complete coverage | VERIFIED | `read_field_with_memory` raises before decoding |
| Scratch is committed and the entire field reread | VERIFIED | `forward_transaction` and `run_transaction` perform two full reads |
| Exact addressable source characters exist | VERIFIED | `AddressableMemory.char_indices` and pointer scatter path |
| Empty region markers cannot be copied | VERIFIED | `char_indices=-1`, masked before pointer scatter |
| Checkpoint resumes bind datasets and RNG state | VERIFIED by code/tests and prior artifacts | checkpoint v5 payload and fingerprint refusal |
| V5 passed behavioral promotion | FALSE, correctly reported as rejected | step-2,000 `gate.json` has `promotion_allowed=false` |
| No 200k run or State promotion occurred | VERIFIED | no training process; no promoted candidate evidence |
| Full suite was green after implementation | independently VERIFIED | 1,092 passed, one expected skip after this audit's two new tests |
| Live council running | FALSE at audit time | loopback connection actively refused |

## Pointer review

The v5 probability mixture is numerically coherent. After excluding invalid
empty-region markers, renormalization produces a valid distribution over real
source characters; when no copy source exists, the model falls back to
generation. The primary shortcut is not that normalization. It is collapsing
all positions containing the same target character into one character-level
reward, compounded by averaged multi-head attention.

The next mechanism should use explicit contiguous-span supervision before a
monotonic pointer state. Minimum alignment labels:

- base field ID and phase/head;
- target region plus target start/end;
- source namespace, region, start/end, and exact text hash;
- one source position or an explicitly enumerated equivalent-occurrence set
  for each supervised target character;
- copy-versus-generate label per supervised target position;
- EOS as generated control, never a copied field character;
- exact page/region coordinates and coverage-manifest ID.

Only unambiguous exact spans receive the auxiliary loss. Ordinary prose remains
under language loss. Anti-shortcut fixtures must include unseen mixed-case
alphanumeric strings, lengths 4-32, balanced character frequencies, repeated
same-token distractors, same token in a wrong region, nearby decoys, page-boundary
spans, first/middle/last positions, and corrupted/deleted sources.

## Decisions on ChatGPT's eight questions

1. Use explicit contiguous-span alignment first. Add monotonic/coverage loss
   only after exact single-span retrieval works.
2. Give committed scratch a learned, supervised source prior for clean response
   spans, not unconditional authority. When scratch conflicts, align to the
   immutable evidence region instead.
3. Recover the verified fact whenever immutable evidence resolves the conflict.
   Mention the conflict only when useful; never train generic conflict prose in
   place of a known answer.
4. Move two-digit multiplication out of the hard R0 promotion boundary. Keep it
   diagnostic until a reasoning/tool lane exists. R0 must prove reading,
   binding, scratch commit, grounded response, and termination.
5. Train one deterministic canonical foundation response per exact field.
   Evaluate with a small frozen accepted set or semantic predicate. Never put
   multiple incompatible targets on an identical input.
6. The marker renormalization is acceptable. The shortcut to fix is
   character-marginal attention without position labels.
7. A separate single-head copy pointer is the safest first throughput change.
   Length bucketing is second. Do not cache trainable page encodings across
   optimizer steps; their weights and recurrent carried state change.
8. Prove exact one-span retrieval and correct evidence-over-scratch conflict
   behavior in the existing two-read transaction before adding a third tick.

## Repair completed in this audit

`training/build_complete_field_r0_curriculum.py` now:

- uses fresh held-out random strings for exact-copy records;
- makes foundation prompts and targets deterministic and variable-bearing;
- gives abstention cases unique marker identities;
- groups arithmetic repeats by operands so identical questions cannot cross
  splits;
- rejects identical exact fields with contradictory targets;
- rejects identical exact fields crossing train/dev/test;
- records `exact_field_isolation_verified: true` in the manifest.

A clean isolated rebuild produced 8,377 records with train/dev/test counts
7,037/662/678, zero contradictory exact fields, zero exact-field overlap for
all three split pairs, and 1,328/127/145 distinct exact-copy values in
train/dev/test. D00 stayed byte-identical at SHA-256
`f7c12a76550df3ad4cee2b594b33cea7888762383714f6054d92f06fe58e6a38`.
The existing private curriculum and prior checkpoints were not overwritten;
dataset fingerprinting will correctly prevent an unsafe resume from them.

## August 24 firing solution

Do not resume v5 and do not run 5k or 200k. The next bounded change is checkpoint
schema v6 plus a tiny alignment shard and evaluator:

1. define source-span/copy-gate labels and evidence-authority behavior;
2. add a separate single-head position pointer that exposes per-position logits;
3. supervise only unambiguous span characters and the copy/generate gate;
4. rebuild the corrected private curriculum into a new versioned directory;
5. run CPU mechanism tests, then a tiny CUDA smoke;
6. require 100% exact unseen random-token retrieval across page positions and
   correct immutable-evidence recovery under empty/conflicting scratch;
7. require terminated scratch/response and byte-exact interrupted resume;
8. only then authorize a bounded 1,000-step comparison against v5.

The shot is not “train longer.” The shot is to make source position and evidence
authority observable, supervise them directly, and demand held-out binding
before scaling.

## Verification grades

- VERIFIED: Git history/worktree, source mechanisms, gate artifacts, curriculum
  contradictions/leakage, corrected isolated rebuild, D00 hash, focused and full
  tests, remote commit parity, stopped council, and absence of training process.
- ATTEMPTED: the first full-suite invocation timed out at 124 seconds without a
  result; it was rerun with an adequate bound and passed.
- ASSUMED: prior CUDA byte-exact replay claims were not rerun because doing so
  would consume GPU time without addressing the newly isolated supervision
  defect; code and preserved artifacts are consistent with the report.

No external spend, new training, State promotion, checkpoint mutation, live
runtime mutation, or D00 write occurred.
