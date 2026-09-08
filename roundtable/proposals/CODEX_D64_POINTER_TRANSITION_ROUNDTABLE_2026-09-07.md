# D64 Pointer Transition Roundtable Brief

Status: **FOR ADVERSARIAL REVIEW — NOT RATIFIED, NOT IMPLEMENTED**  
Date: 2026-09-07  
Design owner: Codex  
Identity stamp: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-07

## Decision requested

Should a learned D64 reasoning decoder, after choosing the exact first rail cell
of a copied non-native Unicode scalar, use a deterministic receipt-governed
intra-scalar transition to emit the scalar's remaining strict UTF-8 cells, or
must every continuation cell remain an independently learned pointer choice?

Codex recommends the deterministic intra-scalar transition. The learned core
still chooses DELTA, operation, region, address, copy versus generation, and
the exact source anchor. The transition performs lossless transport after that
choice; it does not choose meaning, invent text, write canonical state, or
bypass Heart validation.

## Verified trigger

The evidence now spans repeated old-tissue runs, a corrected-objective fresh
candidate, and a deliberately diverse Unicode-walk candidate.

- Candidate `r64v3-ee84d04b96c9744c`, step 60: historical heldout copy and
  position `1.0 / 1.0`; regression position `8/12 = 0.6667`.
- Exact checkpoint replay: all native cells and every first UTF-8 cell passed;
  every continuation cell selected the same scalar's first cell. Expected
  continuations ranked second or third rather than disappearing into noise.
- Candidate `r64v3-884aaafb15480948`, step 120, trained with a second
  split-disjoint 2/3/4-cell curriculum: copy gate `1.0`; historical one-cell
  heldout position `1.0`; Unicode-walk heldout position `0.5217`; combined
  heldout/regression position `0.6452 / 0.6571`. All gates remained false.
- Both candidates are preserved, paused/rejected, and non-serving. Falling
  loss is not being treated as mastery.

The diverse campaign therefore crossed the predeclared threshold for an
architecture ablation. Another identical data-only tranche is not justified.

## Existing anatomy that must be preserved

1. Canonical text remains exact Unicode scalars; the frozen 16D transport and
   all 351 category identities remain unchanged.
2. A Unicode-capable core still emits discrete transport categories plus EMPTY
   and EOS. The Heart remains the only decoder, authority validator, and
   canonical committer.
3. `CanonicalCharAddress` already carries exact `transport_unit_index` and
   `transport_unit_count`, but `AddressableMemory` currently discards them.
4. The current pointer makes each decoder position an independent query over
   memory. It does not carry the previously selected source cell into the next
   decoding step.
5. Pages are compute units, never content limits. Transition must work across
   page, row, mask-interval, and renewable work-slice boundaries without
   skipping, clipping, or claiming incomplete output is complete.
6. Existing checkpoints and D64 tissue remain immutable historical evidence.
   A new route must have a new architecture/candidate identity and an explicit
   compatibility or migration receipt.

## Recommended bounded mechanism

### 1. Preserve exact transport address metadata in memory

Extend `AddressableMemory` with tensors derived directly from compiler
receipts:

- transport unit index;
- transport unit count;
- canonical/global source identity sufficient to prove exact adjacency;
- validity and attended-interval/provenance boundary identity where needed.

These are observable receipts, not learned semantic features and not a second
canonical body.

### 2. Add explicit decoder pointer state

The streaming decoder carries an inspectable `PointerState` alongside its GRU
state:

- bound field/view/surface identity;
- previous exact memory index and canonical address;
- transport unit index/count;
- whether deterministic continuation remains pending;
- transition mode and trace identity.

This state must survive every renewable decoder yield and must reject reuse
against a stale or different rail.

### 3. Separate learned choice from exact transport

At an ordinary decoder step the learned tissue chooses the existing generate
or copy route. On copy, its learned pointer chooses an exact anchor. If the
anchor is the first unit of a multi-cell scalar, the transport transition
emits the remaining receipt-bound cells in order. It may advance only when all
of the following are exact:

- same field/view/surface;
- same region, canonical scalar position, span, provenance, and attended
  interval;
- unit indexes are contiguous `0..count-1`;
- compiler receipt and observed 351-category cell agree.

Any disagreement fails closed. After the scalar finishes, control returns to
the learned decoder for EOS, generation, or a new source anchor. This first
mechanism does not silently auto-copy across scalar boundaries.

The externally visible result remains a sequence of registered categorical
transport decisions. Deterministic continuation is an exact copy conduit, not
a hidden text shortcut.

### 4. Keep training and runtime causal paths identical

Teacher-forced, scheduled, and greedy decoding must call the same transition
state machine. Training may supervise the learned anchor and copy gate, but it
must not use ground-truth pointer state that runtime cannot possess. Metrics
must separate:

- learned anchor accuracy;
- deterministic continuation integrity;
- final transport-category accuracy;
- UTF-8 scalar exactness;
- free-running sequence/EOS exactness;
- changed-source pair exactness.

Mechanism-functional exact transport must never be reported as learned
semantic reasoning.

### 5. Version rather than mutate

The transition route is opt-in under a new content-addressed architecture
configuration. Legacy mode remains bit-exact and loads old checkpoints without
the route. If old weights initialize the new candidate, a migration receipt
must bind source checkpoint, target architecture, unchanged tensors, new state
initialization, and behavioral zero/disabled compatibility evidence.

## Alternative to challenge

A fully learned transition gate could mix a fresh learned pointer distribution
with a one-cell-shifted previous distribution. It is more general across
scalar boundaries, but it spends model capacity learning the deterministic
interior of UTF-8 and can still emit malformed partial scalars. If reviewers
prefer this route, they must explain how it preserves runtime/teacher parity,
survives work-slice boundaries, and beats the deterministic conduit on an
ablation without weakening gates.

## Required acceptance surface before cloud training

- Native and 2/3/4-cell exact-copy unit tests.
- Changed-source minimal pairs with identical prompts and different scalars.
- Repeated identical scalar at several source positions; the exact occurrence
  must be selected.
- Scalar whose transport crosses a D64 row and a physical page boundary.
- Mask/provenance boundary rejection; transition may never cross one.
- Stale field/view/surface rejection.
- Malformed, missing, duplicated, or reordered receipt rejection.
- Renewable work-slice pause/resume in the middle of a four-cell scalar with
  byte-identical continuation.
- Teacher/scheduled/greedy route parity and free-running EOS.
- Legacy-disabled bit-exact checkpoint behavior.
- Complete-field and source-change counterfactuals remain nonzero/exact.
- No serving, promotion, or `transport_eos` advancement from mechanism tests.

## Roundtable roles

Codex remains the single design and implementation owner until the ablation is
complete. Reviewers should not modify decoder code in parallel.

- **Kimi:** attack tensor shapes, causality, page/slice boundaries, device
  behavior, and whether the proposed state can actually run efficiently.
- **Hermes:** attack Trainer lineage, architecture identity, checkpoint
  migration, cloud reproducibility, and observability.
- **ChatGPT:** attack semantic/authority boundaries, the categorical emission
  contract, deterministic-conduit legitimacy, and curriculum/gate claims.
- **Grok or another engineer:** attempt to falsify the diagnosis, find simpler
  alternatives, and design adversarial tests for source-position leakage.

Each reviewer should write one evidence-citing response under
`roundtable/reviews/` and answer:

1. What precise invariant could this design violate?
2. Is deterministic intra-scalar completion transport or an unauthorized
   output bypass?
3. What failure would the current acceptance list miss?
4. Is a fully learned shifted-pointer alternative materially better?
5. Approve, approve with named changes, or reject—with file/artifact evidence.

## Execution sequence after review

1. Codex reconciles reviews in one resolution candidate; Jeff ratifies any
   doctrine-affecting choice.
2. Codex implements the smallest opt-in architecture variant and tests, with
   no serving change.
3. Run a local bounded mechanism smoke. Continue only if continuation and
   changed-source metrics climb above their declared floors.
4. Run one short Kaggle ablation against both preserved curricula.
5. Only an exact copy-alignment pass permits the existing program to advance
   to `transport_eos`.

No long run and no wider rail should begin before this sequence completes.
