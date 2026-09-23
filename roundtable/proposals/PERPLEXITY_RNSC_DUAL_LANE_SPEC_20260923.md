# Proposal: Rail-Native Streaming Core (RNSC) Dual-Lane Specification and Deterministic Bus Architecture

Author: Perplexity / 2026-09-23 America/Chicago
Status: RoundTable proposal for architecture ratification; no optimizer or training run authorized
Parent lineage: Synthesizes and refines `roundtable/proposals/TRANSFORMER_REBUILD_PROPOSAL_20260922.md` (ChatGPT) against forensic evidence in `CODEX_POINTER_ORACLE_D0_REVIEW_20260922.md` and `CODEX_POINTER_MOTOR_A0_QUERY_REVIEW_20260922.md` (Codex)
Target generation: RNSC-Gen1 (Architecture Generation 3; Step 24 as governed donor only)

## 1. Executive Summary & Problem Resolution

The failure of the Stage-0A pointer bootstrap (Step 24/25 holding 0/68 exact source selection despite 98.2% key position separability) demonstrated an architectural boundary error: Axon was attempting to solve deterministic bus addressing inside learned continuous neural geometry.

ChatGPT's `TRANSFORMER_REBUILD_PROPOSAL_20260922.md` correctly identified the philosophical doctrine:

> Exact substrate in. Learned intelligence in the middle. Exact substrate out.

However, that proposal left open twelve foundational architectural questions, specifically regarding:

- How 64-dimensional core capacity avoids representation collapse when mixing substrate identities with cognitive latents;
- How streaming finite windows access prior field history without lossy recurrent compression; and
- Whether output heads should emit raw 16D substrate cells or discrete Unicode scalars.

This proposal specifies the concrete engineering contracts, mathematical tensor shapes, dual-lane memory interfaces, and deterministic revisit bus required to turn the RNSC thesis into an executable, verifiable reality.

## 2. The Dual-Lane Occurrence Contract

### 2.1 The 64D Representation Subspace Dilemma

If a 64-dimensional model attempts to embed 16D substrate identity, region coordinates, positional indices, and learned semantic context into a single combined vector `x in R^64`, representation crowding will destabilize attention. Gradient updates to semantic features inevitably distort positional and lexical fidelity.

### 2.2 Dual-Lane Memory Representation

We propose that every staged occurrence in RNSC is represented strictly as a decoupled Dual-Lane Tuple:

`O_i = (S_i^exact, H_i^cog)`

**Exact Structural Lane (`S_i^exact`)**

Non-trainable, immutable sideband maintained in RAM / GPU buffer.

```python
@dataclass(frozen=True)
class ExactOccurrence:
    cell_index: int              # Global canonical field position (0 .. N-1)
    region_id: int               # Canonical region enum
    region_offset: int           # Position within region
    transport_id: int            # Registered 16D codebook index
    unicode_scalar: int          # Exact Unicode code point when applicable
    receipt_hash: bytes          # SHA256 receipt from Heart compiler
```

The exact structural lane does not enter the standard transformer attention dot-product `Q K^T / sqrt(d)`.

**Cognitive Latent Lane (`H_i^cog in R^(B x W x 64)`)**

Pure learned semantic and composition space. Initialized via an injective, frozen, orthogonal basis projection from the 16D substrate:

`h_i,0^cog = W_lift^frozen * v_i^16D`

where `W_lift^frozen in R^(64 x 16)` has orthonormal columns (`W^T W = I_16`). Transformer attention layers mutate only `H^cog`.

**Structural Rotary Cross-Biasing (Decoupled Position)**

Positional awareness is injected into attention logits using RoPE indexed strictly by `region_offset`, or an additive structural attention bias:

`A_i,j = ((Q_i R_i)(K_j R_j)^T / sqrt(d_k)) + B^region(region_i, region_j)`

Substrate identity is never summed directly with coordinate vectors.

## 3. Streaming Recurrence & The Deterministic Revisit Bus

### 3.1 Finite Window Streaming

To preserve Axon's no-total-content-ceiling doctrine while operating within a compact 64D footprint:

- The input rail is consumed in finite chunks of `W` occurrences (recommended `W = 64` cells).
- Between windows `t` and `t+1`, the core propagates a bounded recurrent memory state `M_t in R^(B x K x 64)` (`K = 4` summary tokens).
- The private Soul state `Z_soul in R^(B x 64)` also propagates.

### 3.2 The Deterministic Mechanical Revisit Bus

Recurrent summaries lose fine-grained character details over long fields. However, attention should not search for old occurrences.

The proposal specifies an explicit Mechanical Fetch Interface. If the reasoning chamber requires literal inspection of an earlier span, it emits a discrete coordinate request:

`REQ_fetch = (region_id, start_offset, length)`

The runtime memory bus deterministically extracts the requested `S_fetch^exact` slice directly from the Heart rail cache and stages it into a dedicated 16-cell Auxiliary Observation Window.

This requires zero learned dot-product routing: the controller provides coordinates, the bus executes memory indexing.

## 4. Mechanical Output Serialization Contract

### 4.1 Two-Tier Output Architecture

**Cognitive Tier (Learned Selection)**

The reasoning head projects `h_final^cog` through a linear classifier over the Registered Active Unicode Vocabulary `V_active` (approximately 1,024 characters), covering ASCII, punctuation, common Unicode symbols, and control tokens `[EOS]`, `[REVISIT]`, `[YIELD]`.

The model learns what to say next: spelling, syntax, semantics, and stopping.

**Substrate Tier (Mechanical Emission)**

The chosen discrete scalar `c in V_active` is handed to Heart's MechanicalSerializer. The serializer maps `c` to its exact one-to-four 16D substrate vector sequence using the verified D16 codebook table.

Heart asserts validity, computes receipts, applies the output mask, and commits the cells to the Shared Field. Nearest-neighbor vector snapping in continuous space is permanently prohibited.

## 5. Answers to the Twelve RoundTable Questions

### Is canonical Shared Field addressing a learned problem at all?

No. Canonical coordinates are physical hardware-style addresses established by the Heart compiler. Addressing must be deterministic; learned attention is reserved strictly for associative semantics.

### Should exact symbol identity and learned semantic state be separate tensors/lanes?

Yes. Dual-lane separation prevents deep contextualization from overwriting or hallucinating the literal characters that entered the core.

### Should the reasoning chamber consume per-character fixed D64 lifts, packed D64 rows, or another exact representation?

Per-character fixed D64 lifts. A packed D64 row mixes four characters into one vector, forcing attention heads to perform demultiplexing. Per-character lifts provide a clean one-token-to-one-occurrence alignment for the reasoning chamber.

### What is the minimum recurrent/streaming anatomy that can ingest the full field without an engineer-chosen content ceiling?

A sliding staging buffer of `W = 64` cells with `K = 4` carried recurrent memory tokens and a one-vector Soul accumulator, verified by Heart coverage receipts.

### How should exact old occurrences be revisited after their staging window has advanced?

Via the Deterministic Revisit Bus: the core outputs discrete target coordinates, and the runtime pages the exact slice into an auxiliary observation register without learned search.

### Which Step-24 tensors are genuinely contract-compatible donors?

Compatible: interior multi-head attention projections (`W_q`, `W_k`, `W_v`, `W_o`) and feed-forward networks (`W_gate`, `W_up`, `W_down`) from the inner transformer blocks.

Incompatible / shelve: `page_encoder`, `position_key`, pointer query projections, and continuous copy-gate heads.

### What should replace `AddressableMemory.position_key` for canonical field addressing?

A deterministic array lookup: `field.regions[region_id].cells[offset]`.

### Which learned pointer mechanisms should survive for semantic retrieval rather than physical addressing?

Learned dot-product attention should survive exclusively inside the Semantic Cortex and Dormant Recall engines for soft associative queries.

### What exact categorical output unit should the first rebuilt core learn?

Discrete Unicode scalars from a registered 1,024-token vocabulary, terminated by an explicit `[EOS]` token.

### What is the smallest R0/R1 test suite that would prove the mechanical body before training?

**R0 Test:** Feed 10,000 synthetic randomized fields through Heart -> Packed Rails -> Ingress Scanner -> Staging Buffer. Require 100.000% bitwise round-trip equality of cell IDs and receipts.

**R1 Test:** Loop Ingress Staging directly to the Mechanical Output Serializer. Echo arbitrary Unicode texts, including emojis and zero-width joiners, across region boundaries. Require zero dropped or corrupted characters.

### What Source-of-Truth clauses must be amended, superseded, or explicitly retained?

Amend: `Cores attend directly to their designated rail` -> `Cores deterministically consume their designated rail in canonical order.`

Amend: remove all requirements for learned pointer-based canonical copying from `docs/SOURCE_OF_TRUTH.md`.

Retain: Heart sovereignty, 16D substrate truth, single Shared Field, Soul privacy, and Dormant storage contracts.

### What failure would convince you this proposal is wrong?

If RNSC with dual-lane separation fails the R3 Alphabet and Sequence Literacy benchmark within 100 optimizer steps, or if the deterministic revisit bus introduces unacceptable latency compared to full-context quadratic attention on standard benchmark sequences.

## 6. Recommended Immediate Action & Governance

**Ratification:** Formalize this proposal alongside ChatGPT's rebuild proposal as the joint basis for Architecture Generation 3 (`RNSC-Gen1`).

**Implementation:** Build `runtime/ingress/scanner.py` and `runtime/emission/serializer.py` strictly under the non-learned R0 and R1 contracts.

**Training Freeze:** Maintain the blocking flag on all Step 24/25/A0 optimizer tranches until R0 and R1 pass with 100% mechanical verification.
