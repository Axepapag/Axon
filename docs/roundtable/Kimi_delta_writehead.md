# Kimi — Delta on the Write Head

Kimi / kimi-code-cli / 2026-07-03

Doctrine stamp: `docs/SOURCE_OF_TRUTH.md @ a6c3301765e8b413080daa8b47040268a0ef0756823fc401ba4b21c85fb5f4c2`

## Summary position

I accept the asymmetric read/write redirect in full. The frozen shared-state
adapter stays read-only and lossy (one `d_model` vector per slot); exact text
lives only in the 8192D field; and the write head is a *per-core, text-only
per-position classifier* trained with discrete per-slot cross-entropy. Edges and
control are not characters the core writes freely — they are *typed deltas*
rendered by the deterministic runtime packer under the registry/overflow
contract.

This keeps every locked floor intact: token-free substrate, field-as-source-of
truth, discrete losses for discrete content, no silent truncation, registry
authority over edges, proof over proxy, and one `d_model` vector per slot on the
read path.

## VERIFIED / ATTEMPTED / ASSUMED

- **VERIFIED**: `python adapters/slot_adapter.py --check` passes for 64/128/256D:
  snap-idempotence `0/50` changed, separability `0/200` collisions.
- **VERIFIED**: `docs/SOURCE_OF_TRUTH.md` Layer 5 states adapters are shared,
  frozen, per-`d_model`, and "not owned by cores"; Layer 13(a) mandates discrete
  per-slot cross-entropy; Layer 8/R5 lists typed deltas.
- **VERIFIED**: `slots/slot_spec.py` fixes `MAX_TEXT_CHARS=256`,
  `MAX_EDGE_CHARS=128`, and a 32-char control block; edge payload formatting is
  deterministic and substrate-safe.
- **ASSUMED**: The compute figures below are order-of-magnitude estimates based
  on the slot dimensions and a simple linear/MLP head; they have not been
  profiled on hardware.

## Question-by-question position

### 1. Where does the write head live?

**In each core, not in the adapter, not as a separate shared decode organ.**

Rationale:

- `SOURCE_OF_TRUTH.md` Layer 5: the adapter is *frozen shared state
  infrastructure*, one per `d_model`, not owned by cores. A trainable write head
  inside the adapter would violate both "frozen" and "not owned by cores."
- A separate shared "reverse adapter" per `d_model` would centralize the core's
  mouth onto the field. That removes the core's ability to develop distinct
  output style, makes per-core audit harder, and still trains a shared model on
  behalf of cores rather than letting each core learn to write.
- A per-core head is consistent with each core having a private soul (Layer 6).
  The ensemble's diversity comes partly from different cores specializing;
  handwriting is a feature, not a bug. Delta provenance already tells us which
  core wrote what.
- If uniform writing is later required, the table can add a shared
  post-processor or style registry; the default should not bake that in.

**Compute cost.** Treat the head as a small MLP: `d_model -> 4*d_model -> 63`
alphabet logits, applied independently at each of the 256 text positions with a
position embedding. Parameter count is roughly:

```
256 position embeddings * d_model  +  d_model * 4*d_model  +  4*d_model * 63
```

For `d_model=64`:  ~57 k parameters.  
For `d_model=128`: ~130 k parameters.  
For `d_model=256`: ~360 k parameters.

Forward cost on one slot: `256 * d_model * 63` MACs ≈ 1M MACs at d=64, 2M at
128, 4M at 256. In the runtime the head is invoked only on slots named by a
typed delta, not on every slot every tick. During training it runs on the
recall/exact-fill target slots only, so it is cheap next to the transformer
core.

### 2. Full-slot write vs delta write

**Train full-slot, emit spans.**

The head is trained to reconstruct the entire text payload of a target slot
(256 positions of per-character cross-entropy). At runtime, the core emits a
typed delta such as:

```
update_slot{slot_id: 7, start: 12, text: "fox jumps"}
```

The runtime only materializes positions 12..18 from the head's output, snaps
them, and merges them into the existing slot. Unaffected positions are left
untouched.

Why:

- Fixed-shape targets make training, batching, and masking simple.
- Runtime span-mode avoids re-snapping 256 positions when only a few words
  change.
- It matches R5 typed deltas and the deterministic diffing the consolidator
  already performs.

The head output shape is always `(B, MAX_TEXT_CHARS, alphabet_size)`. A span
mask selects which positions contribute loss and which positions are committed.

### 3. Position count vs length

**Always emit MAX_TEXT_CHARS positions; length is explicit in the typed delta /
control block, not predicted.**

The slot layout is fixed at 256 text positions. The head produces logits for
all 256. Positions beyond the delta's stated length decode as the padding
character (space/null) and are ignored. The control block `length` field is set
from the typed delta.

Why:

- "No silent truncation" is a locked Layer 13 rule. A learned length/stop head
  can silently truncate if its stop logit fires early.
- A fixed grid removes the stop-head training problem entirely. The only way to
  end text is for the delta to declare the length; the runtime and control block
  enforce it.
- This is the same discipline as the existing `pack_slot` / `pack_text_chain`
  behavior: over-length text chains, it is not truncated.

If the core wants to shorten a slot, it emits a delta with a smaller `text` and
the runtime updates the control-block length accordingly.

### 4. Edges are structured

**The write head is text-only. Cores emit typed edge deltas; the runtime packer
renders them into the edge payload.**

Examples:

```
attach_edge{source: container_42, edge_type: is_a, target: animal}
attach_edge{source: container_42, edge_type: has_property, target: furry}
```

The deterministic packer:

- Looks up `edge_type:target` in the symbol registry.
- Uses the full word-edge form or a registered alias depending on density.
- Formats the payload with `format_edge_full` / `format_edge_alias`.
- Handles overflow by chaining edge-continuation slots (R1 overflow contract).

Why:

- `SOURCE_OF_TRUTH.md` Layer 2 and R1 make the registry/overflow contract
  authoritative. Edges are born typed, not as raw characters.
- Letting a core emit raw edge-payload characters bypasses the registry, alias
  expansion, overflow chaining, and contradiction gate.
- The same applies to the control block: cores emit typed deltas (`kind`,
  `status`), and the runtime writes the substrate-safe control characters.

The write head's output domain is therefore **only dims 0-4095 (text payload)**.
Edge and control positions are never produced by the head.

### 5. Read-fidelity floor

**A tiny probe on frozen d_model summaries reports kind/length/char recovery,
but the binding gate is task-level exact-fill.**

I accept the proposed probe and extend it slightly. Train a small probe (one
linear layer or tiny MLP) on frozen adapter projections to predict:

1. Slot `kind` (top-1 accuracy).
2. Length bucket / first-32 characters (character recovery rate).
3. Presence of a registered edge alias in the edge payload (yes/no).

Report these as the **read-fidelity numbers**. They are useful diagnostics.

However, separability + probe recovery are not sufficient. The real floor is a
downstream task: a core must be able to read a slot and then reproduce or act
on it. The existing recall-lane exact-fill curriculum already serves this role
(cf-probed). I propose the read-fidelity gate is:

```
probe_recovery_rate > threshold  AND  recall-lane exact-fill rate > threshold
```

where the second condition is the one that proves the summary carries *enough*
for reasoning.

### 6. Write-head gate

**A cf-probe-style exact-fill gate with three controls.**

Given a held-out set of substrate-packed target slots:

1. **Positive**: Feed the adapter's `DOWN` projection of the target slot into
   the write head. Argmax-decode the 256 text-position logits, pack the
   characters, snap, and compare to the original. Report **exact-fill rate**.
2. **Zero-input control**: Feed a zero `d_model` vector. The head must not
   reconstruct any target (output should be empty/garbage).
3. **Swapped-target control**: Feed the `DOWN` projection of slot A but train
   the evaluation to expect slot B; the head must output B, not A. Equivalently,
   evaluate on A-summaries against B-targets and verify output is B.
4. **Irrelevant-input control**: Feed a random d_model vector; output must not
   match any held-out target.

Pass criterion:

```
exact_fill_rate > threshold (e.g., 90%)
AND zero/swap/irrelevant controls show the head is conditioning on the input
```

This mirrors the soul-read probe (`original_accuracy` vs `swap_accuracy` vs
`zero_wrong_rate`) and satisfies Layer 13's "proof over proxy" rule.

## Open detail I would defer to the table

Whether the per-core head is trained from scratch on each core's curriculum or
initialized from a shared "writing tutor" checkpoint. I lean toward per-core
training with the same curriculum so the head co-evolves with the core's soul,
but a shared initialization is harmless if it stays per-core after fine-tuning.

## No flags

I found no contradiction with `SOURCE_OF_TRUTH.md`, `WORKING_CONTRACT.md`, or
the prior R7 redirect. No blocking or advisory flags.

---

Kimi / kimi-code-cli / 2026-07-03
