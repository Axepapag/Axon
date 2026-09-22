# Review: refined pointer proposals and revised execution plan

**Author:** Codex / GPT-6 / 2026-09-22 America/Chicago

**Status:** design review only; candidate remains stopped

**Input:** ChatGPT and GLM proposals supplied by Jeff in conversation

## Decision

The proposals materially improve the earlier marker-first repair. I accept the
central diagnosis: the failed step-25 curriculum entangles an unlearned
language-to-address transformation, an unproved address-selection motor, and
first-cell emission. The next work should begin with causal ablations and then
teach those capabilities separately.

I recommend explicit canonical address geometry in the pointer. That is a real
anatomy change, however, so it must create a new architecture generation. The
preserved step-24 checkpoint and Soul may be governed donors; they must not be
presented as an exact same-architecture optimizer resume.

No optimizer step, curriculum edit, architecture edit, Soul transition, or
cloud launch is authorized by this review.

## Revisions to the supplied proposals

### 1. Use a two-part oracle trace

The clean diagnostic chain is:

1. normal first-cell execution;
2. receipt-certified oracle pointer, retaining the normal copy/generate route
   and output machinery; and
3. receipt-certified oracle pointer plus a forced copy route, retaining the
   normal copied-value/output machinery.

The second comparison asks whether exact source selection is sufficient under
the present route. The third separates pointer failure from route/readout
failure if the mixed copy gate masks the result. These are read-only diagnostic
interventions and cannot count as mastery.

### 2. Do not call an invented address query an ablation

The current architecture has no canonical `address_query(region, position)`
module. Constructing one would introduce the proposed repair while claiming to
diagnose the old model. The clean pre-change tests are the oracle-pointer bypass
above plus measurements of:

- variance and target-conditioned separation through `reader_state`, decoder
  fused state, position query, and pointer logits;
- separability of existing memory keys by canonical region and position; and
- correct-source NLL, top-1, and correct-versus-runner-up margin.

An explicit address query belongs in the new scaffolded architecture after
ratification.

### 3. Treat explicit address geometry as a new architecture generation

The proposal

`key = position_key(memory_state) + address_key(region, position)`

is the strongest repair offered. It makes canonical location available to the
selection motor instead of requiring a fresh D64 core to reconstruct location
from content states. Because architecture identity is content-addressed, adding
this path cannot be slipped into the current candidate or called an exact
step-24 resume.

The address representation must not impose a fixed total content ceiling. Use
an unbounded deterministic position representation, with explicit region
identity and a versioned projection, rather than a learned
`Embedding(max_position)` table. Copy all compatible donor tensors exactly,
initialize and receipt the new address modules explicitly, and create a new
candidate/architecture generation with a governed Soul binding.

### 4. Split skill gates while retaining infrastructure preconditions

`complete_field_coverage_rate = 1.0` remains a universal fail-closed
precondition for every stage. It is not evidence of pointer mastery, but it
must not be deferred or removed.

The pointer-motor gate should require:

- exact source top-1 across the declared generalization surfaces;
- a finite minimum correct-versus-runner-up logit margin;
- finite, noncollapsed query/key/logit diagnostics; and
- address-swap counterfactuals proving that selection follows the requested
  address.

Mean correct-source probability remains calibration evidence, not an exact
`1.0` gate. Free-running transport, public Unicode validity, termination, and
multi-cell coverage belong to later skill gates.

### 5. Refine the zero-step and bounded-step rule

Record a zero-step baseline. One gradient step should improve the directly
optimized correct-source NLL or margin; it need not immediately change top-1
on every case. Require exact top-1 mastery by the bounded 8-step smoke. Permit
extension to at most 16 steps only under a predeclared progress rule. Stop
immediately for collapsed query variance, nonfinite values, missing gradient,
or no movement in the direct objective.

### 6. Freeze and receipt the trainable surface literally

For the motor stage, every non-pointer/non-address parameter must have
`requires_grad=False`, and the optimizer manifest must prove that only the
authorized modules are present. Merely omitting a component from one loss is
not a freeze in a shared model.

### 7. Move the structured-command comparison into training

A zero-shot `#read# cortex 31` comparison is not decisive because the current
core has never learned that syntax. Use structured requests as the next
curriculum stage after the address motor passes. Then the comparison cleanly
measures structured request-to-address grounding rather than unknown prompt
syntax.

### 8. Keep Soul evidence active, but make it developmental-stage appropriate

Every core and tranche should run and persist a causal Soul probe. A
word-level delayed-recall probe can be confounded before the core has learned
language and emission. During pointer stages, use a native substrate/address
recall probe with intact, reset, swapped, and irrelevant-Soul controls. Retain
the word-level probe once vocabulary and language are competent. The probe
must show causal behavioral use, not merely nonzero Soul tensors.

## Revised execution order

### D0 — read-only causal diagnosis

Run the normal/oracle-pointer/oracle-pointer-plus-copy chain on the exact
step-24 and step-25 boundaries. Record the representation pipeline variance,
target-conditioned separation, key geometry, first-cell output, and
stage-matched Soul controls. No optimizer.

### A0 — governed architecture transition

Ratify the explicit address-key/query scaffold, deterministic unbounded address
encoding, donor-tensor rules, optimizer surface, candidate-generation identity,
and Soul transition receipt.

### P0 — canonical address motor

Give the target canonical address directly. Freeze everything outside the
pointer/address motor. Train exact source-index CE across independently varied
address, character, filler, and context axes. Run zero-step evidence and the
bounded 8/16-step rule.

### P1 — address generalization

Grade four separated surfaces: known characters/new contexts, unseen
characters/same addresses, new address combinations, and both new. Every
address class used for graduation must have training support; heldout axes
must diagnose composition rather than accidental omission.

### P2 — structured request to address

Teach a simple canonical request to produce the scaffold address query. Gate
query matching and pointer behavior before fading the direct scaffold. Define
the fade schedule before training so train and serve paths cannot silently
diverge.

### P3 — one-cell emission

Use the normal pointer, copy route, transport, and output machinery to emit one
cell. Evaluate the content step without allowing EOS to hide a failed motor.
This diagnostic exception cannot count as production mastery.

### P4 — Unicode completion and termination

Reintroduce complete public Unicode scalars and EOS. Grade exact first content,
valid Unicode, nonempty output, correct stopping, and universal complete-field
coverage separately.

### P5 — multi-cell transport

Progress through fixed two-, three-, and four-cell work before variable length.

### P6 — scheduled sampling and recovery

Only after position zero and one-cell emission work, fade gold prefixes and add
recovery from self-generated errors. Before that boundary, scheduled sampling
would mostly train on prefixes created by a broken motor.

### P7 — natural-language grounding

Expand the already competent structured request-to-address mapping into natural
language. This stage teaches language interpretation rather than asking it to
co-emerge with addressing.

Every optimizer-bearing stage must use the real Trainer/Heart transaction,
canonical receipts, the stage-appropriate Soul probe, and a local bounded smoke
before any Kaggle run.

## Parent and evidence boundary

- Keep step 25 as failed, immutable evidence.
- Use step 24 only as the exact donor boundary for a new address-aware
  generation if A0 is ratified.
- Do not resume pointer-bootstrap v1.
- Do not use more layers, a larger FFN, additional ticks, Soul reset, or cloud
  compute as a substitute for proving the primitive.

The existing blocking flag remains valid. This review replaces its recommended
marker-first option with D0 followed by the address-aware A0-P7 ladder if Jeff
and the Roundtable ratify that architecture transition.
