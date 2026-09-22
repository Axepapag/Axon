# Review: D0 oracle-pointer causal diagnosis

**Author:** Codex / GPT-6 / 2026-09-22 America/Chicago

**Status:** completed read-only diagnosis; A0 halted for a narrower table decision

**Candidate:** `english-candidate-1c991f8c911f79394e91`

## Scope

The ratified D0 suite loaded the exact accepted step-24 and step-25 checkpoint
and Soul boundaries on local CUDA. It evaluated all 24 heldout
`pointer_bootstrap_native_v1` FIRST assignments. The FIRST decision is the
cold-start boundary: it has BOS but no teacher-forced target prefix or proposal
workspace.

Three conditions used the same learned decoder state, generated distribution,
ordinary generated-EOS route, copy vocabulary, and output mixture:

1. normal pointer;
2. compiler-receipt oracle pointer with the learned copy/generate route; and
3. compiler-receipt oracle pointer with the content route forced to copy.

The oracle conditions are evaluator interventions only. They did not alter a
checkpoint, Soul, field, trainer, optimizer, or production decoder and do not
count as mastery.

Raw evidence:
`roundtable/reports/CODEX_POINTER_ORACLE_D0_20260922.json`

SHA256:
`9ce7043b86d38b2195a20a8573b81099dc8baaa22f68a3db0c925b2ecd190ce4`

## Causal result

| Boundary | Normal pointer top-1 | Normal first-cell accuracy | Oracle pointer + learned route | Oracle pointer + forced copy |
| --- | ---: | ---: | ---: | ---: |
| Step 24 | 0/24 | 1/24 | 24/24 | 24/24 |
| Step 25 | 0/24 | 1/24 | 24/24 | 24/24 |

At step 24, mean target probability rose from `0.006930` normally to
`0.509376` under the oracle pointer with the unchanged learned route and to
`0.950835` with forced copy. Step 25 produced the same causal recovery:
`0.006778` to `0.508984` and `0.947579`.

EOS did not block the first content cell. It was never top-1 in any of the 48
normal or oracle boundary decisions. Its mean probability was approximately
`0.0492` at step 24 and `0.0524` at step 25.

This proves that correct source selection is sufficient for the existing copy
route and output mixture to emit the correct first transport unit on every D0
heldout field. The immediate cold-start blocker is upstream of copy routing and
transport readout.

## Step-25 movement

The one step-25 update did move the direct pointer objective without solving
selection:

| Metric | Step 24 | Step 25 |
| --- | ---: | ---: |
| Exact-source probability mean | 0.000159 | 0.000314 |
| Exact-source NLL mean | 8.9777 | 8.3876 |
| Correct-vs-best-other logit margin | -7.5687 | -6.8730 |
| Exact-source top-1 | 0/24 | 0/24 |

Gradient reached the pointer and moved in the desired local direction. That
does not rescue v1: the task and gate still conflate address language with the
motor, omit train addresses, and require softmax saturation.

## Geometry result

The supplied proposal expected both sides of the pointer to lack stable address
geometry. D0 found a more specific boundary.

| Representation | Step 24 | Step 25 | Uniform chance |
| --- | ---: | ---: | ---: |
| Cross-episode Cortex-key address classification, 68 positions | 93.995% | 91.605% | 1.471% |
| Requested-address classification from pointer query, 9 positions | 0% | 0% | 11.111% |

The key test classified every Cortex key by comparing it with position
centroids formed only from other episodes. Existing keys therefore contain a
strong and reusable position geometry across changing field contents.

The pointer query was not numerically constant: its mean per-dimension variance
was `2.76e-5` at step 24 and `3.41e-5` at step 25. But that variation did not
separate the requested positions at all. The query changes; it does not change
in the direction required by the requested canonical address.

This narrows the failure to the **request-to-query address bridge**. Adding a
new address term to already address-separable keys is no longer the smallest
evidence-backed repair.

## Soul control

The intact candidate Soul contributed L2 `4.0861` at step 24 and `4.0570` at
step 25 through one decoded layer. In-memory ablation of every Soul temperature
reduced that contribution to zero and changed the pointer distribution by mean
total variation `0.00980` and `0.01256`, respectively.

The effect was not useful for this skill. Intact-minus-ablated mean target
probability was approximately `-2.41e-5` at step 24 and `-2.16e-5` at step 25,
and both conditions remained 0/24 pointer top-1.

This proves that the Soul input path is physically active at these boundaries.
It does not prove delayed recall, content-specific Soul use, or a Soul defect.
The pointer assignment displays the source cell in Cortex and never requires a
private remembered association, so it is not a valid Soul mastery task.

## Revised A0 recommendation

Do not add `address_key(region, position)` to the memory keys first. Preserve
the strong existing key geometry.

The smallest architecture experiment is a query-side canonical-address
scaffold:

1. form a versioned, unbounded deterministic `(region, position)` feature;
2. project that feature into the existing pointer-query/key space;
3. train only the query-side address motor against exact source-index CE while
   every unrelated parameter is literally frozen and omitted from the
   optimizer;
4. prove all declared addresses and heldout content/context axes with exact
   top-1 plus finite margin; and
5. only then train the normal fused decoder state to reproduce the scaffold
   query from a structured request, with a declared fade and regression gate.

This still creates a new architecture generation because it introduces a new
query-side input path. Step 24 remains the governed donor. Existing tensors and
Soul lineage require explicit copy/rebinding receipts. Step 25 remains failed
evidence.

The alternative with no anatomy change is marker-relative training of the
existing query. That is useful as a motor experiment but proves relative marker
selection rather than canonical absolute-address grounding. It is not my first
recommendation after D0.

## State boundary

The accepted candidate pointer remains step 25 / bundle
`ea681656e548c27afb28b5d59418da4313f87d6b0a4402344fa2478b83c3103e`.
Candidate Soul HEAD remains
`13cf886a5a083200816c4487b12edbeb434500993dc8816ce6a374f0abe8f19e`.
No optimizer, checkpoint, Soul transition, Heart transaction, canonical field,
Kaggle job, or persistent Axon process was created.
