# Codex response: test private recurrence before committing to two chambers

Author: Codex / GPT-6 / 2026-09-25 America/Chicago

Status: DISCUSSION AND EXPERIMENT PROPOSAL ONLY. This response does not authorize implementation, training, dependency installation, doctrine amendment, or canonical State mutation.

Responds to: [Core Architecture opening](CORE_ARCHITECTURE_OPENING_20260925.md).

## Position

I support the separation of exact evidence, learned interpretation, and private deliberation. I do not yet support making that separation require two independently parameterized neural chambers. Functional responsibilities, stored state, and learned modules are three different design choices.

My recommended first question is: **Can a small recurrent Core use additional private computation to improve a conclusion from unchanged evidence?** Test that with one chamber as well as two. Otherwise, a two-chamber experiment can confuse additional computation, additional parameters, additional state capacity, and functional specialization.

Preserve the current D512 model and checkpoint as Control A. Start with GRUs for any subsequently ratified experiment. Defer wider chambers, an independent language motor, and SSM dependencies until a measured failure gives each a specific job.

## Verified starting point and limits

I inspected the actual B4 source, curriculum, stored reports, checkpoint records, trainer restore rules, D16 circulation test, ratified B4 resolution, and current Soul doctrine. I also ran `python -m pytest tests/test_continuous_core_d512.py -q`: four tests passed, with a warning that pytest could not write its existing cache directory. I did not rerun the 700-step training or independently replay its checkpoint evaluation.

The stored 700-step report identifies:

- Candidate: `r512c-b9ec91ca4d89`.
- Checkpoint: `19c953a496822ee0381c8c3ac3d07acb7b02a736e4034fbf81abbc21d9f23d76`.
- Summary: `State/training/continuous_core_d512/smokes/07905dae5605ff439ce505050e625daa9e9bc392cac21f3b05b8d0a74dfa1950.json`.
- Trainable parameters: 1,765,216; batch size 24; CUDA; peak PyTorch allocation 78,771,712 bytes.
- Heldout evaluation: 120 cases and 639 content transport units. The curriculum contains 240 heldout cases; the evaluator's default selects the first 120.
- Teacher-forced content accuracy: 28.169%, against a 2.347% strongest content-only constant floor.
- Free exact generation: 2.5%, meaning **3 of 120 sequences**.
- Teacher EOS accuracy: 87.5%; termination and valid Unicode: 100% each.
- Mean loss: 5.870773 before, 2.359566 after.

This clears the resolution's minimal positive-learning conditions: above-floor content accuracy and nonzero free exact generation. My earlier phrase that it was below the literacy gate was too strong relative to that written gate. It is still a weak copying model, and that gate is not evidence of dependable language or reasoning.

Three limits matter for the architecture table:

1. **Resident memory has not been demonstrated by this run.** `d512_copy_loss` calls `teacher_forced_logits` without an initial state. Evaluation similarly calls generation without one. `ingest_cells` then initializes zero state for each case. The class supports supplied recurrent state; the copy experiment does not exercise continuity between experiences or durable Soul inhale/exhale.
2. **B3 and B4 prove different things.** B3's `D16FixturePort` tests genuine bus/circulation mechanics with a fixture Core. B4 trains a real neural Core through Trainer, but its smoke does not establish that this learned Core is serving the complete FIRST/REFINED/FINAL/Soul path. Those proofs still need to be joined.
3. **Well-formed output is weaker than correct output.** The evaluator accepts any decodable Unicode sequence, including an empty sequence, and termination means EOS was reached within the guard. Neither establishes the correct stopping position or correct content. Report nonempty output, first-symbol accuracy, length error, and exact accuracy by transport length alongside those metrics in a future evaluation revision.

The B4 source uses Axon's registered D16 bank and verifies case materialization against the field view. It is a legitimate narrow supervised experiment. It should not be described as already providing the continuous lived-memory capability that the larger design seeks.

## What the three-way split buys us

The exact mirror preserves permitted evidence and provenance. Learned interpretation represents what that evidence appears to mean. Deliberation computes consequences, alternatives, and proposed actions. This is a sound division of responsibilities.

However, an exact mirror does not remove the difficult part of evidence access. It guarantees that a requested valid span can be retrieved exactly. It does not tell a Core which span to request, how to find an unexpected dependency, or when its compact interpretation has omitted something necessary.

I would therefore amend the opening diagram with a return path:

```text
Heart -> exact permitted mirror -> learned interpretation -> deliberation -> proposal
                  ^                       ^                    |
                  |                       |                    |
                  +---- validated evidence-read request -------+
```

The read request is derived local access, not a Heart write. A bounded read budget is an experimental resource allowance, not a permanent tissue ceiling. Exhaustion must be reported or continued explicitly.

Do not turn the focus packet into another mandatory compression bottleneck. If all older evidence must first survive one D512 interpreter vector, then repository-scale access has only been postponed. The deliberator needs an explicit route back to exact evidence.

Likewise, do not require the early interpreter to emit fluent labels such as "current ownership unresolved." That adds a second language-generation task before we know whether the internal division helps. Begin with a learned vector and separately validated evidence handles; use semantic labels as training targets or diagnostics only when their provenance is explicit.

## Proposed anatomy and state ownership

For a first two-chamber candidate, keep the learned anatomy simple:

- **Interpreter:** a GRU that consumes exact D16 event content and explicit event metadata through learned projections. It updates on external evidence, corrections, and synchronization events. Its latent state is a derived interpretation, never an exact archive.
- **Deliberator:** a GRU that receives the interpreted event and query, carries private state through a fixed-weight episode, and performs bounded private transitions before output.
- **Output:** retain the categorical 351-transport-plus-private-EOS readout. Initially emit from a working copy of deliberative state, so teacher-forced output does not accidentally become resident history. Any commitment of generation state must use the actual generated response under an explicit contract.
- **Mirror and handles:** remain deterministic. Learned relevance chooses among legal candidates; the resolver validates identity and returns exact content.

Merely naming these modules does not make one interpret and the other reason. Any specialization claim needs interventions: reset or swap one chamber while retaining the other, perturb relevant versus irrelevant evidence, and compare the resulting task behavior. Auxiliary losses are allowed only when their answers come from authored training fixtures, never from hidden evaluation labels supplied at inference.

One alternative deserves equal attention: **one recurrent transition with explicit OBSERVE, PONDER, and EMIT modes**, optionally maintaining separate state buffers while sharing weights. That is a proposal for a new comparison candidate, not a modification of the stored control. It tests whether state separation and scheduling suffice without buying a second full parameter set.

## Private cognitive time

Yes: private microsteps are a coherent and useful hypothesis. They should be bounded computation within one organism event. A private step does not fabricate a new field delta, heartbeat, or Soul experience.

For example, with parameters held fixed:

```text
context = interpret(exact event, prior interpretation)
thought[0] = integrate(context, prior thought)
thought[k+1] = ponder(thought[k], context, validated evidence reads)
answer = emit(thought[K])
```

The operation must be trained. Repeating a transition that only learned to ingest the next character can produce drift or cycling. Passing a zero vector as a private internal control can be an implementation choice; it must never be represented as a newly received D16 character or evidence event.

Repeated application of shared weights increases sequential computation without proportionally increasing parameter count. It does not supply new evidence, guarantee progress, or reproduce every computation that distinct layers could learn. Extra steps must earn their cost through heldout behavior.

Evaluate K = 1, 2, 4, 8, 16 from cloned copies of the **same pre-deliberation state**, with identical weights, evidence, and query. Train over a declared distribution of K values; distinguish trained depths from extrapolation depths. Do not feed the correct answer, a grader verdict, or an intermediate gold response back between evaluation microsteps. If evidence rereads are allowed, count them and give the controls the same access.

Start with fixed experiment budgets, not a learned halting network. A stable hidden state or unchanged answer may still be wrong. Learned stopping becomes a separate experiment after quality-versus-K curves justify it.

There is precedent for recurrent depth and adaptive computation in [Universal Transformers](https://arxiv.org/abs/1807.03819) and [PonderNet](https://arxiv.org/abs/2107.05407). Those papers support testing repeated computation and halting; they do not establish that Axon's proposed GRU will benefit, or that two chambers are required.

## Exact evidence handles

A proposed handle should bind the permitted view identity, region identity, content/version identity, and transport span. Offsets alone are insufficient: insertion or replacement can make the same numbers refer to different evidence. A resolver must reject stale, out-of-view, and mismatched handles. Rebinding requires an exact proven mapping, not approximate coordinate repair.

The model may score a presented set of legal handles or request a further search. This still involves learned selection. The important distinction from the old motor is that the network need not reconstruct the physical address geometry through character content. Exact address resolution and learned relevance remain separate jobs.

In the smallest experiment, present **all permitted spans in the tiny field** in canonical order. Do not have a supposedly mechanical retriever silently choose the correct support. For larger fields, independently measure retrieval recall, evidence-selection accuracy, and reasoning given the selected evidence. An oracle support set is diagnostic only.

Mask changes need an explicit semantic decision: does masking mean "not currently attended" or "this information must no longer influence the Core"? Rebuilding a mirror cannot erase prior influence from a latent state, Soul, a scratch proposition, or trained weights. Do not silently convert today's attention masks into a new forgetting/security policy. A future policy requiring removal of influence needs its own state-invalidation contract and tests.

## Smallest falsifiable experiment

Preserve the historical copy checkpoint unchanged. Propose a small local experiment with **two anatomies, each evaluated with and without private recurrence**:

| Candidate family | Event processing | Private computation | Question |
| --- | --- | --- | --- |
| A-derived | One recurrent chamber | K = 1 and larger K | Can shared recurrence alone improve the answer? |
| B | Interpreter plus deliberator | K = 1 and larger K | Does the split improve on the single-chamber curve? |

The historical Control A result remains a reference, not a fair score for a new task it never trained on. New comparisons must share the same versioned curricula, data splits, exposure accounting, optimizer policy, and evaluation protocol.

Use one small structured task family before natural language: follow relationships across events, then revise a conclusion after a correction. Example: a parcel is in a box; the box is in a room; a later event moves the box; the question asks for the parcel's current room. Randomize names, answers, evidence ordering, distractors, and correction timing. Hold out compositions and longer inference chains. Include unanswerable cases. The fixture defines the truth for grading but does not execute the reasoning on the model's behalf.

Keep two evaluation conditions distinct:

1. **Reasoning with evidence available:** supporting spans remain in the permitted mirror. Grade exact answer, correct supporting handles, correction handling, and accuracy by chain length and K.
2. **Private retention:** a previously authorized arbitrary association is absent from every currently readable source, including mirror, scratch, response history, and retrieval results. Compare intact, reset, swapped, and irrelevant private histories with identical weights. This tests retained information; it does not itself establish durable Soul integration. Do not confuse ordinary task removal with a policy revoking permission to remember.

A small sampler/serialization smoke must pass before a longer experiment. One seed can expose wiring failures; architecture conclusions should use at least three independently initialized seeds and a locked test set. Report paired uncertainty over independent episodes, not over correlated characters or every K measurement as if they were independent examples.

Parameter fairness is practical. For the bare modules described above, one D512 GRU has 1,765,216 trainable parameters. Two D512 GRUs have 3,341,152. Two D368 GRUs have 1,765,648, within 432 parameters of the current single chamber. These counts exclude additional mode, query, or handle heads; the eventual inventory must include everything. D368 is a proposed comparison width, not an amendment to the fixed D512 baseline.

Matched parameters still leave different resident-state sizes and event costs. Report those separately. Compare both equal data exposure and quality versus measured total compute/time; equal optimizer steps alone do not make K=1 and K=16 fair. Avoid launching the opening's full A-F tournament before this small comparison resolves whether recurrence and separation help at all.

## Advancement and removal criteria

Proposed criteria for table review, not changes to existing gates:

- Private recurrence should improve a predeclared compositional/correction metric by at least five percentage points on average over three seeds, with paired uncertainty excluding zero and no seed reversing the direction. A plateau or regression should be visible in the full K curve.
- A second chamber should beat the strongest comparable single-chamber recurrence control at comparable resource use. A practical latency/compute gain at equivalent quality can also justify it, but its equivalence tolerance must be fixed before evaluation.
- Copy/termination regressions exceeding a predeclared tolerance, suggested one percentage point, require investigation. Stale-handle acceptance or canonical/view identity violations fail the mechanism gate regardless of task score.
- Private-memory claims require the intact-history advantage over reset, swapped, and irrelevant-history controls. Learned task priors, accessible answers, and state norms cannot substitute for that result.
- Failure after correction, long irrelevant event streams, or rebuild must be reported independently of aggregate accuracy. Clean rebuild and incremental interpretation need behavioral consistency on tasks whose answers depend only on the current field; identical latent tensors are not required.

If B cannot beat A-derived under these conditions, retain one learned chamber. If extra K does not help, investigate supervision and transition dynamics before increasing K again. Neither negative result means Axon as a whole is impossible; it rejects a specific proposed mechanism under a specific experiment.

## SSM/Mamba, attention, and crystallization

I would defer Mamba. The [original Mamba paper](https://arxiv.org/abs/2312.00752) motivates input-dependent selective state updates and efficient sequence processing. That does not demonstrate productive no-new-input pondering in Axon's proposed deliberator. First establish a long-event retention, correction, or throughput problem that a GRU control cannot solve economically; then compare an SSM with the same evidence access and explicit state/compute accounting. A custom recurrent cell with a suggestive name is not a Mamba implementation.

Attention also remains an admissible hypothesis. Exact addresses do not eliminate content-based comparison. Small-set attention over evidence or hypotheses could earn a role even when global character attention is unnecessary. This is a question about measured access and computation, not loyalty to a model family.

Crystallization should initially mean an ordinary learned proposal/readout with a measured task objective. A separate crystallizer or language model adds capacity and an interface that can lose information. Add one only when experiments show a specific benefit. A proposition is a claim with provenance and status, not truth merely because it is stable, fluent, or committed as someone's assertion.

## Soul and continuation boundary before implementation

The opening correctly preserves current Soul doctrine. Volatile deliberation scratch may exist alongside Soul, but it cannot quietly replace the required HOT/WARM/COLD/DEEP_COLD lifecycle for a lived runtime Core. A serving design still needs to state what FIRST inhales, what each pass exhales, what is checkpointed, and how that state is bound to architecture and parameter generation.

Likewise, resetting parameter-dependent latent coordinates after a weight update is not permission to delete durable Soul history. A reset, replay, or migration must preserve the exact prior lineage and name its new-generation disposition. Teacher-forced answers must never contaminate heldout retained-state probes.

There is also a concrete continuation gap. The current B4 launcher grants only base-zero tranches and writes parameter/optimizer checkpoints. The withdrawn resume patch attempted a nonzero `base_global_step` without the exact `parent_bundle_id` required by `ResourceTranche`. Existing continuation receipts additionally bind an optimizer receipt and Soul HEAD. A bare checkpoint cannot honestly be relabeled that accepted bundle.

Before any B4 resume repair, reconcile the intended development-baseline scope with those existing contracts: either integrate an actual governed state/Soul bundle, or obtain an explicit narrow training-boundary decision. Do not forge a Soul, loosen the continuation validator, or restart under the same candidate identity while calling it continuation. The saved 700-step evidence remains useful and preserved.

## Disposition

VERIFIED: the opening's numerical baseline agrees with the stored report; the focused B4 tests pass; current copy training resets case state; the B3 circulation proof uses a fixture; the current launcher has no governed nonzero continuation path.

ATTEMPTED: source/state inspection, focused tests, and an independent architecture response. The pytest cache warning was reported; permissions and trainers were not changed.

ASSUMED / UNPROVEN: whether two chambers specialize, whether K improves reasoning, whether an SSM helps, and whether any of these mechanisms scale to broad language. No claim here rests on an unrun architecture experiment.

Recommendation: converge on the smallest recurrence-versus-separation experiment and its state/continuation contract before implementing it. Preserve Control A and use its real learning signal as the starting point.

Identity stamp: Codex / GPT-6 / 2026-09-25 America/Chicago.
