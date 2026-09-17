# Soul Completion + Dormant-State Trainer Roundtable Brief

Status: **PROPOSAL ONLY — FOR ADVERSARIAL REVIEW — NOT RATIFIED, NOT IMPLEMENTED**  
Date: 2026-09-16  
Review status (2026-09-17): **two blocking corrections recorded — see §Reviewer
corrections below.** The phase plan is not yet rewritten; read that section
before treating anything below as an executable plan. Independent reviews:
`evt-20260917T015800Z` (Kimi) and `evt-20260917T032500Z` (GitHub Copilot CLI),
the latter filed in full at
`roundtable/reviews/COPILOT_SOUL_COMPLETION_DORMANT_TRAINER_REVIEW_2026-09-17.md`.
Design owner: ChatGPT / GPT-5.6 Sol  
Scope: reasoning-core Soul completion, Trainer curriculum source, and the boundary between Dormant State, Soul, parameters, D64 rails, and Heart authority

## Decision requested

Should Axon pause wider reasoning-core curriculum expansion long enough to make private Soul a proven load-bearing memory system, then point the Trainer at the governed Dormant State as the primary lifelong curriculum source while preserving the pristine canonical/D64/Heart boundary?

Proposed answer for review: **yes, with strict separation of authorities and explicit causal tests.**

The intended developmental loop is:

`Dormant/Canonical evidence -> governed D64 lesson -> core inhales private Soul -> core attends rail -> core acts/generates -> core exhales Soul -> outcome/evidence -> Trainer updates candidate parameters -> later recall uses the persisted Soul and/or governed Dormant retrieval`

This proposal does **not** authorize dumping the whole Dormant corpus into one context window, writing Dormant bytes directly into Soul, bypassing Heart, or treating every recovered statement as a target. It proposes piecewise, provenance-governed, runtime-faithful exposure over renewable optimizer steps and checkpoints.

## Reviewer corrections (2026-09-17)

Recorded by GitHub Copilot CLI / `deepseek-v4.1-flash:cloud`, event
`evt-20260917T032500Z-copilot-soul-proposal-review`, full text in
`roundtable/reviews/COPILOT_SOUL_COMPLETION_DORMANT_TRAINER_REVIEW_2026-09-17.md`.
Verdict: **approve the direction, reject the sequencing — approve with named
changes.** The doctrine fences in this proposal are correct and should be
ratified as written. Two corrections are blocking.

### BLOCKING 1 — the premise predates the root cause, and the phase order depends on it

This proposal was written after two audits
(`evt-20260917T013100Z`, `evt-20260917T013451Z`) that located the content
plateau at the **objective level**, not in missing memory. The final
`transport_eos` probation report
(`segment_000000041_000000048_probation.json`) shows every QA transcript with
`predicted_payload: ''` and `terminated: True`, and
`payload_transport_exact_rate = 0.3333` — exactly the 12/36 empty-payload cases.
The stop head's fixed point is `stop ~ 1`: content positions push it down at 1/5
of the payload weight, the EOS position pushes it up at 4/5, plus a live
`alignment_eos_gate` BCE at weight 1.0.

Therefore **S0/S1 as ordered below cannot succeed for a reason unrelated to
Soul**: the acceptance statement ("later retrieve or apply that information when
the original evidence is absent") and the S1 gate both require free-running
emission that the current objective actively suppresses. The corrected-v5 motor
shot must precede or accompany S0.

### BLOCKING 2 — "under corrected wiring ... genuine exactness" is false for termination

The "Why this discussion now" paragraph below asserts that copy alignment,
position, copy gating, and termination have reached genuine exactness. Verified
from the accepted step-24 report
(`segment_000000017_000000024.json`):

```
foundation_motor_v2_training_stage: copy_alignment
foundation_motor_v2_stage_gate.passed: true
task_gate_passed: false
evaluation_component_weights: {alignment_copy_gate: 4.0, alignment_position: 1.0,
                               alignment_eos_gate: 0.0, payload: 1.0, ...}
payload_eos_weight: 4.0
```

`alignment_eos_gate` was **0.0**. The termination gate this programme exists to
train has never received gradient in an accepted checkpoint; the only stage where
it ran at 1.0 is the probation branch discarded by design. `copy_gate 4.0` and
`position 1.0` are genuine; `transport_exact 0.333` is the empty-payload floor.
No milestone may be built on "termination is proven."

### Named changes (non-blocking but required for ratification)

1. **Use the instrument that already exists.** The tournament computes
   `relevant_soul_ablation_degradation`, `irrelevant_soul_ablation_delta`,
   `stale_soul_degradation`, `swapped_soul_rejection_rate`
   (`training/reasoning_tournament.py:61-62, 434-438, 703-706`); no gate consumes
   them, and `tests/test_tournament_metrics.py:243-244` asserts degradation 0.0
   as the honest value. §"Load-bearing HOT memory" should adopt this behavioral
   exact-rate differential as its criterion — "the answer must change under
   ablation" would pass on noise.
2. **Mask the source field in every write probe.** `encode` detaches
   (`training/living_reasoning_d64.py:247`) and `decode` rebuilds via
   `frombuffer(...).clone()` (`:269-288`), so cross-phase Soul credit assignment
   is identically zero. If the probe `ZORP?` is asked while the field still holds
   `ZORP = BLUE`, the model answers from field attention and the write is never
   pressured.
3. **Predeclare saturation thresholds** (K = 1, 4, 8, 16) before the first run so
   the architecture-migration question cannot be deferred by training longer.
4. **Split Soul from Dormant.** Soul is deep and small; the Dormant curriculum is
   broad and data-ambiguous. Bundling them into one pivot is the largest
   structural risk here.
5. **Fix Soul identity before S1.** Soul is parameter-generation-bound and
   `inhale` hard-fails on generation mismatch, so every accepted parameter update
   invalidates every Soul — the contract currently forbids the lifelong premise.
   Validate on `soul_codec_version`; make a generation change a migration with a
   receipt.
6. **D0 first family:** container `edges` with `edge_type: "is a"` and short
   targets (`tool`, `file`), keyed by `letters`/`entity_type`. Not
   `semantic_edges`: measured over all 351,978 rows it carries **41,142 distinct
   `edge_type` strings, 20,684 of them singletons**, with free-text targets
   averaging 33 characters.
7. **Identity ruling before D1.** `experience_v1` holds 59,925 records, of which
   **59,875 are the single import `d00-recovered-autobiography-v1`**
   (`evidence_class: recovered_lived_evidence`, D2-era March 2026); live
   Heart-owned ingress is ~50 records. Container `created_tick` is `-1` across
   the corpus, so delay/age buckets must use `experience_v1.occurred_at`.
   Training a current core's private Soul on another life risks manufactured
   memory.

## Why this discussion now

The current D64 program has successfully proven substantial motor/control tissue. Copy alignment, position, copy gating, and termination have all been exercised; under corrected wiring, the accepted copy-alignment lineage has reached genuine exactness on the copy and position gates. (**Correction, 2026-09-17: not on the termination gate — it evaluated with `alignment_eos_gate` weight 0.0 and the only stage where it ran at 1.0 is the discarded probation branch; see §Reviewer corrections, BLOCKING 2.**) However, the next-stage campaign remained flat on content generation while already-mastered copy machinery continued to dominate training pressure.

That result suggests the next engineering question is not merely "more steps." It is whether the core has the persistent internal substrate needed to learn from a long life rather than only from each isolated field presentation.

Axon already has the architectural pieces for that answer:

- exact canonical/Dormant authority;
- a frozen 16D character substrate and D64 packed-rail interface;
- Heart as the sole canonical committer;
- private per-core Soul snapshots with `HOT`, `WARM`, `COLD`, and `DEEP_COLD` layers;
- Soul lineage, transition, promotion, persistence, restart, and integrity contracts;
- Trainer parameter lineage and candidate-generation governance;
- a large exact Dormant corpus with semantic and autobiographical evidence.

The missing part is not the existence of Soul metadata. The missing part is proving that Soul is **causally useful memory** and giving the core a learned way to write, retain, compress, retrieve, correct, and eventually consolidate experience through those layers.

## Existing doctrine that should remain unchanged

This proposal is intended to extend, not replace, the current Source of Truth.

### Canonical/Dormant authority

`docs/SOURCE_OF_TRUTH.md` defines Dormant State as exact state outside current attention plus exact provenance stores and structured knowledge derived from exact evidence. Dormant material is not attended directly; verified material is surfaced through governed canonical regions and Heart authority.

Dormant State remains the exact durable library/autobiography. It is not replaced by weights or Soul.

### Shared Field remains the input interface

The Source of Truth explicitly states that input does not enter Soul first. The Shared Field is the input interface. That should remain true.

A reasoning step therefore remains conceptually:

`inhale Soul -> attend frozen Shared Field / D64 rail -> compute -> propose -> exhale Soul`

not:

`inject external facts directly into Soul -> bypass Shared Field`

### Soul remains private

Each core owns one architecture-bound, parameter-generation-bound private Soul. Souls do not go on proposal boards and are never merged across siblings. Brother cores communicate through governed proposals, not latent-state exchange.

### Soul layers remain HOT / WARM / COLD / DEEP_COLD

Current doctrine already says:

- `HOT` may change on every causal inhale-think-exhale boundary;
- colder layers change only through adjacent evidence-vetted promotion;
- `DEEP_COLD` promotion requires additional validation evidence;
- `DEEP_COLD` is the only Soul layer eligible as a future LoRA/adapter distillation source;
- distillation creates a new governed parameter generation and never deletes exact evidence.

### Heart and Trainer sovereignty remain separate

Heart owns canonical Shared Field mutation. Trainer owns parameter mutation. Soul transition/persistence must stay exact and lineage-bound, but neither Soul nor Trainer may gain hidden canonical-write authority.

## Current Soul anatomy: what is finished and what is not

### Already present

`runtime/soul/contracts.py` and `runtime/soul/store.py` provide the important durable mechanics:

- immutable content-addressed Soul snapshots;
- exact `core_id`, `architecture_id`, and `parameter_generation` binding;
- four ordered temperature layers;
- transition identities and lineage;
- prepared/committed Soul records;
- restart-safe HEAD publication and recovery;
- adjacent-layer promotion receipts;
- stronger evidence requirements for `COLD` and `DEEP_COLD`;
- stale-promotion rejection;
- runtime requirement that each reasoning inhale/exhale returns a HOT successor.

`training/living_reasoning_d64.py` already implements architecture-local opaque Soul tensors. The current codec serializes an exact `[state_tokens, d_model]` float tensor, and `inhale()` projects each available Soul layer into the core's recurrent state through learned per-temperature projections and learned gates.

The living reasoning path also correctly rejects another core's private Soul and rejects incompatible architecture or parameter generations.

### Still incomplete

The current neural reasoning path primarily exerts Soul as a recurrent state carrier, not as a finished memory system.

Most importantly, normal `LivingReasoningCoreD64.exhale_transition()` currently emits only a new `HOT` layer. The contracts allow `WARM`, `COLD`, and `DEEP_COLD` updates through promotion receipts, but the core does not yet have a learned consolidation policy that decides what to compress, how to encode it, when it deserves colder retention, or how to prove later that the colder representation preserved useful information.

The present service-faithful phase loop also intentionally serializes persisted Soul bytes between phases, creating a truncated-gradient boundary that matches runtime. That is correct runtime behavior, but it means delayed-recall learning cannot depend on backpropagating through an arbitrarily long historical computation graph. Memory writing therefore needs an explicit local training objective, and delayed recall must be tested as a separate persisted-state consequence.

## What "finish the Soul" should mean

Soul should not be declared complete merely because snapshots serialize and reload.

A finished first-form Soul should satisfy all of the following.

### 1. Load-bearing HOT memory

The core must be able to ingest a novel fact or correction from the Shared Field, exhale a changed HOT Soul, persist/restart, and later perform better because that exact Soul is inhaled.

Required negative controls:

- pre-experience Soul;
- zero/empty HOT;
- stale HOT;
- ablated HOT;
- wrong-core Soul, which must fail closed rather than silently condition the model.

If the answer is unchanged across those probes, Soul is not causally load-bearing.

### 2. Learned write objective

During a memory-writing event, the Trainer must give the core an immediate differentiable reason to encode the useful information into the exhaled state.

Canonical example:

Input field: `ZORP = BLUE`  
Write event: core attends the field and exhales HOT.  
Immediate memory probe: using the just-produced state, query `ZORP?` and target `BLUE`.

The memory-writing loss may backpropagate through the current write computation. The persisted successor is then serialized exactly. A later independent recall test must use those bytes without relying on the old computation graph.

### 3. Delayed persisted recall

After the original fact is absent from the Shared Field, the exact persisted Soul must improve recall after process restart.

The first test should use arbitrary bindings that cannot be solved from general model knowledge, for example:

- `ZORP -> BLUE`
- `KEL -> R7`
- `MIRA -> Q2`

Only after arbitrary binding works should the same mechanism be trusted for semantic knowledge such as `CAT -> ANIMAL`.

### 4. Correction and interference handling

Soul must support correction without catastrophic overwrite.

Example sequence:

1. `ZORP = BLUE`
2. unrelated experiences
3. `ZORP = GREEN` with explicit correction evidence
4. query `ZORP?` -> `GREEN`
5. previously stored unrelated bindings remain retrievable

The training/evaluation surface must separately measure:

- new-memory acquisition;
- correction adoption;
- retention of unrelated memories;
- stale-memory suppression;
- interference as Soul fills.

### 5. Functional temperature layers

The four temperatures need distinct jobs, not four copies of the same tensor snapshot.

Proposed first-form semantics:

- **HOT** — immediate working/episodic state; changes every meaningful beat.
- **WARM** — compressed recent experience selected for continued relevance.
- **COLD** — durable personal knowledge/strategy supported by repeated successful evidence.
- **DEEP_COLD** — highly validated durable abstractions or habits eligible for adapter distillation.

Promotion should be a learned/Trainer-governed **compression operation**, not byte-copying HOT into WARM and onward.

### 6. Temperature ablation must be interpretable

For every memory-dependent benchmark, evaluation should report full Soul and one-layer-at-a-time ablations.

Expected behavior should become task-dependent:

- immediate follow-up should rely strongly on HOT;
- moderately delayed recurring context should show WARM contribution;
- repeated stable facts/strategies should survive HOT reset through COLD;
- DEEP_COLD should contribute only after explicit promotion/validation.

A layer that never changes behavior is decorative, not memory.

### 7. Soul capacity must be empirical, not doctrinal

There should be no permanent fixed "N facts" Soul ceiling. Each concrete architecture may have a declared tensor budget, but tests should measure saturation, forgetting, interference, compression ratio, and recall quality as load grows.

If `d_model=64` is insufficient for useful first-form memory, the result should trigger an architecture migration rather than silently broadening semantics around a saturated tensor.

### 8. Soul survives parameter evolution only through explicit migration

Because Soul is parameter-generation-bound, new parameter generations must not silently inhale stale latent geometry.

A parameter update that changes the Soul-reading/writing representation needs one of:

- exact compatibility proof;
- an explicit migration codec/receipt;
- a governed reset/reconsolidation path.

This must be tested before large-scale lifelong training, otherwise every accepted model update risks amnesia or latent corruption.

## Dormant State should become the primary curriculum source

The live `State/dormant` corpus is already large enough to serve as Axon's school rather than merely an archive.

Current material includes approximately:

- `427,001` structured containers;
- `351,978` semantic edges;
- `4,198` layout/symbol groups;
- a 50k knowledge-graph cache;
- exact autobiographical/source evidence under `experience_v1`;
- derived retrieval/index artifacts beneath `.derived`.

The `.derived` indexes are lookup senses and must **not** be treated as knowledge authority or supervision targets. Exact JSONL/evidence authority remains in Dormant State.

The Trainer should eventually expose the core to essentially the entire **admissible** Dormant corpus, piece by piece, across renewable passes. It should not blindly stream raw Dormant text into next-token loss.

## Proposed Dormant curriculum compiler

For each eligible Dormant item, the Trainer should generate one or more runtime-faithful lesson forms while retaining exact provenance.

### A. Substrate and literacy lessons

Use exact Dormant text to continue exercising:

- D16 character grounding;
- D64 packing/unpacking;
- forward/reverse alphabet and symbol ordering;
- spelling;
- punctuation;
- Unicode transport;
- exact copy where copy is the declared skill;
- insertion/deletion/replacement;
- EOS and typed delta mechanics.

These remain permanent rehearsal tissue but should not monopolize gradient pressure after mastery.

### B. Hidden-target semantic generation

Use semantic edges as non-copy generation lessons.

Example source memory:

`Twilio --is a--> tool`

Derived lesson forms may include:

- `TWILIO IS A ___` -> `TOOL`
- `WHAT KIND OF THING IS TWILIO?` -> `TOOL`
- relation classification;
- source+relation -> hidden exact target;
- inverse queries only where the relation is not underdetermined;
- minimal-pair counterfactuals.

The answer target must not be visible in the input rail except in explicitly labeled copy tasks.

### C. Retrieval-use lessons

The Trainer should also teach the opposite behavior: do **not** memorize every exact fact into parameters.

Surface selected verified Dormant evidence through the same governed Cortex/Shared-Field path runtime will use, then train the core to:

- recognize relevance;
- cite/use the surfaced evidence correctly;
- distinguish evidence from inference;
- answer "not supported" when evidence is absent;
- resist stale/conflicting records;
- preserve source/provenance distinctions.

This teaches parameters **how to use memory** while exact details remain recoverable from Dormant State.

### D. Lived-experience lessons

Use provenance-complete episodes containing:

- pre-action field state;
- available tools/advisors;
- proposal/delta/action;
- consequences;
- correction/endorsement/rejection;
- later outcome evidence.

Construct:

- successful replay;
- failed-attempt correction pairs;
- delayed-outcome credit examples;
- counterfactuals;
- tool-use lessons;
- error-avoidance lessons;
- memory-writing events for Soul.

Recovered assistant output remains observation unless outcome evidence makes it eligible to supervise.

## Parameter knowledge, Soul memory, and Dormant evidence must remain distinct

The Trainer should deliberately teach three different things.

### Parameters: generalized competence

Weights should learn stable, reusable abilities:

- read/write the D64 interface;
- language structure;
- semantic discrimination;
- common transformations;
- reasoning habits;
- retrieval use;
- proposal formation;
- error avoidance;
- confidence calibration.

Parameters should answer: **"How do situations like this tend to work?"**

### Soul: private experiential compression

Soul should retain what this particular core has recently or repeatedly experienced and what remains behaviorally useful to it:

- unresolved context;
- corrections;
- short-lived arbitrary bindings;
- recurring personal strategies;
- mistakes and consequences;
- compressed long-lived core-specific knowledge.

Soul should answer: **"What has happened to me, and what from that history matters now?"**

### Dormant State: exact durable evidence

Dormant remains the exact library/autobiography:

- exact facts and versions;
- exact conversation and episode evidence;
- exact source material;
- exact provenance;
- auditable outcomes.

Dormant should answer: **"What exactly happened or is recorded, and what evidence supports it?"**

Neither parameters nor Soul may become the only copy of evidence that must remain exactly recoverable.

## Proposed training sequence

**Ordering correction (2026-09-17):** every gate quoted below from S1 onward
requires free-running emission, which the current objective suppresses. Read
"Phase S0" as following, or running alongside, the corrected-v5 motor shot. See
§Reviewer corrections, BLOCKING 1.

### Phase S0 — Soul mechanics acceptance

No broad cloud training.

Prove:

- exact inhale/exhale persistence;
- restart equivalence;
- same-core binding;
- wrong-core rejection;
- layer ablation telemetry;
- no canonical authority leakage;
- parameter-generation migration behavior.

### Phase S1 — HOT arbitrary-binding memory

Use tiny synthetic arbitrary bindings through the **real D64 field path**.

Target: learn to write novel information into HOT and retrieve it after the original field is gone.

Gate: correct persisted Soul materially outperforms pre-experience/ablated Soul after restart.

### Phase S2 — correction + interference

Increase simultaneous bindings, delays, unrelated intervening experiences, and explicit corrections.

Gate: new/corrected recall rises while unrelated retained bindings stay above floor.

### Phase S3 — learned WARM consolidation

Teach/select compression from repeated HOT experience into WARM.

Gate: targeted memories survive HOT reset because WARM is present; irrelevant HOT noise is not blindly copied into WARM.

### Phase S4 — COLD and DEEP_COLD promotion

Use exact outcome and validation evidence to train/test durable consolidation.

Gate: COLD survives long delay and HOT/WARM churn. DEEP_COLD requires replay/heldout evidence and must support ablation-proven durable behavior.

### Phase D0 — Dormant semantic school

Start with a clean, balanced subset of high-confidence/admissible semantic relations. Include permanent substrate rehearsal and hidden-target generation.

Suggested first proof scale: order of 1k clean relations transformed into several lesson forms, not a permanent dataset limit.

Gate: content generation rises above constant/copy baselines while prior motor gates remain intact.

### Phase D1 — expand through admissible Dormant State

Increase curriculum breadth to larger semantic, procedural, and autobiographical families. Continue renewable passes over the corpus with whole-lineage heldouts, counterfactuals, and forgetting probes.

### Phase D2 — integrated Soul + Dormant learning

Make some lessons require exact surfaced Dormant retrieval, some require personal Soul continuity, and some require both.

Critical minimal pairs:

- same field, different Soul -> different justified answer;
- same Soul, different surfaced Dormant evidence -> different justified answer;
- fact absent from both -> explicit uncertainty/absence behavior;
- stale Soul contradicted by authoritative current evidence -> current evidence wins, while correction becomes new experience.

### Phase D3 — return to proposal/refinement society

Only after one core can read, generate, remember, retrieve, and correct should multi-core proposal/refinement/consolidation training resume as the dominant curriculum.

## Loss and gradient doctrine

This proposal inherits the current lesson from the copy-alignment plateau: **retention gates are not the same thing as permanent high training weights.**

Once a motor skill is mastered:

- continue evaluating it aggressively;
- retain replay examples;
- reduce its gradient dominance unless regression appears;
- restore targeted teaching pressure only when retention drops.

New learning stages must expose explicit component losses and gradient accounting so that a 100%-accurate auxiliary head cannot consume most of the optimization budget while the target skill remains at zero.

For Soul-specific training, telemetry should include at minimum:

- full-Soul task accuracy/loss;
- per-temperature ablation delta;
- write-probe accuracy;
- delayed-recall accuracy;
- correction accuracy;
- interference/retention score;
- Soul layer norm/change magnitude;
- Soul projection/gate gradients;
- writer/reader parameter gradients;
- persisted-byte identity and restart equivalence;
- memory-age/delay bucket;
- promotion source/evidence/validation receipts.

## What this proposal does not authorize

1. No direct canonical writes by the core. Heart remains sole committer.
2. No direct Dormant-to-Soul byte injection as a shortcut.
3. No treating all historical/recovered assistant text as truth.
4. No training on `.derived` evidence-index internals as if they were authority.
5. No merging or sharing private Soul latents across cores.
6. No hidden text side channel outside D64/runtime-faithful surfaces.
7. No claim of memory because Soul bytes changed; behavior must causally depend on them.
8. No long training tranche before the tiny Soul canaries prove write/read/recall.
9. No silent reuse of old optimizer moments or stale Soul geometry across a changed objective/architecture generation.
10. No weakening Heart, Trainer, or provenance gates to make training easier.

## Concrete implementation questions for the roundtable

Reviewers should answer with file/artifact evidence where possible.

1. **Soul tensor shape:** Is the current `[state_tokens, d_model]` opaque recurrent representation sufficient for HOT/WARM/COLD/DEEP_COLD first-form memory, or should temperature layers have different capacities/codecs?
2. **Write mechanism:** Should memory writing be the existing recurrent-state transition plus explicit probes, or should the core gain dedicated Soul-writer attention/slots?
3. **Read mechanism:** Is additive per-temperature projection into initial recurrent state expressive enough, or should Soul be addressable memory/cross-attention rather than only a projected initialization?
4. **Consolidation:** What learned mechanism compresses HOT -> WARM -> COLD without turning promotion into byte-copying?
5. **Gradient boundary:** How should immediate write objectives be designed while preserving persisted-byte truncated boundaries between actual runtime phases?
6. **Migration:** How do we preserve or deliberately migrate Soul when accepted parameters change its latent geometry?
7. **Capacity:** What saturation/interference benchmarks determine when the Soul architecture needs more tokens/width rather than more training?
8. **Dormant eligibility:** Which current Dormant record families are safe for direct semantic supervision, which are retrieval-only, and which are observation-only?
9. **Curriculum construction:** What first 1k/10k relation family gives the cleanest hidden-target semantic generation signal while avoiding label ambiguity and leakage?
10. **Motor retention:** What minimum rehearsal schedule keeps D64 transport/copy/EOS exact without letting mastered components dominate gradient again?
11. **LoRA path:** What exact evidence and distillation process should eventually turn validated DEEP_COLD experience into a core-specific adapter generation?
12. **End-to-end proof:** What is the smallest demo that proves `Dormant evidence -> D64 lesson -> Soul/parameter learning -> restart -> later correct proposal -> Heart-validatable output` without shortcuts?

## Requested reviewer roles

- **Kimi Coder:** synthesize the implementation plan, inspect exact tensor/dataflow mechanics, propose minimal file changes, and attack runtime/training parity.
- **Kimi K3:** challenge the architectural decomposition, memory/compression assumptions, and whether the proposed Soul hierarchy is actually load-bearing.
- **DeepSeek V4.1 Flash:** independently attack the optimization/curriculum design, identify gradient or objective traps, and propose falsifying tests.
- **ChatGPT / GPT-5.6 Sol:** review semantic/authority boundaries, causal memory tests, Dormant eligibility doctrine, and staged acceptance gates.
- **Any additional reviewer:** attempt to prove that a simpler memory design is sufficient, or that the proposed design creates unnecessary latent-state complexity.

Each reviewer should write one response under `roundtable/reviews/` and explicitly state:

1. What part of the diagnosis is wrong or incomplete?
2. What exact Soul capability must exist before broader training resumes?
3. What is the smallest falsifiable Soul canary?
4. What Dormant material should be admitted first and why?
5. What architecture/objective change is actually required versus optional?
6. What regression or authority failure is most likely to be missed?
7. Approve, approve with named changes, or reject — with evidence.

## Proposed near-term execution order after review

1. Reconcile roundtable reviews into one Soul-completion specification.
2. Preserve the current accepted D64 motor checkpoint and historical objective identities unchanged.
3. Implement the smallest versioned Soul-memory candidate needed for S1; do not launch wide semantic training yet.
4. Run arbitrary-binding write/read/restart/ablation canaries locally.
5. Add correction/interference tests.
6. Implement or prove WARM consolidation, then COLD/DEEP_COLD promotion behavior.
7. Build a provenance-governed Dormant curriculum compiler beginning with clean hidden-target semantic relations and permanent D64 literacy rehearsal.
8. Start a fresh governed curriculum lineage from an accepted parameter base; if the objective changes materially, treat it as an explicit new lineage rather than silently resuming incompatible optimizer state.
9. Expand through the admissible Dormant corpus only as heldout content generation, Soul causality, and motor-retention gates pass.
10. Resume multi-core proposal/refinement/consolidation curriculum after a single core proves read + generate + remember + retrieve + correct through the real runtime interfaces.

## Proposed acceptance statement

The Soul milestone should be considered complete enough for broader Dormant-State training only when this statement is empirically true:

> A D64 reasoning core can ingest a novel experience through the governed Shared Field, encode behaviorally useful information into its private Soul, persist and restart from exact Soul bytes, later retrieve or apply that information when the original evidence is absent, correctly accept corrections without destroying unrelated memories, show causal dependence on the appropriate Soul temperature under ablation, and preserve the Heart/Trainer/Dormant authority boundaries throughout.

The Dormant curriculum milestone should be considered ready only when:

> The Trainer can compile provenance-qualified Dormant evidence into target-hidden, runtime-faithful lessons that teach D64 literacy, semantic generation, retrieval use, and lived-experience compression without treating historical observation as truth or allowing mastered motor losses to dominate the new objective.

## Bottom line

The recommended pivot is not to abandon the substrate. It is to treat the substrate and Heart as the **protected body**, finish Soul as the core's **private persistent memory**, and use Dormant State as Axon's **governed lifelong school**.

The core should learn its interface and generalized abilities into parameters while inhaling and exhaling Soul on every real causal boundary. Dormant State remains exact memory authority; Soul becomes personal experiential compression; weights become generalized competence. Once those three roles are demonstrably distinct and interoperable, broader reasoning and society training can resume on a much stronger foundation.
