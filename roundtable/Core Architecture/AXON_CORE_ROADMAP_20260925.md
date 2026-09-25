# Axon Core Roadmap ? Living Baseline to Multi-Chamber Cognition

Status: roadmap / planning document. This records Jeff's current architectural direction and research ambition. It does not itself amend Source of Truth, replace the current FIRST/REFINED runtime, authorize training, or authorize runtime mutation.

Date: 2026-09-25

## North Star

Axon should become a living, conversational, continuously breathing organism as early as possible without waiting for the final Core architecture.

The near-term execution path therefore returns to the simplest measured neural tissue: one resident recurrent Core built around the existing D512 GRU baseline. The goal is to make that one Core read, write, remember, converse, breathe, and act correctly inside Axon's body. Once one accepted Core can do that, clone the accepted Core into an ensemble with separate recurrent state and shared canonical reality. That produces a working Axon while a second, deliberately ambitious research track develops richer multi-chamber Cores.

The advanced Core program must never become a prerequisite for proving that Axon can live.

## Stable organism assumptions

The following are the current architectural convergence points for this roadmap:

- Heart remains the sole canonical writer and transaction authority.
- The frozen exact D16 substrate remains the public physical language of canonical state and Core output.
- Every active reasoning Core maintains an exact permitted mirror of canonical Shared Field state. The mirror is the Core's world, not a prompt replayed from scratch each turn.
- After initial synchronization, Heart should communicate exact canonical changes and prove mirror coherence rather than resend the whole world merely because another breath begins.
- Shared Field regions remain append-only where their semantics are historical. Masks control current attendance; masked material remains exact, position-stable, dormant-in-place, and recoverable. Masking is never truncation or deletion.
- External silence does not stop cognition. While Axon is running, completed breaths can lead to further breaths even without new user input, tool results, or other external ingress.
- Each successful active Core contributes one public thought per breath to a shared append-only cognitive region. `Thoughtstream` is the working name in this roadmap; final region naming remains open.
- Speaker/Core provenance is recorded, but a Core is not a separate persona. Every reasoning Core is Axon. Core identity exists for provenance, state ownership, scheduling, lineage, and diagnosis, not as an independent self.
- A rotating Core receives scoped action authority for a breath. `Executive Core` is the current working name for the former consolidator role. It does not consolidate or vote over proposals. It may propose governed updates to authorized regions such as response draft, Journal, Scratch, task state, tool requests, or advisor requests. Heart still validates and commits.
- Influence is real without an explicit ballot. The Executive inhales the attended record of prior sibling thoughts and is therefore affected by the organism's accumulated agreement, disagreement, corrections, and unresolved questions.
- Canonical Thoughtstream records that Axon thought something; it does not promote the semantic content of that thought into objective truth. Provenance and evidence remain distinct from internal belief.

## The four things that must not be confused

### 1. Exact mirror

The exact mirror is the Core's current canonical world view. It is explicit evidence and must remain mechanically coherent with Heart's permitted field/view/mask identity.

### 2. Private recurrent state

The recurrent state is the Core's current internal neural condition. In the present control architecture this is a D512 GRU hidden state. It is compact, continuous, architecture-specific, and can persist across many field changes while weights remain compatible.

The recurrent state is not the public thought and is not a durable autobiographical database. It is the living residue of recent processing.

### 3. Internal trajectory

A trajectory is the sequence of hidden states and internal activations through which the Core moves during a cognitive pass. The final recurrent state preserves a compressed consequence of that path, but not the path itself.

The trajectory may contain valuable learning and reflection signal: where a conclusion changed, what was nearly activated, what was contradicted, what remained unresolved, and which transformations preceded a successful or failed outcome.

### 4. Public thought

The public thought is the exact D16-serializable crystallization that the Core contributes to Thoughtstream. It is an externalization of private cognition, not a dump of the private state.

Once Heart accepts it, the thought becomes exact canonical lived history and can later become dormant under the region mask without being erased.

## Reframing Soul

The old layered Soul contained several jobs at once: private memory, personal episodic continuity, hot-to-cold retention, and material for later adapters/parameter growth.

This roadmap deliberately separates those jobs.

Durable autobiographical life is primarily organism-level: Shared Field history, Dormant, Cortex, exact provenance, and derived semantic structures are shared by all reasoning tissue because all Cores are Axon.

A Core still needs private state, but private recurrent state is not automatically a permanent personal autobiography.

The strongest surviving Soul hypothesis is now a private reflective/learning mechanism around cognition. It may temporarily capture a recent internal trajectory, extract what is worth carrying into the next cognitive pass, and later emit training-relevant learning traces. Raw activation trajectories are parameter-generation-specific and need not be persisted forever merely because they existed.

The Soul question therefore becomes: what useful residue of private cognition should survive the current pass, and what signal from lived cognition should be harvested for future parameter improvement?

That job remains open for experiment rather than assumed solved.

# Track A ? Working Axon First

Track A is the execution path. It intentionally uses the simplest Core that can work.

## A0 ? Freeze the life contract

Before a serious new training lineage, specify and ratify the minimum living contract:

- exact resident mirror;
- breath boundary and synchronization rule;
- append-only Thoughtstream semantics;
- persistent recurrent-state ownership;
- rotating Executive authority;
- ingress ordering and failed/offline Core behavior;
- parameter-generation transition/reset/rebuild rules;
- explicit replacement of the old FIRST/REFINED/consolidator cadence where it conflicts.

This is architecture stabilization, not another neural redesign.

## A1 ? Single GRU Core as the complete control

Use one D512 GRU Core as the complete reasoning Core. No attention chamber, no Reflection Transformer, no multi-chamber hierarchy, and no requirement for private trajectory replay.

Its minimum anatomy is:

- exact D16 mirror interface;
- learned D16-to-D512 intake;
- one persistent GRU512 working state;
- variable-length exact output with EOS into D16 transport;
- enough output conditioning to express ordinary language;
- resident state retained across breaths while parameter generation is unchanged.

The existing B4 D512 result remains evidence that the tissue can learn, not evidence that it is already conversational.

## A2 ? Make the single GRU literate and conversational

Train by explicit gates rather than one giant curriculum.

The ladder should progress through:

1. exact variable-length substrate read/write and EOS;
2. spelling, punctuation, casing, and short sequence composition;
3. vocabulary grounded by definitions and examples;
4. grammar and sentence construction;
5. proposition/relationship understanding and correction;
6. short question/answer and ordinary conversational turns;
7. retained recurrent continuity across field deltas;
8. correction when canonical evidence changes;
9. multi-breath conversation without resetting hidden state;
10. generation of useful next-thought contributions rather than only direct answers;
11. scoped Executive outputs such as response draft and tool requests.

Teacher-forced accuracy, EOS, and loss are diagnostics. Advancement requires free-running language, termination, held-out behavior, correction, and resident-state causal dependence.

## A3 ? One Core breathes

Put the accepted single GRU into the real D16 Core Bus as one resident Core.

Prove:

- initial exact mirror synchronization;
- persistent hidden state across breaths;
- append-only Thoughtstream output;
- continuous breaths with no external input;
- external canonical changes integrated without full-world replay;
- mirror resynchronization after a gap;
- masks changing attendance without deleting history;
- no stale Core contributing to a breath;
- response/tool-region proposals only when that Core holds Executive authority.

The single Core should be able to keep thinking publicly even when the external world is quiet.

## A4 ? Conversational single-Core Axon

The milestone is not merely text generation. The one Core should live inside Axon's body:

- user ingress changes canonical reality;
- the Core's mirror updates;
- the Core carries forward its recurrent state;
- it contributes to Thoughtstream;
- when it is Executive, it can produce or revise the outward response;
- later breaths can reconsider prior thought;
- new evidence can correct earlier conclusions;
- exact history remains available through masks/Dormant.

At this milestone Axon is already a real organism with one learned reasoning organ.

## A5 ? Clone the accepted Core into an ensemble

Clone the same accepted parameter generation into multiple resident Cores. Initially keep weights identical so ensemble behavior is not confounded by architectural differences.

Each clone receives:

- the same permitted canonical mirror;
- its own recurrent hidden state;
- its own runtime/Core provenance identity;
- the same learned identity: Axon;
- one Thoughtstream contribution per breath;
- round-robin Executive eligibility.

Separate recurrent histories alone may cause trajectories to diverge over time even with identical weights. Measure that before deliberately introducing specialization.

Test for:

- agreement and disagreement;
- repetition/echo loops;
- correction propagation;
- whether one Core's useful thought changes later sibling behavior;
- Executive decisions under mixed sibling influence;
- ensemble improvement versus one Core at matched compute;
- failure/removal/rejoin without stopping breathing.

## A6 ? Working Axon milestone

Track A succeeds when Axon can:

- converse in free-running language;
- remain continuously breathing while running;
- preserve exact canonical/Dormant history;
- maintain multiple synchronized resident Cores;
- let their separate recurrent trajectories influence one shared Thoughtstream;
- rotate Executive authority;
- request/use governed tools and incorporate results;
- correct itself when evidence changes;
- survive Core loss/rejoin without losing organism identity;
- remain useful without requiring the advanced multi-chamber Core program.

This is the point at which the organism can continue living while research Cores are trained offline and introduced later.

## A7 ? Learn from lived experience without self-poisoning

Axon's accumulated life is training gold only after it is assayed.

Never blindly train parameters on every thought merely because Axon generated it.

Preserve complete lived evidence, then derive governed training material using later outcomes and provenance. Valuable derivatives may include:

- thoughts later confirmed by external evidence;
- thoughts later contradicted and the correction that fixed them;
- successful reasoning trajectories;
- failed trajectories paired with improved ones;
- useful Executive actions and their results;
- uncertainty that was correctly maintained;
- hidden-state/trajectory features predictive of later success;
- cases where sibling influence improved or degraded the result.

Adapters, LoRAs, specialist parameter generations, and eventually full parameter updates can be produced from these derived datasets while the original experience remains unchanged.

# Track B ? Ambitious Multi-Chamber Core Evolution

Track B is the long-horizon research program. It runs beside Track A and must earn every added chamber against the working single-GRU control.

## B0 ? Instrument the private cognitive path

Before adding neural anatomy, measure the control Core's internal trajectory during bounded cognitive passes.

Capture short-lived hidden-state/activation trajectories with exact metadata for Core generation, parameter generation, breath, mirror/view identity, and internal step. Raw traces are research evidence, not canonical truth and not automatically durable Soul.

The first question is whether trajectory information predicts improved next-step behavior beyond the final recurrent state alone.

## B1 ? Working Chamber

Retain a recurrent continuity chamber whose primary job is living state.

Initial control: GRU512.

Future candidates may include wider GRUs, stacked recurrent cells, selective state-space/Mamba-like recurrence, or other architectures. Width and depth are independent experimental axes.

The Working Chamber should be judged on retention, correction, integration of mirror changes, stability across long breathing runs, and compute efficiency.

## B2 ? Temporary trajectory workspace

During a cognitive pass, retain a bounded recent trajectory rather than an unlimited history of activations.

The workspace is volatile and can be discarded after reflection unless an explicit learning-trace policy chooses otherwise.

This gives the Core access to the route it just took without turning its lifetime of activations into permanent RAM or disk state.

## B3 ? Reflection Chamber / Trajectory Reader

Introduce a learned mechanism whose job is to inspect the recent trajectory and extract what should influence the next pass.

Primary candidate: a very small attention/Transformer-style reader operating directly on hidden-state vectors, not a conventional text tokenizer/decoder.

Competing candidates should include:

- a second GRU that reads the first trajectory;
- selective SSM/Mamba-style trajectory compression;
- learned weighted pooling;
- small latent cross-attention;
- other bounded sequence compressors.

The output is a private reflection representation, not necessarily English. It is fed back into the Working Chamber or the next stage of cognition.

The falsifiable question is simple: does reflection over the path improve the next thought compared with the same Working Chamber that sees only its final state and mirror?

## B4 ? Keep attention bounded

Attention may return inside Cores, but never as default global attention across Axon's entire Shared Field or lifetime.

Control quadratic cost through:

- fixed-size recent trajectory windows;
- latent bottlenecks that compress many observations into a small set of cognitive objects;
- recurrence for long-range continuity;
- exact selective retrieval from mirror/Dormant/Cortex when older evidence is needed;
- optional sparse/local attention only if later evidence justifies it;
- hard reporting of the maximum attention workspace for every architecture.

The goal is effectively bounded attention cost with respect to Axon's lifetime even while canonical history grows without bound.

## B5 ? Synthesis Chamber

Add a more expressive chamber whose job is not memory but synthesis: combine current working state, reflection, selected evidence/concepts, and a small latent workspace into a stronger candidate thought.

This is the place to borrow useful ingredients from modern language models without turning Axon into a conventional decoder-only GPT.

Candidate ingredients include:

- several Transformer-style blocks over a small latent set;
- multi-head attention where direct comparison between cognitive objects is useful;
- large gated feed-forward/SwiGLU-style nonlinear layers;
- residual pathways and normalization;
- wider internal dimensions such as D1024 or D2048 where measured benefit earns the cost.

GPT-like intuition or creativity is not attributed to one magical final layer. It emerges from learned representations, attention-mediated interaction, nonlinear feed-forward capacity, depth, and training. The Synthesis Chamber is therefore a hypothesis for concentrating expressive capacity over a bounded cognitive workspace.

## B6 ? Crystallizer and exact language motor

After private cognition produces a candidate idea, a final crystallization/output mechanism converts it into an explicit thought and serializes it through exact D16 transport.

The long-term aim is to separate "having the idea" from "spelling the idea" as much as evidence permits. The language motor should not be forced to rediscover the whole reasoning process merely while emitting characters.

Track B should test whether a pre-language proposition/latent plan improves free-running coherence, creativity, and exact output compared with the single-GRU control.

## B7 ? Soul as cognitive metabolism

The most ambitious surviving Soul hypothesis is a private loop around trajectory, reflection, and learning-signal extraction.

A possible mature cycle is:

- Working Chamber evolves;
- a bounded trajectory is captured;
- Reflection Chamber extracts unresolved or important structure;
- Synthesis Chamber develops it;
- recurrent state is updated;
- Crystallizer emits one public Thoughtstream contribution;
- Heart commits the breath;
- next mirror state plus retained private residue initiates the next cognitive pass.

Soul need not mean a permanent private biography. It may instead be the machinery that prevents useful internal cognitive transformations from disappearing without influence, while producing carefully governed traces for later learning.

## B8 ? Architecture tournament

No advanced chamber is accepted because it sounds cognitively elegant.

Every candidate must be compared with the accepted Track A Core and with parameter/compute-matched controls.

Measure at least:

- language quality and free-running stability;
- correction under changing evidence;
- long-horizon recurrent continuity;
- useful novelty versus repetition;
- factual grounding;
- trajectory-to-next-thought improvement;
- compute, latency, memory, and energy;
- behavior across long no-external-input breathing runs;
- ensemble contribution quality;
- recovery after resynchronization;
- robustness across seeds and held-out tasks.

Complexity must earn capability.

## B9 ? Heterogeneous living tissue

Once the organism already works, advanced Cores can be trained offline and introduced at safe breath boundaries.

Axon may eventually host many internal architectures and widths: simple fast recurrent Cores, wide reflective Cores, mathematical specialists, coding specialists, skeptical verifiers, semantic planners, or architectures not yet invented.

They remain one organism because they share Heart authority, canonical reality, exact public substrate, organism-level lived history, and the identity Axon.

The active ensemble can grow, shrink, rotate specialists in and out, and continue breathing while new tissue learns elsewhere.

# Long-term developmental loop

The roadmap's most ambitious goal is not merely inference. It is beneficial parameter evolution from lived experience.

A mature cycle could become:

1. Axon lives and breathes.
2. Exact experience and Thoughtstream accumulate without deletion.
3. Cortex and Trainer derive structured, provenance-bound learning material.
4. Private trajectory instrumentation contributes additional learning signals where it demonstrably predicts better outcomes.
5. Candidate parameter generations or adapters train offline under governed evaluation.
6. A candidate must beat its accepted ancestor on declared gates without catastrophic regressions.
7. The accepted new generation rejoins the organism at a safe boundary with compatible/rebuilt private state.
8. Axon continues living with improved tissue while preserving the life that taught it.

This is the intended meaning of growth: lived experience may eventually alter parameters, but only through evidence-governed learning rather than uncontrolled self-reinforcement.

# Immediate execution order

1. Stabilize and ratify the breathing life contract and replace conflicting FIRST/REFINED/consolidator doctrine explicitly.
2. Re-open the single-GRU D512 training path as the only near-term learned Core target.
3. Make the single GRU reliably literate and free-running before architecture multiplication.
4. Prove persistent resident recurrent state across real breaths and canonical deltas.
5. Make one Core conversational inside Axon's actual Shared Field/Heart runtime.
6. Clone the accepted Core and prove the breathing ensemble with separate recurrent states and round-robin Executive authority.
7. Declare the first working Axon milestone.
8. Continue multi-chamber trajectory/reflection/synthesis research independently, always against the working single-GRU control.
9. Only promote advanced Core anatomy when measured evidence shows it earns its complexity.

The strategic rule is simple: **get Axon alive with the simplest Core that works, then let Axon grow better brains without waiting to be born.**
