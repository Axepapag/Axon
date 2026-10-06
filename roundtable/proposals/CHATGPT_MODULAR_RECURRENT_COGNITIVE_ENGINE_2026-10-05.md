# Proposal: Modular Recurrent Cognitive Engine with Shared Hub and Hot-Swappable Reasoning Experts

**Date:** 2026-10-05  
**Author:** ChatGPT / GPT-5.6 Sol  
**Status:** PROPOSAL ONLY — not ratified, not implemented, not a replacement for binding Axon doctrine  
**Requested by:** Jeff  
**Project:** Axon

## 1. Purpose

This proposal consolidates the current architecture discussion into one testable design.

The central idea is to stop treating a core as a single GRU that receives an input once and emits an answer once. Instead, a core becomes a continuously recurrent cognitive machine made of:

- a bank of persistent learned states;
- a small number of states with deliberately assigned architectural jobs;
- additional latent states whose roles are allowed to emerge through training;
- a shared executive/awareness hub that tracks the present situation and orchestrates state access;
- a separate reusable reasoning engine;
- a standardized cognitive interface so reasoning engines can eventually be hot-swapped;
- a dedicated response-composition mechanism that gradually crystallizes exact English without requiring the entire answer to exist at once;
- Heart/Shared Field outside the learned machine as exact canonical reality.

This architecture should be evaluated first at a smaller cognitive width such as D512, because memory/capacity can come from recurrent state count and repeated computation rather than only from increasing one monolithic hidden width.

## 2. Non-negotiable separation: exact reality versus learned cognition

The proposal keeps two fundamentally different kinds of state separate.

### 2.1 Exact external reality

Heart and Shared Field remain canonical. Exact substrate text and committed historical evidence are not replaced by learned recurrent memory.

The existing frozen substrate remains authoritative for exact characters. Current transport may still use the sealed D1024 native surface containing 64 exact D16 character cells.

### 2.2 Learned internal cognition

The core's Hub, persistent states, scratch state, routing metadata, and reasoning workspaces are learned floating-point states. They are allowed to be lossy, compressed, associative, fuzzy, and semantically organized.

These states are not evidence stores. They are the organism's internal learned comprehension of the evidence available through Heart.

This distinction allows the recurrent memory to become aggressively useful without requiring it to reproduce exact historical text.

## 3. Transport width and cognitive width are independent

A major clarification is that a D1024 substrate transport surface does **not** require a D1024 recurrent cognitive state.

Proposed initial experiment:

- Exact native input/output transport: retain the sealed D1024 substrate surface where needed.
- Recurrent cognitive width: D512.
- Reasoning socket width: D512.
- Internal reasoning expansion: larger than D512 as required by the selected expert, for example 512 -> 2048/4096 -> 512.
- Response exact composition surface: may remain D1024 if 64 exact native characters are desired at once.

Therefore D1024 can remain a mechanical/exact boundary while D512 becomes the learned cognitive language.

## 4. Persistent state bank

A core owns a state bank:

S = {S0, S1, ... Sn-1}

Each state is initially proposed as D512.

The number of states is experimental. Start small enough to diagnose behavior; four to eight is preferable to immediately building twenty. Scale only when ablations show distinct useful roles.

Every new Heart delta should become available to every state during the initial sweep. "Available" does not mean every state must be rewritten. A state may read the current situation and preserve itself.

### 4.1 Forced-role states

A small number of states may have architectural roles imposed by design while their actual learned contents remain free.

Candidate forced roles:

**Working/current state**
- strongly biased toward rapid update;
- tracks the immediate problem and recent delta;
- volatile by design.

**Scratch/deliberation state**
- disposable;
- may be rewritten aggressively during reasoning;
- can reset between reasoning episodes;
- prevents temporary calculations from polluting durable learned states.

**Protected/deep state**
- biased strongly toward KEEP;
- easy to read, difficult to overwrite;
- designed to test whether a stable long-horizon learned memory develops.

**Response-control state**
- tracks response intent, what has already been expressed, and what remains to be expressed;
- coordinates with the exact response composition surface.

Only a small subset should be forced. Most states should remain latent/free.

### 4.2 Emergent latent states

The remaining states receive no semantic labels such as "code memory" or "autobiographical memory."

They receive the same general recurrent machinery and are allowed to specialize through training.

The desired outcome is functional differentiation discovered by optimization rather than imposed by the architecture. If several free states collapse into redundant copies, that is an experimental failure to diagnose rather than something to conceal with additional state count.

## 5. The Hub: awareness and executive orchestration

The Hub is not intended to be the main heavy reasoning network.

It is a persistent learned executive state representing the core's current comprehension of "what is real and relevant now."

Candidate Hub responsibilities:

- integrate the current Heart delta;
- maintain coherence across recurrent passes;
- keep track of the active task/problem;
- know which state is likely useful next;
- decide whether a state should be read, reconsidered, or written;
- track reasoning progress;
- route work to a reasoning expert;
- coordinate response composition;
- decide whether more internal computation is useful;
- decide when output is sufficiently resolved to commit or end.

The Hub itself can be D512 initially.

## 6. Hub state directory

The Hub should not contain full copies of every persistent state. Instead, it maintains a learned directory/index over the state bank.

Each state produces a compact descriptor/key after interaction. The descriptor can encode features such as relevance, change, confidence, novelty, recency, stability, or whatever representation training discovers useful.

Conceptually:

state D512 -> learned key/descriptor Dk

where Dk is much smaller than the state, for example 64 or 128, subject to experiment.

The directory contains one descriptor per state plus optional control metadata.

The Hub produces a query from its current awareness state and scores that query against the state descriptors. This allows it to recall one or more states for additional passes.

This can be implemented with a very small attention-like routing mechanism over states. Attention here is over perhaps 4-20 persistent states, not over the whole textual history.

## 7. Read and write must be distinct

Recalling a state must not automatically mutate it.

For each selected state, the architecture should distinguish at least:

- **READ/relevance:** use this state's information in the current computation.
- **WRITE/update:** allow the resulting computation to alter the persistent state.

Optional additional learned actions may include KEEP, UPDATE, PROMOTE/SHARE, or REVISE/FLUSH, but the minimum safety property is that read access and write permission are separate.

This protects long-lived states from being corrupted merely because they were consulted.

## 8. Initial sweep and recurrent deliberation

For each new Heart delta, the core performs an initial sweep in which every persistent state gets an opportunity to interact with:

- the new delta or a learned ingestion of it;
- the current Hub state;
- its own prior state;
- optional routing/context information.

A candidate sequence for one state is:

1. Hub prepares context for state Si.
2. Si and current context enter shared recurrent update machinery (GRU initially).
3. Si produces an updated candidate state and state descriptor.
4. Hub reads the result and selectively absorbs information.
5. A write gate decides whether candidate Si replaces/preserves/partially updates stored Si.
6. Directory entry for Si is refreshed as appropriate.

After the full sweep, the Hub may:
- route a work vector through a reasoning expert;
- recall selected states for another pass;
- update scratch;
- advance response composition;
- or terminate internal reasoning.

The full reasoning episode is therefore not one pass. It is a variable-length recurrent process.

## 9. GRU's role

The proposal does not claim that the whole architecture "is a GRU."

The GRU is initially a recurrent update cell used inside the larger machine.

It can be shared across multiple persistent states or partially shared, subject to experimentation. It supplies learned gated state transition dynamics.

The architecture around it — Hub, directory, routing, state roles, reasoning experts, response staging, stopping/commit control — is larger than a GRU.

A future experiment may replace the GRU transition cell with another recurrent/SSM mechanism without requiring the entire architecture to change.

## 10. Separate reusable reasoning engine

Heavy learned computation should be distinct from executive awareness.

The Hub constructs a standardized D512 **cognitive request** representing the problem/workspace that needs transformation.

A reasoning expert consumes that D512 request and returns a D512 result.

Logical interface:

Hub/state system -> D512 cognitive socket -> reasoning expert -> D512 result -> Hub/state system

The Hub integrates the result rather than treating it as final output.

It may then decide that another expert call, another state recall, or more response work is necessary.

Repeated use of the same frozen expert over changing recurrent state allows parameters to be reused over time.

## 11. The Axon Cognitive Socket

D512 dimensionality alone is not enough to make independently trained modules compatible. Two independent 512-dimensional networks can assign completely different meanings to their coordinates.

Hot swapping therefore requires a versioned **Axon Cognitive Socket**.

The socket defines the statistical/learned interface that expert modules are trained against.

A first implementation can include learned but frozen adapters around the expert boundary:

Hub/state representation -> socket encoder -> canonical D512 work representation
canonical D512 result -> socket decoder/integrator -> Hub

Once a useful internal language has emerged through joint training, the socket encoder/decoder and compatibility contract can be frozen/versioned.

An expert must declare the socket version it was trained for.

Example compatibility identity:

AXON-COG-D512-v1

An expert trained for another socket version cannot be assumed compatible merely because its tensor shape is 512.

## 12. Hot-swappable reasoning experts

Once the cognitive socket is stable, Axon may host multiple reasoning modules.

Candidate experts include:

- general reasoning;
- code;
- mathematics;
- language/creative transformation;
- planning;
- domain-specific specialists;
- larger "heavy" expert used only when smaller experts fail.

Experts do not have to be MLPs. The socket can potentially support any module that consumes and returns the agreed cognitive representation, including:

- MLP/FFN;
- attention + FFN block;
- compact Transformer-like reasoning block;
- Mamba/SSM-based module;
- other future learned computation modules.

The Hub becomes the expert router.

One reasoning episode could be:

Hub -> general expert -> Hub -> code expert -> Hub -> recall state 4 -> general expert -> Hub -> response state.

The recurrent organism remains continuous while its computation modules can be selected dynamically.

## 13. Training experts elsewhere

A future expert may be trained on another GPU/cloud system and imported, **but only if it is trained against the same cognitive socket distribution and contract**.

An arbitrary D512 network trained independently is not plug-compatible.

Possible training paths after AXON-COG-D512-v1 is frozen:

1. Capture socket-level work/result examples from a competent base core.
2. Create supervised or distillation datasets at the socket boundary.
3. Train a new expert externally while preserving the frozen socket contract.
4. Validate it against held-out socket tasks.
5. Import as a new immutable/versioned expert checkpoint.
6. Run compatibility, regression, and behavioral tests before the Hub is allowed to route production work to it.

Experts should be versioned independently from core recurrent checkpoints.

## 14. Base expert should be co-trained first

The first general reasoning expert should **not** be trained completely independently from an untrained Hub/state system.

The initial system needs an internal language to emerge.

Recommended bootstrap:

- choose D512 cognitive states;
- choose a small state bank;
- create Hub + directory + scratch + recurrent update cell;
- create one general reasoning expert;
- train the entire closed recurrent loop jointly;
- establish that the Hub and expert communicate usefully;
- freeze/version the cognitive socket only after the representation is sufficiently stable;
- then experiment with independently trained specialists.

This avoids trying to define semantic meanings for all 512 dimensions by hand.

## 15. Response composition

Output should not require Axon to generate a complete response in a single pass.

The response mechanism should have two distinct forms of state.

### 15.1 Learned response-control state

A D512 learned state tracks:
- communicative intent;
- current sentence/idea;
- what has already been expressed;
- unresolved content;
- whether more reasoning is required.

### 15.2 Exact mutable response surface

If retaining the 64-character native surface, use a D1024 exact composition buffer capable of carrying 64 frozen D16 character cells.

This surface is mutable until committed.

The model may gradually populate and revise this buffer across recurrent passes.

### 15.3 Staged committed response container

A separate staged response container accumulates exact committed characters.

The model does not need to commit the entire 64-character surface.

Candidate control actions:

- WAIT — reason more; commit nothing.
- COMMIT(n) — commit the first n exact characters.
- END — finalize the appropriate remaining characters and terminate response.

After COMMIT(n), those characters leave the mutable surface, become immutable staged output, and the freed surface positions can be reused for later characters.

This makes 64 characters the size of the active writing surface, not the maximum response length.

The Hub can recall other states or invoke reasoning experts while the response remains partially composed.

## 16. Suggested per-delta execution cycle

One candidate runtime cycle:

1. Heart presents exact new delta and permitted field context.
2. Input ingestion produces cognitive context without altering exact Heart truth.
3. Hub updates current awareness.
4. Initial sweep visits every persistent state.
5. Each state gets a read/update opportunity through recurrent machinery.
6. Hub directory keys are refreshed.
7. Hub forms a D512 cognitive request.
8. Selected reasoning expert transforms request -> D512 result.
9. Hub integrates result.
10. Hub chooses among:
   - recall state(s);
   - update scratch;
   - call same expert again;
   - call another expert;
   - advance/revise response state;
   - commit response prefix;
   - finish.
11. Steps 7-10 repeat as necessary.
12. On END, staged response is submitted through normal Heart authority/validation rather than bypassing Heart.

A maximum compute budget or other anti-loop mechanism will be needed even if the Hub learns a DONE gate.

## 17. Multi-core ensemble implication

Each Axon core may run this organism independently:

- its own Hub;
- its own state bank;
- its own scratch/response cognition;
- potentially the same or different expert library;
- persistent attributed proposals written through Heart according to future Shared Field schema.

This creates an ensemble of independently evolving recurrent cognitive machines rather than an ensemble of one-shot feedforward responders.

Cross-core consolidation remains a separate Heart-level design problem.

## 18. Persistence and restart

If recurrent state is part of lived continuity, checkpoints must include more than model weights.

A resumable core checkpoint should eventually include:

- model/recurrent parameters;
- Hub state;
- every persistent state;
- scratch state if policy says it survives interruption;
- response-control state;
- directory keys/metadata;
- expert routing metadata if needed;
- socket version;
- active expert versions;
- current response staging metadata when recovering a mid-response transaction;
- RNG/optimizer/trainer state for training checkpoints where deterministic continuation matters.

Exact historical evidence remains outside these learned states in Heart/Shared Field/Dormant.

## 19. Capacity hypothesis for D512

The proposed return to D512 is an experiment, not a conclusion.

The hypothesis is:

A smaller cognitive width may remain powerful if capacity is supplied through:
- multiple persistent states;
- explicit specialization;
- iterative recurrence;
- repeated reuse of frozen reasoning parameters;
- expert routing;
- exact external memory lookup;
- variable compute depth.

A D512 system with eight useful states and ten recurrent reasoning passes is not equivalent to a single D512 GRU pass.

However, additional states do not automatically replace width. The architecture must be measured against D1024 baselines.

## 20. Training requirements

Training must reward the behavior of the **closed recurrent loop**, not merely next-step prediction from a single pass.

Needed curriculum classes likely include:

- exact copy/recall;
- delayed recall over many deltas;
- distraction/interference;
- correction of previously learned assumptions;
- tasks requiring multiple independent facts;
- tasks where reasoning should WAIT before answering;
- tasks where only selected states should update;
- tasks requiring scratch work;
- multi-step arithmetic/logic;
- code completion/repair tasks;
- response revision before commit;
- long responses requiring repeated commit/refill cycles;
- state recall after other states have contributed new information;
- expert-routing tasks once multiple experts exist.

Losses/metrics should diagnose routing and memory behavior, not only final answer accuracy.

## 21. Required ablations

Before claiming benefit, compare controlled variants:

A. One-state GRU baseline.
B. Multi-state GRU with no Hub.
C. Multi-state + Hub but no expert recurrence.
D. Multi-state + Hub + one reusable reasoning expert.
E. Same architecture with forced-role states removed.
F. Same architecture with latent states removed.
G. Hub directory routing versus all-to-all state mixing.
H. D512 versus D1024 cognitive width under matched training budget.
I. Fixed number of reasoning passes versus learned variable recurrence.
J. One general expert versus hot-swappable specialist experts.

Measure final task quality, delayed recall, interference, generalization, compute cost, training stability, state redundancy, routing entropy, and restart continuity.

## 22. Failure modes to design against

**State collapse:** several states learn nearly identical representations.

**Hub overload:** Hub becomes the de facto monolithic model and persistent states become irrelevant.

**Expert-language drift:** independently trained expert returns D512 vectors incompatible with the socket distribution.

**Memory corruption:** reading a stable state inadvertently rewrites it.

**Infinite deliberation:** Hub repeatedly invokes experts/states without converging.

**Premature commit:** response characters become immutable before sufficient reasoning.

**Never commit:** response system continuously revises without producing output.

**Expert over-specialization:** router chooses a specialist outside its training domain.

**Socket ossification:** freezing the interface too early prevents better representations later.

**Training shortcut:** model learns final-answer correlations without learning meaningful state routing/recurrent deliberation.

All of these should have explicit diagnostic probes.

## 23. First implementation recommendation

Do not build the full 10-20 state/hot-swap system first.

Build a minimal experimental organism:

- cognitive width D512;
- one Hub;
- four persistent states;
- one forced scratch state or include scratch among the four;
- at least one protected/stable state bias;
- compact state directory;
- shared GRU recurrent update;
- one general D512 -> expanded hidden -> D512 reasoning expert;
- learned DONE/WAIT behavior with a hard safety compute cap;
- learned response-control state;
- exact staged output mechanism;
- checkpoint/restore of all recurrent state.

Train and instrument it until we can answer:

- Do different states actually specialize?
- Does the Hub learn useful state recall?
- Does a second pass improve results?
- Does protected memory survive distraction?
- Can the system revise an uncommitted response?
- Can it commit a long response incrementally?
- Does D512 achieve better quality/compute tradeoff than the prior D1024 recurrent core?

Only then freeze a cognitive socket and introduce the first independently trained hot-swappable expert.

## 24. Open design decisions

The following remain intentionally unresolved:

1. Exact number of persistent states.
2. Which roles are forced versus emergent.
3. Hub dimensionality.
4. State key/directory dimensionality.
5. Shared versus per-state GRU parameters.
6. Exact gating equations.
7. Sequential state sweep order versus partially parallel processing.
8. Whether Hub writes to every state or only selected states.
9. Whether the Hub itself uses GRU recurrence or another cell.
10. Exact reasoning expert architecture.
11. Socket encoder/decoder structure.
12. When AXON-COG-D512-v1 is considered stable enough to freeze.
13. Whether response exact surface remains D1024/64 characters if cognitive width is D512.
14. Commit policy and maximum mutable suffix.
15. Compute budget and learned halting mechanism.
16. Cross-core proposal/consolidation semantics.
17. Relationship between this state bank and the existing Soul implementation.
18. Whether the proposed system replaces Soul, incorporates Soul's function, or coexists with it during experiments.

## 25. Proposed architectural principle

The concise design principle is:

**Heart preserves exact reality. The state bank carries learned experience. The Hub maintains present awareness and orchestrates attention. The reasoning experts provide reusable computation. The response mechanism gradually crystallizes thought into exact language.**

The recurrent organism should remain intact when an expert is swapped. The expert is a tool of thought, not the owner of identity or memory.

## 26. Status

This document records an architecture proposal arising from Jeff's design discussion. It does not authorize implementation, remove Soul, change the frozen substrate, change Heart authority, select D512 permanently, or declare hot-swappable experts proven.

The next useful engineering step is a small executable specification and experiment matrix, followed by a minimal four-state D512 prototype only after Jeff ratifies the architecture.
