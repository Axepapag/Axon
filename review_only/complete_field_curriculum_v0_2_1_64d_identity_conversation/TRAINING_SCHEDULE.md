# Training Schedule v0.2.1 — 64D Identity and Conversation Focus

Status: design only; training is not authorized

Stamp: ChatGPT / GPT-5 / 2026-08-18

## Gate 0: Contract acceptance

Codex independently verifies v0.2.1 from clean directories. Zero coverage,
authority, identity, phase, chunk, or deterministic-replay defects are
allowed. Until acceptance, nothing enters `datasets/`, `runs/`, `State/`, or
the live council.

## Gate 1: CPU R0 mechanics

Implement the smallest additive `CompleteFieldReader` adapter outside the
bootstrap engine. Use fresh 64D reader/writer parameters and exact 16D
character input. Required mechanisms:

- ordered base and sibling page streams;
- carried reader state;
- global region/span positions;
- exact re-read pointers;
- externally computed coverage manifest;
- one phase-bound soul transition;
- variable typed output assembled atomically;
- per-core content and attention policy checks.
- nonempty active access to all ten canonical regions;
- one variable transaction with scratch, response-draft, and diary operations.

Run CPU-only synthetic smokes first. Required comparisons are constant output,
first-page only, last-page only, tail only, shuffled pages, missing siblings,
disabled-region writes, and corrupted masks.

Promotion requires zero structural violations and causal improvement over all
shortcut baselines. Loss reduction alone is insufficient. No private identity
records enter this gate.

## Gate 2: Exact mechanics and focused three-region writing

Train exact reconstruction, offsets, masks, coverage, runtime delta payloads,
one-tick council phases, sibling streams, chunks, and length generalization.
Hold out longer page counts and alternate page sizes.

The core must learn that immutable evidence can be attended and masked but not
rewritten. Mask choices affect only later declared views.

Every focused successful example must produce useful, region-appropriate
content in all three outputs: a compact scratch work product, a natural living
response draft, and a grounded diary note. Train no broad field-write policy
at this stage.

## Gate 3: Identity-grounded conversation families B, E, F, L

Begin with synthetic identity/name continuity and neutral dialogue. Then add
an approved private Grade A/B evidence pack locally, keeping exact source
pointers and matched unknown/contradiction cases. Require natural replies,
causal scratch use, and diary claims supported by the completed exchange.

Identity success requires correct/absent/swapped/corrupted evidence tests. A
model repeating "I am Axon" regardless of evidence fails.

## Gate 4: Retrieval and reasoning families C, G

Add exact needles, cross-page rules, arithmetic, tool interpretation, and
evidence-grounded responses. Evidence position and region must be balanced.
The output must cite exact spans and pass counterfactual evidence swaps.

## Gate 5: Broader language and visible work family H

Add natural response drafting, compact visible scratch products, code work,
creative reading/writing, and correction. Scratch promotion requires
removed/swapped/corrupted/irrelevant causal tests.

## Gate 6: Soul family K

Use neutral synthetic soul fixtures first. Require correct/zero/swapped/
shuffled/stale/irrelevant comparisons and donor-following tests. Diary deltas
must cite completed exact events. Souls never become exact historical truth.

## Gate 7: Council uplift

Compare identical examples and measured compute across:

- best single core;
- three independent single-core attempts;
- full proposal/refinement/consolidation council;
- ablated strongest-only and missing-sibling councils.

Promote only if the complete council improves frozen task accuracy and
calibration without hiding extra compute or structural failures.

## Gate 8: 128D comparison

Train a fresh 128D line on the same logical data and compare capability, CPU
latency, memory, throughput, and council uplift. Width is promoted by evidence,
not by step count.

## Gate 9: Offline adapters

Only after the reader and causal gates pass, introduce quarantined LoRA/soul
candidates with immutable source manifests, replay, regression evaluation,
promotion pointers, and rollback. Production weights never learn directly
from unverified self-output.

## Long-run rule

Every long run follows a bounded smoke in the same configuration. A smoke must
show falling discrete loss, rising task metrics above shortcuts, correct CPU
inference, and zero structural violations. No long training is launched by
this package.
