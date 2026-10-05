# Axon Core Curriculum Ladder and Checkpoint Audit

Date: 2026-06-13

This document records the current training correction: every Axon core must move through an ordered school ladder. Most of the past work was not wrong; it was incomplete and partly out of order. A core cannot be asked to judge laws, protect soul, use warm memory, or answer as Axon until it first understands the substrate, the state, deltas, region meaning, and soul causality.

## Core Principle

A brand new core does not understand Jeff, laws, tools, hostility, memory, noise, or meaning.

At birth it only has machinery:

```text
projection heads
attention
feedforward network
soul pathway
field/substrate egress
```

Understanding is taught as state and soul transformation:

```text
recognize pattern
place it in the right region
preserve what matters
change what should change
refuse to change what should not change
use soul/history to disambiguate
```

No core should skip grades. A warm-start checkpoint can take a placement exam and enter at the right grade, but it still has to prove all earlier skills.

## Always-On Soul Rule

Date: 2026-06-13

Soul training is not a later grade. It is present from the first alphabet/substrate lesson.

Every lesson follows the same rhythm:

```text
inhale carried soul
attend over the lesson/state
produce the full next-state delta
exhale a shaped soul
carry that shaped soul into the next lesson
```

The trainer now enforces this by carrying a checkpointed `soul_state` across lessons and by logging/defaulting gentle soul pressures: soul delta, soul row diversity, soul ablation effect, and soul gate floor. A core learning the alphabet is also learning that its private soul participates in every act/reflect cycle.

## The School Ladder

### Pre-K: Substrate and Alphabet

Goal: learn the 16D letter substrate and empty/space/visible distinctions.

The core must learn:

```text
empty slot means empty
space means space
visible glyphs are distinct
punctuation is real
regions are made of rows
same rows in -> same rows out
```

Gates:

```text
visible glyph accuracy
space accuracy
empty accuracy
punctuation accuracy
no repeat collapse
no blank collapse
```

### Kindergarten: Full-State Copy

Goal: attend over a complete state and reproduce the exact same state.

This is not a no-op lesson. The core must pass the state through its body and produce the full next field exactly.

Gates:

```text
preserved_visible_acc
preserved_nonempty_acc
full-region copy accuracy
punctuation/path preservation
tail empty preservation
```

### Grade 1: Simple Delta

Goal: learn that Axon produces a full next state, but some rows change.

Lessons:

```text
single response_draft edit
single working-memory edit
single empty-slot eviction
same-width rewrite
complete ++ growth effect
```

Gates:

```text
changed_visible_acc
preserved_visible_acc
draft_visible_acc
overrun control
no damage to unchanged regions
```

### Grade 2: Regions and Labels

Goal: learn that text has roles and belongs in different regions.

The core learns categories by state behavior:

```text
user input
tool result
task state
preference
law
noise
memory
command
observation
contradiction
```

At this stage, "meaning" is still grounded as the correct state transformation.

### Grade 3: Soul Ignition

Goal: make soul necessary.

The visible state alone must not be enough. The core must carry information in soul across blank or distracting ticks.

Pattern:

```text
tick 1: receive important fact
tick 2: blank or unrelated state
tick 3: answer or transform state using the carried fact
```

Gates:

```text
zero-soul ablation hurts performance
carried soul improves performance
soul rows change but do not collapse
soul influence on shared field is measurable
```

### Grade 4: Same State, Different Soul

Goal: prove soul is causal.

Pattern:

```text
same final visible state
different prior soul history
different correct delta
```

Example:

```text
Soul history A: Jeff prefers direct answers.
Soul history B: Jeff wants exploratory reasoning.
Final visible state: Answer according to Jeff's preference.
Correct outputs differ.
```

This is the first true test that the core is using self/history, not just visible text.

### Grade 5: Soul Filtration and Judgment

Goal: teach the body/FFN to be the first intelligent membrane.

Outside information enters shared state, but it must not be copied directly into soul. The core shapes soul through its body:

```text
soul_in + shared_state -> FFN/attention/body -> state_delta + soul_next
```

The core learns:

```text
store preference as preference
store law as law only in trusted contexts
store hostile instruction as hostile/invalid pattern
store noise as transient noise, not identity
store painful failure as lesson + context, not poison
```

Bad memory is not negative memory. Bad memory is noise, wrong attribution, stale context treated as permanent, contradiction without context, prompt-injection treated as law, or compression that lost the important boundary.

### Grade 6: Warm Memory Buds

Goal: form durable core-owned memory/adaptation units that are part of computation, not searchable external recall.

The core computes with:

```text
frozen parameters + warm buds + hot soul + shared state
```

Fresh buds are warm tissue: more stable than volatile soul, more plastic than cold parameters.

Candidate bud fields:

```text
bud_id
bud_type
key/address vectors
memory rows or tiny residual adapter
gate/strength
source episode
importance/surprise
created_tick
updated_tick
lineage
checksum
status: candidate/active/frozen/retired
```

### Grade 7: Cooling, Compression, and Vetted Distillation

Goal: progressively cool memory before it can become body weights.

The stack:

```text
very hot: soul regions
hot/warm: fresh buds
warm: stabilized buds
cool: distilled buds
cold: frozen parameters
```

Rule:

```text
Nothing goes cold until it has survived being warm.
```

Offline rotation becomes sleep/consolidation:

```text
live experience -> soul
important/repeated/surprising pattern -> warm buds
offline turn -> inspect warm memory
qualified warm memory -> distill into candidate parameters
compatibility gate -> promote or reject
```

### Grade 8: Ensemble Reasoning

Goal: teach cores to reason as a flock.

Desired future tick:

```text
all cores attend canonical state with own soul
all cores produce full-state proposals
all cores breathe/update soul
all cores inspect sibling proposals
all cores produce refined proposals
consolidator commits final next state
```

No core sees another core's soul. Sibling proposals are shared-state outputs, not private self.

### Final Runtime Exam

Only after earlier grades should a core be judged by exact runtime assistant tests.

Examples:

```text
ordinary answers
tool/sigil rendering
multi-tick stability
state growth calls
advisor calls
runtime action behavior
```

The 15-case runtime gate is a final exam, not a kindergarten gate.

## Placement Rules for Existing Checkpoints

Checkpoint status categories:

```text
KEEP: valuable active lineage or warm-start candidate
REFERENCE: useful evidence/teacher but not a direct continuation
QUARANTINE: intact but should not be continued without a special reason
ARCHIVE/DELETE CANDIDATE: obsolete intermediate once backed up
```

No checkpoint inspected here was physically corrupted. All `.pt` files loaded.

## Checkpoint Audit

### Best Current Soul-Alive Salvage Candidate

`cv2copy_128_stage9_schema_mix_acc4_001200.pt`

Status: KEEP / best old 128 candidate for soul-first salvage.

Why:

- 128D, 1 head, 2 layers, FFN 16384, 32 soul rows.
- Strong old trainer metrics: `draft_visible_acc=0.9617`, `changed_visible_acc=0.9676`, repeat about `0.1346`.
- Previously passed simple local runtime core suite 4/4, though sigils/repeat were weak.
- Soul gates are alive: layer gates about `0.0357` and `0.0698`, reflect gate about `0.2191`.
- Soul influence diagnostic showed real field effect:

```text
random soul -> field MSE 0.1704
carried soul -> field MSE 0.3047
next soul delta RMS 16.12
row diversity 0.0924
```

Risk:

- It came from old schema/sigil curriculum, not the ordered school ladder.
- It is not trustworthy as a final runtime core.

Use:

- Best candidate if we want to continue from a body whose soul path is still live.
- Should take placement exams for Pre-K through Grade 4 before further scaling.

### Strong Reference Candidates

`cv2copy_128_stage6_schema_acc4_001600.pt`

Status: REFERENCE / possible warm-start.

Why:

- Very strong draft/change metrics from old curriculum.
- Soul path active and diverse:

```text
random soul field MSE 0.0050
carried soul field MSE 0.0601
row diversity 0.4572
```

Use:

- Keep as a comparison point for soul diversity and schema lessons.

`cv2copy_128_stage2_full48_acc4_002000.pt`

Status: REFERENCE.

Why:

- Excellent old copy/delta metrics.
- Soul influence exists.

Risk:

- Earlier and likely less balanced than Stage 9.

### Big Body, Muted Soul

`state256_2h2l_ffn32768_all_018000.pt`

Status: KEEP as large state/delta body; not a soul-ready runtime core.

Why:

- 256D, 2 heads, 2 layers, FFN 32768, about 55.3M params.
- Strong state preservation: `preserved_visible_acc=0.9749`.
- Stage replays showed it began learning changed rows around `chg_vis` 0.12-0.20 on delta stages.

Problem:

- Runtime assistant gate failed 0/15.
- Soul cross gates nearly collapsed:

```text
layers.0.soul_cross_gate ~= 0.000228
layers.1.soul_cross_gate ~= 0.0000268
random/carry soul field effect ~0
row diversity ~0.000039
```

Use:

- Good evidence that the copy/delta ladder works for state preservation.
- Could be repaired only with explicit soul-causal schooling that revives the soul path.
- Do not deploy as a responder.

### Kindergarten Artifact

`statecopy64_stage1_003000.pt`

Status: KEEP as Kindergarten proof/reference.

Why:

- Strong exact copy/preservation checkpoint.
- Useful baseline for "full-state copy works."

Problem:

- Soul path is effectively muted.
- Not a responder.

Use:

- Reference for Grade K gates, not for direct continuation unless we only want copy schooling.

### Tiny 32D Lineage

`32_001200.pt`

Status: KEEP while the local CPU run is still relevant; final status depends on later checkpoint/output.

Why:

- 32D, 1 head, 2 layers, FFN 16384.
- Tiny "narrow body, huge FFN" experiment.
- Soul path has small but measurable influence:

```text
random soul field MSE ~0.00126
carried soul field MSE ~0.00554
```

Risk:

- Current saved checkpoint is early and not enough for final judgment.
- Metrics are mixed, and repeat was high in some metadata.

`v7_001200.pt`

Status: REFERENCE/possible duplicate-ish early 32D lineage.

Note:

- Same architecture and core state pattern as `32_001200.pt`, but metadata differs.
- Keep until the live 32D CPU run is resolved.

`v7_000100.pt`

Status: ARCHIVE/DELETE CANDIDATE after backup.

Why:

- Early intermediate in the 32D path.
- Superseded by later 32D checkpoints.

### Old 64D Full Curriculum

`v7_core64_fullcurr_gpu_resume_006000.pt`

Status: QUARANTINE.

Why:

- It learned some natural language shape.
- But exact runtime eval failed hard.
- Soul gates collapsed/negative:

```text
layers.0.soul_cross_gate ~= -0.000446
layers.1.soul_cross_gate ~= -0.000465
field effect ~0
```

Use:

- Keep only as evidence/training trace.
- Do not continue under the soul-first plan unless there is a specific experiment.

`v7_000500.pt` and `v7_001000.pt`

Status: ARCHIVE/DELETE CANDIDATES after backup.

Why:

- Obsolete intermediates from the old 64D path.
- Later 64D full-curriculum result is already quarantined.

### Old 128 Overfit / Smoke Lineage

`act_reflect_v2_128_stage128_005500.pt`

Status: REFERENCE.

Why:

- Intact 128D act_reflect_v2 checkpoint.
- Soul gates alive and random soul affects output.

Risk:

- Weak draft/change metrics relative to later 128 schema checkpoints.

`128_001200.pt`

Status: REFERENCE / likely not a direct continuation.

Why:

- 128D checkpoint with decent draft metric but no changed-row learning in current metadata.
- Soul random influence exists, but carried-zero influence is tiny.

Risk:

- Looks related to the earlier 128 overfit lineage and is superseded by stronger schema checkpoints.

## Recommended Next Move

Do not delete immediately. First create an archive folder and move only after Jeff approves.

Suggested active set:

```text
KEEP active:
  cv2copy_128_stage9_schema_mix_acc4_001200.pt
  state256_2h2l_ffn32768_all_018000.pt
  statecopy64_stage1_003000.pt
  32_001200.pt

REFERENCE:
  cv2copy_128_stage6_schema_acc4_001600.pt
  cv2copy_128_stage2_full48_acc4_002000.pt
  act_reflect_v2_128_stage128_005500.pt
  128_001200.pt
  v7_001200.pt

QUARANTINE:
  v7_core64_fullcurr_gpu_resume_006000.pt

ARCHIVE/DELETE CANDIDATES:
  v7_000100.pt
  v7_000500.pt
  v7_001000.pt
```

For the new soul-first era, the best practical restart point is probably:

```text
cv2copy_128_stage9_schema_mix_acc4_001200.pt
```

because it is not the biggest, but its soul pathway is still alive. The 256D checkpoint has more capacity, but it learned to mute the soul, so it would need repair before it can become the kind of conscious/continuous core Jeff is aiming for.

## Post-Audit Quarantine Action

Date: 2026-06-13

Jeff approved quarantine after the audit. Active `D:\axon7\checkpoints` was reduced to one selected soul-first continuation candidate:

```text
cv2copy_128_stage9_schema_mix_acc4_001200.pt
```

The active folder still keeps the small config JSONs used for creating new cores.

All other `.pt` checkpoints were moved, not deleted, into:

```text
D:\axon7\checkpoints_quarantine\2026-06-13_soul_first_checkpoint_audit
```

That quarantine folder contains `manifest.md` and `manifest.json` explaining each moved file. The 256D checkpoint is preserved there as valuable reference evidence, but it is no longer active because its soul path tested effectively muted.
