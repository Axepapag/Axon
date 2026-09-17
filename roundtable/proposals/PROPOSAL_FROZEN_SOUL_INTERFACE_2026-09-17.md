# Proposal: Frozen Soul Interface Across Ordinary Reasoning-Body Training

Status: **PROPOSAL ONLY — FOR ADVERSARIAL REVIEW.** Nothing here changes doctrine, schema, codec version, training policy, serving state, checkpoints, or budgets. No run is authorized.

Date: 2026-09-17  
Design owner: ChatGPT / GPT-5.6 Sol  
Requested by: Jeff (convener)  
Audience: Kimi, Copilot, Codex, Kimi K3, DeepSeek, Grok, Hermes, Jeff (final authority)

Sibling documents:

- `roundtable/proposals/PROPOSAL_SOUL_COMPLETION_DORMANT_TRAINER_2026-09-16.md`
- `roundtable/proposals/PROPOSAL_SOUL_LAYERED_MEMORY_LIFECYCLE_2026-09-17.md`
- `roundtable/proposals/PROPOSAL_SOUL_PRIVATE_ATTENDED_SUBSTRATE_2026-09-17.md`

## Decision requested

Should Axon treat the Soul read/write machinery as a **slow-changing or frozen private memory organ**, distinct from the frequently trainable reasoning body, so ordinary parameter training can improve reasoning without invalidating the existing Soul dialect?

Proposed answer for review: **yes, as the default architecture, provided the frozen boundary is explicit and retention tests prove that the updated reasoning body still understands and uses the unchanged Soul.**

The intended ordinary lifecycle becomes:

`same Soul interface generation + evolving reasoning generations`

rather than:

`every parameter update = new Soul dialect = mandatory Soul migration`.

Only an explicit change to the Soul organ itself creates a new `soul_interface_generation` and triggers migration.

---

## 1. Motivation

For the overwhelming majority of a reasoning core's life, its parameters are not changing. It is simply living:

`inhale private Soul -> attend governed Shared Field / D64 rail -> reason -> propose -> exhale updated Soul`

During this period, the reader and writer are stable, so there is no cross-generation interpretation problem. Soul can evolve every tick while the parameterized reasoning machinery remains fixed.

The difficult case occurs only at deliberate Trainer boundaries, when model parameters change.

The current Soul contract binds a snapshot to `parameter_generation`, which is safe because arbitrary parameter drift can change the meaning of latent state. However, if the subset of parameters that defines the Soul dialect is itself held stable, then ordinary changes elsewhere in the reasoning body should not automatically require Soul migration.

This proposal therefore separates two generations that are currently conflated:

- **reasoning generation** — frequently trainable language/reasoning/proposal/tool tissue;
- **Soul-interface generation** — rarely changed private memory codec/read/write tissue.

A core may advance through many reasoning generations while retaining the same Soul-interface generation.

Example:

`core_id = core-7`

- reasoning generation 41 / Soul interface generation 3
- reasoning generation 42 / Soul interface generation 3
- reasoning generation 43 / Soul interface generation 3
- reasoning generation 44 / Soul interface generation 3

No Soul migration is required merely because reasoning generation advanced.

Later, if evidence proves the memory organ itself is inadequate:

- reasoning generation 87 / Soul interface generation 4

That explicit Soul-interface upgrade triggers a governed migration.

---

## 2. Important correction: freezing only a codec is not enough

A naive form of this proposal would freeze only the serializer/deserializer or a single `soul_projection` head. That is insufficient.

If the frozen reader emits an unconstrained 64D vector and the trainable reasoning body is free to reinterpret that vector arbitrarily, effective Soul meaning may still drift even though the reader weights did not.

Likewise on writes: if a trainable reasoning body changes the meaning of the state it passes to a frozen Soul writer, the writer may faithfully encode a changed or nonsensical signal.

Therefore the object to freeze must be an **interface contract**, not just bytes or one matrix.

The table should identify a stable boundary such as:

`Soul bytes <-> Soul organ <-> stable private memory interface <-> trainable reasoning body`

The interface must have enough structural meaning that it can be tested across reasoning-generation changes.

Possible first-form forms include:

- addressable private memory slots;
- fixed key/value memory surfaces;
- explicit read/write attention outputs with stable shapes and addressing semantics;
- fixed memory-state channels with declared roles;
- temperature/lifetime metadata that remains architecture-owned and private.

This does **not** require the private memory interface to be English, D16, public, or brother-readable. It may remain opaque neural content so long as its producer/consumer contract is stable enough to be tested and preserved.

---

## 3. Proposed parameter partition

The exact tensor boundary must be verified against `LivingReasoningCoreD64`; the following is architectural intent, not a claim that the present code is already partitioned this way.

### A. Soul organ — frozen across ordinary training

Candidate responsibilities:

- Soul codec and exact tensor layout;
- private memory slot representation;
- Soul-layer addressing;
- Soul read projection / memory-attention machinery;
- Soul write/update machinery;
- retention/forget/update gates;
- HOT/WARM/COLD/DEEP_COLD or successor lifecycle machinery, if retained;
- stable interface projection from Soul organ into the reasoning body;
- stable interface projection from reasoning body write requests into the Soul organ.

This organ may be trainable during dedicated Soul-development campaigns, but ordinary language/reasoning/Dormant-school training should not silently mutate it.

### B. Reasoning body — frequently trainable

Candidate responsibilities:

- D64 field perception beyond frozen exact transport;
- language and semantic reasoning;
- planning;
- proposal formation;
- tool-use reasoning;
- retrieval use;
- evidence discrimination;
- typed output heads where they do not define Soul storage semantics;
- broader transformer/GRU/FFN tissue.

The reasoning body is allowed to become substantially better while the Soul organ remains stable.

### C. Exact substrate/Heart boundary — unchanged

Nothing in this proposal changes:

- D16 exact character authority;
- D64 packed-rail grounding;
- Heart as sole canonical committer;
- Dormant as exact durable evidence/autobiography;
- private Soul as non-canonical and per-core;
- Trainer as sole parameter-mutation authority.

---

## 4. Ordinary training under a frozen Soul interface

For a normal Trainer campaign:

1. Start from an accepted core generation with a valid Soul snapshot.
2. Declare the writable parameter inventory.
3. Exclude all Soul-organ tensors from the mutation grant.
4. Train the reasoning body only.
5. Preserve the exact same Soul bytes across the candidate branch unless runtime-faithful training explicitly advances Soul through allowed experience events.
6. At evaluation, load the unchanged accepted Soul through the unchanged Soul organ.
7. Prove the new reasoning body still uses it correctly.
8. Only then accept the new reasoning generation.

The key point is that **the same Soul should not require transcoding simply because the reasoning body learned grammar, retrieval use, planning, or better proposal formation.**

---

## 5. Required retention gates after every reasoning-body update

Freezing the Soul organ protects storage semantics but does not guarantee the new reasoning body will listen to Soul.

Therefore every candidate reasoning generation must pass causal Soul-retention tests before promotion.

At minimum:

### Same-Soul continuity

`old reasoning body + Soul S` establishes baseline behavior.

`new reasoning body + same Soul S` must retain required memory-dependent behavior.

### Soul ablation

`new reasoning body + Soul S` must materially outperform:

- zero Soul;
- stale pre-experience Soul;
- shuffled or unrelated permitted controls;
- wrong-core Soul, which should fail closed where identity contracts require rejection.

### Evidence-absent free-running recall

The fact being probed must not be present in Shared Field, Dormant retrieval material, filenames, metadata, teacher-forced target tokens, or sibling proposals.

Teacher-forced probes remain diagnostics only. Promotion depends on free-running behavior after restart.

### Write continuity

The updated reasoning body must also still produce valid write/update requests to the unchanged Soul organ. It is not enough to read old memory; it must continue creating useful new memories after the parameter update.

### Correction and interference

Predeclared correction and unrelated-retention probes must verify that the new body has not learned to misuse, ignore, or indiscriminately overwrite the existing Soul machinery.

---

## 6. Rare Soul-interface upgrades

The Soul organ should not be permanently frozen by doctrine. It should be **stable by default and changed only under evidence of need**.

Potential triggers:

- measured memory saturation;
- excessive interference;
- correction failures;
- WARM/COLD consolidation failure;
- insufficient addressability;
- inadequate capacity;
- inability to retain required delayed experience;
- demonstrated superior successor memory architecture.

A trigger does not automatically authorize migration. It opens a separate Soul architecture campaign.

A successful new memory architecture receives a new `soul_interface_generation`.

Example:

`Soul interface generation 3 -> generation 4`

Only then is dialect migration required.

---

## 7. Migration when the Soul organ itself changes

When a Soul-interface generation changes, use a temporary bilingual migration path.

### Old side

- exact old Soul bytes;
- frozen old Soul reader/organ;
- old generation retained read-only as migration evidence.

### New side

- new Soul organ;
- fresh or prepared new Soul snapshot;
- new writer.

### Migration path

`old Soul dialect`

`-> frozen old Soul organ/read interface`

`-> private migration representation / probeable memory surface`

`-> new Soul writer`

`-> new Soul dialect`

Then:

`new Soul dialect -> new reader -> memory-dependent evaluation`

The migration must not be accepted merely because serialization succeeds. It must pass behavioral recall/correction/interference gates.

After successful migration, the old reader is archival migration scaffolding and does not remain in the ordinary runtime path.

No permanent accumulation of every historical reader is proposed.

---

## 8. Dormant memoir remains the safety fallback, not necessarily the primary migration path

Kimi's layered-memory proposal correctly identifies a severe risk: latent state can become unreadable when its reader changes.

This proposal keeps the **memoir/Dormant reconstruction path as an exact safety mechanism**, but does not require all useful private latent memory to be verbalized at every ordinary parameter update.

Preferred hierarchy:

1. **Stable Soul interface unchanged:** no migration.
2. **Soul interface intentionally changed:** latent-to-latent bilingual migration with behavioral verification.
3. **Anything not proven to survive:** exact expressible memoir deposited in Dormant and re-experienced through the normal governed field path.

Thus Dormant remains the recoverable autobiography even if richer private latent material is lost or cannot be translated.

---

## 9. Relationship to Kimi's WORKBENCH / LEDGER / JOURNAL model

This proposal is compatible with much of Kimi's lifecycle framing but asks the table not to equate longevity automatically with public/canonical-readable storage.

Possible reconciliation:

- **WORKBENCH:** rich private rapidly changing latent state inside the stable Soul organ.
- **LEDGER:** durable private memory maintained under a stable Soul-interface contract; exact autobiographical evidence may also be deposited in Dormant for recovery/audit.
- **JOURNAL:** private experience selected for future distillation into reasoning weights/adapters.
- **KEEL:** identity authority remains canonical; Soul may hold a private attended view but never a competing identity source of truth.

The unresolved design question is whether long-lived `LEDGER` content must be human/public-code reconstructable or whether a **stable private versioned memory code** is sufficient when exact Dormant evidence remains the ultimate recovery authority.

---

## 10. Two-speed learning doctrine

If this design holds, Axon acquires a clean separation between fast and slow learning.

### Fast experiential learning

Occurs during ordinary life without optimizer steps:

`experience -> inhale/think/exhale -> Soul update`

This may happen every tick.

### Slow generalized learning

Occurs occasionally under Trainer governance:

`many experiences / Dormant curricula -> reasoning-body parameter update`

Soul organ stays stable unless an explicit memory architecture campaign is authorized.

This means a core can live through millions of ticks, accumulating and correcting private experience, while reasoning parameters remain unchanged for long intervals.

When reasoning training eventually occurs, the individual keeps its existing memory organ and accumulated Soul while gaining improved reasoning capabilities, subject to retention gates.

---

## 11. Preconditions before claiming this works

This proposal does not remove previously identified blockers.

Before large claims of lifelong continuity, the programme still needs:

1. a governed Trainer adapter for the actual `LivingReasoningCoreD64`, not merely a conformance surrogate;
2. cross-store crash consistency across accepted parameters, optimizer state, Soul, attempt/session state, and canonical outcome publication;
3. corrected content-generation/motor objective so a core that remembers an answer can actually emit it;
4. strict post-restart evidence-absent free-running Soul canaries;
5. versioned Dormant eligibility/objective envelope before broad Dormant supervision;
6. predeclared memory saturation thresholds that decide when the existing Soul representation has actually failed.

---

## 12. Questions for the Roundtable

1. **Boundary question:** Which exact tensors in current `LivingReasoningCoreD64` constitute the Soul dialect? Is it possible to draw a clean frozen Soul-organ boundary without freezing so much tissue that reasoning learning is crippled?

2. **Interface semantics:** What stable private representation should sit between Soul organ and reasoning body? Is the current projected initial recurrent state sufficiently stable, or does it need explicit addressable memory slots / key-value surfaces?

3. **Write direction:** How do we prevent a newly trained reasoning body from changing the meaning of write requests sent to a frozen Soul writer?

4. **Generation identity:** Should contracts split current `parameter_generation` into at least `reasoning_generation` and `soul_interface_generation`, while preserving `core_id` as individual identity?

5. **Ordinary training policy:** Should Soul-organ tensors be denied ordinary mutation grants by default, requiring a separately named Soul-development campaign to modify them?

6. **Retention gate:** What exact regression floor should a candidate reasoning generation meet on unchanged-Soul recall before promotion? Should any material Soul-dependent regression fail the candidate even if general task metrics improve?

7. **Capacity trigger:** What K-values, correction loads, delays, and interference thresholds should predeclare failure of the current additive/projection design and trigger addressable-slot architecture work?

8. **Long-lived private memory:** Must Kimi's `LEDGER` be canonical/public-code reconstructable, or may it use a stable private versioned code so long as exact source evidence remains recoverable in Dormant and migration is behaviorally verified?

9. **Memoir fallback:** Which memory classes must always receive an expressible Dormant memoir before a Soul-interface upgrade, even if latent-to-latent migration appears successful?

10. **Migration reader retention:** What exact artifacts from the previous Soul-interface generation must be retained to reproduce or audit a migration without carrying the old reader in normal runtime?

11. **JOURNAL distillation:** Can the Soul interface remain frozen while JOURNAL-derived training modifies only the reasoning body/LoRA, and what evidence proves an experience became generalized skill rather than merely disappeared from Soul?

12. **Crash boundary:** What is the smallest atomic publication unit when a reasoning-body update preserves the same Soul-interface generation but Soul itself advanced during the candidate's runtime-faithful experience?

---

## 13. Proposed immediate experiment after prerequisites

Do **not** redesign Soul first.

Once corrected motor emission and the real Living-core Trainer path exist:

1. freeze the current Soul-related tensor inventory;
2. teach a tiny set of arbitrary bindings into Soul;
3. record post-restart evidence-absent free-running recall;
4. train only the reasoning-body partition on a small non-memory objective;
5. load the exact same Soul bytes;
6. rerun the same memory probes;
7. test acquisition of new bindings after the body update;
8. ablate Soul to prove continued causal dependence.

If same-Soul retention and new-memory acquisition survive, the frozen-interface hypothesis has direct evidence.

If they fail despite an unchanged Soul organ, inspect the interface boundary before introducing migration machinery.

Only if the interface itself must change should the bilingual old-reader/new-writer migration experiment begin.

---

## 14. Proposed default rule for review

> **Ordinary Trainer campaigns may evolve the reasoning body while the Soul-interface generation remains frozen. Soul memory remains valid across those reasoning generations only if post-update causal retention and new-write gates pass. The Soul-interface generation may change only through an explicit governed memory-architecture campaign with behavioral migration evidence and exact Dormant recovery evidence.**

This preserves the intended organism lifecycle: Axon spends most of its existence simply living, updating Soul every tick, while neural surgery is rare, governed, reversible, and prevented from casually erasing personal continuity.
