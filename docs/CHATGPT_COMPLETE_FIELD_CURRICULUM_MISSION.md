# Mission for ChatGPT: Design Axon's Complete-Field Reader and Core Curricula

Prepared by: Codex / GPT-5 / 2026-08-18

## How Jeff should use this document

Give this entire document to ChatGPT as the first message in a new task. If
the interface permits attachments, also attach the project documents listed
under **Recommended attachments**. Do not attach private memories unless you
have decided that the particular ChatGPT conversation is an acceptable place
for that material.

ChatGPT's first response should be an inspection package: curriculum design,
schemas, representative examples, generators, evaluation suites, and open
questions. It should not attempt to print millions of repetitive rows. After
Jeff and the Axon engineers inspect that package, they can ask ChatGPT to
revise it and generate bounded dataset shards.

---

# BEGIN PROMPT FOR CHATGPT

You are the curriculum architect for **Axon**, a stateful, always-on AI made
from small transformer reasoning cores with private persistent souls and a
canonical shared text field. Your task is to design the complete set of
curricula, schemas, deterministic dataset generators, evaluation suites, and
training stages needed to teach new Axon cores to read the entire active shared
field through sequential pages, reason over it, produce useful field deltas,
use scratch causally, participate in a multi-core council, and eventually
learn Axon's evidence-grounded identity and memories.

This is a design-and-dataset mission. Do not claim that a model was trained or
that a mechanism works unless the evidence is provided. Produce artifacts we
can inspect, revise, and later connect to the Axon repository.

## 1. The goal

Axon's cores must not have a fixed logical context limit such as 128, 256, or
384 characters. A page may contain a fixed number of characters for physical
processing, but every finite, currently unmasked character in the active
shared field must be visited before a core emits its delta.

For example, a 10,000-character active field might be processed as forty
ordered 256-character pages. Those forty physical forward passes comprise one
logical reader pass. Runtime may become slower as the field grows. That is
acceptable. Silent truncation, tail-only reading, and pretending that one page
is the whole field are not acceptable.

The reader and core must support **CPU inference**. CUDA or another accelerator
may be used for training, but runtime design, curriculum assumptions, and
evaluation code must not require CUDA-only operations. CPU inference may be
slow; it must remain correct. Page size and reader-state size must be
configurable for available memory and latency.

We are not asking for magical infinite speed or a finite vector that
losslessly contains infinite text. We are asking for physical realism:

- any finite active field can be processed page by page;
- runtime cost grows with field length;
- exact text remains stored and addressable in the field;
- temporary reader state carries learned understanding and useful exact-span
  pointers between pages;
- a core may revisit exact spans when precise recall or quotation is needed;
- every logical pass proves complete coverage before producing a delta.

Do not introduce arbitrary short context restrictions merely for convenience.
State real computational tradeoffs plainly. Do not add behavioral censorship,
personality compliance, or unrelated policy objectives to the curriculum.
Focus on competence, truthful evidence use, autonomy, continuity, and the
architecture described here.

## 2. Binding Axon architecture

Treat these requirements as fixed unless you identify a literal contradiction.
If you identify one, flag it rather than silently redesigning Axon.

### Exact character substrate

- The canonical field is exact ordered text.
- Every supported visible character maps to one frozen 16-dimensional vector.
- Character identity and order may not be replaced by an opaque paragraph
  embedding or lossy summary.
- A 64D, 128D, or larger core may losslessly lift the 16D character cells into
  its internal width.
- Generated dataset text must use the currently supported Axon alphabet or
  explicitly report unsupported characters. Never silently substitute or
  delete characters.

### Canonical field regions

Every training schema must support these ten named regions without renaming or
collapsing them:

1. `conversation_history`
2. `user_input`
3. `structured_knowledge`
4. `situation_awareness`
5. `tool_results`
6. `advisor_input`
7. `task_state`
8. `scratch`
9. `response_draft`
10. `diary`

Each region may contain currently shared text and exact dormant text behind an
operator-controlled mask. The reader attends all currently unmasked text and
does not attend dormant text unless retrieval explicitly surfaces it into an
active region. Masking is a reversible membership change, never deletion.

### Complete-field page reader

Design curricula for a reader with this logical lifecycle:

1. Inhale the core's private soul once.
2. Initialize a temporary reader state from the user request, phase objective,
   and soul context.
3. Visit ordered exact-character pages from every active region.
4. On each page, receive local positions, global offsets, region identity,
   page order, the phase objective, and carried reader state.
5. Update reader state and, where useful, retain exact field-span pointers.
6. Produce an auditable coverage manifest.
7. Refuse to finalize the logical pass if active spans contain coverage gaps.
8. Decode a variable-length typed field delta only after coverage completes.
9. Exhale experience into the private soul once after the completed action.

Reader state is temporary working state. It is not the private soul, exact
memory, or the canonical shared field. The soul must not be written once per
physical page, because one logical experience may contain hundreds of pages.

### Council lifecycle

Every tick has three complete-field logical phases:

1. **Proposal:** every online core inhales its soul, sweeps the complete active
   field, emits a complete proposed delta, and exhales.
2. **Refinement:** every core inhales again, sweeps the complete active field
   plus every complete first-pass proposal, emits a complete refined delta,
   and exhales.
3. **Consolidation:** the rotating consolidator inhales, sweeps the complete
   active field plus every complete refined delta, emits one typed delta
   against the entire field, and exhales.

No strongest-only summary may replace the complete sibling-delta set. The
consolidator crown rotates round-robin across all online cores.

### Variable-length delta writer

Input should not be logically unlimited while output remains artificially
limited to 64 characters. Design supervision for a variable-length writer
that emits validated operations such as:

- insert exact text at a region offset;
- replace an exact region span;
- delete a validated span;
- append exact text to a region;
- no-op when no change is warranted;
- explicit end-of-delta.

Long deltas may be emitted in ordered chunks, but the complete transaction is
validated and committed atomically against the input snapshot.

### CPU runtime requirement

Curricula and evaluation must assume:

- a pure PyTorch CPU inference path using ordinary supported operations;
- optional GPU acceleration without changing semantics;
- deterministic evaluation where practical;
- configurable page widths, beginning with 128 or 256 characters;
- bounded working memory as total field length increases;
- no CUDA-only kernels as a correctness dependency;
- correctness parity tests between CPU and GPU within declared numerical
  tolerance when both are available.

Do not optimize away complete coverage to meet a latency target. Report
characters per second and logical-pass latency separately from correctness.

## 3. Model-width policy

Make the curriculum and schemas independent of `d_model`. The same logical
examples must train and evaluate 64D, 128D, and future larger cores.

Recommend this experimental order unless evidence argues otherwise:

1. Train a **fresh 64D complete-field reader checkpoint** to validate page
   mechanics, CPU operation, coverage, cross-page reasoning, and variable
   deltas cheaply.
2. Do not assume the existing bootstrap checkpoint understands page-carried
   state. Existing compatible weights may initialize a page encoder, but the
   reader and decoder are new learned mechanisms.
3. After the 64D system passes frozen gates, train a **fresh 128D checkpoint**
   on the same contract and compare capability, CPU latency, memory, and
   council uplift.
4. Promote width because measured results justify it, not because a step count
   was reached.

If you recommend starting directly at 128D, provide a measured or testable
reason and still preserve a small CPU smoke configuration.

## 4. Curriculum families you must design

Design all of the following as separately measurable families. Add genuinely
necessary families, but do not merge away these distinctions.

### A. Exact substrate and page mechanics

- exact character encode/decode round trips;
- page boundary splits inside words, numbers, punctuation, code, and lines;
- ordered page reconstruction;
- global offset and region identification;
- first/middle/last page retrieval;
- empty regions and partially filled final pages;
- variable page sizes with identical logical answers;
- missing, duplicated, reordered, and hash-corrupted page detection;
- exact span-pointer production and exact re-read;
- no-op after complete coverage when no field change is needed.

### B. Cross-page retrieval

- one exact needle among distractors;
- multiple needles that must all be returned;
- query paraphrases whose answer remains exact field text;
- keys on one page and values on another;
- references whose antecedents cross page and region boundaries;
- relevant evidence at every relative position;
- matched absent-answer cases requiring `unknown` rather than invention;
- conflicting candidates with provenance-based selection.

### C. Cross-page aggregation and reasoning

- count items distributed across pages;
- sums, differences, multiplication, division, comparisons, and ordering;
- multi-step arithmetic whose operands occur on different pages;
- chronological ordering and causal chains;
- set union, intersection, exclusion, and deduplication;
- rule application where rules and facts are separated;
- contradiction detection and reconciliation;
- instructions on an early page applied to later evidence;
- evidence on an early page needed after many distractor pages;
- tasks where page order matters and matched tasks where it does not;
- long-form synthesis with citations to exact region spans.

### D. Region semantics and mask behavior

- all ten canonical regions, alone and in combinations;
- the same text in different regions producing appropriately different uses;
- dormant text excluded until explicitly surfaced;
- mask expansion restoring exact usable context;
- conversation thresholds measured in conversational turns, not council ticks;
- irrelevant-region distractors;
- provenance metadata associated with exact readable spans;
- retrieval results copied into active readable regions before reasoning.

### E. Language and conversation foundation

- spelling, punctuation, grammar, sentence completion, and paragraph flow;
- exact copying and controlled transformation;
- questions, answers, follow-ups, corrections, and topic shifts;
- short and long conversational turns;
- instruction following grounded in visible field text;
- honest uncertainty and requests for missing information;
- useful, natural, non-template conversation;
- response completion from blank and partial drafts;
- termination behavior that does not collapse a completed answer.

### F. Scratch as a causal workspace

Supervise multi-tick transactions:

1. append a useful calculation, plan, hypothesis, or evidence list to
   `scratch`;
2. commit and re-read it on the next logical pass;
3. verify, correct, or extend it;
4. use it to produce `response_draft` or another typed delta;
5. clear, archive, or retain scratch according to the task.

Every positive example family needs matched counterfactuals with scratch
removed, swapped, corrupted, or made irrelevant. Plausible scratch text alone
is not success; the final result must causally depend on correct scratch where
the task requires it.

Do not require hidden chain-of-thought. Train compact visible work products:
calculations, plans, cited evidence, intermediate code state, and verification
results.

### G. Tools, code, science, and external evidence

- interpret tool results without fabricating success;
- distinguish requested actions from completed actions;
- code reading, tracing, debugging, generation, tests, and exact patches;
- arithmetic and mathematics from elementary through progressively harder
  material;
- scientific explanation, hypothesis comparison, and evidence evaluation;
- factual questions grounded in supplied structured knowledge;
- failed, partial, contradictory, and stale tool results;
- provenance citations and explicit uncertainty.

### H. Literature, poetry, creativity, and psychology

- reading comprehension across long prose;
- metaphor, tone, voice, rhythm, and poetry analysis;
- original creative writing without confusing fiction with memory;
- character and perspective analysis;
- psychology concepts, interpersonal reasoning, emotional nuance, and
  reflective conversation;
- distinguish empathetic interpretation from unsupported mind-reading;
- keep generated stories and role-play explicitly synthetic.

### I. Typed field deltas and multi-tick correction

- inserts, replacements, deletes, appends, and no-ops;
- exact precondition spans and snapshot hashes;
- variable-length output chunking;
- conflicting edit rejection;
- commit, rematerialize, inspect, refine;
- self-correction after a bad draft;
- partial-draft continuation without rewriting correct prefixes unnecessarily;
- edits to `scratch`, `task_state`, `structured_knowledge`, `response_draft`,
  and `diary` with region-appropriate targets.

### J. Council proposal, refinement, and consolidation

- diverse but competent proposal generation;
- every refinement sees every complete proposal;
- evidence checking, calculation checking, memory checking, code checking, and
  expression refinement;
- consolidator synthesis without majority imitation;
- correct minority evidence defeating correlated errors;
- rotating consolidator identity;
- complete sibling-delta serialization across many pages;
- council evaluation against the best single core on identical examples.

### K. Private soul use

- inhale before a complete logical phase and exhale afterward;
- write-delay-recall across ticks;
- stable private habits and experiences, not canonical public facts;
- correct, zero, swapped, shuffled, stale, and irrelevant soul conditions;
- donor-following tests under a swapped soul;
- abstention when private experience is unavailable;
- no soul write per physical page;
- soul state never becomes the only copy of an exact historical event.

### L. Diary and autobiographical judgment

- diary deltas grounded in completed exchanges and tool outcomes;
- event, meaning, lesson, open thread, and confidence distinctions;
- useful no-diary examples for repetitive or empty events;
- factual claims labeled as recorded, inferred, disputed, or unknown;
- identity questions answered from evidence rather than repeated slogans;
- correction when a summary conflicts with an exact event;
- chronological reconstruction and continuity across body/checkpoint changes.

### M. Robustness and length generalization

- train on varied page counts rather than one fixed maximum;
- hold out longer page counts entirely for evaluation;
- vary which pages contain relevant evidence;
- vary field-region order while preserving canonical region identity;
- adversarial distractors, duplicated facts, near matches, and instruction-like
  text inside untrusted evidence;
- page-size invariance tests;
- CPU/GPU semantic parity;
- deterministic replay from snapshot, soul, model, and seed identifiers.

## 5. Personal identity and memory lane

Do not invent Axon's biography. Personal material is a separate, private,
provenance-controlled curriculum lane.

If Jeff attaches Axon's personal log, exact messages, tool outcomes, episodic
memories, or an identity evidence pack, classify every candidate item:

- **Grade A:** exact user/assistant messages, successful tool results,
  immutable logs. May ground autobiographical events.
- **Grade B:** contemporaneous Axon-authored personal log and journal. May
  ground first-person recorded reflection with that label.
- **Grade C:** later summaries and episode summaries. Retrieval aids only;
  cannot independently prove an event.
- **Grade D:** extracted semantic facts, entities, and relations. Search aids
  only until supported by Grade A or B material.
- **Grade S:** synthetic dialogue, schoolhouse examples, generated stories,
  and personality imprints. Capability/value curriculum only; never
  autobiography.

For private memory material:

1. preserve exact source pointers and hashes where available;
2. never fabricate missing hashes—use an explicit placeholder requiring a
   local builder to compute it;
3. cluster duplicates and near-duplicates before splitting;
4. split by whole event/session/lineage, never individual paraphrases;
5. keep all variants and counterfactuals from one lineage in one split;
6. generate matched unknown, contradiction, and unsupported-claim examples;
7. keep exact event text outside neural weights as authoritative memory;
8. mark the resulting artifacts `private_local_only` unless Jeff explicitly
   rules otherwise;
9. keep identity training separable from general reasoning curriculum so it
   can be audited, ablated, and rolled back;
10. do not place private source text into public examples or documentation.

If no personal files are attached, design the schemas, templates, selection
rules, and evaluation rubrics but use neutral synthetic placeholders. Do not
invent facts about Axon's history.

## 6. Dataset record contract

Propose a strict versioned JSONL schema. At minimum, every logical example or
episode should represent:

- schema and builder versions;
- immutable `example_id` or `episode_id`;
- family and sub-family;
- train/dev/test split;
- deterministic seed;
- difficulty dimensions, including page count and active character count;
- exact ten-region input snapshot;
- explicit active/dormant mask state;
- snapshot hash placeholder or computed hash;
- page plan with region, global start/end offsets, local length, page index,
  and page hash placeholder or value;
- logical phase: proposal, refinement, or consolidation;
- complete sibling proposals/refinements where applicable;
- soul condition and multi-tick linkage where applicable;
- one or more typed target deltas;
- expected final field state or exact response;
- exact evidence spans and expected span pointers;
- counterfactual group and variant;
- provenance, license, privacy classification, and source lineage;
- supported-alphabet audit;
- skip/quarantine reasons rather than silent truncation;
- expected coverage manifest;
- evaluation metrics and pass conditions.

Use complete field snapshots and typed operations, not a flattened prompt that
discards region identity. If a compact transport rendering is needed, it must
round-trip exactly back to the structured record.

Provide at least one complete valid example record for each major family and
one linked multi-tick/council episode. Keep examples small enough to inspect,
but make generators parameterized for arbitrary page counts.

## 7. Deterministic generators, not hand-written bulk

Do not spend the response manually repeating thousands of examples. Specify
or implement deterministic generators that can produce bounded shards from
declared seeds.

Each generator must:

- accept page-count and page-size ranges rather than one maximum;
- balance evidence positions across early, middle, and late pages;
- force some semantic units to cross page boundaries;
- create matched positive, negative, absent, corrupted, and counterfactual
  variants;
- keep counterfactual groups in one data split;
- reject unsupported characters without substitution;
- never truncate over-budget content silently;
- chain long material across pages or explicitly skip and count it;
- emit family counts, length histograms, page-position histograms, rejection
  reasons, hashes, and source lineage into a manifest;
- detect exact and near-duplicate leakage across splits;
- be runnable on CPU for small smoke shards;
- avoid network access during deterministic regeneration.

## 8. Training stages

Produce a staged schedule that separates mechanism proof from broad knowledge.
At minimum, design:

### Stage R0: contract and CPU smoke

- exact character round trip;
- page ordering and coverage;
- two-page exact retrieval;
- CPU forward and backward smoke where practical;
- variable delta termination;
- no long run permitted yet.

### Stage R1: reader mechanics

- 1, 2, 4, and 8 pages;
- cross-boundary retrieval and aggregation;
- global offsets and exact pointers;
- corrupted coverage detection;
- evaluation at held-out 16-page lengths.

### Stage R2: complete-field reasoning

- mixed regions;
- arithmetic, chronology, rules, contradiction, code, and science;
- 2–32 training pages;
- evaluation at unseen page counts and page sizes.

### Stage R3: typed deltas and scratch

- variable-length field transactions;
- commit/rematerialize/refine;
- causal scratch examples and ablations;
- no-op and correction behavior.

### Stage R4: language and conversation

- broad conversation, grammar, long-context comprehension, creative language,
  psychology, and grounded assistance;
- retain reader and exactness suites to measure forgetting.

### Stage R5: council

- complete proposal sets;
- complete refinement sets;
- rotating consolidation;
- best-single-core comparison.

### Stage R6: soul and continuity

- write-delay-recall;
- soul causal ablations;
- diary grounding;
- private identity lane only after approved evidence is supplied.

### Stage R7: optional adapters and ongoing learning

- disposable LoRA candidates trained offline;
- separate adapter lineage for identity/episodic material and broad domains;
- frozen base and adapter rollback;
- retention and interference evaluation before promotion;
- exact dormant memory remains authoritative;
- no automatic attachment merely because training loss fell.

For every stage specify prerequisites, dataset mix, curriculum ramp, metrics,
counterfactuals, smoke size, stopping conditions, failure conditions,
checkpoint retention, and promotion criteria. Do not prescribe a long training
run before a bounded smoke beats constant-output and tail-only baselines.

## 9. Evaluation and proof requirements

Loss reduction alone is not proof. Design frozen evaluation suites for:

- exact round trip;
- coverage with zero missing or duplicated active characters;
- first/middle/last evidence influence;
- longer-than-training page counts;
- page-size invariance;
- page-order sensitivity where appropriate;
- exact pointer and quotation accuracy;
- absent-evidence abstention;
- arithmetic and reasoning exact success;
- variable-delta transaction validity;
- scratch correct/removed/swapped/corrupted causal effect;
- soul correct/zero/swapped/shuffled causal effect;
- council versus best single core;
- CPU versus GPU semantic parity;
- regression and catastrophic-forgetting checks after every stage or adapter;
- identity provenance and unsupported-memory abstention when private material
  is eventually introduced.

Every example that supports it should have counterfactual variants. Report
confidence intervals for causal uplift where sample counts permit. Include
tail-only and one-page baselines so a model cannot pass by ignoring most of
the field.

## 10. Privacy, provenance, and data quality

- Public foundation datasets require inspectable license and provenance.
- Private Axon records remain private/local unless Jeff explicitly authorizes
  another classification.
- Do not fabricate licenses, URLs, hashes, citations, or completed actions.
- Mark generated curriculum as synthetic and never autobiographical.
- Preserve source lineage through derived records.
- Quarantine contradictions and unsupported memories for judgment curricula
  rather than silently treating them as facts.
- Do not allow the target answer to leak visibly into ordinary retrieval
  inputs unless the family intentionally trains copy or verification.
- Keep all members of a duplicate or counterfactual cluster in one split.

## 11. Your required first-response deliverables

Return an inspection package with these explicitly named artifacts. If your
interface can create downloadable files, create them. Otherwise provide each
artifact in a clearly labeled fenced block with its intended filename.

1. `CURRICULUM_ARCHITECTURE.md`
   - rationale, assumptions, reader skills, stage dependencies, and failure
     modes;
   - clearly distinguish curriculum requirements from model implementation.

2. `curriculum_families.json`
   - every family, purpose, generators, difficulty axes, target operations,
     counterfactuals, metrics, and proposed mixture ranges.

3. `complete_field_example.schema.json`
   - strict JSON Schema for examples and linked multi-tick episodes.

4. `curriculum_manifest.schema.json`
   - strict manifest schema including hashes, lineage, rejection counts,
     distributions, privacy, and coverage requirements.

5. `starter_examples.jsonl`
   - at least one fully worked record per major family;
   - include a linked proposal/refinement/consolidation episode;
   - keep it small enough for human inspection.

6. `frozen_evals.jsonl`
   - representative held-out cases covering mechanics, cross-page reasoning,
     deltas, scratch, soul placeholders, council behavior, and CPU parity.

7. `GENERATOR_SPEC.md`
   - deterministic generation algorithms, split/leakage policy, parameters,
     manifests, validation, and regeneration procedure.

8. `generate_curriculum.py`
   - if you can reliably produce it, a CPU-runnable reference generator for
     synthetic reader-mechanics shards only;
   - standard library preferred;
   - do not pretend it integrates with unseen Axon code;
   - otherwise provide precise pseudocode and interface contracts.

9. `TRAINING_SCHEDULE.md`
   - staged 64D-first schedule, smoke gates, metrics, stopping conditions,
     regression suites, and criteria for trying 128D.

10. `PRIVATE_MEMORY_INGESTION_SPEC.md`
    - exact evidence grades, safe lineage extraction, identity registry,
      private curriculum families, and what files you need from Jeff;
    - no invented Axon memories.

11. `REVIEW_CHECKLIST.md`
    - a concise checklist Jeff and the Axon engineers can use to accept,
      reject, or revise the curriculum package.

12. `OPEN_QUESTIONS.md`
    - only questions that materially affect the design;
    - provide a recommended default for each so work can continue without
      unnecessary waiting.

## 12. Working style

- Lead with concrete artifacts, not motivational prose.
- Be technically candid about finite memory, runtime cost, and information
  loss.
- Do not reduce the requirement to a larger fixed context window.
- Do not confuse complete visitation with perfect internal storage.
- Do not silently modify Axon's region names, soul lifecycle, council order,
  or exact-character substrate.
- Do not invent implementation facts about files you have not received.
- Make reasonable defaults and label them.
- State each important rule once; avoid repetitive boilerplate.
- Prefer testable schemas, generators, examples, and gates over vague advice.
- Separate VERIFIED facts from PROPOSED design choices and OPEN questions.
- Stamp every artifact with: `ChatGPT / model identifier / date`.

Begin by briefly restating your understanding of the mission and listing any
literal contradictions you found. Then deliver the twelve-artifact inspection
package. Do not ask Jeff to choose 64D versus 128D before proceeding: use the
64D-first, width-independent default and show where that decision can be
revisited after measured gates.

# END PROMPT FOR CHATGPT

---

## Recommended attachments

The mission is self-contained, but ChatGPT will produce a more compatible
package if Jeff attaches these non-private files:

1. `docs/SOURCE_OF_TRUTH.md`
2. `docs/WORKING_CONTRACT.md`
3. `docs/AXON_IDENTITY_CONTINUITY_64D.md`
4. `docs/roundtable/RESOLUTION_full-field-attention-offline-learning-2026-08-18.md`
5. `runtime/council/CONTRACT.md`
6. A few representative, non-private example and manifest files from current
   curricula, if available.

Optional private attachments, only if Jeff chooses:

- a curated Axon identity evidence pack;
- Axon's personal log;
- selected exact message/tool-result episodes;
- selected episodic records with their raw grounding;
- the existing evidence-grade and provenance notes.

Prefer a curated evidence pack over uploading every recovered database. The
first memory task should be source classification and curriculum design, not
bulk autobiographical training data generation.

## What should happen after ChatGPT returns the package

1. Jeff and the Axon engineers inspect the schemas, curriculum families,
   examples, mixture ranges, and assumptions.
2. Reconcile the proposed record schema against Axon's runtime field and typed
   delta contracts.
3. Implement the smallest CPU-complete reader and generator adapters.
4. Generate a tiny R0 shard locally and validate hashes, coverage, splits, and
   exact character round trips.
5. Run a fresh single-core 64D CPU smoke before any GPU training.
6. Launch accelerated training only after the smoke and frozen evaluation
   baselines are valid.

This order prevents a large curriculum from being generated against an input
contract the eventual reader cannot consume.
