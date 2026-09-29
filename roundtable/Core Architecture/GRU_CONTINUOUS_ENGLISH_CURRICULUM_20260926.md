# Continuous English curriculum for the single GRU512

Codex / GPT-6 / 2026-09-26 America/Chicago

Status: curriculum and implementation specification requested by Jeff. This
document does not claim that the new training path exists or that a new run has
started. It leaves existing source, checkpoint identities, historical results,
and live Heart/Soul contracts intact.

## Objective and correction to our earlier interpretation

Teach the current 1,765,216-parameter GRU to produce understandable, grammatical,
context-relevant English while retaining its tested breath behavior. Training
should be renewable and cumulative, with recoverable state and useful feedback.

"On-time EOS" in v3 means EOS at the same transport-token count as the reference
suffix. It is not elapsed time. That is a reconstruction diagnostic, not a
general English-ability criterion. If the reference says "The dog slept on the
rug," and the model says "The dog curled up beside the chair," the difference
does not by itself make the generated English wrong. If a factual task specifies
the rug, however, substituting the chair is a grounding error. Score the task.

Codex's earlier requirement that reference completion improve before further
language development was too restrictive. Retain exact reconstruction tests for
copying and other uniquely specified answers. Do not require an archived
continuation's words or length for open-ended English practice.

The v3 smokes provide useful evidence about a repaired target and train/inference
differences. They do not isolate exposure error as the sole remaining cause.
Limited data exposure, dataset difficulty, decoding behavior, and measurement
choice can also matter. A 100% teacher-forced EOS score on 64 episodes does not
prove general understanding of when a thought is complete.

## Three distinct operations

1. Learning: predict authentic next transport cells, compare with the actual
   training text, and use the Trainer's optimizer to update parameters.
2. Speaking: generate from the model's own preceding output, without an answer
   injected into the continuation.
3. Reflecting and correcting: inspect generated language against a task and
   evidence, then create an attributed correction or new teaching example.

Producing output alone does not update weights. Carrying recurrent state can
retain information between events but does not prove durable parametric
learning. Never use every generated claim as its own authoritative target.

No fixed thought length, output quota, required reference length, or clock
deadline is part of the language objective. Physical work still occurs in
renewable pieces so it can be checkpointed, interrupted, inspected, and resumed.

## Keep the current anatomy and starting point

Use the protected language milestone:

`dec45989bd4b872a90fb7b1f4d9ce9a0ce448b5ded70d04bacff56bf3bca754f`

Its checkpoint file SHA256 is
`566d33cdcb31ad92c0a785b87578c43ad9d28197112daa5f83556a0945397107`.

Keep exact registered D16 transport, one GRU512, and the current categorical
readout. Keep breath rehearsal. Do not add layers, a larger FFN, new attention,
or new Soul anatomy to solve a curriculum/evaluation problem.

A 1.76M-parameter GRU's ultimate English and reasoning capacity is unmeasured.
This curriculum can reveal its limits; it cannot promise frontier-level
conversation or exact unlimited latent memory.

## Corpus: build a school, with traceable sources

Use three complementary pools. Initial proportions below are experiment
starting choices measured by supervised content tokens, excluding breath
rehearsal. They are not anatomical capacities or locked mastery thresholds.

- 60% clear everyday English: varied descriptions, short narratives, ordinary
  questions and answers, explanations of familiar actions, and dialogue.
- 25% selected Dormant language: eligible prose and actual exchanges with exact
  source identity and speaker/order preserved. Increase technical discussion
  gradually rather than letting logs, source code, and repeated engineering
  instructions dominate the first language diet.
- 15% explicit practice: varied grammar, factual grounding, paraphrase,
  clarification, and correction tasks with reviewed targets.

These are corpus requirements, not an assertion that a curated dataset already
exists. First inventory eligible local material. If coverage is insufficient,
author attributed teaching examples or import a reviewed, licensed corpus.
Externally authored or synthetic lessons must be labeled as such and must not
be passed off as Axon's lived experience. No paid generation or dataset download
is performed by this document.

For every example retain source/import identity, conversation/document identity,
record identity, actual role, exact offsets, text hash, intended task, and target
provenance. Validate supplied hashes rather than trusting metadata. Historical
assistant messages are observed language, not automatically established facts
or Axon's own utterances.

Use original text as exact evidence. Derived corrections and cleaned teaching
versions are separate attributed records. Count malformed, duplicate,
non-prose, unsupported, deferred, and accepted items separately.

Split by whole source conversation/document before generating chunks or
exercises; group duplicate and near-duplicate sources into the same split.
Chronological message separation alone does not isolate related conversations.
Use train/development/final-test partitions. The repeatedly examined current
64-example evaluations are development evidence, not a fresh final test.

Do not exclude a message merely because it is longer than 512 transport cells.
Stream it with a retained cursor. A genuinely unavailable resource produces a
visible, counted deferral with source retained for later service.

## Teaching progression: overlapping practice, not locked stair steps

Start from the existing learned core; do not send it back to weeks of alphabet
copying. Mix earlier skills into later lessons and adjust the diet according to
observed errors. No stage requires 100% reference completion before the core
is allowed to see richer language.

| Practice emphasis | What the core does | What success means |
| --- | --- | --- |
| Words and sentences | Predict natural prose, then continue unfamiliar sentence openings | Fewer spelling/grammar errors, improving heldout content likelihood |
| Connected descriptions and stories | Produce several connected sentences about people, objects, and events | Stable subject, understandable relations, less repetition |
| Conversation | Answer ordinary questions and requests with actual preceding turns available | Relevant, intelligible response; reasonable turn completion |
| Grounded correction | Use provided facts, handle a changed fact, ask when information is missing | Answers track evidence and corrections, not memorized wording |
| English across breaths | Read own/sibling textual contributions and produce a useful next contribution | Retained language and demonstrable use of available prior information |

Illustrative authored lesson families, with many independently varied examples:

- Subject/verb agreement: "The bird sings. The birds sing." Then use unseen
  subjects and actions.
- Tense: describe what someone did yesterday and plans for tomorrow.
- Reference: "Lena has a cup. She puts it on the shelf." Ask what "it" refers to.
- Relations: "The red cup is beside the blue bowl." Ask for a description.
- Causality: "Rain soaked the path." Ask why the person brought boots.
- Dialogue: greet, answer a question, ask a relevant follow-up, and yield a turn.
- Paraphrase: explain a supplied sentence in different simple words.
- Missing information: ask where an object is when its location was never given.
- Correction: first the key is in the drawer; later it is moved to a bag.
- Revision: improve a rough sentence while preserving its intended meaning.
- Silent breath: revisit an unresolved question using existing evidence.
- Useful completion: give a sufficient answer without adding repetitive filler.

Do not train one tiny template family until its score is perfect and call that
English. Hold out combinations, contexts, speakers, and independently authored
examples; include real text beyond templates.

## Learning objective and continuous state

Make token-weighted next-transport categorical cross-entropy the main language
objective. Supervise ordinary language throughout each selected stream instead
of using only the suffix of a few fixed prefix cuts. Track content loss
separately from any utterance-end and breath losses.

Begin with ordinary teacher forcing. Providing the correct preceding character
is how this learner receives a stable next-character target; it is not evidence
that free generation has succeeded. Sample free generation regularly as a
separate evaluation.

Persist recurrent state across successive compute chunks of the same logical
stream. Carry it across true chronological turns when that is the declared
conversation task. Start a separate state for unrelated documents, separate
batch streams, and held-out evaluation. Never carry training-example state into
a held-out prompt.

Truncated backpropagation may detach the graph every 64 transport decisions
initially, retaining numerical state and the exact next cursor. That limits
gradient history for an update; it is not a text limit or a memory reset.
Longer credit-assignment horizons can be tested later.

Make memory use actually bounded: accumulating graph-bearing losses for every
chunk and calling backward only after an entire long message still retains
chunk graphs. Use Trainer-owned gradient accumulation/microsteps that release
each graph while accumulating correctly normalized gradients. Do not call an
ungoverned optimizer from a helper to work around the Trainer.

For a fixed-parameter inference replay, chunked and uninterrupted execution must
agree. Across training updates, parameters intentionally change; record the
state/update policy and do not falsely label that trajectory a fixed-weight
replay.

## EOS and free generation

EOS is optional for learning ordinary next-character statistics. It is useful
for expressing "I have finished this utterance" so a listener, sibling, or
another turn can proceed. Finishing one utterance does not mean the organism
stops breathing, loses its private state, or stops learning.

Retain the existing private EOS category. For utterance examples, supervise it
only at an actual authored end, initially at ordinary weight 1. For continuing
stream segments, supervise actual next content and do not invent an EOS at an
observation boundary. Report its objective contribution separately. Do not
return to heavy EOS weighting merely to improve a headline stopping score.

Provide resumable free generation. A work page exhausting its allocation returns
PAUSED_WITH_CONTINUATION, with hidden state, previous token, decoding policy/RNG,
UTF-8 partial-scalar state, output identity/cursor, and exact model/context
bindings. It never pretends to be MODEL_FINISHED and never injects EOS.
Continuation must not re-ingest the prompt or duplicate the last emitted token.

This needs an implementation beyond the current simple evaluation guard.
Labels above describe proposed experiment statuses, not a silent alteration of
a locked runtime schema.

A model-generated EOS ends that utterance. Preserve that fact rather than
suppressing it to manufacture longer output. In continuous operation, the next
breath can attend the committed utterance and continue cognition under the
organism's cadence. Live Heart integration remains its own implementation task.

For open-ended prompts, use one declared greedy evaluation policy and a
separate fixed-seed sampling evaluation, initially temperature 0.8/top-p 0.9 as
a comparison. Keep decoding fixed when comparing checkpoints; do not tune it
against each output or apply hidden cleanup/repetition tricks. Sampling may
change repetition and diversity; it does not constitute new learning.

Inspect successively longer resumable excerpts for degeneration. Excerpt size
belongs to the inspection budget and must be reported as censored/unfinished
when the core has not ended the utterance. No output deadline is inferred.

## Evaluation: score the ability being taught

Record at baseline and every review point:

1. Heldout content cross-entropy over all scored transport decisions, with
   source-domain breakdowns and actual evaluated token counts. Compare on the
   same data and tokenizer/transport. Keep EOS and breath losses separate.
2. Readable free outputs for a fixed prompt panel, saved exactly with seed,
   checkpoint, source, length, and whether generation ended or was paused.
3. Grammar, local coherence, prompt relevance, evidence consistency, and
   repetition rated separately. Use a written rubric: 0 failed, 1 partial,
   2 adequate for each applicable dimension; retain the text and explanation.
   Blind checkpoint identity when comparing samples. Human or external-model
   judgments are attributed judgments, not deterministic proof.
4. Deterministic checks where justified: exact transport, correct extracted
   names/numbers for unambiguous tasks, contradictions to supplied facts, and
   sustained repetition. Do not use a dictionary score as a proxy for fluency.
5. Breath retention and private-state controls, reported separately.

A practical development panel starts with 48 prompts: 12 open continuations,
12 everyday dialogues, 12 evidence-grounded questions/corrections, and 12
multi-breath/continuity cases. Version prompt identities; audit source overlap.
Use that same panel for paired comparisons, and reserve a separate final panel.
These are evaluation sample counts, not core capacity or output limits.

Reference exactness and reference-length EOS remain available for explicit
copy/reconstruction tasks. They are not requirements for open English fluency.
For free conversation judge whether the response is sufficient and coherent,
not whether its length matches an archived speaker.

Loss is a valuable training signal, not a complete fluency score. It measures
prediction with supplied context; generated errors change the future context.
Likewise, 100% teacher-forced EOS can coexist with weak generation.

## Breath and Soul: retain behavior and test the claim

Keep the currently successful breath rehearsal each optimizer step, with its
existing weight 1.0 and batch 16 as the initial retention anchor. Measure its
cost and gradient contribution; these settings may be adjusted through a
controlled comparison, not silently removed.

Keep the current 120-episode synthetic breath suite as regression evidence.
Add a separately reported English breath suite; neither is a declaration of
general autonomous reasoning.

A source-versus-zero logits difference proves input sensitivity only. A random
untrained recurrent model can pass that check. For useful memory dependence,
compare otherwise identical tasks under intact, reset, swapped, and irrelevant
private state. Include a task whose needed fact was available earlier but is
no longer in the currently attended view, with the exact source still preserved
canonically. Account for public transcript cues so they cannot substitute for
the private mechanism being measured.

Durable Soul is more than an in-memory GRU tensor. Integrate through the
existing generation-bound private Soul codec/receipts before claiming Soul
inhale/exhale or mode-switch continuity. Preserve canonical evidence and never
publish private Soul contents to siblings. Current experiment state carry and
a future live Soul integration must remain accurately distinguished.

## Continued learning and corrective practice

Renew training while the evidence supports useful progress. Track unique
source coverage, cumulative supervised transport tokens, optimizer updates,
replay fraction, elapsed time, and evaluation history. A process restart must
not silently mean fresh optimizer state or replaying the first messages again.

Introduce corrective practice from observed errors: collect a generated sample,
retain its prompt/evidence and provenance, author a corrected continuation
appropriate to that actual prefix, and train on the pair. A different but valid
continuation is not an error.

Do not automatically train the core on its own unreviewed output. Do not require
scheduled sampling as the price of further learning. If later tested, keep it
a separate matched experiment: after a divergent prefix, forcing the original
continuation may no longer be an appropriate language target.

For every new material batch retain older language and breath replay.
Evaluation corpus stays out of gradients; freeze final-test material before
tuning and do not turn repeated development inspection into claimed blind
generalization.

## First implementation and run recipe

Implement a versioned streaming-language path, preserving v3 artifacts.

- Inventory/qualify local text and publish the corpus/split manifest, coverage,
  deferred counts, authored-lesson provenance, and a real prompt panel.
- Add stream state/cursor management, Trainer-owned bounded accumulation,
  explicit per-task loss masks, and resumable free generation.
- Add meaningful checks for exact long-message coverage, UTF-8 boundary
  continuation, chunk/resume equivalence, checkpoint restoration, split
  isolation, real end labels, and failed-integrity refusal.
- Start from protected dec45989 as an explicitly recorded experimental branch.
  Initial AdamW LR 5e-5, weight decay 0, gradient clip 1, FP32, four logical
  language streams, 64-decision BPTT chunks; retain the breath settings above.
  These are initial hypotheses and must be logged with actual throughput.
- Run a 32-update wiring/health smoke with measured preflight and baseline/post
  content loss, simple prediction baseline, generated samples, and breath
  retention. A wiring smoke is not a competency graduation.
- After a healthy learning signal, use a renewable pilot allocation of 1,000,000
  supervised language transport targets, reviewed every 100,000. Those counts
  are initial resource allocations, not a statement that English is learned
  by that many tokens or that learning stops there. Measure throughput before
  estimating duration on the GTX.
- Grow exposure while reviewing trends and retained skills. Do not restart the
  whole effort after a noisy 24-step metric fluctuation, and do not require a
  perfect prose match to renew training.

Retain a small verified rolling checkpoint set under existing retention law,
plus the protected language milestone and evidence-linked landmarks. Checkpoint
model, optimizer/schedule, RNG, per-stream recurrent/decoder state, exact corpus
cursor/order, objective, and any participating Soul receipts atomically.
Verify reload before retiring a recoverable artifact. Retention cleanup remains
separately governed; this document does not delete anything.

## What the checks control

Integrity checks stop a broken trainer: nonfinite updates, corrupt/ambiguous
lineage, invalid exact transport, missing source, missing continuation, or data
leakage. They are necessary for trustworthy learning.

Progress reviews decide the next training diet and whether a larger compute
allocation is justified. Use repeated comparable loss/sample trends and actual
forgetting evidence. Preserve a learning candidate while diagnosing a plateau;
a failed review never erases the core's history.

Serving/promotion checks decide which learned parameters enter the running
organism. They do not grant or revoke each breath based on novelty, English
perfection, or external activity.

Low reference-exactness, a different sensible utterance length, and an unfinished
observation excerpt are not reasons to forbid language practice or breathing.

## Evidence and scope

This specification is grounded in direct inspection of the current v3
curriculum, loss, evaluator, launcher, tests, and the 2026-09-26 notes. V3
currently skips messages above a configured length and starts a new recurrent
state per completion example; it is not the continuous streaming school
specified here. Its measured counterfactual is a useful wiring diagnostic,
not proof of learned semantic/Soul use.

Primary research supports the direction without proving this particular GRU:

- [Mikolov et al., recurrent neural language modeling](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html):
  recurrent networks can be trained as predictive language models.
- [Eldan and Li, TinyStories](https://arxiv.org/abs/2305.07759):
  carefully simplified language data enabled coherent generation by small
  models in their experiments. Their architectures and representation differ
  from this GRU; do not transfer their results as an Axon capacity guarantee.
- [Bengio et al., Scheduled Sampling](https://arxiv.org/abs/1506.03099):
  identifies the discrepancy between supplied and generated previous tokens.
  It motivates a controlled future comparison, not a mandatory next repair.

Only this curriculum and both ledgers were written for this request. No new
dataset, implementation, optimizer run, promotion, or live cadence change is
claimed.

