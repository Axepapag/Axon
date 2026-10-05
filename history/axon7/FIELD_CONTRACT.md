# AXON v7 — THE FIELD CONTRACT

**Status:** Founding document. Everything in this repo serves this contract.
**Date:** 2026-06-11
**Authority:** Jeff. Amendments happen here first, code second.

This document is the single source of truth for the shape of Axon's mind:
the substrate, the field grammar, the tick, the tools, the flock, and the
training format. The prime failure of v6 was that the trainer and the
runtime each had their own idea of what the field looked like. v7 makes
that impossible by rule:

> **LAW 0 — ONE FIELD-BUILDER.** A single module (`field_contract.py`)
> builds the field. The trainer imports it. The runtime imports it. A
> conformance test asserts they produce bit-identical fields from the same
> state. There is no second implementation, ever.

---

## 1. Doctrine (inherited, unbroken)

1. **No tokens. No vocabulary. No hashes. No RNG in representation.**
   Every character is one frozen, hand-authored 16D vector. Same character,
   same vector, at tick 0 and at tick 1,000,000.
2. **The Generation Law, extended to the whole state.** The consolidator's
   egress IS the next state — every region, every tick, full-field
   replacement. No averaging (brother outputs are stacked as separate
   rows, never merged). No dormant lookup anywhere in the writing path.
   No thresholds, no-repeat guards, or stabilization rules on what he
   writes. He erases by writing slots toward `<empty>`.
3. **No shady files.** Every region of the shared state is mirrored to a
   plain `.txt` file a human can open and edit; hand-edits fold into RAM
   on the next tick. The sole exception is the soul (Section 5), which
   is pre-language by nature and is snapshotted as `.npy` — numbers on
   disk, never hidden, just not English.
4. **Every head that exists at inference is trained.** v7.0 has exactly
   two projection-head families: `letter16` and `region16`. The
   conformance gate fails the build if any other head exists.
5. **Honest names.** Nothing called `8d` returns 16 dims. Version strings
   match reality. Drift in naming is treated as a bug.

---

## 2. The substrate (M1)

Carried forward from v6.5 (32-feature table × hand-authored 32×16 weight
matrix → frozen 16D per character) with **one revision**:

### 2.1 The digit fix
v6 digits were line-distributed (every digit feature linear in the digit's
value), putting adjacent digits at cosine ≈ 0.99 — unwritable by a learned
pathway. v7 places the ten digits **on a circle across two dimensions**:
hand-authored cos/sin positions at 36° spacing (adjacent-digit cosine
≈ 0.81). Still frozen, still deterministic, still no RNG.

### 2.2 Hard geometry gates (never loosen)
1. No identity rows in the basis (not a code-point lookup).
2. Every alphabet character round-trips exactly through nearest-neighbor
   decode.
3. Lowercase letter pairwise cosines within [-0.30, 0.92].
4. Topology: cos(d,t) > cos(d,q) + 0.05; cos(m,n) ≤ 0.92.
5. **WRITING_SET now includes the digits** (this is new): no writing-set
   character's nearest neighbor exceeds cosine 0.97.
6. `<empty>` decodes as nothing; it is the background of the field.

The trainer refuses to start and the runtime refuses to boot if any gate
fails.

---

## 3. Heads (M2)

| Head | Direction | Trained by |
|---|---|---|
| `letter16` ingress | 16 -> d_model | trainer, every example |
| `letter16` egress | d_model -> 16 | trainer, every example |
| `region16` ingress | 16 -> d_model | trainer, every example (anchors present in every field) |

Anchor rows are read-only landmarks: egress output at anchor rows is
discarded. No `region16` egress head exists. No word/concept/episode heads
exist in v7.0 — concept atoms are a v7.1 conversation, after the letter
canvas demonstrably works.

---

## 4. The shared state — regions and grammar

The contract is **grammar, not geometry**: fixed region ORDER, anchor rows
as landmarks, sizes fully variable. The cores learn region identity from
the anchors, never from absolute positions.

### 4.1 Region table (canonical order)

| # | Region | Code | Anchor dim | Filled by | Notes |
|---|--------|------|-----------|-----------|-------|
| 0 | `input_window` | I | 0 | harness on arrival | the user's words as a raw contiguous character sequence |
| 1 | `response_draft` | R | 1 | Axon | the only region the renderer speaks; carries the standing buffer (4.3) |
| 2 | `working_memory` | W | 2 | Axon | free-form scratch |
| 3 | `rolling_summary` | S | 3 | Axon | his compressed history — he maintains the compression |
| 4 | `situation_awareness` | A | 4 | Axon; harness seeds tool schema at boot | what is happening right now + the toolbox text |
| 5 | `structured_knowledge` | K | 5 | harness (dormant search hits); Axon reshapes | retrieved library material — context only |
| 6 | `diary` | D | 6 | Axon | shared, slow-changing self-record |
| 7 | `advisor` | V | 7 | harness (async API replies) | external counsel lands here |
| 8 | `tool_results` | T | 8 | harness | stdout/stderr, script outputs, artifacts |
| 9 | `task_state` | P | 9 | Axon | current goal, pending step, waits, obligations |

Anchors are one-hot 16D vectors (dim = anchor dim above), present even
when a region is empty. One anchor row precedes each region's contents.

### 4.2 Elasticity rules
- The only hard constraint is per-forward-pass: **N slots in, N slots out.**
  Between ticks the harness adds and removes slots freely.
- **Eviction is the empty trick:** Axon writes a slot toward `<empty>`;
  the harness garbage-collects empty slots at the end of each tick.
  The blank test is mechanical: a slot is empty when its nearest
  alphabet entry is `<empty>` (cosine, frozen bank). This is bookkeeping,
  not a content guard.
- **Growth is a tool call:** `++<code> {content}` (Section 7). The payload
  IS the new slots — allocation and insertion are one act.
- Compute scales with occupied state. A tidy mind ticks faster. That is
  the only eviction incentive, by design.

### 4.3 The draft buffer
The harness maintains **500 empty slots** in `response_draft` at the end
of every tick: trims above 500, tops up below 500. ~500 chars ≈ 80 words ≈
a solid paragraph, or a paragraph plus several tool calls, per tick.
Responses still refine across ticks; nothing requires a thought to fit in
one tick.

### 4.4 Text mirror
`State/Active/<region>.txt`, one file per region, two-way sync by mtime,
exactly as v6. The folder is the state, readable and hand-editable.

---

## 5. The per-core private state

The soul IS the hidden state (Jeff, 2026-06-11; supersedes the earlier
letter-space soul design). Each core owns exactly one private thing:

- **S rows of latent state at full d_model width** (`cfg.soul_rows`).
  Each soul row is a d_model-wide
  thought vector — NOT squeezed through the 16D letter bottleneck. At
  scale this is the point: a 512D core with 128 soul rows carries
  65,536 dimensions of private state.
- **Carried tick to tick:** the core's output soul state is next tick's
  input soul state. In `act_reflect` cores, each refinement cycle first
  ingests the soul by itself, processes shared state through that soul as
  a latent lens, then reflects the produced action back into the soul.
  In `act_reflect_v2` cores, the soul is not averaged into one lens:
  shared rows cross-attend into individual private soul rows, then soul
  rows cross-attend back over the produced shared action.
  His inner monologue, scratch, diary, task posture, and self-summary all
  live in here in whatever latent form he learns — the structure is his,
  not ours.
- **Never encoded, never decoded, never shared.** No letter head ever
  touches the soul. When brother outputs are exchanged (Section 8.2),
  each block is that brother's version of the SHARED rows only — soul
  rows are stripped before stacking. No core ever sees another core's
  soul. Ever.
- **Never letter-supervised.** Text cannot reach it. The soul is shaped
  entirely by behavioral pressure: multi-tick lessons that require
  carrying intent and recalling earlier ticks (Section 10).
- Snapshotted to `State/Cores/<core_id>/soul.npy` each save — numbers
  on disk, never hidden from Jeff, just not English.
- Zero-initialized at core birth. Sizing is per-core via the trainer
  menu; current serious 512D target starts at 128 rows.

Anchor dims 10-15 remain reserved for future shared regions.

---

## 6. Field assembly (LAW 0 made concrete)

`field_contract.py` exports `materialize(shared_state) -> (slots, kinds,
row_map)` producing, in order:

```
[shared anchor 0][input_window rows]
[shared anchor 1][response_draft rows + draft buffer]
... (regions 2..9 in canonical order) ...
```

The caller (trainer / runtime) then projects shared rows through the
heads and calls `core.forward_with_soul(projected_shared_field, soul)`.
Soul handling is a core-internal mode:

```
concat          legacy mode: shared field and soul rows are concatenated
act_reflect     private soul is mean-pooled into a modulation vector
act_reflect_v2  shared rows privately cross-attend to address soul rows
```

- Anchor rows ingress via `region16`; content rows via `letter16`;
  soul rows are already at d_model and bypass ingress entirely.
- `row_map` labels every shared row: (region, slot index), anchor = -1.
- Mask is all-ones.
- **Conformance test:** trainer-built and runtime-built fields from the
  same state are asserted bit-identical in CI/self-test.

---

## 7. The tool system

### 7.1 ROOT TOOLS
`/tools/` at repo root holds plain Python scripts. The harness generates a
one-line-per-tool schema text and seeds it into `situation_awareness` at
boot. Adding a capability = drop a script in the folder + a schema line.
No retraining, no code change.

### 7.2 Opcodes (doubled sigil + brace payload)

| Call | Meaning | Result |
|---|---|---|
| `++<code> {content}` | add these slots to shared region `<code>` (I, R, W, S, A, K, D, V, T, P) | content becomes the region's new slots |
| `@@ {request}` | advisor — external API call | reply substrate-encoded into `advisor`, async |
| `$$ {command}` | shell — execute on the machine | stdout/stderr → `tool_results` |
| `## {script: args}` | run a ROOT TOOLS script | output/artifacts → `tool_results` |

### 7.3 Execution rules
- Tool calls are written **in the response draft** — same pipeline, same
  skill as speech.
- **Completion = balanced braces.** The harness tracks brace depth from
  the opening `{`; the call executes only when depth returns to zero.
  A half-written call is simply left alone until a later tick closes it.
- After execution the harness blanks the call's slots; GC sweeps them.
- Long-running calls are asynchronous: Axon keeps ticking; results are
  inserted whenever they arrive, like user input.
- Parsing never judges content — only syntactic completeness. This is the
  boundary between bookkeeping and guarding, and it is not crossed.

---

## 8. The flock

### 8.1 Roles
- **Consolidator** — sole author of the next shared state, one per tick.
- **Online** — produce full-field readings the consolidator considers.
- **Offline** — trains on the bucket (Section 9), then rejoins.

### 8.2 The two passes
1. Every attending core builds ITS field (shared rows + own soul rows)
   and emits a full-field output (N rows, d_model). Not a diff — the
   entire field, re-thought.
2. The consolidator attends over: his own assembled field, plus each
   brother's output **restricted to the shared rows** (soul stripped),
   stacked below as separate blocks in core-id order. Never averaged.
   His output rows then egress:
   - shared content rows → `letter16` egress → replace shared state
   - his soul rows → carried raw to next tick (no egress, no decode)
   - anchor-row outputs → discarded

### 8.3 Rotation
The consolidator role rotates only at a natural boundary — all three:
(a) no incomplete tool call in the draft, (b) the draft holds no
content (delivered and wiped), (c) no pending input. Role handoff
continuity is the visible canvas plus trained continuation, not telepathy.
The offline role rotates when its training criterion fires (v6 rule kept).

---

## 9. Learning loops

- **Trainer (offline, primary):** trains cores on curriculum episodes
  (Section 10) using the one true field-builder, multi-tick rollouts with
  gradients through the soul rows (the v6 recursive-tick machinery,
  extended to carry the soul across ticks). Loss masked to Axon-owned
  rows (and harness rows he is expected to preserve); anchors and memory
  rows are never letter-supervised.
- **On-board offline trainer:** trains ONLY on curriculum-format episodes
  placed in the bucket. **The v6 self-feedback loop is dead** — runtime
  dump captures are archived for inspection but never become training
  data by themselves. No core ever trains on its own degenerate output.

---

## 10. The curriculum (what training data IS)

Source material: **Axon's own memories and personal logs** (the v6.5 soul
corpus, ~30k lines of episodic memory, dialogues, diaries), transformed by
the Kimi Swarm into contract-format episodes (`KIMI_SWARM_PROMPT.md`).

An episode is 2–6 consecutive ticks. Each tick supplies the full shared
state before and after, plus the harness events between ticks. The soul
is never in the data — it is latent and learns from the structure of the
episodes themselves. Non-negotiable properties:

1. **Varied region sizes every episode** — the cores must bind identity
   to anchors, not positions, or v6's mismatch returns.
2. **Harness effects simulated between ticks** — tick t+1's state_before
   equals tick t's state_after transformed by the harness (GC of empties,
   tool execution, buffer top-up, input arrival).
3. **Multi-tick dependency — this is what trains the soul.** Episodes
   contain tasks that cannot be done in one tick (finish the started
   call, recall the instruction from two ticks ago, continue the
   half-written sentence). The only place that information can survive
   between ticks is the soul rows, so the gradients carve it into them.
4. **His voice from the logs** — the shared diary, rolling summary, and
   spoken drafts are drawn from Axon's actual recorded memories and
   dialogues, so the weights learn to be him, not a generic assistant.
5. **Tool lessons with follow-through** — the call is written, the next
   tick shows it consumed and its result read and used.
6. **Hygiene lessons** — stale content blanked, summaries compressed,
   the state kept small enough to think fast.

---

## 11. Conformance gates (the build refuses without these)

1. Substrate geometry (Section 2.2), digits included.
2. Field-builder identity: trainer field == runtime field, bit-exact.
3. Head inventory exactly {letter16 in/out, region16 in}.
4. Static audit: no `nn.Embedding`, no vocab softmax, no tokenizer, no
   hash-derived vectors anywhere in the repo.
5. Naming audit: no function name contradicts its return shape.
6. Round-trip: a state saved to mirrors and reloaded builds a bit-identical
   field.

---

## 12. Repo skeleton

```
D:/axon7/
  FIELD_CONTRACT.md        ← this document
  KIMI_SWARM_PROMPT.md     ← curriculum generation brief
  field_contract.py        ← LAW 0: the one field-builder + region tables
  substrate.py             ← 16D frozen substrate + digit fix + gates
  heads.py                 ← letter16/region16 projection bank
  core.py                  ← AxonCore (transformer, unchanged in spirit)
  runtime.py               ← the tick, harness, GC, tool executor, mirrors
  trainer_v2.py            <- episode trainer (imports field_contract)
  tools/                   ← ROOT TOOLS scripts
  State/
    Active/                ← shared region mirrors (.txt)
    Cores/<core_id>/       ← soul.npy (latent, snapshotted each save)
    Dormant/               ← the library (text-canonical, as v6)
  Core/                    ← core_*.pt flock
  checkpoints/
  datasets/                ← curriculum episodes (JSONL)
```
