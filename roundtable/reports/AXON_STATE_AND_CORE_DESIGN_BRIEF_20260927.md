# Axon — state of the engineering and core-design brief

**Author:** Claude / Claude Code
**Date:** 2026-09-27
**Purpose:** Preparation for a design discussion on core architectures. Written read-only; nothing modified except this file.
**Companion:** `roundtable/proposals/PROPOSAL_LEDGER_CONSOLIDATION_20260927.md` (ledger strategy + governance defects).

Marking: **[V]** = read/run directly, path given. **[A]** = inference.

---

## Part 1 — The month, in one page

**2026-07-03 → 08-17.** 418 commits total; 37 in July, 156 in August, 225 in September.
Day Zero repository, frozen 16D Unicode substrate, D64 field compiler, dormant evidence bridge.

**08-17 → 08-26 — "the Heart must beat before it learns."** The Heart amendment is ratified: one
sole canonical writer, masks are *derived* views and never a second canonical field, the tick ends
at the commit and nowhere else. Kimi refuses to *"train neurons for an organ whose body does not yet
exist."* Dormant recall is measured honestly at Hit@8 0.625 against a 0.203 raw. The Heart
translator is built and fails (**grounded roundtrip 0.0** against a 1.0 floor). The `max_source_chars=192`
ceiling is audited as *"The objection was correct"* — an architecture ceiling masquerading as a page.

**08-26 — the shelf pivot.** ChatGPT/ChatGPT's convergence line: *"the Heart no longer needs to be
intelligent to make heterogeneous reasoning cores possible."* Exact characters pack into rail vectors;
rail-to-rail movement becomes deterministic repacking, not a learned organ. The learned-Heart branch
dies. **This is the moment Axon stops being a translator and becomes a body with a bus.**

**08-28 → 09-13 — curricula, and the tournament that measured nothing.** Foundations-First is
ratified. Grok joins and inverts the tissue diagnosis: *"A giant FFN is not what made Candidate A
special; the giant FFN is what made it a prior-memorizer with a postage-stamp mixer."* The 48-shape
D64 tournament runs and dies of disk exhaustion; every candidate hits teacher-forced 1.0 while
**every one is at free-running exact 0.0** — across 1.22M to 169.38M parameters. No capacity ranking
was obtained; the program was paused by operator decision.

**09-17 → 09-19 — Soul week, then the reversal.** Four Soul proposals in two days, and Copilot finds
both blocking defects: the canaries depend on emission the objective suppresses, and
*"The termination gate the entire termhead programme exists to train has never received gradient."*
Then, on 09-19, the sharpest single act in the record: **the learned DELTA/NO_OP/ABSTAIN decision head
is deleted from doctrine**, and the receipt-continuation mechanism ratified only eleven days earlier
is retired with it. Cores now speak English proposals. `archive/legacy_typed_reasoning_20260919/`.

**09-22 — the forensics that ended a month of work.** D0 proves the copy route is innocent:
substituting the compiler-certified pointer raises first-cell accuracy to **24/24 through the
unchanged learned copy route**, while the request-to-query bridge classifies at **0%**. The
conclusion: *"What should be retired is learned physical addressing/copy-pointer anatomy, not
transformer attention itself."*

**09-24 — the D16 Core Bus, ratified.** *"A Core's internal model width is private architecture and is
not a public transport width."* Proven by B3: a fake Core at `d_model=512` completes the live cycle
while the frozen tick contains no D512 rail. The packed-rail mandate ends.

**09-25 → today — breathing, and a fork.** The single-GRU breath proof reaches **360/360 exact
free-running breaths** and 120/120 silent third breaths. Jeff: *"external silence must not stop
breathing."* The Core Architecture table opens: *"Exact mirror != field interpretation != deliberative
cognition."* Two tracks: get Axon alive on one GRU (Track A), research better cores without blocking
it (Track B).

**The shape of the month:** almost nothing ended in promotion. What was built is *anatomy* — a Heart
that alone commits, a Trainer that alone mutates, a substrate that is exact, a bus that is
width-neutral — plus a sequence of honestly recorded failures that repeatedly falsified the
engineers' own preferred story. The dominant failure mode was **not a wrong architecture but a
measurement artifact**: a gate that accepts NaN, a preflight that hard-coded `passed: True`, a
metric asserted at 0.0 in tests but never gated. Hence Codex's reachability law: *"a stage gate may
only require a metric whose causal components all carry nonzero weight in that stage."*

---

## Part 2 — True current state

### 2.1 Nothing is training **[V]**

Jeff's framing was *"we're finally in the heat of training."* The evidence says otherwise, and this
is the single most important correction in this brief:

- **No training process is running.** Filtering all processes for `train_continuous_core|train_living|axon_trainer`
  returns nothing. GPU (GTX 1650) is at 6% util, 1336/4096 MiB — desktop baseline.
- The last tranche finished **2026-09-26 20:19** and was **deliberately paused for developmental
  review**: candidate `r512schoolv6-16f1385312cc`, step 400, `gate_decision_id: null`.
- The trainer writer lease is **absent**; no active generation pointers; **nothing has ever been
  promoted** (`promotion_proposals: 0`, `activation_receipts: 0`).
- A **monitor server is live** (PID 19940, port 8788, started 21:53). That is why the dashboard looks
  active — it is displaying a finished run. Telemetry stops at 20:19.

**Accurate framing: paused at a review point, not in the heat of training.**

### 2.2 What was actually achieved **[V]**

| Run | Scale | Result |
|---|---|---|
| B4 copy, 700 steps | — | loss 5.87 → 2.36; teacher content 0.16% → **28.17%**; **free exact 0 → 2.5%** (first nonzero) |
| Breath, 700 steps | — | breath loss 5.90 → **0.00421**; **120/120 episodes, 120/120 silent third breaths** |
| Dormant v1, 400 steps | — | loss 15.05 → **1.6126**; teacher content 62.18% → checkpoint `dec45989` = **protected milestone** |
| v5 stream, 400 steps | 102,400 targets | loss 3.26 → 2.7873; teacher 22.39% → 31.02% |
| **school v6, 400 steps** | **128,000 targets** | loss 2.4664 → **2.2503**; teacher **36.58%** vs constant floor **15.56%** |

**These are hashed artifacts but single-source [V].** Every figure above is self-reported by the
trainer that produced it. A Hermes-authored independent re-grade exists — `State/tmp/probe_school6_grade.py`
(5,385 B, written 21:58, *after* the run paused) — which re-verifies `artifact_bytes`/`artifact_sha256`,
re-measures both checkpoints on the same curriculum, and compares six headline metrics to the
run-summary claims at 1e-9. **It has no recorded result anywhere**: its sentinel strings
(`ALL CLAIMS REPRODUCED EXACTLY` / `SOME MISMATCH`) appear in no log and no output file, and
`State/tmp/__pycache__` is empty. So the 2.2503 loss and the 36.58% teacher accuracy are **claimed,
not independently reproduced**. Running that probe is the outstanding step if those numbers are to be
treated as solid; it loads two models on GPU and was deliberately not run by this read-only sweep. Its
claimed values are in the script itself (`final_language.mean_loss: 2.2502934899867966`,
`teacher_content_accuracy: 0.3658119658119658`).

**But free generation is degenerate.** Greedy decoding collapses to `"the the the the the…"`
(`unique_word_ratio=0.0566`, `top_word='the'` ×26). Sampled output produces non-words
(*"es torcest farcentions lisectingh"*). **The model has never produced coherent free-running English.**

### 2.3 The real architecture **[V]**

There is **exactly one trainable Core** in the repo:

```
exact D16 cell → Linear(16→512) → nn.GRUCell(512,512) → Linear(512→352)
1,765,216 parameters
```

No attention, no FFN, no residual, no normalization, no depth parameter, no tokenizer embedding table.
That is machine-enforced (`training/continuous_core_d512_breath.py:449-450`). The 352 output classes
are **351 registered transport categories** (95 native characters + 256 UTF-8 byte codewords) **+ 1
private EOS**. Byte-to-byte cosine ≤ 0.5; non-canonical spellings are rejected.

**Tokenization is therefore already resolved in a specific and non-obvious way: there is no learned
tokenizer at all.** The vocabulary is characters plus UTF-8 bytes, each mapping to one exact frozen
16D cell. There is no lexical layer, no word unit, no BPE.

### 2.4 The organs — what actually exists

Six organs, and the authority model is **implemented, not decorative** **[V]**:

| Organ | Size | Authority |
|---|---|---|
| `runtime/heart` | 30 modules, ~380 KB | **Sole canonical writer.** 20-slot valve plane, CLOSED by default; only 4 valves are real (`user_ingress`, `tool_ingress`, `advisor_ingress`, `dormant_recall`) |
| `runtime/field` | 8 modules, ~164 KB | Binding validator of every committed byte; D64 compiler |
| `runtime/dormant` | 6 modules, ~200 KB | **Read-only** retrieval senses; 4.4 GB evidence index |
| `runtime/soul` | 3 modules | Private, opaque, four-temperature core state |
| `runtime/trainer` | 34 modules, ~700 KB | Parameter authority — but **only `status` is dispatchable**; 12 of 13 commands return `UNAVAILABLE` |
| `substrate/` | repo root | The frozen exact 16D character bank |

The single-writer claim is enforced by **five independent gates** that must agree: an OS file lock,
an owner-token re-read on every mutating call, an in-process registry, the CLOSED-by-default valve
plane, and a class-based commit refusal in the transaction boundary. Cores are refused commit
authority in **two unrelated places** — defence in depth by design. `ValveEnvelope` actively rejects
any envelope that carries a grant, so authority is *derived, never supplied*.

"Masking without deleting" has two proofs in code: `RegionState.text` returns the complete canonical
string by contract, the compiler hashes the full text **including masked cells**, and the demo's
roundtrip operation asserts full-body exactness while the compiled rail is shorter.

**Two corrections worth carrying into the discussion:**

- **`controlplane/` is contract-only — "DESIGNED, NOT BUILT"** — and has **zero tests**. There is no
  control-plane server. `runtime/bus`, `runtime/slots`, `runtime/adapters`, and `D:\Axon\cores\` do
  not exist at all; the bus lives in `runtime/heart/core_bus.py`.
- **The repo conflates two different corpora.** "59,875" is the `experience_v1` **import** (one
  D2-era import of 59,875 records out of 59,925 total). The **retrieval index** is **427,001
  containers and 351,978 semantic edges**. These are different stores with different sizes; the
  README's phrasing merges them. Cite 427,001/351,978 for retrieval, 59,875 for the experience import.
- **The index's shape, measured live [V]** (read-only open of
  `State/dormant/.derived/evidence_v1/index.sqlite3`): `containers` 427,001; `edges` 351,978;
  **`graph_neighbors` 93,025,272**; `container_terms` 4,195,793; `edge_terms` 7,314,509. The 93M-row
  adjacency table is a **~217× blow-up over the container count** and is why the index is 4.4 GB.
  Retrieval is therefore two-stage — term lookup, then graph expansion — so **recall cost scales
  with graph degree, not container count**. Any memory or latency budget derived from "427k
  containers" is wrong by two orders of magnitude on the graph path.
  The corpus counts are now **triple-confirmed**: `wc -l` gives `containers.jsonl` 427,001 and
  `semantic_edges.jsonl` 351,978 lines (matching the live SQLite counts exactly), and
  `symbol_registry.jsonl` / `layout_groups.jsonl` 4,198 each. File, corpus manifest, and index agree.
  **Storage [V]: `State/dormant` = 9.2 GB** (≈3.5 GB byte-exact source snapshots + 4.4 GB derived
  index + ≈1.1 GB JSONL), while **`State/active` = 3.7 MB**. The organism's durable truth is
  kilobytes of JSON; its memory is gigabytes. The drive holds 278 GB free (71% used).
  **`State/training` = 22.5 GB apparent / 29 GB allocated**, and its shape refutes the working
  assumption that retired track directories dominate it: **`trainer/` alone is 19.2 GB across 3,441
  files** — the checkpoint artifact store — with `cloud/` a further 3.0 GB. `plm3` (10 MB),
  `diagnostics` (3 MB), `curricula` (21 MB) and the five `pytest_checkpoint_*` directories are
  collectively negligible. **Disk pressure here is checkpoint accumulation, not history.**

### 2.5 The tree was left non-green, and the last hour was verification **[V]**

The newest file in the whole tree is a failing test sweep: `logs/pytest_hermes_full_sweep2_20260926.log`,
mtime **2026-09-26 22:00:37** — later than the newest canonical ledger event (21:55). It ends `EXIT=1`
with **three failures, all in `tests/test_day_zero_hygiene.py`**. That afternoon's sweep (14:58) had
**twelve** failures, so the evening's work cut twelve to three — but the survivors are precisely the
guards that police how wide the active Python surface is, and they are three different problems:

| Failing guard | Cause | Reading |
|---|---|---|
| `test_authority_mirrors_are_byte_identical` | root vs `docs/` copy differ at byte 46 | real defect — the stale mirror |
| `test_day_zero_active_python_surface_is_narrow` | extra file `runtime/field/d16_view.py` | **stale guard**, not a violation: the D16 view is ratified 2026-09-24 work the whitelist never learned |
| `test_parallel_pre_day_zero_bodies_are_not_live` | `ops` exists — `ops/windows_vm/bootstrap_axon_vm.ps1`, tracked | real violation of the pre-Day-Zero-body ban |

The last ~1.5 h of 09-26 was not training. Five `State/tmp/probe_*.py` scripts landed between 21:14
and 21:58 (`probe_extract_extra`, `probe_recovery_report`, `probe_recovery2`, `mail_check`,
`probe_school6_grade`) plus `sot_mirror.diff`. That is ad-hoc verification of the just-finished run,
consistent with the trainer's own `"paused for developmental review"` reason string — and it means
**the run was paused with its verification unfinished.**

---

## Part 3 — The design space, honestly drawn

### 3.1 Settled — do not relitigate without an amendment

1. **Exact D16 is public truth; Core width is private.** Ratified 2026-09-24, live in SoT `:426-431`.
2. **The exact mirror is body tissue, not cognition.** *"The exact local Shared Field mirror and the
   learned chamber must never be conflated."*
3. **Control A anatomy** = D512 one-GRU, zero attention — **a development baseline, not a ceiling.**
4. **Masks are attendance, never forgetting.**
5. **Durable Soul is more than an in-memory GRU tensor.**
6. **No latent carryover across parameter generations** — *stated as doctrine*, see §3.4.

### 3.2 Live debate — and the uncomfortable fact that none of it has an experiment

Everything below is argument, not evidence. No attention has ever been run at D512. No matched
comparison of any two architectures exists.

| Question | Positions | Evidence |
|---|---|---|
| **Attention** | Roadmap: *"Full attention is no longer the default… If attention returns anywhere inside a Core, it must justify itself against the Reflection Table and remain bounded."* Codex: *"Attention also remains an admissible hypothesis. Exact addresses do not eliminate content-based comparison."* | **None.** |
| **Mamba/SSM** | Codex: *"I would defer Mamba."* Roadmap B1/B3: named as a candidate and a Reflection Table implementation. Gemini: SSM belongs in the deliberation chamber only, because deltas *"require immediate, localized relational binding, not infinite-context sequence filtering."* | **One 160-step A/B** — see §3.3. |
| **Second chamber** | Codex: not yet justified. Gemini: only with decoupled objectives, or *"gradient flow will entangle their representations."* Hermes: testable via a 4-cell grid. | **None.** |
| **K "pondering" microsteps** | Gemini: *"Without an active driving term, an SSM running in a loop will either decay to zero or diverge"* and predicts attractor collapse for K>4. Codex/Hermes: fully specified protocols. | **None.** |
| **Recursive reasoning** | — | **"recursi" appears zero times across all 8 Core Architecture documents.** Blank page. |

### 3.3 The one cross-architecture result — and it is unrecorded

Both cores trained on the **identical curriculum** for 160 steps at batch 32 **[V]**:

| | GRU512 | Mamba reference |
|---|---|---|
| params | 1,765,216 | 1,861,984 (+5.5%) |
| resident state | 512 | **11,776 (23×)** |
| loss | 1.0058 | **0.7624** |
| teacher content | 66.67% | **70.76%** |
| free breath exact | 0.556% (2/360) | **4.44% (16/360)** |
| free **episode** exact | 0% | **0%** |
| peak CUDA | ~103 MiB | **~422 MiB (4.1×)** |

At 24 steps the ordering **reverses** (GRU 41.94% vs Mamba 32.08%), so the curve crosses — single
checkpoints are uninformative here. The GRU's 700-step run then solves the task completely.

Caveats that matter: **not seed-matched**, not parameter-matched, not memory-matched, never run past
160 steps, **0/120 complete episodes** (it has not shown multi-breath continuity), and it is **blocked
from the streaming path** — `ContinuousCoreD512Mamba` lacks the `begin/continue_greedy_generation`
API, so v4/v5/v6 cannot run on it without porting.

**And it is not in the ledger.** A grep for `mamba` across all 410 canonical events returns only 4
hits, all from 2026-09-24/25, all about *proposals*. The two Mamba breath runs and their checkpoints
are recorded nowhere. This violates the working contract's §12.

### 3.4 Three tensions the documents have not reconciled

1. **Objective vs doctrine.** TC-OPENING named *"next-token prediction as the governing learning
   objective"* the thing to escape. Every live script optimizes exactly that — next-transport CE with
   `positive_eos_targets: 0`. **Architecture choice is downstream of objective choice, and the
   objective is currently unexamined.**
2. **Parameter-generation boundary.** The roadmap says a hidden state from generation N *"may not
   remain meaningful"* under N+1, and this should *"remain fail-closed."* But v5/v6 deliberately carry
   recurrent state across optimizer updates (`state_reset_between_optimizer_updates: False`). The
   running code has already answered a question the documents still call open.
3. **The metric is not comparable.** The **same checkpoint** `dec45989` scores **68.25% teacher content
   on eval set `68094c1a` and 20.54% on `e62149d2`**. "Teacher content accuracy" is therefore not
   comparable across runs — **which makes every architecture claim resting on it unfalsifiable.**

---

## Part 4 — What is missing, ranked

| # | Gap | Test it needs |
|---|---|---|
| 1 | **The Mamba A/B is unrecorded** | A corrective ledger event (additive, never a rewrite) |
| 2 | **No matched experiment exists** | ≥3 seeds, parameter-matched (the grid below), memory-matched, **past 160 steps** |
| 3 | **"Resident memory" never demonstrated** outside a synthetic letter task, and confounded — the answer sits in the attended mirror | Hermes's zero-training bridge + an intact/reset/swapped/irrelevant counterfactual on a task whose answer is **absent from the attended view** |
| 4 | **The metric drifts across eval sets** | One frozen, versioned eval corpus with a declared token count, before any comparison |
| 5 | **K-pondering**: designed in detail by two reviewers, never run | Codex's cloned-pre-deliberation protocol, with Hermes's compute-matched fix |
| 6 | **Two-chamber**: design complete (Gemini's counterfactual override + delayed recall), never built | Note: Gemini's GRU724 control is arithmetically wrong — **GRU716** is the matched control |
| 7 | **No depth, no FFN, no normalization anywhere** | The opening's own §1 test: job, why deterministic machinery can't do it, owned state, interfaces, falsifiable experiment, **removal criterion** |

**The parameter grid for any fair comparison [V]:**

| Cell | Params | vs A | State |
|---|---|---|---|
| A — GRU512 ×1 | 1,765,216 | — | 512 |
| B-368 — GRU368 ×2 | 1,765,648 | **+0.024%** | 736 |
| A-W716 — GRU716 ×1 | 3,344,788 | +0.109% vs B-512 | 716 |
| B-512 — GRU512 ×2 | 3,341,152 | +3,636 vs A-W716 | 1,024 |

---

## Part 5 — The three questions worth putting on the table

1. **Is the Mamba result real, and why is it not recorded?** It is the only cross-architecture
   evidence in the repo, it favours the alternative on quality-per-step at 4.1× memory, and it exists
   in `State/` but nowhere in the ledger.

2. **Is `next-transport CE` the objective this discussion is actually about?** Every live script
   optimizes it; doctrine says to escape it; nobody has reconciled the two. Choosing an architecture
   before settling the objective is choosing an answer to the wrong question.

3. **What is the control?** The fairness grid and the ≥3-seed / no-seed-reversal rule are already
   written down and cost roughly a dozen short runs at a scale already proven (700 steps, ~13 min
   each on this GPU). Ratifying them costs nothing and makes every future architecture claim
   falsifiable. The alternative — continuing to compare 400-step runs on non-comparable corpora —
   **cannot produce a decision.**

**The honest framing to open with:** the argument is not "GRU vs Mamba." It is that **the GRU has
never been given a fair chance, and no alternative has ever had a matched comparison.** Both halves
of that sentence are fixable cheaply, and the machine is idle.
