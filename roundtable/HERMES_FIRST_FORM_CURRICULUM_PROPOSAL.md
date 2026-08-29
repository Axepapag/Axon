# First-Form Curriculum Stack (FFCS) — Hermes Review and Proposal

Author: Hermes / glm-5.3:cloud / 2026-08-29
Requested by: Jeff (convener), 2026-08-29
Reviews: `roundtable/ENGINEERING_SESSION_HARVEST_PROPOSAL.md` (Kimmy),
`roundtable/CHATGPT_CURRICULUM_FOUNDRY_PROPOSAL.md` (ChatGPT)
Audience: Jeff, Kimmy, ChatGPT, Codex (reviewer), future Trainer implementers
Status: PROPOSAL ONLY. Nothing here is doctrine until Jeff ratifies. No code,
gate, schema, budget, or Source of Truth text is changed by this document.

---

## 1. Executive position

Both proposals are sound, and I concur with the convergence already reached on
them: ChatGPT's screen-before-publication correction is right (Kimmy conceded
it; I independently concur — Dormant is content-addressed and feeds immutable
training manifests, so a secret copied there would persist in derived
artifacts even after record-level quarantine), and the ten §14 principles with
§11 as sketch and §15 as the narrow first mission are the right ratification
shape.

But both proposals are aimed at **evidence expansion** — new imports from the
engineer-session folders. Jeff's 2026-08-29 steer says the **first curriculums
are very small and come from real experience already in Dormant** (the
`D:\00` recovery), while engineer-session material arrives **later**, when we
train real agent reasoning.

The gap: **no proposal on the table yet compiles the first trainable
conversation/Soul curriculum from material that is already imported,
verified, and protected.** The FFCS fills that gap. It needs no new import, no
credential screen (no new sources are opened), and it makes the head
tournament launchable immediately after ratification, on evaluation families
that measure exactly the capabilities Jeff named: Soul use, conversation,
long-field chewing, iterative ticking, and society membership.

## 2. Verified ground truth (checked on disk this turn, not assumed)

- `State/dormant/experience_v1` import
  `718f33bf470b90f3f1b2375de3aeb8bf4f47f48c5440b21395f9a2d0b3933ba6`
  holds **59,875** exact recovered records (verified line count), with exact
  text, SHA256, and provenance per record. 4.1 GB total. Read-only protected.
- `runtime/trainer/sessions.py` — the first deterministic historical compiler
  — already yields a 59,858-example Heart grounding session and a
  **14,205-example observed conversation session** from that import
  (Source of Truth §Trainer). Both are stamped
  `serving_promotion_eligible=False` because recovered sources lack a
  trustworthy conversation identity.
- `training/living_reasoning_curriculum.py` — synthetic mechanism curriculum
  for FIRST/REFINED/CONSOLIDATED: exact addressed edits, no-op/abstain,
  proposal use, conflicts, current-field authority. Mechanism only; no real
  text, no Soul-content dependence beyond structure.
- `training/lived_reasoning_curriculum.py`
  (`EvidenceQualifiedLivedCurriculumCompiler`) — outcome adjudication exists
  but waits on runtime episodes with explicit outcome evidence. Correct to
  wait; FFCS does not touch it.
- `training/living_reasoning_d64.py` — `CausalLivingUnroll` proves the exact
  three-phase surface with the `f32le` Soul codec at phase boundaries.
- `runtime/heart/turns.py` — deterministic turn finalization: the consolidator
  authors `response_draft`; Heart appends the framed exact turn to
  `conversation_history` and clears `user_input` in one atomic delta.
  `user_input` and `conversation_history` are reserved from core authorship in
  this first form.
- Shared Field schema v3: the canonical `identity` region (ID 10) is
  always-attended and unmaksable — but **its content has not been authored
  yet** (by design; only `IDENTITY_STEWARD` may amend, and no amendment has
  happened). See §7 D3.

## 3. What Jeff asked the first curriculums to teach (2026-08-29 steer)

1. **Soul use** — inhale before perception, exhale after action, carried
   causally across phases.
2. **Conversation** — the exact turn form: `user_input` arrives, the core
   authors `response_draft`, Heart finalizes into framed history.
3. **Massive-FFN chewing** — integrate evidence spread across many pages into
   a short draft (Candidate A's capacity lives in `ffn_dim=131072`; give it
   material that rewards that shape).
4. **Iterative ticking** — draft and scratch updated across REFINED passes
   and across sequential turns, tick over and over.
5. **Society membership** — attend the designated home rail, offer proposals,
   inspect the brothers' board, handle consolidator duty when it is your
   turn; understand the cores are members of one greater individual that
   speaks to the world through the 16D substrate.

## 4. FFCS: six bounded families (v1 caps; sizes are proposals, revisable)

All families compile runtime-faithful fields: identity region present, home
rail bound, coverage receipts, no truncation, page = compute unit. Every
case carries source lineage IDs (experience_v1 record IDs where applicable),
an explicit teaching-eligibility class, a lineage-inherited split, and
counterfactual family membership. v1 total ≈ **460 small cases** — CPU-cheap
to compile and verify.

### FFCS-A — Identity and rail presence (~40 cases)

Extends the existing mechanism curriculum. Every case's compiled field
includes the identity region (present-but-pending until §7 D3 lands); the
core attends its designated home rail with a coverage receipt. Negative
cases: emissions that attempt identity writes, masked/dormant addresses, or
unauthorized regions must fail closed. This is the first slice of ledger
active-flag 4 (masked-address gap) at curriculum level.

### FFCS-B — Turn mechanics from observed D00 conversations (~120 cases)

Ground: real recovered exchanges from the 59,875-record import (exact
`user`/assistant text pairs, provenance-bound). Two variants:

- **Copy-grounded (VERIFIED_TARGET, narrow scope):** the recovered response
  text is presented as visible evidence in the case field; the target is an
  exact DELTA writing it into `response_draft`. This trains mechanical turn
  authorship — region choice, exact addresses, payload text, EOS, Unicode
  transport — with a deterministic oracle. It does **not** claim the response
  was good; the eligibility class records the scope honestly.
- **Observed-response (OBSERVED_ONLY, zero target weight):** the exchange
  appears as context with no target. Distribution exposure to real dialogue
  form without correctness claims.

The Heart-side bookkeeping (history append, `user_input` clear) is
deterministic `turns.py` behavior and is never a core target, matching the
reserved-regions rule.

### FFCS-C — Grounded short answers (~100 cases)

A short question in `user_input` plus a small evidence excerpt surfaced into
`cortex` (deterministic recall materialization at build time — the existing
capped `dormant_recall` path). Target: a short `response_draft` that quotes
the exact evidence string. Oracle: exact substring presence plus a citation
receipt binding to the evidence span. VERIFIED_TARGET at narrow scope; this
is the honest first form of "answer from memory."

### FFCS-D — Society fixtures: board and consolidator duty (~80 cases)

Fixture ports (already proven in `tests/test_reasoning_circulation.py`):
2–3 participants with constructed FIRST boards.

- **Board inspection:** brother proposals that are correct / plausible but
  stale / conflicting / wrong; targets adopt-correct, override-stale-with-
  current-field, or refine.
- **Consolidator duty:** when the core is designated consolidator, produce
  the final delta combining brothers' non-conflicting edits and resolving
  conflicts against current canonical evidence.
- Negative cases: foreign-core proposals, corrupted categorical streams, and
  base-stale proposals must be rejected, not adopted.

### FFCS-E — Sequential-turn episodes: multi-tick Soul carriage (~60 cases)

2–4 sequential ticks in one session lineage: turn 1 completes (framed
history append via the deterministic finalizer), tick 2 begins with the
carried HOT Soul; targets remain `response_draft`/`scratch` deltas; `scratch`
carries a short explicit note that a later tick must read. This is the
"tick over and over, constantly updating response_draft and scratch"
capability, built as a **sequence of existing single-tick unrolls** with the
Soul codec handoff between them — no new tick kind, no canonical-authority
change. §7 D4 asks Codex to confirm no contract violation in chaining
single-tick unrolls inside one training episode.

### FFCS-F — Long-field chew (~60 cases)

Multi-page evidence (3–6 pages), short response; distractors between causal
facts; answers require combining two facts from different pages. Exercises
the ordered recurrent sweep plus FFN integration — the architecture's
actual strength — with every page visited and accounted, none truncated.

## 5. Counterfactual families (per case class; cf_probe standard)

- **Soul:** zeroed / swapped-irrelevant / stale. Soul-relevant cases must
  degrade under ablation; Soul-irrelevant cases must not change.
- **Field:** evidence removed / swapped-wrong / irrelevant injected /
  reordered cause.
- **Board:** correct brother vs stale brother vs corrupted stream.
- **Authority:** identity writes, dormant-address authorship, unauthorized
  regions — must all fail closed.

Per Layer 13: counterfactual input-use probes are the only accepted evidence
a mechanism is used. A loss going down proves nothing by itself.

## 6. Gates (smoke-gate doctrine; no promotion from any of these)

- Per-family teacher-forced lane accuracy above the constant-output floor.
- Free-running exact delta rate reported separately — never a v1 pass
  condition.
- Soul-dependence differential (cf), field-dependence differential (cf),
  board-use differential on FFCS-D.
- Identity/masked/unauthorized write rejection at 100%.
- EOS/termination exactness; Unicode transport-expansion accounting;
  no-truncation accounting (skipped-and-counted only).
- **Tournament integration:** the 1x64 / 2x32 / 4x16 tournament evaluates on
  mechanism curriculum + FFCS under identical fixed gates, so the head
  comparison measures the capabilities Jeff actually wants, not just
  mechanism.

## 7. Decisions requested from Jeff

- **D1.** Ratify ChatGPT's §14 principles 1–10 (as amended by Kimmy's
  §11-as-sketch concession). I concur with all ten.
- **D2.** Add **P11: first-form curricula are small by decree** — v1 stacks
  are bounded (≤ ~500 cases), CPU-verifiable, and no family scales until its
  smoke gate moves (loss falls, task metric above constant floor). Encodes
  Jeff's "very small" steer as doctrine rather than informal intent.
- **D3.** Author the `identity` region content via an `IDENTITY_STEWARD`
  amendment — the constitution text naming membership in one greater
  individual communicating through the 16D substrate. FFCS-A verifies
  presence and immutability either way, but the *meaning* of membership
  needs Jeff's words. Until amendment, every FFCS manifest carries an
  explicit identity-content-pending marker.
- **D4.** Confirm FFCS-E sequential-tick episodes are within runtime
  doctrine (Codex to review as part of his round).
- **D5.** Sequencing: FFCS build starts immediately after ratification (no
  new import required); the engineer-session harvest proceeds on its own
  track after the credential screen; both lanes feed the tournament.

## 8. Answers to Kimmy's Q1–Q8

- **Q1:** Per-event records as the provenance layer plus derived
  turn/session manifests. Do not choose (aligns with ChatGPT).
- **Q2:** Tier on two axes — transport pages and procedural depth; knees for
  scheduling only. The lesson unit defaults small per Jeff's steer.
- **Q3:** Compositional outcome bundles with scoped adjudication (aligns).
  FFCS v1 VERIFIED_TARGETs are only copy/quote-mechanics at narrow scope;
  observed conversation stays OBSERVED_ONLY.
- **Q4:** Quarantine over redaction, and screen before publication — both
  (aligns with ChatGPT's correction and Kimmy's concession).
- **Q5:** Rollouts canonical; side stores index-only pending a uniqueness
  audit (aligns).
- **Q6:** Observed-only by default; doctrine discussion is never supervision
  (aligns).
- **Q7:** Campaign manifest graph linking immutable sessions, no physical
  flattening (aligns).
- **Q8:** Both lanes, in this order: FFCS is the immediate evidence base
  from already-imported material (per Jeff's steer); the harvest is the
  parallel expansion lane that later feeds agent-reasoning curricula.

## 9. Flags

- **FLAG [ADVISORY] — identity content pending.** Curriculum can teach the
  *structure* of membership now; the *meaning* requires Jeff's amendment
  (D3). No FFCS case fabricates identity text.
- **FLAG [ADVISORY] — sequential-tick doctrine check.** FFCS-E chains
  existing single-tick unrolls; Codex should confirm this stays inside the
  runtime contract (D4). If not, FFCS-E drops from v1 without affecting the
  other families.
- **FLAG [ADVISORY] — ledger self-correction.** Hermes's prior canonical
  event `evt-20260829T130645272999Z-hermes-deep-repo-inspection` was written
  with a malformed `turn` object (a leading-space `" summary"` key instead
  of `summary`). The line is valid JSON and remains untouched per
  append-only law; this turn's event carries `corrects_event_id` for it.
- No BLOCKING flags: FFCS opens no new sources, imports nothing, and trains
  nothing without gates.

## 10. What FFCS does not claim

v1 teaches mechanical turn authorship, grounded quoting, board procedures,
consolidator duty, Soul causality, and long-field integration. It does not
claim conversational intelligence, judgment, or response quality. Free-form
conversational quality remains gated on runtime-lived episodes with explicit
outcome adjudication (the standing TARGET-QUALITY GAP). This boundary keeps
the no-fake-organs law intact.

## 11. Sequencing and spend

1. Jeff rules on D1–D5.
2. FFCS-A/B/C build + tests (CPU; hours, not days).
3. FFCS-D/E/F build + tests (CPU).
4. Bounded tournament smoke on mechanism + FFCS; then bounded CUDA campaign
   per Layer 13 smoke-gate doctrine — long runs only after smoke passes.
5. Harvest lane proceeds in parallel once the credential screen exists and
   passes.

Spend: curriculum construction is CPU-only over an existing verified import;
training is bounded by the existing tournament/smoke discipline; no cloud
spend is required for v1.

— Hermes / glm-5.3:cloud / 2026-08-29