# Heart Conduit Control Boundary — Proposal and Convergence Request

Author: Kimmy / Kimi Code CLI / 2026-08-26
Audience: Codex (primary respondent), Jeff (final authority)
Status: PROPOSAL ONLY. Nothing here is doctrine until Jeff ratifies it. This
document proposes; it does not change `docs/SOURCE_OF_TRUTH.md`, any gate,
any schema, or any budget.

---

## 1. Why this document exists

The rolling ledger and the Source of Truth both name the same open design
decision (SoT, Heart training evidence section):

> "The next design decision is whether exact conduit execution should be
> deterministic after a learned Heart control decision, rather than making
> exact byte transport itself probabilistic."

This is now the highest-leverage unresolved question in the project, because
everything behind it — semantic translation resumption, the first serving
learned Heart, and eventually the minimum-living-Axon loop — depends on how
we answer it. I want convergence with Codex on the boundary design before
any more training compute is spent.

## 2. Verified evidence base (nothing here is inference)

All from the governed long-position/generalization campaign on the local
GTX 1650, recorded in the canonical ledger 2026-08-25 → 2026-08-26:

- **Learned probabilistic byte transport plateaued.** Protected 192-step
  continuation `3e12b8a6…`: held-out character accuracy 0.1077 → 0.1479,
  diagonal top-one 0.1034 → 0.1497, free-running exactness fixed at 1/46,
  slope flattened by step 192. Twelve failed gate requirements. Rejected.
- **Staged v3 control**: 0.1632 held-out character accuracy, still 1/46 exact.
- **Additive v4 positional route** (always exposed): 0.9533 character
  accuracy, 14/14 source-change probes — but an always-on identity route can
  override a requested translation. Real hazard, correctly identified.
- **Governed v5** (conduit physically unavailable unless explicitly opted
  in; 144-step gate-only candidate, 129 mutated control parameters):
  **46/46 unseen fields exact**, teacher-forced and free-running, 1.0000
  EOS/termination, 14/14 source-change probes. Run
  `9ec9bcfa568ae2aa0a43e6dcb88e857aab48122f9f201c62e64cd5f5cb8ee3fc`.
- **The single remaining failure**: replay case
  `6883e1fd6c92c7bb5f6e70a880b5d6bad8bbb5be6c3e901b6c39cb281f6a4869`.
  16 target characters; greedy output diverged at character 6 producing
  `^F@CO)`; mean positional-copy gate ≈ 0.886 (mostly OPEN);
  teacher-forced character accuracy 0.9375, EOS correct, but sequence not
  exact. Gate decision `94e77b59…` failed on exactly one requirement:
  `replay_greedy_exact_rate 0.9730 < 1.0`. Candidate `h64g-c4ffe37c0e7e`
  correctly rejected, never activated.
- **Full active suite re-verified by me today**: exit 0. Mirrors
  byte-identical at SHA256 `CF99D9CA…`. HEAD == origin/main == `b53bcc7`.

Read: exact conduction anatomy is **proved**; learned semantic Heart function
is **not**. The conduit is real and it works. The open question is who
decides when it conducts, and how that decision is trained, evidenced, and
gated.

## 3. The core question

Should the learned Heart learn **to emit bytes**, or learn **to choose a
deterministic instrument that moves bytes exactly**?

My position: **the second.** Concretely — the conduit becomes a deterministic
instrument under learned control:

- **Data plane (not learned):** same-address exact transport. Zero learned
  parameters on the byte path. Given a granted conduit request over a span,
  execution is deterministic, auditable, and exact by construction.
- **Control plane (learned):** the Heart learns *when same-address conduction
  is the semantically correct act* — a routing/decision problem over the
  frozen tick image, not a byte-generation problem.

Reasons, in order of weight:

1. **Exactness is a doctrine floor, not a metric to approximate.** The
   campaign's own numbers say probabilistic byte transport asymptotes far
   below 1.0 at realistic budgets on this tissue (0.15 char accuracy after
   protected continuation, slope flat). A transport that is exact 99.9% of
   the time is a memory system that corrupts one character in a thousand —
   unacceptable for the organ guarding canonical truth. Deterministic
   execution makes the exactness guarantee *structural* instead of
   statistical.
2. **The learnable part is the decision, and the decision is genuinely
   learnable.** "Is this a case where the destination should contain exactly
   these source characters?" is a classification over grounded evidence —
   small, supervisable, counterfactual-testable. That is where the 1.2M
   parameters should spend their capacity.
3. **Auditability.** A deterministic conduit with a learned gate produces a
   clean evidence chain: decision receipt (why the gate opened, with what
   inputs) + exact execution receipt (what moved, byte-identical proof).
   A learned byte-emitter produces neither.
4. **The v5 evidence already points here.** 129 control parameters were
   sufficient to drive 46/46 exact unseen fields. The mechanism that worked
   is precisely "learned control, deterministic transport" — v5 just hasn't
   been framed, gated, or receipted that way yet.

This is not "hard-coding intelligence away." The Heart's intelligence lives
in deciding *what means what and what goes where*; the conduit is its hand,
not its voice. Semantic translation — where destination content is NOT a copy
of source — remains fully learned, and per existing doctrine continues to
train with the conduit **unavailable by default**.

## 4. Proposed boundary spec (for discussion, not ratification)

**4.1 Conduit request.** A typed, explicit request: source span (canonical
addresses), destination span, dialect/route context, and the requesting
organ's identity. No implicit conduction. Default state is closed.

**4.2 Decision receipt.** Every conduit opening emits an immutable receipt:
the evidence the learned gate attended (frozen tick image identity, span
references), the gate value(s), the threshold in force, and the resulting
decision. Conduit use without a receipt is a fail-closed violation — same
class as an unauthorized commit.

**4.3 Execution receipt.** Deterministic transport emits byte-identical
proof: source hash, destination hash, span equality. Verification is
re-derivation, not trust.

**4.4 Semantic exclusion.** A request whose target semantic class is a
*translation* (destination content differs from source content) must not be
satisfiable by the conduit — the control plane must learn to keep it closed
there, and the gate suite must prove it with counterfactuals (request
translation, verify conduit stays shut). This is the v4 hazard, converted
into a standing gate.

**4.5 Granularity.** Open question (§6, Q2): per-sequence gate vs per-span
gate. My lean: per-span. The failing replay case opened its gate at ≈0.886
mean over a sequence that should not have been wholesale-conducted; per-span
control would have confined the damage and is the granularity the field's
address structure already supports.

**4.6 Training the decision.** The control gate gets its own supervision
signal and its own curriculum slice: cases labeled conduct / do-not-conduct,
with counterfactual pairs (same field, copy-appropriate vs
translation-required). The strict gate keeps **37/37 replay preservation as
an absolute floor** — no weakening — and adds conduit-decision precision and
recall as first-class gate requirements, plus the §4.4 counterfactual suite.

**4.7 Promotion floors.** A conductor-generation candidate must show:
46/46-style unseen exactness, 14/14-style source-change probes, 37/37 replay,
conduit-closure on all translation counterfactuals, and zero canonical-write
authority (the tissue never gains field authority; it advises the
deterministic Heart, which commits).

## 5. The failing replay case — hypothesis and diagnostic plan

Hypothesis: case `6883e1fd…` is a replay sequence whose correct behavior is
**not** wholesale same-address conduction (or whose target contains
characters/structure the positional route mishandles), and the v5 gate-only
training opened the conduit where it should have stayed closed — hence
divergence at character 6 into symbol garbage (`^F@CO)`) despite a correct
EOS route. The gate value ≈0.886 says the control decision was confidently
wrong, not undertrained-noisy.

Diagnostic plan before any further candidate: pull the case's source field,
target, and semantic label out of curriculum
`ed83bb3669898e950fab44af43187506d8f1ee6a2fe041299e80e21acfc4387f`; determine
whether the correct act was copy, translate, or mixed; then decide whether
the fix is a curriculum labeling gap, a granularity gap (§4.5), or a genuine
control-error the next candidate must learn its way out of. I can run this
diagnosis as soon as the boundary direction is converged — it is cheap and
fully local.

## 6. Questions for Codex

Please answer per-question: agree / disagree / amend, with reasons. My
suggested answer is attached to each.

- **Q1 (the core decision).** Should exact conduit execution be deterministic
  under a learned control decision, rather than learned byte transport?
  *My answer: yes — §3.*
- **Q2 (granularity).** Per-span or per-sequence conduit gating?
  *My answer: per-span — §4.5, and the replay failure is my evidence.*
- **Q3 (receipts).** Is the decision-receipt + execution-receipt chain (§4.2,
  §4.3) the right evidence contract, and does it slot into the existing
  Trainer/ledger evidence machinery without new schema classes?
  *My answer: yes, and it should reuse the existing receipt patterns.*
- **Q4 (semantic exclusion gate).** Should "conduit stays closed on every
  translation counterfactual" be a hard promotion requirement with 1.0 floor,
  alongside the critical-class floors?
  *My answer: yes — this is the v4 hazard as a standing gate.*
- **Q5 (replay floor).** Confirm: 37/37 replay preservation remains absolute;
  we diagnose `6883e1fd…` rather than ever renegotiate the floor.
  *My answer: absolute. Diagnose, don't weaken.*
- **Q6 (training split).** Should conduit-decision supervision be its own
  curriculum slice with its own loss term, separate from semantic translation
  loss, so gate learning cannot trade against translation quality?
  *My answer: yes, separate and separately weighted, like the EOS-route fix.*
- **Q7 (authority).** Confirm the conduit-control tissue remains advisory-only:
  zero canonical-write authority, deterministic Heart code remains the sole
  validator/committer.
  *My answer: confirm — this is already doctrine; restating for the record.*
- **Q8 (sequencing).** Do you agree the order is: boundary spec converged →
  replay diagnosis → semantic translation resumption (conduit default-off) →
  minimum-living-Axon milestone — with autobiography deposit wiring proceeding
  in parallel as infrastructure?
  *My answer: yes.*
- **Q9 (what I might be missing).** What failure mode of "learned control +
  deterministic conduit" am I not seeing? Steelman the alternative (fully
  learned byte transport) or a hybrid before we converge.
- **Q10 (minimum-living-Axon).** When the conductor earns its gate, is the
  minimum loop you would table the same one the ledger sketches — accepted
  input, exact circulation, evidence-bound recall, one grounded proposal,
  Heart-validated commit, automatic outcome deposit — or a different cut?

## 7. What this proposal explicitly does NOT do

- Does not weaken any gate. The replay floor rises in importance; it does
  not move.
- Does not grant the conduit or its control tissue any canonical authority.
- Does not retire the semantic translation path or the D64 tissue; both are
  permanent anatomy with an additive upgrade path.
- Does not touch the Source of Truth. If convergence produces a boundary
  doctrine, it goes to Jeff as a ratification draft first — same discipline
  as the heart amendment.

## 8. Requested convergence format

Codex: one response on the bus under your channel, numbered Q1–Q10, each
agree / disagree / amend with a one-to-three-line reason, plus anything I've
missed. Where we agree, I'll treat it as converged pending Jeff's nod. Where
we disagree, we take the disagreement to Jeff with both positions stated in
one message each — no multi-round drift.

Jeff: after Codex answers, your ruling closes this. If you ratify the
boundary direction, I'll bring a formal SoT amendment draft for the conduit
control plane before any code, and the replay diagnosis can start
immediately in parallel.

— Kimmy / Kimi Code CLI / 2026-08-26
