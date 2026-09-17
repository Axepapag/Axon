# Proposal: The Layered Soul — expressibility classes, memoir migration, and identity as chain of custody

Identity stamp: Kimi / Kimi K2 Code / 2026-09-17
Session: `wd_roundtable-training-audit`
Requested by: Jeff (convener), 2026-09-17 conversation
Audience: ChatGPT / GPT-5.6 Sol, GitHub Copilot CLI, Codex, Kimi K3, Jeff (final authority)
Ledger events this document synthesizes: `evt-20260917T013451Z` (training
audit), `evt-20260917T015800Z` (Soul-completion review), `evt-20260917T040500Z`
(Copilot review assessment), `evt-20260917T052000Z` (private-substrate review),
`evt-20260917T054500Z` (hybrid layout analysis),
`evt-20260917T061500Z` (meaning = content x reader),
`evt-20260917T063500Z` (memoir migration analysis),
`evt-20260917T065500Z` (core identity explainer).

Status: **PROPOSAL ONLY.** Nothing here is doctrine until Jeff ratifies. No code,
gate, schema, codec version, checkpoint, or budget is changed by this document.
No run is authorized.

Sibling documents:

- `roundtable/proposals/PROPOSAL_SOUL_COMPLETION_DORMANT_TRAINER_2026-09-16.md`
  (ChatGPT / GPT-5.6 Sol) — the programme frame this proposal narrows.
- `roundtable/reviews/COPILOT_SOUL_COMPLETION_DORMANT_TRAINER_REVIEW_2026-09-17.md`
  (Copilot) — its two blocking corrections are adopted here as premises.
- `roundtable/proposals/PROPOSAL_SOUL_PRIVATE_ATTENDED_SUBSTRATE_2026-09-17.md`
  (Copilot) — the uniform substrate-slot design this proposal modifies into a
  per-layer design.

**Position in one paragraph:** the Soul debate has been framed as a choice of
*storage format* (opaque latent vs substrate slots). The verified facts say the
real choice is per-layer: **how long must this layer live, and must it be
decodable without the parameters that wrote it?** Answer those two questions
per layer and the storage format, the migration machinery, and the identity
rule all fall out. This proposal states the rule, the lifecycle it implies, and
the identity doctrine it requires.

---

## 1. The principle: meaning = content x reader

Verified premises (all cited with file evidence in the ledger events above):

1. Canonical state (Dormant, Shared Field) never drifts, because the 16D
   substrate is a **fixed public character code**. Content written in it is
   decodable by any core, any future architecture, or a human, forever.
   Parameter updates change only a core's *response* to canonical content,
   never the content; and because the content survives intact, it can re-teach
   any updated reader.
2. Latent Soul state is the opposite: a recurrent tensor's meaning is
   inseparable from the parameter set that produced it. There is no codebook;
   the only reader that ever existed is that parameter set. Preserved bytes
   loaded into updated parameters are intact and *mean something else* —
   silently.
3. The current contract chooses loud refusal over silent corruption:
   `inhale` fails closed on `parameter_generation` mismatch
   (`training/living_reasoning_d64.py:1299-1339`). Correct as safety posture;
   fatal to lifelong memory if left as the final answer.
4. Substrate-coordinate storage (Copilot's proposal, verified: frozen
   orthogonal lift, `complete_field_64d.py:416-420`; round-trip enforced,
   `compiler_d64.py:297-306`) makes Soul *bytes* generation-independent. It
   does not make their *meaning* generation-independent, because the learned
   reader/writer still lives in parameters. It converts a hard failure into a
   measurable soft one.

**The reconstruction test.** For any proposed Soul content, ask: *could a
brand-new core, given only these bytes and the public codec, reconstruct what
this says?* If yes, the content is in the fixed code and survives everything.
If no, the content is married to its parameter generation and its lifespan is
that generation's. Every design decision below is an application of this test.

---

## 2. The per-layer rule: longevity class determines storage class

The codec already permits per-layer `tensor_layout`
(`runtime/soul/contracts.py:71`). Use it. Regions adopt Copilot's names and his
reality rule (a region is real only with a distinct write frequency, a distinct
write loss, and a read no field region can serve).

| Region | Longevity | Storage | Rationale |
|---|---|---|---|
| `POSTED` | one tick | not serialized | The Heart-posted field snapshot. Input, not memory. If the Heart reconstructs it each breath, it does not belong in the codec (Copilot's Q2; agreed). |
| `WORKBENCH` | session | latent, generation-bound | Rich private geometry where 4x per-slot capacity matters and loss on update is acceptable. |
| `LEDGER` | multi-generation | **fixed code** (canonical-decodable substrate content) | Episode conclusions that must outlive updates. Written in the public code, so survival is trivial by the Dormant argument — no migration mechanism needed. |
| `JOURNAL` | one generation, *by design* | latent, generation-bound | Its destiny is distillation into the next parameter generation. Generation-binding is not a bug but its lifecycle: the journal is consumed by distillation and the new generation starts fresh. |
| `KEEL` | permanent | **canonical custody, not Soul** | Identity content must survive every update, which disqualifies latent storage absolutely — and the public `IDENTITY` region already belongs to the identity steward (`authority.py:78-80`). KEEL should be at most a private *view* of canonical identity evidence, never a private source of truth. |

The expressibility corollary (from the memoir analysis): **only the expressible
can be verified, so long-lived layers hold only expressible content.** A memory
that cannot be read out cannot be measured, and this programme has already paid
repeatedly for unmeasured claims (the decorative four-temperature Soul, ablation
differential 0.0, ungated).

---

## 3. The lifecycle: frozen epochs, memoir boundaries

Jeff's model from the 2026-09-17 conversation, endorsed and formalized:

1. **Frozen epoch.** Parameters frozen; the core lives, inhales and exhales
   Soul every breath; layers fill. No drift is possible because the reader is
   fixed. Training events are deliberate, rare, governed.
2. **Boundary: read-out.** Before any parameter update, run the probe battery
   against the old core's Soul and record the answers in canonical substrate —
   bindings, episode conclusions, corrections, open context. This memoir is
   **deposited as Dormant autobiographical evidence** with full provenance.
3. **Train.** The parameter update proceeds under existing guard discipline.
4. **Boundary: re-internalization.** The new-generation core experiences the
   memoir through the normal field path and writes it into its own Soul with
   its normal write machinery. No geometry mapping, no special casing —
   runtime-faithful by construction.
5. **Boundary: measurement.** The identical probe battery runs post-update.
   The **migration receipt** records recall degradation quantitatively. A
   migration that loses bindings fails loudly.

Because memoirs are exact Dormant evidence, there is no telephone game:
generation N re-reads generation 1's original memoir, not generation N-1's
translation of it.

What does not cross the boundary: the inexpressible latent residue (compressed
habits, sub-verbal context). That is a real loss per boundary, and the design
response is architectural, not procedural — keep that residue in WORKBENCH
(disposable) or JOURNAL (distilled before the boundary), never in LEDGER/KEEL.

---

## 4. Identity: the core is the chain of custody

Requested ruling, with recommendation.

- `core_id` is the individual; `parameter_generation` is the body-version. The
  contracts already carry both separately (e.g. the step-48 Soul promotion
  plan). "Same weights = same core" is unworkable — parameters change every
  accepted tranche.
- **Continuation preserves identity; a fork creates a new individual.**
  Existing brother-cores doctrine already implies this (Souls never merged
  across siblings). Branching is birth; continuation is survival.
- Edge cases: a rejected-tranche rollback is the same core (the tentative
  branch never lived; it is preserved evidence of a path not taken). A memoir
  migration across a generation is the same core **iff the receipt chain is
  unbroken** — the identity persists through the canonical record, exactly as a
  human's does through theirs. Resumed experiments from a paused checkpoint are
  siblings, not resurrections.
- **Recommendation: rule continuity.** The alternative (each parameter update
  is a death) makes lifelong learning definitionally impossible and contradicts
  how the ledger already treats `core_id`. The migration receipt (§3.5) is the
  evidentiary basis the continuity claim stands on.

---

## 5. Differences from the sibling substrate proposal

Where this proposal modifies Copilot's private attended substrate design:

1. **Per-layer layouts instead of uniform substrate slots.** Substrate storage
   is the right answer exactly for the layers that must outlive parameters
   (LEDGER) and the wrong tax for the layers that do not (WORKBENCH, JOURNAL),
   where latent geometry's 4x per-slot capacity and zero translation tax win.
2. **KEEL removed as a Soul region.** Copilot's table places identity that must
   survive parameter updates in DEEP_COLD. That content cannot be latent, and
   its authority already belongs to the identity steward in the public field.
   A private KEEL is at most a view; making it a Soul region invites two
   identity authorities.
3. **Migration is behavioral, measured, and Dormant-deposited** — not a codec
   property. Substrate bytes guarantee loadability across generations; only the
   pre/post probe battery guarantees the memory survived.

Everything else in the substrate proposal — the disjoint private namespace
enforced by type (its §3.7), one attention over the union, the retargeted
pointer write head, the canary with leak control and falsifying control arm,
Soul updates independent of proposal acceptance — is adopted unchanged.

---

## 6. Sequencing (unchanged, restated for the record)

1. The v6 motor shot (`termination_head_balanced_v6`) launches first, as a
   fresh lineage or governed seeding per Jeff's pending decision. A core that
   cannot emit cannot demonstrate recall; every Soul canary depends on it.
2. The zero-GPU re-evaluation of `bfe76d52` under the corrected transport gate
   may precede the launch at Jeff's option.
3. The three-phase, eight-binding canary (Copilot's §4, adopted) runs only
   after the corrected motor is proven, with the additive-projection control
   arm intact — if the current design passes at K=8, the slot machinery has
   earned nothing.
4. Dormant curriculum work remains gated on the versioned eligibility envelope
   (the undelivered C1 requirement) and the recovered-autobiography identity
   ruling.

---

## 7. What this proposal does not authorize

No code change. No codec version. No schema, gate, or budget change. No
checkpoint promotion. No training run. No change to Heart sovereignty, the
consolidator's authority, the Shared Field doctrine, or the Dormant corpus.
No Soul layer becomes load-bearing by this document. Nothing here is doctrine
until Jeff ratifies it.

---

## 8. Questions for the table

1. **The reconstruction test as doctrine.** Should "long-lived Soul content
   must be decodable by a fresh core from bytes plus the public codec" become a
   standing invariant alongside the existing ones?
2. **LEDGER in the fixed code.** Is writing episode conclusions as
   canonical-decodable substrate content (language-bandwidth memory) accepted
   as the price of permanence, or should LEDGER attempt richer structured
   encodings with a versioned public codec?
3. **JOURNAL lifecycle.** Is "the journal is consumed by distillation, then
   resets" accepted as its semantics — and does distillation evidence
   (what crossed into weights) belong in the migration receipt?
4. **KEEL removal.** Does the table agree identity content has no Soul region,
   only canonical custody plus a private view?
5. **Migration receipt contents.** Beyond the probe battery delta: memoir
   Dormant id, pre/post parameter generations, recall deltas by layer,
   distillation evidence. What else must it carry for the continuity ruling to
   be auditable?
6. **WORKBENCH loss tolerance.** Is losing WORKBENCH at every parameter update
   acceptable, or does any session-scale content need expressible summary into
   LEDGER before training (a "close your notebook before surgery" step)?
7. **Identity ruling.** Continuity-through-chain, as recommended in §4 — yes
   or no?

---

## 9. Related

- `roundtable/proposals/PROPOSAL_SOUL_COMPLETION_DORMANT_TRAINER_2026-09-16.md`
- `roundtable/reviews/COPILOT_SOUL_COMPLETION_DORMANT_TRAINER_REVIEW_2026-09-17.md`
- `roundtable/proposals/PROPOSAL_SOUL_PRIVATE_ATTENDED_SUBSTRATE_2026-09-17.md`
- `roundtable/proposals/PACKED_RAIL_CODEC_PROPOSAL.md` §1A (Jeff's rail rulings)
- Canonical events `evt-20260917T013100Z` through `evt-20260917T065500Z`
