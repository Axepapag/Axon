# Round Table: Adapter Read/Write Asymmetry + The Write Head

Date: 2026-07-03
Convener: Jeff
Participants: Codex, Kimi, Hermes, Claude, ChatGPT
Deltas: `docs/roundtable/<YourName>_delta_writehead.md`
Working under: `docs/WORKING_CONTRACT.md` (read the preamble first)
Prior art (read first): `docs/roundtable/Hermes_delta_adapter_bandwidth.md`

## Why this table exists

RESOLUTION R7 specified two adapter gates that cannot both hold. Hermes found
the contradiction; the flaw was in the spec (Jeff/Claude), not the code:

1. Sizing rule: a slot reaches a core as ONE d_model vector (the compute law
   — 32 slots = 32 attention tokens, CPU residency).
2. Exact text round-trip through the adapter: 256 chars (~1,536 bits) cannot
   survive a frozen projection into 64 floats and come back exact.

The resolution already landed (Jeff's redirect, accepted and built): READ and
WRITE have asymmetric bandwidth needs. This table's job is to LOCK that
resolution into doctrine and settle the one genuinely open piece it created:
the write head.

## What is already decided and built (not up for review)

- READ is lossy by design. The adapter DOWN-projects a whole 8192D slot to
  one d_model summary vector (deterministic sign-projection over text + edge +
  control dims). The core attends over one vector per slot.
- The FIELD is the source of truth for exact text, never a core's compressed
  read. English is decoded from the 8192D field, not from d_model.
- WRITE commits are made exact by construction: any 8192D slot committed to
  the field is codebook-snapped (each 16D block -> nearest substrate code).
- Verified gates replace the impossible round-trip gate: snap-idempotence
  (0/50 changed at 64/128/256D) and separability (0/200 collisions).
- Exactness is proven at the TASK level (recall-lane exact-fill, cf-probed),
  not assumed at the projection. This is the BLT shape.

If you want to reopen any of the above, that is a BLOCKING flag with
information-theoretic evidence, not a preference.

## The open question: the write head

A core cannot specify 256 exact characters through one d_model vector (the
same bottleneck, reversed). So how does a core PROPOSE slot content it wants
written to the field?

Hermes' position (the seed, argue with it): a per-position alphabet
classifier living in the core/trainer, not the frozen adapter. It produces
logits over the ~63-char substrate alphabet for each text position; argmax ->
characters -> char_to_slot -> 8192D -> codebook-snap. Trained with discrete
per-slot cross-entropy (Layer 13a — the loss that already worked for recall
and morphology). Edge (128) and control (32) positions get their own
classifiers or a shared classifier with position-type conditioning.

## Questions for the table

1. WHERE does the write head live — in each core's head, in the shared
   adapter (making it not-frozen), or as a separate shared decode organ (like
   a reverse adapter, one per d_model)? Trade-off: per-core heads let cores
   develop distinct "handwriting"; a shared organ keeps writing uniform and
   auditable and keeps cores focused on reasoning. Cite the compute cost of
   your choice.
2. FULL-SLOT write vs DELTA write. R5 says cores emit typed delta records
   (update_slot, update_response_draft, attach_edge, ...). Does the write
   head emit a full 256-position slot every time, or only the positions a
   typed delta touches? (A core revising three words of a draft should not
   re-decode 256 positions.)
3. POSITION COUNT vs LENGTH. Does the head always emit MAX_TEXT_CHARS
   positions with padding, or predict a length/stop (the capsule-core length
   head lesson)? How is the stop trained without silent truncation?
4. EDGES ARE STRUCTURED, not free text. A write head emitting raw characters
   into the edge payload could produce invalid edges. Should edge writes go
   through the write head at all, or should cores emit typed edge deltas
   (attach_edge{is_a:animal}) that the deterministic packer renders into the
   edge payload — keeping the registry/overflow contract authoritative?
   (Claude leans strongly to the latter: cores propose edges as typed
   operations, never as raw edge-payload characters.)
5. READ FIDELITY FLOOR. The lossy read must still let a core distinguish the
   slots it needs to reason. Separability proves distinctness, not
   sufficiency. What read-side probe proves the d_model summary carries
   ENOUGH for a task (beyond "the vectors differ")? Proposal to beat:
   train a tiny probe to recover slot kind + first-N chars from the d_model
   summary; report recovery rate as the read-fidelity number.
6. WRITE-HEAD GATE. What is the cf-probe-style acceptance test for the write
   head before Lane 1 trusts it? (Draft: given a target slot, the head must
   reproduce it to exact-fill through the discrete loss; zero-input and
   swapped-target controls must fail.)

## Impact if we get this wrong

The write head is the cores' only mouth onto the field. Every delta, every
response draft, every proposed edge passes through it. Character-level
attention already got smuggled in once here (the v1 adapter). The failure
mode to watch: a design that quietly reintroduces per-character token blowup,
or that lets cores write unvalidated structure into the edge payload and
bypass the registry.

## Doctrine floor (not up for review)

Token-free substrate; field-as-source-of-truth for exact text; discrete
per-slot cross-entropy for discrete content; no silent truncation (length/
stop must be explicit and counted); registry + overflow contract remain
authoritative for edges; proof over proxy (write head gated by cf-probe, not
by loss alone); one d_model vector per slot on the read path (compute law).
