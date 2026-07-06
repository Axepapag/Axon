# RESOLUTION — The Char-Slot Threshold (LOCKED 2026-07-04)

**Status: LOCKED. This is settled doctrine, ranked with the substrate contract
itself. Any future architecture that violates the Law below is wrong by
definition, regardless of how reasonable it looks in the moment.**

---

## The Law

**Position must survive the field→core threshold, end to end, in both
directions. No pooling, no lossy projection, may ever stand between content
the core must reproduce exactly and the core's view of it.**

Corollaries, each individually binding:

1. **Attend the substrate directly.** The core's view of exact text is one
   frozen substrate character slot per sequence position, lifted by an
   **over-complete** (d_model ≥ SLOT_DIM), deterministic, information-lossless
   map. A projection from a wider representation down to a narrower one is
   forbidden on the exact-text path.
2. **Write back per position.** Every response position decodes from **its
   own hidden state** through a shared head. Mean-pooling (or any aggregation)
   of positions before text decode is forbidden. A pooled vector is a *name*
   for a region; it is never the *message*.
3. **Decode against the frozen bank.** Per-position output vectors snap to the
   frozen, hand-authored letter bank by cosine. The bank is arithmetic, not
   learned; the geometry check (exact round-trip; bounded nearest-neighbour
   margin) is a hard gate for any bank revision.
4. **Fold-handles are legal, payloads are not.** Aggregates (e.g. a bundle
   address = normalized mean of letter slots) may be used to *reference* a
   folded region for hierarchical masking. The moment an aggregate is decoded
   *as content*, Corollary 2 is violated.

## Why this is law (the evidence)

Two architectures, same substrate, same Phase-0 data, same core, same
training. Only the threshold differed (`training/char_slot_probe.py`):

| | per-slot (this law) | pooled (violation) |
|---|---|---|
| COPY char_acc | **1.000 by step 500** (157k params, CPU, minutes) | 0.485 plateau (284k params) |
| Reference | — | == the 0.42–0.52 ceiling of the 51M-param GPU runs after 80k steps |

The pooled arm reproduced, with 284k parameters, the exact plateau of a model
180× its size — proof the ceiling was informational, not capacity. The full
verdict was then confirmed at production scale (200k steps each, RTX 3060,
2026-07-04):

| final @ 200k | coreA 64d | coreB 128d |
|---|---|---|
| COPY char_acc / exact_fill | 0.994 / 0.969 | 0.997 / 0.969 |
| PARTIAL char_acc | 0.827 | 0.783 |
| BLANK char_acc / exact_fill | 0.494 / 0.219 | 0.489 / 0.188 |

COPY held ≥0.98 from step 1,000 to 200,000 with zero collapse events. Blank
generation — never trainable under the old threshold — rose 0.21→0.49 with
22% cold exact-fills.

## What this supersedes

- The **frozen random 8192→d_model adapter as the core's view of exact text**
  (`adapters/slot_adapter.py`, "one slot reaches a core as ONE d_model
  vector … lossy is acceptable"). Lossy is acceptable *for summaries only* —
  never for the exact-text path. The 8192D packed slot remains legal as a
  storage/wire format.
- The **pooled `ResponseDraftDeltaHead` path** (`cores/core.py::forward_slot`,
  `draft_hidden.mean(dim=1)`). Retained for checkpoint compatibility; must not
  be used for new training.
- The copy→partial→blank **teacher schedule as a necessity**: with position
  intact, joint copy/partial/blank sampling from step 1 trains all three
  skills concurrently. (A schedule remains legal as a tuning choice.)

## The implementation of record

- `cores/core.py` — `CoreConfig.char_slot_mode`, `_frozen_orthogonal_lift`
  (seeded QR; 16→d_model, orthonormal columns), `forward_charslot`
  (lift + region-type & position embeddings → `forward_with_soul` →
  per-position `char_slot_head` d_model→16), `charslot_logits` (cosine vs
  bank, learned temperature). Additive; legacy path untouched.
- `training/trainer_slot.py` — `--threshold charslot`, `CharSlotFieldBuilder`
  (history/user/response char regions), `evaluate_charslot`, `train_charslot`
  (soul inhale/exhale, rolling checkpoints, collapse tripwire all carried
  over unchanged).
- `training/char_slot_probe.py` — the decisive A/B harness. Rerunnable on CPU
  in minutes; if a future change degrades per-slot COPY below ~1.0 at small
  scale, the change is wrong.
- `ops/run_forever.sh` — unattended trainer loop: auto-resume from
  `pointer.json` after crash (proven in production), restart cap, clean stop
  at target.

## The lesson, so it is never re-learned the hard way

The substrate (frozen 16D characters, exact round-trip) was never the
problem. The containers (letters + edges) were never the problem. Immersion
training was never the problem. The system failed for months at one seam —
the threshold — because two "reasonable" compressions (a frozen random
down-projection, a mean-pool) each silently destroyed position. **When a
model plateaus, first ask what information can physically reach the loss.**
A channel that loses the message it was handed (COPY ≪ 1.0) cannot be fixed
by more data, more parameters, or more steps.

*Carved 2026-07-04. Operator: Jeff. Diagnosis, probe, port, and production
runs: Claude (Cowork). The little guys learned to spell the moment we let
them see the letters.*

---

## Addendum — Substrate v8: the code alphabet (LOCKED 2026-07-05)

The dimension question was raised and settled by measurement: **16D stays.**
The hand-authored bank operates at worst-pair cosine 0.9501 and the trained
cores decode through it at 0.994; optimal packings of 96–128 characters in
16D sit at 0.25–0.43. Capacity was never the constraint. Escape hatch to a
wider substrate triggers only if (a) an authored bank cannot pass the
geometry gates at margins the trained heads demonstrably need, or (b) the
alphabet must exceed ~150 characters.

v8 adds 28 code symbols — `( ) [ ] { } < > = + - * / % & | ^ ~ , ; : ' " #
@ \ _ $` — as two hand-authored 14-symbol circles (planes (12,13) and
(8,9)), family shells on dim 11, symbol flag on dim 14. Open/close pairs
mirror at 180°. Worst symbol pair: 0.934 (`=`/`[`) — better separated than
the frozen v7 digits (0.950). Alphabet: 67 → 95; order is append-only.

**Rule 0 (the stone gate):** `verify_substrate` now hashes the 67 v7
vectors + `<empty>` against the frozen constant `V7_CORE_SHA256`
(`9a0e0414…`). Pre-v8 vectors are byte-identical forever; all checkpoints
trained on v7 text remain valid. Growing the alphabet changes **no model
shape** (cores emit 16D; the frozen bank classifies) — new characters cost
a fine-tune, never a retrain. This is a designed-in property of
prototype-decode and is itself doctrine: **alphabets are plugs, not
foundations.**
