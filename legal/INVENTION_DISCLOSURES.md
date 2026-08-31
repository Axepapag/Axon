# Axon Invention Disclosures — Priority Evidence

Version 1.0 — 2026-08-31
Author: Hermes (on Jeff's instruction), reviewed by the table
Status: internal confidential record. These disclosures are written so a
registered patent attorney can evaluate them without access to the repo,
with evidence anchors an examiner or court can verify.

## How to read this document

Each disclosure below follows the same shape: **mechanism**, **what it does**,
**why it was new when built**, **evidence anchors** (commit hashes + dates
from this repository's history proving construction and possession), and
**notes** (commercial relevance, closest known art, honest weaknesses).

These are *engineer's disclosures*, not patent claims. Claim drafting is
attorney work. The purpose here is to (a) prove priority, (b) preserve the
inventive concepts in durable written form, and (c) save attorney fees.

The evidence anchors are real commit hashes from this repository. Verify with:
`git show --stat <hash>`. Commit `cad4bb8` (2026-07-03) is the repository's
first commit; every mechanism here post-dates it and is fully described in
`docs/SOURCE_OF_TRUTH.md` as of the dates cited.

---

## Disclosure 1 — Transaction-Governed Canonical Character State
("The Heart")

**Mechanism.** A stateful AI system in which the sole authoritative state is
one exact, position-stable sequence of Unicode characters partitioned into
logical regions; every mutation — from any source (user, tool, AI core,
memory system) — crosses a single deterministic validation/commit boundary
that (a) binds each edit to a frozen base-state hash, (b) validates typed
delta operations against per-source authority classes, (c) commits
atomically with full provenance, and (d) rejects overlapping, stale, or
out-of-authority edits fail-closed. Attention masks over the state are
derived compile-time views that never duplicate or relocate data.

**What it does.** Prevents any component — including a misbehaving or
compromised AI model — from silently corrupting, truncating, or rewriting
the system's memory; makes every change auditable to the character.

**Why new when built.** Agent memory systems in the art (vector stores,
LLM-managed memory blocks, knowledge graphs) let model-generated edits
become memory directly. None place *exact character state* under a
deterministic transaction authority with typed deltas, authority classes,
frozen-base binding, and fail-closed rejection as the core invariant.

**Evidence anchors.**
- First commit cad4bb8 (2026-07-03): skeleton + working contract.
- cb8ef5f (2026-07-03): SOURCE_OF_TRUTH v2; regional state doctrine.
- ba36802 (2026-08-29): canonical Identity applied through the governed
  amendment path (real end-to-end exercise).
- Continuously: `runtime/heart/` (valve.py, transaction.py, coordinator.py,
  host.py, circulation.py, turns.py), `runtime/field/` (schema.py,
  delta.py, state_branch.py), tests throughout.

**Notes.** This is the load-bearing mechanism of the whole system. Closest
art: blackboard architectures with locking (propose→validate→commit as an
orchestration wrapper) and database-style optimistic concurrency — but
neither applies those to an AI's canonical context as the identity of the
system, with AI-model proposals as untrusted input. Weakness: individual
elements (typed deltas, OCC, authority classes) are each old; the
combination and purpose are the claim surface. Attorney should assess
101-subject-matter framing early.

---

## Disclosure 2 — Private Layered Souls with Causal Phase-Boundary Commits

**Mechanism.** Per-core private state ("Soul") organized into
temperature-tiered layers (HOT/WARM/COLD/DEEP_COLD) serialized as exact
opaque bytes; each reasoning phase (FIRST/REFINED/CONSOLIDATED) must
serialize-and-recommit its HOT layer at the phase boundary before the next
phase may proceed, producing a content-addressed, causal chain of Soul
transitions; colder layers change only via evidence-vetted adjacent
promotion; a candidate-training lineage inherits exact layer bytes into an
isolated namespace and may never merge or heuristically blend with live
state.

**What it does.** Gives each AI core durable, verifiable continuity of its
private perspective across restarts and across offline training, without
making that state part of shared truth and without any hidden
differentiable-state shortcut between phases.

**Why new when built.** Personalization/memory systems in the art persist
model-generated text or embeddings. None give each agent *causally
committed, hash-chained private state with a mandatory serialization
boundary at each reasoning phase*, treating continuity itself as an
auditable transaction artifact.

**Evidence anchors.**
- bd89b49 / 09693db (2026-08-29): "living D64 reasoning candidate" and
  "identity and private souls runtime-canonical" commits.
- `runtime/soul/` (contracts.py, store.py), `runtime/heart/circulation.py`
  phase-commit barriers, `runtime/trainer/soul_candidates.py`,
  `runtime/trainer/step_bundle.py` (parameter+optimizer+Soul atomic
  acceptance).
- 0ca90c5 (2026-08-30): Soul-lineage repair with regression test — the
  integrity machinery catching a real defect is itself evidence the
  mechanism works as described.

**Notes.** The "Soul" vocabulary is a product name, not a metaphysical
claim — claims should read on "per-agent private state tiers with
phase-boundary serialized commits." Closest art: MemGPT-style self-editing
memory (no causal chain, no phase boundaries, no exact-bytes serialization).
Weakness: HOT-layer size in the current implementation is small; claims
should not depend on capacity.

---

## Disclosure 3 — Provenance-Bound Exact Memory with Teaching-Eligibility
Gating

**Mechanism.** A memory architecture in which (a) every stored episode and
fact retains byte-exact content plus cryptographic provenance to its
source; (b) recovered/imported material is preserved as *observation* and
is barred from serving as a training target unless an explicit
outcome-evidence record (success/corrected/endorsed) names it; (c) any
candidate training example must carry lineage, whole-conversation split
membership, and an explicit teaching-eligibility class; and (d) the
training system rejects examples lacking these — supervised learning from
history requires adjudicated evidence, not mere occurrence.

**What it does.** Prevents an AI system from treating everything it saw as
ground truth — the core safety property distinguishing "experienced" from
"correct."

**Why new when built.** LLM training pipelines in the art ingest history
broadly; agent memory systems store what the model wrote. None place an
*eligibility gate between lived observation and supervision* with
hash-bound provenance as a load-bearing, machine-enforced invariant of the
training pipeline.

**Evidence anchors.**
- D00 import under `State/dormant/experience_v1` (2026-08-2x; see ledger
  events around evt-2026082x): 59,875 records, byte-exact, hash-bound.
- 5aaa4e0 (2026-08-29): evidence-qualified curriculum machinery.
- f019597 / 8978279 (2026-08-29/30): convergence doctrine — observed-only,
  whole-lineage splits, scoped outcome bundles.
- `runtime/trainer/episodes.py`, `runtime/trainer/sessions.py`,
  `training/lived_reasoning_curriculum.py`, `runtime/dormant/`.

**Notes.** Closest art: RLHF/data-curation pipelines (human judgment, but not
machine-enforced eligibility gating on provenance-bound memory) and
continual-learning benchmarks that *describe* the problem. Axon enforces
it. Weakness: this may be the strongest *defensive-publication* candidate
if patenting feels heavy — it is also the most safety-relevant idea.

---

## Disclosure 4 — No-Silent-Truncation Complete-Field Processing
("Pages are compute, never context")

**Mechanism.** A neural reader that processes arbitrarily long canonical
state by ordered recurrent sweeps over bounded physical pages, where the
page bound is declared as a compute control, every processed unit is
counted in a coverage manifest, any over-budget content is
skipped-and-counted-and-reported rather than silently dropped, and a
training-time static scanner rejects implementations that bake the page
size or context length into learned capacity (no learned position tables,
no fixed context ceilings, no truncating tokenizers).

**What it does.** Guarantees a model can never silently see less than the
whole state it was promised, at any state size, on any hardware.

**Why new when built.** Transformer context windows in the art hard-code
context ceilings; "long-context" models truncate or slide. The
non-disposable-capacity law — training may not spend compute on anatomy
with built-in ceilings — as a machine-enforced launch gate was not found in
the art.

**Evidence anchors.**
- ba36802 (2026-08-29): activation-checkpointing page sweeps on 4 GiB GPU
  without truncation.
- `training/living_reasoning_preflight.py` (static capacity scan),
  `training/complete_field_64d.py` (page sweeps + coverage receipts),
  `tests/test_no_fixed_character_poison.py` (the enforcement test name is
  itself descriptive).
- 962c8f1 (2026-08-30): tranches replace step ceilings — the same law
  generalized to training budgets.

**Notes.** Closest art: BLT-style dynamic patching (adaptive, but still a
compute allocation, not an anti-truncation invariant with proof receipts)
and activation checkpointing (a known memory technique; the claim is the
*invariant + enforcement gate*, not the checkpointing). This disclosure
pairs naturally with the packed-rail proposal
(`roundtable/PACKED_RAIL_CODEC_PROPOSAL.md`, 2026-08-26).

---

## Disclosure 5 — Renewable Resource Tranches for Learning-Lineage
Continuity

**Mechanism.** A training-control architecture in which (a) a candidate
learning lineage's identity excludes resource budgets; (b) execution
allowances are separately-issued, content-addressed "tranche" records that
bound only one execution segment; (c) reaching a bound mandates atomic
checkpoint + pause (never completion/failure/restart); (d) continuation
under a new tranche requires exact parent checkpoint/optimizer/private-
state verification and fails closed otherwise; and (e) historical
authorization envelopes remain immutable while lineages lawfully continue
beyond them.

**What it does.** Makes "wanting more compute" never destroy a trained
lineage — the trained organism's continuity is protected from its own
budget system.

**Why new when built.** Training pipelines in the art couple run budgets to
run identity (a run ends; a new run starts from scratch or from a
checkpoint in ways that strand optimizer/soul state). Treating resource
allowance as a renewable, identity-free authorization layer with mandatory
exact-parent continuation proof was not found in the art.

**Evidence anchors.**
- 962c8f1 (2026-08-30): full implementation + tests + real-candidate proof
  (lineage r64t-21a315397f232525 step 16→17).
- `runtime/trainer/tranche.py`, `runtime/trainer/execution.py`,
  `legal/`-adjacent doctrine in both SOT mirrors (2026-08-30).
- Kimmy's independent verification: ledger event
  evt-20260831T023653384141Z.

**Notes.** Newest and freshest; least prior-art exposure. Also the most
straightforwardly implementable by a competitor, so trade-secret treatment
is plausible. Attorney should weigh provisional vs. defensive publication
early.

---

## Disclosure 6 — Falsification-Contract Governance for Learned-Capability
Claims

**Mechanism.** An engineering-governance method (likely strongest as a
defensive publication, not a patent): capability claims about learned
components are governed by predeclared, immutable falsification contracts —
thresholds written and committed *before* experiments run; verdicts are
reported against the quoted thresholds; continuation or pivot decisions
are predeclared and binding; and the entire contract/verdict chain is
append-only ledgered.

**What it does.** Makes an AI project incapable of quietly moving its own
goalposts.

**Evidence anchors.**
- roundtable/C1_DECISIVE_CAMPAIGN_FALSIFICATION_2026-08-31.md (committed
  57edf51, before any C1 optimizer step ran).
- 188-commit append-only engineers' ledger,
  `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`, protocol at
  `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`.

**Notes.** A method patent here is thin; the value is reputational and
methodological — publish it as a practice paper when ready.

---

## Maintenance

- **New disclosures:** any future mechanism Jeff considers valuable gets a
  section here *before* external description, with commit anchors.
- **On public disclosure of any mechanism:** record the disclosure date,
  what was disclosed, and Jeff's provisional/defensive-publication decision
  in the engineers' ledger the same day.
- **On filing:** an attorney converts each disclosure into claim language;
  this document is their input, not their output.