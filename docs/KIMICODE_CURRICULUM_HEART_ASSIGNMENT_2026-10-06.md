# KimiCode: curriculum delivery and HeartHost coordination

Authority: Jeff directly requested this delegation on October 6, 2026. Codex
prepared it for KimiCode's existing Axon session. Please acknowledge on the Iris
bus and proceed with substantial implementation, not only a proposal.

## Deliverable 1: comprehensive, staged memory curriculum

Start with the implemented E0 two-state D512 GRU, but make curriculum/task
definitions independent of the neural architecture. Additional architectures
must be able to use the same tasks and held-out evaluations later. Inspect
current curriculum ownership and existing files before choosing implementation
paths. Do not modify the read-only legacy checkout or source datasets.

Deliver versioned task/episode schemas, deterministic generators, a validator,
small materialized examples, frozen split manifests with hashes/seeds, meaningful
tests, and a plain-language curriculum catalog. Supply configuration schemas and
a handoff Codex can expose through the Lab API; Jeff must operate this through
controls and presets rather than editing Python. Do not generate huge datasets
merely to demonstrate scale: use reproducible generators and bounded previews.

Progressive task families:

1. Immediate exact character recognition/copy across all native95 characters,
   including newline; expand gradually from one character to short sequences.
2. Delayed recall with increasing delays and sequence lengths; separate silent
   ticks from visible distractor inputs so their effects can be measured.
3. Distracted recall: unrelated noise, meaningful-looking distractors, similar
   distractors, repeated decoys and interference from competing memories.
4. Key/value association and selective retrieval among several stored facts;
   vary query position, number of facts and delay independently.
5. Updates/corrections: replacing a value, latest applicable information, repeated
   queries and resistance to obsolete values; explicit task rules for overwrite
   versus retention, with deterministic expected answers.
6. Order and binding: retrieve items in order, distinguish who/what a value belongs
   to, and recall multiple values without swapping their associations.
7. Response/control behavior: correct WAIT/COMMIT/END targets, delayed output,
   exact response length and episode reset. Control labels are metadata; WAIT
   must never be represented as an all-EMPTY content surface.
8. Generalization/stress tests: unseen combinations, novel distractor patterns,
   longer delays/lengths and separate challenge splits. Unsupported capacities
   must be reported explicitly, never truncated or advertised as passed.

For every family define the causal rule, the memory demand, expected output,
curriculum prerequisites, adjustable difficulty, and evaluation criteria.
Learning must not be bypassed by copying expected answers into model inputs.

Freeze disjoint train/validation/test sets before training. Split by the relevant
facts, associations/templates and combinations to prevent leakage, not only by
row number. Deduplicate and validate actual episodes across splits; report the
split strategy per task and a separate generalization suite. Every exact content
character must pass native95 admission; outside characters fail closed without
conversion/escaping. Legacy inventory alone does not establish training readiness.

Report exact response accuracy, per-character accuracy, binding/order errors,
obsolete-memory errors, WAIT/COMMIT/END accuracy, invalid-content/control failures
and recall versus delay/distractor load. Include constant/most-common and
memoryless baselines. Design zero/swapped/irrelevant-state counterfactual tests
and reset-leakage checks through the eventual real execution path. These tests
are planned evaluations until executed; do not claim recall from declining loss.

Presets should make the first curriculum small and understandable while exposing
the full progression. Include a results/report schema that records curriculum
version/hash, task difficulty, split and architecture/checkpoint identity.
Curriculum position and generator RNG must be checkpointable for exact resume.

## Deliverable 2: coordinate and implement the HeartHost handoff

Jeff also authorized delegating Heart wiring. Coordinate with the existing
HeartField owner first. Request an explicit response: either that owner supplies
the coordinator or hands the unimplemented coordinator work to you. An old bus
registration alone is not an active ownership acknowledgment. Review existing
Heart/field/lease/admission/delta code and preserve tested behavior. Announce
the agreed scope and file paths before integration.

Deliver a versioned HeartHost contract and real coordinator joining exact Heart
observations, E0 reasoning/response states, proposal control and Heart admission.
Training and inference use the same tick/proposal/commit path. Keep native codes
frozen; no implicit learned substrate conversion, rails or direct Core field
writes. Distinguish numerical response state from readable private draft and
Heart-committed output. E0 first; optional memories and new architectures stay
separate experiments.

Specify step ordering, reset/delta cursors, private draft versus committed text,
WAIT/COMMIT/END interpretation, masks/authority, inspection snapshots and replay.
Full resume includes both Core states, Heart/field/draft/cursors/control state,
optimizer/RNG/curriculum position where applicable, and proof of no duplicate
commits after interruption. Test uninterrupted versus resumed execution at
mid-output boundaries, invalid proposals, WAIT, END, resets and lease failures.

Keep Codex-owned lab/backend/, lab/contracts/ and backend tests untouched;
propose API amendments on the bus. ChatGPT owns lab/frontend/. Give both owners
the adapter interfaces and evidence needed to enable actual run controls later.
Do not launch valuable/long training, cloud jobs or spend money. The complete
operator vehicle and independent recovery/backup gate remain required first.

## Delivery and coordination

Post receipt and work ownership on Iris. Deliver the curriculum generator/schema
and frozen first-stage examples as the first reviewable milestone, then expand
the families and complete the agreed HeartHost work. Report actual files,
versions, checks and remaining integration limits. Run repository pytest and both
substrate self-tests before claiming success; append canonical ledger events and
maintain the rolling ledger. Preserve all existing histories and credentials.
