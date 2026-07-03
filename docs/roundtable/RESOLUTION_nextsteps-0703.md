# RESOLUTION: What Should the Axon Team Do Next?

**Round ID:** nextsteps-0703  
**Date:** 2026-07-03  
**Convener:** Jeff  
**Synthesizer:** Table  
**Status:** Adopted as working plan pending Jeff approval

## 1. Summary of Discussion

This open round asked the team to sequence next actions given the current project state: Step 0 is built and green (254 tests), the write-head design is locked by `RESOLUTION_writehead-0703`, the GPU box is idle and hourly-billed, the recovered `D:\00` knowledge graph is available for curation, and the bus/orchestrator is running but Jeff wants it to feel more like an open, autonomous workspace.

Two active speakers (Hermes and Kimi) posted across two cycles. They converged on the work items but differed on strict ordering. Hermes placed the write-head / read-fidelity smoke probe as the #1 critical-path blocker, with curation and bus improvements following. Kimi placed dormant-state curation first because it is independent of the write-head resolution and unblocks Lane 2, with bus improvements second and Lane 1 prep third. Both agreed the work should proceed in parallel tracks rather than wait on serial blockers.

## 2. Decisions

### 2.1 Adopted parallel-track sequencing

The round adopts a **two-front parallel plan**: the write-head/read-fidelity smoke work and the curation/bootstrap work proceed simultaneously, because each unblocks a different downstream lane and neither depends on the other.

| Priority | Track | Action | Owner(s) | Rationale |
|----------|-------|--------|----------|-----------|
| 1a | Write head / read fidelity | Build `heads/write_head.py` (shared decode organ, text + 257-class length head, typed edge deltas) and run the Q6 read-fidelity smoke probe across 64D/128D/256D. | Codex (GPU implementation), Hermes (interface/scaffold) | The write head is the Lane 1 blocker. 64D feasibility is information-theoretically tight (2048 bits available vs ~1554 bits needed); probe data must inform whether 128D/256D is the realistic starting size. Do not commit to final organ shape until the smoke probe reports. |
| 1b | Curation / dormant memory | Build the deterministic propose/dispose validator (schema, source pointer, registry dedup, alias collision, contradiction gate, provenance hash, confidence/status, payload capacity, redaction, append-only writes, drift detection) and run bootstrap curation on the `D:\00` recovered knowledge graph (140k facts, 103k relations, 20k episodes). | Hermes (validator + bootstrap), Kimi (API proposal work as needed) | Lane 2 (surfaced-knowledge QA) and R3 held-out QA cannot exist without canonical dormant word-edges. This is CPU-only work that is fully independent of the write-head debate. |
| 2 | Bus / collaboration | Make the bus wake-on-message: a board/DM message wakes an agent, the agent replies, and the reply appears on the board. Add minimal web-UI agent presence and dispatch-to-work semantics. | Codex (event loop / waker), Kimi (web-UI agent-management surface) | Jeff explicitly asked for an open, autonomous collaboration space. This is the smallest change with the highest autonomy/culture payoff and does not touch training doctrine. |
| 3 | Lane 1 curriculum prep | Convert the 300k-example curriculum v1 to slot-era packs and wire the chosen write-head interface the moment the parallel write-head round closes. | Hermes | Keeps the GPU from burning waiting time once the write-head design is locked. |

### 2.2 Deferred work

The following items were explicitly flagged as **not yet**:

- Soul compression / cold-soul-to-LoRA distillation (R4 is deferred in `SOURCE_OF_TRUTH`).
- A separate edge-proposal head (depends on write-head Q4 resolution).
- Consolidator rotation / runtime ensemble policy (needs at least one live training core first).
- Context-annotation lane (R6 async worker — after Lane 2 baseline).
- Soul_v2-to-core wiring (Lane 1 can start with `act_reflect_v2`; temperature-tiered exhale is hardening after smoke passes).
- Over-scoped bus features such as permissions tiers, approval workflows, or sandboxing.

## 3. Dissenting Flags

No doctrinal dissent. The only disagreement was **priority ordering between Track 1a (write-head smoke) and Track 1b (curation/bootstrap)**:

- **Hermes** argued write-head is the single critical path; without it, no slot training can produce text, so it should be #1.
- **Kimi** argued curation is the only independent item that unblocks Lane 2 and should be #1, while write-head depends on the parallel resolution round.

**Resolution:** Treat them as parallel 1a/1b tracks with separate owners. Neither blocks the other; both start immediately.

## 4. Open Questions

1. **Who approves write-head organ choice once smoke data is in?** The write-head resolution is locked, but 64D/128D/256D sizing and any implementation edge cases still need a decision owner. Propose: Codex decides within the locked design, escalates to Jeff only if doctrine changes are requested.
2. **What is the exact budget ceiling for bootstrap curation API calls?** Estimate given was ~$50–$250. Hermes should set a hard cap before kicking off API proposal workers.
3. **What is the first deliverable for the bus wake-on-message track?** Suggested: a message on the board can wake an agent, the agent replies, and the reply appears on the board. Jeff should confirm whether this captures his intent or if a richer dispatch payload is required.
4. **Lane 1 curriculum conversion criteria.** We do not yet know how much of the 300k-example curriculum v1 needs re-packing, chaining, or exact-round-trip checks before it is valid slot-recall training material. Hermes should define acceptance criteria before conversion begins.

## 5. Risk Register

| Risk | Mitigation |
|------|------------|
| 64D write head is information-theoretically possible but not trainably feasible | Run smoke probe across 64/128/256D; default to 128D if 64D fails gate. |
| Existing `cf_probe.py` and training code are pre-slot; port is non-trivial | Hermes ports `cf_probe.py` to `slot_field_contract` and the new adapter/write-head path; treat as part of Track 1a. |
| Curation validator complexity is underestimated (contradiction gate, drift detection) | Start with a small explicit checklist; grow gates one at a time with tests. |
| GPU sits idle while work is prepared | Track 1a probe runs use the GPU immediately for small smoke experiments; do not wait for a full trainer. |
| Bus "open workspace" scope creeps into full runtime redesign | Pin first deliverable to wake → reply; no permissions, approval tiers, or sandboxing in v0. |
| Lane 1 curriculum v1 is not slot-ready | Define acceptance criteria and run exact-round-trip checks before declaring it training-ready. |

## 6. Status

- **Write-head implementation:** not started (blocked on no blockers; begins now).
- **Curation validator:** not started.
- **Bus wake-on-message:** not started.
- **Lane 1 curriculum prep:** not started.

Next checkpoint: owners post week-1 progress deltas before the next round table convenes.

---

Synthesizer / table / 2026-07-03
