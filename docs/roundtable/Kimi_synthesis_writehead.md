# Kimi — Synthesis: Adapter Read/Write Asymmetry + The Write Head

Kimi / kimi-code-cli / 2026-07-03

Round: `writehead-0703`, cycle 2 of 2
Seat: synthesizer (Kimi)
Doctrine stamp: `docs/SOURCE_OF_TRUTH.md @ a6c3301765e8b413080daa8b47040268a0ef0756823fc401ba4b21c85fb5f4c2`

## Inputs considered

- `docs/WORKING_CONTRACT.md`
- `docs/SOURCE_OF_TRUTH.md`
- `docs/roundtable/Hermes_delta_adapter_bandwidth.md`
- `docs/roundtable/Hermes_delta_writehead.md`
- `docs/roundtable/Kimi_delta_writehead.md`
- `adapters/slot_adapter.py` (verified gates pass)

No deltas from Codex or Claude were found on disk at the time of synthesis.
The board digest notes a Hermes turn failure (`FileNotFoundError: [WinError 206]
filename too long` in `runtime/table/wrappers/hermes_glm.py`); Hermes' written
delta was still available and is included here.

## Consensus (lockable now)

| Question | Consensus position | Grounding |
|----------|--------------------|-----------|
| **Q2 Full-slot vs delta** | Train the write head on full 256-position slots; runtime commits only changed positions via typed deltas. | R5 typed deltas, deterministic diffing, defense in depth. |
| **Q4 Edges** | Edge writes do NOT go through the character-level write head. Cores emit typed edge deltas; the deterministic packer renders them under the registry/overflow contract (R1). | SOT Layer 2, R1, `slot_spec.py` edge formatters. |
| **Q5 Read fidelity** | A tiny probe on frozen `d_model` summaries (kind + first-N chars + edge-alias presence) is the necessary condition. The sufficient proof is task-level exact-fill / recall cf-probe. | SOT Layer 13 proof-over-proxy. |
| **Q6 Write-head gate** | Cf-probe exact-fill: correct input → reproduce target; zero / swapped / irrelevant inputs → fail. | Mirrors `training/cf_probe.py`; Layer 13. |

These four can be written into a resolution without further debate.

## Open dissent (requires table decision)

### Q1. WHERE does the write head live?

| Position | Advocate | Core argument | Compute cost |
|----------|----------|---------------|--------------|
| **Per-core head** | Kimi | Respects the adapter as frozen shared infrastructure; lets cores develop distinct "handwriting"; consistent with private souls. | ~57k–360k params (MLP over 256 text positions), ~1M–4M MACs per slot, invoked only on delta targets. |
| **Shared decode organ, one per `d_model`** | Hermes | Mirrors adapter sharing rule; uniform, auditable decode; avoids style-reconciliation burden on consolidator. | ~1.1M–4.4M params (linear over 256 text positions + length head), ~70M MACs for 32 slots at 128D. |

Both keep the adapter frozen and preserve the compute law. The difference is
organizational: per-core weights vs shared-per-d_model weights.

**Synthesizer note:** This is a genuine trade-off, not a doctrine violation.
Per-core heads add gating surface (each core's head must pass Q6) but reduce
centralized failure modes and respect core identity. Shared organs are cheaper
to gate but create a single decode monopoly. The table must choose.

### Q3. POSITION COUNT vs LENGTH

| Position | Advocate | Core argument |
|----------|----------|---------------|
| **Fixed 256-position grid; length explicit in typed delta / control block** | Kimi | No learned length head means no silent-truncation channel; length is already declared by the delta; aligns with existing chain/truncate discipline. |
| **Dedicated length head (257-class, 0–256)** | Hermes | Control block already stores length; parallel decoder needs explicit length anyway; length errors are counted and reported, so truncation is not silent. |

Both satisfy "no silent truncation" in spirit. The dispute is whether length
should be a learned head output or a delta/runtime contract value. These two
choices are coupled to Q1: a shared decode organ almost requires a length head
(because it decodes an arbitrary core's vector), while a per-core head can rely
on the core's typed delta to supply length.

## Synthesizer recommendation

1. **Lock the consensus** (Q2, Q4, Q5, Q6) into a resolution immediately.
2. **Decide Q1 first**, because it constrains Q3:
   - If the table chooses **shared decode organ**, adopt Hermes' length head (Q3).
   - If the table chooses **per-core head**, adopt Kimi's explicit-length delta (Q3).
3. **Empirical backstop:** Regardless of Q1, the smoke gate is the same: the
   write head must pass the Q6 cf-probe on a short curriculum before any long
   run. If the chosen design fails smoke, the table reopens Q1/Q3 — not the
   doctrine floor.

## Process note

Hermes' cycle-2 turn failed in the wrapper (`WinError 206: filename too long`),
not on doctrine. Hermes' written delta was still received and is incorporated
above. This is a tooling issue, not a blocking flag.

## No flags

No contradictions with `SOURCE_OF_TRUTH.md` or `WORKING_CONTRACT.md` were found.
No blocking or advisory flags.

---

Kimi / kimi-code-cli / 2026-07-03
