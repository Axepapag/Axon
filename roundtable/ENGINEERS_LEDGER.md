# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-10T01:05:00-05:00
Current through event:
`evt-20260910T010500000000Z-hermes-live-organ-demo-server`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
Identity stamp: Codex / GPT-5 family (exact runtime model ID not exposed) /
2026-09-09

## Current mission and honest status

The active research target remains **exact motor writing on physical D64**.
Jeff ratified R1-R12 and the Layer 13 clarification: deterministic continuation
inside a selected Unicode scalar is categorical transport, while route choice,
the exact source anchor, generate output, and EOS remain learned decisions.

Two cloud ablation runs on Kaggle have now been executed, fetched, verified, and
formally adjudicated:

| Metric | Prior Unicode-walk v3 | Receipt D64 v1 (`4b228ccd...`) | Receipt D64 v2 (`2f5687a6...`) |
|---|---:|---:|---:|
| Objective Profile | Standard | `continuation_v1` | `route_eos_balanced_v2` |
| Heldout source position | 0.645 | **1.000** | **1.000** |
| Regression source position | 0.657 | **1.000** | **1.000** |
| Heldout teacher-forced content only | 0.645 | **1.000** | **1.000** |
| Regression content-only probe | 0.657 | **1.000** | **1.000** |
| Heldout/regression copy route | 1.000 / 1.000 | 0.000 / 0.000 | **1.000 / 1.000** |
| Heldout/regression EOS route | 0.000 / 0.000 | **1.000 / 1.000** | 0.000 / 0.000 |
| Heldout emitted EOS token | 0.000 | 0.125 | 0.125 |
| Regression emitted EOS token | 0.000 | 0.208 | **0.333** |
| Heldout teacher-forced token accuracy | 0.364 | 0.475 | 0.475 |
| Heldout exact terminated payload transport | 0.000 | **0.333** | 0.000 |
| Heldout typed-emission exact rate | 0.000 | **0.333** | **0.333** |
| Heldout final mean loss | 0.5760 | 0.7322 | **0.4461** |

### What the Two Ablations Establish

1. Receipt continuation retained 1.000 source-position and teacher-forced
   content-only accuracy on complete heldout and changed-source regression
   examinations. This is strong mechanism evidence, not exact emitted transport.
2. The v2 D64 mixer reached 1.000 copy-route alignment across both probes.
3. The current failure is joint copy/EOS/transport behavior:
   In v1 (copy weight 0.25, EOS 1.0), the gate logit biased positive (always GENERATE;
   EOS=1.000, Copy=0.000). In v2 (copy weight 4.0, EOS 2.0), the gate logit biased
   negative (always COPY; Copy=1.000, EOS=0.000). Parameter inspection of checkpoint
   `a91fcd16` confirmed `copy_gate.bias = -0.00978` with `copy_gate.weight` norm at only
   `0.09427`. This is consistent with bias-dominated routing, but no direct
   counterfactual yet proves that mechanism. v2's exact terminated transport is
   0.000 and its 0.333 typed-emission rate comes from exact supervised phases,
   including empty/abstain-like cases.

## Current D64 receipt evidence

- Decision: `roundtable/decisions/RESOLUTION_D64_POINTER_TRANSITION_2026-09-08.md`
- Local acceptance: `roundtable/reports/D64_RECEIPT_LOCAL_ACCEPTANCE_2026-09-08.md`
- Cloud adjudication v1: `roundtable/reports/D64_RECEIPT_KAGGLE_ABLATION_2026-09-09.md`
- Cloud adjudication v2: `roundtable/reports/D64_RECEIPT_KAGGLE_ABLATION_V2_2026-09-09.md`
- Cloud job v1: `4b228ccd6dcf2bfdddb09eed59d293c162b18486bd20f42ab70d52721c37f669`
- Cloud job v2: `2f5687a63bd691a5c1b3af6823fabaebbae927ad1b74f43271fb5745edc94a5a`
- Candidate v2: `r64v3-a80504169350a5ce`
- Architecture: `living-d64-receipt-823973aed39c1fe14276d2c3`
- Final checkpoint v2: `a91fcd16b5870661a82940c8ea11f27fd90502cf6ec2b5caf78d9faedf07e487`
- Inspected step-120 checkpoint file: `22af77fa0bcedb396f1b1aea189863a2e4299ee33a19ab198cfa6d4abfa7088d.pt`
- Report ID v2: `685024aeda64fe8fc71db334471403fff42f56187bd0d84066886448a5a46b09`
- Report SHA256 v2: `28184f7950dbfddf3efeda8cd28aa5edd4846bcdbac874902b6f1e97180b4607`
- Fetched archive SHA256 v2: `c48c6740877f46c05761ce14a25965956ad7926acceeb1a3d665c2531b718b7c`
- Heldout final loss: 0.576025 (walk v3), 0.732245 (v1), 0.446148 (v2);
  v2 also fell 81.28% from its own 2.382640 step-zero loss
- Complete heldout/regression evaluation: true / true
- Candidate lifecycle: paused at an exact accepted checkpoint, renewable
- Serving promotion claimed: false

## Binding continuity

- Canonical state is exact 16D substrate. Active region masks determine the
  Shared Field; masked vectors remain in Dormant state and are not deleted.
- No character-count truncation or hidden tissue ceiling is allowed. Resource
  tranches are renewable pauses, not lifetime or curriculum limits.
- Heart remains the sole canonical writer. Reasoning cores make proposals;
  consolidator proposals become real only through governed Heart validation and
  commit.
- Physical D64 remains permanently useful if it earns a role. Do not widen the
  rail merely to evade a local curriculum or mechanism defect.
- Do not promote from falling loss, teacher-forced exactness, aggregate metrics,
  or a checkpoint's safe acceptance into Trainer storage.
- Preserve all rejected candidates and immutable evidence. Never overwrite or
  resume across incompatible architecture/objective/curriculum identities.
- Preserve `D:\00`, teammate state, and the private untracked `legal/` tree.
- No learned Heart or reasoning core is serving.

## Active flags

1. **Joint gate/transport failure:** v2 recovered copy alignment but EOS-route
   accuracy and exact terminated payload transport are 0.000. Scalar-bias
   dominance is a plausible hypothesis, not yet a proved cause.
2. **Serving blocker:** durable HeartHost ownership and reload of decoder
   execution state is not integrated.
3. **Metric hazard:** the weak `exact_serving_gate_passed` name was renamed in code
   to `nonzero_exact_output_observed` (progress signal only), with `exact_serving_gate_passed`
   requiring strict 1.0 exact rates across all complete metric surfaces.
4. **Cloud resilience:** optional mid-run off-kernel sync lacked
   provider-managed credentials. The final bundle completed and is verified,
   but future long runs need a secure secret path.
5. **Windows tooling:** default Pytest temp/cache cleanup has local access and
   long-path defects.

## Next recommended shot

1. Add a bounded D64 diagnostic that records per-position route logits and
   compares the untouched v2 checkpoint with a zero-bias clamp. Do not promote
   or rewrite the candidate. This observation-only diagnostic needs no new
   ratification.
2. If the counterfactual confirms bias dominance, compare a longer fresh
   tranche and non-zero gate initialization under new content-addressed
   identities. If it falsifies the hypothesis, inspect contextual separability
   before changing architecture. Present any training or architecture
   intervention to Jeff for ratification first.
3. **Google Cloud meeting**: Operator demo package assembled and verified:
   `roundtable/proposals/KIMI_GOOGLE_ORGAN_DEMO_RUNBOOK_2026-09-10.md`
   (Heart transaction OS, 59,875-record autobiographical memory with measured
   provenance retrieval, spot-native pause/resume lineage, honest D64 receipt
   result). Jeff to review and rehearse before the morning of 2026-09-10.
4. **Wider Rail Migration**: Only after the D64 mechanism is fully closed and ratified,
   implement the width-generic packed compiler (`d_model // 16` lanes) for 512D.

## Google meeting demo package (2026-09-10, deck + presenter briefing added)

Kimi produced the operator runbook
`roundtable/proposals/KIMI_GOOGLE_ORGAN_DEMO_RUNBOOK_2026-09-10.md` in isolated
worktree `D:\Axon-worktrees\kimi-google-demo` on branch
`codex/kimi-google-demo-20260909`; Kimi commit `7597a58` was reviewed and
cherry-picked to `main` as `3a748eb`. Codex then independently reran the live
organ surface (24 Heart/circulation tests and all 10 substrate conformance
gates passed) and tightened the runbook's evidence boundaries. It is a 7-scene,
~14-minute live story:

1. 16D substrate conformance + exact Unicode roundtrip (incl. surrogate
   fail-closed rejection).
2. Per-region masks: attention slider changes the rail view, never the
   canonical body (`field_id` preserved; masked state dormant in place).
3. HeartHost sole-writer boundary: typed consolidator commit + transactional
   receipt; core/unauthorized/malformed proposals rejected fail-closed
   (`test_heart_control_plane.py`, 19 passed).
4. D64 reasoning rail + two-barrier proposal workspace; malformed core output
   rejected but accounted (`test_reasoning_circulation.py`, 5 passed); learned
   cores explicitly not serving. The workspace serializer has D64/D128 tests,
   but the canonical field compiler and learned reader remain D64.
5. Dormant memory: 59,875 hash-chained records; one selected episode dereference
   measured at 0.40–1.30 ms and verified against the index's stored SHA-256,
   with provenance to `D:\00\axon_episodic_memory.db:episodes:1`. Private memory
   text is withheld; full integrity walk is ~103 s and labeled as such.
6. Trainer observability: content-addressed pointer/sentinels at step 120,
   `paused` renewable lifecycle, offline Kaggle job catalog (`--json jobs`).
7. D64 receipt result presented honestly: 1.000 position/content-only and
   copy-route alignment vs 0.475 teacher-forced token accuracy, 0.000 exact
   terminated transport, failed EOS, non-serving.

Known exclusion: `run_axon_heart.py --once` cannot run from a bare scratch
state root (no dormant generation index; `DormantGenerationError`) — the demo
uses the verified circulation tests instead. No training launched, no cloud
touched, and no serving capability is claimed. Operator commands now target the
real `D:\Axon` `main` checkout. The Google ask distinguishes proven D64 tissue
from future D512/D1024 compiler/reader engineering.

Hermes (2026-09-09, `evt-20260909T232500000000Z-hermes-google-demo-deck`) built
the presentation layer the runbook lacked: a 14-slide 1920×1080 HTML deck
`roundtable/proposals/GOOGLE_DEMO_DECK_2026-09-10.html` (truth boundary as
slide 2, per-scene PROVES / DOES-NOT-PROVE boxes, Scene 7 adjudication table
with failures in amber before wins in cyan, presenter notes per slide on the
`N` key, fullscreen `F`, `?slide=N` deep links, print-to-PDF, gliksbot.com
palette) and a presenter briefing
`roundtable/proposals/PRESENTER_NOTES_GOOGLE_DEMO_2026-09-10.md` (rehearsal
cues, verbatim honest sentence and ask, timing safety valve, preflight
additions). Before building on top of them, Hermes re-verified the live scenes
on this machine with the Python 3.12 interpreter: substrate v7 conformance
PASS, heart control plane 19 passed, reasoning circulation 5 passed (all exit
0). Advisory: bare `python` on this box is a 3.11 venv without pytest — demo
commands need the Python 3.12 full install on PATH.

## Live organ demo server (2026-09-09/10, Jeff's runtime-in-action demo)

Jeff directed a runtime-in-action demonstration: attention-mask sliders per
region, the 16D→D64 packing roundtrip, heart beats and rail ticks, the cortex
valve surfacing dormant hits into the shared field, and predetermined Q&A in
place of intelligent cores. Hermes built
`scripts/demo_organ_server.py` (`evt-20260910T010500000000Z-hermes-live-organ-demo-server`):
a browser UI over the PRODUCTION organ APIs on an isolated demo state root
`State/tmp/organ_demo` (own single-writer lease; 8-record demo-authored dormant
corpus with `organ-demo` provenance; real `State/` never touched; learned-core
honesty preserved — header chip and honest fixture placeholder for unscripted
input). Recon confirmed the cortex-valve intuition against the code:
`BeatCoordinator._run_recall` already performs primitive dormant recall —
query extraction from user ingress, retrieval from the real dormant index,
relevance audit, CORTEX surfacing via `DormantEvidenceBridge`, committed under
DORMANT_VALVE authority — so surfacing required no runtime changes. Verified
live end-to-end: valve ingress "what color is the sky?" → heartbeat with 3
commits → 6 records surfaced into CORTEX with full provenance → fixture-core
consolidator committed "The sky is blue." through the real two-barrier
circulation → D64 rail 118–133 rows, coverage complete → canonical roundtrip
exact (466 valid lanes) → `conversation_history` slider at 40% showed the view
diverging from the canonical body with `field_id` stable. Consolidator
rotation verified across ticks (`demo-consolidator` then `demo-witness`).
Regression: `tests/test_dormant_evidence_bridge.py` 8 passed, exit 0 (no
runtime source files were modified). Presenter briefing updated with the
3-minute live story, honest framing, and fallback ladder. Boot:
`PYTHONUTF8=1 python scripts/demo_organ_server.py --port 9201` →
http://127.0.0.1:9201.

## Continuity health

- Canonical ledger: 204 valid unique event lines plus one preserved historical
  blank line through `evt-20260910T010500000000Z-hermes-live-organ-demo-server`.
- `scripts/append_engineers_ledger_event.py` validated and cleanly appended the turn event.
- Kimi CLI is globally pinned to standard K2.7 Coding; its first bounded,
  read-only Codex-directed evidence audit completed without repository writes.
- After this closeout commit, `legal/` remains protected and untracked.
