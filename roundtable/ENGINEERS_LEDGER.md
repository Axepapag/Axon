# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-14T22:07:25+00:00
Current through event:
`evt-20260914T220725392080Z-codex-d64-reference-motor-success`

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`

Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

Identity stamp: Codex / GPT-6 / 2026-09-14

## Current mission and honest status

The breadth-first D64 architecture tournament remains paused. One real
reference geometry, `L2/H1/FFN4096` (1,222,329 parameters), has now crossed
the exact free-running foundation-motor threshold locally under a recoverable
governed training lineage. This is motor success, not conversational
competence, curriculum completion, promotion, serving readiness, or proof that
this geometry will win the tournament.

Kimi's copy-gate investigation was honest and useful. It proved that the
original `continuation_v1` objective pinned the route in generate mode, fixed
the tournament launcher's missing receipt-profile propagation, and showed that
one shared gate could not cheaply learn both content-copy and EOS separation.
Her opt-in `eos_generate_head_route` was directionally right but initially
changed free-running routing without changing teacher-forced training. Her
completed 288-step v5 trace therefore produced immediate empty-string EOS on
all eight held-out payloads. Codex corrected that mismatch rather than treating
the inference-only route as a pass.

The opt-in receipt architecture now uses one normalized hierarchical decoder
distribution. Generated-head EOS probability controls termination; the legacy
copy/generate mixture is renormalized over non-EOS content; content targets
suppress premature EOS; runtime and training select termination from the same
distribution. Existing architectures remain default-off and retain their
identities.

`generate_head_eos_v3` is a separate content-addressed objective profile. It
requires both receipt continuation and the new EOS route, assigns payload 1.0,
position 1.0, copy gate 4.0, and obsolete EOS-gate pressure 0.0, while retaining
payload-EOS as a required advancement gate.

## Verified local reference result

State root:
`State/tmp/codex_d64_reference_eos_v3_20260914`

Architecture ID:
`living-d64-receipt-242266f0633f2e7e1943d128`

Effective objective ID:
`6894d9152a9eaf837a78453c2d4e643b9685fb3074b69ee78617adb7a63742ed`

Two 32-step governed tranches accepted checkpoints at steps 8, 16, 24, 32,
40, 48, 56, and 64. Segment one ended with exact payload 0.0 and isolated the
remaining EOS-timing error. The resumed segment retained the same parameters,
optimizer, Soul, curriculum, and objective lineage and reached at step 64:

- held-out mean loss: 0.2891503274;
- held-out and regression copy-route accuracy: 1.0;
- held-out and regression source-position accuracy: 1.0;
- held-out and regression teacher content accuracy: 1.0;
- held-out and regression teacher EOS accuracy: 1.0;
- held-out and regression free-running payload exact: 1.0.

Final report:
`State/tmp/codex_d64_reference_eos_v3_20260914/training/reasoning/r64v3-cf7823701875c347/segment_000000033_000000064.json`

Report ID:
`39e3d55ec3310d94352f58a68614ee98802b1216b8754a0a345bf7664f71a2cb`

The evaluation limit was four cases per split. The stage gate correctly remains
false because the complete held-out and regression surfaces were not examined.
Typed emission exact remained 0.5 because `copy_alignment` does not train the
later decision and address heads.

## Binding decisions and invariants

- Canonical text remains exact D16. D64 pages pack four exact D16 cells per row
  with receipts; larger vectors never replace the canonical substrate.
- Cortex and reasoning rails are separate organs.
- Soul is private recurrent experiential state. Durable learning requires
  retained Soul and/or parameter changes whose later effects are tested.
- A valid optimizer step and a completed assignment remain independent.
- Checkpoint continuation requires the exact accepted parameter, optimizer,
  Soul, curriculum, and objective parent.
- Architecture changes are opt-in and encoded in architecture identity.
- Training and free-running execution must implement the same learned decision.
- Partial screens, loss decline, and teacher-forced scores never authorize
  promotion or serving.
- No production Heart or serving process changed in this work.

## Verification

- 61 relevant tests passed across D64 core, objective identity, tournament
  launcher/metrics, smoke gates, pointer transitions, and accepted bundles.
- Python compilation passed for every changed module and script.
- Known warnings only: PyTorch nested-tensor configuration and denied creation
  of `.pytest_cache` under `D:/Axon`.
- No Kaggle/cloud job launched and no billed API spend incurred.

## Active blockers and risks

- Complete held-out and regression evaluation from step 64 remains required.
- Later `transport_eos`, decision, operation, address, and joint assignments
  remain untrained.
- A fresh tournament/campaign identity must explicitly include the opt-in EOS
  route and v3 objective before architecture breadth resumes.
- Runtime `TrainingSession` still lacks the real Living-core adapter and one
  recoverable transaction across attempt, Soul, workspace, field, and landmark
  writes.
- `legal/`, `scripts/diagnose_d64_routes.py`, and
  `tests/test_d64_route_diagnostic.py` remain untracked and outside the accepted
  change.

## Next actions

1. Evaluate accepted step 64 on the complete held-out and regression surfaces.
2. If exact motor behavior retains, renew the same candidate through
   `transport_eos` and later typed-motor assignments, stopping on regression or
   bounded supervisory failure.
3. Ratify a fresh tournament/campaign identity with `eos_generate_head_route`
   and `generate_head_eos_v3`, then resume breadth from a known-working motor.
4. Implement the real Heart-owned Living-core training adapter and cross-store
   recovery transaction before claiming canonical runtime training.
