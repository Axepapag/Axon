# Codex review — query-side pointer-motor A0 boundary

**Agent:** Codex / GPT-5
**Date:** 2026-09-22 (America/Chicago)
**Mission:** run the ratified query-side canonical-address pointer motor on the local GTX and decide whether the address gate is ready for the next stage.

## Verified implementation

The run used the canonical `Trainer` entrypoint in
`scripts/train_pointer_motor_scaffold.py`, on CUDA, with a fresh candidate
lineage derived from the pinned step-24 donor. The candidate architecture is
`living-d64-english-461ae22e32b8f62c13776f95`; all donor parameters remained
frozen except the four tensors in the bounded query scaffold:

```text
pointer_address_query.0.weight
pointer_address_query.0.bias
pointer_address_query.2.weight
pointer_address_query.2.bias
```

The trainer executed the complete FIRST/REFINED/CONSOLIDATED Soul transition
chain for each whole episode and recorded accepted bundles, optimizer receipts,
checkpoint identities, and Soul receipts. It did not promote the candidate and
did not mutate the live reasoning module, Heart, or cloud state.

## GTX evidence

The first dense continuation ended at accepted step 9. Its 68-case heldout
evaluation recorded:

| metric | value |
| --- | ---: |
| exact source top-1 | 1/68 (0.0147059) |
| mean target probability | 0.00991396 |
| minimum target-vs-runner margin | -1.40187 |
| query variance mean | 0.0463118 |
| target-address coverage | 68/68 |
| frozen key cross-episode address accuracy | 0.982050 |

The additional eight-step continuation ended at accepted step 17. Its exact
68-case heldout evaluation recorded:

| metric | value |
| --- | ---: |
| exact source top-1 | 0/68 |
| mean target probability | 0.0122075 |
| minimum target-vs-runner margin | -1.98360 |
| query variance mean | 0.143629 |
| target-address coverage | 68/68 |
| frozen key cross-episode address accuracy | 0.982050 |

The gate requires exact top-1 `1.0`, minimum margin `0.25`, positive query
variance, and complete target-position coverage. The candidate therefore has
`assignment_completed=false` and `mastery_gate.passed=false`.

The result is not a compute or checkpoint failure. The query scaffold is
receiving gradients and producing nonconstant representations, while the
existing memory keys already carry canonical position information at 98.2%
cross-episode accuracy. The learned request-to-query address bridge still does
not select the certified source. The change in target probability is not a
mastery signal; it remains below the uniform 1/68 target probability and exact
accuracy regressed from 1/68 to 0/68 at the later boundary.

## Interpretation

This run falsifies the narrow claim that adding a directly parameterized
canonical-address query scaffold and training only its four tensors is already
enough to teach the current decoder to point. It does **not** falsify the D0
diagnosis that the request-to-query bridge is the failed mechanism, and it does
not justify changing the memory keys, increasing the model, changing the Soul,
or launching Kaggle. No production or live-core claim is made from this
candidate.

The candidate lineage and all accepted checkpoints remain valuable forensic
evidence. They must remain separate from the live module until a later,
explicitly governed gate passes.

## Verification

- `python -m pytest tests/test_pointer_motor_training.py -q` — 4 passed.
- `python -m pytest tests/test_pointer_oracle_diagnostic.py tests/test_pointer_bootstrap_curriculum.py tests/test_d64_pointer_transition.py -q` — 24 passed.
- New modules compiled with `python -m py_compile`.
- Both reports identify `device: cuda`; `nvidia-smi` after completion showed a
  GTX 1650 at 0% utilization with 439 MiB baseline allocation.
- No Axon trainer or diagnostic process remained active after the run.

Raw report: `State/tmp/pointer_motor_query_horizon.json` (SHA256
`FE2634479301621CB77AE100D16F924DEB9225FF33D4F0F6D7A3B50F0A43C50A`).
Accepted pointer receipt: `State/training/trainer/candidates/r64-english-reasoning/pointer-motor-candidate-b78f14c368ed6943d5e2/accepted_steps/pointer.json` (SHA256
`2F8704D3382A4EDAE3CBE4FB6DE14172C70151F155B0DE66B6CDB007D9AA9BBC`).

The next action is governed by the blocking flag
`roundtable/flags/CODEX_POINTER_MOTOR_QUERY_A0_GATE_FLAG_20260922.md`.
