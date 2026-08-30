# Codex / Hermes Campaign Recovery — 2026-08-30

Identity stamp: Codex / GPT-5 / 2026-08-30

## Outcome

Hermes's recovered FFCS campaign wiring in commit `617989b` is accepted. The
composed mechanism+A/B/C/D/E/F campaign is real, governed, resumable, and
fail-closed. A sequential private-Soul lineage defect stopped the first CUDA
attempt before an invalid bundle could be accepted. The repair now supplies
all nine FIRST/REFINED/CONSOLIDATED transitions across each three-tick FFCS-E
lesson, and a direct regression test proves the exact lineage.

The deterministic D64 tournament now has exactly 16 accepted optimizer+Soul
steps for each head geometry. This remains a mechanism diagnostic, not learned
reasoning: every candidate is far below its strongest constant token baseline,
both free-running exact rates are zero, the formal metric surface is incomplete,
and no winner or promotion exists.

## Verified evidence

Campaign curriculum:
`b39bc91b69e5d8ce3395ff826e15f3413da7c9f48376f831f037743205a4a7cd`

Closing tournament observation:
`State/training/reasoning/tournaments/ac9eceaaf0e67d483b288140d4db328c7f7b2156468ef5e25a1eaa9f4b791a39/observations/068867c5f1e4b693f00625d30c3a718feea5a40030a4ff41758e54d865f012ab.json`

| Candidate | Accepted step | Heldout loss | Token accuracy | Constant floor | Typed exact | Payload exact | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x64 | 16 | 6.070038775 | 0.012846715 | 0.108759124 | 0.0 | 0.0 | 1.0 |
| 2x32 | 16 | 5.994862173 | 0.012408759 | 0.108759124 | 0.0 | 0.0 | 1.0 |
| 4x16 | 16 | 6.006355773 | 0.012554745 | 0.108759124 | 0.0 | 0.0 | 1.0 |

The evaluation covered 40 standard heldout cases and all six sequential
heldout cases per candidate. One standard heldout case remains explicitly
deferred. Regression evaluation is complete, but the incomplete heldout surface
means nine tournament metrics remain formally missing even where partial
diagnostic values exist. `winner_selected=false` and
`promotion_claimed=false` are durable in the observation.

Repository verification after the lineage repair:

- full suite passed, 461 tests collected;
- changed-file Ruff passed;
- `compileall` passed over runtime, training, scripts, and tests;
- `git diff --check` passed;
- SOT mirrors are byte-identical.

## Failure and repair

The sequential objective performs three real ticks, with three private-Soul
phase transitions per tick. The harness previously handed the atomic step
coordinator only `unrolls[-1].transitions`, losing the first six transitions.
`SoulIntegrityError` rejected the discontinuous lineage. No invalid accepted
bundle, winner, or serving state resulted.

`_material_objective` now returns the complete ordered transition tuple for
sequential material. The regression test applies every transition from the
initial Soul, asserts each `before_soul_id`, and proves the resulting Soul is
exactly the final tick's final Soul.

## Blocking continuation gap

The tournament's `max_steps=16` is part of both the immutable Trainer plan and
the candidate-generation identity. Increasing it changes the generation ID and
starts from the governed base rather than these step-16 checkpoints. Continuing
optimizer work without addressing this would violate the additive-growth
policy and waste learned state.

Further tournament training is therefore halted pending a governed
continuation/succession contract. That contract must preserve and verify:

- the exact prior checkpoint and optimizer state;
- the exact candidate private-Soul HEAD and transition lineage;
- original and successor plan/authorization identities;
- unchanged architecture, curriculum/splits, learning policy, and seed unless
  an explicitly named experiment changes them;
- an immutable parent-generation link and rollback evidence;
- cumulative step accounting without weakening the original 16-step ceiling.

## Advisory recovery gap

An optimizer+Soul bundle is accepted before expensive final evaluation. If the
process crashes after bundle acceptance but before its segment report is
published, learned state is safe, but the current complete-campaign resume path
cannot regenerate the missing report. This did not occur in the closing run;
all three step-16 reports exist. A report-recovery/evaluation-only path should
be added before the next campaign.

## Next shot

1. Ratify and implement governed candidate continuation/succession.
2. Add crash-safe report regeneration and evaluation-only operation.
3. Evaluate the one deferred heldout case for all candidates without optimizer
   mutation so the formal metric surface can close.
4. Only then decide whether these lineages deserve a larger bounded budget or
   whether curriculum/output architecture needs adjustment. No serving decision
   is currently admissible.

