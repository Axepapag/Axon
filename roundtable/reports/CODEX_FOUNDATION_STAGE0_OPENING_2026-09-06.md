# Foundations Stage 0 — Exact-Motor Opening Report

Author: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-06

Status: IMPLEMENTED AND PREFLIGHT-VERIFIED; CLOUD OUTCOME PENDING AT THE TIME
OF THIS OPENING REPORT. This document records the launch contract. The durable
segment report and canonical engineer-ledger event record the observed outcome.

## Decision

Axon will not advance to sequence recitation, broad language, conversation, or
reasoning merely because teacher-forced loss falls. The first learned D64 core
must demonstrate exact control of Axon's existing body: typed decision,
operation, region and scalar address; exact payload transport; explicit EOS;
and correct no-op/abstain behavior. This is the ratified Foundations Stage 0.

The prior Stage-1 smoke was useful falsification, not a pass. Kaggle job
`37389c3dce3ab86afedcaf0b5230bb9473e09538a499e93a820a8846257e13b8`
completed 60 accepted steps on a Tesla T4. Heldout loss fell from
10.6677380204 to 5.5501824021 and the core learned the typed control shell, but
its payload remained empty. Free-running exact stayed zero and its
teacher-forced payload accuracy, 0.055556, remained below the 0.062500
constant-category floor. Nothing was promoted and no long continuation was
authorized.

## Implemented training surface

The content-addressed `F0` curriculum manifest is
`a273737cdaa399f9b85e1a8ac91326a255a6c3b63ef880fd1bc1273832be857c`.
It contains 120 complete runtime-faithful episodes: 72 train, 24 heldout and
24 regression. Every split covers copy, insert, replace, delete, no-op and
abstain, and every action is tested in changed-source pairs whose source
identities do not cross splits.

Copy, insert and replace targets carry an additive exact-alignment contract.
The living objective supervises the decoder's already-existing copy pointer at
the exact current-field scalar address. A Unicode scalar expands to all of its
one-to-four native transport cells, in exact order; EOS is still supervised as
generated output. No alternate substrate, semantic token, normalization,
character truncation, or hidden content limit was introduced. Existing target
IDs remain byte-identical whenever the optional alignment contract is absent.

The Stage-0 advancement gate requires, over complete heldout and regression
surfaces:

- at least 0.95 free-running exact case rate;
- at least 0.95 changed-source-pair and payload-pair exact rate;
- at least 0.95 exact rate for every typed action;
- exactly 1.0 complete-field coverage; and
- teacher-forced payload accuracy strictly above its declared constant floor.

This gate governs curriculum advancement only. It is not serving promotion and
it is not a lifetime step, tissue, output, context, field, or curriculum
ceiling. A failed bounded tranche pauses for diagnosis or an explicitly
renewed continuation from the exact accepted checkpoint.

## Trainer and cloud integrity correction

Kimi's ratified hash-verified end-of-run bundle and opt-in mid-run sync were
reviewed before launch. An adversarial concurrency defect was found and fixed:
a slow step-30 upload could previously discover an artifact created at step 60
and mislabel it as step 1-30. Each boundary now freezes only the paths first
observed at that boundary; the worker advances the accepted range after the
prior upload outcome; and transient failures carry exact members into the next
extended retry. A delayed-uploader regression test proves later artifacts
cannot leak backward across the boundary.

Mid-run sync intentionally remains observation-only and opt-in. Final mutable
pointers and progress records are recovered from the complete end-of-run
bundle. No credential is included in a packet, report, or repository file.

## Launch evidence

The bounded launch recipe is
`configs/kaggle/axon_foundation_motor_stage0_alignment_smoke.json`: fresh
non-serving D64 candidate, one 64D head, two layers, FFN 131072, checkpoint
every 15 accepted steps, and a renewable 60-step smoke. Mechanism replay and
F0 teaching remain separate round-robin lanes.

Full-size local CUDA preflight passed with:

- architecture ID `living-d64-675b5ec0f0053cd54c0fbda6`;
- 33,981,879 parameters;
- campaign curriculum ID
  `bc5dfd28fdfd098b45c734a7ff86f349935be4e03718fd18e33954c61a7015e3`;
- plan ID `d28b8abe25f9d89734a07890d149e40975f847f24de9337ddca30e6a3db9156b`;
- candidate generation `r64v2-d39ec38a0f38f523`; and
- preflight receipt
  `096c3e2b63a463580c8a2d6b72ee386d921fa84303480de039fd192f0b92d89c`.

Preflight proves launch anatomy, exact compilation, Trainer authority and GPU
execution only. It is not learned-motor evidence. Cloud launch must originate
from a clean committed revision, and any continuation decision must use the
fetched, hash-verified complete report rather than live-log impressions.

## Verification known at opening

- focused Foundation/Unicode/living-objective tests: 25 passed;
- focused cloud-bundle tests: 21 passed;
- renewal and active-surface hygiene regression: 7 passed;
- changed-file Ruff, compileall, mirror equality and diff checks: passed;
- final whole-repository suite: 578 passed in 1,645.87 seconds; 54 warnings
  (known PyTorch nested-tensor notices plus a non-fatal local pytest-cache ACL
  warning), zero failures.

No serving activation, promotion, destructive migration, long run, or paid
cloud spend is authorized by this report.
