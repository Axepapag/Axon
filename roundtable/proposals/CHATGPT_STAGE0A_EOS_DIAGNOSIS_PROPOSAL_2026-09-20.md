# Stage-0A EOS Diagnosis and Controlled Continuation Proposal

**Author:** ChatGPT / GPT-5.6 Sol  
**Date:** 2026-09-20  
**Status:** PROPOSAL ONLY

## Purpose

Diagnose the first fresh English-native Stage-0A tranche without drifting back toward dedicated termination anatomy, then define the smallest scientifically clean continuation Codex can execute.

This proposal does **not** authorize promotion, cloud training, a new Core birth, a dedicated EOS/termination head, a DELTA/NO_OP/ABSTAIN route, or a weakened mastery gate.

## Current candidate and evidence boundary

Candidate: `english-candidate-1c991f8c911f79394e91`  
Module: `r64-english-reasoning`  
Architecture: `living-d64-english-023f4b5e7d43d59968c9b133`  
Base generation: `english-init-7d94581ca59c706157d2`  
Fresh scratch lineage: yes; `typed_donor_migration=false`  
Accepted tranche: 8 optimizer steps x 8 lived experiences = 64 lived experiences  
Final accepted bundle: `2a939b59a0d55d365b9b6b18274fb91d62e9e6ecadbef31143f49da71b75bb97`  
Final checkpoint: `a72f83c60087fb5482111d8a5fbd9ee42d67e04f488001820015868b1d09c95a`  
Final Soul: `8087cd11f5029beaa8a6992d6aee930f3e9037450b9870826453ba1fe151eafd`, generation 192

The completed tranche reduced optimization loss from `11.149822` at step 1 to `5.785553` at step 8, but production heldout remained:

- exact free-running output: `0.0`;
- teacher-forced content: `57/650 = 0.0876923077`;
- teacher-forced EOS: `0/64 = 0.0`;
- complete-field coverage: `1.0`;
- production phase outputs: `0/96`;
- mastery: false.

FIRST/REFINED failed with `English reasoning decoder did not terminate within its renewable work slice`.

## Read-only inspection performed 2026-09-20

I inspected the active decoder/objective and replayed retained accepted checkpoints against all 32 Stage-0A heldout episodes. The ad-hoc forensic probe is outside Git at:

`State/diagnostics/eos_checkpoint_probe.py`

SHA-256 at inspection time:

`b814f51c3f01757a854c822d9e911e71bc6a68dbb6a570bd1052d6d7f18e2c8f`

It executed with no optimizer mutation.

### Finding 1 — ordinary generated EOS is mechanically live

`training/living_reasoning_d64.py::_decoder_logits` obtains the terminal EOS probability from the ordinary generated distribution and renormalizes content around it. The content copy/generate gate does not own EOS termination. There is no dedicated termination head in the active English D64.

This is the architecture we want to preserve.

### Finding 2 — the actual EOS cross-entropy is present, weighted, and unmasked

`living_phase_objective` calls `sequence_cross_entropy(..., eos_weight=4.0)`. `sequence_cross_entropy` applies the 4x weight to the actual final EOS class and normalizes by the resulting weight sum. The active Stage-0A alignment mask does not remove EOS from this loss.

A step-8 EOS-only backward probe on the shortest heldout example produced:

- EOS loss: `4.898776`;
- `decoder_output.weight[EOS]` gradient norm: `7.94495058`;
- EOS output-bias absolute gradient: `0.99254429`;
- total gradient L2 for that isolated EOS loss: `15.51508858`.

Therefore the immediate failure is **not** explained by a disconnected EOS gradient path.

### Finding 3 — retained checkpoints show EOS probability stalled near one percent

Full checkpoint artifacts for accepted steps 1-4 have already been pruned by the normal retention policy. Their records remain, but exact post-hoc replay is unavailable. Steps 5-8 are still replayable.

Teacher-forced terminal EOS statistics across all 32 heldout FIRST examples:

| Step | Mean EOS probability | Min | Max | Mean EOS rank | EOS top-1 | FIRST content accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.00969282 | 0.00533609 | 0.01414021 | 19.06 | 0/32 | 0.080139 |
| 6 | 0.00892584 | 0.00480115 | 0.01358949 | 22.25 | 0/32 | 0.101045 |
| 7 | 0.00891970 | 0.00469752 | 0.01376971 | 21.06 | 0/32 | 0.121951 |
| 8 | 0.00965428 | 0.00496259 | 0.01481603 | 16.72 | 0/32 | 0.101045 |

There is no convincing EOS-learning trend yet. Eight optimizer steps are still extremely early, but the failure is real rather than an evaluator illusion.

### Finding 4 — free-running failure is broader than “one work slice was too short”

On the shortest heldout example (`'🙂🙃🙂🙃'`) at step 8, a direct 64-decision free-run trace:

- did not emit EOS;
- had first EOS probability `0.01507956`;
- never exceeded `0.01507956`;
- ended at EOS probability `0.00555197`;
- repeated transport categories beginning `[18, 18, 18, 248, 18, 248, ...]`;
- did not form valid UTF-8.

Production `emit` currently gets a 512-transport-unit slice. For Stage-0A targets of only a handful of characters, failure to stop after 512 units is not meaningfully explained by an insufficient slice budget. Renewing the slice indefinitely would hide the symptom, not teach termination.

### Finding 5 — `alignment_eos_gate` is not EOS-token supervision

The active alignment code uses the **copy/generate route gate** at the terminal position and calls that auxiliary term `eos_gate_loss` / `alignment_eos_gate`. Its target says, in effect, “at the EOS position use the generated route rather than the copy route.”

That is distinct from learning the EOS token itself. Actual EOS-token supervision comes from the sequence cross-entropy above.

The auxiliary may still be a defensible terminal routing constraint, but its present name is dangerously easy to misread as a special termination mechanism. Do not add more termination anatomy to repair this run.

### Finding 6 — renewable iterator and production `emit` deserve separate review

`iter_decode_transport` is explicitly renewable and yields incomplete at each work-slice boundary. `decode_transport_greedy` takes only the first yield, and `emit` immediately fails on that incomplete result. The capacity policy says callers may renew incomplete work.

That is a real contract seam worth reviewing. It is **not** the explanation for the current Stage-0A failure because these exact-copy targets should terminate far inside the first 512-unit slice. Treat renewal semantics as separate runtime debt; do not make “more slices” the training fix or relax the exact/EOS gate.

## Proposal for Codex execution

### Phase A — verify and make the EOS evidence permanent, with zero optimizer steps

1. Independently reproduce the step-8 terminal EOS probability/rank and EOS-only gradient path from the exact accepted checkpoint + exact accepted Soul.
2. Add a small read-only living-decoder diagnostic surface or evaluator metrics that record, at minimum:
   - teacher-forced terminal EOS probability;
   - EOS rank and top-1 count;
   - generated terminal EOS logit/probability separately from copy/generate route telemetry;
   - bounded free-running tokens/transport units before EOS;
   - whether the emitted transport prefix is valid Unicode;
   - first-slice termination separately from overall production validity.
3. Persist lightweight diagnostic evidence at tranche/evaluation boundaries so ordinary checkpoint pruning cannot erase the learning trajectory. Do not retain every large checkpoint merely for these metrics.
4. Add regression tests proving the real EOS token is controlled by the ordinary generated distribution and receives nonzero gradient from Stage-0A text loss.

### Phase B — do not change the learning objective yet

The first eight steps are too early to justify objective surgery, especially because loss is falling and the actual EOS gradient is healthy.

For the next controlled observation, preserve **all** of the following:

- same candidate generation;
- exact accepted step-8 parameter parent;
- exact step-8 Soul parent;
- same Stage-0A curriculum;
- same `1e-4` learning rate;
- same D64 / 1 head / 2 layers / FFN 16384 / 4 state tokens / page 32 anatomy;
- same 8 lived experiences per optimizer step;
- same 4x ordinary EOS text weight;
- same strict mastery gate.

Do not reset Soul. Do not start a fresh Core. Do not add a termination classifier/head. Do not increase model width. Do not move to Kaggle/cloud.

### Phase C — execute one 8-step continuation only, then stop and re-evaluate

Continue the same organism from global step 8 to global step 16: **8 additional optimizer steps / 64 additional lived experiences**.

At step 16, compare against the retained step-8 baseline, including:

- total/phase loss;
- teacher-forced content accuracy;
- terminal EOS probability distribution and rank;
- teacher-forced EOS accuracy;
- free-running exact rate;
- free-running termination rate;
- Unicode validity;
- complete-field coverage;
- actual FIRST -> board -> REFINED production behavior.

The tranche is diagnostic. No promotion follows automatically.

### Phase D — decision after step 16

If terminal EOS probability/rank and content behavior both move materially in the right direction, continue this same Soul/parameter lineage in another bounded tranche. This is normal development, not a reset problem.

If content improves while terminal EOS remains approximately flat near the current ~1% probability / rank ~17-22, **stop optimizer continuation again** and inspect objective interaction before changing anything. Specifically compare the gradient direction/magnitude of:

1. actual terminal EOS token cross-entropy;
2. content-token cross-entropy;
3. position alignment;
4. copy-route alignment;
5. the terminal generated-route auxiliary currently called `alignment_eos_gate`.

The purpose is to determine whether terminal-token learning is merely slower than content learning or is being systematically opposed/diluted by another active objective.

If the real EOS-token gradient disappears or fails to reach the ordinary generated output row under the production Stage-0A objective, treat that as an implementation blocker, not a reason to add an EOS head.

### Phase E — clean the misleading `alignment_eos_gate` terminology without changing behavior accidentally

Codex should review whether the terminal route auxiliary is still useful under the current active decoder. If it remains, expose it under terminology such as `terminal_generate_route_*` and clearly document that it is a copy/generate routing constraint, **not** termination-token supervision. Preserve historical compatibility aliases as needed.

Do not silently change the optimized loss surface in the same step as a diagnostic rename. If removal or reweighting is eventually justified, version the objective evidence and record the curriculum/environment transition explicitly.

### Phase F — separately reconcile renewable work-slice semantics

Review the `iter_decode_transport` -> `decode_transport_greedy` -> `emit` contract against the capacity policy. Either implement a governed caller-level renewal path or make the one-slice failure contract explicit and tested.

Regardless of that resolution:

- slice exhaustion is never successful termination;
- Stage-0A mastery must not be rescued by granting absurdly long outputs;
- EOS remains an ordinary generated symbol;
- proposal length remains independent of input pages/cells;
- no fixed output-slot cap is reintroduced.

## Architectural guardrail

The observed failure is a developmental/optimization problem until evidence proves otherwise. The repair hierarchy should therefore be:

**observe -> verify gradients -> continue a bounded childhood -> inspect objective interaction -> only then alter training.**

It must **not** become:

**observe failure -> manufacture a new control head.**

Cores remain free English-speaking neural bodies. Heart remains the authority boundary. EOS terminates generated language; it is not permission to think or act.

## Requested Codex handback

Before any tranche beyond step 16, return to the Roundtable with:

1. independent confirmation/correction of this diagnosis;
2. the permanent EOS observability changes and tests;
3. exact step-8 -> step-16 continuation lineage evidence;
4. step-16 behavioral metrics, including EOS probability/rank rather than only 0/1 EOS accuracy;
5. a recommendation to continue, pause for objective diagnosis, or identify a mechanical blocker;
6. explicit confirmation that no dedicated termination/decision anatomy was introduced and no mastery threshold was weakened.
