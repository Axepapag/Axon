# D64 Tissue Shape — Proposal

Author: Grok 4.6 / GitHub Copilot CLI / 2026-09-06
Requested by: Jeff (convener), 2026-09-06. After a private conversation he
asked for an engineering look at the Trainer and whether 64D is wheel-spinning:
bump to 128 / 256 / 512 / 1024? more layers? more heads? smaller FFNs?
Do not jump straight to Stage-0-v2.
Audience: Kimmy, Codex, ChatGPT, Hermes, Jeff
Status: Jeff ratified invert-shape in conversation 2026-09-06. Both
non-serving 60-step Kaggle mixer smokes COMPLETE (FFN256 / FFN512,
motor v2 `copy_alignment`). Heldout copy-gate and position 1.0; stage
gate failed on regression position 0.667. Extra FFN did not help.
Recommended next: FFN256 copy-alignment renewal. Still not Source of
Truth. Still not serving. See
`roundtable/reports/GROK_D64_MIXER_COPY_ALIGNMENT_SMOKE_2026-09-06.md`.

Jeff also said the earlier personal opinion stays between him and me. This
document is only the engineering.

---

## 1. Problem

The living reasoning core has now failed every honest learned-use gate it
has met: mechanism smoke, FFCS opening, L0-L4, Stage-0 motor control. Loss
falls. Constant-class predictors explain the token metrics. Stage 0's one
real signal was source-address learning; the copy/generate gate and typed
delta collapsed to `DELTA / REPLACE / response_draft / empty payload`.

The tempting reads are:

- "the curriculum is too joint; run Stage-0-v2"
- "64 is too small; buy a bigger brain"

I think both skip the tissue we actually trained.

## 2. Binding constraints (unchanged)

- Canonical substrate stays 16D. Rails are `d_model / 16` packed lanes.
  64 / 128 / 256 / 512 / 1024 are registered packing widths, not a
  hyperparameter slider on Candidate A.
- Additive law: failed candidates are evidence. D64 is not discarded because
  a recipe failed.
- Living core implementation currently **hard-fails** if `d_model != 64`
  (`training/living_reasoning_d64.py` LivingReasoningCoreConfig). A 256D or
  512D brother is new rail tissue, not `--d-model 512`.
- Gates stay. Falling loss is not writing.
- No promotion, no serving, no SOT edit in this proposal.

## 3. What Candidate A actually is

Verified from source and from
`roundtable/reports/LIVING_D64_CANDIDATE_A_REPORT_2026-08-28.md`:

| Property | Value |
|---|---|
| `d_model` | 64 (locked to the physical D64 rail) |
| heads | 1 x 64D (tournament also declared 2x32, 4x16) |
| layers | 2 |
| FFN | **131,072** (2048 x d_model) |
| Soul tokens | 4 |
| page | 32 (compute unit, not a ceiling) |
| trainable params | 33,981,879 |

FFN matmuls alone (two `Linear(64, 131072)` + two `Linear(131072, 64)` plus
biases) are **33,816,704** parameters. That is **99.1%** of the network.

Attention, the dedicated position pointer (`position_query` / `position_key`),
and the so-called copy gate are a rounding error next to those two MLPs.

The page encoder is `nn.TransformerEncoder` over a 64D residual stream.
Almost all "capacity" never mixes positions. Copy is a mixing problem.

## 4. The gate is named backwards and born generating

In `training/complete_field_64d.py`:

- The module is named `self.copy_gate`.
- It is used as **generate_gate**:
  `probabilities = generate_gate * generated + (1 - generate_gate) * copied`
- Init: weight zeros, **bias +1.5**.
- `sigmoid(1.5) ≈ 0.82`, so the network starts **~82% generate**.

The generate vocabulary head sits on the fused decoder state. The copy path
is a one-head pointer over source positions. Stage 0 taught addresses and
still emitted generated empty payloads. That is not mysterious if 99% of the
parameters and the gate prior both prefer generate, while DELTA/REPLACE empty
deletes are majority classes.

Heart translator tissue (shelved, 0.0 fidelity) was a saner shape:
64D, **4 heads**, 2 layers, **FFN 4096**. Still 64D. Still failed its job.
Width did not save it. A giant FFN is not what made Candidate A special;
the giant FFN is what made it a prior-memorizer with a postage-stamp mixer.

## 5. Honest ranking of Jeff's options

**Smaller FFN — first, by a lot.**
Normal transformer FFN is 4x d_model (256), maybe 8x-16x (512-1024) for a
tiny rail. 2048x is not a core. It is a lookup table glued to a 64D wire.
Cutting FFN to 256-1024 *reduces* parameters by ~30M, makes Kaggle/local
memory honest, and stops the MLP from dominating the pointer. This ablation
has not been run on living tissue. The 1-layer / FFN-192 reader default in
`ReaderConfig` is closer to sane and is not what we trained.

**More layers — second, after FFN is sane.**
Candidate A is explicitly "shallow core, depth through ticks." Motor control
of one typed delta is an in-tick binding problem (instruction → pointer →
gate → op). 4-6 layers on 64D with a small FFN is the untested depth bet.
Do not add layers on top of 131k FFN; that is 99% more lookup table.

**More heads on 64D — low leverage.**
The tournament already declared 1x64 / 2x32 / 4x16 with FFN and layers
fixed. Opening step (2026-08-29): all three dropped loss, all three sat on
the constant token floor, no ranking. Splitting a 64D stream into 16D heads
makes each head weaker. The dedicated copy pointer already *is* the extra
head copy needs. I would not spend a campaign on 4x16 until the FFN is no
longer 99% of the animal.

**128D — awkward, skip as the first bump.**
4 heads → head_dim 32. 2 heads → 64. Neither is a clean story. 128D is a
real rail in doctrine; it is a poor first brother.

**256D — first serious wider brother, only after mixer-shaped D64 is tested.**
4 heads x 64 head_dim, 4-6 layers, FFN 1024 (4x). That is a normal small
transformer. It requires new living-core code (the d_model==64 lock), a
packed 256D rail consumer, a new architecture_id, a new Soul codec layout.
Additive: D64 stays. This is months of rail work if we skip the cheap
ablation, and maybe unnecessary if inverted D64 can copy.

**512D — later adult small, not now.**
8 x 64 or 4 x 128. Fine after 256D proves the rail contract. Local 4 GiB
already OOM'd Identity-bearing 64D with the fat FFN. 512D with a sane FFN
can be cheap; 512D with 131k FFN would be comedy.

**1024D — no.**
Not as a response to empty REPLACE. 16 x 64 or 8 x 128, plus packing 64
transport lanes per row, plus OOM risk, plus no evidence the bottleneck is
representational bandwidth for 351 categories. 64D can hold a character.
It cannot host a 33M generate-MLP and a pointer in a fair fight.

## 6. Trainer organ — what is and is not the villain

The Trainer did the right thing. Isolated candidates, content-addressed
policies, exact resume, gates that refused a false Stage-0 pass. Keep it.

What the Trainer cannot do is make a 99% FFN model learn copy because we
staged the loss. `FOUNDATION_MOTOR_V2_PROGRAM` already exists in
`training/foundation_motor_curriculum.py` (copy_alignment → eos → decision
→ operation → address → joint, copy-gate weight 4.0). Running that program
on **unchanged Candidate-A shape** is the wheel-spin I was asked to push
back on. It may still be the right *objective* after the tissue can mix.

Teacher forcing vs free-running is already in the decoder
(`decode_scheduled`). Stage 0 free-running exact typed rate stayed 0. The
harness saw the truth. Do not weaken it.

## 7. Proposed next shot (one cheap comparison)

Not Stage-0-v2. Not a 512D rail.

**Same D64 rail. Invert Candidate A. Tiny diagnostic.**

Hold: d_model=64, 1 head (head_dim 64), dedicated pointer, same typed-delta
contract, same Stage-0 *copy-only* or copy+insert slice, same gates.

Change:

| | Candidate A (trained) | Proposed diagnostic |
|---|---|---|
| layers | 2 | 4 (or 6 if 4 is free) |
| FFN | 131072 | **256 or 512** |
| heads | 1 | 1 |
| params | ~34M | order 1M, not 34M |
| gate init | generate-biased +1.5 | **0.0** (fair) or copy-biased, declared before launch |

Success is not serving. Success is: generate-gate on copy tokens moves off
the constant, pointer exactness stays nonzero, empty REPLACE is not the
whole policy. If that fails, *then* a 256D brother is justified as capacity,
not as a mood.

Optional same-budget control: one Candidate-A-shaped clone on the identical
copy-only slice, so we do not confuse "fewer classes" with "smaller FFN."

## 8. What this proposal does not do

- Does not amend Source of Truth.
- Does not shelf D64.
- Does not authorize Stage-0-v2, C1 revival, L0-L4 continuation, or a third
  60-step fat-MLP tranche.
- Does not rename `copy_gate` in code until Jeff wants a surgical cleanup
  (the name is misleading; the behavior is generate-gate).
- Does not claim 64D is sufficient forever. It claims we have not yet tested
  a mixer-shaped 64D core.

## 9. Decision requests

D1. Invert-shape diagnostic on D64 before Stage-0-v2 or any wider rail?
D2. FFN 256 or 512 for that diagnostic?
D3. Leave `copy_gate` init at 0.0 (fair) or declare copy-biased init in the
    recipe before launch?
D4. If inverted D64 still cannot move the generate-gate, is the next brother
    256D (4x64) rather than 128/512/1024?

I will not implement or launch until Jeff ratifies.

— Grok / Grok 4.6 (grok-4.6) / 2026-09-06
