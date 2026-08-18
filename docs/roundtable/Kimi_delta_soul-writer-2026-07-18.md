# Kimi delta: differentiable 168-row soul writer

Review mode: read-only focused design review with Kimi Code `0.22.1`.

## Recommendation

Use one additive `DifferentiableSoulWriter` outside the core:

- authoritative shape: 168 rows at each core's native `d_model`,
- tier boundaries: hot `0:128`, warm `128:160`, cold `160:168`,
- learned content-addressed keys over only the 128 hot rows,
- a shared value projection from masked tick-A field output,
- a small residual write gate,
- warm/cold rows copied exactly,
- existing core soul cross-attention retained as the read path,
- conflicting core writer and no-grad manager writer frozen/disabled.

Tensor sequence:

1. Tick A reads `soul0` and the lesson-bearing field.
2. The external writer maps tick-A field output to a differentiable hot-row
   update and produces `soul1`.
3. Tick B receives the same shape/mask but a field from which the fact is
   absent.
4. Tick-B proposal cross-entropy backpropagates through soul read attention,
   `soul1`, and the tick-A writer.
5. Persisted soul mutation, if any, happens only after optimization using a
   detached, contract-validated `soul1`.

## Required masks and ablations

- identical `(B,168)` active/read masks across conditions,
- write mask restricted to active hot rows,
- field pooling honors the exact active-field mask,
- pre-write, zero, swapped/donor, no-writer, uniform-routing, and fact-leak
  ablations,
- warm/cold byte identity,
- standard copy/partial/blank regression,
- 100-percent named optimizer/gradient coverage for declared trainables.

## Promotion blockers

- any row-boundary change,
- either legacy writer remaining active,
- writer touching warm/cold rows,
- tick-A supervision masking a weak tick-B causal signal,
- the expected fact appearing anywhere in tick-B's visible field,
- missing correct/zero/swapped probes,
- evaluation mutating persisted state,
- missing writer/optimizer/RNG/sampler/freeze-manifest checkpoint state,
- mixing 64D and 128D writer artifacts.

Kimi authorized no launch. The recommendation was implemented only as an
additive local pilot module and CPU proof; behavioral thresholds still require
real fixed-suite training before promotion.
