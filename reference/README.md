# Reference code: read-only, never imported

Copies of the old trainers and v6 helpers, kept so the new trainer and memory tools can be designed from what worked.
Nothing in `runtime/`, `substrate/`, `tools/` or `tests/` imports these files, and they are not expected to run: they
target the retired v6/v7 substrate and the transformer core.

## `axon7/` (the trainer Jeff remembers as good)

Worth keeping as ideas:

- `train_menu.py`: the operator flow (resume or create; checkpoint picker that shows metadata; guided new-core prompts
  with sensible defaults; local or portable run).
- `field_contract.py`: **one field builder shared by trainer and runtime**, the rule that prevents training/runtime drift.
- `trainer_v2.py`: episode parsing, loss, EMA, atomic and emergency checkpoint saves.
- `runtime_eval.py`: a fixed exam run against a checkpoint.
- Curriculum ladder (see `history/axon7/AXON_CORE_CURRICULUM_AND_CHECKPOINT_AUDIT.md`): alphabet, copy, delta, regions,
  Soul ignition, same-state-different-soul, cooling, ensemble, runtime final exam.

Known gaps to fix in the new trainer (found by inspection 2026-10-05, nothing was executed):

1. Ordinary checkpoint names are built from architecture and step only, so different runs can collide.
2. Resume restores weights/optimizer/step but no curriculum cursor, dataset hash, RNG tied to the data, or parent checkpoint.
3. A resume can silently run against a different or edited dataset.
4. Warm-start can silently create a fresh Soul.
5. `trainer_v2.py` "repairs" unsupported characters by replacing them with spaces. That violates fail-closed; conversion
   must be an explicit tool that writes a new, versioned file and a report.
6. No durable run registry (what each core studied, progress, ancestry), no pause button or paused-job state.
7. Transformer-first; it is not an architecture-neutral engine.

## `ashes_v6/`

Ideas, not code to port: `idle_refinery.py` (compress raw experience into structured memory while idle: the shape of
the new Soul compression), `dormant_store.py` (plain-text, provenance-bearing memory), `pump_soul.py` and
`pump_dialogue.py` (how the soul and dialogue corpora were extracted and filtered), `commandments.py`,
`axon_guard.py` (halt on architecture violations), `training_bucket.py` (validated, exactly-once training queue),
`letter_substrate.py` and `tune_substrate.py` (how the frozen letter geometry was built and tested).