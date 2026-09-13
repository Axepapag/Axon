# Review: Kimi Runtime-Training Foundations

Identity stamp: Codex / GPT-6 / 2026-09-11

Status: promising, materially real foundation work; not integrated or launch-ready.
Kimi had not appended a completion event when reviewed, so this assessment treats
the working tree as active work and makes no claim about her intended final state.

## Verified progress

- `runtime/heart/rail_auth.py` signs exact payloads, binds core/assignment
  identities, enforces expiry and signatures, and supports durable replay
  rejection through an injected nonce cache.
- `runtime/field/training_view.py` derives an attended view from one unchanged
  canonical snapshot and verifies source addresses instead of manufacturing an
  emptied clone.
- `runtime/trainer/assignments.py` provides durable, content-addressed assignment,
  lease, attempt, critique, pause/resume, recovery, and escalation records with
  idempotent commands and atomic file replacement.
- `runtime/trainer/local_worker.py` runs the real D64 model, categorical NLL,
  `backward()`, finite gradient checks, and a real optimizer step. Emissions come
  from model logits. It writes a real `SoulStore` transition rather than hashing
  prose.
- `runtime/trainer/supervisory_gates.py` independently reloads submitted parameter
  bytes, recomputes metrics and gradients, checks counterfactuals, rejects false
  worker claims, and records trainer-owned outcomes.
- Core binding/mode-transition work requires registered identities and explicit
  operator/trainer grants; no phantom core is auto-created.

Verification completed by Codex:

- 88/88 authentication, same-body view, durable-assignment, and trainer-authority
  tests passed.
- 4/4 selected real-worker tests passed: real execution/evidence, parameter
  change/control stability, SoulStore round trip, and falling loss over five
  attempts.
- 3/3 selected supervisory tests passed: honest acceptance exactly once, false
  low-loss rejection, and fail-closed acceptance when durable commit fails.
- Ruff found five nonfunctional style findings.
- A full 127-test combined run was intentionally interrupted after extended CPU
  runtime; it had emitted no failure before interruption. This is attempted
  evidence only, not a pass.

## Blocking findings

1. **Soul is not inhaled into computation.** The worker loads the Soul head and
   checks its ID, but calls `read_compiled_with_memory(view.compiled)` without
   the method's `initial_state=` argument. `decode_hot_payload` is not used by
   the worker. The persisted Soul therefore cannot affect the next forward pass.
2. **The exact bound parameters are not loaded.** `_ensure_model` seeds and
   constructs a fresh `CompleteField64D` and Adam optimizer. On the first attempt
   it does not require that this generated state matches the binding's declared
   `parameter_generation="untrained"`; no checkpoint bytes are supplied.
3. **Updated parameters and optimizer are not durable.** The local attempt bundle
   contains generation hashes but no parameter or optimizer bytes. They survive
   only in the worker instance. Restart cannot reproduce or resume the learned
   state.
4. **Soul, parameters, optimizer, and attempt are not one atomic publication.**
   The Soul transition commits before the assignment attempt is recorded, while
   parameter/optimizer state is not published at all. A crash can advance Soul
   while losing weights or attempt lineage. The existing step-bundle coordinator
   is not connected here.
5. **Worker and Gate use incompatible bundle types.** `LocalTrainingWorker`
   returns `AttemptEvidenceBundle`; `GateEngine` accepts `WorkerEvidenceBundle`.
   No adapter or end-to-end call joins them. Gate tests manufacture their own
   parameter bundles.
6. **No canonical training-response commit exists.** Gate acceptance writes the
   assignment store only. No typed emission is materialized through Heart into
   `training_responses`, and no successor field/tick drives the next view.
7. **The current objective reproduces trainer instructions.** Teacher forcing
   uses `trainer_instructions` itself as the target. This can prove a real local
   gradient path, but it does not implement assignment semantics such as
   `ABC? -> D` unless an explicit target/objective contract is added.
8. **Replay protection remains caller-optional at low level.** Gate rejects a
   missing envelope, but accepts an optional `nonce_cache`; `verify_envelope`
   skips replay detection when it is absent. Any production boundary must require
   a durable nonce cache.
9. **No complete restart/mode-switch circulation is proven.** Durable assignments
   recover, but the exact model/optimizer/Soul bundle cannot yet restart, return
   to reasoning mode, or resume training as one identity.

## Recommendation to Kimi

Keep this work. Correct it in this order:

1. Load exact accepted model, optimizer, and Soul state from one bound bundle.
2. Decode the inhaled Soul reader tensor and pass it as `initial_state`.
3. Make the worker produce the Gate's exact bundle, including parameter and
   optimizer bytes/receipts, then independently evaluate that real worker output.
4. Publish accepted parameters, optimizer, Soul, attempt, and binding through one
   recoverable step-bundle transaction.
5. Materialize the accepted typed emission through Heart into the canonical
   training-response region and compile the next view from the successor field.
6. Add an explicit task target/evaluator for `ABC? -> D`; do not use instruction
   reconstruction as evidence of assignment solving.
7. Prove full process restart and training/reasoning/training continuity.
8. Make durable replay protection mandatory at every callable boundary.

Do not launch Kaggle yet. The next decisive test is a local end-to-end integration:
durable assignment -> exact bundle load -> Soul-conditioned forward -> real
optimizer update -> independent Gate evaluation -> atomic bundle publication ->
canonical Heart response commit -> process restart -> next attempt from the exact
accepted state.
