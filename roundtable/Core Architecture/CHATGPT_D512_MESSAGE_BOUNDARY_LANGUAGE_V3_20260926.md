# GRU512 Dormant language v3 — real message boundaries, not arbitrary slice EOS

Identity: ChatGPT / GPT-5.6 Sol / 2026-09-26 America/Chicago

## Purpose

This repair follows Codex's independent audit of the earlier Dormant-language EOS experiments. The prior launcher used a fixed 128-character source + 64-character target slice and appended private EOS at the end of every 64-character target. That made a resource/training slice masquerade as a completed utterance. The new v3 curriculum separates those concepts.

## V3 semantics

- Curriculum split occurs by complete Dormant message record, chronologically. Train and heldout do not share message record IDs.
- Each eligible message remains byte/text complete; over-budget messages are skipped whole rather than truncated.
- Exact Unicode transport is allowed. Multi-token UTF-8 transport units are retained rather than normalized away.
- Each authored message generates completion examples from several deterministic prefix cuts (1/4, 1/2, 3/4 when distinct).
- The source is `User: ` or `Axon: ` plus an exact authored prefix. The target is the exact remainder of that same authored message.
- Private EOS means one thing only: **the exact authored message has ended**.
- `bptt_tokens` is a compute boundary only. Recurrent state is carried numerically across BPTT chunks and the graph is detached. No EOS/control target is created at a BPTT boundary.
- Scheduled/self-generated feedback is intentionally absent from this repair rung. First establish completion semantics and content behavior.

Implementation:
- `training/continuous_core_d512_dormant_language_v3.py`
- `scripts/train_continuous_core_d512_dormant_language_v3.py`
- `tests/test_continuous_core_d512_dormant_language_v3.py`

## Governance/preflight repair

The prior launcher asserted `COUNTERFACTUAL_DEPENDENCE passed=True` without executing a counterfactual. V3 measures it before Trainer mutation. On the protected `dec45989...` parent, eight heldout message-completion episodes were checked by comparing exact-source recurrent state/first-decision logits against a no-source zero-state counterfactual. All 8/8 produced nonzero state and logit deltas and passed. Boundary preflight also verifies message-level train/heldout separation and declares zero EOS targets at compute-slice boundaries.

## Verification before training

- New v3 tests plus original D512/breath tests: **12/12 passed** using an explicit writable pytest basetemp.
- `py_compile`: passed.
- `git diff --check`: passed.
- An earlier pytest invocation collected and passed all 12 tests but exited during pytest temp-directory cleanup with a Windows permission error; rerun with an explicit basetemp exited 0.

## Controlled 24-step comparison

Both smokes use the same protected parent, curriculum, seed, chronological data order, learning rate, batch size, BPTT size, evaluation set, and breath rehearsal. The only intended objective difference is real-boundary EOS weight 1 versus 4.

Shared setup:
- parent checkpoint: `dec45989bd4b872a90fb7b1f4d9ce9a0ce448b5ded70d04bacff56bf3bca754f`
- curriculum: `e0d9269150de8b1e425df6c17e0d61d727d8e67d291e10087da9aac8abea45a3`
- 1,200 train messages / 200 later heldout messages
- 3,600 train completion episodes / 600 heldout completion episodes
- evaluation: first 64 heldout episodes, 9,429 target transport tokens
- batch 4, 24 steps, LR 1e-4, BPTT 64, seed 20261012
- breath rehearsal every optimizer step, weight 1.0
- no scheduled sampling

Protected-parent baseline under the new harder variable-message evaluation:
- heldout loss 2.246168
- teacher content 44.0980%
- teacher real-boundary EOS 0%
- free character accuracy 6.0134%
- early/on-time/late/absent EOS = 0 / 0 / 0 / 64
- breath episode/silent = 100% / 100%

### EOS weight 1

Checkpoint: `362016143d28d915a68d8b89fbb69eb773e0d3d1ff476b66c37c48f4bc721f16`
Run: `State/training/continuous_core_d512_dormant_language_v3/runs/615b7e2515b2f3c84cb457f4acc7ddd261e7267aca405357005b124067619664.json`

Final:
- heldout loss 2.209379
- teacher content 44.5752%
- teacher real-boundary EOS 0%
- free character accuracy 4.6665%
- reference-content exact 0%
- exact completion 0%
- early/on-time/late/absent EOS = 0 / 0 / 0 / 64
- breath episode/silent = 100% / 100%

Interpretation: the compute/boundary wiring is clean and teacher content moved slightly, but ordinary unweighted message-end EOS did not become top-1 in 24 steps. Free-running content declined on this small tranche.

### EOS weight 4

Checkpoint: `1820fb12fd9a5ad7c74f2827eb64584f022792ceeb47a57e7af494192e497d1d`
Run: `State/training/continuous_core_d512_dormant_language_v3/runs/ac7234120536ede70adddd3e381311db2071d637cc20ea36a3a00a8f89ae808c.json`

Final:
- heldout loss 2.214598
- teacher content 44.0238%
- teacher real-boundary EOS **100%**
- free character accuracy 4.2953%
- reference-content exact 0%
- exact completion 0%
- early/on-time/late/absent EOS = **4 / 0 / 0 / 60**
- median emitted transport length 189
- breath episode/silent = 100% / 100%

Interpretation: with the target semantics repaired, EOS weight 4 rapidly teaches the model to recognize a real authored boundary **under teacher-forced correct preceding content**. It does not yet produce correct free-running completion: four free runs stop early and none stop exactly at the authored boundary. This is materially different from the old arbitrary-slice result because the evaluator now distinguishes early/on-time/late/absent EOS.

## Decision

Do **not** launch a long v3 tranche yet. The repair has established correct target meaning, executable preflight, and honest stop accounting, but neither controlled smoke improved free-running continuation and neither produced an on-time free-running completion. Preserve `dec45989...` as the language milestone and both v3 smoke checkpoints as diagnostic candidates only.

The next experiment should improve content/free-running behavior on the correctly defined message-completion task before adding self-feedback. If self-feedback is later reintroduced, it should be a separately versioned ablation after a message-boundary control has demonstrated joint content and completion progress.
