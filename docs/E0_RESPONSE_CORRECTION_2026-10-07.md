# E0 response learning correction

Jeff authorized Codex to correct the answer-copy shortcut on 2026-10-07 UTC.
This increment implements `axon-e0-autoregressive-v2` in `host_e0.py`.

Both training and inference observe all exercise steps and the query, then
enter the response phase. The first response input is the existing EMPTY
cell, used as a generation cue; it is never a fabricated WAIT answer. The
next training input is the previous target character, with the next character
as its prediction target. Inference uses its previous emitted character. It
never reads expected answer text or length to determine the response walk.

Each target character has a character loss and COMMIT control loss. An extra
tick after the last character has END control loss, so END cannot swallow the
last character. Control-only WAIT/END exercises also receive a control loss.
Teacher forcing supplies actions only during training; its text and scores are
explicitly labelled `teacher_forced_training`. Evaluation uses model choices.

The loop performs one optimizer update per complete exercise, from the mean
response loss with a full gradient path through the observations and response
states. It no longer detaches away the memory-learning path at every tick.
`optimizer_steps` counts actual updates; `loss_samples` retains the last 256
response losses; `objective_loss` is the actual mean used for the update.
The Lab's main loss reports this objective, rather than only the final END loss.
This baseline keeps an episode graph in memory; very long exercises may need
an explicit future memory-bounded training strategy. No truncated credit is
silently substituted.

Mid-exercise training recovery reconstructs the deterministic prefix graph at
unchanged checkpoint weights, including prior losses, without any Heart write
or optimizer step during replay. Only the completed exercise updates weights.
Checkpoint values still carry both states, optimizer, all RNG artifacts, Heart
bookkeeping and cursor. The additive `execution` block pins protocol, mode,
episode digest and response budget. Incompatible older active walks fail closed;
they require an explicit fresh episode or migration. No autograd graph is saved.
Replay relies on E0's deterministic GRU, which has no stochastic layers.

The optional constructor keyword `response_tick_budget` defaults to 256 and
must be a positive integer. Inference stops on its own END; otherwise it reports
`budget_exhausted`, preserving any partial draft without calling it completed.
WAIT produces no draft mutation, control submission or canonical commit, while
recurrent states can continue changing. `generation.response_ticks` reports
the actual count. A nonempty answer's `prediction_control` is COMMIT; its closing
END is reported separately by `generation.termination`. No target label is used
to choose this result control. Empty results retain the model's last decision.

## What the staging area actually does

`HeartHost.set_draft` holds exact native text privately, separately from the
latent response tensor. E0 appends a chosen character there, then asks the Heart
consolidator to commit that draft into canonical `RESPONSE_DRAFT` on every COMMIT.
END closes the episode and performs no final canonical output write.
The response can therefore grow over many ticks beyond one tensor's lane count.
It does not yet edit/reorder words or wait to publish the whole sentence in one
final write. This correction preserves that agreed path; a different staging
policy needs its own explicit implementation and tests.

## Acceptance scope

Regression tests check shifted inputs, separate END, no answer-dependent
generation inputs/length, memory gradients, WAIT zero-write behavior, explicit
budget exhaustion, frozen substrate, and bit-identical CPU weights/optimizer/
losses on training resume. Backend tests exercise real episode-boundary recovery.
These checks establish correct plumbing; held-out delayed/distracted recall,
input-aware baselines, memory counterfactuals, CUDA recovery and verified current
backup remain separate acceptance gates. Public Start stays locked.

Codex / GPT-6 / 2026-10-07 UTC
