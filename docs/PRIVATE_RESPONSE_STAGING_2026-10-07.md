# Private response staging

Jeff requested that END hand off the collected response to the Heart, which
then writes the Shared Field. This supersedes incremental E0 publication.

The learned control head still has three outputs: 0=WAIT, 1=STAGE, 2=END.
STAGE accepts the chosen exact native character into the private collector.
WAIT changes recurrent states without adding text. END invokes the Heart's
`finish_response(core_id)`: the current consolidator validates and commits the
complete private draft to `response_draft`, then the episode closes. An empty
END writes no filler. Budget exhaustion leaves unfinished text private.

The Heart's generic low-level COMMIT operation still means a canonical write.
The former model control name COMMIT is retained as a numeric alias for STAGE
for weights/curriculum label compatibility; it no longer means per-character
publication in E0. State updates, chosen characters and END decisions remain
learned. Their action meanings and the Heart authority boundary are explicit
runtime rules. Training supervises every character and a separate END through
the same private-collector/publication functions as independent generation.

Response component version is 0.1.2; the unchanged input/reasoning components
remain 0.1.1. The registered reference graph is version 0.1.2. Execution policy
is `axon-e0-private-draft-v3`. Older prepared runs and active walks cannot silently
switch policies. Weights retain the same shapes; old run/checkpoint migration
is separate, and existing artifacts are preserved.

Result rows add `response_publication`, which distinguishes private, published,
unchanged and empty responses, and includes actual commit evidence. Exact text
in `prediction_text` alone is not evidence of publication. Cursor checkpoints
preserve the publication record alongside both states, private draft, weights,
optimizer and RNG. Restoring after a durable publication skips the already
committed text instead of duplicating it; the training graph is rebuilt without
new Heart writes before the remaining optimizer update.

This implements collection and handoff. Arbitrary draft revision, a separate
user-facing delivery transport, and a sentence-quality checker are separate
features. END remains a learned decision that can be premature.

Codex / GPT-6 / 2026-10-07 UTC
