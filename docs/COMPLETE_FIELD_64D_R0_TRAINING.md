# Complete-Field 64D R0 Training Contract

Status: active implementation contract, authorized by Jeff on 2026-08-18.

## R0 outcome

Train one shallow 64D core to make two causally linked field updates:

1. sweep every active character in all ten canonical regions and replace `scratch`;
2. teacher-commit or runtime-commit scratch, sweep the complete field again, and replace `response_draft`.

`diary` is attended like every other active region but is sealed in this
curriculum. Diary writing is deferred to a later, separately evaluated stage.
Conversation history and tool results remain immutable evidence.

## Physical reader

The frozen 16D substrate is lifted losslessly into 64D. A configurable physical
page defaults to 256 characters. Four temporary reader-state tokens pass from
page to page through one shallow Transformer encoder layer. Empty regions still
receive a region marker. The decoder is locked until a coverage manifest proves
all active characters and all ten region identities were visited without gaps
or duplicates.

Page size bounds one physical operation, not logical context. Runtime cost grows
with field length. Exact text stays external and addressable.

## Writer and ticks

Scratch and response use separate autoregressive heads with explicit end tokens
and a configurable 512-character R0 training bound. Targets longer than the
declared bound fail loudly; they are never silently clipped. The writer emits
typed whole-region replacement operations only for `scratch` and
`response_draft`.

## Evidence classes

- Grade A exact raw user messages plus exact, quality-filtered assistant
  substrings teach conversation. Source IDs, timestamps, source-pair hashes,
  excerpt offsets/hashes, and newline-normalization audits stay with each
  local-only example. Hidden reasoning and tool wrappers are excluded.
- The exact March 6 Axon naming event is a Grade A autobiographical anchor.
- Grade D grounded procedure episodes teach scratch behavior only and are
  explicitly prohibited from becoming sole autobiographical evidence.
- Grade S deterministic examples teach page mechanics, retrieval, arithmetic,
  conversation foundations, and abstention. Every synthetic family contains
  matched empty-scratch and conflicting-scratch interventions with explicitly
  different verified response targets.

Private output remains under `State/private_curriculum` and is not committed.
The D00 SQLite source opens with URI `mode=ro` and `PRAGMA query_only=ON`.

## Observability and recovery

Every step records component losses, scratch/response character accuracy,
matched-counterfactual loss/accuracy and intervention identity, coverage
characters/pages, gradient norm, rate, family, and example ID.
`samples.jsonl` and `live.json` show the user input, gold and predicted scratch,
gold and predicted response, termination, coverage, and typed delta.

Checkpoints are atomic, keep a rolling three, and update `pointer.json` plus
`checkpoint_done.json`. Interruption writes a recovery checkpoint.

## Promotion sequence

1. contract and CPU tests;
2. bounded GPU smoke with baseline and actual samples;
3. a 5,000-step behavioral promotion stage with frozen eval, matched scratch
   interventions, forced correct/counterfactual decodes, and actual free-running
   scratch/response samples;
4. only if loss, output, coverage, termination, diversity, matched causal-use,
   and free-running exact-match gates for copy, conversation, cross-page
   retrieval, abstention, and arithmetic all pass, automatically resume the
   same checkpoint lineage toward 200,000 steps. The root training launcher
   enforces this boundary and independently rereads the gate artifact before
   continuation.

Step count alone is never a success metric.
