# Complete-Field 64D R0 Training Contract

Status: preserved V6 mechanism contract; direct-record training is blocked as an active Axon path by Jeff's 2026-08-20 canonical-anatomy ruling. The complete-sweep reader remains the D64 mechanism baseline while training is moved behind the canonical State/compiler interface.

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
receive a region marker. Every encoded page token is retained as addressable
read-only memory together with its immutable source character and region identity.
Each autoregressive decoder step cross-attends that memory for semantic context.
V6 then uses a separate single-head position pointer over exact source tokens,
plus a learned copy/generate gate. Supervised copy spans bind target characters
to one exact `(region, character_position)` occurrence before probability is
scattered onto the source-character alphabet. Empty-region markers remain
context but cannot be copied. The four state tokens carry a compact recurrent
summary; they are not required to memorize exact names, spans, or tool output.
The decoder is locked until a coverage manifest proves all active characters
and all ten region identities were visited without gaps or duplicates.

Page size bounds one physical operation, not logical context. Runtime and decoder
attention cost grow with field length. Exact field text remains external,
immutable where sealed, and directly addressable through its encoded page tokens.

## Writer and ticks

Scratch and response use separate autoregressive heads with explicit end tokens
and a configurable 512-character R0 training bound. Targets longer than the
declared bound fail loudly; they are never silently clipped. The writer emits
typed whole-region replacement operations only for `scratch` and
`response_draft`.

Training supports deterministic scheduled prefix exposure. With a ratio below
one, each scratch, response, and matched-counterfactual decoder sometimes
receives its own previous argmax token instead of the gold previous character.
This prevents high teacher-path accuracy from hiding a free-running decoder
that ignores reader/scratch state. The chosen ratio is recorded in every metric
and checkpoint.

## Evidence classes

- Grade A exact raw user messages plus exact, quality-filtered assistant
  substrings teach conversation. Source IDs, timestamps, source-pair hashes,
  excerpt offsets/hashes, and newline-normalization audits stay with each
  local-only example. Hidden reasoning and tool wrappers are excluded.
- The exact March 6 Axon naming event is a Grade A autobiographical anchor.
- Grade D grounded procedure episodes teach scratch behavior only and are
  explicitly prohibited from becoming sole autobiographical evidence.
- Grade S deterministic examples teach page mechanics, retrieval, arithmetic,
  conversation foundations, and abstention. V6 counterfactuals test evidence
  authority rather than scratch obedience: when immutable `user_input` or
  `tool_results` resolves the answer, empty or conflicting scratch preserves
  that verified answer. A separate anti-shortcut shard uses unseen mixed-case
  alphanumeric tokens, same-region and wrong-region duplicates, nearby decoys,
  first/middle/last placement, and exact page-boundary crossings.

Preserved private curriculum output belongs under `State/training/curriculum` and is not committed. Direct D00 rebuilding is legacy-only and requires an explicit CLI opt-in; canonical D64 training will consume copy-on-write State branches through the shared field/compiler interface instead of treating detached JSON as runtime truth.

## Observability and recovery

Every step records component losses, scratch/response character accuracy,
matched-counterfactual loss/accuracy and intervention identity, coverage
characters/pages, gradient norm, rate, family, and example ID.
`samples.jsonl` and `live.json` show the user input, gold and predicted scratch,
gold and predicted response, termination, coverage, and typed delta.

Checkpoints are atomic, keep a rolling three active checkpoints, archive older
checkpoints without deleting them, and update `pointer.json` plus
`checkpoint_done.json`. Interruption writes a recovery checkpoint. Checkpoint
schema v6 binds every resume to the exact SHA-256 fingerprints of both training
and evaluation datasets and records Python, NumPy, Torch, CUDA, scaler, optimizer,
and sampler state. V5 and earlier checkpoints are intentionally incompatible
with the V6 exact-position pointer.

## Promotion sequence

1. contract, compilation, focused CPU tests, and the full repository suite;
2. build a versioned V6 private curriculum plus an isolated anti-shortcut
   alignment shard; never resume V5 against the new fingerprints;
3. run a tiny bounded CUDA alignment smoke and require complete ten-region
   coverage, finite optimization, actual free-running samples, and exact
   interrupted/resumed CUDA trajectory;
4. before any 1,000-step comparison, require **100%** held-out V6 alignment
   behavior: exact scratch, exact response, exact labeled source position,
   correct copy/generate gate, termination, and exact answer recovery under
   empty/conflicting scratch where immutable evidence resolves the answer;
5. only after those gates pass may a separately authorized bounded 1,000-step
   V6-vs-V5 comparison run. Arithmetic remains diagnostic in R0 rather than a
   hard identity/conversation promotion gate.

The root launcher contains no 5,000-step or 200,000-step continuation while V6
is below the exact held-out binding gate. Step count alone is never a success
metric.
