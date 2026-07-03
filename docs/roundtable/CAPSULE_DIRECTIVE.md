# Round Table Directive: The Capsule Pivot

Date: 2026-07-02
Issued by: Jeff (via round table with Claude)
Executor: Kimi (repo restructure), reviewed by Claude

This directive supersedes the 8192D wide-page pivot. Update
`docs/SOURCE_OF_TRUTH.md` in the same change, per project doctrine.

## Why (context for the amendment)

The 8192D wide-page design left one question unresolved: who encodes a page,
and how is a page kept honest? The capsule architecture answers it. It is the
same architecture as patch-based byte modeling (MEGABYTE 2023, Byte Latent
Transformer 2024), built on Axon's own frozen 16D substrate. Characters stay
visible, capsules are auditable, and every capsule must round-trip to exact
text through a gate. This keeps the token-free doctrine intact: no vocabulary
lookup table, no opaque embeddings — packed characters and gated learned
compression only.

## Locked Principles (write these into SOURCE_OF_TRUTH)

1. Layer 0 is the frozen 16D character substrate (`substrate.py`). It is the
   alphabet. The 8192D wide substrate lane is superseded and becomes legacy.
2. Capsules are the unit of the shared field. A capsule is a fixed-width
   vector row packed from or decodable to exact substrate characters.
3. Word capsules are DETERMINISTIC: the 16D character vectors of a word
   (text between whitespace boundaries) concatenated into slots. Lossless by
   construction. Exact round-trip (pack -> unpack -> identical text) is a hard
   substrate gate, same as the old alphabet round-trip gate.
4. Sentence and paragraph capsules are LEARNED, produced by the Capsule Core.
   They must pass an exact-reconstruction gate: capsule -> decoder -> exact
   original character sequence. A capsule that cannot decode to its source
   text fails the gate. No silent lossy pages, ever.
5. The Capsule Core is a shared-state organ, like the projection rails and
   state adapters. It is NOT a member of the reasoning ensemble, has no soul,
   does not tick. Its sole job is construct (text -> capsule) and deconstruct
   (capsule -> exact text), so reasoning cores can focus on reasoning and all
   communication with the outside world stays coherent English.
6. Multiple reasoning cores at different d_model sizes (64, 128, 256, 512,
   1024) attend over the same capsule field through shared per-d_model
   adapters and work together as a single mind. Ensemble, rotating
   consolidator, private temperature-tiered souls: unchanged from current
   SOURCE_OF_TRUTH.
7. Deltas are proposed capsule rows. A committed response deconstructs
   through the Capsule Core to exact English before leaving the system.
8. Semantic edges attach to capsules exactly as the container schema attaches
   them to containers today: additive symbol overlays backed by the permanent
   registry. Letters remain recoverable; symbols never replace them.
9. Three data lanes feed training, each doing its own job:
   - public text corpora streamed through substrate -> capsules (language)
   - the recovered `D:\00` memory DBs (identity + dormant knowledge)
   - future API-ensemble tick traces (behavior; separate directive)

## Working Defaults (revisable at the table, record as such in SOT)

- Word capsule: up to 16 characters x 16D = 256D, zero-padded, with an
  explicit length slot in metadata. Words longer than 16 chars chain across
  capsules linked by container metadata.
- Canonical shared-field width: 1024D. A field row is 1024D. Word capsules
  occupy their 256D deterministic pack within the row (rest reserved /
  metadata); sentence and paragraph capsules are learned 1024D rows.
- Sentence capsule target: up to 256 source characters per capsule
  initially. Paragraphs chain sentence capsules or use a paragraph capsule
  once the sentence gate is proven.
- Capsule Core geometry: small encoder/decoder transformer pair; exact
  geometry is the trainer's choice, but the gate is not negotiable.

## Restructure Tasks (execute in order)

1. AMEND `docs/SOURCE_OF_TRUTH.md`:
   - Replace the "Layer 0: 8192D Substrate" story with the capsule
     architecture above (16D alphabet + capsule layers + Capsule Core organ).
   - Mark `wide_substrate.py`, `wide_field_contract.py`, `state_adapter.py`,
     `mint_state_adapter.py`, `substrate4096.py` as the superseded 8192D lane
     (historical note, kept under `legacy_8192/`).
   - Keep all soul, ensemble, consolidator, delta, training-contract,
     provenance, autonomy, and bus doctrine intact except where the field
     width/capsule change touches it.
   - Update the "Next Architecture Contracts To Implement" list: capsule
     spec, Capsule Core trainer, capsule curriculum, capsule field contract
     move to the top.
2. MOVE (do not delete) the superseded 8192D lane into `legacy_8192/`:
   `wide_substrate.py`, `wide_field_contract.py`, `state_adapter.py`,
   `mint_state_adapter.py`, `substrate4096.py`, plus their tests into
   `tests/` updated to import from `legacy_8192`. All tests must still pass.
3. CREATE `capsule_spec.py` (top level, matching the repo's flat style):
   - deterministic word-capsule pack/unpack against `substrate.py`
   - whitespace segmentation of input text into word capsules
   - chaining rule for >16-char words
   - `python capsule_spec.py` runs a self-test: round-trips a paragraph of
     mixed text word-by-word with zero mismatches, prints PASS/FAIL.
4. CREATE `capsule_core.py`:
   - `CapsuleEncoder` (char-slot rows -> one 1024D capsule) and
     `CapsuleDecoder` (1024D capsule -> char-slot rows -> text via
     nearest-code decode), small transformer pair, PyTorch
   - a `roundtrip_gate(model, texts)` helper returning exact-fill rate
   - identity probe helper in the cf_probe spirit: swapped capsules must
     decode to swapped texts.
5. CREATE `build_capsule_curriculum.py`:
   - streams plain-text corpus files (`--corpus path ...`) and/or sqlite DBs
     (`--db path`, extracting text from messages/summaries/facts tables as
     found in the recovered `D:\00` DBs) into JSONL examples:
     `{text, kind: word|sentence, source, provenance_hash}`
   - sentence segmentation on `.!?` boundaries with a max-256-char cap,
     deterministic, no external NLP dependencies.
6. CREATE `trainer_capsule_core.py`:
   - trains encoder/decoder jointly on the curriculum, loss = exact
     character reconstruction (per-slot nearest-code cross-entropy or MSE to
     frozen char vectors — trainer's choice, must report exact-fill rate)
   - checkpoint discipline: rolling checkpoints with `pointer.json` +
     `checkpoint_done.json`, full config in payload, per SOT Layer 16
   - gates reported every eval: exact-fill rate, swapped-capsule identity
     probe. `--device cpu|cuda`. Must run a 60-second smoke on CPU:
     `python trainer_capsule_core.py --smoke`.
7. CREATE tests: `tests/test_capsule_spec.py` (round-trip, segmentation,
   chaining), `tests/test_capsule_core.py` (shapes, gate helper, smoke
   forward/backward on CPU), `tests/test_capsule_curriculum.py` (builder on
   a tiny fixture corpus).
8. UPDATE `README.md`: new one-paragraph description of the capsule
   direction, new quick checks (`python capsule_spec.py`, capsule tests),
   training example for `trainer_capsule_core.py`.
9. RUN `python -m pytest` and `python capsule_spec.py`; iterate until green.

## Constraints

- Do not delete anything. Move to `legacy_8192/` only as specified.
- Do not touch `ssh.py`, `ssh_helper.py` (uncommitted, credential-bearing),
  `checkpoints/`, `datasets/`, `runs/`, `rails/`, `State/`.
- Do not modify the proven trainers (`trainer_recall.py`,
  `train_field_surfacing.py`, `trainer_semantic.py`, `trainer_soul_v2.py`,
  `core.py`, `field_recall.py`) except minimal import fixes if a move
  requires it.
- No tool lockdowns, approval gates, truncation, or sandbox behavior — SOT
  Layer 17 doctrine applies to this restructure too.
- Keep the flat module style. Python 3.12, torch, no new heavy dependencies.
- Leave the working tree uncommitted; Claude reviews and commits.
