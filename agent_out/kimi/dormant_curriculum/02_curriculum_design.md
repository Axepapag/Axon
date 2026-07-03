# Dormant-State + Semantic-Edge + Curriculum Pipeline — Design

**Date:** 2026-07-03  
**Active repo:** `D:\Axon`  
**Sources (read-only):** `D:\00\axon_semantic_memory.db`, `D:\00\axon_episodic_memory.db`, `D:\00\axon_memory_backlog.db`, `D:\00\axon_personal_log.json`

---

## 1. Goals

1. Convert recovered SQLite/JSON sources into structured `Container` records and layout metadata.
2. Emit dormant-state artifacts (`containers.jsonl`, `symbol_registry.jsonl` as layout metadata, `semantic_edges.jsonl`, `layout_groups.jsonl`) and a corpus manifest.
3. Build a family of training curriculum datasets from those artifacts.
4. Keep all extraction streaming, read-only, budgeted, and fully provenance-tracked.
5. Never feed raw SQL/JSON rows directly to cores/soul/trainers.

---

## 2. Dormant container record layout

Every recovered item becomes a `Container` from `curator/container_schema.py` with these fields populated:

| Field | Rule |
|-------|------|
| `container_id` | `c-{uuid4.hex[:12]}` generated at import time. A deterministic stable id is **not** required at bootstrap; the provenance pointer (`source`) links back to the source record. |
| `kind` | Broad kind classification: `person`, `place`, `organization`, `tool`, `procedure`, `fact`, `relation`, `concept`, `entity`, `unknown`, `episode`, `backlog_job`, `diary`. |
| `text` | Primary surface text. For entities = `name`; for facts = `"{key}: {value}"`; for relations = `"{subject} {predicate} {object}"`; for procedures = `name`; for episodes = `summary` (truncated); for backlog = `operation`; for diary = `content` (truncated). |
| `normalized_text` | Lowercased / stripped `text`. |
| `letters` | Same as `text` (the materializer later encodes character-by-character through `substrate.py`). |
| `edges` | List of `SemanticEdge` records derived from the source. See §3. |
| `symbols` | Legacy compatibility field. Active no-symbol path keeps this empty. Layout/index IDs live in `metadata["layout_symbols"]`. |
| `source` | `"{db_path}:{table}:{id}"` or `"{json_path}:entry:{index}"`. Uniquely identifies the originating record. |
| `source_tick` | `-1` at bootstrap (no runtime tick yet). |
| `created_tick` | `-1` at bootstrap. |
| `updated_tick` | `-1` at bootstrap. |
| `confidence` | `0.8` for imported records; copied from `extracted_relations.confidence` when available. |
| `provenance` | `"recovered_corpus_builder:{table}_import:v1"`. |
| `status` | `ContainerStatus.DORMANT`. |
| `metadata` | Open dict holding source-specific fields, layout symbols, redaction flags, and raw JSON blobs that are useful for provenance but **not** ensemble-visible. |

### 2.1 Container metadata conventions

```json
{
  "source_table": "extracted_entities",
  "source_id": 42,
  "entity_type": "person",
  "attributes": {...},
  "layout_symbols": ["AA", "AAB"],
  "redaction": "none",
  "origin_family": "semantic_entity"
}
```

- `layout_symbols` is **metadata only**. It is never promoted to `Container.symbols` as semantic meaning.
- `redaction` values: `"none"`, `"diary_only"`, `"episodic_filter"`, `"withheld"`.
- `origin_family` tags the source table/family for curriculum routing.

---

## 3. Semantic edge / layout registry layout

### 3.1 Layout registry

The registry file is produced by `SymbolRegistry` from `curator/semantic_layout_machine.py` for layout/search grouping metadata:

| Field | Meaning |
|-------|---------|
| `symbol` | Substrate-safe layout/group ID (A-Z, a-z, 0-9). |
| `label` | Full human-readable layout/group label. |
| `edge_type` | Empty for active layout records. |
| `target` | Empty for active layout records. |
| `sense` | Empty unless used by future layout grouping. |
| `directionality` | Empty unless used by future layout grouping. |
| `description` | Human-readable note. |
| `created_by` | `"recovered_corpus_builder:v1"`. |
| `created_tick` | `-1` at bootstrap. |
| `examples` | List of source container ids that motivated the symbol. |
| `confidence` | Registry-level confidence. |
| `status` | `"dormant"` for bootstrap layout metadata. |

### 3.2 Edge policy

1. Every semantic edge is born as readable English: edge_type plus target.
2. No semantic-edge symbols are assigned.
3. `Container.symbols` remains empty in the active recovered pipeline.
4. Layout/index IDs stay in metadata only.

### 3.3 Edge derivation by source

| Source table | Edge derivation |
|--------------|-----------------|
| `extracted_entities` | If `attributes` contains edges, parse them; otherwise create `{is_a: type}` if `type` is present. |
| `extracted_facts` | `{key: value}` → `SemanticEdge(edge_type=key, target=value)`. |
| `extracted_relations` | Direct `(subject, predicate, object)` → edge on a container whose text is subject. |
| `extracted_procedures` | No automatic object-level edge; stored as `procedure` container with `steps` metadata. |
| `episodes` | Extracted JSON entities/relations, if present, become edges; otherwise episode is edge-free. |
| `backlog_jobs` | `{operation: payload_summary}` edge when useful; otherwise edge-free. |
| `diary` | Edge-free by default; may later receive self-reflection edges via curation. |

---

## 4. Provenance / source pointer rules

### 4.1 Source pointer format

- SQLite: `"{absolute_db_path}:{table_name}:{primary_key}"`
- JSON: `"{absolute_json_path}:entry:{index}"`

Example: `D:\00\axon_semantic_memory.db:extracted_entities:42`

### 4.2 Manifest provenance

Every artifact manifest records:

```json
{
  "source_files": [
    {"path": "D:\\00\\axon_semantic_memory.db", "sha256": "...", "rows_approx": 497000}
  ],
  "options": {"limit": 1000, "max_items": -1, "no_personal_log": true},
  "output_counts": {"containers": 1000, "edges": 1200, "layout_ids": 45, ...},
  "created_at": "2026-07-03T13:00:00Z"
}
```

### 4.3 Source-pointer required rule

Per SOURCE_OF_TRUTH R2, every container/edge must carry a source pointer. Claims without a source pointer are emitted as `status="candidate"`, never as facts. The bootstrap importer satisfies this by construction.

---

## 5. Personal-log sensitivity rules

1. **Default exclusion.** `axon_personal_log.json` is excluded unless `--include-personal-log` is passed.
2. **Redaction tag.** When included, every diary container gets `metadata.redaction = "diary_only"`.
3. **Region isolation.** Diary containers are routed only to the `diary_region_self_reflection` curriculum family.
4. **No mixing.** Diary content never appears in generic QA, edge prediction, or retrieval QA families.
5. **Length cap.** Diary entries are truncated to `max_diary_chars` (default 2048) at the builder stage; full content stays in the source file.
6. **Content warning.** The manifest records `"personal_log_included": true` and `"redaction_policy": "diary_only"`.

---

## 6. Training dataset families

All families are emitted as JSONL with a `family` field and a per-family manifest.

### 6.1 `field_surfacing_v1`

- **Input:** dormant `containers.jsonl` edge-bearing containers.
- **Output:** `field_surfacing_examples.jsonl` from `dormant_materializer.py`.
- **Format:** `{kind, container_id, source_text, query, edge_type, target, answer, structured_knowledge, provenance}`.
- **Budget:** `max_edges_per_container` defaults to unlimited; materializer already enforces substrate capacity.

### 6.2 `edge_prediction_v1`

- **Goal:** Predict the typed edge given the source text.
- **Input:** containers with edges.
- **Output:** `{family, container_id, source_text, edge_type, target, is_negative, provenance}`.
- **Negative sampling:** For every positive edge, sample one unrelated target from recovered edge targets. Negative examples are tagged `is_negative: true`.

### 6.3 Removed: semantic symbol assignment

The former `symbol_assignment_v1` family is removed. Cores should read English
edge text directly rather than learn opaque edge-code meanings.

### 6.4 `retrieval_qa_v1`

- **Goal:** Answer a question using surfaced structured knowledge.
- **Input:** fact and relation containers.
- **Output:** `{family, query, answer, context_slots_text, provenance}`.
- **Query templates:**
  - Fact: `"What is {key}?"` → answer = `value`.
  - Relation: `"{subject} {predicate} ?"` → answer = `object`.
- **Context:** `render_container_visible(container)` with explicit edge budget.

### 6.5 `procedure_next_step_v1`

- **Goal:** Predict the next step or outcome given a procedure.
- **Input:** `extracted_procedures` containers.
- **Output:**
  - Step prediction: `{family, procedure_name, trigger, steps_before, next_step, provenance}`.
  - Outcome prediction: `{family, procedure_name, trigger, steps, outcome, provenance}`.
- **Budget:** `max_steps` defaults to 10; longer procedures are truncated and counted.

### 6.6 `episodic_exhale_filter_v1`

- **Goal:** Train the soul exhale filter (Tick A/B) per SOURCE_OF_TRUTH Layer 13.
- **Tick A:** Episode summary + extracted facts are placed in `structured_knowledge`; supervise delta + exhale.
- **Tick B:** Mask the episode; require recall of salient facts from soul.
- **Output pairs:**
  - `tick_a.jsonl`: `{family, phase:"a", episode_id, summary, facts_to_retain, structured_knowledge}`.
  - `tick_b.jsonl`: `{family, phase:"b", episode_id, query, expected_answer, mask_regions:[...]}`.
- **Filter:** Episodes with overly long content or explicit personal identifiers are skipped and counted.

### 6.7 `contradiction_alias_curation_v1`

- **Goal:** Produce training material for the propose/dispose validator and contradiction gate (R2 / R3).
- **Contradictions:** Detect pairs of edges with same `(source_text, edge_type)` but different target; emit `{family, type:"contradiction", edge_a, edge_b, resolution_policy:"higher_confidence_or_review"}`.
- **Aliases:** Detect normalized target strings that are lexically similar (Jaccard ≥ 0.85); emit `{family, type:"alias_candidate", target_a, target_b, container_ids_a, container_ids_b}`.
- **No auto-merge:** Both are flagged for review, not silently merged.

---

## 7. Manifest / hash / version strategy

### 7.1 Artifact versioning

- All manifests carry `"version": 1` and `"kind": "axon_recovered_corpus_manifest"` or `"axon_recovered_curriculum_manifest"`.
- Schema changes bump `version` and update `migrations` notes.

### 7.2 Source hashes

- SHA-256 of every source file is computed once at the start of a run and recorded in the manifest.
- Output files also carry SHA-256 so downstream steps can verify integrity.

### 7.3 Counts and budgets

- Manifests record exact output counts and skipped counts (over-length, no edges, personal-log excluded, etc.).
- Budgets are explicit views; every truncation/skip is counted, never silent.

### 7.4 Determinism

- Same source files + same CLI options → same output counts and layout metadata (excluding vector sampling, which is seeded).
- UUID-based `container_id` is the only non-deterministic field; it does not affect training targets.

---

## 8. CLI design

### 8.1 `curator/recovered_corpus_builder.py`

```bash
python curator/recovered_corpus_builder.py \
  --semantic-db D:\00\axon_semantic_memory.db \
  --episodic-db D:\00\axon_episodic_memory.db \
  --backlog-db D:\00\axon_memory_backlog.db \
  --personal-log D:\00\axon_personal_log.json \
  --out-dir D:\Axon\datasets\recovered\dormant_state_v1 \
  --limit 1000 \
  --max-items 5000 \
  --no-personal-log
```

Flags:
- `--semantic-db`, `--episodic-db`, `--backlog-db`, `--personal-log` — source paths.
- `--out-dir` — output directory.
- `--limit` — max rows per source table.
- `--max-items` — max total containers.
- `--no-personal-log` — exclude personal log (default true).
- `--include-personal-log` — override.
- `--max-diary-chars` — truncate diary entries.
- `--seed` — for vector sampling determinism.
- `--dry-run` — count only, no writes.
- `--smoke` — alias for `--limit 100 --max-items 500 --no-vectors`.

### 8.2 `training/build_recovered_curriculum.py`

```bash
python training/build_recovered_curriculum.py \
  --containers D:\Axon\datasets\recovered\dormant_state_v1\containers.jsonl \
  --registry D:\Axon\datasets\recovered\dormant_state_v1\symbol_registry.jsonl \
  --out-dir D:\Axon\datasets\recovered\curriculum_v1 \
  --families all \
  --max-examples 1000 \
  --no-personal-log
```

Flags:
- `--containers`, `--registry`, `--edges`, `--groups` — inputs.
- `--out-dir` — output directory.
- `--families` — comma-separated list or `all`.
- `--max-examples` — per-family example cap.
- `--no-personal-log` — skip diary family.
- `--negative-ratio` — for edge-prediction negatives.

---

## 9. Implementation boundaries

- `curator/recovered_corpus_builder.py` owns SQL/JSON → `Container` + layout metadata + manifest.
- `training/build_recovered_curriculum.py` owns `Container` + edge artifacts → curriculum JSONL + manifest.
- `curator/dormant_materializer.py` is reused for `field_surfacing_v1` (optional).
- `curator/container_schema.py` is the single source of truth for `Container`/`SemanticEdge`.
- `curator/semantic_layout_machine.py` remains the reference symbol-registry implementation; new code imports from it rather than duplicating.

---

## 10. Open decisions

- Exact Jaccard threshold for alias detection (start at 0.85; tune via held-out eval).
- Whether to decode and use `knowledge_vectors` / `episode_vectors` for retrieval QA reranking (deferred; smoke mode skips vectors).
- Exact Tick A/B mask regions (start with `structured_knowledge` active, all other regions masked in Tick B).
