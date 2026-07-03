# Dormant-State + Semantic-Edge + Curriculum Pipeline — Schema Report

**Date:** 2026-07-03  
**Source files (read-only):**
- `D:\00\axon_semantic_memory.db`
- `D:\00\axon_episodic_memory.db`
- `D:\00\axon_memory_backlog.db`
- `D:\00\axon_personal_log.json`

**Scope:** schema, counts, field names, and tiny redacted examples only. No raw private memory content is reproduced.

---

## 1. axon_semantic_memory.db

**File size:** ~2.5 GB  
**Mode:** read-only  
**Tables:** 6 (5 user tables + `sqlite_sequence`)

| Table | Approx. rows (sqlite_sequence.seq) | Purpose |
|-------|------------------------------------|---------|
| `extracted_entities` | 117,685 | Named entities/concepts |
| `extracted_facts` | 140,894 | Subject-predicate-object style facts |
| `extracted_relations` | 103,296 | Typed relations between entities |
| `extracted_procedures` | 36,951 | Named procedures / workflows |
| `knowledge_vectors` | 338,141 | Vector embeddings keyed by `item_type`/`item_id` |

### 1.1 extracted_entities

**Columns:** `id`, `name`, `type`, `attributes`, `source`, `created_at`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | Stable record id |
| `name` | TEXT | Entity surface text / label |
| `type` | TEXT | Entity type (e.g. `person`, `project`, `tool`) |
| `attributes` | TEXT (JSON object) | Key/value properties as JSON string |
| `source` | TEXT | Originating agent/worker (e.g. `memory_agent`) |
| `created_at` | TEXT | ISO-ish timestamp |

Tiny redacted example:
- `name="Jeffrey"`, `type="person"`, `attributes='{"role": "user_requesting_feature"}'`
- `name="Axon Voice Bridge"`, `type="project"`, `attributes='{"purpose": "Real-time voice conversation system", ...}'`

### 1.2 extracted_facts

**Columns:** `id`, `key`, `value`, `source`, `created_at`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | |
| `key` | TEXT | Fact key / short subject |
| `value` | TEXT | Fact statement |
| `source` | TEXT | |
| `created_at` | TEXT | |

Tiny redacted example:
- `key="current_phone_capability"`, `value="Uses Twilio with webhook connections, can speak greetings but lacks bidirectional streaming"`
- `key="desired_feature"`, `value="Real-time TTS with back-and-forth conversation capability on phone calls"`

### 1.3 extracted_relations

**Columns:** `id`, `subject`, `predicate`, `object`, `confidence`, `source`, `created_at`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | |
| `subject` | TEXT | Source entity |
| `predicate` | TEXT | Relation type (often snake_case) |
| `object` | TEXT | Target entity |
| `confidence` | REAL | 0.0–1.0 |
| `source` | TEXT | |
| `created_at` | TEXT | |

Tiny redacted example:
- `subject="phone.make_call"`, `predicate="currently_uses"`, `object="Twilio"`, `confidence=1.0`

### 1.4 extracted_procedures

**Columns:** `id`, `name`, `trigger`, `steps_json`, `outcome`, `applicability`, `source`, `created_at`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | |
| `name` | TEXT | Procedure title |
| `trigger` | TEXT | When to invoke |
| `steps_json` | TEXT (JSON array) | Ordered step strings |
| `outcome` | TEXT | Expected result |
| `applicability` | TEXT | Scope / context |
| `source` | TEXT | |
| `created_at` | TEXT | |

Tiny redacted example:
- `name="Legacy Phone Call"`, `trigger="phone.make_call invoked without streaming"`, `steps_json='["Establish Twilio connection", "Connect to webhook endpoint", ...]'`

### 1.5 knowledge_vectors

**Columns:** `id`, `item_type`, `item_id`, `vector`, `created_at`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | |
| `item_type` | TEXT | `entity`, `fact`, `relation`, `procedure`, etc. |
| `item_id` | INTEGER | Foreign key to the corresponding table |
| `vector` | BLOB | float32 vector bytes |
| `created_at` | TEXT | |

The BLOB is decoded as `np.frombuffer(blob, dtype=np.float32)` (fallback to JSON / struct.unpack). These vectors are **not** used as ensemble-visible substrate content; they are optional grouping/retrieval signals and are skipped by default in smoke mode.

---

## 2. axon_episodic_memory.db

**File size:** ~491 MB  
**Mode:** read-only  
**Tables:** 3 (2 user tables + `sqlite_sequence`)

| Table | Approx. rows (sqlite_sequence.seq) | Purpose |
|-------|------------------------------------|---------|
| `episodes` | 20,407 | Conversation / interaction episodes |
| `episode_vectors` | 17,429 | Per-episode float32 vectors |

### 2.1 episodes

**Columns:** `id`, `episode_type`, `turns_json`, `summary`, `extracted_json`, `source`, `created_at`, `payload_json`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | |
| `episode_type` | TEXT | e.g. `conversation_summary` |
| `turns_json` | TEXT (JSON) | Raw turn list, often `null` |
| `summary` | TEXT | Narrative summary (multi-line, may be long) |
| `extracted_json` | TEXT (JSON) | Extracted entities/facts/relations, often `null` |
| `source` | TEXT | |
| `created_at` | TEXT | ISO timestamp |
| `payload_json` | TEXT (JSON) | Full payload incl. turns and assistant response |

Tiny redacted example:
- `episode_type="conversation_summary"`, `summary="**ROLLING SUMMARY**\n\n**Active Initiative: Real-Time Voice Conversation System**..."`, `payload_json='{"turns": [{"user": "Want you to use your software engineer Context template...", "assistant": "..."}]}'`

Episodic content is **high-sensitivity**: it contains verbatim user/assistant turns and project context. It must be surfaced only through explicit filters (Tick A/B exhale filter) and only when `--no-personal-log` is not set.

### 2.2 episode_vectors

**Columns:** `id`, `episode_id`, `vector`, `created_at`

Per-episode float32 BLOB, keyed to `episodes.id`. Optional; skipped by default in smoke mode.

---

## 3. axon_memory_backlog.db

**File size:** ~184 MB  
**Mode:** read-only  
**Tables:** 2 (1 user table + `sqlite_sequence`)

| Table | Approx. rows (sqlite_sequence.seq) | Purpose |
|-------|------------------------------------|---------|
| `backlog_jobs` | 8,567 | Deferred memory operations / tasks |

### 3.1 backlog_jobs

**Columns:** `id`, `operation`, `payload_json`, `budget`, `status`, `attempts`, `last_error`, `created_at`, `updated_at`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PRIMARY KEY | |
| `operation` | TEXT | e.g. `summarize` |
| `payload_json` | TEXT (JSON) | Operation-specific payload |
| `budget` | INTEGER | Token/cost budget |
| `status` | TEXT | e.g. `done` |
| `attempts` | INTEGER | |
| `last_error` | TEXT | |
| `created_at` | TEXT | |
| `updated_at` | TEXT | |

Tiny redacted example:
- `operation="summarize"`, `payload_json='{"turns": [{"user": "u1", "assistant": "a1"}]}'`, `budget=5000`, `status="done"`

Backlog rows are operational metadata. They are useful for procedure/next-step drills and provenance, but are low-density training signal compared to semantic tables.

---

## 4. axon_personal_log.json

**File size:** ~35 KB  
**Mode:** read-only  
**Top-level type:** object

| Field | Type | Notes |
|-------|------|-------|
| `version` | string | e.g. `"1.0.0"` |
| `created_at` | string | ISO timestamp |
| `about` | string | Description of the journal |
| `entries` | list | 33 entries |
| `last_updated` | string | ISO timestamp |
| `entry_count` | integer | 33 (matches `len(entries)`) |

### 4.1 entries[i]

| Field | Type | Notes |
|-------|------|-------|
| `timestamp` | string | ISO timestamp |
| `type` | string | Entry type tag |
| `content` | string | Journal content; can be very long (one entry ~16 KB) |

**Sensitivity:** `axon_personal_log.json` is Axon's own diary. It is the most sensitive source and is **excluded by default** unless a dedicated `--include-personal-log` flag is passed. When included, each entry is redaction-tagged and surfaced only through the `diary` region curriculum family, never mixed into generic QA.

---

## 5. Cross-source summary

| Source | Records (approx.) | Primary curriculum value | Default inclusion |
|--------|-------------------|--------------------------|-------------------|
| `axon_semantic_memory.db` | ~497 K semantic records | Field surfacing, edge prediction, retrieval QA | Yes |
| `axon_episodic_memory.db` | ~20 K episodes | Episodic exhale filter Tick A/B, procedure/next-step, contradiction/alias curation | Yes, filtered |
| `axon_memory_backlog.db` | ~8.6 K jobs | Procedure/next-step, operational provenance | Yes |
| `axon_personal_log.json` | 33 entries | Diary-region training, self-reflection | **No** |

**Total high-value semantic records:** ~497 K from the semantic DB, plus ~28 K episodic/backlog records. This is the bootstrap dormant-state foundation.

---

## 6. Compatibility notes with existing curator code

- `curator/semantic_layout_machine.py` was authored against a slightly different recovered schema (e.g. `entity_type` instead of `type`, `definition` column on entities). The new `recovered_corpus_builder.py` must map the **actual** columns above into `Container` records using a flexible column-finder, not hard-coded names.
- `curator/dormant_materializer.py` already consumes `containers.jsonl` and produces `field_surfacing` examples. The new pipeline reuses that path for one curriculum family and adds additional families in `training/build_recovered_curriculum.py`.
- `curator/container_schema.py` is the canonical record envelope; all SQL rows become `Container` records before any training use.
- `slots/slot_spec.py` and `slots/slot_field_contract.py` enforce substrate-safe text and surfacing budgets; layout IDs stay metadata and are not semantic edge meaning.

---

## 7. Redaction and privacy posture

- No raw row content is reproduced in this report beyond tiny, truncated, schema-illustrative snippets.
- Personal-log entries are excluded by default.
- Episodic turns/summaries are treated as potentially-sensitive and are filtered through Tick A/B exhale gates before becoming curriculum.
- All artifact manifests record source hashes, counts, and redaction flags so downstream consumers can audit provenance.
