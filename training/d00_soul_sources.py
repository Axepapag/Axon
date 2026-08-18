"""Fail-closed, read-only adapter for the structured memory stores in ``D:\00``.

The semantic database is the only source of training targets.  A semantic
fact, relation, or procedure is eligible only when the same canonical
assertion occurs in an episode's ``extracted_json``.  The old memory databases
are audited and quarantined in their entirety; the personal log is recorded as
manual-only and never emitted.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from substrate import assert_supported_text
from runtime.field import (
    LogicalRegion,
    SharedFieldSnapshot,
    SlotKind,
    compile_field_view,
)
from training.build_multitick_curriculum import (
    SourceMutationError,
    SourceRecord,
    SourceSpec,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
    sqlite_ro_uri,
)


D00_PRIVATE_ADAPTER = "d00_private_grounded_semantic_v2"
D00_SEMANTIC_DATABASE = "axon_semantic_memory.db"
D00_EPISODIC_DATABASE = "axon_episodic_memory.db"
D00_RESIDUAL_DATABASES = (
    "axon_memory.db",
    "axon_memory_backlog.db",
)
D00_PERSONAL_LOG = "axon_personal_log.json"
D00_MEMORY_DATABASES = (
    D00_SEMANTIC_DATABASE,
    D00_EPISODIC_DATABASE,
    *D00_RESIDUAL_DATABASES,
)
DEFAULT_PRODUCTION_SAMPLE_CAP = 8192
# Backward-compatible name retained for callers of the first adapter.
DEFAULT_LIMIT_PER_KIND = DEFAULT_PRODUCTION_SAMPLE_CAP
FAMILY_ORDER = ("fact", "relation", "procedure")
FAMILY_WEIGHTS = {
    "fact": 0.45,
    "relation": 0.45,
    "procedure": 0.10,
}
TEMPLATE_CAP = 32
MIN_TARGET_CHARS = 8
SOUL_WRITE_INSTRUCTION = "Store visible fact in soul."
_SAMPLE_CAP_PATTERN = re.compile(r"d00_sample_cap=(\d+)")
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:password|passwd|secret|token|api[\s_-]*key|credential|"
    r"private[\s_-]*key|phone|e[\s_-]*mail|address|ssn|"
    r"social[\s_-]*security|auth|cookie|session)",
    re.IGNORECASE,
)


class D00GroundingError(RuntimeError):
    """Raised when the private adapter cannot satisfy its strict contract."""


@dataclass(frozen=True)
class D00GroundedSelection:
    """Selected source records plus source and quarantine audits."""

    records: tuple[SourceRecord, ...]
    source_manifest: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def render_soul_memory_text(fact: str, evidence: str) -> str:
    """Render the exact tick-A payload shared by adapter and builder."""

    return f"Fact: {fact} Evidence: {evidence}"


def _compiled_visible_text(
    region: LogicalRegion,
    text: str,
    *,
    snapshot_texts: dict[LogicalRegion, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    snapshot = SharedFieldSnapshot.from_texts(
        snapshot_texts or {region: text},
        source="d00_capacity_audit",
        provenance="runtime_compiled_field_capacity",
    )
    view = compile_field_view(
        snapshot,
        proposal_region=LogicalRegion.RESPONSE_DRAFT,
    )
    visible = "".join(
        ref.rendered_char or ""
        for ref, active in zip(
            view.slot_refs,
            view.attention_mask.tolist(),
            strict=True,
        )
        if (
            active
            and ref.kind is SlotKind.SPAN
            and ref.logical_region is region
        )
    )
    omissions = [
        {
            "reason": omission.reason,
            "region_char_start": omission.region_char_start,
            "region_char_end": omission.region_char_end,
        }
        for omission in view.omissions
        if omission.logical_region is region
    ]
    return visible, omissions


@lru_cache(maxsize=1)
def _compiled_payload_capacity_items() -> tuple[tuple[str, int], ...]:
    """Derive payload limits from runtime slot refs, including region tags."""

    probe = "x" * 512
    capacities: list[tuple[str, int]] = []
    for label, region in (
        ("recall_query_user_input", LogicalRegion.USER_INPUT),
        ("target_response_draft", LogicalRegion.RESPONSE_DRAFT),
        (
            "tick_a_structured_knowledge",
            LogicalRegion.STRUCTURED_KNOWLEDGE,
        ),
    ):
        visible, _ = _compiled_visible_text(region, probe)
        if not visible or visible != probe[: len(visible)]:
            raise D00GroundingError(
                f"cannot derive compiled payload capacity for {region.value}"
            )
        capacities.append((label, len(visible)))
    return tuple(capacities)


def runtime_field_payload_capacities() -> dict[str, int]:
    """Return exact tag-aware capacities derived from the runtime compiler."""

    return dict(_compiled_payload_capacity_items())


def _compiled_region_fit_audit(
    *,
    label: str,
    region: LogicalRegion,
    text: str,
    snapshot_texts: dict[LogicalRegion, str] | None = None,
) -> dict[str, Any]:
    visible, omissions = _compiled_visible_text(
        region,
        text,
        snapshot_texts=snapshot_texts,
    )
    return {
        "label": label,
        "logical_region": region.value,
        "required_chars": len(text),
        "visible_chars": len(visible),
        "required_sha256": sha256_text(text),
        "visible_sha256": sha256_text(visible),
        "omissions": omissions,
        "passed": visible == text and not omissions,
    }


def audit_compiled_field_capacity(
    *,
    target: str,
    evidence: str,
    recall_query: str,
) -> dict[str, Any]:
    """Compile the exact three D00 payloads and prove full visibility."""

    memory_text = render_soul_memory_text(target, evidence)
    tick_a_texts = {
        LogicalRegion.USER_INPUT: SOUL_WRITE_INSTRUCTION,
        LogicalRegion.STRUCTURED_KNOWLEDGE: memory_text,
    }
    checks = {
        "target_response_draft": _compiled_region_fit_audit(
            label="target_response_draft",
            region=LogicalRegion.RESPONSE_DRAFT,
            text=target,
        ),
        "recall_query_user_input": _compiled_region_fit_audit(
            label="recall_query_user_input",
            region=LogicalRegion.USER_INPUT,
            text=recall_query,
        ),
        "tick_a_structured_knowledge": _compiled_region_fit_audit(
            label="tick_a_structured_knowledge",
            region=LogicalRegion.STRUCTURED_KNOWLEDGE,
            text=memory_text,
            snapshot_texts=tick_a_texts,
        ),
    }
    return {
        "schema": "axon_d00_compiled_field_capacity_audit_v1",
        "capacity_source": (
            "runtime.field.compile_field_view active SlotKind.SPAN refs"
        ),
        "derived_payload_capacities": runtime_field_payload_capacities(),
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }


def is_d00_private_adapter(adapter: str) -> bool:
    return str(adapter).startswith("d00_private_")


def _validate_sample_cap(sample_cap: int) -> int:
    if (
        isinstance(sample_cap, bool)
        or not isinstance(sample_cap, int)
        or sample_cap <= 0
    ):
        raise ValueError("sample_cap must be a positive integer")
    if sample_cap > DEFAULT_PRODUCTION_SAMPLE_CAP:
        raise ValueError(
            "sample_cap cannot exceed the production cap of "
            f"{DEFAULT_PRODUCTION_SAMPLE_CAP}"
        )
    return sample_cap


def structured_memory_query(
    *,
    limit_per_kind: int = DEFAULT_LIMIT_PER_KIND,
) -> str:
    """Return the inert marker query used by the grounded adapter.

    The query is intentionally empty.  Cross-database grounding cannot be
    represented by the generic one-database SQLite iterator, so the soul
    builder recognizes this adapter and invokes :func:`select_grounded_records`.
    The sample cap is encoded in a strict comment for the immutable
    :class:`SourceSpec`.
    """

    sample_cap = _validate_sample_cap(limit_per_kind)
    return (
        "SELECT 1 AS d00_grounded_adapter_sentinel WHERE 0 "
        f"/* d00_sample_cap={sample_cap} */"
    )


def sample_cap_from_spec(spec: SourceSpec) -> int:
    """Extract and validate the bounded sample cap from a private spec."""

    if not is_d00_private_adapter(spec.adapter):
        raise ValueError("not a D00 private adapter")
    match = _SAMPLE_CAP_PATTERN.search(spec.query or "")
    if match is None:
        raise D00GroundingError("D00 source spec is missing its sample cap")
    return _validate_sample_cap(int(match.group(1)))


def allocate_family_counts(sample_cap: int) -> dict[str, int]:
    """Allocate 45/45/10 counts by deterministic largest remainder."""

    total = _validate_sample_cap(sample_cap)
    raw = {family: total * FAMILY_WEIGHTS[family] for family in FAMILY_ORDER}
    counts = {family: int(raw[family]) for family in FAMILY_ORDER}
    remaining = total - sum(counts.values())
    ranked = sorted(
        FAMILY_ORDER,
        key=lambda family: (
            -(raw[family] - counts[family]),
            FAMILY_ORDER.index(family),
        ),
    )
    for family in ranked[:remaining]:
        counts[family] += 1
    return counts


def discover_d00_private_soul_sources(
    source_root: str | Path,
    *,
    limit_per_kind: int = DEFAULT_LIMIT_PER_KIND,
) -> list[SourceSpec]:
    """Discover the canonical semantic+episodic pair without opening it.

    One sentinel spec represents the pair.  Discovery fails closed (returns no
    private source) unless both canonical databases exist.  Residual databases
    and the personal log are optional audit-only inputs.
    """

    root = Path(source_root)
    semantic = root / D00_SEMANTIC_DATABASE
    episodic = root / D00_EPISODIC_DATABASE
    if not semantic.is_file() or not episodic.is_file():
        return []
    return [
        SourceSpec.sqlite(
            semantic,
            query=structured_memory_query(limit_per_kind=limit_per_kind),
            adapter=D00_PRIVATE_ADAPTER,
        )
    ]


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _first_step(value: Any) -> str:
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    if isinstance(parsed, dict):
        for key in ("steps", "step", "text", "action", "description"):
            if key in parsed:
                return _first_step(parsed[key])
        return ""
    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            step = _first_step(item)
            if step:
                return step
        return ""
    return _text(parsed)


def _canonical_assertion(
    family: str,
    fields: tuple[Any, ...],
) -> tuple[str, str, str] | None:
    """Return ``(target, answer component, natural query)``."""

    values = tuple(_text(value) for value in fields)
    if not all(values):
        return None
    if family == "fact":
        key, value = values
        target = f"{key}: {value}"
        component = value
        query = f"What value was stored for {key}?"
    elif family == "relation":
        subject, predicate, object_ = values
        target = f"{subject} {predicate} {object_}"
        component = object_
        query = f"What does {subject} {predicate}?"
    elif family == "procedure":
        name, first_step = values
        target = f"{name}: {first_step}"
        component = first_step
        query = f"What is the first step for {name}?"
    else:
        raise ValueError(f"unknown D00 family {family!r}")
    return target, component, query


def _query_is_answer_free(query: str, answer_component: str) -> bool:
    answer = _normalized(answer_component)
    return bool(answer) and answer not in _normalized(query)


def _template_key(family: str, fields: tuple[str, ...]) -> str:
    if family == "fact":
        seed = fields[0]
    elif family == "relation":
        seed = fields[1]
    else:
        seed = fields[0]
    normalized = re.sub(r"\d+", "#", seed.casefold())
    normalized = re.sub(r"[a-f0-9]{8,}", "<id>", normalized)
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"{family}:{normalized}"


def _supported(*texts: str) -> bool:
    try:
        for text in texts:
            assert_supported_text(text)
    except (TypeError, ValueError):
        return False
    return True


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(sqlite_ro_uri(path), uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _episode_grounding(
    episodic_path: Path,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, int],
]:
    grounding: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in FAMILY_ORDER
    }
    counters: dict[str, int] = defaultdict(int)
    connection = _open_read_only(episodic_path)
    try:
        for row in connection.execute(
            "SELECT id, extracted_json FROM episodes ORDER BY id"
        ):
            counters["episode_rows"] += 1
            raw = row["extracted_json"]
            if not isinstance(raw, str) or not raw.strip():
                counters["empty_extracted_json"] += 1
                continue
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                counters["malformed_extracted_json"] += 1
                continue
            if not isinstance(payload, dict):
                counters["non_object_extracted_json"] += 1
                continue
            collections = {
                "fact": payload.get("facts", []),
                "relation": payload.get("relations", []),
                "procedure": payload.get("procedures", []),
            }
            for family in FAMILY_ORDER:
                items = collections[family]
                if not isinstance(items, list):
                    counters[f"{family}_non_list"] += 1
                    continue
                for index, item in enumerate(items):
                    counters[f"{family}_items_seen"] += 1
                    if not isinstance(item, dict):
                        counters[f"{family}_invalid_items"] += 1
                        continue
                    if family == "fact":
                        fields = (item.get("key"), item.get("value"))
                    elif family == "relation":
                        fields = (
                            item.get("subject"),
                            item.get("predicate"),
                            item.get("object"),
                        )
                    else:
                        fields = (
                            item.get("name"),
                            _first_step(
                                item.get("steps", item.get("steps_json"))
                            ),
                        )
                    canonical = _canonical_assertion(family, fields)
                    if canonical is None:
                        counters[f"{family}_invalid_items"] += 1
                        continue
                    target, _, _ = canonical
                    grounding[family].setdefault(
                        target,
                        {
                            "episode_id": int(row["id"]),
                            "item_index": index,
                            "item_sha256": sha256_bytes(
                                canonical_json_bytes(item)
                            ),
                            "assertion_sha256": sha256_text(target),
                        },
                    )
        for family in FAMILY_ORDER:
            counters[f"{family}_unique_assertions"] = len(grounding[family])
    finally:
        connection.close()
    return grounding, dict(sorted(counters.items()))


def _semantic_candidates(
    semantic_path: Path,
    grounding: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {
        family: [] for family in FAMILY_ORDER
    }
    counters: dict[str, dict[str, int]] = {
        family: defaultdict(int) for family in FAMILY_ORDER
    }
    seen: dict[str, set[str]] = {family: set() for family in FAMILY_ORDER}
    payload_capacities = runtime_field_payload_capacities()
    connection = _open_read_only(semantic_path)
    queries = {
        "fact": (
            "SELECT id, key, value, source, created_at "
            "FROM extracted_facts ORDER BY id"
        ),
        "relation": (
            "SELECT id, subject, predicate, object, source, created_at "
            "FROM extracted_relations ORDER BY id"
        ),
        "procedure": (
            "SELECT id, name, steps_json, source, created_at "
            "FROM extracted_procedures ORDER BY id"
        ),
    }
    try:
        for family in FAMILY_ORDER:
            for row in connection.execute(queries[family]):
                counter = counters[family]
                counter["semantic_rows_seen"] += 1
                if family == "fact":
                    raw_fields = (row["key"], row["value"])
                elif family == "relation":
                    raw_fields = (
                        row["subject"],
                        row["predicate"],
                        row["object"],
                    )
                else:
                    raw_fields = (
                        row["name"],
                        _first_step(row["steps_json"]),
                    )
                fields = tuple(_text(value) for value in raw_fields)
                canonical = _canonical_assertion(family, fields)
                if canonical is None:
                    counter["quarantined_invalid_shape"] += 1
                    continue
                target, component, query = canonical
                if (
                    len(target) < MIN_TARGET_CHARS
                    or len(_normalized(component)) < 4
                ):
                    counter["quarantined_contract_length"] += 1
                    continue
                if family == "fact" and _SENSITIVE_KEY_PATTERN.search(fields[0]):
                    counter["quarantined_sensitive_key"] += 1
                    continue
                if not _query_is_answer_free(query, component):
                    counter["quarantined_query_answer_leakage"] += 1
                    continue
                grounding_ref = grounding[family].get(target)
                if grounding_ref is None:
                    counter["quarantined_missing_exact_grounding"] += 1
                    continue
                if target in seen[family]:
                    counter["quarantined_semantic_duplicate"] += 1
                    continue
                evidence = (
                    f"episode:{grounding_ref['episode_id']} "
                    f"{family}:{grounding_ref['item_index']}"
                )
                if not _supported(target, component, query, evidence):
                    counter["quarantined_unsupported_text"] += 1
                    continue
                memory_text = render_soul_memory_text(target, evidence)
                required_chars = {
                    "target_response_draft": len(target),
                    "recall_query_user_input": len(query),
                    "tick_a_structured_knowledge": len(memory_text),
                }
                if any(
                    required_chars[label] > payload_capacities[label]
                    for label in required_chars
                ):
                    counter[
                        "quarantined_compiled_field_capacity"
                    ] += 1
                    continue
                seen[family].add(target)
                counter["grounded_contract_eligible"] += 1
                candidates[family].append(
                    {
                        "family": family,
                        "semantic_row_id": int(row["id"]),
                        "semantic_table": f"extracted_{family}s",
                        "semantic_source": _text(row["source"]),
                        "semantic_created_at": _text(row["created_at"]),
                        "target": target,
                        "answer_component": component,
                        "query": query,
                        "evidence": evidence,
                        "field_capacity_required_chars": required_chars,
                        "template_key": _template_key(family, fields),
                        "grounding": grounding_ref,
                    }
                )
    finally:
        connection.close()
    return candidates, {
        family: dict(sorted(counters[family].items()))
        for family in FAMILY_ORDER
    }


def _apply_template_cap(
    rows: Iterable[dict[str, Any]],
    *,
    family: str,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["template_key"])].append(row)
    kept: list[dict[str, Any]] = []
    quarantined = 0
    for template in sorted(grouped):
        ranked = sorted(
            grouped[template],
            key=lambda row: (
                sha256_text(
                    f"{seed}|template|{family}|{row['target']}"
                ),
                int(row["semantic_row_id"]),
            ),
        )
        kept.extend(ranked[:TEMPLATE_CAP])
        quarantined += max(0, len(ranked) - TEMPLATE_CAP)
    return kept, quarantined


def _residual_audit(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    connection = _open_read_only(path)
    try:
        table_names = sorted(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        for table in table_names:
            escaped_table = table.replace('"', '""')
            counts[table] = int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{escaped_table}"'
                ).fetchone()[0]
            )
    finally:
        connection.close()
    structured = sum(
        count
        for table, count in counts.items()
        if table.startswith("extracted_")
    )
    all_rows = sum(counts.values())
    return {
        "path": str(path.resolve()),
        "table_counts": dict(sorted(counts.items())),
        "quarantined_rows": all_rows,
        "quarantined_structured_rows": structured,
        "excluded_message_rows": counts.get("messages", 0),
        "policy": "all_tables_audit_only_never_emit",
    }


def _personal_log_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    count = len(entries) if isinstance(entries, list) else 0
    return {
        "path": str(path.resolve()),
        "entries": count,
        "manual_only_entries": count,
        "emitted_rows": 0,
        "policy": "manual_review_only_never_emit",
    }


def select_grounded_records(
    spec: SourceSpec,
    *,
    seed: int,
) -> D00GroundedSelection:
    """Select a deterministic grounded sample while proving source immutability."""

    if spec.kind != "sqlite" or spec.adapter != D00_PRIVATE_ADAPTER:
        raise ValueError("select_grounded_records requires the D00 sentinel spec")
    sample_cap = sample_cap_from_spec(spec)
    semantic_path = Path(spec.path).resolve()
    root = semantic_path.parent
    episodic_path = (root / D00_EPISODIC_DATABASE).resolve()
    if semantic_path.name != D00_SEMANTIC_DATABASE:
        raise D00GroundingError(
            f"canonical target database must be {D00_SEMANTIC_DATABASE}"
        )
    if not episodic_path.is_file():
        raise D00GroundingError(
            f"required grounding database is missing: {episodic_path}"
        )

    roles: list[tuple[str, Path]] = [
        ("canonical_semantic_targets", semantic_path),
        ("exact_episodic_grounding", episodic_path),
    ]
    residual_paths = [
        (root / name).resolve()
        for name in D00_RESIDUAL_DATABASES
        if (root / name).is_file()
    ]
    roles.extend(("quarantined_old_memory", path) for path in residual_paths)
    personal_path = (root / D00_PERSONAL_LOG).resolve()
    if personal_path.is_file():
        roles.append(("manual_only_personal_log", personal_path))

    before_hashes = {path: sha256_file(path) for _, path in roles}
    before_mtimes = {path: path.stat().st_mtime_ns for _, path in roles}
    try:
        grounding, episodic_counters = _episode_grounding(episodic_path)
        family_candidates, semantic_counters = _semantic_candidates(
            semantic_path,
            grounding,
        )
        residual_audits = [
            _residual_audit(path) for path in residual_paths
        ]
    except sqlite3.Error as exc:
        raise D00GroundingError(
            "D00 database schema/read contract failed"
        ) from exc
    personal_audit = (
        _personal_log_audit(personal_path)
        if personal_path.is_file()
        else {
            "path": str(personal_path),
            "present": False,
            "entries": 0,
            "manual_only_entries": 0,
            "emitted_rows": 0,
            "policy": "manual_review_only_never_emit",
        }
    )

    quotas = allocate_family_counts(sample_cap)
    selected: list[dict[str, Any]] = []
    family_audit: dict[str, dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        capped, template_quarantine = _apply_template_cap(
            family_candidates[family],
            family=family,
            seed=seed,
        )
        quota = quotas[family]
        ranked = sorted(
            capped,
            key=lambda row: (
                sha256_text(f"{seed}|sample|{family}|{row['target']}"),
                int(row["semantic_row_id"]),
            ),
        )
        family_selected: list[dict[str, Any]] = []
        exact_capacity_quarantine = 0
        for row in ranked:
            capacity_audit = audit_compiled_field_capacity(
                target=str(row["target"]),
                evidence=str(row["evidence"]),
                recall_query=str(row["query"]),
            )
            if not capacity_audit["passed"]:
                exact_capacity_quarantine += 1
                continue
            selected_row = dict(row)
            selected_row["field_capacity_audit"] = capacity_audit
            family_selected.append(selected_row)
            if len(family_selected) == quota:
                break
        if len(family_selected) < quota:
            raise D00GroundingError(
                f"{family} has {len(family_selected)} exact compiled-view "
                f"eligible rows but requires {quota}"
            )
        selected.extend(family_selected)
        family_audit[family] = {
            "weight": FAMILY_WEIGHTS[family],
            "quota": quota,
            "grounded_contract_eligible": len(family_candidates[family]),
            "eligible_after_template_cap": len(capped),
            "quarantined_template_concentration": template_quarantine,
            "quarantined_exact_compiled_view_mismatch": (
                exact_capacity_quarantine
            ),
            "not_selected_after_cap": (
                len(capped) - quota - exact_capacity_quarantine
            ),
            "emitted": len(family_selected),
            **semantic_counters[family],
        }

    semantic_sha = before_hashes[semantic_path]
    records: list[SourceRecord] = []
    for row in sorted(
        selected,
        key=lambda item: (
            FAMILY_ORDER.index(str(item["family"])),
            sha256_text(f"{seed}|order|{item['target']}"),
        ),
    ):
        family = str(row["family"])
        grounding_ref = row["grounding"]
        target = str(row["target"])
        answer_component = str(row["answer_component"])
        source_row = {
            "schema": "axon_d00_grounded_semantic_source_v2",
            "source_id": (
                f"semantic:{family}:{int(row['semantic_row_id'])}"
            ),
            "lineage_id": (
                f"semantic:{family}:{sha256_text(target).lower()}"
            ),
            "fact_text": target,
            "evidence_text": str(row["evidence"]),
            "recall_query": str(row["query"]),
            "answer_bearing_texts": [
                target,
                answer_component,
                str(row["evidence"]),
            ],
            "memory_kind": family,
            "answer_component_sha256": sha256_text(answer_component),
            "semantic_table": str(row["semantic_table"]),
            "semantic_row_id": int(row["semantic_row_id"]),
            "semantic_source": str(row["semantic_source"]),
            "semantic_created_at": str(row["semantic_created_at"]),
            "grounding_episode_id": int(grounding_ref["episode_id"]),
            "grounding_item_index": int(grounding_ref["item_index"]),
            "grounding_item_sha256": str(grounding_ref["item_sha256"]),
            "grounding_assertion_sha256": str(
                grounding_ref["assertion_sha256"]
            ),
            "exact_episodic_grounding": True,
            "field_capacity_audit": row["field_capacity_audit"],
            "local_only": True,
            "cloud_export_allowed": False,
        }
        if (
            source_row["local_only"] is not True
            or source_row["cloud_export_allowed"] is not False
        ):
            raise D00GroundingError("private row privacy contract failed")
        pointer = (
            f"sqlite:{semantic_path}#"
            f"{row['semantic_table']}:{int(row['semantic_row_id'])}"
        )
        records.append(
            SourceRecord(
                row=source_row,
                pointer=pointer,
                source_path=str(semantic_path),
                source_sha256=semantic_sha,
                row_sha256=sha256_bytes(canonical_json_bytes(source_row)),
                adapter=D00_PRIVATE_ADAPTER,
            )
        )

    after_hashes = {path: sha256_file(path) for _, path in roles}
    after_mtimes = {path: path.stat().st_mtime_ns for _, path in roles}
    for _, path in roles:
        if (
            after_hashes[path] != before_hashes[path]
            or after_mtimes[path] != before_mtimes[path]
        ):
            raise SourceMutationError(
                f"source changed during read-only build: {path}"
            )

    rows_seen_by_role = {
        "canonical_semantic_targets": sum(
            int(family_audit[family]["semantic_rows_seen"])
            for family in FAMILY_ORDER
        ),
        "exact_episodic_grounding": int(
            episodic_counters.get("episode_rows", 0)
        ),
        "manual_only_personal_log": int(personal_audit["entries"]),
    }
    residual_by_path = {
        Path(item["path"]).resolve(): item for item in residual_audits
    }
    source_manifest: list[dict[str, Any]] = []
    for role, path in roles:
        residual = residual_by_path.get(path)
        rows_seen = (
            sum(residual["table_counts"].values())
            if residual is not None
            else rows_seen_by_role.get(role, 0)
        )
        source_manifest.append(
            {
                "kind": "json" if path.suffix.casefold() == ".json" else "sqlite",
                "path": str(path),
                "adapter": D00_PRIVATE_ADAPTER,
                "role": role,
                "sha256_before": before_hashes[path],
                "sha256_after": after_hashes[path],
                "mtime_ns_before": before_mtimes[path],
                "mtime_ns_after": after_mtimes[path],
                "read_only_verified": True,
                "rows_seen": rows_seen,
                "episodes_emitted": (
                    len(records)
                    if role == "canonical_semantic_targets"
                    else 0
                ),
            }
        )

    old_memory_quarantine = sum(
        int(item["quarantined_structured_rows"])
        for item in residual_audits
    )
    old_memory_all_rows = sum(
        int(item["quarantined_rows"]) for item in residual_audits
    )
    semantic_missing_grounding = sum(
        int(
            family_audit[family].get(
                "quarantined_missing_exact_grounding", 0
            )
        )
        for family in FAMILY_ORDER
    )
    audit = {
        "schema": "axon_d00_grounded_adapter_audit_v2",
        "canonical_target_store": D00_SEMANTIC_DATABASE,
        "grounding_store": (
            f"{D00_EPISODIC_DATABASE}:episodes.extracted_json"
        ),
        "sample_seed": seed,
        "production_sample_cap": DEFAULT_PRODUCTION_SAMPLE_CAP,
        "requested_sample_cap": sample_cap,
        "allocation_policy": (
            "deterministic_largest_remainder_45_45_10_no_redistribution"
        ),
        "family_order": list(FAMILY_ORDER),
        "family_counts": family_audit,
        "template_cap_per_family_template": TEMPLATE_CAP,
        "grounding": {
            "required": True,
            "exact_canonical_assertion_match": True,
            "episodic_counters": episodic_counters,
            "semantic_rows_missing_exact_grounding": (
                semantic_missing_grounding
            ),
            "emitted_rows_exactly_grounded": len(records),
            "passed": all(
                bool(record.row["exact_episodic_grounding"])
                for record in records
            ),
        },
        "compiled_field_capacity": {
            "capacity_source": (
                "runtime.field.compile_field_view active SlotKind.SPAN refs"
            ),
            "derived_payload_capacities": (
                runtime_field_payload_capacities()
            ),
            "exact_audit_required_before_emission": True,
            "emitted_rows_exact_compiled_view_passed": sum(
                1
                for record in records
                if record.row["field_capacity_audit"]["passed"]
            ),
            "passed": all(
                bool(record.row["field_capacity_audit"]["passed"])
                for record in records
            ),
        },
        "quarantine_counts": {
            "semantic_missing_exact_grounding": semantic_missing_grounding,
            "semantic_invalid_or_disallowed": sum(
                sum(
                    int(value)
                    for key, value in family_audit[family].items()
                    if key.startswith("quarantined_")
                    and key
                    not in {
                        "quarantined_missing_exact_grounding",
                        "quarantined_template_concentration",
                    }
                )
                for family in FAMILY_ORDER
            ),
            "semantic_template_concentration": sum(
                int(
                    family_audit[family][
                        "quarantined_template_concentration"
                    ]
                )
                for family in FAMILY_ORDER
            ),
            "old_memory_all_rows": old_memory_all_rows,
            "old_memory_structured_residuals": old_memory_quarantine,
            "personal_log_manual_only_entries": int(
                personal_audit["manual_only_entries"]
            ),
        },
        "old_memory_residuals": residual_audits,
        "personal_log": personal_audit,
        "privacy": {
            "all_emitted_rows_local_only": all(
                record.row.get("local_only") is True for record in records
            ),
            "all_emitted_rows_cloud_export_allowed_false": all(
                record.row.get("cloud_export_allowed") is False
                for record in records
            ),
            "personal_log_manual_only": True,
        },
        "source_hashes": {
            str(path): before_hashes[path] for _, path in roles
        },
        "emitted_rows": len(records),
        "passed": (
            len(records) == sample_cap
            and all(
                bool(record.row["field_capacity_audit"]["passed"])
                for record in records
            )
            and all(
                family_audit[family]["emitted"] == quotas[family]
                for family in FAMILY_ORDER
            )
        ),
    }
    if not audit["grounding"]["passed"] or not audit["passed"]:
        raise D00GroundingError("D00 grounded selection audit failed")
    return D00GroundedSelection(
        records=tuple(records),
        source_manifest=tuple(source_manifest),
        audit=audit,
    )


__all__ = [
    "D00_PRIVATE_ADAPTER",
    "D00_SEMANTIC_DATABASE",
    "D00_EPISODIC_DATABASE",
    "D00_RESIDUAL_DATABASES",
    "D00_PERSONAL_LOG",
    "D00_MEMORY_DATABASES",
    "DEFAULT_PRODUCTION_SAMPLE_CAP",
    "DEFAULT_LIMIT_PER_KIND",
    "FAMILY_ORDER",
    "FAMILY_WEIGHTS",
    "TEMPLATE_CAP",
    "SOUL_WRITE_INSTRUCTION",
    "D00GroundingError",
    "D00GroundedSelection",
    "render_soul_memory_text",
    "runtime_field_payload_capacities",
    "audit_compiled_field_capacity",
    "is_d00_private_adapter",
    "structured_memory_query",
    "sample_cap_from_spec",
    "allocate_family_counts",
    "discover_d00_private_soul_sources",
    "select_grounded_records",
]
