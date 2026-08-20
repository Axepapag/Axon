#!/usr/bin/env python3
r"""build_recovered_curriculum.py - build training curriculum from recovered corpus.

Reads the dormant-state artifacts produced by curator/recovered_corpus_builder.py
and emits a family of training JSONL datasets plus a curriculum manifest.

Training families:
    field_surfacing_v1          - surfaced structured knowledge examples
    edge_prediction_v1          - predict edge type/target from source text
    retrieval_qa_v1             - answer questions from fact/relation containers
    procedure_next_step_v1      - predict next step / outcome for procedures
    episodic_exhale_filter_v1   - Tick A/B pairs for soul exhale training
    contradiction_alias_curation_v1 - contradiction/alias review material

Architecture alignment (SOURCE_OF_TRUTH.md):
  - Raw SQL/JSON rows are never used; inputs are Container records.
  - Semantic edges are spelled out in English; no semantic-edge symbols are
    training targets.
  - Surfacing budgets are explicit views; no silent truncation.
  - Personal-log/diary content is isolated to its own family.

CLI:
    python training/build_recovered_curriculum.py --smoke
    python training/build_recovered_curriculum.py \
        --containers D:\Axon\datasets\recovered\dormant_state_v1\containers.jsonl \
        --registry D:\Axon\datasets\recovered\dormant_state_v1\symbol_registry.jsonl \
        --edges D:\Axon\datasets\recovered\dormant_state_v1\semantic_edges.jsonl \
        --out-dir D:\Axon\datasets\recovered\curriculum_v1 \
        --families all \
        --max-examples 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Ensure repo root and curator/training directories are importable
_ROOT = str(Path(__file__).resolve().parent.parent)
_TRAINING = str(Path(__file__).resolve().parent)
_CURATOR = str(Path(__file__).resolve().parent.parent / "curator")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _TRAINING not in sys.path:
    sys.path.insert(0, _TRAINING)
if _CURATOR not in sys.path:
    sys.path.insert(0, _CURATOR)

from container_schema import Container, SemanticEdge, normalize_container

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONTAINERS = os.path.join("datasets", "recovered", "dormant_state_v1", "containers.jsonl")
DEFAULT_REGISTRY = os.path.join("datasets", "recovered", "dormant_state_v1", "symbol_registry.jsonl")
DEFAULT_EDGES = os.path.join("datasets", "recovered", "dormant_state_v1", "semantic_edges.jsonl")
DEFAULT_OUT_DIR = os.path.join("datasets", "recovered", "curriculum_v1")

ALL_FAMILIES = [
    "field_surfacing_v1",
    "edge_prediction_v1",
    "retrieval_qa_v1",
    "procedure_next_step_v1",
    "episodic_exhale_filter_v1",
    "contradiction_alias_curation_v1",
    "diary_region_self_reflection_v1",
]

RETIRED_FAMILIES = [
    "symbol_assignment_v1",
]

TRIM_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(value: Any, max_chars: int | None = None) -> str:
    """Conservative substrate-friendly text normalization."""
    raw = "" if value is None else str(value)
    out: list[str] = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(" ")
    text = TRIM_RE.sub(" ", "".join(out)).strip()
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def sha256_file(path: str) -> str:
    """Return SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_containers(path: str) -> Iterator[Container]:
    """Stream Container records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield normalize_container(json.loads(line), source=path)


def load_registry(path: str) -> Dict[str, Dict[str, Any]]:
    """Load symbol registry as symbol -> entry dict."""
    registry: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sym = obj.get("symbol")
            if sym:
                registry[sym] = obj
    return registry


def load_edges(path: str) -> List[Dict[str, Any]]:
    """Load semantic edges list from JSONL."""
    edges: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edges.append(json.loads(line))
    return edges


# ---------------------------------------------------------------------------
# Visible rendering (substrate-safe)
# ---------------------------------------------------------------------------


def render_container_visible(container: Container, *, max_edges: int | None = None) -> str:
    """Render container as plain substrate-safe words.

    Compatible with dormant_materializer.render_container_visible but avoids
    the numpy/substrate dependency.
    """
    source_text = clean_text(container.letters or container.text)
    parts: list[str] = []
    if source_text:
        parts.extend(["word", source_text])
    rendered = 0
    for edge in container.edges:
        if max_edges is not None and rendered >= max_edges:
            break
        edge_type = clean_text(edge.edge_type)
        target = clean_text(edge.target)
        if not edge_type or not target:
            continue
        parts.extend(["edge", edge_type, "target", target])
        rendered += 1
    return TRIM_RE.sub(" ", " ".join(parts)).strip()


def query_source(container: Container) -> str:
    """Best-effort source entity text for query generation."""
    metadata = container.metadata
    for key in ("source_entity", "subject", "name", "entity", "key"):
        value = clean_text(metadata.get(key, ""))
        if value:
            return value
    return clean_text(container.text or container.letters)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStats:
    """Per-family and aggregate stats."""

    family_counts: Dict[str, int] = field(default_factory=dict)
    family_skipped: Dict[str, Dict[str, int]] = field(default_factory=dict)
    source_files: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, family: str) -> None:
        self.family_counts[family] = self.family_counts.get(family, 0) + 1

    def skip(self, family: str, reason: str) -> None:
        d = self.family_skipped.setdefault(family, {})
        d[reason] = d.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_counts": dict(self.family_counts),
            "family_skipped": {k: dict(v) for k, v in self.family_skipped.items()},
            "source_files": list(self.source_files),
        }


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------


def build_field_surfacing(
    containers: Iterable[Container],
    stats: CurriculumStats,
    max_examples: int = -1,
) -> Iterator[Dict[str, Any]]:
    """Build field_surfacing_v1 examples from edge-bearing containers."""
    produced = 0
    for container in containers:
        if not container.edges:
            stats.skip("field_surfacing_v1", "no_edges")
            continue
        source = query_source(container)
        for edge in container.edges:
            edge_type = clean_text(edge.edge_type)
            target = clean_text(edge.target)
            if not edge_type or not target:
                stats.skip("field_surfacing_v1", "empty_edge")
                continue
            query = TRIM_RE.sub(" ", f"{source} {edge_type}".strip()).strip()
            structured = render_container_visible(container)
            if not structured:
                stats.skip("field_surfacing_v1", "empty_structured")
                continue
            yield {
                "family": "field_surfacing_v1",
                "container_id": container.container_id,
                "source_text": source,
                "query": query,
                "edge_type": edge_type,
                "target": target,
                "answer": target,
                "structured_knowledge": structured,
                "provenance": container.provenance or container.source,
            }
            stats.add("field_surfacing_v1")
            produced += 1
            if max_examples >= 0 and produced >= max_examples:
                return


def build_edge_prediction(
    containers: Iterable[Container],
    stats: CurriculumStats,
    max_examples: int = -1,
    negative_ratio: float = 0.5,
    rng: random.Random | None = None,
) -> Iterator[Dict[str, Any]]:
    """Build edge_prediction_v1 examples, including negatives."""
    rng = rng or random.Random(42)
    containers_list = list(containers)
    targets = [clean_text(e.target) for c in containers_list for e in c.edges if clean_text(e.target)]
    produced = 0

    for container in containers_list:
        source = query_source(container)
        if not source:
            stats.skip("edge_prediction_v1", "empty_source")
            continue
        for edge in container.edges:
            edge_type = clean_text(edge.edge_type)
            target = clean_text(edge.target)
            if not edge_type or not target:
                stats.skip("edge_prediction_v1", "empty_edge")
                continue
            yield {
                "family": "edge_prediction_v1",
                "container_id": container.container_id,
                "source_text": source,
                "edge_type": edge_type,
                "target": target,
                "is_negative": False,
                "provenance": container.provenance or container.source,
            }
            stats.add("edge_prediction_v1")
            produced += 1
            if max_examples >= 0 and produced >= max_examples:
                return

            # Negative sample
            if negative_ratio > 0 and targets:
                if rng.random() < negative_ratio:
                    neg_target = target
                    attempts = 0
                    while neg_target == target and attempts < 10:
                        neg_target = rng.choice(targets)
                        attempts += 1
                    if neg_target != target:
                        yield {
                            "family": "edge_prediction_v1",
                            "container_id": container.container_id,
                            "source_text": source,
                            "edge_type": edge_type,
                            "target": neg_target,
                            "is_negative": True,
                            "provenance": container.provenance or container.source,
                        }
                        stats.add("edge_prediction_v1")
                        produced += 1
                        if max_examples >= 0 and produced >= max_examples:
                            return


def build_retrieval_qa(
    containers: Iterable[Container],
    stats: CurriculumStats,
    max_examples: int = -1,
) -> Iterator[Dict[str, Any]]:
    """Build retrieval_qa_v1 examples from fact/relation containers."""
    produced = 0
    for container in containers:
        if container.kind not in ("fact", "relation"):
            stats.skip("retrieval_qa_v1", "wrong_kind")
            continue
        source = query_source(container)
        for edge in container.edges:
            edge_type = clean_text(edge.edge_type)
            target = clean_text(edge.target)
            if not edge_type or not target:
                stats.skip("retrieval_qa_v1", "empty_edge")
                continue
            if container.kind == "fact":
                query = f"What is {edge_type}?"
            else:
                query = TRIM_RE.sub(" ", f"{source} {edge_type} ?".strip()).strip()
            context = render_container_visible(container, max_edges=3)
            yield {
                "family": "retrieval_qa_v1",
                "container_id": container.container_id,
                "query": query,
                "answer": target,
                "context_slots_text": context,
                "edge_type": edge_type,
                "provenance": container.provenance or container.source,
            }
            stats.add("retrieval_qa_v1")
            produced += 1
            if max_examples >= 0 and produced >= max_examples:
                return


def build_procedure_next_step(
    containers: Iterable[Container],
    stats: CurriculumStats,
    max_examples: int = -1,
    max_steps: int = 10,
) -> Iterator[Dict[str, Any]]:
    """Build procedure_next_step_v1 examples from procedure containers."""
    produced = 0
    for container in containers:
        if container.kind != "procedure":
            stats.skip("procedure_next_step_v1", "wrong_kind")
            continue
        steps = container.metadata.get("steps", [])
        if not isinstance(steps, list) or not steps:
            stats.skip("procedure_next_step_v1", "no_steps")
            continue
        steps = [clean_text(s) for s in steps if clean_text(s)]
        if len(steps) > max_steps:
            stats.skip("procedure_next_step_v1", "truncated_steps")
            steps = steps[:max_steps]

        name = clean_text(container.text)
        trigger = clean_text(container.metadata.get("trigger", ""))
        outcome = clean_text(container.metadata.get("outcome", ""))

        # Outcome prediction
        if outcome:
            yield {
                "family": "procedure_next_step_v1",
                "sub_family": "outcome",
                "container_id": container.container_id,
                "procedure_name": name,
                "trigger": trigger,
                "steps": steps,
                "answer": outcome,
                "provenance": container.provenance or container.source,
            }
            stats.add("procedure_next_step_v1")
            produced += 1
            if max_examples >= 0 and produced >= max_examples:
                return

        # Step prediction: predict step i+1 from steps[:i+1]
        for i in range(len(steps) - 1):
            yield {
                "family": "procedure_next_step_v1",
                "sub_family": "next_step",
                "container_id": container.container_id,
                "procedure_name": name,
                "trigger": trigger,
                "steps_before": steps[: i + 1],
                "next_step_index": i + 1,
                "answer": steps[i + 1],
                "provenance": container.provenance or container.source,
            }
            stats.add("procedure_next_step_v1")
            produced += 1
            if max_examples >= 0 and produced >= max_examples:
                return


def build_episodic_exhale_filter(
    containers: Iterable[Container],
    stats: CurriculumStats,
    max_examples: int = -1,
) -> Iterator[Dict[str, Any]]:
    """Build episodic_exhale_filter_v1 Tick A/B pairs."""
    produced = 0
    for container in containers:
        if container.kind != "episode":
            stats.skip("episodic_exhale_filter_v1", "wrong_kind")
            continue
        summary = clean_text(container.metadata.get("summary", "") or container.text, max_chars=1024)
        if not summary:
            stats.skip("episodic_exhale_filter_v1", "empty_summary")
            continue

        facts: List[str] = []
        for edge in container.edges:
            et = clean_text(edge.edge_type)
            tgt = clean_text(edge.target)
            if et and tgt:
                facts.append(f"{et} {tgt}")

        # Tick A: episode present in structured_knowledge
        yield {
            "family": "episodic_exhale_filter_v1",
            "phase": "a",
            "container_id": container.container_id,
            "episode_summary": summary,
            "facts_to_retain": facts,
            "structured_knowledge": render_container_visible(container, max_edges=5),
            "active_regions": ["structured_knowledge"],
            "provenance": container.provenance or container.source,
        }
        stats.add("episodic_exhale_filter_v1")
        produced += 1
        if max_examples >= 0 and produced >= max_examples:
            return

        # Tick B: episode masked; require recall
        if facts:
            yield {
                "family": "episodic_exhale_filter_v1",
                "phase": "b",
                "container_id": container.container_id,
                "query": "What facts from the previous episode are relevant?",
                "expected_answer": facts[:3],
                "mask_regions": ["conversation_history", "structured_knowledge"],
                "provenance": container.provenance or container.source,
            }
            stats.add("episodic_exhale_filter_v1")
            produced += 1
            if max_examples >= 0 and produced >= max_examples:
                return


def build_diary_region_self_reflection(
    containers: Iterable[Container],
    stats: CurriculumStats,
    max_examples: int = -1,
) -> Iterator[Dict[str, Any]]:
    """Build diary-isolated examples from personal-log containers."""
    produced = 0
    for container in containers:
        if container.kind != "diary":
            stats.skip("diary_region_self_reflection_v1", "wrong_kind")
            continue
        diary_text = clean_text(container.metadata.get("content", "") or container.text, max_chars=2048)
        if not diary_text:
            stats.skip("diary_region_self_reflection_v1", "empty_diary_text")
            continue
        yield {
            "family": "diary_region_self_reflection_v1",
            "container_id": container.container_id,
            "region": "diary",
            "diary_text": diary_text,
            "prompt": "Use this diary-region memory only for self-reflection and continuity.",
            "redaction": container.metadata.get("redaction", "diary_only"),
            "allowed_regions": ["diary"],
            "blocked_families": ["retrieval_qa_v1", "edge_prediction_v1"],
            "provenance": container.provenance or container.source,
        }
        stats.add("diary_region_self_reflection_v1")
        produced += 1
        if max_examples >= 0 and produced >= max_examples:
            return


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity over token sets."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def build_contradiction_alias_curation(
    containers: Iterable[Container],
    stats: CurriculumStats,
    max_examples: int = -1,
    alias_threshold: float = 0.85,
    max_alias_pairs: int = 250_000,
) -> Iterator[Dict[str, Any]]:
    """Build contradiction/alias review material.

    Contradictions: same (source_text, edge_type) but different target.
    Aliases: different target strings with high lexical similarity.
    """
    produced = 0

    # Build index: (source_text, edge_type) -> list of (target, container_id)
    index: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    target_containers: Dict[str, Set[str]] = {}
    for container in containers:
        source = query_source(container)
        for edge in container.edges:
            et = clean_text(edge.edge_type)
            tgt = clean_text(edge.target)
            if not source or not et or not tgt:
                stats.skip("contradiction_alias_curation_v1", "empty_field")
                continue
            key = (source, et)
            index.setdefault(key, []).append((tgt, container.container_id))
            target_containers.setdefault(tgt, set()).add(container.container_id)

    # Contradictions
    for (source, et), entries in index.items():
        if len(entries) < 2:
            continue
        seen_targets: Set[str] = set()
        for tgt, cid in entries:
            if tgt in seen_targets:
                continue
            seen_targets.add(tgt)
        if len(seen_targets) > 1:
            yield {
                "family": "contradiction_alias_curation_v1",
                "type": "contradiction",
                "source_text": source,
                "edge_type": et,
                "candidates": [
                    {"target": tgt, "container_id": cid}
                    for tgt, cid in entries
                ],
                "resolution_policy": "higher_confidence_or_review",
                "provenance": "recovered_corpus_builder",
            }
            stats.add("contradiction_alias_curation_v1")
            produced += 1
            if max_examples >= 0 and produced >= max_examples:
                return

    # Alias candidates across spelled-out targets.
    buckets: Dict[str, List[Tuple[str, Set[str]]]] = {}
    for target, container_ids in target_containers.items():
        bucket_key = target.lower()[:4]
        buckets.setdefault(bucket_key, []).append((target, container_ids))

    checked_pairs = 0
    for target_list in buckets.values():
        for i in range(len(target_list)):
            target_a, ids_a = target_list[i]
            for j in range(i + 1, len(target_list)):
                if max_alias_pairs >= 0 and checked_pairs >= max_alias_pairs:
                    stats.skip("contradiction_alias_curation_v1", "alias_pair_budget_exhausted")
                    return
                checked_pairs += 1
                target_b, ids_b = target_list[j]
                if target_a == target_b:
                    continue
                sim = _jaccard(target_a, target_b)
                if sim >= alias_threshold:
                    yield {
                        "family": "contradiction_alias_curation_v1",
                        "type": "alias_candidate",
                        "target_a": target_a,
                        "target_b": target_b,
                        "container_ids_a": sorted(ids_a),
                        "container_ids_b": sorted(ids_b),
                        "jaccard": round(sim, 4),
                        "resolution_policy": "review_before_merge",
                        "provenance": "recovered_corpus_builder",
                    }
                    stats.add("contradiction_alias_curation_v1")
                    produced += 1
                    if max_examples >= 0 and produced >= max_examples:
                        return


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RecoveredCurriculumBuilder:
    """Build training curriculum JSONL artifacts from recovered corpus."""

    def __init__(
        self,
        containers_path: str = DEFAULT_CONTAINERS,
        registry_path: str = DEFAULT_REGISTRY,
        edges_path: str = DEFAULT_EDGES,
        out_dir: str = DEFAULT_OUT_DIR,
        families: Sequence[str] | None = None,
        max_examples: int = -1,
        negative_ratio: float = 0.5,
        max_steps: int = 10,
        alias_threshold: float = 0.85,
        max_alias_pairs: int = 250_000,
        seed: int = 42,
        no_personal_log: bool = True,
    ):
        self.containers_path = containers_path
        self.registry_path = registry_path
        self.edges_path = edges_path
        self.out_dir = out_dir
        self.families = list(families) if families is not None else list(ALL_FAMILIES)
        self.max_examples = max_examples
        self.negative_ratio = negative_ratio
        self.max_steps = max_steps
        self.alias_threshold = alias_threshold
        self.max_alias_pairs = max_alias_pairs
        self.seed = seed
        self.no_personal_log = no_personal_log
        self.rng = random.Random(seed)
        self.stats = CurriculumStats()

    def run(self) -> CurriculumStats:
        """Build all requested curriculum families and write manifests."""
        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self._remove_retired_family_outputs(out_path)

        self.stats.source_files = self._record_source_files()
        registry = load_registry(self.registry_path)

        # Load containers once; some builders need multiple passes.
        containers = list(load_containers(self.containers_path))
        if self.no_personal_log:
            containers = [c for c in containers if c.kind != "diary"]

        for family in self.families:
            examples = self._build_family(family, containers, registry)
            self._write_family(family, examples, out_path)

        self._write_manifest(out_path)
        return self.stats

    def _remove_retired_family_outputs(self, out_path: Path) -> None:
        """Remove stale datasets from families that are no longer valid."""
        for family in RETIRED_FAMILIES:
            path = out_path / f"{family}.jsonl"
            if path.exists():
                path.unlink()

    def _record_source_files(self) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        for path in [self.containers_path, self.registry_path, self.edges_path]:
            if os.path.exists(path):
                files.append({
                    "path": os.path.abspath(path),
                    "sha256": sha256_file(path),
                    "size_bytes": os.path.getsize(path),
                })
            else:
                files.append({
                    "path": os.path.abspath(path),
                    "sha256": "",
                    "size_bytes": 0,
                    "note": "file not found",
                })
        return files

    def _build_family(
        self,
        family: str,
        containers: List[Container],
        registry: Dict[str, Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        if family == "field_surfacing_v1":
            return build_field_surfacing(containers, self.stats, self.max_examples)
        if family == "edge_prediction_v1":
            return build_edge_prediction(containers, self.stats, self.max_examples, self.negative_ratio, self.rng)
        if family == "retrieval_qa_v1":
            return build_retrieval_qa(containers, self.stats, self.max_examples)
        if family == "procedure_next_step_v1":
            return build_procedure_next_step(containers, self.stats, self.max_examples, self.max_steps)
        if family == "episodic_exhale_filter_v1":
            return build_episodic_exhale_filter(containers, self.stats, self.max_examples)
        if family == "contradiction_alias_curation_v1":
            return build_contradiction_alias_curation(
                containers,
                self.stats,
                self.max_examples,
                self.alias_threshold,
                self.max_alias_pairs,
            )
        if family == "diary_region_self_reflection_v1":
            return build_diary_region_self_reflection(containers, self.stats, self.max_examples)
        raise ValueError(f"Unknown family: {family}")

    def _write_family(self, family: str, examples: Iterator[Dict[str, Any]], out_path: Path) -> None:
        path = out_path / f"{family}.jsonl"
        count = 0
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        self.stats.family_counts[family] = count

    def _write_manifest(self, out_path: Path) -> None:
        manifest = {
            "kind": "axon_recovered_curriculum_manifest",
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "families": self.families,
            "source_files": self.stats.source_files,
            "family_counts": self.stats.family_counts,
            "family_skipped": {k: dict(v) for k, v in self.stats.family_skipped.items()},
            "options": {
                "max_examples": self.max_examples,
                "negative_ratio": self.negative_ratio,
                "max_steps": self.max_steps,
                "alias_threshold": self.alias_threshold,
                "max_alias_pairs": self.max_alias_pairs,
                "seed": self.seed,
                "no_personal_log": self.no_personal_log,
            },
            "notes": "Training curriculum families from recovered dormant state. "
            "Budgets are explicit; skipped examples are counted.",
        }
        with open(out_path / "curriculum_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_families(value: str) -> List[str]:
    if value.lower() == "all":
        return list(ALL_FAMILIES)
    return [f.strip() for f in value.split(",") if f.strip()]


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build training curriculum from recovered dormant corpus."
    )
    ap.add_argument("--containers", default=DEFAULT_CONTAINERS, help="Path to containers.jsonl")
    ap.add_argument("--registry", default=DEFAULT_REGISTRY, help="Path to symbol_registry.jsonl")
    ap.add_argument("--edges", default=DEFAULT_EDGES, help="Path to semantic_edges.jsonl")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--families", default="all", help="Comma-separated families or 'all'")
    ap.add_argument("--max-examples", type=int, default=-1, help="Per-family example cap")
    ap.add_argument("--negative-ratio", type=float, default=0.5, help="Edge-prediction negative ratio")
    ap.add_argument("--max-steps", type=int, default=10, help="Procedure step budget")
    ap.add_argument("--alias-threshold", type=float, default=0.85, help="Jaccard alias threshold")
    ap.add_argument("--max-alias-pairs", type=int, default=250_000, help="Explicit alias pair comparison budget")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--no-personal-log", action="store_true", help="Exclude diary containers")
    ap.add_argument("--include-personal-log", action="store_true", help="Include diary containers and diary-only family")
    ap.add_argument("--smoke", action="store_true", help="Smoke mode: max-examples=200, families=all")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        args.max_examples = 200 if args.max_examples < 0 else args.max_examples
        args.families = "all"
        args.no_personal_log = True

    families = _parse_families(args.families)
    no_personal_log = args.no_personal_log or not args.include_personal_log

    builder = RecoveredCurriculumBuilder(
        containers_path=args.containers,
        registry_path=args.registry,
        edges_path=args.edges,
        out_dir=args.out_dir,
        families=families,
        max_examples=args.max_examples,
        negative_ratio=args.negative_ratio,
        max_steps=args.max_steps,
        alias_threshold=args.alias_threshold,
        max_alias_pairs=args.max_alias_pairs,
        seed=args.seed,
        no_personal_log=no_personal_log,
    )
    stats = builder.run()

    print("Recovered curriculum build complete.")
    for family, count in stats.family_counts.items():
        print(f"  {family}: {count}")
    print(f"  Output dir: {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
