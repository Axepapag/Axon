#!/usr/bin/env python3
"""Materialize dormant containers into substrate-safe surfacing examples.

This module is the bridge between the deterministic dormant corpus produced by
semantic_layout_machine.py and training/runtime shared fields.

Important contract:
    - Containers are envelopes, not vectors.
    - Every visible character is materialized through substrate.py one row at a
      time.
    - Container.symbols must equal actual semantic edge symbols only.
    - Layout symbols remain metadata/search indexes unless promoted to real
      semantic edges.

The default CLI writes a compact JSONL surfacing curriculum. It does not dump
large tensor blobs by default; trainers materialize examples on demand through
field_contract.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from container_schema import Container, SemanticEdge, normalize_container
from substrate import SLOT_DIM, char_to_slot, default_alphabet, text_to_field


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LAYOUT_DIR = REPO_ROOT / "datasets" / "recovered" / "semantic_layout"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "recovered" / "field_surfacing"

# Keep first-lane training strings conservative. The substrate can currently
# render a small punctuation set, but alnum+space strings avoid target-bank
# ambiguity while still preserving every letter and digit.
TRAINING_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
SYMBOL_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SanitizedText:
    """Text after conversion to the conservative substrate-visible alphabet."""

    text: str
    changed_chars: int = 0
    examples: tuple[str, ...] = ()


@dataclass
class MaterializerStats:
    """Counters written into materializer_manifest.json."""

    containers_seen: int = 0
    containers_valid: int = 0
    containers_invalid: int = 0
    edge_bearing_containers: int = 0
    examples_written: int = 0
    skipped_no_edges: int = 0
    skipped_empty_answer: int = 0
    non_training_chars: int = 0
    non_training_examples: list[str] = field(default_factory=list)
    validation_errors: dict[str, int] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.validation_errors[message] = self.validation_errors.get(message, 0) + 1

    def add_sanitized(self, result: SanitizedText) -> None:
        self.non_training_chars += result.changed_chars
        for item in result.examples:
            if item not in self.non_training_examples and len(self.non_training_examples) < 40:
                self.non_training_examples.append(item)


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def native_text(value: Any, *, max_chars: int | None = None) -> SanitizedText:
    """Convert arbitrary text into conservative substrate-visible text.

    Letters, digits, and spaces are retained. Other whitespace and punctuation
    become spaces. This does not compress words into semantic units; it only
    normalizes characters before each remaining character is passed to
    substrate.char_to_slot.
    """

    raw = "" if value is None else str(value)
    out: list[str] = []
    changed = 0
    samples: list[str] = []
    for char in raw:
        if char in TRAINING_CHARS:
            out.append(char)
        elif char.isspace():
            out.append(" ")
            if char != " ":
                changed += 1
                if len(samples) < 12:
                    samples.append(repr(char))
        else:
            out.append(" ")
            changed += 1
            if len(samples) < 12:
                samples.append(repr(char))
    text = SPACE_RE.sub(" ", "".join(out)).strip()
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return SanitizedText(text=text, changed_chars=changed, examples=tuple(samples))


def safe_symbol(value: Any) -> str:
    """Return a substrate-safe edge symbol or empty string."""

    symbol = "" if value is None else str(value).strip()
    if not symbol:
        return ""
    return "".join(ch for ch in symbol if ch in SYMBOL_CHARS)


def edge_symbols(container: Container) -> list[str]:
    """Sorted unique semantic symbols found on actual edges."""

    return sorted({safe_symbol(edge.symbol) for edge in container.edges if safe_symbol(edge.symbol)})


def validate_container(container: Container) -> list[str]:
    """Return contract violations for a normalized dormant container."""

    errors: list[str] = []
    actual = edge_symbols(container)
    declared = sorted({safe_symbol(symbol) for symbol in container.symbols if safe_symbol(symbol)})
    if declared != actual:
        errors.append("container.symbols must equal actual edge symbols")

    for symbol in declared:
        if not symbol or any(ch not in SYMBOL_CHARS for ch in symbol):
            errors.append("container.symbols contains non-substrate-safe symbol")

    for edge in container.edges:
        if not edge.edge_type.strip():
            errors.append("edge has empty edge_type")
        if not edge.target.strip():
            errors.append("edge has empty target")
        if edge.symbol and safe_symbol(edge.symbol) != str(edge.symbol).strip():
            errors.append("edge symbol contains non-substrate-safe characters")

    layout_symbols = container.metadata.get("layout_symbols", [])
    if isinstance(layout_symbols, list):
        visible_layout_symbols = sorted(
            safe_symbol(symbol) for symbol in layout_symbols
            if safe_symbol(symbol) in declared
        )
        if visible_layout_symbols and not container.edges:
            errors.append("layout symbols leaked into visible container symbols")
    return errors


def load_container_records(path: Path) -> Iterator[Container]:
    """Yield normalized Container objects from JSONL."""

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield normalize_container(json.loads(line), source=str(path))


def materialize_visible_text(text: str) -> np.ndarray:
    """Materialize visible text one character per 16D substrate row."""

    slots = text_to_field(text)
    if slots.ndim != 2 or slots.shape[1] != SLOT_DIM:
        raise ValueError(f"expected (N,{SLOT_DIM}) substrate field, got {slots.shape}")
    return slots


def assert_charwise_materialization(text: str, slots: np.ndarray) -> None:
    """Validate that slots exactly equal char_to_slot(c) for each character."""

    if slots.shape != (len(text), SLOT_DIM):
        raise AssertionError(f"shape mismatch: {slots.shape} for {len(text)} chars")
    for i, char in enumerate(text):
        if not np.array_equal(slots[i], char_to_slot(char)):
            raise AssertionError(f"slot {i} does not match char_to_slot({char!r})")


def _edge_to_parts(edge: SemanticEdge) -> tuple[str, str, str]:
    edge_type = native_text(edge.edge_type).text
    target = native_text(edge.target).text
    symbol = safe_symbol(edge.symbol)
    return edge_type, target, symbol


def render_container_visible(container: Container, *, max_edges: int | None = None) -> str:
    """Render one container as visible training text.

    The rendered string is intentionally plain words and spaces:
        word dog edge is a target animal symbol AA

    Layout symbols are not rendered. Every rendered character can be mapped
    through substrate.py independently.
    """

    source_text = native_text(container.letters or container.text).text
    parts: list[str] = []
    if source_text:
        parts.extend(["word", source_text])
    rendered_edges = 0
    for edge in container.edges:
        if max_edges is not None and rendered_edges >= max_edges:
            break
        edge_type, target, symbol = _edge_to_parts(edge)
        if not edge_type or not target:
            continue
        parts.extend(["edge", edge_type, "target", target])
        if symbol:
            parts.extend(["symbol", symbol])
        rendered_edges += 1
    return SPACE_RE.sub(" ", " ".join(parts)).strip()


def _query_source(container: Container) -> str:
    metadata = container.metadata
    for key in ("source_entity", "subject", "name", "entity", "label"):
        value = native_text(metadata.get(key, "")).text
        if value:
            return value
    return native_text(container.text or container.letters).text


def build_surfacing_example(
    container: Container,
    edge: SemanticEdge,
    *,
    max_structured_chars: int | None = None,
) -> dict[str, Any] | None:
    """Build one trainable surfaced-memory example from one edge."""

    edge_type = native_text(edge.edge_type).text
    answer = native_text(edge.target).text
    source = _query_source(container)
    symbol = safe_symbol(edge.symbol)
    if not edge_type or not answer:
        return None

    query = SPACE_RE.sub(" ", f"{source} {edge_type}".strip()).strip()
    structured = native_text(
        render_container_visible(container),
        max_chars=max_structured_chars,
    ).text
    if not structured:
        return None

    return {
        "kind": "field_surfacing_v1",
        "container_id": container.container_id,
        "source_text": source,
        "query": query,
        "edge_type": edge_type,
        "edge_symbol": symbol,
        "target": answer,
        "answer": answer,
        "structured_knowledge": structured,
        "provenance": container.provenance or container.source,
    }


def iter_surfacing_examples(
    containers_path: Path,
    *,
    limit: int = -1,
    max_examples: int = -1,
    stats: MaterializerStats | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield field-surfacing examples from edge-bearing containers."""

    produced = 0
    for container in load_container_records(containers_path):
        if limit >= 0 and stats is not None and stats.containers_seen >= limit:
            break
        if stats is not None:
            stats.containers_seen += 1
            for result in (
                native_text(container.text),
                native_text(container.letters),
                native_text(container.normalized_text),
            ):
                stats.add_sanitized(result)

        errors = validate_container(container)
        if errors:
            if stats is not None:
                stats.containers_invalid += 1
                for err in errors:
                    stats.add_error(err)
            continue

        if stats is not None:
            stats.containers_valid += 1
        if not container.edges:
            if stats is not None:
                stats.skipped_no_edges += 1
            continue
        if stats is not None:
            stats.edge_bearing_containers += 1

        for edge in container.edges:
            example = build_surfacing_example(container, edge)
            if example is None:
                if stats is not None:
                    stats.skipped_empty_answer += 1
                continue
            # Preflight the visible field now, before this reaches a trainer.
            visible = example["structured_knowledge"]
            slots = materialize_visible_text(visible)
            assert_charwise_materialization(visible, slots)
            yield example
            produced += 1
            if stats is not None:
                stats.examples_written += 1
            if max_examples >= 0 and produced >= max_examples:
                return


def write_examples(
    containers_path: Path,
    out_dir: Path,
    *,
    limit: int = -1,
    max_examples: int = -1,
) -> dict[str, Any]:
    """Write surfacing_examples.jsonl and materializer_manifest.json."""

    out_dir.mkdir(parents=True, exist_ok=True)
    examples_path = out_dir / "surfacing_examples.jsonl"
    manifest_path = out_dir / "materializer_manifest.json"
    stats = MaterializerStats()

    started = time.time()
    with examples_path.open("w", encoding="utf-8", newline="\n") as f:
        for example in iter_surfacing_examples(
            containers_path,
            limit=limit,
            max_examples=max_examples,
            stats=stats,
        ):
            f.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "kind": "axon_field_surfacing_materializer_manifest",
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_sec": round(time.time() - started, 3),
        "source_containers": str(containers_path),
        "source_containers_sha256": sha256_file(containers_path) if containers_path.exists() else "",
        "output_examples": str(examples_path),
        "output_examples_sha256": sha256_file(examples_path) if examples_path.exists() else "",
        "training_alphabet": "".join(sorted(TRAINING_CHARS)),
        "substrate_default_alphabet_size": len(default_alphabet()),
        "stats": asdict(stats),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build field-surfacing examples from dormant containers")
    ap.add_argument(
        "--containers",
        type=Path,
        default=DEFAULT_LAYOUT_DIR / "containers.jsonl",
        help="Path to semantic_layout containers.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for surfacing_examples.jsonl and materializer_manifest.json",
    )
    ap.add_argument("--limit", type=int, default=-1, help="Max containers to scan; -1 means all")
    ap.add_argument("--max-examples", type=int, default=-1, help="Max examples to write; -1 means all")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = write_examples(
        args.containers,
        args.out_dir,
        limit=args.limit,
        max_examples=args.max_examples,
    )
    stats = manifest["stats"]
    print(
        "materializer OK: "
        f"containers={stats['containers_seen']} "
        f"valid={stats['containers_valid']} "
        f"edge_containers={stats['edge_bearing_containers']} "
        f"examples={stats['examples_written']} "
        f"out={manifest['output_examples']}"
    )
    if stats["validation_errors"]:
        print(f"validation_errors={stats['validation_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
