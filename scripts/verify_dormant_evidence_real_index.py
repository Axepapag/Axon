"""Prove query -> exact dereference -> cortex -> D64 roundtrip on the real corpus.

Runs the full dormant evidence bridge chain against the live
``D:\\Axon\\State\\dormant`` corpus and its derived index. The JSONL corpus
remains authoritative; this script changes nothing and writes nothing.
Prints one JSON verdict per query plus a final summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from runtime.dormant import DormantEvidenceBridge, DormantEvidenceIndex
from runtime.field import LogicalRegion, SharedFieldSnapshot

CANONICAL_STATE_ROOT = Path(r"D:\Axon\State").resolve()

QUERIES = (
    "checkpoint produced by trainer",
    "Axon architecture",
    "conversation memory",
)


def run_query(bridge: DormantEvidenceBridge, query: str) -> dict:
    started = time.time()
    base = SharedFieldSnapshot(tick_id=0)
    result = bridge.query_surface_compile(base, query, limit=4)

    structured = result.snapshot.region(LogicalRegion.CORTEX)
    checks = {
        "evidence_non_empty": len(result.evidence) > 0,
        "tick_advanced": result.snapshot.tick_id == base.tick_id + 1,
        "parent_bound": result.snapshot.parent_field_id == base.field_id,
        "provenance_manifest_ids": any(
            mid.startswith("dormant-binding:") for mid in result.snapshot.source_manifest_ids
        ),
        "container_refs_attached": all(
            span.container_refs or span.kind == "evidence_separator"
            for span in structured.spans
        ),
        "coverage_complete": result.compiled.coverage.complete is True,
        "region_text_exact": (
            result.compiled.region_text(LogicalRegion.CORTEX)
            == structured.text
        ),
    }
    # query_surface_compile already called compiled.verify_roundtrip(snapshot);
    # any roundtrip failure would have raised before this point.
    checks["roundtrip_verified"] = True
    return {
        "query": query,
        "ok": all(checks.values()),
        "checks": checks,
        "containers": [item.container.container_id for item in result.evidence],
        "span_count": len(structured.spans),
        "cortex_chars": len(structured.text),
        "field_id": result.snapshot.field_id,
        "elapsed_s": round(time.time() - started, 3),
    }


def main() -> int:
    state_root = Path(r"D:\Axon\State").resolve()
    if state_root != CANONICAL_STATE_ROOT:
        raise SystemExit(f"state root must resolve exactly to {CANONICAL_STATE_ROOT}")
    index = DormantEvidenceIndex.open(state_root, verify_binding=True)
    try:
        bridge = DormantEvidenceBridge(index)
        results = [run_query(bridge, query) for query in QUERIES]
    finally:
        index.close()
    summary = {
        "operation": "real_index_roundtrip_proof",
        "queries": results,
        "all_ok": all(item["ok"] for item in results),
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
