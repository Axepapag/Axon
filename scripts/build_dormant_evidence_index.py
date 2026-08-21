"""Build or verify Axon's derived dormant evidence index.

The JSONL corpus beneath D:\\Axon\\State\\dormant remains authoritative.
This script only manages the disposable lookup sense beneath
State\\dormant\\.derived and never rewrites dormant corpus records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime.dormant import DormantEvidenceIndex, default_index_path

CANONICAL_STATE_ROOT = Path(r"D:\Axon\State").resolve()


def _require_canonical_state_root(value: str | Path) -> Path:
    path = Path(value).resolve()
    if path != CANONICAL_STATE_ROOT:
        raise SystemExit(f"state root must resolve exactly to {CANONICAL_STATE_ROOT}; got {path}")
    return path


def _require_derived_index_path(state_root: Path, value: str | Path | None) -> Path:
    path = default_index_path(state_root) if value is None else Path(value).resolve()
    derived_root = (state_root / "dormant" / ".derived").resolve()
    try:
        path.relative_to(derived_root)
    except ValueError as exc:
        raise SystemExit(f"index path must remain beneath {derived_root}; got {path}") from exc
    return path


def _summary(index: DormantEvidenceIndex, *, operation: str) -> dict[str, Any]:
    meta = dict(index._connection.execute("SELECT key, value FROM meta"))
    return {
        "operation": operation,
        "index_path": str(index.path),
        "index_id": index.index_id,
        "binding_id": index.binding.binding_id,
        "container_count": int(meta["container_count"]),
        "edge_count": int(meta["edge_count"]),
        "index_bytes": index.path.stat().st_size,
        "authority": str(index.corpus.root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", default=str(CANONICAL_STATE_ROOT))
    parser.add_argument("--index-path")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    state_root = _require_canonical_state_root(args.state_root)
    index_path = _require_derived_index_path(state_root, args.index_path)
    if args.verify_only:
        index = DormantEvidenceIndex.open(state_root, index_path=index_path, verify_binding=True)
        operation = "verify"
    else:
        index = DormantEvidenceIndex.build(state_root, index_path=index_path)
        operation = "build"
    try:
        print(json.dumps(_summary(index, operation=operation), sort_keys=True))
    finally:
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
