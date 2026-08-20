"""Build the isolated R0 v6 exact-position alignment shard."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.build_complete_field_r0_curriculum import (
    ALIGNMENT_SCHEMA,
    SCHEMA,
    TARGET_ALIGNMENT_SCHEMA,
    BUILDER_VERSION,
    digest,
    validate_exact_field_isolation,
    v6_alignment_records,
    write_jsonl,
)


def build(output_dir: Path, count: int, seed: int, page_size: int) -> dict[str, Any]:
    records = list(v6_alignment_records(count, seed=seed, page_size=page_size))
    validate_exact_field_isolation(records)
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "dev", "test")
    }
    if any(not rows for rows in by_split.values()):
        raise ValueError("v6 alignment shard must populate train, dev, and test")
    files: dict[str, Any] = {}
    for split, rows in by_split.items():
        path = output_dir / f"{split}.jsonl"
        records_written, sha256 = write_jsonl(path, rows)
        files[split] = {
            "path": str(path),
            "records": records_written,
            "sha256": sha256,
        }
    token_lengths = [len(record["targets"]["response_draft"]) for record in records]
    manifest: dict[str, Any] = {
        "schema": "axon-r0-v6-alignment-shard-manifest-v1",
        "builder_version": BUILDER_VERSION,
        "example_schema": SCHEMA,
        "alignment_schema": ALIGNMENT_SCHEMA,
        "target_alignment_schema": TARGET_ALIGNMENT_SCHEMA,
        "local_only": True,
        "cloud_export_allowed": False,
        "count": len(records),
        "seed": seed,
        "page_size": page_size,
        "layouts": dict(Counter(record["provenance"]["layout"] for record in records)),
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
        "wrong_region_duplicates": sum(
            bool(record["provenance"]["wrong_region_duplicate"]) for record in records
        ),
        "same_region_duplicates": sum(
            bool(record["provenance"]["same_region_duplicate"]) for record in records
        ),
        "page_boundary_records": sum(
            record["provenance"]["layout"] == "page_boundary" for record in records
        ),
        "files": files,
    }
    manifest["manifest_sha256"] = digest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build isolated R0 v6 alignment shard")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"State\training\curriculum\complete_field_r0_v6_alignment"),
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=70024)
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()
    manifest = build(args.output_dir, args.count, args.seed, args.page_size)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
