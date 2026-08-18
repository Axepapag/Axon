"""Build the minimal exact-v4 multi-tick Kaggle training bundle.

This bundle intentionally overrides the exact-v4 curriculum manifest's
``privacy.cloud_export_allowed=false`` flag.  The dataset contains read-only
grounded records that the user (Jeff) has given blanket approval to use for
Kaggle experiments; do not upload this bundle without an equivalent explicit
authorization.

The bundle includes:
- clean source for cores, substrate, training, runtime, tests, and kaggle
- the exact-v4 curriculum (episodes.jsonl, manifest.json, per-family jsonls)
- the selected core checkpoint
- a bundle manifest with SHA-256 fingerprints and resume metadata

It does not include raw memory databases, old runs, or 8192-slot artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
import zipfile
from typing import Any, Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SOURCE_DIRS = ("cores", "substrate", "training", "runtime")
SOURCE_FILES = ("pyproject.toml", "README.md", "PROVENANCE.md")

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".part"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: pathlib.Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts) or path.suffix in SKIP_SUFFIXES


def iter_source_files(root: pathlib.Path) -> Iterable[tuple[pathlib.Path, pathlib.PurePosixPath]]:
    for dirname in SOURCE_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for src in sorted(base.rglob("*")):
            if src.is_file() and not should_skip(src):
                arcname = pathlib.PurePosixPath("axon") / src.relative_to(root).as_posix()
                yield src, arcname

    for rel in SOURCE_FILES:
        src = root / rel
        if src.exists() and src.is_file():
            yield src, pathlib.PurePosixPath("axon") / rel.replace("\\", "/")

    for dirname in ("tests", "kaggle"):
        base = root / dirname
        if not base.exists():
            continue
        for src in sorted(base.rglob("*")):
            if src.is_file() and not should_skip(src):
                arcname = pathlib.PurePosixPath("axon") / src.relative_to(root).as_posix()
                yield src, arcname


def read_checkpoint_step(checkpoint: pathlib.Path) -> int | None:
    """Best-effort read of the checkpoint step from sidecar files."""
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            value = data.get("final_step", data.get("step"))
            return int(value)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    pointer = checkpoint.parent / "pointer.json"
    if pointer.exists():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8-sig"))
            if data.get("active") == checkpoint.name:
                return int(data["step"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            pass
    return None


def add_file(
    zf: zipfile.ZipFile,
    src: pathlib.Path,
    arcname: pathlib.PurePosixPath,
    entries: list[dict[str, object]],
) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    zf.write(src, arcname.as_posix())
    entries.append(
        {
            "path": arcname.as_posix(),
            "bytes": src.stat().st_size,
            "sha256": sha256_file(src),
        }
    )


def build_bundle(args: argparse.Namespace) -> pathlib.Path:
    root = pathlib.Path(args.repo_root).resolve()
    output = pathlib.Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = pathlib.Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()

    dataset_dir = pathlib.Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    dataset_dir = dataset_dir.resolve()

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"exact-v4 manifest missing: {manifest_path}")
    curriculum_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if curriculum_manifest.get("schema") != "axon_multitick_curriculum_manifest_v1":
        raise ValueError(
            f"unexpected exact-v4 manifest schema: {curriculum_manifest.get('schema')!r}"
        )
    if curriculum_manifest.get("privacy", {}).get("cloud_export_allowed") is not False:
        # The manifest is expected to forbid cloud export.  The bundle builder
        # explicitly overrides this on the user's authority (see module docstring).
        print(
            "warning: exact-v4 manifest does not set cloud_export_allowed=false; "
            "proceeding anyway"
        )

    start_step = args.start_step
    if start_step is None:
        start_step = read_checkpoint_step(checkpoint)
    if start_step is None:
        raise ValueError(
            f"cannot determine checkpoint step for {checkpoint}; "
            "provide --start-step or a .pt.json sidecar / pointer.json"
        )
    start_step = int(start_step)

    target_step = args.target_step
    if target_step is None:
        target_step = start_step + 2000
    target_step = int(target_step)
    if target_step <= start_step:
        raise ValueError(
            f"--target-step {target_step} must be greater than start_step {start_step}"
        )

    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    for name in ("eval_every", "checkpoint_every"):
        value = int(getattr(args, name))
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    checkpoint_name = checkpoint.name
    checkpoint_sha256 = sha256_file(checkpoint)
    manifest_sha256 = sha256_file(manifest_path)

    entries: list[dict[str, object]] = []
    archive_dataset_dir = pathlib.PurePosixPath("axon/datasets/recovered") / dataset_dir.name
    manifest_archive_path = archive_dataset_dir / "manifest.json"

    manifest: dict[str, Any] = {
        "schema": "axon_exact_v4_bundle_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": [
            "Exact-v4 multi-tick continuation bundle.",
            "Curriculum manifest sets privacy.cloud_export_allowed=false; this upload relies on the user's explicit blanket approval for Kaggle.",
            "No raw memory DBs, old runs, or 8192-slot artifacts are included.",
        ],
        "checkpoint": {
            "archive_path": f"checkpoints/{checkpoint_name}",
            "source_relative": checkpoint.relative_to(root).as_posix()
            if checkpoint.is_relative_to(root)
            else str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "start_step": start_step,
        },
        "dataset": {
            "archive_path": str(archive_dataset_dir),
            "source_relative": dataset_dir.relative_to(root).as_posix()
            if dataset_dir.is_relative_to(root)
            else str(dataset_dir),
            "manifest_path": str(manifest_archive_path),
            "manifest_sha256": manifest_sha256,
            "manifest_schema": curriculum_manifest.get("schema"),
        },
        "training": {
            "run_name": args.run_name,
            "start_step": start_step,
            "target_step": target_step,
            "lr": args.lr,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "train_soul": bool(args.train_soul),
            "soul_continuity_weight": float(args.soul_continuity_weight),
            "soul_l2_weight": float(args.soul_l2_weight),
            "soul_compression_weight": float(args.soul_compression_weight),
        },
        "files": entries,
    }

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arcname in iter_source_files(root):
            add_file(zf, src, arcname, entries)

        add_file(
            zf,
            checkpoint,
            pathlib.PurePosixPath("checkpoints") / checkpoint_name,
            entries,
        )

        add_file(zf, manifest_path, manifest_archive_path, entries)

        for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
            add_file(
                zf,
                jsonl_path,
                archive_dataset_dir / jsonl_path.name,
                entries,
            )

        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Axon exact-v4 multi-tick Kaggle bundle"
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default="dist/axon_exact_v4_bundle.zip")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/core64D_2L_1H_FFN131072_charslot384_phase0b_step400000_from287000.pt",
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/recovered/phase0b_exact_curriculum_v4",
    )
    parser.add_argument("--run-name", default="core64D_exact_v4_leg1")
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-soul",
        action="store_true",
        help="Enable trainable soul-write in the bundled leg.",
    )
    parser.add_argument(
        "--soul-continuity-weight",
        type=float,
        default=0.01,
        help="Weight for the soul continuity regularizer.",
    )
    parser.add_argument(
        "--soul-l2-weight",
        type=float,
        default=1e-5,
        help="Weight for the soul L2 magnitude regularizer.",
    )
    parser.add_argument(
        "--soul-compression-weight",
        type=float,
        default=0.1,
        help="Weight for the Phase-A hot->warm soul compression loss.",
    )
    args = parser.parse_args()
    output = build_bundle(args)
    print(output)
    print(f"bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
