"""Build the minimal Axon phase0b Kaggle training bundle.

The bundle intentionally includes only:
- current clean source needed by the charslot trainer
- phase0b curriculum JSONL files
- the selected checkpoint and verified resume-step metadata
- Kaggle notebook/docs

It does not include raw memory databases, old runs, or experimental state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import pathlib
import random
import sys
import time
import zipfile
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_PHASE0B_FAMILIES = (
    "runtime_response_delta_chain_v3",
    "structured_knowledge_delta_chain_v3",
    "scratchpad_delta_chain_v3",
)

LEG_KINDS = ("standard", "quarantined_repair", "quarantined_pilot")

OPTIONAL_SOUL_FAMILIES = (
    "soul_exhale_pair_v1",
    "soul_ablation_probe_v1",
)

OPTIONAL_DIARY_FAMILIES = ("diary_continuity_delta_v1",)

SOURCE_DIRS = ("cores", "substrate", "training")
SOURCE_FILES = ("pyproject.toml", "README.md", "PROVENANCE.md")
TEST_FILES = (
    "tests/test_trainer_slot_smoke.py",
    "tests/test_trainer_resume_prereqs.py",
    "tests/test_charslot_roundtrip.py",
    "tests/test_checkpoint_continuity.py",
    "tests/test_checkpoint_contract.py",
    "tests/test_phase0b_exact_curriculum.py",
    "tests/test_promotion_gate.py",
)
KAGGLE_FILES = (
    "kaggle/README.md",
    "kaggle/axon_phase0b_train.ipynb",
    "kaggle/pilot_gate_128d_250000_252000_exact_v3.json",
)

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".part"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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
                yield src, pathlib.PurePosixPath("axon") / src.relative_to(root).as_posix()

    for rel in SOURCE_FILES + TEST_FILES + KAGGLE_FILES:
        src = root / rel
        if src.exists() and src.is_file():
            yield src, pathlib.PurePosixPath("axon") / rel.replace("\\", "/")


def read_checkpoint_step(checkpoint: pathlib.Path) -> int | None:
    pointer = checkpoint.parent / "pointer.json"
    if pointer.exists():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            if data.get("active") == checkpoint.name:
                return int(data["step"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            pass

    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            value = data.get("final_step", data.get("step"))
            return int(value)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
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


def fixed_eval_contract(
    *,
    root: pathlib.Path,
    phase0b_dir: pathlib.Path,
    families: list[str],
    family_sampling: str,
    family_weights: list[float],
    eval_n: int,
    eval_partial_frac: float,
    seed: int,
) -> dict[str, object]:
    """Reproduce the trainer's deterministic fixed dev-suite identity."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from training.trainer_slot import Phase0bCurriculum

    curriculum = Phase0bCurriculum(
        str(phase0b_dir),
        families,
        max_chars=64,
        history_chars=256,
        user_chars=64,
        max_examples=100_000,
        rng=random.Random(seed),
        family_sampling=family_sampling,
        family_weights=family_weights or None,
    )
    examples = curriculum.eval_examples(eval_n)
    if len(examples) != eval_n:
        raise ValueError(f"fixed evaluation suite has {len(examples)} cases, expected {eval_n}")
    case_ids = [str(example["eval_case_id"]) for example in examples]
    suite_sha256 = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()
    return {
        "schema": "axon_charslot_fixed_eval_contract_v1",
        "seed": seed,
        "n": eval_n,
        "fixed_suite_sha256": suite_sha256,
        "family_counts": dict(
            sorted(Counter(str(example["family"]) for example in examples).items())
        ),
        "partial_fraction": eval_partial_frac,
    }


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

    phase0b_dir = pathlib.Path(args.phase0b_dir)
    if not phase0b_dir.is_absolute():
        phase0b_dir = root / phase0b_dir
    phase0b_dir = phase0b_dir.resolve()

    families = [item.strip() for item in args.phase0b_families.split(",") if item.strip()]
    if not families:
        families = list(DEFAULT_PHASE0B_FAMILIES)
    if args.include_soul:
        families.extend(OPTIONAL_SOUL_FAMILIES)
    if args.include_diary:
        families.extend(OPTIONAL_DIARY_FAMILIES)
    source_families = [
        item.strip() for item in args.source_phase0b_families.split(",") if item.strip()
    ]
    if not source_families:
        source_families = list(families)
    family_weights: list[float] = []
    if args.phase0b_family_weights.strip():
        family_weights = [
            float(value.strip()) for value in args.phase0b_family_weights.split(",")
        ]
    if args.phase0b_family_sampling == "weighted":
        if (
            len(family_weights) != len(families)
            or any(not math.isfinite(value) or value <= 0 for value in family_weights)
            or not math.isclose(sum(family_weights), 1.0, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError(
                "weighted family sampling requires one positive finite weight per family summing to 1"
            )
    elif family_weights:
        raise ValueError("--phase0b-family-weights is only valid with weighted family sampling")

    start_step = read_checkpoint_step(checkpoint)
    if start_step is None:
        raise ValueError(
            f"cannot verify checkpoint step for {checkpoint}; provide an active pointer.json "
            "or a checkpoint .pt.json sidecar"
        )
    if args.expected_start_step is not None and start_step != args.expected_start_step:
        raise ValueError(
            f"checkpoint step {start_step} does not match --expected-start-step {args.expected_start_step}"
        )
    target_step = int(args.target_steps)
    if target_step <= start_step:
        raise ValueError(f"target step {target_step} must be greater than checkpoint step {start_step}")
    leg_steps = target_step - start_step
    checkpoint_name = args.checkpoint_name or f"{args.run_name}_step{start_step}.pt"
    checkpoint_sha256 = sha256_file(checkpoint)

    mode_weights = [float(value.strip()) for value in args.charslot_mode_weights.split(",")]
    if (
        len(mode_weights) != 3
        or any(not math.isfinite(value) or value < 0 for value in mode_weights)
        or not math.isclose(sum(mode_weights), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("--charslot-mode-weights wants three finite non-negative values summing to 1")
    if not math.isfinite(args.charslot_partial_frac) or not 0.0 <= args.charslot_partial_frac <= 1.0:
        raise ValueError("--charslot-partial-frac must be between 0 and 1")
    if not math.isfinite(args.eval_partial_frac) or not 0.0 <= args.eval_partial_frac <= 1.0:
        raise ValueError("--eval-partial-frac must be between 0 and 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    for name in ("eval_every", "eval_n", "checkpoint_every"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.resume_curriculum_state_policy == "reset" and not args.curriculum_reset_reason.strip():
        raise ValueError("curriculum reset policy requires --curriculum-reset-reason")
    if args.resume_curriculum_state_policy == "restore" and args.curriculum_reset_reason.strip():
        raise ValueError("--curriculum-reset-reason is only valid with reset policy")

    curriculum_manifest_path = phase0b_dir / "phase0b_exact_curriculum_manifest.json"
    if not curriculum_manifest_path.is_file():
        raise FileNotFoundError(curriculum_manifest_path)
    curriculum_manifest = json.loads(curriculum_manifest_path.read_text(encoding="utf-8"))
    curriculum_schema = str(curriculum_manifest.get("schema", ""))
    if not curriculum_schema.startswith("axon_phase0b_exact_curriculum_manifest_v"):
        raise ValueError(f"unexpected exact curriculum schema: {curriculum_schema!r}")
    if args.leg_kind == "quarantined_pilot":
        if curriculum_schema != "axon_phase0b_exact_curriculum_manifest_v3":
            raise ValueError("quarantined pilot requires the exact-v3 curriculum schema")
        chunk_policy = curriculum_manifest.get("chunk_policy", {})
        if (
            chunk_policy.get("lossless") is not True
            or chunk_policy.get("prefer_word_or_punctuation_boundary") is not True
        ):
            raise ValueError("exact-v3 lossless boundary chunk policy is missing")
        source_balance = curriculum_manifest.get("source_balance", {})
        if (
            source_balance.get("truncate_chains") is not False
            or int(source_balance.get("max_emitted_examples_per_source", -1)) != 24
        ):
            raise ValueError("exact-v3 emitted-source balance contract is missing")
    manifest_families = curriculum_manifest.get("families")
    if not isinstance(manifest_families, dict):
        raise ValueError("exact curriculum manifest has no family map")
    for family in families:
        family_meta = manifest_families.get(family)
        if not isinstance(family_meta, dict):
            raise ValueError(f"exact curriculum manifest is missing family {family}")
        family_path = phase0b_dir / f"{family}.jsonl"
        if not family_path.is_file():
            raise FileNotFoundError(family_path)
        expected_sha256 = str(family_meta.get("output_sha256", "")).lower()
        actual_sha256 = sha256_file(family_path)
        if expected_sha256 != actual_sha256:
            raise ValueError(f"curriculum family hash mismatch: {family}")
        if args.leg_kind == "quarantined_pilot":
            selected_audit = family_meta.get("selected_audit")
            if not isinstance(selected_audit, dict):
                raise ValueError(f"exact-v3 selected audit is missing: {family}")
            for zero_key in (
                "avoidable_midword_boundaries",
                "broken_chains",
                "midword_boundaries",
                "silent_clips",
                "unsupported_substitutions",
            ):
                if int(selected_audit.get(zero_key, -1)) != 0:
                    raise ValueError(f"exact-v3 selected audit failed: {family}.{zero_key}")
    sampler_contract = curriculum_manifest.get("training_sampler_contract")
    if args.phase0b_family_sampling == "weighted":
        if not isinstance(sampler_contract, dict) or sampler_contract.get("required") is not True:
            raise ValueError("weighted exact curriculum is missing its required sampler contract")
        contract_weights = sampler_contract.get("family_weights")
        if not isinstance(contract_weights, dict):
            raise ValueError("weighted exact curriculum sampler contract has no family weights")
        expected_weights = [float(contract_weights.get(family, math.nan)) for family in families]
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(family_weights, expected_weights)
        ):
            raise ValueError("CLI family weights do not match the exact curriculum manifest")
    evaluation_contract = fixed_eval_contract(
        root=root,
        phase0b_dir=phase0b_dir,
        families=families,
        family_sampling=args.phase0b_family_sampling,
        family_weights=family_weights,
        eval_n=args.eval_n,
        eval_partial_frac=args.eval_partial_frac,
        seed=args.seed,
    )

    if args.gate_file and args.promotion_gate_file:
        raise ValueError("use only one of --gate-file and --promotion-gate-file")
    gate_file = args.gate_file or args.promotion_gate_file
    leg_gate = None
    if gate_file:
        gate_path = pathlib.Path(gate_file)
        if not gate_path.is_absolute():
            gate_path = root / gate_path
        leg_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_schema = leg_gate.get("schema")
        if gate_schema not in {"axon_promotion_gate_v1", "axon_charslot_pilot_gate_v1"}:
            raise ValueError(
                "leg gate schema must be axon_promotion_gate_v1 or axon_charslot_pilot_gate_v1"
            )
        if int(leg_gate.get("source_step", -1)) != start_step:
            raise ValueError("leg gate source_step does not match checkpoint step")
        if int(leg_gate.get("target_step", -1)) != target_step:
            raise ValueError("leg gate target_step does not match training target")
        if leg_gate.get("leg_kind") != args.leg_kind:
            raise ValueError("leg gate leg_kind does not match --leg-kind")
        default_gate_role = "continuation" if gate_schema == "axon_charslot_pilot_gate_v1" else "promotion"
        gate_role = str(leg_gate.get("gate_role", default_gate_role))
        if gate_role not in {"continuation", "promotion"}:
            raise ValueError("leg gate gate_role must be continuation or promotion")
    else:
        gate_role = None
    if args.leg_kind in {"quarantined_repair", "quarantined_pilot"} and leg_gate is None:
        raise ValueError("quarantined legs require --gate-file")
    if args.leg_kind == "quarantined_pilot" and gate_role != "continuation":
        raise ValueError("quarantined pilot requires a continuation gate")
    if (
        args.leg_kind == "quarantined_pilot"
        and leg_gate is not None
        and leg_gate.get("schema") != "axon_charslot_pilot_gate_v1"
    ):
        raise ValueError("quarantined exact-v3 pilot requires axon_charslot_pilot_gate_v1")
    if args.leg_kind == "quarantined_pilot" and leg_gate is not None:
        if leg_gate.get("fixed_suite_sha256") != evaluation_contract["fixed_suite_sha256"]:
            raise ValueError("pilot gate fixed-suite SHA-256 does not match the rebuilt evaluator suite")

    entries: list[dict[str, object]] = []
    manifest = {
        "bundle": "axon_phase0b_kaggle_bundle",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {
            "archive_path": f"checkpoints/{checkpoint_name}",
            "source_relative": checkpoint.relative_to(root).as_posix() if checkpoint.is_relative_to(root) else str(checkpoint),
            "step": start_step,
            "start_step": start_step,
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "curriculum_families": source_families,
        },
        "training": {
            "run_name": args.run_name,
            "core_cfg": args.core_cfg,
            "leg_kind": args.leg_kind,
            "start_step": start_step,
            "target_step": target_step,
            "leg_steps": leg_steps,
            "target_steps": target_step,
            "charslot_mode_weights": ",".join(str(value) for value in mode_weights),
            "charslot_partial_frac": args.charslot_partial_frac,
            "charslot_rung_preset": "manual",
            "policy_fork_step": start_step,
            "learning_rate": args.learning_rate,
            "resume_lr_policy": args.resume_lr_policy,
            "resume_curriculum_state_policy": args.resume_curriculum_state_policy,
            "curriculum_reset_reason": args.curriculum_reset_reason.strip(),
            "eval_every": args.eval_every,
            "eval_n": args.eval_n,
            "checkpoint_every": args.checkpoint_every,
            "eval_source_before_train": bool(args.eval_source_before_train),
            "eval_partial_frac": args.eval_partial_frac,
            "seed": args.seed,
            "evaluation_contract": evaluation_contract,
            "phase0b_family_sampling": args.phase0b_family_sampling,
            "phase0b_family_weights": family_weights,
            "gate_role": gate_role,
            "leg_gate": leg_gate,
            "promotion_gate": leg_gate if gate_role == "promotion" else None,
            "continuation_gate": leg_gate if gate_role == "continuation" else None,
            "continuity": {
                "optimizer_state_required_at_target": True,
                "rng_state_required_at_target": True,
                "curriculum_state_required_at_target": True,
                "source_curriculum_families": source_families,
                "target_curriculum_families": families,
                "curriculum_reset_audited": args.resume_curriculum_state_policy == "reset",
                "expected_optimizer_policy": (
                    "restore_state_override_lr"
                    if args.resume_lr_policy == "override"
                    else "restore_state_preserve_lr"
                ),
                "expected_optimizer_state_restored": True,
                "expected_effective_lr": args.learning_rate,
                "expected_sampler_policy": (
                    "reset_exact_v3"
                    if args.resume_curriculum_state_policy == "reset"
                    else "restore_exact_curriculum"
                ),
                "expected_sampler_reset": args.resume_curriculum_state_policy == "reset",
            },
        },
        "phase0b": {
            "source_relative": phase0b_dir.relative_to(root).as_posix()
            if phase0b_dir.is_relative_to(root)
            else str(phase0b_dir),
            "families": families,
            "archive_relative": f"datasets/recovered/{phase0b_dir.name}",
            "manifest_schema": curriculum_schema,
            "manifest_sha256": sha256_file(curriculum_manifest_path),
        },
        "notes": [
            "Current bridge is checkpoint-compatible 384-slot charslot.",
            "No raw memory DBs, old runs, or 8192-slot artifacts are included.",
        ],
        "files": entries,
    }

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arcname in iter_source_files(root):
            add_file(zf, src, arcname, entries)

        add_file(zf, checkpoint, pathlib.PurePosixPath("checkpoints") / checkpoint_name, entries)

        archive_dir = pathlib.PurePosixPath("axon/datasets/recovered") / phase0b_dir.name
        manifest_paths = sorted(phase0b_dir.glob("*manifest*.json"))
        for manifest_path in manifest_paths:
            add_file(zf, manifest_path, archive_dir / manifest_path.name, entries)

        for family in families:
            src = phase0b_dir / f"{family}.jsonl"
            add_file(
                zf,
                src,
                archive_dir / src.name,
                entries,
            )

        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Axon phase0b Kaggle bundle")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default="dist/axon_phase0b_kaggle_bundle.zip")
    parser.add_argument("--checkpoint", default="runs/cpu_coreA64_charslot_no8192_tight/ckpt_0.pt")
    parser.add_argument("--checkpoint-name", default="")
    parser.add_argument("--run-name", default="coreA64_phase0b")
    parser.add_argument("--core-cfg", default="A")
    parser.add_argument("--expected-start-step", type=int)
    parser.add_argument("--target-steps", type=int, default=400000)
    parser.add_argument("--leg-kind", choices=LEG_KINDS, default="standard")
    parser.add_argument("--charslot-mode-weights", default="0.3,0.3,0.4")
    parser.add_argument("--charslot-partial-frac", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resume-lr-policy", choices=["preserve", "override"], default="preserve")
    parser.add_argument(
        "--resume-curriculum-state-policy", choices=["restore", "reset"], default="restore"
    )
    parser.add_argument("--curriculum-reset-reason", default="")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-n", type=int, default=64)
    parser.add_argument("--eval-partial-frac", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-source-before-train", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-file", default="")
    parser.add_argument("--promotion-gate-file", default="")
    parser.add_argument("--phase0b-dir", default="datasets/recovered/phase0b_exact_curriculum_v3")
    parser.add_argument(
        "--phase0b-families",
        default=",".join(DEFAULT_PHASE0B_FAMILIES),
        help="Comma-separated curriculum filenames without .jsonl",
    )
    parser.add_argument(
        "--source-phase0b-families",
        default="",
        help="Comma-separated sampler families expected in the source checkpoint; defaults to target families",
    )
    parser.add_argument(
        "--phase0b-family-sampling",
        choices=["global", "weighted"],
        default="global",
    )
    parser.add_argument("--phase0b-family-weights", default="")
    parser.add_argument("--include-soul", action="store_true", help="Include soul probe/pair curriculum files")
    parser.add_argument("--include-diary", action="store_true", help="Include diary continuity curriculum file")
    args = parser.parse_args()

    output = build_bundle(args)
    print(output)
    print(f"bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
