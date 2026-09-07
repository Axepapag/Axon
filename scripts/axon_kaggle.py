#!/usr/bin/env python3
"""Jeff-friendly control surface for persistent private Kaggle training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.trainer import (  # noqa: E402
    TrainerCommandKind,
    TrainerOrgan,
    TrainerOrganCommand,
)
from runtime.trainer.cloud_jobs import CloudPacketError  # noqa: E402
from runtime.trainer.kaggle_adapter import KaggleTrainerAdapter  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=ROOT / "State",
        help="Axon's canonical State root",
    )
    parser.add_argument(
        "--owner",
        default=None,
        help="Kaggle account slug (default: whoever the Kaggle CLI is authenticated as)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor",
        help="verify CLI, login, and live quota (not accelerator entitlement)",
    )
    prepare = commands.add_parser("prepare", help="build a local packet; uploads nothing")
    prepare.add_argument("config", type=Path)
    launch = commands.add_parser("launch", help="upload privately and submit to Kaggle")
    launch.add_argument("job_id")
    launch.add_argument(
        "--yes",
        action="store_true",
        help="explicitly authorize private provider upload and accelerator submission",
    )
    status = commands.add_parser("status", help="show Kaggle's current job status")
    status.add_argument("job_id")
    monitor = commands.add_parser(
        "monitor",
        help="pick a local job if needed, then follow the live training dashboard",
    )
    monitor.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="64-hex Axon cloud job id; omit to list running jobs and choose",
    )
    monitor.add_argument("--follow", action="store_true")
    monitor.add_argument(
        "--raw",
        action="store_true",
        help="follow raw Kaggle kernel logs instead of the training dashboard",
    )
    fetch = commands.add_parser("fetch", help="download outputs into canonical cloud job State")
    fetch.add_argument("job_id")
    fetch.add_argument(
        "--no-bundle",
        action="store_true",
        help="skip the hash-manifested single-archive path and force legacy per-file download",
    )
    sync_pull = commands.add_parser(
        "sync-pull",
        help="download, verify, and extract the latest mid-run sync dataset version (observation-only)",
    )
    sync_pull.add_argument("job_id")
    sync_status = commands.add_parser(
        "sync-status",
        help="show which synced step ranges are locally verified (observation-only)",
    )
    sync_status.add_argument("job_id")
    sync_status.add_argument(
        "--no-rehash",
        action="store_true",
        help="trust pull-time receipts without rehashing members (faster, weaker)",
    )
    commands.add_parser("jobs", help="list locally known cloud jobs")
    return parser.parse_args()


def _format_job_catalog(rows: list[dict[str, Any]], *, limit: int = 12) -> str:
    lines = [
        "Axon Kaggle jobs on this machine",
        "Pick a number. Closing the monitor never stops cloud training.",
        "",
    ]
    shown = rows[:limit]
    for index, row in enumerate(shown, start=1):
        live = str(row.get("live_status") or row.get("phase") or "unknown").upper()
        name = str(row.get("name") or "Axon job")
        bits = []
        if row.get("candidate_label"):
            bits.append(str(row["candidate_label"]))
        if row.get("shape"):
            bits.append(str(row["shape"]))
        if row.get("tranche_steps"):
            bits.append(f"{row['tranche_steps']} steps")
        if row.get("resume"):
            bits.append("resume")
        if row.get("teach_multicell_copy"):
            bits.append("multi-cell teach")
        if row.get("accelerator"):
            bits.append(str(row["accelerator"]))
        lines.append(f" {index:2d}  {live:<14} {name}")
        if bits:
            lines.append("     " + "  ".join(bits))
        lines.append(f"     {row.get('job_id', '')}")
        lines.append("")
    if len(rows) > limit:
        lines.append(f"({len(rows) - limit} older jobs not shown)")
    return "\n".join(lines).rstrip()


def _pick_job(adapter: "KaggleTrainerAdapter") -> str:
    print("Checking Kaggle for live status of recent jobs...")
    catalog = adapter.job_catalog(refresh_live=True)
    if not catalog:
        raise CloudPacketError("No local Kaggle jobs yet.")
    print()
    print(_format_job_catalog(catalog))
    print()
    default_index = 0
    for index, row in enumerate(catalog):
        if row.get("live_status") in {"running", "queued"}:
            default_index = index
            break
    default_number = default_index + 1
    if not sys.stdin.isatty():
        raise CloudPacketError(
            "job id required when not interactive; listed jobs above"
        )
    raw = input(f"Choose a job [Enter = {default_number}]: ").strip()
    if not raw:
        return str(catalog[default_index]["job_id"])
    if raw.isdigit():
        number = int(raw)
        if 1 <= number <= min(len(catalog), 12):
            return str(catalog[number - 1]["job_id"])
        raise CloudPacketError(f"choice {number} is not on the list")
    lowered = raw.lower()
    if len(lowered) == 64 and all(character in "0123456789abcdef" for character in lowered):
        return lowered
    raise CloudPacketError("not a listed number or a 64-hex Axon job id")


def _short_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.1f} {suffix}"
        value /= 1024
    return f"{size} B"  # pragma: no cover


def _display(value: Any, *, machine: bool) -> None:
    if machine:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
        return
    if isinstance(value, dict) and value.get("schema") == "axon-kaggle-doctor-v1":
        print("Kaggle CLI/account is READY")
        print(f"Account: {value['owner']}")
        print(f"CLI: {value['cli']}")
        for quota in value["quota"]:
            print(
                f"{quota['resource']}: {quota['remaining']} remaining of "
                f"{quota['total']} (refresh {quota['refreshAt']})"
            )
        print("Accelerator entitlement: NOT proven by quota; each job must pass a real device compute probe")
        print("If Kaggle requests phone/identity verification, complete it before GPU/TPU launch")
        print("Credentials: official user store; never copied into Axon")
        return
    if isinstance(value, dict) and value.get("schema") == "axon-mid-run-sync-status-v1":
        print(f"Mid-run sync status for job {value['job_id']} (observation-only)")
        print(f"Sync dataset: {value.get('sync_dataset_ref') or '(not recorded)'}")
        if value["verified_ranges"]:
            ranges = ", ".join(f"{a}-{b}" for a, b in value["verified_ranges"])
            print(f"Locally verified step ranges: {ranges}")
            print(f"Contiguously verified through step: {value['verified_through_step']}")
            print(f"Verified members: {value['verified_member_count']} (rehashed: {value['rehashed']})")
        else:
            print("No locally verified sync bundles yet; the live monitor auto-pulls while watching.")
        released = value.get("released_ranges") or []
        if released:
            text = ", ".join(f"{a}-{b}" for a, b in released)
            print(
                f"Released older payload windows (receipts remain, keep {value.get('payload_keep', 3)}): {text}"
            )
        for problem in value["mismatches"]:
            print(f"MISMATCH: {problem}")
        for item in value["quarantined"]:
            print(f"QUARANTINED: {item}")
        print("Synced artifacts are observations; they never authorize a continuation.")
        return
    if isinstance(value, dict) and value.get("schema") == "axon-mid-run-sync-pull-v1":
        print(f"Sync pull for job {value['job_id']} from {value['sync_dataset_ref']} (observation-only)")
        for receipt in value["pulled"]:
            start, end = receipt["step_range"]
            print(
                f"  verified steps {start}-{end}: {receipt['member_count']} members, "
                f"archive {receipt['archive_sha256'][:16]}…"
            )
        for name in value["already_present"]:
            print(f"  already verified locally: {name}")
        if not value["pulled"] and not value["already_present"]:
            print("  no sync bundles in the latest dataset version")
        print(f"Extracted under: {value['sync_root']}")
        return
    if isinstance(value, dict) and "job_id" in value:
        print(f"Axon cloud job: {value['job_id']}")
        print(f"Phase: {value.get('phase', 'unknown')}")
        if value.get("git_revision"):
            print(f"Git revision: {value['git_revision']}")
        if value.get("uncompressed_bytes") is not None:
            print(f"Packet: {_short_bytes(int(value['uncompressed_bytes']))} in {value['file_count']} files")
        if value.get("dataset_ref"):
            print(f"Private dataset: {value['dataset_ref']}")
        if value.get("kernel_ref"):
            print(f"Private Kaggle job: {value['kernel_ref']}")
        if value.get("provider_status"):
            print(f"Kaggle says: {value['provider_status']}")
        if value.get("output_dir"):
            print(f"Downloaded outputs: {value['output_dir']}")
        if value.get("fetch_mode"):
            print(f"Fetch mode: {value['fetch_mode']}")
        return
    if isinstance(value, list):
        if not value:
            print("No local Kaggle jobs yet.")
            return
        if value[0].get("schema") == "axon-kaggle-job-catalog-row-v1":
            print(_format_job_catalog(value))
            return
        print("JOB ID                                                            PHASE")
        for item in value:
            print(f"{item['job_id']}  {item.get('phase', 'unknown')}")
        return
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _organ(adapter: KaggleTrainerAdapter) -> TrainerOrgan:
    organ = TrainerOrgan(state_root=adapter.state_root)
    organ.register_handler(
        TrainerCommandKind.EXPORT_CLOUD_PACKET,
        lambda command: adapter.prepare(command.arguments["config_path"]),
    )
    organ.register_handler(
        TrainerCommandKind.START,
        lambda command: adapter.launch(
            command.arguments["job_id"],
            confirmed=bool(command.arguments.get("confirmed")),
        ),
    )
    organ.register_handler(
        TrainerCommandKind.IMPORT_CLOUD_RESULT,
        lambda command: adapter.fetch(
            command.arguments["job_id"],
            bundle=bool(command.arguments.get("bundle", True)),
        ),
    )
    return organ


def main() -> int:
    args = _arguments()
    adapter = KaggleTrainerAdapter(
        repo_root=ROOT,
        state_root=args.state_root,
        owner=args.owner,
    )
    organ = _organ(adapter)
    try:
        if args.command == "doctor":
            value = adapter.doctor()
        elif args.command == "prepare":
            result = organ.dispatch(
                TrainerOrganCommand(
                    kind="export_cloud_packet",
                    arguments={"provider": "kaggle", "config_path": str(args.config.resolve())},
                    requested_by="kaggle-cli",
                )
            )
            if result.status.value != "completed":
                raise CloudPacketError(result.error or "packet export failed")
            value = dict(result.payload)
        elif args.command == "launch":
            result = organ.dispatch(
                TrainerOrganCommand(
                    kind="start",
                    arguments={"job_id": args.job_id, "provider": "kaggle", "confirmed": args.yes},
                    requested_by="kaggle-cli",
                )
            )
            if result.status.value != "completed":
                raise CloudPacketError(result.error or "Kaggle launch failed")
            value = dict(result.payload)
        elif args.command == "status":
            value = adapter.status(args.job_id)
        elif args.command == "monitor":
            job_id = args.job_id
            if not job_id:
                job_id = _pick_job(adapter)
            value = adapter.status(job_id)
            _display(value, machine=args.json)
            follow = bool(args.follow or not args.job_id)
            if not follow:
                return 0
            print()
            print("Closing this window does NOT stop cloud training.")
            if args.raw:
                print("Following raw Kaggle logs.\n")
                return adapter.follow_logs(job_id)
            print("Opening the live training dashboard.\n")
            from scripts.axon_training_watch import follow_job

            return follow_job(job_id, qa=True)
        elif args.command == "jobs":
            value = adapter.job_catalog(refresh_live=False)
        elif args.command == "fetch":
            result = organ.dispatch(
                TrainerOrganCommand(
                    kind="import_cloud_result",
                    arguments={
                        "job_id": args.job_id,
                        "provider": "kaggle",
                        "bundle": not args.no_bundle,
                    },
                    requested_by="kaggle-cli",
                )
            )
            if result.status.value != "completed":
                raise CloudPacketError(result.error or "Kaggle result import failed")
            value = dict(result.payload)
        elif args.command == "sync-pull":
            value = adapter.sync_pull(args.job_id)
        elif args.command == "sync-status":
            value = adapter.sync_status(args.job_id, rehash=not args.no_rehash)
        else:
            raise CloudPacketError(f"unsupported command {args.command}")
    except (CloudPacketError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"AXON KAGGLE STOPPED SAFELY: {exc}", file=sys.stderr)
        return 2
    _display(value, machine=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
