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
    monitor = commands.add_parser("monitor", help="show status or follow live Kaggle logs")
    monitor.add_argument("job_id")
    monitor.add_argument("--follow", action="store_true")
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
            print("No locally verified sync bundles yet; run sync-pull while the job runs.")
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
            value = adapter.status(args.job_id)
            _display(value, machine=args.json)
            if args.follow:
                print("\nFollowing Kaggle logs. Closing this window does NOT stop cloud training.\n")
                return adapter.follow_logs(args.job_id)
            return 0
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
            value = adapter.jobs()
    except (CloudPacketError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"AXON KAGGLE STOPPED SAFELY: {exc}", file=sys.stderr)
        return 2
    _display(value, machine=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
