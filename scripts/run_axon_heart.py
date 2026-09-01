"""Run Axon's permanent Heart host against the canonical active State branch."""

from __future__ import annotations

import argparse
from pathlib import Path

from runtime.field import LogicalRegion
from runtime.heart.host import HeartHost, HeartHostConfig
from runtime.source_of_truth import capacity_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the permanent Axon Heart host")
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=capacity_policy().number("heart.idle_interval_seconds"),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="perform one heartbeat and exit (diagnostic/runtime proof)",
    )
    parser.add_argument(
        "--user",
        help="durably submit one primitive user ingress item before running",
    )
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        metavar="REGION=PERCENT",
        help=("durably set one region's unmasked newest-suffix slider (0..100); repeat for multiple regions"),
    )
    return parser


def _parse_mask(value: str) -> tuple[LogicalRegion, int]:
    try:
        raw_region, raw_percent = value.split("=", 1)
        region = LogicalRegion(raw_region.strip())
        percent = int(raw_percent.strip())
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid --mask {value!r}; expected REGION=PERCENT") from exc
    if not 0 <= percent <= 100:
        raise argparse.ArgumentTypeError("mask percent must be in [0, 100]")
    return region, percent


def main() -> int:
    args = _parser().parse_args()
    config = HeartHostConfig(idle_interval_seconds=args.idle_seconds)
    host = HeartHost(state_root=args.state_root, host_config=config)
    host.start()
    try:
        for value in args.mask:
            try:
                region, percent = _parse_mask(value)
            except argparse.ArgumentTypeError as exc:
                raise SystemExit(str(exc)) from exc
            host.set_region_unmasked_percent(region, percent)
        if args.user is not None:
            host.submit_user(args.user, provenance="scripts/run_axon_heart.py")
        if args.once:
            result = host.heartbeat()
            print(f"heartbeat={result.heartbeat_sequence} state={result.state.value} field={result.field.field_id}")
            if result.tick_image is not None:
                print(f"tick={result.tick_image.identity.tick_sequence} view={result.tick_image.view_id}")
            return 0
        host.run_forever()
        return 0
    except KeyboardInterrupt:
        host.request_stop()
        return 0
    finally:
        if host.started:
            host.stop()


if __name__ == "__main__":
    raise SystemExit(main())
