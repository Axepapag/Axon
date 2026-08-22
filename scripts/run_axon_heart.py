"""Run Axon's permanent Heart host against the canonical active State branch."""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime.heart.host import HeartHost, HeartHostConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the permanent Axon Heart host")
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="perform one heartbeat and exit (diagnostic/runtime proof)",
    )
    parser.add_argument(
        "--user",
        help="durably submit one primitive user ingress item before running",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = HeartHostConfig(idle_interval_seconds=args.idle_seconds)
    host = HeartHost(state_root=args.state_root, host_config=config)
    host.start()
    try:
        if args.user is not None:
            host.submit_user(args.user, provenance="scripts/run_axon_heart.py")
        if args.once:
            result = host.heartbeat()
            print(
                f"heartbeat={result.heartbeat_sequence} state={result.state.value} "
                f"field={result.field.field_id}"
            )
            if result.tick_image is not None:
                print(
                    f"tick={result.tick_image.identity.tick_sequence} "
                    f"view={result.tick_image.view_id}"
                )
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
