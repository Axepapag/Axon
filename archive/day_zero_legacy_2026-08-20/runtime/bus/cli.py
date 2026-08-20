"""CLI entry point for the Axon collaboration bus sidecar."""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .server import make_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Axon Agent Collaboration Bus sidecar")
    parser.add_argument("--host", default="0.0.0.0", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    parser.add_argument("--state-dir", default="State/bus", help="directory for board/logs")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    app = make_app(state_dir=state_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
