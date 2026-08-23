from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.trainer import inspect_trainer_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only inspection of Axon's Trainer parameter-control state.")
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    args = parser.parse_args()
    snapshot = inspect_trainer_state(state_root=args.state_root)
    print(json.dumps(snapshot.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
