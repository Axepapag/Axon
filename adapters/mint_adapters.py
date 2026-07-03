"""Mint canonical slot adapters for the 8192D slot field.

Produces adapters/slot8192_adapter_{d_model}d.pt for the locked d_model sizes.
The minting is deterministic and reproducible: same d_model + version always
produces the same adapter.

Usage:
    python adapters/mint_adapters.py                  # mint 64, 128, 256
    python adapters/mint_adapters.py --d-model 64      # mint one
    python adapters/mint_adapters.py --check            # verify after minting
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from slot_adapter import (  # noqa: E402
    ADAPTER_D_MODELS,
    ADAPTER_VERSION,
    SlotAdapter,
    run_check,
)
from slots.slot_spec import SLOT_WIDTH  # noqa: E402


def mint_adapter(d_model: int, output_dir: Path | None = None) -> Path:
    """Mint one adapter and save to disk."""
    if output_dir is None:
        output_dir = Path(_ROOT) / "adapters"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"slot8192_adapter_{d_model}d.pt"
    adapter = SlotAdapter(d_model=d_model)
    adapter.save(path)
    print(f"  minted {path.name} (d_model={d_model}, slot_width={SLOT_WIDTH})")
    return path


def main() -> int:
    args = sys.argv[1:]

    d_model_override = None
    check_only = False
    for arg in args:
        if arg == "--check":
            check_only = True
        elif arg.startswith("--d-model="):
            d_model_override = int(arg.split("=", 1)[1])
        elif arg == "--d-model":
            # next arg is the value
            idx = args.index(arg)
            if idx + 1 < len(args):
                d_model_override = int(args[idx + 1])

    if check_only:
        return 0 if run_check(verbose=True) else 1

    if d_model_override is not None:
        d_models = [d_model_override]
    else:
        d_models = list(ADAPTER_D_MODELS)

    print(f"Minting {len(d_models)} adapter(s): {d_models}")
    print(f"Adapter version: {ADAPTER_VERSION}")
    print(f"Slot width: {SLOT_WIDTH}")
    print()

    for d_model in d_models:
        mint_adapter(d_model)

    print()
    print("Verifying...")
    ok = run_check(verbose=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())