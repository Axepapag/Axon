"""Write a state root under a short tmp base to dodge Windows MAX_PATH."""

import os
import tempfile
from pathlib import Path

# Windows MAX_PATH is 260; pytest's default tmp root nests deeply and the
# repo's own ledger records this exact failure (verbose names + long hashes).
# Build a shallow base once per test run.
_SHORT_BASE = Path(tempfile.gettempdir()).resolve() / "axw"
_SHORT_BASE.mkdir(parents=True, exist_ok=True)
try:
    _SHORT_BASE = _SHORT_BASE / str(os.getpid())
    _SHORT_BASE.mkdir(parents=True, exist_ok=True)
except OSError:
    pass


def short_state_root(prefix: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=prefix[:8], dir=_SHORT_BASE))
    return root