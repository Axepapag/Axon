"""Day Zero runtime-facing D64 adapter surface.

The pre-D64 ExactV4/bootstrap/projection/runtime stack is archived under
``archive/day_zero_legacy_2026-08-20``. New runtime work must build on the
canonical D64 compiler/state contracts rather than importing archived modules.
"""

from .d64_adapter import CanonicalD64RuntimeAdapter

__all__ = ["CanonicalD64RuntimeAdapter"]
