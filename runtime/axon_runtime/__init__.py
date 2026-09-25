"""Runtime-facing Axon model/adaptation surfaces.

The legacy D64 adapter remains preserved. Continuous reasoning Cores may instead
consume the Heart-served exact D16 Core Bus and widen only inside private learned
tissue.
"""

from .continuous_core_d512 import (
    CONTINUOUS_CORE_D512_ARCHITECTURE,
    CONTINUOUS_CORE_D512_OUTPUT_SCHEMA,
    ContinuousCoreD512,
    ContinuousCoreD512Config,
)
from .d64_adapter import CanonicalD64RuntimeAdapter

__all__ = [
    "CONTINUOUS_CORE_D512_ARCHITECTURE",
    "CONTINUOUS_CORE_D512_OUTPUT_SCHEMA",
    "CanonicalD64RuntimeAdapter",
    "ContinuousCoreD512",
    "ContinuousCoreD512Config",
]
