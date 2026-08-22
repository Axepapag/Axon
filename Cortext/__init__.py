"""Reserved source boundary for Axon's future Semantic Cortex.

Only stable contracts live here in Build C.1. No specialist is active merely
because this package exists; the Heart's semantic_cortex valve remains CLOSED.
"""

from .contracts import (
    SEMANTIC_SERVICE_SCHEMA,
    ExactEvidenceRef,
    SemanticObservation,
    SemanticQuery,
    SemanticSpecialist,
)

__all__ = [
    "SEMANTIC_SERVICE_SCHEMA",
    "ExactEvidenceRef",
    "SemanticQuery",
    "SemanticObservation",
    "SemanticSpecialist",
]
