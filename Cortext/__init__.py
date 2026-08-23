"""Reserved source boundary for Axon's grounded Semantic Cortex work.

D.2 adds evaluation-first consumer anatomy around the accepted D64 dual
surface.  No specialist becomes active merely because this package exists;
the Heart's ``semantic_cortex`` valve remains CLOSED until an evaluated real
organ earns explicit promotion.
"""

from .contracts import (
    SEMANTIC_PROJECTION_SCHEMA,
    SEMANTIC_SERVICE_SCHEMA,
    SEMANTIC_SLOT_REF_SCHEMA,
    ExactEvidenceRef,
    SemanticObservation,
    SemanticProjectionRef,
    SemanticQuery,
    SemanticSlotRef,
    SemanticSpecialist,
)
from .evaluation import (
    D64_RERANK_BASELINE_GENERATION,
    D64_RERANK_EVALUATION_SCHEMA,
    D64RerankCaseResult,
    D64RerankEvaluationResult,
    StructuralLexicalD64Reranker,
    evaluate_d64_reranking,
)

__all__ = [
    "SEMANTIC_SERVICE_SCHEMA",
    "SEMANTIC_PROJECTION_SCHEMA",
    "SEMANTIC_SLOT_REF_SCHEMA",
    "ExactEvidenceRef",
    "SemanticSlotRef",
    "SemanticProjectionRef",
    "SemanticQuery",
    "SemanticObservation",
    "SemanticSpecialist",
    "D64_RERANK_EVALUATION_SCHEMA",
    "D64_RERANK_BASELINE_GENERATION",
    "D64RerankCaseResult",
    "D64RerankEvaluationResult",
    "StructuralLexicalD64Reranker",
    "evaluate_d64_reranking",
]
