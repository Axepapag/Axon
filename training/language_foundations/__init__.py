"""L0-L4 Language Foundations curriculum package.

Objective first bounded campaign of the Language Foundations Schoolhouse:
character/transport fluency (L0), spelling/punctuation (L1), vocabulary (L2),
morphology (L3), and grammar mechanics (L4), all authored deterministic
fixtures with exact or categorical targets and rule-transfer holdouts.
"""

from .characters import l0_lesson_specs
from .compiler import (
    DEFAULT_LANGUAGE_SPLIT_COUNTS,
    LanguageFoundationsCompiler,
    all_language_lesson_specs,
    publish_language_foundations_curriculum,
)
from .contracts import (
    COMPETENCY_PREREQUISITES,
    LANGUAGE_FOUNDATIONS_AUTHORED_SOURCE_ID,
    LANGUAGE_FOUNDATIONS_FAMILIES,
    LanguageLessonSpec,
    TargetVisibility,
)
from .grammar import l4_lesson_specs
from .morphology import l3_lesson_specs
from .orthography import l1_lesson_specs
from .validators import (
    LanguageTargetLeak,
    find_language_target_leaks,
    verify_language_no_target_leakage,
    verify_language_transfer_disjointness,
    verify_language_visibility_consistency,
)
from .vocabulary import l2_lesson_specs

__all__ = [
    "COMPETENCY_PREREQUISITES",
    "DEFAULT_LANGUAGE_SPLIT_COUNTS",
    "LANGUAGE_FOUNDATIONS_AUTHORED_SOURCE_ID",
    "LANGUAGE_FOUNDATIONS_FAMILIES",
    "LanguageFoundationsCompiler",
    "LanguageLessonSpec",
    "LanguageTargetLeak",
    "TargetVisibility",
    "all_language_lesson_specs",
    "find_language_target_leaks",
    "l0_lesson_specs",
    "l1_lesson_specs",
    "l2_lesson_specs",
    "l3_lesson_specs",
    "l4_lesson_specs",
    "publish_language_foundations_curriculum",
    "verify_language_no_target_leakage",
    "verify_language_transfer_disjointness",
    "verify_language_visibility_consistency",
]
