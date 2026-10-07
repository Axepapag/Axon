"""The Axon native-95 memory curriculum: architecture-independent tasks and contract.

The curriculum decides *what* is asked and *what the answer is*; it never
decides how a model represents it. Episodes are plain JSON-serializable dicts
(:mod:`curriculum.schema`), generated deterministically from seeds and validated
before use, so the E0 two-state organism can run them first and other
architectures can run the identical tasks later.

Three laws hold everywhere in this package:

* Only the 95 frozen native characters are admitted; text outside them fails
  closed and is never converted, escaped or clamped (newline is native, TAB and
  backtick are not).
* WAIT is control state, never an all-EMPTY content surface: an ``"input"`` step
  always carries non-empty native text, while ``"tick"``/``"reset"`` steps carry
  none and the control triple (0 WAIT, 1 COMMIT, 2 END) lives only in
  ``expected.control``.
* The expected answer is never copied into model inputs.
"""

from __future__ import annotations

from .metrics import RESULTS_FIELDS, baselines, evaluate_episode
from .presets import PRESETS, materialize
from .schema import (
    CONTROL_COMMIT,
    CONTROL_END,
    CONTROL_VALUES,
    CONTROL_WAIT,
    CURRICULUM_VERSION,
    EPISODE_FIELDS,
    EPISODE_JSON_SCHEMA,
    EPISODE_SCHEMA_ID,
    FAMILIES,
    RESULTS_SCHEMA_ID,
    SPLIT_SCHEMA_ID,
    STEP_KINDS,
    canonical_hash,
    canonical_json,
    validate_episode,
)
from .splits import assign_split, check_manifest, curriculum_sha256, freeze_split

__all__ = [
    "CONTROL_COMMIT",
    "CONTROL_END",
    "CONTROL_VALUES",
    "CONTROL_WAIT",
    "CURRICULUM_VERSION",
    "EPISODE_FIELDS",
    "EPISODE_JSON_SCHEMA",
    "EPISODE_SCHEMA_ID",
    "FAMILIES",
    "PRESETS",
    "RESULTS_FIELDS",
    "RESULTS_SCHEMA_ID",
    "SPLIT_SCHEMA_ID",
    "STEP_KINDS",
    "assign_split",
    "baselines",
    "canonical_hash",
    "canonical_json",
    "check_manifest",
    "curriculum_sha256",
    "evaluate_episode",
    "freeze_split",
    "materialize",
    "validate_episode",
]
