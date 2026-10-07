"""Versioned curriculum presets and their bounded, reproducible materialization.

A preset names the families, the requested size of each split, a seed base and
the difficulty knobs; ``materialize`` turns it into an exact, frozen artifact:

* ``manifest.json`` - the frozen split manifest (canonical JSON, sha256),
* ``episodes.jsonl`` - one canonical episode per line,
* ``REPORT.txt`` - a plain-language report for the operator.

Difficulty convention (the generators take one exact value per knob):

* ``{"length": [1, 2]}`` - an inclusive range; each candidate seed resolves to
  a value in the range,
* ``{"scenario": ["commit", "end"]}`` - a list of choices,
* a scalar is used as is,
* a list of complete dicts (used by ``generalization``, whose two mechanics
  take different knobs) - one option is chosen per candidate seed.

The resolution is deterministic from the candidate seed alone (a short hash of
``[seed, knob]``), so the same preset always produces byte-identical episodes,
generation can stop and resume at any attempt, and a checkpoint can record
exactly how far it got.

Splits come from content identity (``curriculum.splits``): candidates are
generated (oversampled beyond the requested size), classified by their
``split_key`` hash, and accepted until each bucket reaches its requested size.
One lesson body is never accepted twice, and a bucket that cannot be filled
stops the run with ``ValueError`` - a smaller split is never shipped silently.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Union

from .generators import generate_episode
from .schema import CURRICULUM_VERSION, FAMILIES, canonical_hash, canonical_json, validate_episode
from .splits import (
    SPLIT_NAMES,
    assign_split,
    curriculum_sha256,
    freeze_split,
)

__all__ = ["CANDIDATE_MULTIPLIER", "PRESETS", "materialize"]

#: Candidate attempts per required episode in one bucket. A content-identity
#: split sends roughly 1 in 3 candidates to a given bucket, so 12 attempts per
#: required episode is ~4x the expected yield: far beyond the brief's x3
#: oversampling while staying strictly bounded. If a bucket still cannot be
#: filled, materialize raises instead of shipping a smaller split.
CANDIDATE_MULTIPLIER = 12

#: Gap between per-(family, bucket) candidate seed streams.
_CANDIDATE_STRIDE = 10_000

_SPLIT_RULE = (
    "content-identity: sha256(canonical_json(split_key)) mod 3 -> 0 train / 1 validation / 2 test; "
    "candidate seed = seed_base + (family_index * 3 + bucket) * 10000 + attempt"
)

#: The presets. "e0-first" is the default first school for the E0 two-state
#: organism: small, inspectable, and limited to three families. "full-progression"
#: exposes all eight families with the full difficulty range.
PRESETS: dict[str, dict[str, Any]] = {
    "e0-first": {
        "name": "e0-first",
        "version": CURRICULUM_VERSION,
        "description": (
            "First school for the E0 two-state organism: exact character copy, short delayed recall, "
            "and explicit WAIT/COMMIT/END control ticks. Small enough to run and inspect by hand "
            "(40 train / 15 validation / 15 test per family)."
        ),
        "families": {
            "char_copy": {
                "train": 40,
                "validation": 15,
                "test": 15,
                "seed_base": 1000,
                "difficulty": {"length": [1, 2]},
            },
            "delayed_recall": {
                "train": 40,
                "validation": 15,
                "test": 15,
                "seed_base": 1000,
                "difficulty": {"length": [1, 2], "delay": [0, 4]},
            },
            "control": {
                "train": 40,
                "validation": 15,
                "test": 15,
                "seed_base": 1000,
                "difficulty": {"scenario": ["commit", "end"], "length": [1, 8]},
            },
        },
    },
    "full-progression": {
        "name": "full-progression",
        "version": CURRICULUM_VERSION,
        "description": (
            "All eight progressive families with the full difficulty range: delays up to 16 silent ticks, "
            "up to 6 distractors, 1-4 stored facts, and both generalization mechanics "
            "(60 train / 20 validation / 20 test per family)."
        ),
        "families": {
            "char_copy": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": {"length": [1, 4]},
            },
            "delayed_recall": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": {"length": [1, 4], "delay": [0, 16]},
            },
            "distracted_recall": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": {
                    "length": [1, 4],
                    "delay": [0, 16],
                    "distractors": [0, 6],
                    "distractor_kind": ["noise", "similar", "repeated"],
                },
            },
            "key_value": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": {"facts": [1, 4], "delay": [0, 16]},
            },
            "correction": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": {"delay": [0, 16]},
            },
            "order_binding": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": {"items": [2, 4], "delay": [0, 16]},
            },
            "control": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": {"scenario": ["wait", "commit", "end"], "length": [1, 8]},
            },
            "generalization": {
                "train": 60,
                "validation": 20,
                "test": 20,
                "seed_base": 2000,
                "difficulty": [
                    {"mechanic": "novel_key_value", "facts": [1, 4], "delay": [0, 16]},
                    {
                        "mechanic": "novel_distracted_recall",
                        "length": [1, 4],
                        "delay": [0, 16],
                        "distractors": [0, 6],
                    },
                ],
            },
        },
    },
}

if set(FAMILIES) - set(PRESETS["full-progression"]["families"]):  # pragma: no cover - preset guard
    raise RuntimeError(
        "full-progression must cover every family from curriculum.schema.FAMILIES; "
        f"missing: {sorted(set(FAMILIES) - set(PRESETS['full-progression']['families']))}"
    )


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _knob_index(seed: int, name: str, span: int) -> int:
    """Deterministic index in 0..span-1 from the candidate seed and knob name."""
    return int(canonical_hash([seed, name])[:12], 16) % span


def _resolve_knob(seed: int, name: str, value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"preset difficulty[{name!r}] must not be an empty list")
        if len(value) == 2 and all(_is_plain_int(item) for item in value):
            low, high = int(value[0]), int(value[1])
            if low > high:
                raise ValueError(f"preset difficulty[{name!r}] range is inverted: {value!r}")
            return low + _knob_index(seed, name, high - low + 1)
        return value[_knob_index(seed, name, len(value))]
    return value


def _difficulty_for(seed: int, difficulty: Any) -> dict:
    """Resolve a preset difficulty spec to the exact knobs for one candidate seed."""
    if isinstance(difficulty, list):
        if not difficulty or not all(isinstance(option, dict) for option in difficulty):
            raise ValueError("a preset difficulty list must hold complete difficulty dicts")
        difficulty = difficulty[_knob_index(seed, "difficulty", len(difficulty))]
    if not isinstance(difficulty, dict):
        raise ValueError(f"preset difficulty must be a dict or a list of dicts; got {difficulty!r}")
    return {name: _resolve_knob(seed, name, value) for name, value in sorted(difficulty.items())}


def _content_identity(episode: dict) -> str:
    """The lesson body without its per-seed identity: used to refuse duplicates."""
    body = {key: value for key, value in episode.items() if key not in ("episode_id", "seed")}
    return canonical_json(body)


def _preset_seed_base(preset: dict) -> int:
    return min(int(spec["seed_base"]) for spec in preset["families"].values())


def materialize(preset_name: str, out_dir: Union[str, Path]) -> dict:
    """Generate a preset, freeze its splits, and write the frozen artifact.

    Writes ``manifest.json``, ``episodes.jsonl`` and ``REPORT.txt`` under
    ``out_dir`` (created if needed) and returns a summary dict carrying the
    counts, the hashes, the written file paths and the manifest itself. Fails
    with ``ValueError`` if a bucket cannot be filled or a generator emits an
    episode that does not pass ``validate_episode``; it never ships a partial
    or invalid set.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset {preset_name!r}; available: {sorted(PRESETS)}")
    preset = PRESETS[preset_name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    used_episode_ids: set[str] = set()
    used_bodies: set[str] = set()

    for family_index, (family, spec) in enumerate(preset["families"].items()):
        counts[family] = {}
        for bucket_index, split in enumerate(SPLIT_NAMES):
            target = int(spec[split])
            kept = 0
            stream_offset = (family_index * len(SPLIT_NAMES) + bucket_index) * _CANDIDATE_STRIDE
            for attempt in range(CANDIDATE_MULTIPLIER * target):
                if kept >= target:
                    break
                seed = int(spec["seed_base"]) + stream_offset + attempt
                episode = generate_episode(
                    family=family, seed=seed, difficulty=_difficulty_for(seed, spec["difficulty"])
                )
                if assign_split(episode) != bucket_index:
                    continue
                episode_id = episode.get("episode_id")
                body = _content_identity(episode)
                if not isinstance(episode_id, str) or episode_id in used_episode_ids or body in used_bodies:
                    continue
                problems = validate_episode(episode)
                if problems:
                    raise ValueError(
                        f"preset {preset_name!r}: generator produced an invalid {family!r} episode "
                        f"(seed {seed}): {problems[:5]}"
                    )
                used_episode_ids.add(episode_id)
                used_bodies.add(body)
                episodes.append(episode)
                kept += 1
            if kept < target:
                raise ValueError(
                    f"preset {preset_name!r} cannot be filled: family {family!r} bucket {split!r} needed "
                    f"{target} episodes, only {kept} of {CANDIDATE_MULTIPLIER * target} candidates landed "
                    "in that bucket. Refusing to ship a smaller split"
                )
            counts[family][split] = kept

    manifest = freeze_split(episodes, rule=_SPLIT_RULE, seed=_preset_seed_base(preset))
    curriculum_digest = curriculum_sha256(episodes)

    manifest_path = out / "manifest.json"
    episodes_path = out / "episodes.jsonl"
    report_path = out / "REPORT.txt"

    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    episodes_text = "".join(canonical_json(episode) + "\n" for episode in episodes)
    episodes_path.write_text(episodes_text, encoding="utf-8")
    episodes_digest = hashlib.sha256(episodes_text.encode("utf-8")).hexdigest()

    report_lines = [
        f"Axon curriculum preset: {preset_name}",
        f"Version: {CURRICULUM_VERSION}",
        f"Rule: {_SPLIT_RULE}",
        f"Seed base: {_preset_seed_base(preset)} (candidate streams are disjoint per family and bucket)",
        "Difficulty: [min, max] means an inclusive range resolved per candidate seed; other lists are",
        "  choices; a list of dicts means one complete difficulty option per candidate seed.",
        f"Episodes: {len(episodes)} total",
        "",
        "Counts per family and split:",
    ]
    for family, split_counts in counts.items():
        report_lines.append(
            f"  {family}: " + ", ".join(f"{split} {split_counts[split]}" for split in SPLIT_NAMES)
        )
    report_lines.extend(
        [
            "",
            f"curriculum_sha256: {curriculum_digest}",
            f"manifest_sha256: {manifest['manifest_sha256']}",
            f"episodes.jsonl sha256: {episodes_digest}",
            "",
            "Files written:",
            "  manifest.json  - the frozen split manifest (canonical JSON)",
            "  episodes.jsonl - one canonical episode per line",
            "  REPORT.txt     - this plain-language report",
            "",
            "How to regenerate (same bytes, same splits):",
            "",
            f"  python -c \"from curriculum.presets import materialize; "
            f"materialize('{preset_name}', '<out_dir>')\"",
            "",
            "How to verify later: run the episodes through the model, score each prediction with",
            "curriculum.metrics.evaluate_episode, and record curriculum_sha256 with every result row",
            "so the exact bytes are provable.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "preset": preset_name,
        "version": CURRICULUM_VERSION,
        "out_dir": str(out),
        "counts": counts,
        "episode_count": len(episodes),
        "curriculum_sha256": curriculum_digest,
        "manifest_sha256": manifest["manifest_sha256"],
        "files": {
            "manifest": str(manifest_path),
            "episodes": str(episodes_path),
            "report": str(report_path),
        },
        "manifest": manifest,
    }
