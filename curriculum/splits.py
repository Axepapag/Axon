"""Split assignment and frozen split manifests for the Axon curriculum.

Splits are assigned by CONTENT IDENTITY, never by row number: an episode lands
in train, validation or test according to ``sha256(canonical_json(split_key))``
modulo the number of buckets. Reordering rows or adding new rows never moves an
episode across a boundary, and two episodes that share a split_key can never
straddle two splits.

A frozen manifest is canonical JSON plus sha256 digests, so a trainer can
record ``curriculum_sha256`` and later prove which exact bytes it saw. Nothing
here knows about neural architectures: the same frozen splits serve the E0
two-state organism and every later architecture.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .schema import (
    CURRICULUM_VERSION,
    SPLIT_SCHEMA_ID,
    canonical_hash,
    canonical_json,
)

__all__ = [
    "SPLIT_NAMES",
    "assign_split",
    "check_manifest",
    "curriculum_sha256",
    "episode_sha256",
    "freeze_split",
    "manifest_sha256",
]

#: Bucket ids are frozen: 0 = train, 1 = validation, 2 = test.
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")

_MISSING = object()


def episode_sha256(episode: dict) -> str:
    """sha256 hex digest of the canonical JSON of one episode."""
    return canonical_hash(episode)


def manifest_sha256(manifest: dict) -> str:
    """sha256 hex digest of a manifest with its own ``manifest_sha256`` removed."""
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return canonical_hash(body)


def assign_split(episode: dict, buckets: int = 3) -> int:
    """Return the bucket id for an episode: hash of its split_key, mod buckets.

    Bucket ids are stable forever: 0 = train, 1 = validation, 2 = test. The
    assignment depends only on the episode's ``split_key`` (its content
    identity), so the same facts/templates/combinations never straddle a split
    boundary. A missing split_key fails closed instead of guessing.
    """
    if isinstance(buckets, bool) or not isinstance(buckets, int):
        raise TypeError(f"buckets must be an int; got {type(buckets).__name__}")
    if not 1 <= buckets <= len(SPLIT_NAMES):
        raise ValueError(f"buckets must be in 1..{len(SPLIT_NAMES)}; got {buckets}")
    if not isinstance(episode, dict):
        raise TypeError(f"episode must be a dict; got {type(episode).__name__}")
    if "split_key" not in episode:
        raise ValueError(
            "episode has no split_key: splits are assigned by content identity and a missing key fails closed"
        )
    digest = canonical_hash(episode["split_key"])
    return int(digest, 16) % buckets


def freeze_split(episodes: Iterable[dict], rule: str, seed: int) -> dict:
    """Freeze a disjoint train/validation/test manifest for these episodes.

    Splits are assigned from each episode's split_key, so episodes that share a
    key (a template whose values differ) land in the same bucket by
    construction and can never leak across buckets. Duplicate episode ids fail
    closed; empty buckets are kept in the manifest so its shape is constant.
    """
    if not isinstance(rule, str) or not rule:
        raise ValueError(f"rule must be a non-empty str; got {rule!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int; got {type(seed).__name__}")

    buckets: dict[str, dict[str, Any]] = {
        name: {"id": position, "split": name, "episode_ids": [], "episode_sha256": {}}
        for position, name in enumerate(SPLIT_NAMES)
    }
    split_keys: dict[str, Any] = {}
    seen_episode_ids: dict[str, str] = {}

    for episode in episodes:
        if not isinstance(episode, dict):
            raise TypeError(f"episode must be a dict; got {type(episode).__name__}")
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"episode_id must be a non-empty str; got {episode_id!r}")
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate episode_id {episode_id!r} cannot be frozen")
        bucket = assign_split(episode)  # raises when split_key is missing
        name = SPLIT_NAMES[bucket]
        seen_episode_ids[episode_id] = name
        buckets[name]["episode_ids"].append(episode_id)
        buckets[name]["episode_sha256"][episode_id] = episode_sha256(episode)
        split_keys[episode_id] = episode["split_key"]

    body = {
        "schema": SPLIT_SCHEMA_ID,
        "curriculum_version": CURRICULUM_VERSION,
        "rule": rule,
        "seed": seed,
        "buckets": [buckets[name] for name in SPLIT_NAMES],
        "split_keys": split_keys,
    }
    manifest = dict(body)
    manifest["manifest_sha256"] = canonical_hash(body)
    return manifest


def check_manifest(manifest: dict, episodes: Optional[Iterable[dict]] = None) -> list[str]:
    """Audit a frozen split manifest; an empty list means clean.

    Detects duplicate episode ids across buckets, split-key leakage between
    buckets, bucket ids out of range, ids that no bucket or ``split_keys``
    entry covers, a tampered ``manifest_sha256``, and (when ``episodes`` is
    provided) sha256 mismatches against the episodes the manifest was frozen
    from plus any episode sitting in a bucket its split_key does not assign to.
    """
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return [f"manifest must be a dict; got {type(manifest).__name__}"]

    for key in ("schema", "curriculum_version", "rule", "seed", "buckets", "split_keys", "manifest_sha256"):
        if key not in manifest:
            errors.append(f"manifest is missing {key!r}")
    if "schema" in manifest and manifest["schema"] != SPLIT_SCHEMA_ID:
        errors.append(f"manifest schema is {manifest['schema']!r}; expected {SPLIT_SCHEMA_ID!r}")
    if "manifest_sha256" in manifest:
        recomputed = manifest_sha256(manifest)
        if manifest["manifest_sha256"] != recomputed:
            errors.append(
                f"manifest_sha256 mismatch: manifest says {manifest['manifest_sha256']!r}, "
                f"recomputed over canonical JSON it is {recomputed!r}"
            )

    split_keys = manifest.get("split_keys")
    if not isinstance(split_keys, dict):
        errors.append("manifest split_keys must be a mapping episode_id -> split_key")
        split_keys = {}

    buckets = manifest.get("buckets")
    if not isinstance(buckets, list):
        errors.append("manifest buckets must be a list")
        buckets = []

    seen_episode_ids: dict[str, str] = {}
    seen_split_keys: dict[str, str] = {}
    seen_bucket_ids: set[int] = set()
    declared_hashes: dict[str, Any] = {}
    listed_ids: set[str] = set()

    for position, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            errors.append(f"bucket at position {position} must be a dict")
            continue

        bucket_id = bucket.get("id")
        split = bucket.get("split")
        if isinstance(bucket_id, bool) or not isinstance(bucket_id, int):
            errors.append(f"bucket at position {position} has no integer id; got {bucket_id!r}")
        else:
            if not 0 <= bucket_id < len(SPLIT_NAMES):
                errors.append(f"bucket id {bucket_id} is out of range 0..{len(SPLIT_NAMES) - 1}")
            elif split in SPLIT_NAMES and split != SPLIT_NAMES[bucket_id]:
                errors.append(
                    f"bucket id {bucket_id} is named {split!r}; id {bucket_id} means {SPLIT_NAMES[bucket_id]!r}"
                )
            if bucket_id in seen_bucket_ids:
                errors.append(f"bucket id {bucket_id} is used by more than one bucket")
            seen_bucket_ids.add(bucket_id)
        if split not in SPLIT_NAMES:
            errors.append(f"bucket at position {position} has split {split!r}; expected one of {list(SPLIT_NAMES)}")
            continue

        episode_ids = bucket.get("episode_ids")
        if not isinstance(episode_ids, list):
            errors.append(f"bucket {split!r} episode_ids must be a list")
            episode_ids = []
        listed = {episode_id for episode_id in episode_ids if isinstance(episode_id, str)}
        for episode_id in episode_ids:
            if not isinstance(episode_id, str):
                errors.append(f"bucket {split!r} lists a non-string episode_id: {episode_id!r}")
                continue
            owner = seen_episode_ids.get(episode_id)
            if owner is not None:
                if owner == split:
                    errors.append(f"episode_id {episode_id!r} appears twice in bucket {split!r}")
                else:
                    errors.append(f"duplicate episode_id {episode_id!r} across buckets {owner!r} and {split!r}")
            else:
                seen_episode_ids[episode_id] = split
            listed_ids.add(episode_id)

            key = split_keys.get(episode_id, _MISSING)
            if key is _MISSING:
                errors.append(f"episode {episode_id!r} has no split_key in the manifest")
                continue
            try:
                key_text = canonical_json(key)
            except (TypeError, ValueError) as exc:
                errors.append(f"episode {episode_id!r} has a split_key that cannot be serialized: {exc}")
                continue
            key_owner = seen_split_keys.get(key_text)
            if key_owner is not None and key_owner != split:
                errors.append(f"leakage: split_key {key_text} is shared by buckets {key_owner!r} and {split!r}")
            elif key_owner is None:
                seen_split_keys[key_text] = split

        hashes = bucket.get("episode_sha256")
        if not isinstance(hashes, dict):
            errors.append(f"bucket {split!r} episode_sha256 must be a mapping id -> sha256")
        else:
            for episode_id in listed:
                if episode_id not in hashes:
                    errors.append(f"bucket {split!r} episode_sha256 is missing {episode_id!r}")
            for episode_id in sorted(set(hashes) - listed):
                errors.append(f"bucket {split!r} episode_sha256 lists {episode_id!r}, absent from episode_ids")
            for episode_id in listed:
                if episode_id in hashes:
                    declared_hashes[episode_id] = hashes[episode_id]

    for episode_id in sorted(set(split_keys) - listed_ids):
        errors.append(f"split_keys lists {episode_id!r}, which no bucket contains")

    if episodes is not None:
        provided: dict[str, dict] = {}
        for episode in episodes:
            if not isinstance(episode, dict):
                errors.append(f"provided episode must be a dict; got {type(episode).__name__}")
                continue
            episode_id = episode.get("episode_id")
            if not isinstance(episode_id, str):
                errors.append("provided episode has no string episode_id")
                continue
            if episode_id in provided:
                errors.append(f"provided episodes repeat episode_id {episode_id!r}")
                continue
            provided[episode_id] = episode

        for episode_id in sorted(provided):
            episode = provided[episode_id]
            owner = seen_episode_ids.get(episode_id)
            if owner is None:
                errors.append(f"provided episode {episode_id!r} is not listed in the manifest")
                continue
            try:
                digest = episode_sha256(episode)
            except (TypeError, ValueError) as exc:
                errors.append(f"provided episode {episode_id!r} cannot be hashed: {exc}")
                digest = None
            declared = declared_hashes.get(episode_id)
            if digest is not None and declared is not None and declared != digest:
                errors.append(
                    f"sha mismatch for episode {episode_id!r}: manifest declares {declared}, "
                    f"the episode hashes to {digest}"
                )
            try:
                bucket = assign_split(episode)
            except (TypeError, ValueError) as exc:
                errors.append(f"provided episode {episode_id!r} cannot be split-assigned: {exc}")
            else:
                if SPLIT_NAMES[bucket] != owner:
                    errors.append(
                        f"episode {episode_id!r} sits in bucket {owner!r} but its split_key assigns to "
                        f"{SPLIT_NAMES[bucket]!r} (bucket id {bucket})"
                    )
        for episode_id in sorted(set(seen_episode_ids) - set(provided)):
            errors.append(f"manifest lists episode {episode_id!r} but no provided episode matches it")

    return errors


def curriculum_sha256(episodes: Iterable[dict]) -> str:
    """The version hash trainers record: sha256 over the sorted per-episode hashes."""
    digests = sorted(episode_sha256(episode) for episode in episodes)
    return canonical_hash(digests)
