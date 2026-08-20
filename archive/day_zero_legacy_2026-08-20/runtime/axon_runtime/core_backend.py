"""Shared exact-v4 inference models with isolated logical-core state."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np
import torch

from runtime.field import (
    FieldViewCursor,
    LogicalRegion,
    PAGED_CONTEXT_REGIONS,
    PROPOSAL_END,
    PROPOSAL_START,
    SharedFieldSnapshot,
    RegionVisibility,
    canonical_json_bytes,
    canonical_sha256,
    compile_next_read_page,
)
from runtime.multi_tick_refiner import RegionProposal
from substrate import ALPHABET_SET, get_letter_bank

from .checkpoint import (
    ExactV4Checkpoint,
    inspect_exact_v4_checkpoint,
)
from .soul_store import (
    DecodedPrivateRuntimeState,
    PrivateRuntimeStateStore,
    SoulBlob,
    SoulBlobStore,
    SoulStoreError,
    StoredSoulBlob,
    StoredPrivateRuntimeState,
    decode_soul_blob,
    encode_private_runtime_state,
    encode_soul_blob,
)


_CANDIDATE_SCHEMA = "axon-runtime-core-candidate-v1"
_CURSOR_ANCHOR_SCHEMA = "axon-runtime-cursor-anchors-v1"
_ANCHORED_REGIONS = (
    *PAGED_CONTEXT_REGIONS,
    LogicalRegion.USER_INPUT,
)


class CoreBackendError(RuntimeError):
    """Base class for exact-v4 runtime backend failures."""


class ModelRegistrationError(CoreBackendError):
    """A model ID was reused for different immutable weights."""


class CandidateValidationError(CoreBackendError):
    """A staged candidate is stale, modified, or not the accepted action."""


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoreBackendError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoreBackendError(f"{label} must be a non-negative integer")
    return value


def _json_manifest(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise CoreBackendError(f"{label} must be a string-keyed mapping")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CoreBackendError(f"{label} must be finite JSON-safe data") from exc
    if not isinstance(decoded, dict):
        raise CoreBackendError(f"{label} must encode an object")
    return decoded


def _cursor_copy(cursor: FieldViewCursor) -> FieldViewCursor:
    return FieldViewCursor(
        page_index=cursor.page_index,
        context_offsets=tuple(cursor.context_offsets),
        user_offset=cursor.user_offset,
    )


def _cursor_hash(cursor: FieldViewCursor) -> str:
    return canonical_sha256(cursor.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class CursorRegionAnchor:
    region: str
    character_length: int
    text_sha256: str

    def __post_init__(self) -> None:
        try:
            logical = LogicalRegion(self.region)
        except (TypeError, ValueError) as exc:
            raise CoreBackendError("cursor anchor region is invalid") from exc
        if logical not in _ANCHORED_REGIONS:
            raise CoreBackendError("cursor anchor region is not pageable")
        if (
            isinstance(self.character_length, bool)
            or not isinstance(self.character_length, int)
            or self.character_length < 0
        ):
            raise CoreBackendError(
                "cursor anchor character_length must be non-negative"
            )
        if (
            not isinstance(self.text_sha256, str)
            or len(self.text_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.text_sha256
            )
        ):
            raise CoreBackendError(
                "cursor anchor text_sha256 must be lowercase SHA-256"
            )
        object.__setattr__(self, "region", logical.value)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "character_length": self.character_length,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class CursorAnchorManifest:
    anchors: tuple[CursorRegionAnchor, ...] = ()
    anchor_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        anchors = tuple(self.anchors)
        if not all(isinstance(item, CursorRegionAnchor) for item in anchors):
            raise CoreBackendError(
                "cursor anchor manifest must contain CursorRegionAnchor values"
            )
        regions = tuple(item.region for item in anchors)
        expected = tuple(region.value for region in _ANCHORED_REGIONS)
        if anchors and regions != expected:
            raise CoreBackendError(
                "cursor anchor manifest must exactly cover pageable regions"
            )
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(
            self,
            "anchor_sha256",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": _CURSOR_ANCHOR_SCHEMA,
            "anchors": [
                anchor.to_canonical_dict() for anchor in self.anchors
            ],
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "CursorAnchorManifest":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "anchors",
        }:
            raise CoreBackendError("cursor anchor manifest keys mismatch")
        if value["schema"] != _CURSOR_ANCHOR_SCHEMA:
            raise CoreBackendError("unknown cursor anchor manifest schema")
        raw = value["anchors"]
        if not isinstance(raw, list):
            raise CoreBackendError("cursor anchor entries must be a list")
        anchors: list[CursorRegionAnchor] = []
        for item in raw:
            if not isinstance(item, Mapping) or set(item) != {
                "region",
                "character_length",
                "text_sha256",
            }:
                raise CoreBackendError("cursor region anchor keys mismatch")
            anchors.append(
                CursorRegionAnchor(
                    region=item["region"],
                    character_length=item["character_length"],
                    text_sha256=item["text_sha256"],
                )
            )
        return cls(tuple(anchors))


@dataclass(slots=True)
class SharedAxonModel:
    """One immutable loaded model shared by every clone of a model ID."""

    model_id: str
    checkpoint: ExactV4Checkpoint
    inference_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    @property
    def core(self):
        return self.checkpoint.core

    @property
    def soul_compressor(self):
        return self.checkpoint.soul_compressor

    @property
    def soul_tier_status(self) -> str:
        return self.checkpoint.soul_tier_status


@dataclass(frozen=True, slots=True)
class CandidateStateManifest:
    core_id: str
    soul_id: str
    model_id: str
    checkpoint_sha256: str
    base_core_generation: int
    candidate_core_generation: int
    base_soul_generation: int
    candidate_soul_generation: int
    prior_private_field_id: str
    canonical_base_field_id: str
    observed_field_id: str
    observed_tick_id: int
    observed_ancestor_field_ids: tuple[str, ...]
    tick_index: int
    target_region: str
    proposed_text: str
    view_hash: str
    cursor_before_sha256: str
    rebased_cursor_sha256: str
    cursor_after_sha256: str
    cursor_anchor_before_sha256: str
    cursor_anchor_after_sha256: str
    base_soul_blob_sha256: str
    candidate_soul_blob_sha256: str
    adapter_manifest_sha256: str
    rng_manifest_sha256: str
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "core_id",
            "soul_id",
            "model_id",
            "prior_private_field_id",
            "canonical_base_field_id",
            "observed_field_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self,
                name,
            ):
                raise CandidateValidationError(f"{name} must be non-empty")
        if not isinstance(self.proposed_text, str):
            raise CandidateValidationError("proposed_text must be a string")
        for name in (
            "base_core_generation",
            "candidate_core_generation",
            "base_soul_generation",
            "candidate_soul_generation",
            "observed_tick_id",
            "tick_index",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CandidateValidationError(
                    f"{name} must be a non-negative integer"
                )
        if self.candidate_core_generation != self.base_core_generation + 1:
            raise CandidateValidationError(
                "candidate core generation must be the exact successor"
            )
        if self.candidate_soul_generation != self.base_soul_generation + 1:
            raise CandidateValidationError(
                "candidate soul generation must be the exact successor"
            )
        ancestors = tuple(self.observed_ancestor_field_ids)
        if any(not isinstance(item, str) or not item for item in ancestors):
            raise CandidateValidationError(
                "observed_ancestor_field_ids must contain non-empty strings"
            )
        if len(ancestors) != len(set(ancestors)):
            raise CandidateValidationError(
                "observed_ancestor_field_ids cannot contain duplicates"
            )
        if self.observed_field_id != self.canonical_base_field_id:
            if not ancestors or ancestors[0] != self.canonical_base_field_id:
                raise CandidateValidationError(
                    "observed lineage must begin at canonical base"
                )
        object.__setattr__(self, "observed_ancestor_field_ids", ancestors)
        hashes = (
            "checkpoint_sha256",
            "view_hash",
            "cursor_before_sha256",
            "rebased_cursor_sha256",
            "cursor_after_sha256",
            "cursor_anchor_before_sha256",
            "cursor_anchor_after_sha256",
            "base_soul_blob_sha256",
            "candidate_soul_blob_sha256",
            "adapter_manifest_sha256",
            "rng_manifest_sha256",
        )
        for name in hashes:
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise CandidateValidationError(
                    f"{name} must be a lowercase SHA-256"
                )
        try:
            LogicalRegion(self.target_region)
        except (TypeError, ValueError) as exc:
            raise CandidateValidationError("target_region is invalid") from exc
        object.__setattr__(
            self,
            "manifest_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": _CANDIDATE_SCHEMA,
            "core_id": self.core_id,
            "soul_id": self.soul_id,
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "base_core_generation": self.base_core_generation,
            "candidate_core_generation": self.candidate_core_generation,
            "base_soul_generation": self.base_soul_generation,
            "candidate_soul_generation": self.candidate_soul_generation,
            "prior_private_field_id": self.prior_private_field_id,
            "canonical_base_field_id": self.canonical_base_field_id,
            "observed_field_id": self.observed_field_id,
            "observed_tick_id": self.observed_tick_id,
            "observed_ancestor_field_ids": list(
                self.observed_ancestor_field_ids
            ),
            "tick_index": self.tick_index,
            "target_region": self.target_region,
            "proposed_text": self.proposed_text,
            "view_hash": self.view_hash,
            "cursor_before_sha256": self.cursor_before_sha256,
            "rebased_cursor_sha256": self.rebased_cursor_sha256,
            "cursor_after_sha256": self.cursor_after_sha256,
            "cursor_anchor_before_sha256": (
                self.cursor_anchor_before_sha256
            ),
            "cursor_anchor_after_sha256": self.cursor_anchor_after_sha256,
            "base_soul_blob_sha256": self.base_soul_blob_sha256,
            "candidate_soul_blob_sha256": self.candidate_soul_blob_sha256,
            "adapter_manifest_sha256": self.adapter_manifest_sha256,
            "rng_manifest_sha256": self.rng_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedCoreAction:
    """Pure proposal plus an uninstalled, unpersisted private-state candidate."""

    proposal: RegionProposal
    candidate_soul: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_cursor: FieldViewCursor
    candidate_cursor_anchors: CursorAnchorManifest
    candidate_blob: SoulBlob
    manifest: CandidateStateManifest


@dataclass(frozen=True, slots=True)
class ValidatedCandidateInstall:
    """A pure validation receipt; it performs no persistence or mutation."""

    action: PreparedCoreAction
    expected_core_generation: int
    expected_soul_generation: int
    canonical_base_field_id: str
    expected_observed_field_id: str
    expected_observed_tick_id: int
    final_accepted_text: str
    validation_hash: str


@dataclass(frozen=True, slots=True)
class PreparedPersistedRuntimeState:
    """Pre-commit content addresses produced without installing live state."""

    receipt: ValidatedCandidateInstall
    intended_committed_field_id: str
    stored_soul: StoredSoulBlob
    stored_private_state: StoredPrivateRuntimeState

    @property
    def soul_state_sha256(self) -> str:
        return self.stored_soul.sha256

    @property
    def cursor_state_sha256(self) -> str:
        return self.stored_private_state.sha256


class LogicalCore:
    """One identity over shared weights and private soul/runtime metadata."""

    def __init__(
        self,
        *,
        core_id: str,
        soul_id: str,
        shared_model: SharedAxonModel,
        committed_field_id: str,
        core_generation: int,
        soul_generation: int,
        soul: torch.Tensor,
        soul_mask: torch.Tensor,
        cursor: FieldViewCursor,
        cursor_anchors: CursorAnchorManifest,
        adapter_manifest: Mapping[str, Any],
        rng_manifest: Mapping[str, Any],
        last_candidate_manifest_id: str | None = None,
    ) -> None:
        self.core_id = _nonempty(core_id, "core_id")
        self.soul_id = _nonempty(soul_id, "soul_id")
        self.shared_model = shared_model
        self.committed_field_id = _nonempty(
            committed_field_id,
            "committed_field_id",
        )
        self.core_generation = _nonnegative_int(
            core_generation,
            "core_generation",
        )
        self.soul_generation = _nonnegative_int(
            soul_generation,
            "soul_generation",
        )
        seed_blob = encode_soul_blob(
            soul,
            soul_mask,
            soul_id=self.soul_id,
            generation=self.soul_generation,
            base_field_id=self.committed_field_id,
        )
        decoded = decode_soul_blob(seed_blob.data, seed_blob.sha256)
        cfg = shared_model.checkpoint.cfg
        if tuple(decoded.soul.shape) != (
            cfg.total_soul_rows(),
            cfg.d_model,
        ):
            raise CoreBackendError("logical soul shape does not match model")
        self.soul = decoded.soul.clone()
        self.soul_mask = decoded.mask.clone()
        self.cursor = _cursor_copy(cursor)
        if not isinstance(cursor_anchors, CursorAnchorManifest):
            raise TypeError("cursor_anchors must be CursorAnchorManifest")
        if (
            not cursor_anchors.anchors
            and (
                self.cursor.page_index
                or self.cursor.context_offsets
                or self.cursor.user_offset
            )
        ):
            raise CoreBackendError(
                "non-zero cursor cannot be restored without source anchors"
            )
        self.cursor_anchors = cursor_anchors
        self.adapter_manifest = _json_manifest(
            adapter_manifest,
            "adapter_manifest",
        )
        self.rng_manifest = _json_manifest(rng_manifest, "rng_manifest")
        if last_candidate_manifest_id is not None and (
            not isinstance(last_candidate_manifest_id, str)
            or len(last_candidate_manifest_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in last_candidate_manifest_id
            )
        ):
            raise CoreBackendError(
                "last_candidate_manifest_id must be a lowercase SHA-256"
            )
        self.last_candidate_manifest_id = last_candidate_manifest_id
        self._state_lock = threading.RLock()

    @property
    def model_id(self) -> str:
        return self.shared_model.model_id

    @property
    def core(self):
        return self.shared_model.core

    @property
    def soul_compressor(self):
        return self.shared_model.soul_compressor

    @property
    def soul_tier_status(self) -> str:
        return self.shared_model.soul_tier_status

    def prepare_action(
        self,
        snapshot: SharedFieldSnapshot,
        *,
        canonical_base_field_id: str,
        expected_observed_field_id: str,
        expected_observed_tick_id: int,
        observed_ancestor_field_ids: tuple[str, ...] | None = None,
        target_region: LogicalRegion | str = LogicalRegion.RESPONSE_DRAFT,
        tick_index: int | None = None,
        max_output_chars: int = PROPOSAL_END - PROPOSAL_START,
    ) -> PreparedCoreAction:
        return prepare_action(
            self,
            snapshot,
            canonical_base_field_id=canonical_base_field_id,
            expected_observed_field_id=expected_observed_field_id,
            expected_observed_tick_id=expected_observed_tick_id,
            observed_ancestor_field_ids=observed_ancestor_field_ids,
            target_region=target_region,
            tick_index=tick_index,
            max_output_chars=max_output_chars,
        )

    def validate_candidate(
        self,
        action: PreparedCoreAction,
        *,
        expected_core_generation: int,
        expected_soul_generation: int,
        canonical_base_field_id: str,
        expected_observed_field_id: str,
        expected_observed_tick_id: int,
        final_accepted_text: str,
    ) -> ValidatedCandidateInstall:
        return validate_candidate_install(
            self,
            action,
            expected_core_generation=expected_core_generation,
            expected_soul_generation=expected_soul_generation,
            canonical_base_field_id=canonical_base_field_id,
            expected_observed_field_id=expected_observed_field_id,
            expected_observed_tick_id=expected_observed_tick_id,
            final_accepted_text=final_accepted_text,
        )

    def install_candidate(
        self,
        action: PreparedCoreAction,
        *,
        expected_core_generation: int,
        expected_soul_generation: int,
        canonical_base_field_id: str,
        expected_observed_field_id: str,
        expected_observed_tick_id: int,
        final_accepted_text: str,
        committed_field_id: str,
    ) -> ValidatedCandidateInstall:
        receipt = self.validate_candidate(
            action,
            expected_core_generation=expected_core_generation,
            expected_soul_generation=expected_soul_generation,
            canonical_base_field_id=canonical_base_field_id,
            expected_observed_field_id=expected_observed_field_id,
            expected_observed_tick_id=expected_observed_tick_id,
            final_accepted_text=final_accepted_text,
        )
        install_validated_candidate(
            self,
            receipt,
            committed_field_id=committed_field_id,
        )
        return receipt


class CoreBackend:
    """Registry that loads each immutable model ID at most once."""

    def __init__(self) -> None:
        self._models: dict[str, SharedAxonModel] = {}
        self._cores: dict[str, LogicalCore] = {}
        self._soul_ids: set[str] = set()
        self._registry_lock = threading.RLock()

    def register_loaded_model(
        self,
        model_id: str,
        checkpoint: ExactV4Checkpoint,
    ) -> SharedAxonModel:
        identity = _nonempty(model_id, "model_id")
        if not isinstance(checkpoint, ExactV4Checkpoint):
            raise TypeError("checkpoint must be ExactV4Checkpoint")
        with self._registry_lock:
            existing = self._models.get(identity)
            if existing is not None:
                if (
                    existing.checkpoint.checkpoint_sha256
                    != checkpoint.checkpoint_sha256
                ):
                    raise ModelRegistrationError(
                        f"model_id {identity!r} is pinned to another checkpoint"
                    )
                return existing
            shared = SharedAxonModel(identity, checkpoint)
            self._models[identity] = shared
            return shared

    def load_model(
        self,
        *,
        model_id: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        device: str | torch.device = "cpu",
    ) -> SharedAxonModel:
        identity = _nonempty(model_id, "model_id")
        with self._registry_lock:
            existing = self._models.get(identity)
            if existing is not None:
                if (
                    existing.checkpoint.checkpoint_sha256
                    != checkpoint_sha256.lower()
                ):
                    raise ModelRegistrationError(
                        f"model_id {identity!r} is pinned to another checkpoint"
                    )
                return existing
            checkpoint = inspect_exact_v4_checkpoint(
                checkpoint_path,
                checkpoint_sha256,
                device=device,
            )
            return self.register_loaded_model(identity, checkpoint)

    def create_logical_core(
        self,
        *,
        model_id: str,
        core_id: str,
        soul_id: str,
        committed_field_id: str,
        core_generation: int = 0,
        soul_generation: int = 0,
        soul: torch.Tensor | None = None,
        soul_mask: torch.Tensor | None = None,
        cursor: FieldViewCursor | None = None,
        cursor_anchor_manifest: (
            CursorAnchorManifest | Mapping[str, Any] | None
        ) = None,
        adapter_manifest: Mapping[str, Any] | None = None,
        rng_manifest: Mapping[str, Any] | None = None,
        last_candidate_manifest_id: str | None = None,
    ) -> LogicalCore:
        with self._registry_lock:
            try:
                shared = self._models[model_id]
            except KeyError as exc:
                raise ModelRegistrationError(
                    f"model_id {model_id!r} is not loaded"
                ) from exc
            if core_id in self._cores:
                raise CoreBackendError(f"duplicate core_id {core_id!r}")
            if soul_id in self._soul_ids:
                raise CoreBackendError(f"duplicate soul_id {soul_id!r}")
            if (soul is None) != (soul_mask is None):
                raise CoreBackendError(
                    "soul and soul_mask must be supplied together"
                )
            soul_value = (
                shared.checkpoint.initial_soul
                if soul is None
                else soul
            )
            mask_value = (
                shared.checkpoint.initial_soul_mask
                if soul_mask is None
                else soul_mask
            )
            cursor_value = FieldViewCursor() if cursor is None else cursor
            if not isinstance(cursor_value, FieldViewCursor):
                raise TypeError("cursor must be FieldViewCursor")
            if cursor_anchor_manifest is None:
                cursor_anchors = CursorAnchorManifest()
            elif isinstance(
                cursor_anchor_manifest,
                CursorAnchorManifest,
            ):
                cursor_anchors = cursor_anchor_manifest
            else:
                cursor_anchors = CursorAnchorManifest.from_mapping(
                    cursor_anchor_manifest
                )
            adapters = (
                {
                    "schema": "axon-runtime-adapter-manifest-v1",
                    "adapters": [],
                }
                if adapter_manifest is None
                else adapter_manifest
            )
            rng = (
                {
                    "schema": "axon-runtime-rng-manifest-v1",
                    "stream_id": core_id,
                    "generation": 0,
                }
                if rng_manifest is None
                else rng_manifest
            )
            logical = LogicalCore(
                core_id=core_id,
                soul_id=soul_id,
                shared_model=shared,
                committed_field_id=committed_field_id,
                core_generation=core_generation,
                soul_generation=soul_generation,
                soul=soul_value,
                soul_mask=mask_value,
                cursor=cursor_value,
                cursor_anchors=cursor_anchors,
                adapter_manifest=adapters,
                rng_manifest=rng,
                last_candidate_manifest_id=last_candidate_manifest_id,
            )
            self._cores[logical.core_id] = logical
            self._soul_ids.add(logical.soul_id)
            return logical

    clone_core = create_logical_core

    def logical_core(self, core_id: str) -> LogicalCore:
        with self._registry_lock:
            try:
                return self._cores[core_id]
            except KeyError as exc:
                raise CoreBackendError(f"unknown core_id {core_id!r}") from exc

    def shared_model(self, model_id: str) -> SharedAxonModel:
        """Return one already-loaded immutable model without duplicating it."""

        with self._registry_lock:
            try:
                return self._models[model_id]
            except KeyError as exc:
                raise ModelRegistrationError(
                    f"model_id {model_id!r} is not loaded"
                ) from exc

    def prepare_action(
        self,
        core_id: str,
        snapshot: SharedFieldSnapshot,
        **kwargs: Any,
    ) -> PreparedCoreAction:
        return self.logical_core(core_id).prepare_action(snapshot, **kwargs)

    def restore_logical_core(
        self,
        *,
        cursor_state_sha256: str,
        state_store: PrivateRuntimeStateStore,
        soul_store: SoulBlobStore,
    ) -> LogicalCore:
        """Reconstruct one committed logical core after a process crash."""

        if not isinstance(state_store, PrivateRuntimeStateStore):
            raise TypeError("state_store must be PrivateRuntimeStateStore")
        if not isinstance(soul_store, SoulBlobStore):
            raise TypeError("soul_store must be SoulBlobStore")
        state = state_store.load(cursor_state_sha256)
        return restore_logical_core_from_state(self, state, soul_store)

    def restore_or_replace_logical_core(
        self,
        *,
        cursor_state_sha256: str,
        state_store: PrivateRuntimeStateStore,
        soul_store: SoulBlobStore,
    ) -> LogicalCore:
        """Install journal-referenced private state, replacing the same identity.

        This is the process-start/reconciliation path.  It is intentionally
        separate from candidate installation: the referenced private-state
        blob must already be durable and named by the canonical runtime head.
        """

        if not isinstance(state_store, PrivateRuntimeStateStore):
            raise TypeError("state_store must be PrivateRuntimeStateStore")
        if not isinstance(soul_store, SoulBlobStore):
            raise TypeError("soul_store must be SoulBlobStore")
        state = state_store.load(cursor_state_sha256)
        return restore_or_replace_logical_core_from_state(
            self,
            state,
            soul_store,
        )


def _device_dtype(logical: LogicalCore) -> tuple[torch.device, torch.dtype]:
    parameter = next(logical.core.parameters())
    return parameter.device, parameter.dtype


def _validate_runtime_soul(
    logical: LogicalCore,
    soul: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    cfg = logical.shared_model.checkpoint.cfg
    if tuple(soul.shape) != (cfg.total_soul_rows(), cfg.d_model):
        raise CoreBackendError("soul shape does not match exact-v4 checkpoint")
    if tuple(mask.shape) != (cfg.total_soul_rows(),):
        raise CoreBackendError("soul mask shape does not match checkpoint")
    if soul.dtype is not torch.float32 or mask.dtype is not torch.bool:
        raise CoreBackendError("soul/mask dtypes must be float32/bool")
    if not bool(torch.isfinite(soul).all()):
        raise CoreBackendError("soul contains non-finite values")


def _authenticated_observed_lineage(
    snapshot: SharedFieldSnapshot,
    canonical_base_field_id: str,
    supplied: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Bind an observed overlay to the caller-authenticated canonical head."""

    if snapshot.field_id == canonical_base_field_id:
        if supplied not in (None, ()):
            raise CandidateValidationError(
                "canonical observation cannot carry overlay ancestry"
            )
        return ()
    if snapshot.parent_field_id is None:
        raise CandidateValidationError(
            "non-canonical observed snapshot has no authenticated parent"
        )
    ancestors = (
        (canonical_base_field_id,)
        if supplied is None
        else tuple(supplied)
    )
    if (
        not ancestors
        or ancestors[0] != canonical_base_field_id
        or ancestors[-1] != snapshot.parent_field_id
        or snapshot.field_id in ancestors
        or len(ancestors) != len(set(ancestors))
        or any(not isinstance(item, str) or not item for item in ancestors)
    ):
        raise CandidateValidationError(
            "observed snapshot is not an authenticated descendant of "
            "canonical base"
        )
    return ancestors


def _supported_region_text(
    snapshot: SharedFieldSnapshot,
    region: LogicalRegion,
) -> str:
    state = snapshot.region(region)
    if state.visibility is not RegionVisibility.ATTENDED:
        return ""
    return "".join(
        character
        for span in state.spans
        for character in span.text
        if character in ALPHABET_SET
    )


def _anchor_manifest_for_snapshot(
    snapshot: SharedFieldSnapshot,
) -> CursorAnchorManifest:
    anchors = []
    for region in _ANCHORED_REGIONS:
        text = _supported_region_text(snapshot, region)
        anchors.append(
            CursorRegionAnchor(
                region=region.value,
                character_length=len(text),
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return CursorAnchorManifest(tuple(anchors))


def _rebase_cursor(
    snapshot: SharedFieldSnapshot,
    cursor: FieldViewCursor,
    prior: CursorAnchorManifest,
) -> tuple[FieldViewCursor, CursorAnchorManifest]:
    """Retain offsets only when their prior source is an exact prefix."""

    current = _anchor_manifest_for_snapshot(snapshot)
    if not prior.anchors:
        if cursor.page_index or cursor.context_offsets or cursor.user_offset:
            raise CandidateValidationError(
                "non-zero field cursor has no persisted source anchors"
            )
        return _cursor_copy(cursor), current

    prior_by_region = {anchor.region: anchor for anchor in prior.anchors}
    offsets = cursor.context_mapping()
    rebased_offsets: list[tuple[str, int]] = []
    user_offset = cursor.user_offset
    reset_any = False
    for current_anchor in current.anchors:
        region = LogicalRegion(current_anchor.region)
        old = prior_by_region[region.value]
        text = _supported_region_text(snapshot, region)
        anchored_prefix = text[: old.character_length]
        prefix_matches = (
            len(text) >= old.character_length
            and hashlib.sha256(
                anchored_prefix.encode("utf-8")
            ).hexdigest()
            == old.text_sha256
        )
        if region is LogicalRegion.USER_INPUT:
            if user_offset > old.character_length:
                raise CandidateValidationError(
                    "user cursor exceeds its anchored source length"
                )
            if not prefix_matches:
                user_offset = 0
                reset_any = True
            continue
        offset = offsets.get(region, 0)
        if offset > old.character_length:
            raise CandidateValidationError(
                f"cursor for {region.value} exceeds anchored source length"
            )
        if not prefix_matches:
            offset = 0
            reset_any = True
        if offset:
            rebased_offsets.append((region.value, offset))
    return (
        FieldViewCursor(
            page_index=0 if reset_any else cursor.page_index,
            context_offsets=tuple(rebased_offsets),
            user_offset=user_offset,
        ),
        current,
    )


def prepare_action(
    logical: LogicalCore,
    snapshot: SharedFieldSnapshot,
    *,
    canonical_base_field_id: str,
    expected_observed_field_id: str,
    expected_observed_tick_id: int,
    observed_ancestor_field_ids: tuple[str, ...] | None = None,
    target_region: LogicalRegion | str = LogicalRegion.RESPONSE_DRAFT,
    tick_index: int | None = None,
    max_output_chars: int = PROPOSAL_END - PROPOSAL_START,
) -> PreparedCoreAction:
    """Compile and run one exact 384x16 view without changing live state."""

    if not isinstance(logical, LogicalCore):
        raise TypeError("logical must be LogicalCore")
    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("snapshot must be SharedFieldSnapshot")
    canonical_base = _nonempty(
        canonical_base_field_id,
        "canonical_base_field_id",
    )
    observed_field = _nonempty(
        expected_observed_field_id,
        "expected_observed_field_id",
    )
    observed_tick = _nonnegative_int(
        expected_observed_tick_id,
        "expected_observed_tick_id",
    )
    if snapshot.field_id != observed_field or snapshot.tick_id != observed_tick:
        raise CandidateValidationError(
            "observed snapshot does not match caller-authenticated field/tick"
        )
    observed_ancestors = _authenticated_observed_lineage(
        snapshot,
        canonical_base,
        observed_ancestor_field_ids,
    )
    try:
        region = (
            target_region
            if isinstance(target_region, LogicalRegion)
            else LogicalRegion(target_region)
        )
    except (TypeError, ValueError) as exc:
        raise CoreBackendError("target_region is invalid") from exc
    if region not in {
        LogicalRegion.SCRATCH,
        LogicalRegion.RESPONSE_DRAFT,
    }:
        raise CoreBackendError("target_region is not core-writable")
    tick = snapshot.tick_id if tick_index is None else tick_index
    tick = _nonnegative_int(tick, "tick_index")
    if (
        isinstance(max_output_chars, bool)
        or not isinstance(max_output_chars, int)
        or not 1 <= max_output_chars <= PROPOSAL_END - PROPOSAL_START
    ):
        raise CoreBackendError("max_output_chars must be in [1, 64]")

    with logical._state_lock:
        _validate_runtime_soul(logical, logical.soul, logical.soul_mask)
        core_generation = logical.core_generation
        soul_generation = logical.soul_generation
        live_cursor = _cursor_copy(logical.cursor)
        anchor_before = logical.cursor_anchors
        cursor_before, candidate_anchors = _rebase_cursor(
            snapshot,
            live_cursor,
            anchor_before,
        )
        soul_before = logical.soul.detach().cpu().contiguous().clone()
        mask_before = logical.soul_mask.detach().cpu().contiguous().clone()
        adapter_hash = canonical_sha256(logical.adapter_manifest)
        rng_hash = canonical_sha256(logical.rng_manifest)
        base_blob = encode_soul_blob(
            soul_before,
            mask_before,
            soul_id=logical.soul_id,
            generation=soul_generation,
            base_field_id=logical.committed_field_id,
        )
        page = compile_next_read_page(
            snapshot,
            proposal_region=region,
            cursor=cursor_before,
        )
        view = page.view
        if view.field16.shape != (384, 16):
            raise CoreBackendError("compiled field view is not exactly 384x16")
        if view.role_ids.shape != (384,) or view.attention_mask.shape != (384,):
            raise CoreBackendError("compiled field metadata has the wrong shape")
        if set(np.unique(view.role_ids).tolist()) - {0, 1, 2}:
            raise CoreBackendError("compiled field has unknown physical roles")

        device, dtype = _device_dtype(logical)
        field16 = torch.from_numpy(
            np.array(view.field16, copy=True)
        ).unsqueeze(0).to(device=device, dtype=dtype)
        role_ids = torch.from_numpy(
            np.array(view.role_ids, copy=True)
        ).unsqueeze(0).to(device=device, dtype=torch.long)
        attention = torch.from_numpy(
            np.array(view.attention_mask, copy=True)
        ).unsqueeze(0).to(device=device, dtype=torch.bool)
        soul_input = soul_before.clone().unsqueeze(0).to(
            device=device,
            dtype=dtype,
        )
        mask_input = mask_before.clone().unsqueeze(0).to(
            device=device,
            dtype=torch.bool,
        )
        bank = get_letter_bank()
        bank_unit = torch.from_numpy(bank.vecs_unit.copy()).to(
            device=device,
            dtype=dtype,
        )

        core = logical.core
        module_modes = tuple(
            (module, bool(module.training)) for module in core.modules()
        )
        output: Mapping[str, Any]
        indices: list[int]
        with logical.shared_model.inference_lock:
            try:
                core.eval()
                with torch.inference_mode():
                    output = core.forward_charslot(
                        field16,
                        role_ids,
                        soul_input,
                        mask=attention,
                        soul_mask=mask_input,
                        response_slice=slice(PROPOSAL_START, PROPOSAL_END),
                    )
                    delta16 = output.get("response_delta_16")
                    if not isinstance(delta16, torch.Tensor):
                        raise CoreBackendError(
                            "AxonCore did not emit response_delta_16"
                        )
                    if tuple(delta16.shape) != (
                        1,
                        PROPOSAL_END - PROPOSAL_START,
                        16,
                    ):
                        raise CoreBackendError(
                            "response_delta_16 has the wrong shape"
                        )
                    if not bool(torch.isfinite(delta16).all()):
                        raise CoreBackendError(
                            "response_delta_16 contains non-finite values"
                        )
                    logits = core.charslot_logits(delta16, bank_unit)
                    if not bool(torch.isfinite(logits).all()):
                        raise CoreBackendError(
                            "charslot logits contain non-finite values"
                        )
                    indices = logits.argmax(dim=-1)[0].detach().cpu().tolist()
            finally:
                for module, was_training in module_modes:
                    module.training = was_training

        soul_output = output.get("soul")
        if not isinstance(soul_output, torch.Tensor):
            raise CoreBackendError("AxonCore did not emit a candidate soul")
        candidate_soul = (
            soul_output[0].detach().to(device="cpu", dtype=torch.float32).clone()
        )
        candidate_mask = mask_before.clone()
        _validate_runtime_soul(logical, candidate_soul, candidate_mask)

        decoded: list[str] = []
        for index in indices[:max_output_chars]:
            if index == bank.empty_index:
                break
            if not 0 <= index < len(bank.chars):
                raise CoreBackendError(
                    f"AxonCore emitted invalid alphabet index {index}"
                )
            decoded.append(bank.chars[index])
        text = "".join(decoded)
        proposal = RegionProposal(
            target_region=region,
            text=text,
            provenance=(
                f"axon_exact_v4:{logical.model_id}:{logical.core_id}:"
                f"tick:{tick}:view:{view.view_hash}"
            ),
        )
        candidate_blob = encode_soul_blob(
            candidate_soul,
            candidate_mask,
            soul_id=logical.soul_id,
            generation=soul_generation + 1,
            base_field_id=canonical_base,
        )
        candidate_cursor = _cursor_copy(page.next_cursor)
        manifest = CandidateStateManifest(
            core_id=logical.core_id,
            soul_id=logical.soul_id,
            model_id=logical.model_id,
            checkpoint_sha256=(
                logical.shared_model.checkpoint.checkpoint_sha256
            ),
            base_core_generation=core_generation,
            candidate_core_generation=core_generation + 1,
            base_soul_generation=soul_generation,
            candidate_soul_generation=soul_generation + 1,
            prior_private_field_id=logical.committed_field_id,
            canonical_base_field_id=canonical_base,
            observed_field_id=snapshot.field_id,
            observed_tick_id=snapshot.tick_id,
            observed_ancestor_field_ids=observed_ancestors,
            tick_index=tick,
            target_region=region.value,
            proposed_text=text,
            view_hash=view.view_hash,
            cursor_before_sha256=_cursor_hash(live_cursor),
            rebased_cursor_sha256=_cursor_hash(cursor_before),
            cursor_after_sha256=_cursor_hash(candidate_cursor),
            cursor_anchor_before_sha256=anchor_before.anchor_sha256,
            cursor_anchor_after_sha256=candidate_anchors.anchor_sha256,
            base_soul_blob_sha256=base_blob.sha256,
            candidate_soul_blob_sha256=candidate_blob.sha256,
            adapter_manifest_sha256=adapter_hash,
            rng_manifest_sha256=rng_hash,
        )
        return PreparedCoreAction(
            proposal=proposal,
            candidate_soul=candidate_soul.clone(),
            candidate_mask=candidate_mask.clone(),
            candidate_cursor=_cursor_copy(candidate_cursor),
            candidate_cursor_anchors=candidate_anchors,
            candidate_blob=candidate_blob,
            manifest=manifest,
        )


def _candidate_error(message: str) -> CandidateValidationError:
    return CandidateValidationError(message)


def validate_candidate_install(
    logical: LogicalCore,
    action: PreparedCoreAction,
    *,
    expected_core_generation: int,
    expected_soul_generation: int,
    canonical_base_field_id: str,
    expected_observed_field_id: str,
    expected_observed_tick_id: int,
    final_accepted_text: str,
) -> ValidatedCandidateInstall:
    """Validate acceptance exactly; perform neither install nor persistence."""

    if not isinstance(logical, LogicalCore):
        raise TypeError("logical must be LogicalCore")
    if not isinstance(action, PreparedCoreAction):
        raise TypeError("action must be PreparedCoreAction")
    core_generation = _nonnegative_int(
        expected_core_generation,
        "expected_core_generation",
    )
    soul_generation = _nonnegative_int(
        expected_soul_generation,
        "expected_soul_generation",
    )
    canonical_base = _nonempty(
        canonical_base_field_id,
        "canonical_base_field_id",
    )
    observed_field = _nonempty(
        expected_observed_field_id,
        "expected_observed_field_id",
    )
    observed_tick = _nonnegative_int(
        expected_observed_tick_id,
        "expected_observed_tick_id",
    )
    if not isinstance(final_accepted_text, str):
        raise CandidateValidationError(
            "final_accepted_text must be a string"
        )
    manifest = action.manifest

    with logical._state_lock:
        if (
            logical.core_id != manifest.core_id
            or logical.soul_id != manifest.soul_id
            or logical.model_id != manifest.model_id
        ):
            raise _candidate_error("candidate identity does not match logical core")
        if (
            manifest.checkpoint_sha256
            != logical.shared_model.checkpoint.checkpoint_sha256
        ):
            raise _candidate_error("candidate base checkpoint has changed")
        if (
            core_generation != logical.core_generation
            or core_generation != manifest.base_core_generation
        ):
            raise _candidate_error("candidate core generation is stale")
        if (
            soul_generation != logical.soul_generation
            or soul_generation != manifest.base_soul_generation
        ):
            raise _candidate_error("candidate soul generation is stale")
        if logical.committed_field_id != manifest.prior_private_field_id:
            raise _candidate_error("candidate prior private field is stale")
        if canonical_base != manifest.canonical_base_field_id:
            raise _candidate_error("candidate canonical base field is stale")
        if (
            observed_field != manifest.observed_field_id
            or observed_tick != manifest.observed_tick_id
        ):
            raise _candidate_error(
                "candidate observed field/tick evidence does not match"
            )
        if manifest.observed_field_id != manifest.canonical_base_field_id:
            ancestors = manifest.observed_ancestor_field_ids
            if (
                not ancestors
                or ancestors[0] != manifest.canonical_base_field_id
                or len(ancestors) != len(set(ancestors))
            ):
                raise _candidate_error("candidate observed lineage is invalid")
        if (
            final_accepted_text != action.proposal.text
            or final_accepted_text != manifest.proposed_text
        ):
            raise _candidate_error(
                "final accepted text differs from the staged proposal"
            )
        if action.proposal.target_region.value != manifest.target_region:
            raise _candidate_error("candidate target region was modified")
        if manifest.manifest_id != canonical_sha256(
            manifest.to_canonical_dict()
        ):
            raise _candidate_error("candidate manifest hash mismatch")
        if _cursor_hash(logical.cursor) != manifest.cursor_before_sha256:
            raise _candidate_error("candidate cursor is stale")
        if (
            logical.cursor_anchors.anchor_sha256
            != manifest.cursor_anchor_before_sha256
        ):
            raise _candidate_error("candidate cursor anchors are stale")
        if _cursor_hash(action.candidate_cursor) != manifest.cursor_after_sha256:
            raise _candidate_error("candidate cursor was modified")
        if (
            action.candidate_cursor_anchors.anchor_sha256
            != manifest.cursor_anchor_after_sha256
        ):
            raise _candidate_error("candidate cursor anchors were modified")
        if (
            canonical_sha256(logical.adapter_manifest)
            != manifest.adapter_manifest_sha256
        ):
            raise _candidate_error("adapter manifest changed after preparation")
        if (
            canonical_sha256(logical.rng_manifest)
            != manifest.rng_manifest_sha256
        ):
            raise _candidate_error("RNG manifest changed after preparation")

        try:
            current_blob = encode_soul_blob(
                logical.soul,
                logical.soul_mask,
                soul_id=logical.soul_id,
                generation=logical.soul_generation,
                base_field_id=logical.committed_field_id,
            )
            if current_blob.sha256 != manifest.base_soul_blob_sha256:
                raise _candidate_error("live soul changed after preparation")
            rebuilt = encode_soul_blob(
                action.candidate_soul,
                action.candidate_mask,
                soul_id=logical.soul_id,
                generation=manifest.candidate_soul_generation,
                base_field_id=manifest.canonical_base_field_id,
            )
        except SoulStoreError as exc:
            raise _candidate_error(
                f"candidate soul/blob is invalid: {exc}"
            ) from exc
        if (
            rebuilt.sha256 != manifest.candidate_soul_blob_sha256
            or rebuilt.sha256 != action.candidate_blob.sha256
            or rebuilt.data != action.candidate_blob.data
        ):
            raise _candidate_error("candidate soul/blob was modified")
        try:
            decoded = decode_soul_blob(
                action.candidate_blob.data,
                action.candidate_blob.sha256,
            )
        except SoulStoreError as exc:
            raise _candidate_error(
                f"candidate soul/blob is invalid: {exc}"
            ) from exc
        if not torch.equal(decoded.soul, action.candidate_soul) or not torch.equal(
            decoded.mask,
            action.candidate_mask,
        ):
            raise _candidate_error("candidate blob does not match candidate tensors")

        validation_payload = {
            "schema": "axon-runtime-candidate-install-validation-v1",
            "manifest_id": manifest.manifest_id,
            "core_id": logical.core_id,
            "soul_id": logical.soul_id,
            "expected_core_generation": core_generation,
            "expected_soul_generation": soul_generation,
            "prior_private_field_id": manifest.prior_private_field_id,
            "canonical_base_field_id": canonical_base,
            "expected_observed_field_id": observed_field,
            "expected_observed_tick_id": observed_tick,
            "final_accepted_text": final_accepted_text,
        }
        return ValidatedCandidateInstall(
            action=action,
            expected_core_generation=core_generation,
            expected_soul_generation=soul_generation,
            canonical_base_field_id=canonical_base,
            expected_observed_field_id=observed_field,
            expected_observed_tick_id=observed_tick,
            final_accepted_text=final_accepted_text,
            validation_hash=canonical_sha256(validation_payload),
        )


def persist_candidate_soul(
    store: SoulBlobStore,
    action: PreparedCoreAction,
) -> StoredSoulBlob:
    """Persist staged bytes only; this never installs them into a core."""

    if not isinstance(store, SoulBlobStore):
        raise TypeError("store must be SoulBlobStore")
    if not isinstance(action, PreparedCoreAction):
        raise TypeError("action must be PreparedCoreAction")
    return store.persist(action.candidate_blob)


def install_validated_candidate(
    logical: LogicalCore,
    receipt: ValidatedCandidateInstall,
    *,
    committed_field_id: str,
) -> None:
    """Install an already accepted candidate in memory, without persistence."""

    if not isinstance(logical, LogicalCore):
        raise TypeError("logical must be LogicalCore")
    if not isinstance(receipt, ValidatedCandidateInstall):
        raise TypeError("receipt must be ValidatedCandidateInstall")
    output_field_id = _nonempty(committed_field_id, "committed_field_id")
    with logical._state_lock:
        refreshed = validate_candidate_install(
            logical,
            receipt.action,
            expected_core_generation=receipt.expected_core_generation,
            expected_soul_generation=receipt.expected_soul_generation,
            canonical_base_field_id=receipt.canonical_base_field_id,
            expected_observed_field_id=receipt.expected_observed_field_id,
            expected_observed_tick_id=receipt.expected_observed_tick_id,
            final_accepted_text=receipt.final_accepted_text,
        )
        if refreshed.validation_hash != receipt.validation_hash:
            raise CandidateValidationError("validation receipt hash mismatch")
        action = receipt.action
        logical.soul = action.candidate_soul.detach().cpu().contiguous().clone()
        logical.soul_mask = (
            action.candidate_mask.detach().cpu().contiguous().clone()
        )
        logical.cursor = _cursor_copy(action.candidate_cursor)
        logical.cursor_anchors = action.candidate_cursor_anchors
        logical.core_generation = action.manifest.candidate_core_generation
        logical.soul_generation = action.manifest.candidate_soul_generation
        logical.committed_field_id = output_field_id
        logical.last_candidate_manifest_id = action.manifest.manifest_id


def persist_committed_runtime_state(
    state_store: PrivateRuntimeStateStore,
    logical: LogicalCore,
    receipt: ValidatedCandidateInstall,
    stored_soul: StoredSoulBlob,
) -> StoredPrivateRuntimeState:
    """Persist a post-install crash image, separate from acceptance/install."""

    if not isinstance(state_store, PrivateRuntimeStateStore):
        raise TypeError("state_store must be PrivateRuntimeStateStore")
    if not isinstance(logical, LogicalCore):
        raise TypeError("logical must be LogicalCore")
    if not isinstance(receipt, ValidatedCandidateInstall):
        raise TypeError("receipt must be ValidatedCandidateInstall")
    if not isinstance(stored_soul, StoredSoulBlob):
        raise TypeError("stored_soul must be StoredSoulBlob")
    action = receipt.action
    manifest = action.manifest
    with logical._state_lock:
        if (
            logical.core_id != manifest.core_id
            or logical.soul_id != manifest.soul_id
            or logical.model_id != manifest.model_id
            or logical.core_generation != manifest.candidate_core_generation
            or logical.soul_generation != manifest.candidate_soul_generation
            or logical.last_candidate_manifest_id != manifest.manifest_id
        ):
            raise CandidateValidationError(
                "logical core is not the installed candidate"
            )
        if stored_soul.sha256 != manifest.candidate_soul_blob_sha256:
            raise CandidateValidationError(
                "stored soul does not match installed candidate"
            )
        if not torch.equal(logical.soul, action.candidate_soul) or not torch.equal(
            logical.soul_mask,
            action.candidate_mask,
        ):
            raise CandidateValidationError(
                "installed soul differs from the accepted candidate"
            )
        if _cursor_hash(logical.cursor) != manifest.cursor_after_sha256:
            raise CandidateValidationError(
                "installed cursor differs from the accepted candidate"
            )
        if (
            logical.cursor_anchors.anchor_sha256
            != manifest.cursor_anchor_after_sha256
        ):
            raise CandidateValidationError(
                "installed cursor anchors differ from accepted candidate"
            )
        if (
            canonical_sha256(logical.adapter_manifest)
            != manifest.adapter_manifest_sha256
            or canonical_sha256(logical.rng_manifest)
            != manifest.rng_manifest_sha256
        ):
            raise CandidateValidationError(
                "installed private manifests changed after acceptance"
            )
        blob = encode_private_runtime_state(
            core_id=logical.core_id,
            soul_id=logical.soul_id,
            model_id=logical.model_id,
            core_generation=logical.core_generation,
            soul_generation=logical.soul_generation,
            committed_field_id=logical.committed_field_id,
            soul_blob_sha256=stored_soul.sha256,
            candidate_manifest_id=manifest.manifest_id,
            cursor=logical.cursor,
            cursor_anchor_manifest=(
                logical.cursor_anchors.to_canonical_dict()
            ),
            adapter_manifest=logical.adapter_manifest,
            rng_manifest=logical.rng_manifest,
        )
        return state_store.persist(blob)


def prepare_persisted_runtime_state(
    *,
    soul_store: SoulBlobStore,
    state_store: PrivateRuntimeStateStore,
    logical: LogicalCore,
    receipt: ValidatedCandidateInstall,
    intended_committed_field_id: str,
) -> PreparedPersistedRuntimeState:
    """Persist candidate artifacts before DB commit without changing live RAM.

    The returned hashes are suitable for the next ``CoreStateManifest``.  If
    the database transaction fails, these immutable content-addressed blobs
    are merely unreachable and the logical core still needs no rollback.
    """

    if not isinstance(soul_store, SoulBlobStore):
        raise TypeError("soul_store must be SoulBlobStore")
    if not isinstance(state_store, PrivateRuntimeStateStore):
        raise TypeError("state_store must be PrivateRuntimeStateStore")
    if not isinstance(logical, LogicalCore):
        raise TypeError("logical must be LogicalCore")
    if not isinstance(receipt, ValidatedCandidateInstall):
        raise TypeError("receipt must be ValidatedCandidateInstall")
    output_field_id = _nonempty(
        intended_committed_field_id,
        "intended_committed_field_id",
    )
    action = receipt.action
    manifest = action.manifest
    with logical._state_lock:
        refreshed = validate_candidate_install(
            logical,
            action,
            expected_core_generation=receipt.expected_core_generation,
            expected_soul_generation=receipt.expected_soul_generation,
            canonical_base_field_id=receipt.canonical_base_field_id,
            expected_observed_field_id=receipt.expected_observed_field_id,
            expected_observed_tick_id=receipt.expected_observed_tick_id,
            final_accepted_text=receipt.final_accepted_text,
        )
        if refreshed.validation_hash != receipt.validation_hash:
            raise CandidateValidationError("validation receipt hash mismatch")
        stored_soul = soul_store.persist(action.candidate_blob)
        private_blob = encode_private_runtime_state(
            core_id=logical.core_id,
            soul_id=logical.soul_id,
            model_id=logical.model_id,
            core_generation=manifest.candidate_core_generation,
            soul_generation=manifest.candidate_soul_generation,
            committed_field_id=output_field_id,
            soul_blob_sha256=stored_soul.sha256,
            candidate_manifest_id=manifest.manifest_id,
            cursor=action.candidate_cursor,
            cursor_anchor_manifest=(
                action.candidate_cursor_anchors.to_canonical_dict()
            ),
            adapter_manifest=logical.adapter_manifest,
            rng_manifest=logical.rng_manifest,
        )
        stored_private = state_store.persist(private_blob)
        return PreparedPersistedRuntimeState(
            receipt=receipt,
            intended_committed_field_id=output_field_id,
            stored_soul=stored_soul,
            stored_private_state=stored_private,
        )


def restore_logical_core_from_state(
    backend: CoreBackend,
    state: DecodedPrivateRuntimeState,
    soul_store: SoulBlobStore,
) -> LogicalCore:
    """Restore a logical clone from validated state and referenced soul bytes."""

    if not isinstance(backend, CoreBackend):
        raise TypeError("backend must be CoreBackend")
    if not isinstance(state, DecodedPrivateRuntimeState):
        raise TypeError("state must be DecodedPrivateRuntimeState")
    if not isinstance(soul_store, SoulBlobStore):
        raise TypeError("soul_store must be SoulBlobStore")
    decoded_soul = soul_store.load(state.soul_blob_sha256)
    header = decoded_soul.header
    if (
        header.get("soul_id") != state.soul_id
        or header.get("generation") != state.soul_generation
    ):
        raise CandidateValidationError(
            "runtime-state soul reference has incompatible identity/generation"
        )
    return backend.create_logical_core(
        model_id=state.model_id,
        core_id=state.core_id,
        soul_id=state.soul_id,
        committed_field_id=state.committed_field_id,
        core_generation=state.core_generation,
        soul_generation=state.soul_generation,
        soul=decoded_soul.soul,
        soul_mask=decoded_soul.mask,
        cursor=state.cursor,
        cursor_anchor_manifest=state.cursor_anchor_manifest,
        adapter_manifest=state.adapter_manifest,
        rng_manifest=state.rng_manifest,
        last_candidate_manifest_id=state.candidate_manifest_id,
    )


def restore_or_replace_logical_core_from_state(
    backend: CoreBackend,
    state: DecodedPrivateRuntimeState,
    soul_store: SoulBlobStore,
) -> LogicalCore:
    """Restore one exact private state and atomically publish it in a backend."""

    if not isinstance(backend, CoreBackend):
        raise TypeError("backend must be CoreBackend")
    if not isinstance(state, DecodedPrivateRuntimeState):
        raise TypeError("state must be DecodedPrivateRuntimeState")
    if not isinstance(soul_store, SoulBlobStore):
        raise TypeError("soul_store must be SoulBlobStore")
    decoded_soul = soul_store.load(state.soul_blob_sha256)
    header = decoded_soul.header
    if (
        header.get("soul_id") != state.soul_id
        or header.get("generation") != state.soul_generation
    ):
        raise CandidateValidationError(
            "runtime-state soul reference has incompatible identity/generation"
        )
    with backend._registry_lock:
        try:
            shared = backend._models[state.model_id]
        except KeyError as exc:
            raise ModelRegistrationError(
                f"model_id {state.model_id!r} is not loaded"
            ) from exc
        existing = backend._cores.get(state.core_id)
        if existing is not None and existing.soul_id != state.soul_id:
            raise CoreBackendError(
                "cannot replace a logical core with a different soul identity"
            )
        if state.soul_id in backend._soul_ids and (
            existing is None or existing.soul_id != state.soul_id
        ):
            raise CoreBackendError(
                f"soul_id {state.soul_id!r} belongs to another logical core"
            )
        restored = LogicalCore(
            core_id=state.core_id,
            soul_id=state.soul_id,
            shared_model=shared,
            committed_field_id=state.committed_field_id,
            core_generation=state.core_generation,
            soul_generation=state.soul_generation,
            soul=decoded_soul.soul,
            soul_mask=decoded_soul.mask,
            cursor=state.cursor,
            cursor_anchors=CursorAnchorManifest.from_mapping(
                state.cursor_anchor_manifest
            ),
            adapter_manifest=state.adapter_manifest,
            rng_manifest=state.rng_manifest,
            last_candidate_manifest_id=state.candidate_manifest_id,
        )
        backend._cores[state.core_id] = restored
        backend._soul_ids.add(state.soul_id)
        return restored


__all__ = [
    "CoreBackendError",
    "ModelRegistrationError",
    "CandidateValidationError",
    "CursorRegionAnchor",
    "CursorAnchorManifest",
    "SharedAxonModel",
    "CandidateStateManifest",
    "PreparedCoreAction",
    "ValidatedCandidateInstall",
    "PreparedPersistedRuntimeState",
    "LogicalCore",
    "CoreBackend",
    "prepare_action",
    "validate_candidate_install",
    "persist_candidate_soul",
    "install_validated_candidate",
    "persist_committed_runtime_state",
    "prepare_persisted_runtime_state",
    "restore_logical_core_from_state",
    "restore_or_replace_logical_core_from_state",
]
