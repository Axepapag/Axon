"""Production adapter between exact-v4 logical cores and the tick engine.

The adapter is deliberately two phase:

1. inference stages an uninstalled candidate;
2. acceptance persists immutable soul/private-state blobs and returns a
   ``CoreStateManifest``;
3. only after SQLite commits that manifest is the candidate installed in RAM.

At every tick boundary ``reconcile_head`` reloads the journal-referenced
private state.  A process loss between the database commit and RAM installation
therefore cannot make the next tick run with stale private state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from runtime.field import FieldViewCursor, LogicalRegion, SharedFieldSnapshot

from .contracts import (
    CoreIdentity,
    CoreStateManifest,
    RuntimeHead,
    validate_identity_population,
)
from .core_backend import (
    CoreBackend,
    CandidateValidationError,
    CursorAnchorManifest,
    LogicalCore,
    PreparedCoreAction,
    PreparedPersistedRuntimeState,
    install_validated_candidate,
    prepare_persisted_runtime_state,
)
from .engine import (
    ObservedField,
    PreparedCoreCommit,
    StagedCoreTurn,
)
from .soul_store import (
    DecodedPrivateRuntimeState,
    PrivateRuntimeStateStore,
    SoulBlobStore,
)
from runtime.field import canonical_sha256


class ExactV4DriverError(RuntimeError):
    """Exact-v4 private-state evidence did not match canonical runtime truth."""


@dataclass(frozen=True, slots=True)
class ExactV4CommitToken:
    """The exact pre-commit artifact set later installed or reloaded."""

    prepared_state: PreparedPersistedRuntimeState
    prior_manifest_id: str


def _manifest_map(
    manifests: Sequence[CoreStateManifest],
) -> dict[str, CoreStateManifest]:
    values = tuple(manifests)
    result = {item.core_id: item for item in values}
    if len(result) != len(values):
        raise ExactV4DriverError("duplicate core-state manifest")
    return result


class ExactV4RuntimeDriver:
    """Run cloned exact-v4 cores over shared immutable model weights."""

    def __init__(
        self,
        *,
        backend: CoreBackend,
        soul_store: SoulBlobStore,
        private_state_store: PrivateRuntimeStateStore,
        identities: Sequence[CoreIdentity],
    ) -> None:
        if not isinstance(backend, CoreBackend):
            raise TypeError("backend must be CoreBackend")
        if not isinstance(soul_store, SoulBlobStore):
            raise TypeError("soul_store must be SoulBlobStore")
        if not isinstance(private_state_store, PrivateRuntimeStateStore):
            raise TypeError(
                "private_state_store must be PrivateRuntimeStateStore"
            )
        values = tuple(identities)
        validate_identity_population(values)
        self.backend = backend
        self.soul_store = soul_store
        self.private_state_store = private_state_store
        self.identities = values
        self._identity_by_id = {item.core_id: item for item in values}

    def _logical(self, core_id: str) -> LogicalCore:
        if core_id not in self._identity_by_id:
            raise ExactV4DriverError(f"unknown logical core {core_id!r}")
        return self.backend.logical_core(core_id)

    def _validate_persisted_manifest(
        self,
        manifest: CoreStateManifest,
    ) -> DecodedPrivateRuntimeState:
        try:
            identity = self._identity_by_id[manifest.core_id]
        except KeyError as exc:
            raise ExactV4DriverError(
                f"manifest names unknown core {manifest.core_id!r}"
            ) from exc
        if (
            manifest.soul_id != identity.soul_id
            or manifest.model_id != identity.model_id
            or manifest.core_state_sha256 != identity.core_state_sha256
        ):
            raise ExactV4DriverError(
                "core-state manifest does not match immutable identity"
            )
        state = self.private_state_store.load(manifest.cursor_state_sha256)
        if (
            state.core_id != manifest.core_id
            or state.soul_id != manifest.soul_id
            or state.model_id != manifest.model_id
            or state.committed_field_id != manifest.committed_field_id
            or state.soul_blob_sha256 != manifest.soul_state_sha256
        ):
            raise ExactV4DriverError(
                "private runtime-state blob does not match its manifest"
            )
        decoded_soul = self.soul_store.load(state.soul_blob_sha256)
        if (
            decoded_soul.header.get("soul_id") != manifest.soul_id
            or decoded_soul.header.get("generation") != state.soul_generation
        ):
            raise ExactV4DriverError(
                "soul blob identity/generation does not match private state"
            )
        checkpoint = self.backend.shared_model(identity.model_id).checkpoint
        if (
            checkpoint.checkpoint_sha256
            != identity.base_checkpoint_sha256
            or checkpoint.core_state_sha256 != identity.core_state_sha256
        ):
            raise ExactV4DriverError(
                "loaded immutable model does not match pinned identity"
            )
        adapter_hash = canonical_sha256(dict(state.adapter_manifest))
        rng_hash = canonical_sha256(dict(state.rng_manifest))
        if (
            manifest.adapter_set_sha256 is not None
            and manifest.adapter_set_sha256 != adapter_hash
        ):
            raise ExactV4DriverError("adapter manifest hash mismatch")
        if (
            manifest.rng_state_sha256 is not None
            and manifest.rng_state_sha256 != rng_hash
        ):
            raise ExactV4DriverError("private RNG manifest hash mismatch")
        return state

    def reconcile_head(self, head: RuntimeHead) -> None:
        """Make every live logical core equal durable journal head evidence."""

        by_id = _manifest_map(head.core_state_manifests)
        if set(by_id) != set(self._identity_by_id):
            raise ExactV4DriverError(
                "runtime head does not exactly cover configured core identities"
            )
        # Validate the complete population before publishing any replacement.
        decoded = {
            core_id: self._validate_persisted_manifest(manifest)
            for core_id, manifest in by_id.items()
        }
        for core_id in sorted(decoded):
            manifest = by_id[core_id]
            restored = self.backend.restore_or_replace_logical_core(
                cursor_state_sha256=manifest.cursor_state_sha256,
                state_store=self.private_state_store,
                soul_store=self.soul_store,
            )
            if (
                restored.core_id != core_id
                or restored.committed_field_id != manifest.committed_field_id
            ):
                raise ExactV4DriverError(
                    "restored logical core does not match canonical manifest"
                )

    def stage_turn(
        self,
        *,
        core_id: str,
        observation: ObservedField,
        canonical_base_field_id: str,
        tick_seq: int,
        target_region: LogicalRegion,
    ) -> StagedCoreTurn:
        logical = self._logical(core_id)
        action = logical.prepare_action(
            observation.snapshot,
            canonical_base_field_id=canonical_base_field_id,
            expected_observed_field_id=observation.snapshot.field_id,
            expected_observed_tick_id=observation.snapshot.tick_id,
            observed_ancestor_field_ids=observation.ancestor_field_ids,
            target_region=target_region,
            tick_index=tick_seq,
        )
        return StagedCoreTurn(
            core_id=core_id,
            proposal=action.proposal,
            observed_field_id=action.manifest.observed_field_id,
            observed_tick_id=action.manifest.observed_tick_id,
            candidate_token=action,
            evidence_ids=(
                action.manifest.manifest_id,
                action.manifest.view_hash,
            ),
        )

    def prepare_commit(
        self,
        *,
        turn: StagedCoreTurn,
        accepted_text: str,
        output_field_id: str,
        runtime_generation: int,
        next_tick_seq: int,
        prior_manifest: CoreStateManifest,
    ) -> PreparedCoreCommit:
        if not isinstance(turn.candidate_token, PreparedCoreAction):
            raise ExactV4DriverError(
                "staged turn does not contain an exact-v4 candidate"
            )
        action = turn.candidate_token
        if (
            action.proposal != turn.proposal
            or action.manifest.core_id != turn.core_id
            or action.manifest.observed_field_id != turn.observed_field_id
            or action.manifest.observed_tick_id != turn.observed_tick_id
        ):
            raise ExactV4DriverError("staged exact-v4 turn was modified")
        logical = self._logical(turn.core_id)
        if (
            prior_manifest.core_id != logical.core_id
            or prior_manifest.soul_id != logical.soul_id
            or prior_manifest.model_id != logical.model_id
            or prior_manifest.committed_field_id != logical.committed_field_id
            or prior_manifest.core_state_sha256
            != logical.shared_model.checkpoint.core_state_sha256
        ):
            raise ExactV4DriverError(
                "prior core-state manifest does not match live private state"
            )
        # This also proves that the current canonical manifest references a
        # readable, hash-valid crash image before deriving its successor.
        self._validate_persisted_manifest(prior_manifest)
        receipt = logical.validate_candidate(
            action,
            expected_core_generation=logical.core_generation,
            expected_soul_generation=logical.soul_generation,
            canonical_base_field_id=action.manifest.canonical_base_field_id,
            expected_observed_field_id=turn.observed_field_id,
            expected_observed_tick_id=turn.observed_tick_id,
            final_accepted_text=accepted_text,
        )
        persisted = prepare_persisted_runtime_state(
            soul_store=self.soul_store,
            state_store=self.private_state_store,
            logical=logical,
            receipt=receipt,
            intended_committed_field_id=output_field_id,
        )
        manifest = CoreStateManifest(
            core_id=logical.core_id,
            soul_id=logical.soul_id,
            model_id=logical.model_id,
            core_state_sha256=(
                logical.shared_model.checkpoint.core_state_sha256
            ),
            generation=runtime_generation,
            tick_seq=next_tick_seq,
            committed_field_id=output_field_id,
            soul_state_sha256=persisted.soul_state_sha256,
            cursor_state_sha256=persisted.cursor_state_sha256,
            adapter_set_sha256=canonical_sha256(logical.adapter_manifest),
            rng_state_sha256=canonical_sha256(logical.rng_manifest),
            parent_manifest_id=prior_manifest.manifest_id,
        )
        self._validate_persisted_manifest(manifest)
        return PreparedCoreCommit(
            core_id=turn.core_id,
            manifest=manifest,
            commit_token=ExactV4CommitToken(
                prepared_state=persisted,
                prior_manifest_id=prior_manifest.manifest_id,
            ),
        )

    def finalize_commit(self, prepared: PreparedCoreCommit) -> None:
        token = prepared.commit_token
        if not isinstance(token, ExactV4CommitToken):
            raise ExactV4DriverError(
                "prepared commit does not contain exact-v4 state"
            )
        if token.prior_manifest_id != prepared.manifest.parent_manifest_id:
            raise ExactV4DriverError("prepared commit parent changed")
        persisted = token.prepared_state
        if (
            persisted.soul_state_sha256
            != prepared.manifest.soul_state_sha256
            or persisted.cursor_state_sha256
            != prepared.manifest.cursor_state_sha256
            or persisted.intended_committed_field_id
            != prepared.manifest.committed_field_id
        ):
            raise ExactV4DriverError(
                "prepared private state does not match committed manifest"
            )
        logical = self._logical(prepared.core_id)
        install_validated_candidate(
            logical,
            persisted.receipt,
            committed_field_id=prepared.manifest.committed_field_id,
        )

    def recover_committed(self, prepared: PreparedCoreCommit) -> None:
        self._validate_persisted_manifest(prepared.manifest)
        self.backend.restore_or_replace_logical_core(
            cursor_state_sha256=prepared.manifest.cursor_state_sha256,
            state_store=self.private_state_store,
            soul_store=self.soul_store,
        )

    def discard_turn(self, turn: StagedCoreTurn) -> None:
        if not isinstance(turn.candidate_token, PreparedCoreAction):
            raise ExactV4DriverError(
                "discarded turn does not contain an exact-v4 candidate"
            )
        # Candidate tensors are immutable, unpersisted values.  Releasing the
        # staged object is sufficient; live private state never changed.


def initialize_logical_core_state(
    *,
    logical: LogicalCore,
    identity: CoreIdentity,
    genesis: SharedFieldSnapshot,
    soul_store: SoulBlobStore,
    private_state_store: PrivateRuntimeStateStore,
) -> CoreStateManifest:
    """Persist one generation-zero private state for store initialization."""

    if not isinstance(logical, LogicalCore):
        raise TypeError("logical must be LogicalCore")
    if not isinstance(identity, CoreIdentity):
        raise TypeError("identity must be CoreIdentity")
    if not isinstance(genesis, SharedFieldSnapshot):
        raise TypeError("genesis must be SharedFieldSnapshot")
    if (
        logical.core_id != identity.core_id
        or logical.soul_id != identity.soul_id
        or logical.model_id != identity.model_id
        or logical.committed_field_id != genesis.field_id
        or logical.shared_model.checkpoint.checkpoint_sha256
        != identity.base_checkpoint_sha256
        or logical.shared_model.checkpoint.core_state_sha256
        != identity.core_state_sha256
    ):
        raise ExactV4DriverError(
            "generation-zero logical core does not match its identity/genesis"
        )
    stored_soul = soul_store.write(
        logical.soul,
        logical.soul_mask,
        soul_id=logical.soul_id,
        generation=logical.soul_generation,
        base_field_id=genesis.field_id,
    )
    genesis_candidate_id = canonical_sha256(
        {
            "schema": "axon-runtime-genesis-private-state-v1",
            "core_id": logical.core_id,
            "soul_id": logical.soul_id,
            "model_id": logical.model_id,
            "field_id": genesis.field_id,
            "soul_state_sha256": stored_soul.sha256,
        }
    )
    stored_private = private_state_store.write(
        core_id=logical.core_id,
        soul_id=logical.soul_id,
        model_id=logical.model_id,
        core_generation=logical.core_generation,
        soul_generation=logical.soul_generation,
        committed_field_id=genesis.field_id,
        soul_blob_sha256=stored_soul.sha256,
        candidate_manifest_id=genesis_candidate_id,
        cursor=FieldViewCursor(),
        cursor_anchor_manifest=CursorAnchorManifest().to_canonical_dict(),
        adapter_manifest=logical.adapter_manifest,
        rng_manifest=logical.rng_manifest,
    )
    return CoreStateManifest(
        core_id=logical.core_id,
        soul_id=logical.soul_id,
        model_id=logical.model_id,
        core_state_sha256=identity.core_state_sha256,
        generation=0,
        tick_seq=0,
        committed_field_id=genesis.field_id,
        soul_state_sha256=stored_soul.sha256,
        cursor_state_sha256=stored_private.sha256,
        adapter_set_sha256=canonical_sha256(logical.adapter_manifest),
        rng_state_sha256=canonical_sha256(logical.rng_manifest),
    )


__all__ = [
    "ExactV4DriverError",
    "ExactV4CommitToken",
    "ExactV4RuntimeDriver",
    "initialize_logical_core_state",
]
