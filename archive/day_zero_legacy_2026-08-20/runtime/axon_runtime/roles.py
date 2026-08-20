"""Deterministic round-robin role scheduling."""
from __future__ import annotations

from collections.abc import Sequence

from runtime.field import canonical_sha256

from .contracts import (
    ACTIVE_ROLES,
    CoreIdentity,
    RoleAssignment,
    RoleContractError,
    validate_identity_population,
)


class RoleScheduler:
    """Assign proposer, consolidator, and sleeper over one fixed core ring."""

    def __init__(self, identities: Sequence[CoreIdentity]) -> None:
        values = tuple(identities)
        validate_identity_population(values)
        enabled = tuple(identity for identity in values if identity.enabled)
        if len(enabled) < 3:
            raise RoleContractError(
                "the role ring requires at least three unique enabled cores"
            )
        for identity in enabled:
            missing = ACTIVE_ROLES - set(identity.role_capabilities)
            if missing:
                raise RoleContractError(
                    f"enabled core {identity.core_id!r} lacks roles "
                    f"{sorted(missing)}"
                )
        self._identities = enabled
        self._ring = tuple(identity.core_id for identity in enabled)
        self._schedule_hash = canonical_sha256(
            {
                "schema": "axon-role-ring-v1",
                "core_ids": list(self._ring),
                "identity_hashes": [
                    identity.identity_hash for identity in enabled
                ],
            }
        )

    @property
    def core_ids(self) -> tuple[str, ...]:
        return self._ring

    @property
    def size(self) -> int:
        return len(self._ring)

    @property
    def schedule_hash(self) -> str:
        return self._schedule_hash

    def role_index(self, tick_seq: int) -> int:
        if isinstance(tick_seq, bool) or not isinstance(tick_seq, int):
            raise RoleContractError("tick_seq must be an integer")
        if tick_seq < 0:
            raise RoleContractError("tick_seq must be non-negative")
        return tick_seq % self.size

    def assignment(self, tick_seq: int) -> RoleAssignment:
        index = self.role_index(tick_seq)
        proposer = self._ring[index]
        consolidator = self._ring[(index - 1) % self.size]
        sleeper = self._ring[(index - 2) % self.size]
        active = {proposer, consolidator, sleeper}
        standby = tuple(core_id for core_id in self._ring if core_id not in active)
        return RoleAssignment(
            tick_seq=tick_seq,
            proposer_core_id=proposer,
            consolidator_core_id=consolidator,
            sleeper_core_id=sleeper,
            standby_core_ids=standby,
        )

    def verify(self, assignment: RoleAssignment) -> None:
        if not isinstance(assignment, RoleAssignment):
            raise RoleContractError("assignment must be RoleAssignment")
        expected = self.assignment(assignment.tick_seq)
        if assignment != expected:
            raise RoleContractError(
                "role assignment does not match the persisted core ring"
            )


__all__ = ["RoleScheduler"]
