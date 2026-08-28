"""Derived, width-generic proposal-board rail workspaces.

The proposal workspace is deliberately noncanonical.  It renders exact board
accounting and typed deltas beside a frozen tick image so every destination
rail can inspect the same grounded decisions without pairwise neural
translation.  D64 is the first physical consumer; the renderer itself works for
every positive ``d_model`` divisible by the permanent 16D substrate width.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from runtime.field import canonical_json_bytes, canonical_sha256

from .board import ParticipantRecord, ProposalPass
from .reasoning_output import CategoricalTextFrame
from .tick import FrozenTickImage

PROPOSAL_WORKSPACE_SCHEMA = "axon-heart-proposal-workspace-v1"
PROPOSAL_RAIL_SCHEMA = "axon-heart-proposal-rail-v1"


@dataclass(frozen=True, slots=True)
class ProposalWorkspaceEntry:
    core_id: str
    d_model: int
    state: str
    detail: str
    emission_id: str | None
    delta: dict[str, Any] | None

    @classmethod
    def from_record(cls, record: ParticipantRecord) -> "ProposalWorkspaceEntry":
        proposal = record.proposal
        return cls(
            core_id=record.core_id,
            d_model=record.d_model,
            state=record.state.value,
            detail=record.detail,
            emission_id=None if proposal is None else proposal.emission_id,
            delta=None if proposal is None else proposal.delta.to_canonical_dict(),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "core_id": self.core_id,
            "d_model": self.d_model,
            "state": self.state,
            "detail": self.detail,
            "emission_id": self.emission_id,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class RenderedProposalRail:
    d_model: int
    source_workspace_id: str
    text_frame: CategoricalTextFrame
    rail_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.text_frame.d_model != self.d_model:
            raise ValueError("proposal rail frame width differs from rail d_model")
        if not isinstance(self.source_workspace_id, str) or not self.source_workspace_id:
            raise ValueError("source_workspace_id must be non-empty")
        object.__setattr__(self, "rail_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def rows(self):
        return self.text_frame.rows

    @property
    def text(self) -> str:
        return self.text_frame.text

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PROPOSAL_RAIL_SCHEMA,
            "d_model": self.d_model,
            "source_workspace_id": self.source_workspace_id,
            "text_frame": self.text_frame.to_canonical_dict(),
        }
        if include_id:
            value["rail_id"] = self.rail_id
        return value


@dataclass(frozen=True, slots=True)
class ProposalWorkspace:
    image_id: str
    tick_uid: str
    pass_kind: ProposalPass
    entries: tuple[ProposalWorkspaceEntry, ...]
    workspace_id: str = field(init=False)
    rendered_rails: tuple[RenderedProposalRail, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        pass_kind = self.pass_kind if isinstance(self.pass_kind, ProposalPass) else ProposalPass(self.pass_kind)
        entries = tuple(self.entries)
        object.__setattr__(self, "pass_kind", pass_kind)
        object.__setattr__(self, "entries", entries)
        if not self.image_id or not self.tick_uid:
            raise ValueError("proposal workspace identities must be non-empty")
        if not entries:
            raise ValueError("proposal workspace requires participant accounting")
        if len({item.core_id for item in entries}) != len(entries):
            raise ValueError("proposal workspace entries must have unique core ids")
        object.__setattr__(self, "workspace_id", canonical_sha256(self.to_canonical_dict(include_id=False, include_rails=False)))
        rails = tuple(self.rendered_rails)
        if rails and any(item.source_workspace_id != self.workspace_id for item in rails):
            raise ValueError("rendered proposal rail is not bound to this workspace")
        object.__setattr__(self, "rendered_rails", tuple(sorted(rails, key=lambda item: item.d_model)))

    def rail_for(self, d_model: int) -> RenderedProposalRail | None:
        return next((item for item in self.rendered_rails if item.d_model == d_model), None)

    def require_rail(self, d_model: int) -> RenderedProposalRail:
        rail = self.rail_for(d_model)
        if rail is None:
            raise KeyError(f"proposal workspace has no rendered d_model {d_model} rail")
        return rail

    def readable_text(self) -> str:
        value = self.to_canonical_dict(include_id=True, include_rails=False)
        return canonical_json_bytes(value).decode("utf-8")

    def to_canonical_dict(
        self,
        *,
        include_id: bool = True,
        include_rails: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": PROPOSAL_WORKSPACE_SCHEMA,
            "image_id": self.image_id,
            "tick_uid": self.tick_uid,
            "pass_kind": self.pass_kind.value,
            "entries": [item.to_canonical_dict() for item in self.entries],
        }
        if include_id:
            value["workspace_id"] = self.workspace_id
        if include_rails:
            value["rendered_rails"] = [item.to_canonical_dict() for item in self.rendered_rails]
        return value


class ExactProposalWorkspaceRenderer:
    """Render one exact board identically into every registered rail width."""

    def render(
        self,
        image: FrozenTickImage,
        pass_kind: ProposalPass,
        records: Iterable[ParticipantRecord],
    ) -> ProposalWorkspace:
        if not isinstance(image, FrozenTickImage):
            raise TypeError("proposal workspace rendering requires FrozenTickImage")
        entries = tuple(ProposalWorkspaceEntry.from_record(item) for item in records)
        bare = ProposalWorkspace(
            image_id=image.image_id,
            tick_uid=image.identity.tick_uid,
            pass_kind=pass_kind,
            entries=entries,
        )
        text = bare.readable_text()
        rails = tuple(
            RenderedProposalRail(
                d_model=binding.d_model,
                source_workspace_id=bare.workspace_id,
                text_frame=CategoricalTextFrame.from_text(text, d_model=binding.d_model),
            )
            for binding in image.rails
        )
        return ProposalWorkspace(
            image_id=bare.image_id,
            tick_uid=bare.tick_uid,
            pass_kind=bare.pass_kind,
            entries=bare.entries,
            rendered_rails=rails,
        )


class D64ProposalWorkspaceRenderer(ExactProposalWorkspaceRenderer):
    """First physical renderer; fails closed if the frozen image lacks D64."""

    def render(self, image, pass_kind, records):
        image.require_rail(64)
        return super().render(image, pass_kind, records)


__all__ = [
    "PROPOSAL_RAIL_SCHEMA",
    "PROPOSAL_WORKSPACE_SCHEMA",
    "D64ProposalWorkspaceRenderer",
    "ExactProposalWorkspaceRenderer",
    "ProposalWorkspace",
    "ProposalWorkspaceEntry",
    "RenderedProposalRail",
]
