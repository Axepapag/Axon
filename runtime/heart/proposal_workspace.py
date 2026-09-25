"""Derived, width-generic English proposal-board rail workspaces.

The proposal workspace is deliberately noncanonical.  It carries exact English
FIRST/REFINED proposals plus runtime participant accounting beside a frozen tick
image.  Every destination rail receives the same text through deterministic 16D
transport repacking; no pairwise neural translation and no typed-delta proposal
language is involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from runtime.field import canonical_sha256

from .board import ParticipantRecord, ParticipantState, ProposalPass
from .core_bus import D16TextFrame
from .reasoning_output import CategoricalTextFrame
from .tick import FrozenTickImage

PROPOSAL_WORKSPACE_SCHEMA = "axon-heart-english-proposal-workspace-v3"
PROPOSAL_RAIL_SCHEMA = "axon-heart-english-proposal-rail-v2"


@dataclass(frozen=True, slots=True)
class ProposalWorkspaceEntry:
    core_id: str
    d_model: int
    state: str
    detail: str
    proposal_id: str | None
    text: str | None
    evidence: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, record: ParticipantRecord) -> "ProposalWorkspaceEntry":
        proposal = record.proposal
        return cls(
            core_id=record.core_id,
            d_model=record.d_model,
            state=record.state.value,
            detail=record.detail,
            proposal_id=None if proposal is None else proposal.proposal_id,
            text=None if proposal is None else proposal.text,
            evidence=() if proposal is None else proposal.evidence,
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "core_id": self.core_id,
            "d_model": self.d_model,
            "state": self.state,
            "detail": self.detail,
            "proposal_id": self.proposal_id,
            "text": self.text,
            "evidence": list(self.evidence),
        }

    def readable_text(self) -> str:
        if self.state == ParticipantState.RETURNED.value:
            return f"[{self.core_id}]\n{self.text}"
        detail = f": {self.detail}" if self.detail else ""
        return f"[{self.core_id} — {self.state.upper()}{detail}]"


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
    d16_frame: D16TextFrame | None = field(default=None, compare=False)
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
        for entry in entries:
            if entry.state == ParticipantState.RETURNED.value and (not entry.proposal_id or entry.text is None):
                raise ValueError("returned proposal workspace entry must carry exact English text")
            if entry.state != ParticipantState.RETURNED.value and (entry.proposal_id is not None or entry.text is not None):
                raise ValueError("failed/timed-out proposal workspace entries cannot carry proposal text")
        object.__setattr__(
            self,
            "workspace_id",
            canonical_sha256(self.to_canonical_dict(include_id=False, include_d16=False, include_rails=False)),
        )
        d16_frame = self.d16_frame
        if d16_frame is not None:
            if not isinstance(d16_frame, D16TextFrame):
                raise TypeError("proposal workspace d16_frame must be D16TextFrame")
            if d16_frame.text != self.readable_text():
                raise ValueError("proposal workspace D16 frame does not preserve exact readable text")
        object.__setattr__(self, "d16_frame", d16_frame)
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

    def require_d16_frame(self) -> D16TextFrame:
        frame = self.d16_frame
        if frame is None:
            raise KeyError("proposal workspace has no exact D16 frame")
        return frame

    def readable_text(self) -> str:
        heading = f"{self.pass_kind.value.upper()} PROPOSALS"
        return "\n\n".join((heading, *(item.readable_text() for item in self.entries)))

    def to_canonical_dict(
        self,
        *,
        include_id: bool = True,
        include_d16: bool = True,
        include_rails: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": PROPOSAL_WORKSPACE_SCHEMA,
            "image_id": self.image_id,
            "tick_uid": self.tick_uid,
            "pass_kind": self.pass_kind.value,
            "entries": [item.to_canonical_dict() for item in self.entries],
            "readable_text": self.readable_text(),
        }
        if include_id:
            value["workspace_id"] = self.workspace_id
        if include_d16:
            value["d16_frame"] = None if self.d16_frame is None else self.d16_frame.to_canonical_dict()
        if include_rails:
            value["rendered_rails"] = [item.to_canonical_dict() for item in self.rendered_rails]
        return value


class ExactProposalWorkspaceRenderer:
    """Render one exact English board to D16 once and to any legacy rails still bound."""

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
        d16_frame = D16TextFrame(text)
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
            d16_frame=d16_frame,
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
