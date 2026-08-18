"""Bounded multi-tick refinement over canonical shared-field snapshots.

This is intentionally additive: the legacy one-shot runtime remains available
as a control path.  A refiner tick always materializes the current committed
snapshot, obtains one typed proposal, validates/commits it, and only then
materializes the next snapshot.  Free-running mode never substitutes a gold
draft for the core's own committed text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Mapping, Protocol, Sequence

from runtime.field import (
    CORE_WRITABLE_REGIONS,
    FieldDelta,
    FieldView,
    LogicalRegion,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    compile_field_view,
    replay_deltas,
)


class MultiTickContractError(RuntimeError):
    """A proposal, schedule, or commit violated the multi-tick contract."""


@dataclass(frozen=True, slots=True)
class RegionProposal:
    """One complete proposed snapshot for one writable logical region."""

    target_region: LogicalRegion | str
    text: str
    evidence_refs: tuple[str, ...] = ()
    provenance: str = ""

    def __post_init__(self) -> None:
        try:
            region = (
                self.target_region
                if isinstance(self.target_region, LogicalRegion)
                else LogicalRegion(self.target_region)
            )
        except (TypeError, ValueError) as exc:
            raise MultiTickContractError(
                f"unknown proposal region {self.target_region!r}"
            ) from exc
        if region not in CORE_WRITABLE_REGIONS:
            raise MultiTickContractError(
                f"proposal region {region.value!r} is sealed"
            )
        if not isinstance(self.text, str):
            raise TypeError("RegionProposal.text must be a string")
        object.__setattr__(self, "target_region", region)
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(sorted(set(str(ref) for ref in self.evidence_refs))),
        )


class Proposer(Protocol):
    def __call__(
        self,
        *,
        snapshot: SharedFieldSnapshot,
        view: FieldView,
        tick_index: int,
        target_region: LogicalRegion,
    ) -> RegionProposal:
        """Return one proposal without mutating ``snapshot`` or ``view``."""


@dataclass(frozen=True, slots=True)
class TickRecord:
    tick_index: int
    pass_id: str
    target_region: LogicalRegion
    input_field_id: str
    input_view_hash: str
    prior_text: str
    proposed_text: str
    committed_text: str
    teacher_forced: bool
    no_op: bool
    delta_id: str | None
    output_field_id: str
    evidence_refs: tuple[str, ...]
    reference_edit_distance: int | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["target_region"] = self.target_region.value
        return value


@dataclass(frozen=True, slots=True)
class MultiTickResult:
    initial_snapshot: SharedFieldSnapshot
    final_snapshot: SharedFieldSnapshot
    records: tuple[TickRecord, ...]
    deltas: tuple[FieldDelta, ...]
    free_running: bool
    stopped_stable: bool

    @property
    def run_hash(self) -> str:
        payload = {
            "initial_field_id": self.initial_snapshot.field_id,
            "final_field_id": self.final_snapshot.field_id,
            "records": [record.to_dict() for record in self.records],
            "deltas": [delta.to_canonical_dict() for delta in self.deltas],
            "free_running": self.free_running,
            "stopped_stable": self.stopped_stable,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def replay(self) -> SharedFieldSnapshot:
        """Replay committed deltas from the exact initial snapshot."""

        return replay_deltas(self.initial_snapshot, self.deltas)


TargetSchedule = (
    Sequence[LogicalRegion | str]
    | Callable[[int, SharedFieldSnapshot], LogicalRegion | str]
)
OutcomeHook = Callable[[SharedFieldSnapshot, TickRecord], None]
ViewProvider = Callable[
    [SharedFieldSnapshot, LogicalRegion, int],
    FieldView,
]


def character_edit_distance(left: str, right: str) -> int:
    """Deterministic Levenshtein distance for refinement reporting."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _resolve_target(
    schedule: TargetSchedule,
    tick_index: int,
    snapshot: SharedFieldSnapshot,
) -> LogicalRegion:
    if callable(schedule):
        raw = schedule(tick_index, snapshot)
    else:
        if tick_index >= len(schedule):
            raise MultiTickContractError(
                "target schedule is shorter than the requested tick count"
            )
        raw = schedule[tick_index]
    try:
        target = raw if isinstance(raw, LogicalRegion) else LogicalRegion(raw)
    except (TypeError, ValueError) as exc:
        raise MultiTickContractError(f"unknown scheduled region {raw!r}") from exc
    if target not in CORE_WRITABLE_REGIONS:
        raise MultiTickContractError(
            f"scheduled proposal region {target.value!r} is sealed"
        )
    return target


def _full_replace_delta(
    *,
    snapshot: SharedFieldSnapshot,
    proposal: RegionProposal,
    committed_text: str,
    author_core_id: str,
    pass_id: str,
) -> FieldDelta:
    prior = snapshot.region(proposal.target_region).text
    return FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=snapshot.tick_id,
        author_core_id=author_core_id,
        pass_id=pass_id,
        operations=(
            ReplaceText(
                region=proposal.target_region,
                start=0,
                end=len(prior),
                text=committed_text,
                provenance=(
                    proposal.provenance
                    or f"multi_tick:{author_core_id}:{pass_id}"
                ),
                edge_refs=proposal.evidence_refs,
            ),
        ),
        evidence=proposal.evidence_refs,
    )


def run_multi_tick_refinement(
    *,
    initial_snapshot: SharedFieldSnapshot,
    proposer: Proposer,
    target_schedule: TargetSchedule,
    author_core_id: str,
    max_ticks: int,
    min_ticks: int = 1,
    stable_ticks: int = 1,
    teacher_targets: Mapping[int, str] | None = None,
    reference_targets: Mapping[int, str] | None = None,
    on_outcome: OutcomeHook | None = None,
    view_provider: ViewProvider | None = None,
) -> MultiTickResult:
    """Run a bounded canonical commit/materialize/refine transaction.

    ``teacher_targets`` is an explicit training-only mechanism.  When omitted,
    the rollout is free-running and every next tick receives only the model's
    actually committed proposal.  The proposer is always called, including in
    teacher-forced mode, so proposal quality can be measured separately.
    """

    if not isinstance(initial_snapshot, SharedFieldSnapshot):
        raise TypeError("initial_snapshot must be SharedFieldSnapshot")
    if not isinstance(author_core_id, str) or not author_core_id:
        raise ValueError("author_core_id must be non-empty")
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    if min_ticks <= 0 or min_ticks > max_ticks:
        raise ValueError("min_ticks must be in [1, max_ticks]")
    if stable_ticks <= 0:
        raise ValueError("stable_ticks must be positive")
    if teacher_targets is not None:
        unknown = [
            index
            for index in teacher_targets
            if not isinstance(index, int) or index < 0 or index >= max_ticks
        ]
        if unknown:
            raise ValueError(f"invalid teacher target tick indexes: {unknown}")
        if not all(isinstance(text, str) for text in teacher_targets.values()):
            raise TypeError("teacher target values must be strings")

    current = initial_snapshot
    records: list[TickRecord] = []
    deltas: list[FieldDelta] = []
    stable_count = 0
    stopped_stable = False

    for tick_index in range(max_ticks):
        target = _resolve_target(target_schedule, tick_index, current)
        if view_provider is None:
            view = compile_field_view(current, proposal_region=target)
        else:
            try:
                view = view_provider(current, target, tick_index)
            except Exception as exc:
                raise MultiTickContractError(
                    f"tick {tick_index} view provider failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(view, FieldView):
                raise MultiTickContractError(
                    f"tick {tick_index} view provider returned "
                    f"{type(view).__name__}, expected FieldView"
                )
            if view.source_field_id != current.field_id:
                raise MultiTickContractError(
                    f"tick {tick_index} view belongs to a different field"
                )
            if view.proposal_region is not target:
                raise MultiTickContractError(
                    f"tick {tick_index} view targets "
                    f"{view.proposal_region.value!r}, expected {target.value!r}"
                )
        input_field_id = current.field_id
        input_view_hash = view.view_hash
        prior_text = current.region(target).text

        try:
            proposal = proposer(
                snapshot=current,
                view=view,
                tick_index=tick_index,
                target_region=target,
            )
        except Exception as exc:
            raise MultiTickContractError(
                f"tick {tick_index} proposer failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(proposal, RegionProposal):
            raise MultiTickContractError(
                f"tick {tick_index} proposer returned {type(proposal).__name__}, "
                "expected RegionProposal"
            )
        if proposal.target_region is not target:
            raise MultiTickContractError(
                f"tick {tick_index} proposed {proposal.target_region.value!r}, "
                f"scheduled {target.value!r}"
            )

        teacher_forced = teacher_targets is not None and tick_index in teacher_targets
        committed_text = (
            teacher_targets[tick_index]
            if teacher_forced and teacher_targets is not None
            else proposal.text
        )
        no_op = committed_text == prior_text
        pass_id = f"tick-{tick_index:04d}"
        delta: FieldDelta | None = None

        # An empty->empty no-op cannot be represented as a non-empty typed
        # operation.  It is still an accepted, auditable outcome.
        if no_op:
            next_snapshot = current
            stable_count += 1
        else:
            try:
                delta = _full_replace_delta(
                    snapshot=current,
                    proposal=proposal,
                    committed_text=committed_text,
                    author_core_id=author_core_id,
                    pass_id=pass_id,
                )
                next_snapshot = apply_delta(current, delta)
            except Exception as exc:
                raise MultiTickContractError(
                    f"tick {tick_index} commit failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            deltas.append(delta)
            stable_count = 0

        reference = (
            None
            if reference_targets is None
            else reference_targets.get(tick_index)
        )
        if reference is not None and not isinstance(reference, str):
            raise TypeError("reference target values must be strings")
        record = TickRecord(
            tick_index=tick_index,
            pass_id=pass_id,
            target_region=target,
            input_field_id=input_field_id,
            input_view_hash=input_view_hash,
            prior_text=prior_text,
            proposed_text=proposal.text,
            committed_text=committed_text,
            teacher_forced=teacher_forced,
            no_op=no_op,
            delta_id=None if delta is None else delta.delta_id,
            output_field_id=next_snapshot.field_id,
            evidence_refs=proposal.evidence_refs,
            reference_edit_distance=(
                None
                if reference is None
                else character_edit_distance(committed_text, reference)
            ),
        )
        records.append(record)
        current = next_snapshot
        if on_outcome is not None:
            on_outcome(current, record)

        if len(records) >= min_ticks and stable_count >= stable_ticks:
            stopped_stable = True
            break

    result = MultiTickResult(
        initial_snapshot=initial_snapshot,
        final_snapshot=current,
        records=tuple(records),
        deltas=tuple(deltas),
        free_running=teacher_targets is None,
        stopped_stable=stopped_stable,
    )
    replayed = result.replay()
    if replayed.field_id != current.field_id:
        raise MultiTickContractError(
            "internal replay gate failed: committed deltas do not reproduce "
            "the final field hash"
        )
    return result


__all__ = [
    "MultiTickContractError",
    "RegionProposal",
    "Proposer",
    "TickRecord",
    "MultiTickResult",
    "TargetSchedule",
    "OutcomeHook",
    "ViewProvider",
    "character_edit_distance",
    "run_multi_tick_refinement",
]
