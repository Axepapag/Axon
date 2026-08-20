"""Bounded, crash-replayable dormant-state work for rotating sleeper cores.

The idle worker never shortens the canonical shared field.  It captures exact
whole spans selected for masking, queues deterministic extraction jobs, and
processes only a configured quantum.  Bootstrap triples remain *candidates*;
the runtime must not silently promote a syntactic match into accepted truth.
"""
from __future__ import annotations

from dataclasses import dataclass

from runtime.field import LogicalRegion, SharedFieldSnapshot

from .dormant import (
    BOOTSTRAP_EXTRACTOR_ID,
    BOOTSTRAP_RULESET_SHA256,
    DormantStore,
    SourceRecord,
    bootstrap_extract,
)
from .projection import MaskedSpan, ProjectionResult


class IdleWorkerError(RuntimeError):
    """A bounded dormant-state quantum could not preserve its contract."""


@dataclass(frozen=True, slots=True)
class IdleQuantumResult:
    owner_core_id: str
    tick_id: int
    captured_source_ids: tuple[str, ...]
    leased_job_keys: tuple[str, ...]
    completed_job_keys: tuple[str, ...]
    candidate_revision_ids: tuple[str, ...]
    dormant_span_ids: tuple[str, ...]


def _stream_id(region: LogicalRegion, span_id: str) -> str:
    return f"field-span/{region.value}/{span_id}"


def _stream_span_id(stream_id: str) -> str:
    prefix = "field-span/"
    if not stream_id.startswith(prefix):
        raise IdleWorkerError("source stream is not a captured field span")
    remainder = stream_id[len(prefix) :]
    separator = remainder.find("/")
    if separator <= 0 or separator == len(remainder) - 1:
        raise IdleWorkerError("captured field-span stream is malformed")
    try:
        LogicalRegion(remainder[:separator])
    except ValueError as exc:
        raise IdleWorkerError("captured field-span region is invalid") from exc
    return remainder[separator + 1 :]


class DormantIdleWorker:
    """Capture masked spans and process a finite extraction work quantum."""

    def __init__(self, store: DormantStore) -> None:
        if not isinstance(store, DormantStore):
            raise TypeError("store must be DormantStore")
        self.store = store

    def capture_projection_omissions(
        self,
        projection: ProjectionResult,
        *,
        tick_id: int,
        limit: int,
    ) -> tuple[str, ...]:
        if not isinstance(projection, ProjectionResult):
            raise TypeError("projection must be ProjectionResult")
        if isinstance(tick_id, bool) or not isinstance(tick_id, int) or tick_id < 0:
            raise ValueError("tick_id must be non-negative")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be positive")

        full = projection.full_snapshot
        by_key = {
            (region.name, span.span_id): span
            for region in full.regions
            for span in region.spans
        }
        captured: list[str] = []
        for omission in projection.manifest.masked:
            if len(captured) >= limit:
                break
            if omission.lifecycle_state == "dormant":
                continue
            try:
                span = by_key[(omission.logical_region, omission.span_id)]
            except KeyError as exc:
                raise IdleWorkerError(
                    "projection omission has no exact source span"
                ) from exc
            stream_id = _stream_id(
                omission.logical_region,
                omission.span_id,
            )
            record = self.store.source_at(stream_id, 0)
            if record is None:
                record = SourceRecord.from_field_span(
                    span,
                    stream_id=stream_id,
                    sequence=0,
                    region=omission.logical_region,
                    tick_id=tick_id,
                )
                self.store.append_source(record)
            elif (
                record.region is not omission.logical_region
                or record.exact_text != span.text
                or record.source != (span.source or "shared_field")
                or record.provenance
                != (span.provenance or f"field_span:{span.span_id}")
            ):
                raise IdleWorkerError(
                    "captured field span conflicts with its exact source position"
                )
            latest = self.store.latest_span_lifecycle(omission.span_id)
            if latest is None:
                latest = self.store.transition_span(
                    span_id=omission.span_id,
                    source_record_id=record.record_id,
                    state="active",
                    reason="exact field span captured",
                    tick_id=tick_id,
                )
            if latest.source_record_id != record.record_id:
                raise IdleWorkerError(
                    "span lifecycle points to a different exact source"
                )
            if latest.state in {"active", "resurfaced"}:
                latest = self.store.transition_span(
                    span_id=omission.span_id,
                    source_record_id=record.record_id,
                    state="masked",
                    reason=f"active projection omitted: {omission.reason}",
                    tick_id=tick_id,
                )
            if latest.state == "masked":
                self.store.enqueue_extraction(record.record_id)
            captured.append(record.record_id)
        return tuple(captured)

    def process_quantum(
        self,
        *,
        owner_core_id: str,
        now_tick: int,
        job_limit: int,
        lease_ticks: int = 2,
    ) -> IdleQuantumResult:
        if not isinstance(owner_core_id, str) or not owner_core_id:
            raise ValueError("owner_core_id must be non-empty")
        if (
            isinstance(now_tick, bool)
            or not isinstance(now_tick, int)
            or now_tick < 0
        ):
            raise ValueError("now_tick must be non-negative")
        if (
            isinstance(job_limit, bool)
            or not isinstance(job_limit, int)
            or job_limit <= 0
        ):
            raise ValueError("job_limit must be positive")
        if (
            isinstance(lease_ticks, bool)
            or not isinstance(lease_ticks, int)
            or lease_ticks <= 0
        ):
            raise ValueError("lease_ticks must be positive")

        leased = self.store.lease_jobs(
            owner=owner_core_id,
            now_tick=now_tick,
            lease_ticks=lease_ticks,
            limit=job_limit,
        )
        completed: list[str] = []
        candidates: list[str] = []
        dormant_spans: list[str] = []
        for job in leased:
            if (
                job.extractor_id != BOOTSTRAP_EXTRACTOR_ID
                or job.ruleset_sha256 != BOOTSTRAP_RULESET_SHA256
            ):
                raise IdleWorkerError(
                    "idle worker received an unregistered extractor contract"
                )
            source = self.store.get_source(job.source_record_id)
            bundle = bootstrap_extract(source)
            self.store.append_extraction_bundle(bundle)
            for revision_id in bundle.output_revision_ids:
                if self.store.latest_knowledge(revision_id) is None:
                    self.store.transition_knowledge(
                        target_kind=(
                            "container"
                            if revision_id.startswith("containerrev-")
                            else (
                                "entity"
                                if revision_id.startswith("entityrev-")
                                else "triple"
                            )
                        ),
                        target_revision_id=revision_id,
                        state="candidate",
                        reason=(
                            "deterministic bootstrap extraction; "
                            "not semantic proof"
                        ),
                        tick_id=now_tick,
                    )
                candidates.append(revision_id)
            self.store.complete_job(
                job.job_key,
                owner=owner_core_id,
                output_revision_ids=bundle.output_revision_ids,
            )
            completed.append(job.job_key)

            span_id = _stream_span_id(source.stream_id)
            latest = self.store.latest_span_lifecycle(span_id)
            if latest is not None and latest.state == "masked":
                self.store.transition_span(
                    span_id=span_id,
                    source_record_id=source.record_id,
                    state="dormant",
                    reason="bounded idle extraction completed",
                    tick_id=now_tick,
                )
                dormant_spans.append(span_id)

        return IdleQuantumResult(
            owner_core_id=owner_core_id,
            tick_id=now_tick,
            captured_source_ids=(),
            leased_job_keys=tuple(job.job_key for job in leased),
            completed_job_keys=tuple(completed),
            candidate_revision_ids=tuple(sorted(set(candidates))),
            dormant_span_ids=tuple(sorted(set(dormant_spans))),
        )

    def run_projection_quantum(
        self,
        projection: ProjectionResult,
        *,
        owner_core_id: str,
        now_tick: int,
        capture_limit: int,
        job_limit: int,
        lease_ticks: int = 2,
    ) -> IdleQuantumResult:
        captured = self.capture_projection_omissions(
            projection,
            tick_id=now_tick,
            limit=capture_limit,
        )
        processed = self.process_quantum(
            owner_core_id=owner_core_id,
            now_tick=now_tick,
            job_limit=job_limit,
            lease_ticks=lease_ticks,
        )
        return IdleQuantumResult(
            owner_core_id=processed.owner_core_id,
            tick_id=processed.tick_id,
            captured_source_ids=captured,
            leased_job_keys=processed.leased_job_keys,
            completed_job_keys=processed.completed_job_keys,
            candidate_revision_ids=processed.candidate_revision_ids,
            dormant_span_ids=processed.dormant_span_ids,
        )


class DormantPostCommitHook:
    """Run one bounded dormant quantum under the tick's rotating sleeper."""

    def __init__(
        self,
        worker: DormantIdleWorker,
        *,
        policy: "ProjectionPolicy | None" = None,
        capture_limit: int = 4,
        job_limit: int = 2,
        lease_ticks: int = 2,
    ) -> None:
        # Local import keeps the idle worker usable without projection policy
        # construction at module import time.
        from .projection import ProjectionPolicy

        if not isinstance(worker, DormantIdleWorker):
            raise TypeError("worker must be DormantIdleWorker")
        self.worker = worker
        self.policy = ProjectionPolicy() if policy is None else policy
        if not isinstance(self.policy, ProjectionPolicy):
            raise TypeError("policy must be ProjectionPolicy")
        for name, value in (
            ("capture_limit", capture_limit),
            ("job_limit", job_limit),
            ("lease_ticks", lease_ticks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.capture_limit = capture_limit
        self.job_limit = job_limit
        self.lease_ticks = lease_ticks
        self.last_result: IdleQuantumResult | None = None

    def __call__(
        self,
        runtime_result: object,
        snapshot: SharedFieldSnapshot,
        sleeper_core_id: str,
    ) -> None:
        from .projection import build_active_projection

        tick_id = getattr(getattr(runtime_result, "commit", None), "tick_seq", None)
        if isinstance(tick_id, bool) or not isinstance(tick_id, int):
            raise IdleWorkerError("post-commit result has no valid tick sequence")
        projection = build_active_projection(
            snapshot,
            policy=self.policy,
            store=self.worker.store,
        )
        self.last_result = self.worker.run_projection_quantum(
            projection,
            owner_core_id=sleeper_core_id,
            now_tick=tick_id + 1,
            capture_limit=self.capture_limit,
            job_limit=self.job_limit,
            lease_ticks=self.lease_ticks,
        )


__all__ = [
    "IdleWorkerError",
    "IdleQuantumResult",
    "DormantIdleWorker",
    "DormantPostCommitHook",
]
