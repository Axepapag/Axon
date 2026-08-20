"""Training-side entry points for the canonical D64 compiler contract.

Curriculum records may remain reproducible source material, but before a core
sees them they are materialized as ordinary SharedFieldSnapshot objects and
compiled by the same D64FieldCompiler used by runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from runtime.field import (
    CanonicalStateBranch,
    CompiledD64Field,
    D64FieldCompiler,
    FieldDelta,
    LogicalRegion,
    SharedFieldSnapshot,
    replacement_delta,
)


@dataclass(frozen=True, slots=True)
class CanonicalR0Example:
    example_id: str
    snapshot: SharedFieldSnapshot
    compiled: CompiledD64Field
    scratch_target: str
    response_target: str
    alignment: Mapping[str, Any] | None


def snapshot_from_r0_record(record: Mapping[str, Any]) -> SharedFieldSnapshot:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    field = record.get("field")
    if not isinstance(field, Mapping):
        raise ValueError("R0 record must contain a field mapping")
    example_id = str(record.get("example_id", "unknown-example"))
    source_manifest_ids = [f"r0-example:{example_id}"]
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        refs = provenance.get("evidence_refs", ())
        if isinstance(refs, (list, tuple)):
            source_manifest_ids.extend(str(ref) for ref in refs if str(ref))
    return SharedFieldSnapshot.from_texts(
        {str(name): str(text) for name, text in field.items()},
        tick_id=0,
        source_manifest_ids=tuple(source_manifest_ids),
        source=f"training:{example_id}",
        provenance="canonical_r0_curriculum_materialization",
    )


def canonicalize_r0_record(
    record: Mapping[str, Any],
    *,
    compiler: D64FieldCompiler | None = None,
) -> CanonicalR0Example:
    snapshot = snapshot_from_r0_record(record)
    active_compiler = D64FieldCompiler() if compiler is None else compiler
    compiled = active_compiler.compile(snapshot)
    targets = record.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("R0 record must contain target mapping")
    scratch = targets.get("scratch")
    response = targets.get("response_draft")
    if not isinstance(scratch, str) or not isinstance(response, str):
        raise ValueError("R0 targets must contain text scratch and response_draft")
    alignment = record.get("alignment")
    if alignment is not None and not isinstance(alignment, Mapping):
        raise ValueError("R0 alignment must be a mapping when present")
    return CanonicalR0Example(
        example_id=str(record.get("example_id", "unknown-example")),
        snapshot=snapshot,
        compiled=compiled,
        scratch_target=scratch,
        response_target=response,
        alignment=alignment,
    )


def teacher_region_delta(
    snapshot: SharedFieldSnapshot,
    compiled: CompiledD64Field,
    *,
    region: LogicalRegion | str,
    text: str,
    example_id: str,
    pass_id: str | int,
    evidence: Iterable[str] = (),
) -> FieldDelta:
    return replacement_delta(
        snapshot,
        compiled,
        region=region,
        text=text,
        author_core_id="teacher",
        pass_id=pass_id,
        evidence=evidence,
        provenance=f"training_teacher:{example_id}:{region}",
    )


def training_branch(
    branch_id: str,
    *,
    state_root: Path | str = Path(r"D:\Axon\State"),
) -> CanonicalStateBranch:
    return CanonicalStateBranch.training(branch_id, state_root=state_root)


__all__ = [
    "CanonicalR0Example",
    "snapshot_from_r0_record",
    "canonicalize_r0_record",
    "teacher_region_delta",
    "training_branch",
]
