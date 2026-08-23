"""Runtime-facing wrapper around the canonical deterministic D64 compiler.

The adapter intentionally contains no model and no commit authority.  It gives
runtime orchestration the same exact compiler/delta contract used by training
while the future D64 neural driver is developed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from runtime.field import (
    CanonicalStateBranch,
    CompiledD64DualSurface,
    CompiledD64Field,
    D64FieldCompiler,
    D64SemanticSurfaceCompiler,
    FieldDelta,
    LogicalRegion,
    SharedFieldSnapshot,
    apply_compiled_delta,
    replacement_delta,
)


@dataclass(slots=True)
class CanonicalD64RuntimeAdapter:
    compiler: D64FieldCompiler = D64FieldCompiler()
    semantic_compiler: D64SemanticSurfaceCompiler = D64SemanticSurfaceCompiler()
    branch: CanonicalStateBranch | None = None

    def compile(self, snapshot: SharedFieldSnapshot) -> CompiledD64Field:
        return self.compiler.compile(snapshot)

    def compile_dual(self, snapshot: SharedFieldSnapshot) -> CompiledD64DualSurface:
        """Compile exact D64 plus deterministic grounded first-form semantic slots."""

        exact = self.compiler.compile(snapshot)
        return self.semantic_compiler.compile_dual(snapshot, exact)

    def propose_region_replacement(
        self,
        snapshot: SharedFieldSnapshot,
        compiled: CompiledD64Field,
        *,
        region: LogicalRegion | str,
        text: str,
        author_core_id: str,
        pass_id: str | int,
        evidence: Iterable[str] = (),
        provenance: str = "runtime_d64_exact_replace",
        container_refs: Iterable[str] = (),
        edge_refs: Iterable[str] = (),
    ) -> FieldDelta:
        return replacement_delta(
            snapshot,
            compiled,
            region=region,
            text=text,
            author_core_id=author_core_id,
            pass_id=pass_id,
            evidence=evidence,
            provenance=provenance,
            container_refs=container_refs,
            edge_refs=edge_refs,
        )

    def apply_candidate(
        self,
        snapshot: SharedFieldSnapshot,
        compiled: CompiledD64Field,
        delta: FieldDelta,
    ) -> SharedFieldSnapshot:
        """Validate/apply in memory only; this does not grant commit authority."""

        return apply_compiled_delta(snapshot, compiled, delta)

    def commit_to_branch(
        self,
        snapshot: SharedFieldSnapshot,
        compiled: CompiledD64Field,
        delta: FieldDelta,
    ) -> SharedFieldSnapshot:
        """Persist through an explicitly supplied canonical branch.

        This method exists for controlled runtime/training harnesses.  Production
        consolidation still decides whether a delta may be committed.
        """

        if self.branch is None:
            raise RuntimeError("no canonical state branch is bound to this adapter")
        compiled.assert_fresh(snapshot)
        head = self.branch.load_head()
        if head.field_id != snapshot.field_id:
            raise RuntimeError("canonical branch HEAD does not match supplied snapshot")
        return self.branch.commit(delta)


__all__ = ["CanonicalD64RuntimeAdapter"]
