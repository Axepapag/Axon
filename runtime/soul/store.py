"""Crash-safe content-addressed branches for private per-core souls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from runtime.field import canonical_json_bytes, canonical_sha256

from .contracts import (
    SoulCommitReceipt,
    SoulIntegrityError,
    SoulSnapshot,
    SoulTransition,
    apply_soul_transition,
    empty_soul_layers,
)

SOUL_BRANCH_SCHEMA = "axon-private-soul-branch-v1"
SOUL_HEAD_SCHEMA = "axon-private-soul-head-v1"
SOUL_PREPARED_SCHEMA = "axon-private-soul-prepared-v1"
DEFAULT_SOUL_ROOT = Path(r"D:\Axon\State\active\souls")


def _safe_component(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or any(char in value for char in '\\/:*?"<>|')
    ):
        raise ValueError(f"{label} is not a safe path component")
    return value.strip()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SoulIntegrityError(f"cannot read valid private-soul JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SoulIntegrityError(f"private-soul JSON must be an object: {path}")
    return value


@dataclass(frozen=True, slots=True)
class SoulBranchHead:
    core_id: str
    branch_id: str
    soul_id: str
    generation: int
    parent_soul_id: str | None
    head_id: str

    @classmethod
    def from_snapshot(cls, branch_id: str, snapshot: SoulSnapshot) -> "SoulBranchHead":
        body = {
            "schema": SOUL_HEAD_SCHEMA,
            "core_id": snapshot.core_id,
            "branch_id": branch_id,
            "soul_id": snapshot.soul_id,
            "generation": snapshot.generation,
            "parent_soul_id": snapshot.parent_soul_id,
        }
        return cls(
            core_id=snapshot.core_id,
            branch_id=branch_id,
            soul_id=snapshot.soul_id,
            generation=snapshot.generation,
            parent_soul_id=snapshot.parent_soul_id,
            head_id=canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOUL_HEAD_SCHEMA,
            "core_id": self.core_id,
            "branch_id": self.branch_id,
            "soul_id": self.soul_id,
            "generation": self.generation,
            "parent_soul_id": self.parent_soul_id,
            "head_id": self.head_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SoulBranchHead":
        item = dict(value)
        if set(item) != {
            "schema",
            "core_id",
            "branch_id",
            "soul_id",
            "generation",
            "parent_soul_id",
            "head_id",
        } or item["schema"] != SOUL_HEAD_SCHEMA:
            raise SoulIntegrityError("serialized private-soul HEAD fields are invalid")
        body = dict(item)
        head_id = body.pop("head_id")
        if canonical_sha256(body) != head_id:
            raise SoulIntegrityError("private-soul HEAD identity mismatch")
        return cls(
            core_id=item["core_id"],
            branch_id=item["branch_id"],
            soul_id=item["soul_id"],
            generation=item["generation"],
            parent_soul_id=item["parent_soul_id"],
            head_id=head_id,
        )


class SoulBranch:
    """One live or candidate private-soul lineage for exactly one core."""

    def __init__(self, path: Path | str, *, core_id: str, branch_id: str) -> None:
        self.path = Path(path).resolve()
        self.core_id = _safe_component(core_id, "core_id")
        self.branch_id = _safe_component(branch_id, "branch_id")
        self.snapshots_dir = self.path / "snapshots"
        self.transitions_dir = self.path / "transitions"
        self.prepared_dir = self.path / "prepared"
        self.receipts_dir = self.path / "receipts"
        self.head_path = self.path / "HEAD.json"
        self.branch_path = self.path / "branch.json"
        self.journal_path = self.path / "journal.jsonl"

    @property
    def initialized(self) -> bool:
        return self.head_path.is_file() and self.branch_path.is_file()

    def initialize(
        self,
        *,
        architecture_id: str,
        parameter_generation: str,
        snapshot: SoulSnapshot | None = None,
        forked_from: Mapping[str, Any] | None = None,
    ) -> SoulSnapshot:
        if self.initialized:
            existing = self.load_head()
            if (
                existing.architecture_id != architecture_id
                or existing.parameter_generation != parameter_generation
            ):
                raise SoulIntegrityError("existing soul branch architecture/generation mismatch")
            return existing
        if snapshot is None:
            snapshot = SoulSnapshot(
                core_id=self.core_id,
                architecture_id=architecture_id,
                parameter_generation=parameter_generation,
                generation=0,
                parent_soul_id=None,
                layers=empty_soul_layers(),
            )
        elif (
            snapshot.core_id != self.core_id
            or snapshot.architecture_id != architecture_id
            or snapshot.parameter_generation != parameter_generation
        ):
            raise SoulIntegrityError("fork snapshot does not match target branch identity")
        self._persist_snapshot(snapshot)
        descriptor = {
            "schema": SOUL_BRANCH_SCHEMA,
            "core_id": self.core_id,
            "branch_id": self.branch_id,
            "architecture_id": architecture_id,
            "parameter_generation": parameter_generation,
            "forked_from": None if forked_from is None else dict(forked_from),
        }
        descriptor["branch_identity"] = canonical_sha256(descriptor)
        _atomic_json(self.branch_path, descriptor)
        _atomic_json(self.head_path, SoulBranchHead.from_snapshot(self.branch_id, snapshot).to_dict())
        self._append_event(
            {
                "event": "initialized" if forked_from is None else "forked",
                "soul_id": snapshot.soul_id,
                "generation": snapshot.generation,
                "forked_from": None if forked_from is None else dict(forked_from),
            }
        )
        return snapshot

    def load_head_record(self) -> SoulBranchHead:
        head = SoulBranchHead.from_mapping(_read_json(self.head_path))
        if head.core_id != self.core_id or head.branch_id != self.branch_id:
            raise SoulIntegrityError("private-soul HEAD belongs to another branch")
        return head

    def load_head(self) -> SoulSnapshot:
        head = self.load_head_record()
        snapshot = self.load_snapshot(head.soul_id)
        if snapshot.generation != head.generation or snapshot.parent_soul_id != head.parent_soul_id:
            raise SoulIntegrityError("private-soul HEAD metadata disagrees with its snapshot")
        return snapshot

    def load_snapshot(self, soul_id: str) -> SoulSnapshot:
        snapshot = SoulSnapshot.from_mapping(_read_json(self.snapshots_dir / f"{soul_id}.json"))
        if snapshot.soul_id != soul_id or snapshot.core_id != self.core_id:
            raise SoulIntegrityError("private-soul snapshot path/identity mismatch")
        return snapshot

    def load_transition(self, transition_id: str) -> SoulTransition:
        transition = SoulTransition.from_mapping(
            _read_json(self.transitions_dir / f"{transition_id}.json")
        )
        if transition.transition_id != transition_id or transition.core_id != self.core_id:
            raise SoulIntegrityError("private-soul transition path/identity mismatch")
        return transition

    def prepare_transition(
        self,
        transition: SoulTransition,
        *,
        requires_external_commit: bool = False,
    ) -> SoulSnapshot:
        before = self.load_head()
        after = apply_soul_transition(before, transition)
        self._persist_transition(transition)
        self._persist_snapshot(after)
        prepared = {
            "schema": SOUL_PREPARED_SCHEMA,
            "core_id": self.core_id,
            "branch_id": self.branch_id,
            "transition_id": transition.transition_id,
            "before_soul_id": before.soul_id,
            "after_soul_id": after.soul_id,
            "after_generation": after.generation,
            "requires_external_commit": bool(requires_external_commit),
            "automatic_commit_binding": f"reasoning-request:{transition.request_id}",
        }
        prepared["prepared_id"] = canonical_sha256(prepared)
        path = self.prepared_dir / f"{transition.transition_id}.json"
        if path.exists():
            if canonical_json_bytes(_read_json(path)) != canonical_json_bytes(prepared):
                raise SoulIntegrityError("prepared soul transition identity collision")
        else:
            _atomic_json(path, prepared)
        return after

    def finalize_transition(
        self,
        transition_id: str,
        *,
        commit_binding: str,
    ) -> SoulCommitReceipt:
        prepared = self._load_prepared(transition_id)
        transition = self.load_transition(transition_id)
        before = self.load_snapshot(prepared["before_soul_id"])
        after = self.load_snapshot(prepared["after_soul_id"])
        head = self.load_head()
        if head.soul_id == after.soul_id:
            return self.load_receipt(transition_id)
        if head.soul_id != before.soul_id:
            raise SoulIntegrityError("private-soul HEAD moved before prepared transition finalized")
        receipt = SoulCommitReceipt(
            core_id=self.core_id,
            branch_id=self.branch_id,
            transition_id=transition_id,
            before_soul_id=before.soul_id,
            after_soul_id=after.soul_id,
            generation=after.generation,
            phase=transition.phase,
            tick_uid=transition.tick_uid,
            request_id=transition.request_id,
            commit_binding=commit_binding,
        )
        receipt_path = self.receipts_dir / f"{transition_id}.json"
        if receipt_path.exists():
            existing = SoulCommitReceipt.from_mapping(_read_json(receipt_path))
            if existing.receipt_id != receipt.receipt_id:
                raise SoulIntegrityError("soul receipt disagrees with requested commit binding")
        else:
            _atomic_json(receipt_path, receipt.to_canonical_dict())
        _atomic_json(self.head_path, SoulBranchHead.from_snapshot(self.branch_id, after).to_dict())
        self._append_event({"event": "committed", **receipt.to_canonical_dict()})
        return receipt

    def commit_transition(
        self,
        transition: SoulTransition,
        *,
        commit_binding: str | None = None,
    ) -> SoulCommitReceipt:
        self.prepare_transition(transition)
        return self.finalize_transition(
            transition.transition_id,
            commit_binding=commit_binding or f"reasoning-request:{transition.request_id}",
        )

    def recover(
        self,
        *,
        external_commit_bindings: Mapping[str, str] | None = None,
    ) -> tuple[SoulCommitReceipt, ...]:
        """Finish durable prepared work when its commit condition is proven."""

        self.load_head()
        bindings = {} if external_commit_bindings is None else dict(external_commit_bindings)
        recovered: list[SoulCommitReceipt] = []
        for path in sorted(self.prepared_dir.glob("*.json")):
            prepared = self._load_prepared(path.stem)
            after_id = prepared["after_soul_id"]
            head = self.load_head()
            if head.soul_id == after_id:
                continue
            receipt_path = self.receipts_dir / f"{path.stem}.json"
            if receipt_path.exists():
                receipt = SoulCommitReceipt.from_mapping(_read_json(receipt_path))
                if (
                    receipt.before_soul_id != prepared["before_soul_id"]
                    or receipt.after_soul_id != prepared["after_soul_id"]
                ):
                    raise SoulIntegrityError("soul receipt disagrees with its prepared transition")
                if head.soul_id != receipt.after_soul_id:
                    # A prepared record is permanent evidence.  Once later
                    # transitions have advanced HEAD, an older committed
                    # record is historical, not pending recovery.
                    self.receipts_after(receipt.after_soul_id)
                    continue
                recovered.append(
                    self.finalize_transition(path.stem, commit_binding=receipt.commit_binding)
                )
                continue
            if prepared["requires_external_commit"]:
                binding = bindings.get(path.stem)
                if binding is None:
                    continue
            else:
                binding = prepared["automatic_commit_binding"]
            recovered.append(self.finalize_transition(path.stem, commit_binding=binding))
        return tuple(recovered)

    def load_receipt(self, transition_id: str) -> SoulCommitReceipt:
        receipt = SoulCommitReceipt.from_mapping(
            _read_json(self.receipts_dir / f"{transition_id}.json")
        )
        if receipt.core_id != self.core_id or receipt.branch_id != self.branch_id:
            raise SoulIntegrityError("soul receipt belongs to another branch")
        return receipt

    def receipts_after(self, soul_id: str) -> tuple[SoulCommitReceipt, ...]:
        """Return the exact linear receipt chain after a known ancestor."""

        receipts = {
            receipt.before_soul_id: receipt
            for receipt in (
                SoulCommitReceipt.from_mapping(_read_json(path))
                for path in self.receipts_dir.glob("*.json")
            )
        }
        chain: list[SoulCommitReceipt] = []
        cursor = soul_id
        while cursor in receipts:
            receipt = receipts[cursor]
            chain.append(receipt)
            cursor = receipt.after_soul_id
        if cursor != self.load_head().soul_id:
            raise SoulIntegrityError("requested soul is not an ancestor of branch HEAD")
        return tuple(chain)

    def fork_to(self, target: "SoulBranch") -> SoulSnapshot:
        source = self.load_head()
        return target.initialize(
            architecture_id=source.architecture_id,
            parameter_generation=source.parameter_generation,
            snapshot=source,
            forked_from={
                "core_id": self.core_id,
                "branch_id": self.branch_id,
                "soul_id": source.soul_id,
                "generation": source.generation,
            },
        )

    def _persist_snapshot(self, snapshot: SoulSnapshot) -> None:
        path = self.snapshots_dir / f"{snapshot.soul_id}.json"
        value = snapshot.to_canonical_dict()
        if path.exists():
            if canonical_json_bytes(_read_json(path)) != canonical_json_bytes(value):
                raise SoulIntegrityError("content-addressed soul snapshot collision")
        else:
            _atomic_json(path, value)

    def _persist_transition(self, transition: SoulTransition) -> None:
        path = self.transitions_dir / f"{transition.transition_id}.json"
        value = transition.to_canonical_dict()
        if path.exists():
            if canonical_json_bytes(_read_json(path)) != canonical_json_bytes(value):
                raise SoulIntegrityError("content-addressed soul transition collision")
        else:
            _atomic_json(path, value)

    def _load_prepared(self, transition_id: str) -> dict[str, Any]:
        value = _read_json(self.prepared_dir / f"{transition_id}.json")
        prepared_id = value.pop("prepared_id", None)
        if value.get("schema") != SOUL_PREPARED_SCHEMA or canonical_sha256(value) != prepared_id:
            raise SoulIntegrityError("prepared soul transition identity mismatch")
        value["prepared_id"] = prepared_id
        if value.get("transition_id") != transition_id:
            raise SoulIntegrityError("prepared soul transition path mismatch")
        return value

    def _append_event(self, value: Mapping[str, Any]) -> None:
        event = {
            "schema": "axon-private-soul-branch-event-v1",
            "core_id": self.core_id,
            "branch_id": self.branch_id,
            **dict(value),
        }
        self.path.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class SoulStore:
    """Registry of live private-soul branches beneath canonical State."""

    def __init__(self, root: Path | str = DEFAULT_SOUL_ROOT) -> None:
        self.root = Path(root).resolve()

    @classmethod
    def active(cls, state_root: Path | str) -> "SoulStore":
        return cls(Path(state_root).resolve() / "active" / "souls")

    def branch(self, core_id: str, branch_id: str = "live") -> SoulBranch:
        safe_core = _safe_component(core_id, "core_id")
        safe_branch = _safe_component(branch_id, "branch_id")
        return SoulBranch(
            self.root / safe_core / "branches" / safe_branch,
            core_id=safe_core,
            branch_id=safe_branch,
        )

    def ensure_core(
        self,
        *,
        core_id: str,
        architecture_id: str,
        parameter_generation: str,
    ) -> SoulSnapshot:
        return self.branch(core_id).initialize(
            architecture_id=architecture_id,
            parameter_generation=parameter_generation,
        )

    def load_snapshot(self, core_id: str, soul_id: str, branch_id: str = "live") -> SoulSnapshot:
        return self.branch(core_id, branch_id).load_snapshot(soul_id)


__all__ = [
    "DEFAULT_SOUL_ROOT",
    "SOUL_BRANCH_SCHEMA",
    "SOUL_HEAD_SCHEMA",
    "SOUL_PREPARED_SCHEMA",
    "SoulBranch",
    "SoulBranchHead",
    "SoulStore",
]
