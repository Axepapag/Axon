"""Durable transparent state for Axon's Trainer control plane."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from runtime.field import canonical_json_bytes

from .authority import AuthorizedParameterMutation
from .contracts import ParameterInventory, ParameterMutationPlan, ParameterPromotionProposal
from .telemetry import ParameterTelemetryFrame


class TrainerStoreError(RuntimeError):
    pass


class TrainerStateStore:
    """Immutable lineage artifacts plus append-only live parameter telemetry."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve(strict=False)
        self.inventories_dir = self.root / "inventories"
        self.plans_dir = self.root / "plans"
        self.authorizations_dir = self.root / "authorizations"
        self.promotions_dir = self.root / "promotion_proposals"
        self.telemetry_path = self.root / "telemetry.jsonl"
        self.latest_telemetry_path = self.root / "latest_telemetry.json"

    @classmethod
    def active(cls, *, state_root: Path | str = Path(r"D:\Axon\State")) -> "TrainerStateStore":
        state = Path(state_root).resolve(strict=False)
        return cls(state / "training" / "trainer")

    def write_inventory(self, inventory: ParameterInventory) -> Path:
        if not isinstance(inventory, ParameterInventory):
            raise TypeError("inventory must be ParameterInventory")
        path = self.inventories_dir / f"{inventory.inventory_id}.json"
        self._write_immutable(path, inventory.to_canonical_dict())
        return path

    def write_plan(self, plan: ParameterMutationPlan) -> Path:
        if not isinstance(plan, ParameterMutationPlan):
            raise TypeError("plan must be ParameterMutationPlan")
        path = self.plans_dir / f"{plan.plan_id}.json"
        self._write_immutable(path, plan.to_canonical_dict())
        return path

    def write_authorization(self, authorization: AuthorizedParameterMutation) -> Path:
        if not isinstance(authorization, AuthorizedParameterMutation):
            raise TypeError("authorization must be AuthorizedParameterMutation")
        path = self.authorizations_dir / f"{authorization.authorization_id}.json"
        self._write_immutable(path, authorization.to_canonical_dict())
        return path

    def write_promotion_proposal(self, proposal: ParameterPromotionProposal) -> Path:
        if not isinstance(proposal, ParameterPromotionProposal):
            raise TypeError("proposal must be ParameterPromotionProposal")
        path = self.promotions_dir / f"{proposal.proposal_id}.json"
        self._write_immutable(path, proposal.to_canonical_dict())
        return path

    def append_telemetry(self, frame: ParameterTelemetryFrame) -> None:
        if not isinstance(frame, ParameterTelemetryFrame):
            raise TypeError("frame must be ParameterTelemetryFrame")
        value = frame.to_canonical_dict()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._atomic_json(self.latest_telemetry_path, value)

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _write_immutable(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json_bytes(existing) != canonical_json_bytes(dict(value)):
                raise TrainerStoreError(f"immutable Trainer artifact disagrees at {path}")
            return
        self._atomic_json(path, value)


__all__ = ["TrainerStoreError", "TrainerStateStore"]
