"""Persist one base generation's exact bytes so base identity is machine-stable.

Axon already ratified this doctrine.  ``substrate/substrate.py`` records that
"bitwise hashes of computed floats vary across BLAS builds" and therefore
materializes ``v7_reference_bank.npy`` once, then verifies later rebuilds with
``np.allclose(..., atol=1e-5, rtol=0.0)`` -- a tolerance far above BLAS
last-ULP noise and far below any real weight edit.

The untrained base module was the one computed artifact the Trainer still
rebuilt from code on every run.  Because ``registry.parameter_value_sha256``
hashes raw bytes, ``ParameterInventory.inventory_id`` -- and with it
``ParameterMutationPlanV2.plan_id`` -- was a property of the machine's BLAS and
reduction order rather than a property of the governed lineage.  A resumable
lineage therefore became reachable only from the environment that first built
its base, which is exactly the platform lock-in this module removes.

The repair is forward-only and changes no training behaviour:

* The first run of a fresh base generation stores the base module's exact
  parameter and buffer bytes under the trainer state root.
* Every later run rebuilds the base from code, then verifies the rebuild
  against the stored artifact with ``atol=1e-5, rtol=0.0`` and adopts the
  artifact's bit-exact bytes, so ``base_inventory_id`` no longer depends on
  the local BLAS build.
* A rebuild that leaves the doctrine tolerance fails closed and names the
  records, which means a real initialisation change must be re-versioned
  rather than silently masked.
* A resumed lineage whose recorded base inventory is bit-reproducible
  everywhere except this machine also fails closed, and says so, instead of
  surfacing an opaque ``checkpoint plan lineage mismatch``.
* A stored artifact is checked hash-for-hash against the lineage's recorded
  base inventory before it is adopted, so an artifact captured before that
  inventory existed cannot quietly become the lineage's base.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..field.schema import canonical_sha256
from .contracts import ParameterModuleDescriptor
from .registry import capture_module_manifest

BASE_ARTIFACT_SCHEMA = "axon-base-module-artifact-v1"
BASE_ARTIFACT_DIRNAME = "base_artifacts"
BASE_ARTIFACT_ATOL = 1e-5


class BaseArtifactError(RuntimeError):
    """Raised when a base generation's bytes cannot be governed reproducibly."""


@dataclass(frozen=True, slots=True)
class BaseArtifactDisposition:
    """What the base-identity step did, in terms a report can carry."""

    action: str
    artifact_path: Path
    artifact_key: str
    recorded_base_inventory_id: str | None = None
    reconciled_records: tuple[str, ...] = ()
    max_abs_delta: float = 0.0

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "artifact_key": self.artifact_key,
            "artifact_relpath": self.artifact_path.name,
            "recorded_base_inventory_id": self.recorded_base_inventory_id,
            "reconciled_record_count": len(self.reconciled_records),
            "reconciled_records": list(self.reconciled_records),
            "max_abs_delta": self.max_abs_delta,
        }


def base_artifact_key(*, module_id: str, generation_id: str) -> str:
    """Return the content-free identity of one base generation."""

    return canonical_sha256({"module_id": module_id, "generation_id": generation_id})


def base_artifact_path(state_root: Path | str, *, module_id: str, generation_id: str) -> Path:
    return (
        Path(state_root).resolve()
        / "training"
        / "trainer"
        / BASE_ARTIFACT_DIRNAME
        / f"{base_artifact_key(module_id=module_id, generation_id=generation_id)}.pt"
    )


def _live_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in module.state_dict().items()
    }


def _canonical_records(manifest_records: Any) -> list[dict[str, Any]]:
    return [item.to_canonical_dict() for item in manifest_records]


def _recorded_base_inventory(
    state_root: Path | str,
    *,
    module_id: str,
    candidate_generation_id: str,
) -> tuple[str | None, dict[str, str]]:
    """Return the recorded base inventory id and its exact record hashes.

    The resuming parent checkpoint record names the plan that produced it, and
    the plan names the base inventory.  Those two ids are the lineage's own
    statement of which base it was trained on; nothing here is inferred from
    the local machine.
    """

    root = Path(state_root).resolve() / "training" / "trainer"
    latest = root / "candidates" / module_id / candidate_generation_id / "latest_checkpoint.json"
    if not latest.is_file():
        return None, {}
    plan_id = json.loads(latest.read_text(encoding="utf-8")).get("plan_id")
    if not plan_id:
        return None, {}
    plan_path = root / "plans" / f"{plan_id}.json"
    if not plan_path.is_file():
        raise BaseArtifactError(
            f"recorded plan {plan_id} for {module_id}/{candidate_generation_id} is missing from "
            f"{plan_path}; the state root is incomplete and base identity cannot be governed"
        )
    inventory_id = json.loads(plan_path.read_text(encoding="utf-8")).get("base_inventory_id")
    if not inventory_id:
        return None, {}
    inventory_path = root / "inventories" / f"{inventory_id}.json"
    if not inventory_path.is_file():
        raise BaseArtifactError(
            f"recorded base inventory {inventory_id} is missing from {inventory_path}; "
            "the state root cannot prove which base this lineage was trained on"
        )
    body = json.loads(inventory_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    for module_manifest in body.get("manifests", ()):
        descriptor_body = module_manifest.get("descriptor") or {}
        if descriptor_body.get("module_id") != module_id:
            continue
        for group in ("tensors", "buffers"):
            for item in module_manifest.get(group, ()):
                name = item.get("name")
                digest = item.get("value_sha256")
                if name and digest:
                    hashes[name] = digest
    if not hashes:
        raise BaseArtifactError(
            f"recorded base inventory {inventory_id} carries no value hashes for {module_id}; "
            "base identity cannot be governed from it"
        )
    return str(inventory_id), hashes


def _verify_stored_artifact(
    stored: dict[str, Any],
    *,
    live: dict[str, torch.Tensor],
    key: str,
    descriptor: ParameterModuleDescriptor,
    atol: float,
) -> tuple[tuple[str, ...], float]:
    if stored.get("schema") != BASE_ARTIFACT_SCHEMA:
        raise BaseArtifactError(f"base artifact schema mismatch: {stored.get('schema')!r}")
    if stored.get("key") != key:
        raise BaseArtifactError("base artifact identity disagrees with its filename")
    stored_descriptor = stored.get("descriptor") or {}
    current_descriptor = descriptor.to_canonical_dict()
    for field in ("module_id", "generation_id", "architecture"):
        if stored_descriptor.get(field) != current_descriptor.get(field):
            raise BaseArtifactError(
                "base artifact was captured for a different base module "
                f"({field}: artifact {stored_descriptor.get(field)!r} vs live "
                f"{current_descriptor.get(field)!r}); re-version the base generation"
            )
    state = stored.get("state")
    if not isinstance(state, dict):
        raise BaseArtifactError("base artifact carries no parameter state")
    if set(state) != set(live):
        missing = sorted(set(state) - set(live))
        extra = sorted(set(live) - set(state))
        raise BaseArtifactError(
            "base artifact anatomy disagrees with the live module "
            f"(missing {missing[:4]}, unexpected {extra[:4]}); re-version the base generation"
        )
    reconciled: list[str] = []
    worst = 0.0
    for name in sorted(state):
        reference = state[name]
        observed = live[name]
        if tuple(reference.shape) != tuple(observed.shape) or reference.dtype != observed.dtype:
            raise BaseArtifactError(
                f"base artifact tensor {name} changed shape or dtype; re-version the base generation"
            )
        if torch.equal(reference, observed):
            continue
        left = reference.to(torch.float64)
        right = observed.to(torch.float64)
        delta = float((left - right).abs().max().item()) if left.numel() else 0.0
        worst = max(worst, delta)
        if not torch.allclose(right, left, atol=atol, rtol=0.0):
            raise BaseArtifactError(
                f"base initialization changed beyond doctrine tolerance ({name}: max abs delta "
                f"{delta:g} > atol {atol:g}); the base generation must be re-versioned"
            )
        reconciled.append(name)
    return tuple(reconciled), worst


def _write_artifact(
    path: Path,
    *,
    key: str,
    descriptor: ParameterModuleDescriptor,
    records: list[dict[str, Any]],
    state: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BASE_ARTIFACT_SCHEMA,
        "key": key,
        "descriptor": descriptor.to_canonical_dict(),
        "records": records,
        "state": state,
    }
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _assert_artifact_matches_recorded_inventory(
    stored: dict[str, Any],
    *,
    recorded_id: str | None,
    recorded_hashes: dict[str, str],
    module_id: str,
    generation_id: str,
) -> None:
    """Refuse to adopt bytes that the lineage's own record contradicts.

    An artifact captured before this lineage recorded a base inventory would
    otherwise govern forever and silently install a base identity the lineage
    was never trained on -- the same lock-in this module exists to remove, only
    now hidden behind an artifact instead of a BLAS build.  The comparison is
    hash-against-hash, so it costs nothing and cannot drift.
    """

    if not recorded_hashes:
        return
    records = stored.get("records")
    if not isinstance(records, list) or not records:
        raise BaseArtifactError(
            f"persisted base artifact for {module_id}/{generation_id} carries no record hashes; "
            "it cannot be reconciled with the recorded base identity, so it must be rebuilt"
        )
    stored_hashes = {str(item["name"]): str(item["value_sha256"]) for item in records}
    if stored_hashes == recorded_hashes:
        return
    drifted = sorted(
        name
        for name in set(recorded_hashes) | set(stored_hashes)
        if recorded_hashes.get(name) != stored_hashes.get(name)
    )
    raise BaseArtifactError(
        f"persisted base artifact for {module_id}/{generation_id} was captured before the "
        f"lineage recorded base inventory {recorded_id} and disagrees with it "
        f"({len(drifted)} of {len(recorded_hashes)} records differ, e.g. {drifted[:4]}); "
        "adopting it would install a base identity this lineage was never trained on, so the "
        "base generation must be re-versioned rather than resumed"
    )


def resolve_base_module(
    module: nn.Module,
    *,
    state_root: Path | str,
    descriptor: ParameterModuleDescriptor,
    candidate_generation_id: str,
    atol: float = BASE_ARTIFACT_ATOL,
) -> BaseArtifactDisposition:
    """Make the base module's governed bytes machine-independent, or fail closed."""

    key = base_artifact_key(module_id=descriptor.module_id, generation_id=descriptor.generation_id)
    path = base_artifact_path(
        state_root, module_id=descriptor.module_id, generation_id=descriptor.generation_id
    )
    live = _live_state(module)
    recorded_id, recorded_hashes = _recorded_base_inventory(
        state_root,
        module_id=descriptor.module_id,
        candidate_generation_id=candidate_generation_id,
    )
    if path.is_file():
        stored = torch.load(path, map_location="cpu", weights_only=True)
        reconciled, worst = _verify_stored_artifact(
            stored, live=live, key=key, descriptor=descriptor, atol=atol
        )
        _assert_artifact_matches_recorded_inventory(
            stored,
            recorded_id=recorded_id,
            recorded_hashes=recorded_hashes,
            module_id=descriptor.module_id,
            generation_id=descriptor.generation_id,
        )
        module.load_state_dict(stored["state"], strict=True)
        return BaseArtifactDisposition(
            action="adopted",
            artifact_path=path,
            artifact_key=key,
            recorded_base_inventory_id=recorded_id,
            reconciled_records=reconciled,
            max_abs_delta=worst,
        )
    manifest = capture_module_manifest(descriptor, module, exact_value_hashes=True)
    records = _canonical_records(manifest.tensors) + _canonical_records(manifest.buffers)
    live_hashes = {str(item["name"]): str(item["value_sha256"]) for item in records}
    if recorded_hashes:
        drifted = sorted(
            name
            for name in set(recorded_hashes) | set(live_hashes)
            if recorded_hashes.get(name) != live_hashes.get(name)
        )
        if drifted:
            raise BaseArtifactError(
                "no persisted base artifact exists and this machine's rebuild of "
                f"{descriptor.module_id}/{descriptor.generation_id} is not bit-identical to the "
                f"recorded base inventory {recorded_id} ({len(drifted)} of "
                f"{len(recorded_hashes)} records differ, e.g. {drifted[:4]}). The recorded base "
                "bytes are not stored on this machine, so resuming here would install a different "
                "base identity than the lineage was trained on. Run the first post-repair tranche "
                "on the platform that produced the recorded base, which will persist the base "
                "artifact and free the lineage, or re-version the base generation."
            )
    _write_artifact(
        path, key=key, descriptor=descriptor, records=records, state=live
    )
    return BaseArtifactDisposition(
        action="created" if not recorded_hashes else "created-verified",
        artifact_path=path,
        artifact_key=key,
        recorded_base_inventory_id=recorded_id,
    )
