"""Complete parameter inventory for Axon's Trainer organ."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .contracts import (
    ParameterInventory,
    ParameterModuleDescriptor,
    ParameterModuleManifest,
    ParameterTensorRecord,
)


class ParameterRegistryError(RuntimeError):
    pass


class IncompleteParameterInventoryError(ParameterRegistryError):
    pass


@dataclass(slots=True)
class _Registration:
    descriptor: ParameterModuleDescriptor
    module: nn.Module


def parameter_value_sha256(parameter: torch.Tensor, *, chunk_numel: int = 1_000_000) -> str:
    """Hash an exact dense tensor without allocating a second full-size CPU copy."""

    if chunk_numel < 1:
        raise ValueError("chunk_numel must be positive")
    tensor = parameter.detach()
    if tensor.layout is not torch.strided:
        raise ParameterRegistryError("exact parameter hashing currently requires dense strided tensors")
    flat = tensor.reshape(-1)
    digest = hashlib.sha256()
    if flat.numel() == 0:
        digest.update(b"")
        return digest.hexdigest()
    for start in range(0, flat.numel(), chunk_numel):
        chunk = flat[start : start + chunk_numel].to(device="cpu").contiguous()
        # Viewing bytes avoids NumPy dtype limitations such as bfloat16.
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ParameterRegistry:
    """Registry of every live parameter-bearing module known to the Trainer.

    A declared expected set turns inventory completeness into a fail-closed
    invariant.  The registry deliberately accepts heterogeneous ``d_model``
    widths; width belongs to each organ's descriptor, not to Trainer authority.
    """

    def __init__(self) -> None:
        self._expected: dict[str, ParameterModuleDescriptor] = {}
        self._registered: dict[str, _Registration] = {}

    @property
    def expected_module_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._expected))

    @property
    def registered_module_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registered))

    def declare_expected(self, descriptors: Iterable[ParameterModuleDescriptor]) -> None:
        expected: dict[str, ParameterModuleDescriptor] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, ParameterModuleDescriptor):
                raise TypeError("expected descriptors must be ParameterModuleDescriptor values")
            if descriptor.module_id in expected:
                raise ParameterRegistryError(f"duplicate expected module {descriptor.module_id!r}")
            expected[descriptor.module_id] = descriptor
        if self._registered:
            unexpected = sorted(set(self._registered) - set(expected))
            mismatched = sorted(
                module_id
                for module_id, registration in self._registered.items()
                if module_id in expected
                and registration.descriptor.to_canonical_dict() != expected[module_id].to_canonical_dict()
            )
            if unexpected or mismatched:
                raise ParameterRegistryError(
                    f"new expected set conflicts with registered modules; unexpected={unexpected}, mismatched={mismatched}"
                )
        self._expected = expected

    def register(self, descriptor: ParameterModuleDescriptor, module: nn.Module) -> None:
        if not isinstance(descriptor, ParameterModuleDescriptor):
            raise TypeError("descriptor must be ParameterModuleDescriptor")
        if not isinstance(module, nn.Module):
            raise TypeError("module must be torch.nn.Module")
        if descriptor.module_id in self._registered:
            raise ParameterRegistryError(f"module {descriptor.module_id!r} is already registered")
        if self._expected:
            if descriptor.module_id not in self._expected:
                raise ParameterRegistryError(f"module {descriptor.module_id!r} was not declared expected")
            expected = self._expected[descriptor.module_id]
            if descriptor.to_canonical_dict() != expected.to_canonical_dict():
                raise ParameterRegistryError(f"descriptor mismatch for expected module {descriptor.module_id!r}")
        self._registered[descriptor.module_id] = _Registration(descriptor=descriptor, module=module)

    def unregister(self, module_id: str) -> None:
        if module_id not in self._registered:
            raise KeyError(module_id)
        del self._registered[module_id]

    def module(self, module_id: str) -> nn.Module:
        try:
            return self._registered[module_id].module
        except KeyError as exc:
            raise KeyError(module_id) from exc

    def descriptor(self, module_id: str) -> ParameterModuleDescriptor:
        try:
            return self._registered[module_id].descriptor
        except KeyError as exc:
            raise KeyError(module_id) from exc

    def capture_inventory(
        self,
        *,
        exact_value_hashes: bool = True,
        require_complete: bool = True,
    ) -> ParameterInventory:
        missing = tuple(sorted(set(self._expected) - set(self._registered)))
        if require_complete and missing:
            raise IncompleteParameterInventoryError(
                f"parameter inventory is incomplete; missing modules: {', '.join(missing)}"
            )
        if self._expected:
            unexpected = tuple(sorted(set(self._registered) - set(self._expected)))
            if unexpected:
                raise ParameterRegistryError(
                    f"parameter inventory contains undeclared modules: {', '.join(unexpected)}"
                )

        manifests: list[ParameterModuleManifest] = []
        for module_id in sorted(self._registered):
            registration = self._registered[module_id]
            records: list[ParameterTensorRecord] = []
            seen: set[str] = set()
            for name, parameter in registration.module.named_parameters(recurse=True):
                if name in seen:
                    raise ParameterRegistryError(f"duplicate named parameter {module_id}:{name}")
                seen.add(name)
                digest = parameter_value_sha256(parameter) if exact_value_hashes else None
                records.append(
                    ParameterTensorRecord(
                        name=name,
                        shape=tuple(parameter.shape),
                        dtype=str(parameter.dtype),
                        numel=int(parameter.numel()),
                        requires_grad=bool(parameter.requires_grad),
                        value_sha256=digest,
                    )
                )
            manifests.append(
                ParameterModuleManifest(
                    descriptor=registration.descriptor,
                    tensors=tuple(records),
                )
            )

        expected_ids = tuple(sorted(self._expected or self._registered))
        return ParameterInventory(
            manifests=tuple(manifests),
            expected_module_ids=expected_ids,
            missing_module_ids=missing,
        )


__all__ = [
    "ParameterRegistryError",
    "IncompleteParameterInventoryError",
    "parameter_value_sha256",
    "ParameterRegistry",
]
