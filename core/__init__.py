"""Axon cores: PyTorch modules that propose; the Heart remains the sole writer.

The package also re-exports the Core-owned component contracts so callers can
reach them without importing the module path directly.
"""
from __future__ import annotations

from core.manifests import COMPONENT_SCHEMA_VERSION, E0_REFERENCE_GRAPH, component_manifests

__all__ = ["component_manifests", "COMPONENT_SCHEMA_VERSION", "E0_REFERENCE_GRAPH"]
