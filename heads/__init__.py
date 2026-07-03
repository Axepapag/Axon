"""Axon heads package — shared decode organs and diagnostic probes.

Modules:
  - write_head: shared text write head (per-position character decoder + length).
  - probe: read-fidelity probe over frozen adapter summaries.
"""
from __future__ import annotations

from heads.write_head import (
    WRITE_HEAD_VERSION,
    WriteHeadConfig,
    WriteHead,
    compute_loss as compute_write_head_loss,
    encode_decoded_slot,
    commit_diff,
)
from heads.probe import (
    READ_FIDELITY_PROBE_VERSION,
    ReadFidelityProbeConfig,
    ReadFidelityProbe,
    probe_metrics,
)

__all__ = [
    "WRITE_HEAD_VERSION",
    "WriteHeadConfig",
    "WriteHead",
    "compute_write_head_loss",
    "encode_decoded_slot",
    "commit_diff",
    "READ_FIDELITY_PROBE_VERSION",
    "ReadFidelityProbeConfig",
    "ReadFidelityProbe",
    "probe_metrics",
]
