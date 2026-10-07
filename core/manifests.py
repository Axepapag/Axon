"""Core-owned component manifests for the E0 two-state architecture.

These are plain, JSON-serializable contracts. The Heart is the sole writer of
canonical state: the ports below name a proposal authority of ``proposes-only``
wherever a core emits something, so no core can write the Shared Field directly.
"""
from __future__ import annotations

COMPONENT_SCHEMA_VERSION = "axon-core-components-v1"

_JSON_SCHEMA = "https://json-schema.org/draft/2020-12/schema"


def _port(port_id, direction, meaning, dtype, shape, width, stateful, authority, surface, constraints):
    return {"port_id": port_id, "direction": direction, "meaning": meaning, "dtype": dtype, "shape": shape,
            "width": width, "stateful": stateful, "authority": authority, "surface": surface,
            "constraints": constraints}


def _state(role, initialization, reset, retention, checkpoint):
    return {"role": role, "initialization": initialization, "reset": reset, "retention": retention,
            "checkpoint": checkpoint}


def _substrate_input():
    return {
        "id": "axon.substrate_input",
        "name": "Substrate Input (exact 95)",
        "version": "0.1.1",
        "status": "experimental",
        "reason": "E0 exact substrate ingest; frozen codes only",
        "execution_eligible": True,
        "configuration_schema": {
            "$schema": _JSON_SCHEMA,
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "width": {"type": "integer", "enum": [512], "default": 512,
                          "description": "D512 baseline only in 0.1.x; wider widths arrive as a later versioned manifest"},
                "occupancy": {"type": "integer", "minimum": 1, "maximum": 32, "default": 1,
                              "description": "occupied lane prefix count; must be <= width/16 (32 at D512)"},
            },
        },
        "ports": [
            _port("char_ids_in", "input", "exact-native-char-ids", "int64", [-1], None, False, "heart",
                  "substrate-exact", {"codebook": "native95", "ids": "0..94"}),
            _port("cells_out", "output", "frozen-d16-cell-surface", "float32", [32, 16], 512, False,
                  "frozen-substrate", "substrate-exact",
                  {"codebook": "native95+empty95", "occupancy": "prefix-1..32", "tolerance": 0.0,
                   "shape_note": "shape is [width/16, 16]"}),
        ],
        "constraints": {"law": "frozen D16 substrate; values restricted to the 96 frozen codes; never learned or projected"},
        "states": [],
    }


def _core_reasoning_gru():
    return {
        "id": "axon.core_reasoning_gru",
        "name": "Reasoning/Memory GRU State",
        "version": "0.1.1",
        "status": "experimental",
        "reason": "E0 recurrent reasoning/memory state (Jeff R3 ratification 3)",
        "execution_eligible": True,
        "configuration_schema": {
            "$schema": _JSON_SCHEMA,
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hidden_size": {"type": "integer", "enum": [512], "default": 512,
                               "description": "fixed at 512 in 0.1.x; the trusted port shapes are fixed"},
            },
        },
        "ports": [
            _port("observation", "input", "frozen-d16-cell-surface", "float32", [32, 16], 512, False,
                  "frozen-substrate", "substrate-exact",
                  {"codebook": "native95+empty95", "occupancy": "prefix-1..32", "tolerance": 0.0}),
            _port("state_reading", "output", "recurrent-reading", "float32", [512], None, True, "core-internal",
                  "free", {}),
        ],
        "constraints": {"cell": "GRUCell", "state_shape": [512]},
        "states": [
            _state("reasoning/memory", "zeros", "on episode start", "per tick; persisted in checkpoints",
                   "included"),
        ],
    }


def _response_state():
    return {
        "id": "axon.response_state",
        "name": "English Response State",
        "version": "0.1.1",
        "status": "experimental",
        "reason": "E0 dedicated English-facing response state; proposes exact characters with END/WAIT control",
        "execution_eligible": True,
        "configuration_schema": {
            "$schema": _JSON_SCHEMA,
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hidden_size": {"type": "integer", "enum": [512], "default": 512,
                               "description": "fixed at 512 in 0.1.x; the trusted port shapes are fixed"},
                "max_chunk": {"type": "integer", "enum": [64], "default": 64,
                            "description": "fixed at 64 in 0.1.x; the trusted port shape is fixed"},
            },
        },
        "ports": [
            _port("reading", "input", "recurrent-reading", "float32", [512], None, True, "core-internal", "free",
                  {}),
            _port("char_ids_out", "output", "exact-char-proposal", "int64", [64], 64, False, "proposes-only",
                  "substrate-exact",
                  {"codebook": "native95+empty95", "pad": 95, "max_chunk": 64,
                   "note": "Heart validates every proposed id; core never writes the field"}),
            _port("control_out", "output", "emission-control", "int64", [1], None, False, "proposes-only", "free",
                  {"values": "0=WAIT,1=COMMIT,2=END",
                   "note": "WAIT/zero-commit is control state, never an all-EMPTY surface"}),
        ],
        "constraints": {"cell": "GRUCell", "state_shape": [512]},
        "states": [
            _state("response-composition", "zeros", "on episode start", "per tick; persisted in checkpoints",
                   "included"),
        ],
    }


def component_manifests() -> list[dict]:
    """Return fresh, plain-JSON manifests for the E0 core components."""
    return [_substrate_input(), _core_reasoning_gru(), _response_state()]


E0_REFERENCE_GRAPH = {
    "version": "0.1.1",
    "name": "e0-two-state-d512",
    "nodes": [
        {"node_id": "substrate_in", "component_type": "axon.substrate_input", "component_version": "0.1.1",
         "config": {"width": 512, "occupancy": 1}},
        {"node_id": "reasoning", "component_type": "axon.core_reasoning_gru", "component_version": "0.1.1",
         "config": {"hidden_size": 512}},
        {"node_id": "response", "component_type": "axon.response_state", "component_version": "0.1.1",
         "config": {"hidden_size": 512, "max_chunk": 64}},
    ],
    "edges": [
        {"source": {"node_id": "substrate_in", "port_id": "cells_out"},
         "destination": {"node_id": "reasoning", "port_id": "observation"}},
        {"source": {"node_id": "reasoning", "port_id": "state_reading"},
         "destination": {"node_id": "response", "port_id": "reading"}},
    ],
    "ports": [],
}
