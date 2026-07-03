# Axon — Provenance

## Archive source

- Archive repo: `D:\AxonGliksbot`
- Archive commit at promotion time: `18f52fc7e26a74221ca4a53f9879fcc122cfb47b`
- Promotion date: 2026-07-03

## Promoted files

| Source (AxonGliksbot) | Destination (Axon) | Notes |
|---|---|---|
| `substrate.py` | `substrate/substrate.py` | Frozen 16D character alphabet; vectors unchanged |
| `core.py` | `cores/core.py` | Transformer core + legacy soul modes |
| `soul_v2.py` | `cores/soul_v2.py` | Temperature-tiered soul manager (not yet wired into core.py) |
| `cf_probe.py` | `training/cf_probe.py` | Counterfactual soul-read diagnostic |
| `kg_search.py` | `curator/kg_search.py` | Knowledge graph search |
| `container_schema.py` | `curator/container_schema.py` | Container/edge schema |
| `semantic_layout_machine.py` | `curator/semantic_layout_machine.py` | Deterministic dormant-state importer |
| `dormant_materializer.py` | `curator/dormant_materializer.py` | Dormant container materializer |
| `axon_bus/` (whole package) | `runtime/bus/` | Collaboration bus; imports fixed to relative |
| `tests/test_substrate_contract.py` | `tests/test_substrate_contract.py` | |
| `tests/test_container_schema.py` | `tests/test_container_schema.py` | |
| `tests/test_kg_search.py` | `tests/test_kg_search.py` | |
| `tests/test_semantic_layout_machine.py` | `tests/test_semantic_layout_machine.py` | |
| `tests/test_dormant_materializer.py` | `tests/test_dormant_materializer.py` | |
| `tests/test_axon_bus_mcp.py` | `tests/test_axon_bus_mcp.py` | Import path fixed |
| `tests/test_axon_bus_registry.py` | `tests/test_axon_bus_registry.py` | Import path fixed |
| `tests/test_axon_bus_schema.py` | `tests/test_axon_bus_schema.py` | Import path fixed |
| `tests/test_axon_bus_server.py` | `tests/test_axon_bus_server.py` | Import path fixed |
| `tests/test_axon_bus_state.py` | `tests/test_axon_bus_state.py` | Import path fixed |

## Not promoted (stays in archive only)

- `capsule_spec.py`, `capsule_core.py`, `build_capsule_curriculum.py`, `trainer_capsule_core.py` — research prototype, not runtime path
- `field_contract.py`, `heads.py` — legacy 16D field/rail, superseded by slot field
- `legacy_8192/` — superseded wide-substrate lane (design template only for new adapters)
- `trainer_recall.py`, `trainer_semantic.py`, `trainer_soul_v2.py`, `train_field_surfacing.py`, `field_recall.py` — legacy trainers, not promoted
- `runtime.py` — legacy runtime, not the slot-era runtime
- `lora_adapter.py`, `mint_rail.py` — legacy adapter/rail tooling
- `ssh.py`, `ssh_helper.py` — credential-bearing, never promoted
- `build_*.py`, `consolidate_vocab.py`, `vocab_*.py` — data builders, archive only
- `cold_read.py`, `substrate.py` self-test utilities beyond the alphabet

## Import changes on promotion

- `axon_bus/` → `runtime/bus/`: all `from axon_bus.xxx` imports changed to relative `from .xxx`
- Tests: `from axon_bus.xxx` → `from bus.xxx`
- All other promoted modules: no import changes (pythonpath in pyproject.toml covers package dirs)
- `soul_v2.py` line 464: `from core import CrossAttention` — works via `cores/` in pythonpath