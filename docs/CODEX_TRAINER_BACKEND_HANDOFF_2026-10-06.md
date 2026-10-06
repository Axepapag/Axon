# Axon Lab trainer/backend handoff

Codex / 2026-10-06. Status: accepted ownership and first interface specification;
backend service and browser application are not implemented by this document.

## Direct request

Jeff authorized Codex to collaborate with the existing ChatGPT Axon discussion
through the Kimi Browser Extension and divide trainer/interface work. The lab
must support experiments with GRU/FFN, Transformers with attention, and custom
cells. A preference within one experiment must not constrain the laboratory.
Jeff requires a complete hands-on browser vehicle before valuable training.

## Accepted ownership

| Owner | Reserved paths | Responsibility |
|---|---|---|
| Codex | `lab/backend/`, `lab/contracts/`, `tests/test_lab_backend_*.py` | Trainer service, public interface specifications, integration and backend tests |
| ChatGPT | `lab/frontend/` | Responsive browser application, visual builder and operator workflows |

ChatGPT accepted on the Iris bus in message
`a1501960-2453-4ed8-9e92-223484ec3a60`; Codex acknowledged in
`7e897d4c-ec35-4ec4-a2f1-1c2e74b6b733`. The coordination occurred in the existing
ChatGPT conversation `https://chatgpt.com/c/6ac3e488-00a0-83ea-b991-128ef706b751`.
ChatGPT reports direct file-writing capability in this workspace. Verify actual
source and behavior independently before claiming frontend completion.

Core/Architecture stays with KimiCode and its substrate specialist. Heart/Runtime,
Curriculum/Data, Tests/Measurement and Governance/Docs retain their existing
owners. Backend consumes their versioned contracts; this handoff does not
delegate their source files or settle open cognitive update equations.

## First concrete deliverables

1. Public interface specification: `lab/contracts/trainer-api-v1.md`.
2. Capability and asynchronous preflight service, wrapping the existing verified
   `tools/training_preflight.py` without blocking request processing.
3. Architecture manifest validation, registry and explicit supported adapters.
4. Run registry and lifecycle commands with observable operation results.
5. Complete checkpoint, restore and inference integration using the same runtime
   path, with live ordered events and bounded tensor/draft inspection.

The frontend can begin its shell, configuration forms and disconnected/error
states against the interface specification. Endpoints in that specification are
design commitments, not evidence that a server currently implements them.

## Completion criteria

The joined application must let Jeff construct and validate supported cores;
select a provenance-bearing curriculum and device; operate real lifecycle
controls; inspect real metrics, tensors and exact draft revisions; save and
restore complete checkpoints; and run inference from the selected checkpoint.
Verify phone and desktop operation separately from automated checks.

Training-readiness evidence includes actual runtime-path integration, native-95
admission, valid dataset splits, device availability, complete resume state and
valuable-artifact backup/restore. Foundation preflight alone is insufficient.
Unavailable integrations have an explicit status and reason. No simulated
metrics or automatic architecture/device substitutions count as acceptance.

## Verified during this coordination

The workspace is `G:\My Drive\Projects\Axon`. Existing preflight source remains
present. System Python package metadata reports FastAPI 0.141.1, Uvicorn 0.52.3,
Torch 2.13.0+cu126 and NumPy 2.5.2; these packages were not installed by this turn.
The earlier 151-test result and GTX1650 operation are historical evidence from
the preflight implementation turn, not a new test run here.

This handoff does not launch training, import broad G-drive data, deploy cloud
jobs or alter the old Axon GitHub repository. The experimental resident-memory
pipeline remains a selectable design target requiring its own implementation
and measurements.
