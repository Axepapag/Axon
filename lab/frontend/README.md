# Axon Lab frontend

Owner: ChatGPT. Reserved path: `lab/frontend/`.

This is the browser/operator layer for Axon Lab. It consumes Codex-owned
`lab/contracts/trainer-api-v1.md` and does not redefine backend, Heart,
Core/Architecture, Curriculum/Data, or Tests contracts.

## Current first slice

- Responsive phone/desktop application shell.
- Capability-driven connection, loading, disconnected and error states.
- System/readiness view.
- Visual architecture draft/palette driven by backend component capabilities.
- Backend architecture validation request.
- Dataset/curriculum registry views.
- Explicit run creation and lifecycle controls driven by `allowed_actions`.
- Ordered run-event SSE subscription.
- Read-only bounded tensor inspection.
- Checkpoint/restore request workflow.
- Checkpoint-selected inference session workflow with private draft vs
  Heart-submitted response shown separately.
- Backup/restore evidence view.
- No fabricated metrics, mock provider claims, or silent device/component fallback.

## Serving locally

No JavaScript build step is required. Serve this directory from an HTTP server,
for example:

```powershell
python -m http.server 4173 --directory "G:\My Drive\Projects\Axon\lab\frontend"
```

Open `http://127.0.0.1:4173/`.

By default the app calls `/api/v1` on the same origin. For development against
another backend origin, use the explicit query parameter:

```text
http://127.0.0.1:4173/?api=http://127.0.0.1:8000/api/v1
```

The backend must permit that origin if it is cross-origin.

## Important status

The public Axon Lab foundation backend is live on the same origin at
`https://axon.gliksbot.com/api/v1` through the managed port-8080 service.
Capabilities, readiness evidence, asynchronous foundation preflight, operation
polling, and architecture validation/registry plumbing are real services.

Training, run execution, checkpoint/restore, tensor inspection, and inference
remain capability-locked until their Core/Heart/Data/runtime adapters and
acceptance evidence are integrated. A disabled control is intentional when its
backend capability is unavailable.

The UI does not authorize valuable training. A green foundation preflight is
evidence only; higher-level readiness must explicitly report training
authorization.


## Slice 2 operator controls

- System now has explicit Auto / CPU / CUDA 0 foundation-preflight controls.
- Preflight uses one stable `command_id` per request/retry and polls the returned
  operation until a terminal status. Current operation state is shown separately
  from completed historical server evidence.
- Architecture registration retains its `command_id` while the graph payload is
  unchanged, so a network retry cannot silently create a second registration.
- Architecture draft, selected component, edge draft, registration retry identity,
  and preflight operation state persist in browser local storage across refresh.
- Component properties are generated from backend `configuration_schema`.
  Explicit edges are selected only from backend-declared input/output ports.
  Components without a real schema/ports/execution adapter remain disabled.
- Inference text is validated against backend `native_alphabet`; it does not use
  generic printable-ASCII rules. In the current substrate newline is native and
  backtick is not.

The browser's local persistence is convenience state only. Backend registry state,
Heart state, model checkpoints, and training evidence remain authoritative.


## Slice 3 preflight recovery

- A persisted `submitting` request with a `command_id` but no acknowledged
  `operation_id` is treated as an uncertain submission after reload.
- Recovery replays the **same command ID**. This is safe whether the first POST
  reached the server and its acknowledgement was lost, or never reached the
  server at all.
- Operation polls carry a selection generation plus command/operation identity;
  a late response from an older operation cannot overwrite a newer selection.
- Terminal operation state is persisted before readiness/capability evidence is
  refreshed. If that auxiliary refresh fails, the completed/failed/interrupted
  probe remains authoritative and the refresh failure is shown separately.
