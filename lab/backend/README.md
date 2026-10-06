# Axon Lab backend: first live slice

This is the local foundation service, not a working training engine. It serves
the real frontend and same-origin `/api/v1` on loopback port 8080.

Normal public operation uses the existing **Launch Cloudfare** launcher. Its
managed 8080 child now hosts the Axon API and deployed frontend from
`G:\My Drive\Cloudfare\Sites\axon\axon`. The existing tunnel still points to
8080. Other site hostnames keep their original handlers in the same managed
process. Axon's API executes directly on 8080; it does not depend on port 8184.

For isolated development, from the repository root:

```powershell
python -m lab.backend --port 8184
```

Open <http://127.0.0.1:8080> or <https://axon.gliksbot.com> during managed operation.
The frontend should say **Backend connected**. The development command above
instead serves the repository frontend on 8184.
No Python source editing is required. `--port` selects another local port.
The service uses the installed Python, FastAPI, Uvicorn and jsonschema packages;
it does not download a runtime, install packages or launch training.

## Implemented

- Health, actual capability status and explicit training-readiness blockers.
- Asynchronous CPU/CUDA foundation checks through the existing read-only helper;
  accepted operations are persisted and polled separately. Duplicate command IDs
  retry the same check; changed contents conflict. One check runs at a time.
- Versioned architecture registry and validation against trusted component
  contracts, including configuration, version and port meaning/shape/dtype/
  authority and required surface checks. Caller-supplied port declarations cannot
  override contracts; crossing substrate-exact/free surfaces needs an explicit
  versioned adapter.
- Empty dataset/curriculum/run/checkpoint registries until actual manifests and
  adapters are registered. The scoped inventory summary is evidence, not an
  automatically admitted dataset. Backup status accurately remains unverified.
- Structured errors for unavailable training, inference, events and inspection.

The default catalog now includes the Core owner's three E0 v0.1.0 manifests:
`axon.substrate_input`, `axon.core_reasoning_gru`, and `axon.response_state`.
GRU/FFN/transformer/custom placeholders still mark future adapters unavailable.
At service startup a bounded, isolated CPU process checks adapter import, D512
reference-graph validation and registration in disposable storage, and a
bit-identical Core checkpoint roundtrip including next-tick outputs. Readiness
returns cached, source-hashed evidence; GET requests do not run new probes.
These checks neither create an architecture in Jeff's registry nor authorize
training. Core checkpoint evidence excludes optimizer and Heart/draft/replay
state; the complete runtime recovery gate remains pending.

SQLite lives under `%LOCALAPPDATA%\Axon\Lab\<workspace-hash>`, outside Drive
synchronization. Completed operation results and architecture definitions survive
restart. Interrupted checks are labelled interrupted; a failed newer check does
not silently reuse an earlier green result. This registry is not a model
checkpoint or an independently verified backup. Run only one service instance
per registry. The CLI uses a single Uvicorn worker.

## Hosted integration

`hosted.py` dispatches the Axon hostname and local addresses to the real API and
deployed frontend on 8080. Other hosts use the original multi-host handler on a
private dynamically assigned loopback socket within the same process; request
Host, path, body and response headers are preserved. The shared launcher restarts
the entire child if it exits. OS-stored tunnel credentials are unchanged. Existing
unsupported runtime/SSE families remain explicitly unavailable.

The frontend owner has delivered preflight controls, configuration/edge editing
and native-alphabet hints, and is reviewing recovery and real-manifest operation.
Full phone/desktop acceptance, HeartHost execution, curriculum splits, complete
checkpoint restore and independent backup recovery remain required before
valuable training.
