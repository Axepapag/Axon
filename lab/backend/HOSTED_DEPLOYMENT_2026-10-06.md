# Axon frontend/API on port 8080

Jeff authorized connecting the API and clarified that the Axon server should
use the existing tunnel's port **8080**. This supersedes the temporary separate
8184 deployment and the earlier proxy-to-8184 plan.

Verified current deployment:

- `https://axon.gliksbot.com` -> existing tunnel -> `127.0.0.1:8080`.
- Both the deployed frontend and Axon `/api/v1` execute on that listener.
  Frontend directory remains `G:\My Drive\Cloudfare\Sites\axon\axon`.
  API source remains `G:\My Drive\Projects\Axon\lab\backend`.
- Existing launcher `main-sites` child now starts the hosted Axon service.
  PID 29636 is owned by supervisor 10700. Only that child was restarted;
  the tunnel PID 21916 and all other supervised service PIDs remained unchanged.
- Public health, capabilities, readiness and app.js return HTTP 200. Kimi
  Browser Extension observed **Backend connected** on the actual public page.
  The previously completed GTX1650 preflight remains visible after migration.
- Temporary services on 8184 and 8185 were stopped; no listeners remain there.
- The original handler serves other hostnames privately inside the managed
  process. Real staged responses for gliksbot root, chess, Downloads listing and
  the existing Plex 404 were byte-identical to the old listener. Hosted tests
  also verify Host/path/body forwarding and repeated response headers.
- Full suite: **167 passed** (19.00 seconds). Both substrate self-tests exited
  0 separately. Launcher compilation and scoped diff check succeeded.

Changed hosting files: `G:\My Drive\Cloudfare\cloudflare_launcher.py` and
`G:\My Drive\Cloudfare\launcher.json`. Prechange copies retained in
`G:\My Drive\Cloudfare\backups\axon-api-20261006-065229`. Tunnel configuration
and credentials were not changed. Frontend source/deployed copies were not
edited. The normal shared launcher now owns startup and child restart.

This connects the existing foundation API; it does not finish training/runtime
adapters, curriculum preparation, checkpoint recovery or independent backup
restoration. `training_authorized` remains false. No training was launched.

Bus coordination: `6243bfb7-7b6f-40fe-9794-2306a4005132` (8080 plan) and
`f3e7fe55-7b21-46b5-a205-36262171e58a` (public delivery).
