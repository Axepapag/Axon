"""Serve Axon directly on the shared listener, retaining other hosts' handlers."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import http.server
import importlib.util
from pathlib import Path
import threading

from fastapi import FastAPI, Request
import httpx
from starlette.responses import JSONResponse, StreamingResponse

from .app import ROOT, create_app

AXON_HOSTS = frozenset(("axon.gliksbot.com", "127.0.0.1", "localhost"))
HOP_HEADERS = frozenset((b"connection", b"keep-alive", b"proxy-authenticate",
    b"proxy-authorization", b"te", b"trailer", b"transfer-encoding", b"upgrade"))


def end_to_end(headers):
    excluded = set(HOP_HEADERS)
    for key, value in headers:
        if key.lower() == b"connection":
            excluded.update(part.strip().lower() for part in value.split(b","))
    return [(key, value) for key, value in headers if key.lower() not in excluded]


def create_hosted_app(*, root: Path = ROOT, frontend_path: Path | None = None,
                      legacy_script: Path | None = None, legacy_handler=None,
                      state_path: Path | None = None) -> FastAPI:
    if legacy_handler is None:
        if legacy_script is None:
            raise ValueError("The hosted service needs the existing site's handler script.")
        spec = importlib.util.spec_from_file_location("axon_existing_sites", legacy_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        legacy_handler = module.Handler

    axon = create_app(root=root, state_path=state_path, frontend_path=frontend_path)
    client = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal client
        # The old handler serves only the other hostnames, privately in this
        # process. Axon's UI/API execute directly in the 8080 ASGI listener.
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), legacy_handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True,
                                  name="existing-site-handler")
        thread.start()
        try:
            timeout = httpx.Timeout(connect=5, read=None, write=30, pool=5)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{server.server_port}",
                                        timeout=timeout, trust_env=False) as active:
                client = active
                async with axon.router.lifespan_context(axon):
                    yield
        finally:
            client = None
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            await asyncio.to_thread(thread.join, 5)

    hosted = FastAPI(title="Hosted Axon Lab", lifespan=lifespan,
                     docs_url=None, redoc_url=None, openapi_url=None)
    hosted.state.axon = axon

    async def dispatch(scope, receive, send):
        host = dict(scope.get("headers", [])).get(b"host", b"").decode("latin-1").split(":")[0].lower()
        if host in AXON_HOSTS:
            return await axon(scope, receive, send)
        if scope["type"] != "http":
            return await send({"type": "websocket.close", "code": 1008})
        request = Request(scope, receive)
        path = scope.get("raw_path", scope["path"].encode("utf-8"))
        if scope.get("query_string"):
            path += b"?" + scope["query_string"]
        try:
            outbound = client.build_request(request.method, path.decode("ascii"),
                        headers=end_to_end(scope["headers"]), content=request.stream())
            response = await client.send(outbound, stream=True)
        except httpx.HTTPError:
            return await JSONResponse({"code": "site_unavailable",
                                      "message": "The existing site handler is unavailable."},
                                     status_code=502)(scope, receive, send)

        async def body():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        reply = StreamingResponse(body(), status_code=response.status_code)
        reply.raw_headers = end_to_end(response.headers.raw)
        await reply(scope, receive, send)

    hosted.mount("/", dispatch)
    return hosted
