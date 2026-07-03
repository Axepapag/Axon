"""ASGI HTTP + WebSocket sidecar server for the Axon collaboration bus."""
from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from .bus import Bus
from .registry import Participant
from .schema import (
    BroadcastPayload,
    DirectMessagePayload,
    HelloPayload,
    MessageEnvelope,
    StatusPayload,
)


HELLO_TIMEOUT = 5.0
HEARTBEAT_INTERVAL = 15.0
HEARTBEAT_TIMEOUT = 30.0


class BusServer:
    """Binds the Bus to Starlette HTTP/WebSocket routes."""

    def __init__(self, bus: Bus) -> None:
        self.bus = bus

    # -----------------------------------------------------------------------
    # HTTP handlers
    # -----------------------------------------------------------------------
    async def health(self, request: Request) -> JSONResponse:
        return JSONResponse(self.bus.health())

    async def get_board(self, request: Request) -> JSONResponse:
        return JSONResponse(self.bus.store.get().model_dump())

    async def post_board(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        if "field" in body:
            try:
                state = self.bus.patch_board(
                    field=body["field"], value=body["value"], from_agent=body.get("from_agent", "")
                )
            except (ValueError, KeyError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        else:
            try:
                state = self.bus.overwrite_board(data=body, from_agent=body.get("from_agent", ""))
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        # Notify WebSocket participants of board change.
        await self._broadcast_board_state(state)
        return JSONResponse(state.model_dump())

    async def get_participants(self, request: Request) -> JSONResponse:
        return JSONResponse(self.bus.registry.snapshot())

    async def get_messages(self, request: Request) -> JSONResponse:
        with_agent = request.query_params.get("with")
        after_id = request.query_params.get("after_id")
        limit = _int_param(request.query_params.get("limit"), 50)
        # In HTTP fallback, the recipient is inferred from a header.
        recipient = request.headers.get("x-agent-id", "")
        dms = self.bus.direct_messages_for(
            agent_id=recipient, with_agent=with_agent, after_id=after_id, limit=limit
        )
        return JSONResponse([dm.model_dump() for dm in dms])

    async def post_messages(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from_agent = body.get("from_agent", request.headers.get("x-agent-id", "anonymous"))
        try:
            payload = DirectMessagePayload.model_validate(body)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        dm, delivered = self.bus.send_direct_message(payload, from_agent=from_agent)
        ack = MessageEnvelope(
            type="dm_ack",
            from_agent="bus",
            to_agent=from_agent,
            payload={
                "msg_id": dm.msg_id,
                "delivered_now": delivered,
                "to_agent": dm.to_agent,
            },
        )
        self.bus._log_event(ack)
        return JSONResponse({"message": dm.model_dump(), "ack": ack.model_dump()})

    async def post_broadcast(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        try:
            bp = BroadcastPayload.model_validate(body)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        from_agent = body.get("from_agent", request.headers.get("x-agent-id", "anonymous"))
        env = self.bus.broadcast(topic=bp.topic, payload=bp.payload.model_dump() if hasattr(bp.payload, "model_dump") else bp.payload, from_agent=from_agent)
        await self._forward_broadcast(env)
        return JSONResponse(env.model_dump())

    async def get_events(self, request: Request) -> JSONResponse:
        after_id = request.query_params.get("after_id")
        limit = _int_param(request.query_params.get("limit"), 200)
        return JSONResponse(self.bus.recent_events(after_id=after_id, limit=limit))

    # -----------------------------------------------------------------------
    # WebSocket handler
    # -----------------------------------------------------------------------
    async def ws_endpoint(self, websocket: WebSocket) -> None:
        await websocket.accept()
        participant: Participant | None = None
        receive_task: asyncio.Task[Any] | None = None
        send_task: asyncio.Task[Any] | None = None
        try:
            # Wait for hello.
            try:
                raw = await asyncio.wait_for(websocket.receive_json(), timeout=HELLO_TIMEOUT)
            except asyncio.TimeoutError:
                await self._send(websocket, {"type": "error", "payload": {"message": "hello timeout"}})
                await websocket.close(code=4001)
                return
            except Exception:
                await websocket.close(code=4000)
                return

            if raw.get("type") != "hello":
                await self._send(websocket, {"type": "error", "payload": {"message": "expected hello"}})
                await websocket.close(code=4002)
                return

            try:
                hello = HelloPayload.model_validate(raw.get("payload", {}))
            except Exception as exc:
                await self._send(websocket, {"type": "error", "payload": {"message": str(exc)}})
                await websocket.close(code=4003)
                return

            participant, replay_info = self.bus.connect(
                agent_id=hello.agent_id,
                capabilities=hello.capabilities,
                last_event_id=hello.last_event_id,
                last_dm_id=hello.last_dm_id,
            )
            hello_ack = MessageEnvelope(
                type="hello_ack",
                from_agent="bus",
                to_agent=hello.agent_id,
                payload={
                    "board_state": self.bus.store.get().model_dump(),
                    "replay": replay_info,
                },
            )
            await self._send(websocket, hello_ack.model_dump())
            # Replay missed DMs directly.
            for dm_raw in replay_info["missed_dms"]:
                await self._send(websocket, {"type": "direct_message", "payload": dm_raw})

            # Start producer/consumer tasks.
            receive_task = asyncio.create_task(self._ws_receive_loop(websocket, participant))
            send_task = asyncio.create_task(self._ws_send_loop(websocket, participant))
            done, pending = await asyncio.wait(
                [receive_task, send_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                try:
                    task.result()
                except Exception:
                    pass
        finally:
            if participant is not None:
                self.bus.disconnect(participant.agent_id)
                await self._broadcast_board_state(self.bus.store.get())
            try:
                await websocket.close()
            except Exception:
                pass

    async def _ws_receive_loop(self, websocket: WebSocket, participant: Participant) -> None:
        while True:
            try:
                raw = await websocket.receive_json()
            except Exception:
                break
            try:
                env = MessageEnvelope.model_validate(raw)
            except Exception as exc:
                await self._send(websocket, {"type": "error", "payload": {"message": str(exc)}})
                continue
            participant.last_pong_at = asyncio.get_running_loop().time()
            if env.from_agent and env.from_agent != participant.agent_id:
                participant.agent_id = env.from_agent
            await self._handle_client_message(websocket, participant, env)

    async def _handle_client_message(
        self, websocket: WebSocket, participant: Participant, env: MessageEnvelope
    ) -> None:
        if env.type == "ping":
            await self._send(websocket, {"type": "pong", "payload": {}})
            return
        if env.type == "status":
            try:
                StatusPayload.model_validate(env.payload)
            except Exception as exc:
                await self._send(websocket, {"type": "error", "payload": {"message": str(exc)}})
                return
            self.bus._log_event(env)
            # Forward status to subscribers.
            await self._forward_broadcast(env)
            return
        if env.type == "board_update":
            field = env.payload.get("field")
            value = env.payload.get("value")
            if not field or not value:
                await self._send(websocket, {"type": "error", "payload": {"message": "board_update needs field and value"}})
                return
            try:
                state = self.bus.patch_board(field=field, value=value, from_agent=participant.agent_id)
            except Exception as exc:
                await self._send(websocket, {"type": "error", "payload": {"message": str(exc)}})
                return
            await self._broadcast_board_state(state)
            return
        if env.type == "broadcast":
            try:
                bp = BroadcastPayload.model_validate(env.payload)
            except Exception as exc:
                await self._send(websocket, {"type": "error", "payload": {"message": str(exc)}})
                return
            env.topic = bp.topic
            self.bus._log_event(env)
            await self._forward_broadcast(env)
            return
        if env.type == "direct_message":
            try:
                payload = DirectMessagePayload.model_validate(env.payload)
            except Exception as exc:
                await self._send(websocket, {"type": "error", "payload": {"message": str(exc)}})
                return
            dm, delivered = self.bus.send_direct_message(payload, from_agent=participant.agent_id)
            ack = MessageEnvelope(
                type="dm_ack",
                from_agent="bus",
                to_agent=participant.agent_id,
                payload={
                    "msg_id": dm.msg_id,
                    "delivered_now": delivered,
                    "to_agent": dm.to_agent,
                },
            )
            self.bus._log_event(ack)
            await self._send(websocket, ack.model_dump())
            return
        if env.type == "pong":
            return
        await self._send(
            websocket,
            {"type": "error", "payload": {"message": f"unsupported type {env.type}"}},
        )

    async def _ws_send_loop(self, websocket: WebSocket, participant: Participant) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(participant.queue.get(), timeout=HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                if asyncio.get_running_loop().time() - participant.last_pong_at > HEARTBEAT_TIMEOUT:
                    break
                try:
                    await self._send(websocket, {"type": "ping", "payload": {}})
                except Exception:
                    break
                continue
            try:
                await self._send(websocket, msg)
            except Exception:
                break

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    async def _send(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        await websocket.send_json(data)

    async def _forward_broadcast(self, env: MessageEnvelope) -> None:
        data = env.model_dump()
        for p in self.bus.matching_recipients(env.topic or "*"):
            try:
                p.queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    async def _broadcast_board_state(self, state: Any) -> None:
        self.bus.store.set_participants(self.bus.registry.snapshot())
        data = {
            "type": "board_state",
            "from_agent": "bus",
            "topic": "board",
            "payload": state.model_dump(),
        }
        for p in self.bus.registry.all():
            try:
                p.queue.put_nowait(data)
            except asyncio.QueueFull:
                pass


def _int_param(value: str | None, default: int) -> int:
    try:
        return int(value) if value else default
    except (TypeError, ValueError):
        return default


def make_app(state_dir: str | Path = "State/bus") -> Starlette:
    bus = Bus(Path(state_dir))
    server = BusServer(bus)
    routes = [
        Route("/health", server.health, methods=["GET"]),
        Route("/board", server.get_board, methods=["GET"]),
        Route("/board", server.post_board, methods=["POST"]),
        Route("/participants", server.get_participants, methods=["GET"]),
        Route("/messages", server.get_messages, methods=["GET"]),
        Route("/messages", server.post_messages, methods=["POST"]),
        Route("/broadcast", server.post_broadcast, methods=["POST"]),
        Route("/events", server.get_events, methods=["GET"]),
        WebSocketRoute("/ws", server.ws_endpoint),
    ]
    app = Starlette(routes=routes)
    app.state.bus = bus
    return app


app = make_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Axon Agent Collaboration Bus sidecar")
    parser.add_argument("--host", default="0.0.0.0", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    parser.add_argument("--state-dir", default="State/bus", help="directory for board/logs")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run(make_app(state_dir=state_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
