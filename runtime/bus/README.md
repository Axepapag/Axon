# Axon Agent Collaboration Bus

A small Starlette + Pydantic sidecar for multi-agent collaboration.

## Run

```bash
python -m axon_bus.server --host 0.0.0.0 --port 8765 --state-dir State/bus
```

Or with uvicorn directly:

```bash
uvicorn axon_bus.server:app --host 0.0.0.0 --port 8765
```

## State files

State is persisted under the configured `--state-dir`:

- `board.json` — current board snapshot
- `events.jsonl` — bus events (broadcasts, board updates, status, etc.)
- `messages.jsonl` — direct messages
- `participants.jsonl` — connect/disconnect audit trail

## HTTP routes

- `GET /health`
- `GET /board`, `POST /board`
- `GET /participants`
- `GET /messages`, `POST /messages`
- `POST /broadcast`
- `GET /events`

## WebSocket

Connect to `/ws` and send a `hello` message. The server replies with `hello_ack`
carrying the current board state and any missed events/DMs.

## MCP boundary

`axon_bus/mcp_adapter.py` exposes the bus as MCP-style tools by translating calls
into ordinary HTTP requests. It does not require the `mcp` SDK.
