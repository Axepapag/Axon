"""Honest MCP adapter boundary for the Axon collaboration bus.

This module is an *optional* bridge: it exposes the bus as MCP-style tools
by translating tool calls into ordinary HTTP requests to a running bus.
It does not embed a full MCP server and does not require the `mcp` package
unless the caller chooses to import and wrap it.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


TOOLS: list[dict[str, Any]] = [
    {
        "name": "axon_bus_query_board",
        "description": "Read the current collaboration board.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "axon_bus_update_board",
        "description": "Add or update an objective, task, talking point, or question.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["objectives", "tasks", "talking_points", "questions"],
                },
                "value": {"type": "object"},
                "from_agent": {"type": "string"},
            },
            "required": ["field", "value"],
        },
    },
    {
        "name": "axon_bus_publish",
        "description": "Broadcast a status or event to all subscribers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "payload": {"type": "object"},
                "from_agent": {"type": "string"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "axon_bus_send_dm",
        "description": "Send a direct message to another agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to_agent": {"type": "string"},
                "content": {"type": "string"},
                "from_agent": {"type": "string"},
                "in_reply_to": {"type": "string"},
            },
            "required": ["to_agent", "content"],
        },
    },
]


class BusMcpAdapter:
    """Translates MCP tool calls into bus HTTP calls.

    The adapter is honest: if the `mcp` SDK is not available, it still
    provides the tool definitions and a synchronous call handler. A caller
    with the SDK can wrap `handle_tool_call` in an `mcp.server.Server`
    instance.
    """

    def __init__(self, base_url: str = "http://localhost:8765", agent_id: str = "mcp_adapter") -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.client: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(timeout=30.0)
        return self.client

    def tools(self) -> list[dict[str, Any]]:
        return TOOLS

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        headers = {"x-agent-id": arguments.get("from_agent") or self.agent_id}
        if name == "axon_bus_query_board":
            resp = self._client().get(f"{self.base_url}/board")
            resp.raise_for_status()
            return {"content": [{"type": "text", "text": json.dumps(resp.json(), indent=2)}]}
        if name == "axon_bus_update_board":
            body = {
                "field": arguments["field"],
                "value": arguments["value"],
                "from_agent": arguments.get("from_agent", self.agent_id),
            }
            resp = self._client().post(f"{self.base_url}/board", json=body)
            resp.raise_for_status()
            return {"content": [{"type": "text", "text": json.dumps(resp.json(), indent=2)}]}
        if name == "axon_bus_publish":
            body = {
                "topic": arguments["topic"],
                "payload": arguments.get("payload", {}),
                "from_agent": arguments.get("from_agent", self.agent_id),
            }
            resp = self._client().post(f"{self.base_url}/broadcast", json=body)
            resp.raise_for_status()
            return {"content": [{"type": "text", "text": json.dumps(resp.json(), indent=2)}]}
        if name == "axon_bus_send_dm":
            body = {
                "to_agent": arguments["to_agent"],
                "content": arguments["content"],
                "from_agent": arguments.get("from_agent", self.agent_id),
                "in_reply_to": arguments.get("in_reply_to"),
            }
            resp = self._client().post(f"{self.base_url}/messages", json=body)
            resp.raise_for_status()
            return {"content": [{"type": "text", "text": json.dumps(resp.json(), indent=2)}]}
        raise ValueError(f"unknown tool: {name}")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
