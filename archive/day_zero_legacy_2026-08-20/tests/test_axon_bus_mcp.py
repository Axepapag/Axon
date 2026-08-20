"""MCP adapter boundary tests for the Axon collaboration bus."""
from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from bus.mcp_adapter import BusMcpAdapter
from bus.server import make_app


@pytest.fixture
def adapter(tmp_path) -> BusMcpAdapter:
    app = make_app(state_dir=tmp_path)
    client = TestClient(app)
    a = BusMcpAdapter(base_url="http://test", agent_id="test_adapter")
    a.client = client
    yield a
    a.close()


def test_tools_list(adapter: BusMcpAdapter) -> None:
    tools = adapter.tools()
    names = {t["name"] for t in tools}
    assert "axon_bus_query_board" in names
    assert "axon_bus_update_board" in names
    assert "axon_bus_publish" in names
    assert "axon_bus_send_dm" in names


def test_query_board(adapter: BusMcpAdapter) -> None:
    result = adapter.handle_tool_call("axon_bus_query_board", {})
    text = result["content"][0]["text"]
    data = json.loads(text)
    assert "tasks" in data


def test_update_board(adapter: BusMcpAdapter) -> None:
    result = adapter.handle_tool_call(
        "axon_bus_update_board",
        {"field": "tasks", "value": {"id": "t1", "text": "mcp task", "created_at": "now"}},
    )
    data = json.loads(result["content"][0]["text"])
    assert any(t["text"] == "mcp task" for t in data["tasks"])


def test_publish(adapter: BusMcpAdapter) -> None:
    result = adapter.handle_tool_call(
        "axon_bus_publish",
        {"topic": "status", "payload": {"state": "idle"}},
    )
    data = json.loads(result["content"][0]["text"])
    assert data["type"] == "broadcast"


def test_send_dm(adapter: BusMcpAdapter) -> None:
    result = adapter.handle_tool_call(
        "axon_bus_send_dm",
        {"to_agent": "kimi", "content": "from mcp"},
    )
    data = json.loads(result["content"][0]["text"])
    assert data["message"]["to_agent"] == "kimi"


def test_unknown_tool(adapter: BusMcpAdapter) -> None:
    with pytest.raises(ValueError):
        adapter.handle_tool_call("axon_bus_nope", {})
