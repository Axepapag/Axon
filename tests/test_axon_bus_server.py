"""HTTP + WebSocket server tests for the Axon collaboration bus."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from bus.server import make_app


@pytest.fixture
def client(tmp_path) -> TestClient:
    app = make_app(state_dir=tmp_path)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_get_board_empty(client: TestClient) -> None:
    resp = client.get("/board")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks"] == []


def test_post_board_patch(client: TestClient) -> None:
    resp = client.post(
        "/board",
        json={"field": "tasks", "value": {"id": "t1", "text": "wire ws", "created_at": "now"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tasks"]) == 1


def test_post_board_invalid_field(client: TestClient) -> None:
    resp = client.post("/board", json={"field": "nope", "value": {"id": "x"}})
    assert resp.status_code == 400


def test_post_messages_http_fallback(client: TestClient) -> None:
    resp = client.post(
        "/messages",
        json={"to_agent": "kimi", "content": "hello"},
        headers={"x-agent-id": "claude"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"]["to_agent"] == "kimi"


def test_get_messages_filter(client: TestClient) -> None:
    client.post(
        "/messages",
        json={"to_agent": "kimi", "content": "hello"},
        headers={"x-agent-id": "claude"},
    )
    resp = client.get("/messages", headers={"x-agent-id": "kimi"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_post_broadcast(client: TestClient) -> None:
    resp = client.post(
        "/broadcast",
        json={"topic": "status", "payload": {"state": "idle"}},
        headers={"x-agent-id": "claude"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "broadcast"


def test_get_events(client: TestClient) -> None:
    client.post(
        "/broadcast",
        json={"topic": "status", "payload": {}},
        headers={"x-agent-id": "claude"},
    )
    resp = client.get("/events")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_websocket_hello_ack(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "hello",
                "from_agent": "claude",
                "payload": {"agent_id": "claude", "capabilities": ["read"]},
            }
        )
        msg = ws.receive_json()
        assert msg["type"] == "hello_ack"
        assert "board_state" in msg["payload"]


def test_websocket_hello_timeout(client: TestClient) -> None:
    # TestClient's sync websocket doesn't wait easily; send bad first message.
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_websocket_dm_delivery(client: TestClient) -> None:
    # Connect recipient first.
    with client.websocket_connect("/ws") as ws_kimi:
        ws_kimi.send_json(
            {
                "type": "hello",
                "from_agent": "kimi",
                "payload": {"agent_id": "kimi", "capabilities": ["read"]},
            }
        )
        ws_kimi.receive_json()  # hello_ack

        # Send DM over HTTP.
        resp = client.post(
            "/messages",
            json={"to_agent": "kimi", "content": "hello"},
            headers={"x-agent-id": "claude"},
        )
        assert resp.status_code == 200
        assert resp.json()["message"]["delivered"] is True

        # Recipient should receive the DM.
        dm = ws_kimi.receive_json()
        assert dm["type"] == "direct_message"
        assert dm["payload"]["content"] == "hello"


def test_websocket_board_update_broadcast(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "hello",
                "from_agent": "kimi",
                "payload": {"agent_id": "kimi", "capabilities": ["read"]},
            }
        )
        ws.receive_json()  # hello_ack clears queue

        ws.send_json(
            {
                "type": "board_update",
                "from_agent": "kimi",
                "payload": {
                    "field": "objectives",
                    "value": {"id": "o1", "text": "ship it", "created_at": "now"},
                },
            }
        )
        msg = ws.receive_json()
        assert msg["type"] == "board_state"
        assert len(msg["payload"]["objectives"]) == 1


def test_websocket_replay(client: TestClient) -> None:
    # Seed a DM via HTTP.
    resp = client.post(
        "/messages",
        json={"to_agent": "kimi", "content": "first"},
        headers={"x-agent-id": "claude"},
    )
    first_id = resp.json()["message"]["msg_id"]

    # Reconnect with last_dm_id to replay only newer messages.
    client.post(
        "/messages",
        json={"to_agent": "kimi", "content": "second"},
        headers={"x-agent-id": "claude"},
    )

    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "hello",
                "from_agent": "kimi",
                "payload": {
                    "agent_id": "kimi",
                    "capabilities": ["read"],
                    "last_dm_id": first_id,
                },
            }
        )
        ack = ws.receive_json()
        assert ack["type"] == "hello_ack"
        replay = ack["payload"]["replay"]["missed_dms"]
        assert len(replay) >= 1
        contents = {m["content"] for m in replay}
        assert "second" in contents
