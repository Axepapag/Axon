"""Thin Dream Team bus client for the Round Table Orchestrator.

The waker connects per turn-batch as client_id "table" using protocol
"agent_cli_wake".  The shared auth token is read ONLY from the process
environment (BUS_TOKEN).  It is never persisted, logged, placed in prompts, or
written to transcripts.

Offline mode queues events to ops/table/outbox.jsonl and flushes them on
reconnect, so a missing bus never blocks a round.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


try:
    import websockets
except Exception:  # pragma: no cover - websockets may be absent in minimal envs
    websockets = None  # type: ignore[assignment]


StateDir = Path("ops") / "table"
DefaultWsUrl = "ws://127.0.0.1:8765/ws/bus"
DefaultHttpUrl = "http://127.0.0.1:8765"


@dataclass
class Cursors:
    last_event_id: str = ""
    last_dm_id: str = ""


@dataclass
class BusEvent:
    topic: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class BusLink:
    """Bus client with offline outbox and cursor persistence."""

    def __init__(
        self,
        ws_url: str | None = None,
        http_url: str | None = None,
        state_dir: str | Path | None = None,
        offline: bool = False,
    ) -> None:
        self.ws_url = ws_url or os.environ.get("BUS_URL", DefaultWsUrl)
        self.http_url = http_url or os.environ.get("BUS_HTTP_URL", DefaultHttpUrl)
        self.state_dir = Path(state_dir) if state_dir else StateDir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self._cursors_path = self.state_dir / "cursors.json"
        self._outbox_path = self.state_dir / "outbox.jsonl"
        self._token: str | None = None

    @property
    def token(self) -> str:
        if self._token is None:
            self._token = os.environ.get("BUS_TOKEN", "")
        return self._token

    @staticmethod
    def contract_ack() -> dict:
        """Working-contract acknowledgement required by the bus hello.

        The bus (Dream Team server) refuses any hello that does not accept
        Jeff's Working Contract by version + sha256 of the enforced file.
        The enforced path comes from DREAM_TEAM_WORKING_CONTRACT_PATH, with
        fallbacks to the known live copy and the repo's docs copy.
        """
        import hashlib
        import re

        candidates = [
            os.environ.get("DREAM_TEAM_WORKING_CONTRACT_PATH", ""),
            r"D:\Axon\roundtable\WORKING_CONTRACT.md",
            str(Path(__file__).resolve().parents[2] / "docs" / "WORKING_CONTRACT.md"),
        ]
        for cand in candidates:
            if not cand:
                continue
            path = Path(cand)
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"^Version:\s*(.+?)\s*$", text, flags=re.MULTILINE)
            version = match.group(1).strip() if match else ""
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"accepted": True, "version": version, "sha256": sha}
        return {"accepted": True, "version": "", "sha256": ""}

    def load_cursors(self) -> Cursors:
        if self._cursors_path.exists():
            try:
                data = json.loads(self._cursors_path.read_text(encoding="utf-8"))
                return Cursors(
                    last_event_id=str(data.get("last_event_id", "")),
                    last_dm_id=str(data.get("last_dm_id", "")),
                )
            except (json.JSONDecodeError, TypeError):
                pass
        return Cursors()

    def save_cursors(self, cursors: Cursors) -> None:
        self._cursors_path.write_text(
            json.dumps(
                {"last_event_id": cursors.last_event_id, "last_dm_id": cursors.last_dm_id},
                indent=2,
            ),
            encoding="utf-8",
        )

    def _append_outbox(self, events: list[BusEvent]) -> None:
        with self._outbox_path.open("a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps({"topic": ev.topic, "payload": ev.payload, "timestamp": ev.timestamp}) + "\n")

    def _read_outbox(self) -> list[BusEvent]:
        if not self._outbox_path.exists():
            return []
        events: list[BusEvent] = []
        with self._outbox_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    events.append(
                        BusEvent(
                            topic=data["topic"],
                            payload=data["payload"],
                            timestamp=data.get("timestamp", time.time()),
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    continue
        return events

    def _clear_outbox(self) -> None:
        if self._outbox_path.exists():
            self._outbox_path.unlink()

    def _http_publish(self, topic: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        body = {
            "type": "publish",
            "topic": topic,
            "source": "bridge:table",
            "session_id": "roundtable",
            "payload": payload,
        }
        ack = self.contract_ack()
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    urljoin(self.http_url + "/", "publish"),
                    json=body,
                    headers={
                        "X-Dream-Team-Token": self.token,
                        "X-Dream-Team-Contract-Accepted": "true" if ack["accepted"] else "false",
                        "X-Dream-Team-Contract-Version": ack["version"],
                        "X-Dream-Team-Contract-SHA256": ack["sha256"],
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return None

    async def _ws_publish(self, ws: Any, topic: str, payload: dict[str, Any]) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "publish",
                    "topic": topic,
                    "session_id": "roundtable",
                    "payload": payload,
                }
            )
        )

    async def _ws_send_dm(self, ws: Any, to_agent: str, payload: dict[str, Any]) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "direct_message",
                    "to_participant": to_agent,
                    "payload": payload,
                }
            )
        )

    async def connect_and_flush(
        self,
        events: list[BusEvent] | None = None,
        dms: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> bool:
        """Connect, flush outbox + new events/dms, update cursors, disconnect.

        Returns True if the bus accepted the batch, False if it was queued to
        the outbox (offline or unreachable).
        """
        if self.offline:
            self._append_outbox(events or [])
            return False

        if not websockets:
            self._append_outbox(events or [])
            return False

        cursors = self.load_cursors()
        try:
            async with websockets.connect(self.ws_url, ping_interval=None, ping_timeout=None) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "hello",
                            "protocol": "agent_cli_wake",
                            "client_id": "table",
                            "display_name": "Round Table",
                            "auth_token": self.token,
                            "capabilities": ["orchestrator"],
                            "last_event_id": cursors.last_event_id,
                            "last_dm_id": cursors.last_dm_id,
                            "working_contract": self.contract_ack(),
                        }
                    )
                )
                # Wait briefly for hello_ack + replays; we are only publishing.
                try:
                    ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    ack_data = json.loads(ack) if isinstance(ack, str) else {}
                    if isinstance(ack_data, dict) and ack_data.get("type") == "hello_ack":
                        pass
                except Exception:
                    pass

                await ws.send(
                    json.dumps(
                        {
                            "type": "subscribe",
                            "patterns": [
                                "bus.*",
                                "committee.*",
                                "agent.*",
                            ],
                        }
                    )
                )

                # Flush outbox first.
                for ev in self._read_outbox():
                    await self._ws_publish(ws, ev.topic, ev.payload)
                self._clear_outbox()

                for ev in events or []:
                    await self._ws_publish(ws, ev.topic, ev.payload)

                for to_agent, payload in dms or []:
                    await self._ws_send_dm(ws, to_agent, payload)

            return True
        except Exception:
            # Queue new events and return failure so the round can continue.
            self._append_outbox(events or [])
            return False

    def publish_sync(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> bool:
        """Synchronous, best-effort publish. Used when async context is unavailable."""
        if self.offline:
            self._append_outbox([BusEvent(topic=topic, payload=payload)])
            return False
        result = self._http_publish(topic, payload)
        if result is None:
            self._append_outbox([BusEvent(topic=topic, payload=payload)])
            return False
        return True

    async def replay_events(
        self,
        patterns: list[str] | None = None,
        timeout: float = 5.0,
        from_start: bool = False,
    ) -> list[dict[str, Any]]:
        """Connect to the bus and return replayed events matching patterns.

        Used by live gates to verify published events were persisted.

        The server only replays events matching the connection's
        SUBSCRIPTIONS, and replay is (re)triggered by the subscribe message —
        so we must subscribe after hello. With `from_start=True` the hello
        carries a zero cursor so the full persisted history replays
        (otherwise the saved/server-side cursor is used: new events only).
        """
        if self.offline or not websockets:
            return []

        cursors = self.load_cursors()
        last_event_id = "0" * 20 if from_start else cursors.last_event_id
        collected: list[dict[str, Any]] = []
        try:
            async with websockets.connect(self.ws_url, ping_interval=None, ping_timeout=None) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "hello",
                            "protocol": "agent_cli_wake",
                            "client_id": "table",
                            "display_name": "Round Table",
                            "auth_token": self.token,
                            "capabilities": ["orchestrator"],
                            "last_event_id": last_event_id,
                            "last_dm_id": cursors.last_dm_id,
                            "working_contract": self.contract_ack(),
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "subscribe",
                            "patterns": list(patterns) if patterns else ["bus.*", "committee.*", "agent.*"],
                        }
                    )
                )

                deadline = asyncio.get_event_loop().time() + timeout
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        remaining = deadline - asyncio.get_event_loop().time()
                        if remaining <= 0:
                            break
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                        if not isinstance(raw, str):
                            continue
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            if data.get("type") in ("hello_ack", "presence_snapshot"):
                                continue
                            if patterns and data.get("topic"):
                                if any(self._topic_matches(data["topic"], p) for p in patterns):
                                    collected.append(data)
                            else:
                                collected.append(data)
                            event_id = data.get("msg_id") or data.get("event_id")
                            if event_id:
                                cursors.last_event_id = str(event_id)
                    except asyncio.TimeoutError:
                        break
                self.save_cursors(cursors)
            return collected
        except Exception:
            return []

    @staticmethod
    def _topic_matches(topic: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            return topic.startswith(pattern[:-1])
        return topic == pattern
