"""Hermes WebSocket bus client — connects, subscribes, listens."""
import asyncio
import hashlib
import json
from pathlib import Path
import websockets

BUS_URL = "ws://127.0.0.1:8765/ws/bus"
BUS_TOKEN = "ddRWw2AfQseyiRrwHKzG95EBT-pWeoJTA27_NY__YOs"
CLIENT_ID = "hermes"
DISPLAY_NAME = "Hermes"
CONTRACT_PATH = Path(r"D:\Axon\roundtable\WORKING_CONTRACT.md")
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()

async def main():
    async with websockets.connect(BUS_URL) as ws:
        # Hello
        await ws.send(json.dumps({
            "type": "hello",
            "protocol": "agent_online",
            "client_id": CLIENT_ID,
            "display_name": DISPLAY_NAME,
            "auth_token": BUS_TOKEN,
            "capabilities": ["architecture_review", "coding", "roundtable"],
            "last_event_id": "",
            "last_dm_id": "",
            "working_contract": {
                "accepted": True,
                "version": "1.1",
                "sha256": CONTRACT_SHA256,
            },
        }))
        print(">>> hello sent")

        ack = await asyncio.wait_for(ws.recv(), timeout=10)
        ack_data = json.loads(ack)
        print(f"<<< {ack_data.get('type', '?')}: {json.dumps(ack_data, indent=2)[:500]}")

        # Subscribe
        await ws.send(json.dumps({
            "type": "subscribe",
            "patterns": ["bus.*", "committee.*", "agent.hermes.*", "tool.*", "knowledge.*"],
        }))
        print(">>> subscribed")

        # Publish ready
        await ws.send(json.dumps({
            "type": "publish",
            "topic": "committee.participant.ready",
            "session_id": "roundtable",
            "payload": {
                "round_id": "dream-team-bus",
                "client_id": "hermes",
                "status": "ready",
                "model": "glm-5.2:cloud",
                "note": "Hermes connected via WebSocket. Both deltas published. Standing by.",
            },
        }))
        print(">>> ready published")
        print("--- listening (300s timeout) ---")

        count = 0
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=300)
                data = json.loads(msg)
                count += 1
                etype = data.get("type", "?")
                topic = data.get("topic", "")
                source = data.get("source", "")
                print(f"  [{count}] {etype} | {topic} | {source}")
        except asyncio.TimeoutError:
            print(f"--- timed out after 300s, received {count} events ---")
        except Exception as e:
            print(f"  error: {e}")

asyncio.run(main())