"""Local registry storage. Live SQLite stays outside Google Drive synchronization."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def local_state(root: Path) -> Path:
    identity = hashlib.sha256(str(root.resolve()).casefold().encode()).hexdigest()[:12]
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local/share")))
    return base / "Axon" / "Lab" / identity


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS objects (kind TEXT, id TEXT, payload TEXT NOT NULL, PRIMARY KEY(kind,id))")
            db.execute("CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, digest TEXT NOT NULL, operation_id TEXT NOT NULL)")
            # Pending checks cannot be treated as completed after a process restart.
            for row in db.execute("SELECT id,payload FROM objects WHERE kind='operation'").fetchall():
                item = json.loads(row[1])
                if item["status"] in ("queued", "running"):
                    item.update(status="interrupted", error={"code": "service_restarted", "message": "The service restarted before this check completed."})
                    db.execute("UPDATE objects SET payload=? WHERE kind='operation' AND id=?", (canonical(item), row[0]))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, kind: str, identity: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM objects WHERE kind=? AND id=?", (kind, identity)).fetchone()
        return json.loads(row[0]) if row else None

    def items(self, kind: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM objects WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def put(self, kind: str, identity: str, value: dict, *, replace: bool = False):
        with self.lock, self.connect() as db:
            if replace:
                db.execute("INSERT INTO objects VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload", (kind, identity, canonical(value)))
            else:
                db.execute("INSERT INTO objects VALUES (?,?,?)", (kind, identity, canonical(value)))

    def begin_command(self, command_id: str, spec: dict, operation: dict) -> dict:
        digest = hashlib.sha256(canonical(spec).encode()).hexdigest()
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT digest,operation_id FROM commands WHERE id=?", (command_id,)).fetchone()
            if previous:
                if previous[0] != digest:
                    raise ValueError("command_id was already used with different contents")
                row = db.execute("SELECT payload FROM objects WHERE kind='operation' AND id=?", (previous[1],)).fetchone()
                return json.loads(row[0])
            db.execute("INSERT INTO commands VALUES (?,?,?)", (command_id, digest, operation["operation_id"]))
            db.execute("INSERT INTO objects VALUES ('operation',?,?)", (operation["operation_id"], canonical(operation)))
        return operation

    def register_architecture(self, command_id: str, graph: dict, saved: dict) -> dict:
        digest = hashlib.sha256(canonical({"kind": "register_architecture", "graph": graph}).encode()).hexdigest()
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT digest,operation_id FROM commands WHERE id=?", (command_id,)).fetchone()
            if prior:
                if prior[0] != digest:
                    raise ValueError("command_id was already used with different contents")
                row = db.execute("SELECT payload FROM objects WHERE kind='architecture' AND id=?", (prior[1],)).fetchone()
                return json.loads(row[0])
            db.execute("INSERT INTO objects VALUES ('architecture',?,?)", (saved["architecture_id"], canonical(saved)))
            db.execute("INSERT INTO commands VALUES (?,?,?)", (command_id, digest, saved["architecture_id"]))
        return saved
