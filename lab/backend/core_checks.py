"""Bounded CPU-only Core evidence, isolated from operator registries and RNG."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
IDS = ("core_adapter_import", "core_graph_registration", "core_checkpoint_roundtrip")


def check_core() -> list[dict]:
    # Cache only while the exact inspected source version remains unchanged.
    paths = ("core/e0_two_state.py", "core/manifests.py", "lab/backend/core_checks.py",
             "lab/backend/validation.py", "lab/backend/app.py", "lab/backend/store.py", "lab/backend/runs.py")
    digest = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in paths)).hexdigest()
    return json.loads(_checked(digest))


@lru_cache(maxsize=4)
def _checked(source_hash: str) -> str:
    try:
        reply = subprocess.run([sys.executable, "-m", "lab.backend.core_checks"],
                               cwd=ROOT, capture_output=True, text=True, timeout=90,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        if reply.returncode:
            raise RuntimeError("Core evidence subprocess failed: " + reply.stderr[-1000:])
        checks = json.loads(reply.stdout)
    except Exception as exc:
        checks = [{"id": key, "status": "failed", "reason": str(exc)} for key in IDS]
    for check in checks:
        check["source_hash"] = source_hash
    return json.dumps(checks)


def probe() -> list[dict]:
    from copy import deepcopy
    import datetime as dt
    import tempfile
    from fastapi.testclient import TestClient
    import torch
    from core.e0_two_state import E0TwoStateCore, save_checkpoint, load_checkpoint
    from core.manifests import component_manifests, E0_REFERENCE_GRAPH
    from .app import create_app

    checks = [{"id": IDS[0], "status": "passed", "reason": "E0 adapter imported in an isolated CPU process."}]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="axon-core-check-") as folder:
        base = Path(folder)
        graph = deepcopy(E0_REFERENCE_GRAPH)
        # API architecture versions are text; component versions remain unmodified.
        graph["version"] = str(graph["version"])
        with TestClient(create_app(state_path=base / "registry.sqlite3",
                                  components=component_manifests(), core_probe=lambda: [])) as client:
            validation = client.post("/api/v1/architectures/validate", json=graph).json()
            assert validation["valid"] and validation["execution_eligible"], validation
            registered = client.post("/api/v1/architectures", json={**graph, "command_id": "core-smoke"})
            assert registered.status_code == 201, registered.text
            saved = registered.json()
            assert client.get("/api/v1/architectures/" + saved["architecture_id"]).json() == saved
        checks.append({"id": IDS[1], "status": "passed",
                       "reason": "D512 reference graph validates and registers in a disposable registry; architecture version normalized to text.",
                       "architecture_hash": saved["architecture_hash"]})
        core = E0TwoStateCore().eval()
        with torch.no_grad():
            ids = torch.tensor([0], dtype=torch.int64)
            states = core.step(ids, *core.initial_states(1))[:2]
            path = save_checkpoint(base / "smoke.pt", core, *states, core.config())
            restored, reasoning, response, config = load_checkpoint(path, map_location="cpu")
            assert config == core.config()
            assert all(torch.equal(value, restored.state_dict()[key]) for key, value in core.state_dict().items())
            assert torch.equal(states[0], reasoning) and torch.equal(states[1], response)
            assert all(torch.equal(a, b) for a, b in zip(core.step(ids, *states), restored.step(ids, reasoning, response)))
        checks.append({"id": IDS[2], "status": "passed",
                       "reason": "D512 weights, both states and next-tick outputs restore bit-identically on CPU. Heart/draft/replay and optimizer recovery remain pending."})
    for check in checks:
        check["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        check["scope"] = "E0 D512 CPU smoke only; no training authorization"
    return checks


if __name__ == "__main__":
    print(json.dumps(probe()))
