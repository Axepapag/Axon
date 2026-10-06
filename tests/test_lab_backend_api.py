"""Exercise real API boundaries with isolated storage and controlled probe workers."""
from copy import deepcopy
import threading
import time

from fastapi.testclient import TestClient
import pytest

from lab.backend.app import create_app
from lab.backend.store import Registry
from substrate.native import ALPHABET


def wait_for(client, identity):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = client.get(f"/api/v1/operations/{identity}").json()
        if result["status"] in ("completed", "failed", "interrupted"):
            return result
        time.sleep(.01)
    pytest.fail("Probe operation did not reach a terminal state")


def report(**kwargs):
    return {"schema": "axon-foundation-preflight-v1", "foundation_passed": True,
            "training_authorized": False, "checks": [{"id": "device", "status": "pass",
            "details": {"selected_device": kwargs["device"]}}]}


def test_honest_capabilities_empty_registries_and_runtime_refusal(tmp_path):
    with TestClient(create_app(state_path=tmp_path / "db")) as client:
        caps = client.get("/api/v1/capabilities").json()
        assert caps["native_alphabet"] == ALPHABET
        assert "\n" in caps["native_alphabet"] and "`" not in caps["native_alphabet"]
        assert {c["id"] for c in caps["components"] if c["execution_eligible"]} == {
            "axon.substrate_input", "axon.core_reasoning_gru", "axon.response_state"}
        assert all(d["status"] == "unavailable" for d in caps["devices"])
        assert client.get("/api/v1/readiness").json()["training_authorized"] is False
        for kind in ("architectures", "datasets", "curricula", "runs", "checkpoints"):
            assert client.get(f"/api/v1/{kind}").json()["items"] == []
        rejected = client.post("/api/v1/runs", json={"command_id": "run-1"})
        assert rejected.status_code == 503
        assert rejected.json()["code"] == "not_integrated"
        assert client.get("/api/v1/backup/status").json()["status"] == "not_verified"
        assert client.get("/").status_code == 200
        assert "Trainer Control Center" in client.get("/").text


def test_probe_is_async_responsive_deduplicated_and_serialized(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(3)
        return report(**kwargs)

    with TestClient(create_app(state_path=tmp_path / "db", preflight=blocked)) as client:
        try:
            first = client.post("/api/v1/preflight", json={"command_id": "one", "device": "cpu"})
            assert first.status_code == 202 and first.json()["status"] == "queued"
            assert entered.wait(1)
            assert client.get("/api/v1/capabilities").status_code == 200
            retry = client.post("/api/v1/preflight", json={"command_id": "one", "device": "cpu"})
            assert retry.json()["operation_id"] == first.json()["operation_id"]
            conflict = client.post("/api/v1/preflight", json={"command_id": "one", "device": "cuda:0"})
            assert conflict.status_code == 409 and conflict.json()["code"] == "command_conflict"
            busy = client.post("/api/v1/preflight", json={"command_id": "two", "device": "cpu"})
            assert busy.status_code == 409 and busy.json()["code"] == "preflight_busy"
        finally:
            release.set()
        result = wait_for(client, first.json()["operation_id"])
        assert result["result"]["schema"] == "axon-foundation-preflight-v1"
        assert len(calls) == 1
        ready = client.get("/api/v1/readiness").json()
        assert ready["checks"][0]["status"] == "passed"
        assert ready["training_authorized"] is False
        assert client.get("/api/v1/capabilities").json()["devices"][0]["status"] == "available"


def test_failed_probe_releases_worker_and_does_not_reuse_stale_pass(tmp_path):
    calls = 0

    def probe(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Device probe failed")
        return report(**kwargs)

    with TestClient(create_app(state_path=tmp_path / "db", preflight=probe)) as client:
        for identity in ("passed", "failed", "recovered"):
            op = client.post("/api/v1/preflight", json={"command_id": identity, "device": "cpu"})
            assert op.status_code == 202
            result = wait_for(client, op.json()["operation_id"])
            if identity == "failed":
                assert result["status"] == "failed"
                assert client.get("/api/v1/readiness").json()["checks"][0]["status"] == "failed"
                assert client.get("/api/v1/capabilities").json()["devices"][0]["status"] == "unavailable"
            else:
                assert result["status"] == "completed"


def test_completed_operation_survives_service_restart(tmp_path):
    db = tmp_path / "db"
    with TestClient(create_app(state_path=db, preflight=report)) as client:
        first = client.post("/api/v1/preflight", json={"command_id": "persistent", "device": "cpu"}).json()
        result = wait_for(client, first["operation_id"])
    with TestClient(create_app(state_path=db, preflight=report)) as client:
        assert client.get(f'/api/v1/operations/{first["operation_id"]}').json() == result
        retry = client.post("/api/v1/preflight", json={"command_id": "persistent", "device": "cpu"}).json()
        assert retry == result


def test_unfinished_operation_is_interrupted_on_restart(tmp_path):
    db = tmp_path / "db"
    registry = Registry(db)
    registry.begin_command("crashed", {"kind": "preflight"},
                           {"operation_id": "orphan", "kind": "preflight", "status": "running"})
    with TestClient(create_app(state_path=db)) as client:
        assert client.get("/api/v1/operations/orphan").json()["status"] == "interrupted"
        assert client.get("/api/v1/readiness").json()["checks"][0]["status"] == "failed"


def component(identity, meaning, direction):
    return {"id": identity, "name": identity, "version": "1", "status": "experimental",
            "execution_eligible": False,
            "configuration_schema": {"type": "object", "properties": {"width": {"const": 16}},
                                     "required": ["width"], "additionalProperties": False},
            "ports": [{"port_id": "text", "direction": direction, "meaning": meaning,
                       "shape": [16], "dtype": "float32", "authority": "private", "surface": "substrate-exact"}]}


def graph():
    return {"name": "Test-only graph", "version": "1", "nodes": [
        {"node_id": "a", "component_type": "reader", "component_version": "1", "config": {"width": 16}},
        {"node_id": "b", "component_type": "worker", "component_version": "1", "config": {"width": 16}}],
        "edges": [{"source": {"node_id": "a", "port_id": "text"},
                   "destination": {"node_id": "b", "port_id": "text"}}]}


def test_validation_trusts_catalog_and_registry_is_durable(tmp_path):
    catalog = [component("reader", "exact_substrate", "output"), component("worker", "exact_substrate", "input")]
    db = tmp_path / "db"
    with TestClient(create_app(state_path=db, components=catalog)) as client:
        valid = client.post("/api/v1/architectures/validate", json=graph()).json()
        assert valid["valid"] and not valid["execution_eligible"]
        proposed = {**graph(), "command_id": "register-1"}
        saved = client.post("/api/v1/architectures", json=proposed)
        assert saved.status_code == 201
        registered = saved.json()
        assert client.post("/api/v1/architectures", json=proposed).json() == registered
        conflict = client.post("/api/v1/architectures", json={**proposed, "name": "Different"})
        assert conflict.status_code == 409
        assert client.get("/api/v1/architectures").json()["items"] == [registered]
        wrong = graph()
        wrong["nodes"][0]["config"]["width"] = 2048
        assert not client.post("/api/v1/architectures/validate", json=wrong).json()["valid"]
        wrong = graph()
        wrong["nodes"][0]["component_version"] = "2"
        assert not client.post("/api/v1/architectures/validate", json=wrong).json()["valid"]
    with TestClient(create_app(state_path=db, components=catalog)) as client:
        assert client.get(f'/api/v1/architectures/{registered["architecture_id"]}').json() == registered
        assert len(registered["architecture_hash"]) == 64


@pytest.mark.parametrize("field,value", [("meaning", "learned_state"), ("shape", [2048]), ("dtype", "float16"), ("authority", "canonical"), ("surface", "free"), ("surface", None)])
def test_edges_reject_implicit_conversion_and_forged_ports(tmp_path, field, value):
    catalog = [component("reader", "exact_substrate", "output"), component("worker", "exact_substrate", "input")]
    catalog[1]["ports"][0][field] = value
    proposed = graph()
    proposed["nodes"][1]["ports"] = deepcopy(catalog[0]["ports"])
    with TestClient(create_app(state_path=tmp_path / "db", components=catalog)) as client:
        result = client.post("/api/v1/architectures/validate", json=proposed).json()
        assert not result["valid"]
        assert any(field in error["message"] for error in result["errors"])


@pytest.mark.parametrize("bad", [[], {}, None])
def test_malformed_node_types_return_validation_errors(tmp_path, bad):
    proposed = graph()
    proposed["nodes"][0]["component_type"] = bad
    with TestClient(create_app(state_path=tmp_path / "db")) as client:
        response = client.post("/api/v1/architectures/validate", json=proposed)
        assert response.status_code == 200 and not response.json()["valid"]


def test_inventory_is_evidence_not_automatic_dataset_admission(tmp_path):
    import json
    path = tmp_path / "State/curriculum_inventory/legacy_raw_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"file_count": 37, "total_bytes": 30137542}), encoding="utf-8")
    with TestClient(create_app(root=tmp_path, state_path=tmp_path / "db")) as client:
        inventory = client.get("/api/v1/inventory").json()
        assert inventory["file_count"] == 37
        assert inventory["status"] == "inventoried_not_training_ready"
        assert client.get("/api/v1/datasets").json()["items"] == []
