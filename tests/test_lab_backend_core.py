from copy import deepcopy

from fastapi.testclient import TestClient

from core.manifests import component_manifests, E0_REFERENCE_GRAPH
from lab.backend.app import create_app
from lab.backend.core_checks import check_core
from lab.backend.validation import validate_graph


def test_default_catalog_and_real_core_evidence_leave_training_locked(tmp_path):
    with TestClient(create_app(state_path=tmp_path / "db")) as client:
        ready = client.get("/api/v1/readiness").json()
        checks = {c["id"]: c for c in ready["checks"]}
        for key in ("core_adapter_import", "core_graph_registration", "core_checkpoint_roundtrip"):
            assert checks[key]["status"] == "passed", checks[key]
            assert len(checks[key]["source_hash"]) == 64
        assert ready["training_authorized"] is False
        assert checks["core_runtime"]["status"] == "not_verified"
        assert client.get("/api/v1/architectures").json()["items"] == []
        graph = deepcopy(E0_REFERENCE_GRAPH)
        graph["version"] = str(graph["version"])
        registered = client.post("/api/v1/architectures", json={**graph, "command_id": "e0"})
        assert registered.status_code == 201
        assert registered.json()["execution_eligible"] is True
        # Adapter availability does not authorize execution.
        assert client.post("/api/v1/runs", json={}).status_code == 422


def test_surface_conversion_requires_explicit_versioned_adapter():
    def component(identity, incoming, outgoing):
        return {"id": identity, "version": "1", "execution_eligible": True,
                "configuration_schema": {"type": "object"}, "ports": [
                    {"port_id": direction, "direction": direction, "meaning": "test",
                     "shape": [16], "dtype": "float32", "authority": "test", "surface": surface}
                    for direction, surface in (("input", incoming), ("output", outgoing))]}
    catalog = [component("source", "substrate-exact", "substrate-exact"),
               component("adapter", "substrate-exact", "free"), component("destination", "free", "free")]
    graph = {"nodes": [{"node_id": c["id"], "component_type": c["id"], "component_version": "1"} for c in catalog],
             "edges": [{"source": {"node_id": "source", "port_id": "output"},
                        "destination": {"node_id": "destination", "port_id": "input"}}]}
    assert not validate_graph(graph, catalog)["valid"]
    graph["edges"][0]["destination"]["node_id"] = "adapter"
    graph["edges"].append({"source": {"node_id": "adapter", "port_id": "output"},
                           "destination": {"node_id": "destination", "port_id": "input"}})
    assert validate_graph(graph, catalog)["execution_eligible"]
    graph["nodes"][1]["component_version"] = "unregistered"
    assert not validate_graph(graph, catalog)["valid"]
