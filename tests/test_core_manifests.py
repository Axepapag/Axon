"""Contract tests for the Core-owned component manifests and the E0 reference graph."""
import json

from jsonschema import Draft202012Validator

from core.manifests import COMPONENT_SCHEMA_VERSION, E0_REFERENCE_GRAPH, component_manifests
from lab.backend.validation import validate_graph

TOP_LEVEL_FIELDS = {"id", "name", "version", "status", "reason", "execution_eligible", "configuration_schema",
                    "ports", "constraints", "states"}
PORT_FIELDS = {"port_id", "direction", "meaning", "dtype", "shape", "width", "stateful", "authority", "surface",
               "constraints"}
STATE_FIELDS = {"role", "initialization", "reset", "retention", "checkpoint"}
SURFACES = {"substrate-exact", "free"}
CODEBOOKS = {"native95", "native95+empty95"}


def manifests_by_id():
    return {m["id"]: m for m in component_manifests()}


def test_package_re_exports_the_core_contract():
    import core
    assert core.COMPONENT_SCHEMA_VERSION == COMPONENT_SCHEMA_VERSION
    assert core.component_manifests() == component_manifests()
    assert core.E0_REFERENCE_GRAPH == E0_REFERENCE_GRAPH


def test_schema_version_and_manifest_shape():
    assert COMPONENT_SCHEMA_VERSION == "axon-core-components-v1"
    manifests = component_manifests()
    assert {m["id"] for m in manifests} == {"axon.substrate_input", "axon.core_reasoning_gru",
                                            "axon.response_state"}
    for manifest in manifests:
        assert TOP_LEVEL_FIELDS <= set(manifest), manifest["id"]
        assert manifest["version"] == ('0.1.2' if manifest['id']=='axon.response_state' else '0.1.1')
        assert manifest["status"] == "experimental"
        assert manifest["execution_eligible"] is True
        assert isinstance(manifest["ports"], list) and manifest["ports"]
        for port in manifest["ports"]:
            assert PORT_FIELDS <= set(port), (manifest["id"], port["port_id"])
            assert port["direction"] in ("input", "output")
            assert isinstance(port["shape"], list) and port["shape"]
            assert isinstance(port["constraints"], dict)
        for state in manifest["states"]:
            assert STATE_FIELDS <= set(state), manifest["id"]


def test_states_are_declared_where_the_design_requires_them():
    manifests = manifests_by_id()
    assert manifests["axon.substrate_input"]["states"] == []
    assert [(s["role"], s["checkpoint"]) for s in manifests["axon.core_reasoning_gru"]["states"]] == [
        ("reasoning/memory", "included")]
    assert [(s["role"], s["checkpoint"]) for s in manifests["axon.response_state"]["states"]] == [
        ("response-composition", "included")]


def test_configuration_schemas_compile_and_accept_e0_configs():
    manifests = manifests_by_id()
    for manifest in manifests.values():
        Draft202012Validator.check_schema(manifest["configuration_schema"])
    for node in E0_REFERENCE_GRAPH["nodes"]:
        manifest = manifests[node["component_type"]]
        assert node["component_version"] == manifest["version"]
        assert list(Draft202012Validator(manifest["configuration_schema"]).iter_errors(node["config"])) == []


def test_e0_reference_graph_passes_the_backend_validator():
    result = validate_graph(E0_REFERENCE_GRAPH, component_manifests())
    assert result["errors"] == []
    assert result["valid"] is True
    assert result["execution_eligible"] is True


def test_edge_matched_ports_agree_field_for_field():
    graph = E0_REFERENCE_GRAPH
    manifests = manifests_by_id()
    ports = {m["id"]: {(p["port_id"], p["direction"]): p for p in m["ports"]} for m in manifests.values()}
    for edge in graph["edges"]:
        node = {n["node_id"]: n["component_type"] for n in graph["nodes"]}
        source = ports[node[edge["source"]["node_id"]]][(edge["source"]["port_id"], "output")]
        destination = ports[node[edge["destination"]["node_id"]]][(edge["destination"]["port_id"], "input")]
        for field in ("meaning", "dtype", "shape", "authority"):
            assert source[field] == destination[field], field


def test_substrate_exact_ports_declare_codebook_and_zero_tolerance():
    for manifest in component_manifests():
        for port in manifest["ports"]:
            assert port["surface"] in SURFACES, (manifest["id"], port["port_id"])
            if port["surface"] != "substrate-exact":
                continue
            assert port["constraints"].get("codebook") in CODEBOOKS, (manifest["id"], port["port_id"])
            if port["meaning"] == "frozen-d16-cell-surface":
                assert port["constraints"].get("tolerance") == 0.0, (manifest["id"], port["port_id"])
                assert port["dtype"] == "float32"


def test_proposal_and_write_authority_stay_separated():
    manifests = manifests_by_id()
    assert [p["authority"] for p in manifests["axon.substrate_input"]["ports"] if p["direction"] == "input"] == [
        "heart"]
    for component in ("axon.core_reasoning_gru", "axon.response_state"):
        for port in manifests[component]["ports"]:
            if port["direction"] == "output":
                assert port["authority"] in ("core-internal", "proposes-only")
            else:
                assert port["authority"] in ("frozen-substrate", "core-internal")


def test_manifests_survive_a_json_round_trip():
    manifests = component_manifests()
    assert json.loads(json.dumps(manifests)) == manifests
    assert json.loads(json.dumps(E0_REFERENCE_GRAPH)) == E0_REFERENCE_GRAPH


def test_graph_version_is_text_and_configs_match_fixed_port_shapes():
    # Codex handoff regressions: registration requires a text version, and
    # configurable values must not diverge from the fixed trusted port shapes.
    assert isinstance(E0_REFERENCE_GRAPH["version"], str)
    manifests = manifests_by_id()
    for node in E0_REFERENCE_GRAPH["nodes"]:
        schema = manifests[node["component_type"]]["configuration_schema"]
        for key, value in node["config"].items():
            allowed = schema["properties"][key].get("enum")
            if allowed is not None:
                assert value in allowed, (node["node_id"], key)
            if key == "occupancy":
                width = node["config"]["width"]
                assert 1 <= value <= width // 16, (node["node_id"], key)


def test_occupancy_bounds_match_lane_count():
    # Codex coordination catch (2026-10-07): the schema maximum must equal the
    # lane count at the pinned width, never an unrelated constant.
    substrate_input = manifests_by_id()["axon.substrate_input"]
    props = substrate_input["configuration_schema"]["properties"]
    width = props["width"]["enum"]
    for w in width:
        assert props["occupancy"]["maximum"] == w // 16, (w, props["occupancy"]["maximum"])


def test_reference_graph_is_not_shared_mutable_state():
    first = component_manifests()
    first[0]["ports"][0]["constraints"]["codebook"] = "tampered"
    assert component_manifests()[0]["ports"][0]["constraints"]["codebook"] == "native95"
