"""Graph validation against trusted component contracts, never caller-supplied ports."""
from __future__ import annotations

from jsonschema import Draft202012Validator


def validate_graph(graph: dict, components: list[dict]) -> dict:
    errors, warnings = [], []
    catalog = {c["id"]: c for c in components}
    nodes = graph.get("nodes")
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not nodes:
        return {"valid": False, "execution_eligible": False, "errors": [{"path": "nodes", "message": "Add at least one component."}], "warnings": []}
    if not isinstance(edges, list) or len(nodes) > 256 or len(edges) > 1024:
        return {"valid": False, "execution_eligible": False, "errors": [{"path": "graph", "message": "Graph is malformed or exceeds the 256-node/1024-edge limit."}], "warnings": []}
    known = {}
    executable = True
    for index, node in enumerate(nodes):
        path = f"nodes/{index}"
        if not isinstance(node, dict) or not isinstance(node.get("node_id"), str) or not node["node_id"]:
            errors.append({"path": path, "message": "Every node needs a nonempty node_id."})
            continue
        if node["node_id"] in known:
            errors.append({"path": path, "message": "Duplicate node_id."})
            continue
        component_type = node.get("component_type")
        component = catalog.get(component_type) if isinstance(component_type, str) else None
        if not component or not component.get("configuration_schema"):
            errors.append({"path": path, "message": "This component has no integrated configuration contract."})
            executable = False
            continue
        if node.get("component_version") != component.get("version"):
            errors.append({"path": path, "message": "Component version does not match its registered contract."})
        config = node.get("config", {})
        validator = Draft202012Validator(component["configuration_schema"])
        for error in validator.iter_errors(config):
            errors.append({"path": path + "/config/" + "/".join(map(str, error.path)), "message": error.message})
        known[node["node_id"]] = component
        if not component.get("execution_eligible", False):
            executable = False
            warnings.append({"path": path, "message": "Configuration can be saved; its execution adapter is not integrated."})
    occupied = set()
    for index, edge in enumerate(edges):
        path = f"edges/{index}"
        if not isinstance(edge, dict):
            errors.append({"path": path, "message": "An edge must be an object."})
            continue
        resolved = []
        for endpoint, direction in (("source", "output"), ("destination", "input")):
            point = edge.get(endpoint, {})
            node_id = point.get("node_id") if isinstance(point, dict) else None
            component = known.get(node_id) if isinstance(node_id, str) else None
            port = next((p for p in component.get("ports", []) if p["port_id"] == point.get("port_id") and p["direction"] == direction), None) if component else None
            resolved.append(port)
        source, destination = resolved
        if not source or not destination:
            errors.append({"path": path, "message": "Edge must connect a registered output port to an input port."})
            continue
        for field in ("meaning", "dtype", "shape", "authority", "surface"):
            if source.get(field) != destination.get(field):
                errors.append({"path": path, "message": f"Port {field} mismatch; no implicit conversion is permitted."})
        if source.get("surface") not in ("substrate-exact", "free") or destination.get("surface") not in ("substrate-exact", "free"):
            errors.append({"path": path, "message": "Both ports must declare substrate-exact or free surface in their versioned component contract."})
        target = (edge["destination"]["node_id"], edge["destination"]["port_id"])
        if target in occupied:
            errors.append({"path": path, "message": "An input needs an explicit combining component before multiple connections."})
        occupied.add(target)
    if len(nodes) > 1 and not edges:
        warnings.append({"path": "edges", "message": "These components are disconnected; card order does not establish connections."})
        executable = False
    return {"valid": not errors, "execution_eligible": not errors and executable,
            "errors": errors, "warnings": warnings}
