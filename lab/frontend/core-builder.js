const REFERENCE_ONLY = "reference-only";

export function cloneValue(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

export function splitEndpoint(endpoint) {
  const value = String(endpoint || "");
  const index = value.indexOf(".");
  if (index < 1 || index === value.length - 1) return null;
  return [value.slice(0, index), value.slice(index + 1)];
}

export function nodeParameterCount(node) {
  const type = node?.type;
  const c = node?.config || {};
  const n = key => Number(c[key] ?? 0);
  if (type === "gru_cell") {
    const input = n("input_size");
    const hidden = n("hidden_size");
    return 3 * hidden * input + 3 * hidden * hidden + 6 * hidden;
  }
  if (type === "feed_forward") {
    const input = n("input_size");
    const hidden = n("hidden_size");
    const output = Number(c.output_size ?? input);
    return input * hidden + hidden + hidden * output + output;
  }
  if (type === "linear_head") return n("input_size") * n("output_size") + n("output_size");
  if (type === "character_head" || type === "response_head") return n("input_size") * 96 + 96;
  if (type === "control_wait_stage_end") return n("input_size") * 3 + 3;
  if (type === "decision_head") {
    const labels = Array.isArray(c.labels) ? c.labels.length : 0;
    return n("input_size") * labels + labels;
  }
  if (type === "routing_head") return n("input_size") * n("routes") + n("routes");
  if (type === "self_attention") {
    const embed = n("embed_dim");
    return 4 * embed * embed + 4 * embed;
  }
  if (type === "layer_norm") return 2 * n("size");
  if (type === "gated_fusion") {
    const size = n("size");
    return 2 * size * size + size;
  }
  return 0;
}

export function architectureAnatomy(spec, catalog = {}) {
  const nodes = Array.isArray(spec?.nodes) ? spec.nodes : [];
  const states = spec?.states && typeof spec.states === "object" ? spec.states : {};
  const byNode = {};
  let trainableParameters = 0;
  let unknownParameters = false;
  for (const node of nodes) {
    const count = nodeParameterCount(node);
    byNode[node.id] = count;
    trainableParameters += count;
    if (catalog[node.type]?.status === REFERENCE_ONLY) unknownParameters = true;
  }
  return {
    trainableParameters,
    unknownParameters,
    nodeCount: nodes.length,
    stateCount: Object.keys(states).length,
    byNode,
  };
}

function expectedInputWidth(type, config, port) {
  const c = config || {};
  if (type === "gru_cell") return Number(port === "x" ? c.input_size : c.hidden_size);
  if (type === "feed_forward" && port === "x") return Number(c.input_size);
  if (["linear_head","character_head","response_head","control_wait_stage_end","decision_head","routing_head"].includes(type) && port === "x") return Number(c.input_size);
  if (type === "self_attention" && port === "x") return Number(c.embed_dim);
  if (type === "layer_norm" && port === "x") return Number(c.size);
  if (type === "gated_fusion" && ["a","b"].includes(port)) return Number(c.size);
  if (type === "mamba_block") {
    if (port === "x") return Number(c.model_dim);
    if (port === "state") return Number(c.state_dim);
  }
  return null;
}

function fixedOutputWidth(type, config, port) {
  const c = config || {};
  if (type === "exact_substrate_input" && port === "surface") return Number(c.width);
  if (type === "gru_cell" && port === "state") return Number(c.hidden_size);
  if (type === "feed_forward" && port === "out") return Number(c.output_size ?? c.input_size);
  if (type === "linear_head" && port === "logits") return Number(c.output_size);
  if (["character_head","response_head"].includes(type) && port === "logits") return 96;
  if (type === "control_wait_stage_end" && port === "logits") return 3;
  if (type === "decision_head" && port === "logits") return Array.isArray(c.labels) ? c.labels.length : null;
  if (type === "routing_head" && port === "logits") return Number(c.routes);
  if (type === "self_attention" && port === "out") return Number(c.embed_dim);
  if (type === "layer_norm" && port === "out") return Number(c.size);
  if (type === "gated_fusion" && ["out","gate"].includes(port)) return Number(c.size);
  if (type === "mamba_block") {
    if (port === "out") return Number(c.model_dim);
    if (port === "state") return Number(c.state_dim);
  }
  return null;
}

export function validateCoreGraph(spec, catalog = {}) {
  const errors = [];
  const warnings = [];
  if (!spec || typeof spec !== "object") return {valid:false, executable:false, errors:["Graph is missing."], warnings};
  const nodes = Array.isArray(spec.nodes) ? spec.nodes : [];
  const states = spec.states && typeof spec.states === "object" ? spec.states : {};
  const inputs = spec.inputs && typeof spec.inputs === "object" ? spec.inputs : {};
  const nodeMap = new Map();
  const incoming = new Map();
  const adjacency = new Map();
  const widths = new Map();

  for (const [name, item] of Object.entries(inputs)) {
    const shape = item?.shape;
    widths.set(`input.${name}`, Array.isArray(shape) && Number.isFinite(shape.at(-1)) ? Number(shape.at(-1)) : null);
  }
  for (const [name, state] of Object.entries(states)) widths.set(`state.${name}`, Number(state?.size ?? 0) || null);

  for (const node of nodes) {
    if (!node?.id || String(node.id).includes(".")) errors.push(`Invalid node id: ${node?.id ?? "(missing)"}`);
    if (nodeMap.has(node?.id)) errors.push(`Duplicate node id: ${node.id}`);
    if (!catalog[node?.type]) errors.push(`${node?.id || "node"}: unknown part type ${node?.type || "(missing)"}`);
    if (catalog[node?.type]?.status === REFERENCE_ONLY) warnings.push(`${node.id}: ${node.type} is reference-only and cannot execute yet.`);
    nodeMap.set(node?.id, node);
    adjacency.set(node?.id, new Set());
  }

  for (const edge of Array.isArray(spec.edges) ? spec.edges : []) {
    const src = splitEndpoint(edge?.from);
    const dst = splitEndpoint(edge?.to);
    if (!src || !dst) { errors.push("Every edge must use owner.port endpoints."); continue; }
    const [srcOwner, srcPort] = src;
    const [dstOwner, dstPort] = dst;
    if (!nodeMap.has(dstOwner)) { errors.push(`Edge destination node ${dstOwner} does not exist.`); continue; }
    const destPart = catalog[nodeMap.get(dstOwner)?.type];
    if (destPart && !destPart.inputs?.[dstPort]) errors.push(`${dstOwner}.${dstPort} is not an input port.`);
    const key = `${dstOwner}.${dstPort}`;
    if (incoming.has(key)) errors.push(`${key} has more than one source.`);
    incoming.set(key, edge.from);

    if (srcOwner === "input") {
      if (!inputs[srcPort]) errors.push(`External input ${srcPort} is not declared.`);
    } else if (srcOwner === "state") {
      if (!states[srcPort]) errors.push(`State ${srcPort} is not declared.`);
    } else if (!nodeMap.has(srcOwner)) {
      errors.push(`Edge source node ${srcOwner} does not exist.`);
    } else {
      const srcPart = catalog[nodeMap.get(srcOwner)?.type];
      if (srcPart && !srcPart.outputs?.[srcPort]) errors.push(`${srcOwner}.${srcPort} is not an output port.`);
      adjacency.get(srcOwner)?.add(dstOwner);
    }
  }

  for (const node of nodes) {
    const part = catalog[node.type];
    for (const port of Object.keys(part?.inputs || {})) {
      if (!incoming.has(`${node.id}.${port}`)) errors.push(`Required input ${node.id}.${port} is not wired.`);
    }
  }

  const indegree = new Map(nodes.map(node => [node.id, 0]));
  for (const [source, destinations] of adjacency) for (const destination of destinations) indegree.set(destination, (indegree.get(destination) || 0) + 1);
  const queue = nodes.map(node => node.id).filter(id => indegree.get(id) === 0);
  const order = [];
  while (queue.length) {
    const current = queue.shift();
    order.push(current);
    for (const next of adjacency.get(current) || []) {
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) queue.push(next);
    }
  }
  if (order.length !== nodes.length) errors.push("Same-tick graph contains a cycle. Recurrent feedback must cross an explicit state boundary.");

  for (const nodeId of order) {
    const node = nodeMap.get(nodeId);
    const inputWidths = {};
    for (const [key, source] of incoming) {
      if (!key.startsWith(`${nodeId}.`)) continue;
      const port = key.slice(nodeId.length + 1);
      const actual = widths.get(source);
      const expected = expectedInputWidth(node.type, node.config, port);
      if (actual != null && expected != null && Number.isFinite(expected) && actual !== expected) errors.push(`Width mismatch at ${key}: source ${source} is ${actual}, expected ${expected}.`);
      inputWidths[port] = actual;
    }
    for (const port of Object.keys(catalog[node.type]?.outputs || {})) {
      let width = fixedOutputWidth(node.type, node.config, port);
      if (node.type === "concat2" && port === "out") {
        const a = inputWidths.a, b = inputWidths.b;
        width = a != null && b != null ? a + b : null;
      }
      if (node.type === "add" && port === "out") {
        const a = inputWidths.a, b = inputWidths.b;
        if (a != null && b != null && a !== b) errors.push(`Add node ${nodeId} requires equal widths; got ${a} and ${b}.`);
        width = a ?? b ?? null;
      }
      widths.set(`${nodeId}.${port}`, width);
    }
  }

  for (const [name, endpoint] of Object.entries(spec.state_updates || {})) {
    if (!states[name]) errors.push(`State update ${name} has no declared state.`);
    const parsed = splitEndpoint(endpoint);
    if (!parsed || !nodeMap.has(parsed[0]) || !catalog[nodeMap.get(parsed[0])?.type]?.outputs?.[parsed[1]]) errors.push(`State update ${name} points to invalid endpoint ${endpoint}.`);
    const actual = widths.get(endpoint);
    const expected = Number(states[name]?.size ?? 0);
    if (actual != null && expected && actual !== expected) errors.push(`State ${name} expects width ${expected}, but ${endpoint} is width ${actual}.`);
  }

  for (const [name, endpoint] of Object.entries(spec.outputs || {})) {
    const parsed = splitEndpoint(endpoint);
    if (!parsed || !nodeMap.has(parsed[0]) || !catalog[nodeMap.get(parsed[0])?.type]?.outputs?.[parsed[1]]) errors.push(`Output ${name} points to invalid endpoint ${endpoint}.`);
  }

  const executable = errors.length === 0 && nodes.every(node => catalog[node.type]?.status !== REFERENCE_ONLY);
  return {valid: errors.length === 0, executable, errors, warnings};
}

export function structuralFingerprint(spec) {
  const compact = {
    inputs: spec?.inputs || {},
    states: spec?.states || {},
    nodes: (spec?.nodes || []).map(node => ({id:node.id,type:node.type,config:node.config || {}})),
    edges: spec?.edges || [],
    outputs: spec?.outputs || {},
    state_updates: spec?.state_updates || {},
  };
  return JSON.stringify(compact);
}

export function compareArchitectures(base, candidate, catalog = {}) {
  const baseNodes = new Map((base?.nodes || []).map(node => [node.id, node]));
  const nextNodes = new Map((candidate?.nodes || []).map(node => [node.id, node]));
  const added = [...nextNodes.keys()].filter(id => !baseNodes.has(id));
  const removed = [...baseNodes.keys()].filter(id => !nextNodes.has(id));
  const changed = [...nextNodes.keys()].filter(id => baseNodes.has(id) && JSON.stringify(baseNodes.get(id)) !== JSON.stringify(nextNodes.get(id)));
  const edgeKey = edge => `${edge.from}->${edge.to}`;
  const baseEdges = new Set((base?.edges || []).map(edgeKey));
  const nextEdges = new Set((candidate?.edges || []).map(edgeKey));
  return {
    added,
    removed,
    changed,
    edgesAdded: [...nextEdges].filter(key => !baseEdges.has(key)),
    edgesRemoved: [...baseEdges].filter(key => !nextEdges.has(key)),
    parameterDelta: architectureAnatomy(candidate, catalog).trainableParameters - architectureAnatomy(base, catalog).trainableParameters,
  };
}
