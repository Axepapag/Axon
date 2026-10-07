import {AxonApi} from "./api.js?v=20261006-s4";
import {
  analyzeNativeText,
  defaultsFromSchema,
  displayCharacter,
  getAtPath,
  portCompatibility,
  registrationPayload,
  setAtPath,
  stableFingerprint,
} from "./ui-utils.js?v=20261006-s4";
import {PreflightController, isPreflightBusy} from "./preflight-controller.js?v=20261006-s5";
import {RunMutationController, defaultRunMutationState, isRunMutationBusy} from "./run-controller.js?v=20261006-s5";

const api = new AxonApi();
const STORAGE_KEY = "axon-lab-ui-v2";

function loadPersisted() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

const persisted = loadPersisted();
function freshArchitecture() {
  return {
    architecture_id: null,
    architecture_hash: null,
    version: "draft-1",
    name: "Untitled experiment",
    nodes: [],
    edges: [],
    ports: [],
  };
}

const state = {
  connection: "loading",
  connectionError: null,
  capabilities: null,
  readiness: null,
  view: "system",
  architecture: {...freshArchitecture(), ...(persisted.architecture || {})},
  architectureValidation: null,
  architectureRegistration: {
    commandId: null,
    fingerprint: null,
    status: "idle",
    result: null,
    error: null,
    ...(persisted.architectureRegistration || {}),
  },
  architectUi: {
    selectedNodeId: null,
    edgeDraft: {sourceNode: "", sourcePort: "", destinationNode: "", destinationPort: ""},
    ...(persisted.architectUi || {}),
  },
  preflight: {
    device: "auto",
    commandId: null,
    operationId: null,
    status: "idle",
    operation: null,
    error: null,
    cachedResult: null,
    generation: 0,
    evidenceRefreshError: null,
    ...(persisted.preflight || {}),
  },
  datasets: [],
  curricula: [],
  registeredArchitectures: [],
  runs: [],
  selectedRun: null,
  runEvents: [],
  runEventCursor: 0,
  eventMode: "idle",
  eventSource: null,
  eventPollTimer: null,
  runMutation: {...defaultRunMutationState(), ...(persisted.runMutation || {})},
  tensors: [],
  runSnapshot: null,
  selectedTensor: null,
  tensorError: null,
  checkpoints: [],
  backup: null,
  inferenceSession: null,
};

let preflightController = null;
let runMutationController = null;

const dom = {
  main: document.querySelector("#main-view"),
  pill: document.querySelector("#connection-pill"),
  retry: document.querySelector("#retry-connection"),
  nav: [...document.querySelectorAll(".nav-item")],
};

function persistUiState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      architecture: state.architecture,
      architectureRegistration: state.architectureRegistration,
      architectUi: state.architectUi,
      preflight: {
        device: state.preflight.device,
        commandId: state.preflight.commandId,
        operationId: state.preflight.operationId,
        status: state.preflight.status,
        operation: state.preflight.operation,
        error: state.preflight.error,
        cachedResult: state.preflight.cachedResult,
        generation: state.preflight.generation,
        evidenceRefreshError: state.preflight.evidenceRefreshError,
      },
      runMutation: state.runMutation,
    }));
  } catch {
    // Storage failure must never break the operator surface.
  }
}

function safeArray(value) {
  if (Array.isArray(value)) return value;
  if (Array.isArray(value?.items)) return value.items;
  if (Array.isArray(value?.results)) return value.results;
  return [];
}

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function id() {
  return crypto.randomUUID ? crypto.randomUUID() : `cmd-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function statusClass(status) {
  if (["available", "ready", "connected", "passed", "pass", "ok", "running", "completed"].includes(status)) return "good";
  if (["experimental", "warning", "queued", "submitting", "uncertain", "paused", "starting", "pausing", "resuming", "poll_error"].includes(status)) return "warn";
  if (["unavailable", "not_integrated", "failed", "error", "stopped", "interrupted"].includes(status)) return "bad";
  return "neutral";
}

function setConnection(mode, error = null) {
  state.connection = mode;
  state.connectionError = error;
  dom.pill.className = "status-pill";
  if (mode === "connected") {
    dom.pill.classList.add("status-connected");
    dom.pill.textContent = "Backend connected";
  } else if (mode === "loading") {
    dom.pill.classList.add("status-loading");
    dom.pill.textContent = "Connecting";
  } else {
    dom.pill.classList.add("status-disconnected");
    dom.pill.textContent = "Disconnected";
  }
}

async function connect() {
  setConnection("loading");
  render();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const [capabilities, readiness] = await Promise.all([
      api.capabilities(controller.signal),
      api.readiness(controller.signal).catch(error => ({checks: [], unavailable: true, error: error.message})),
    ]);
    state.capabilities = capabilities;
    state.readiness = readiness;
    setConnection("connected");
    await refreshCatalogs();
    void resumePreflightPolling();
    ensureRunMutationController().resume();
  } catch (error) {
    setConnection("error", error);
  } finally {
    clearTimeout(timer);
    render();
  }
}

async function refreshEvidence() {
  if (state.connection !== "connected") return;
  const [capabilities, readiness] = await Promise.all([
    api.capabilities(),
    api.readiness().catch(error => ({checks: [], unavailable: true, error: error.message})),
  ]);
  state.capabilities = capabilities;
  state.readiness = readiness;
}

async function refreshCatalogs() {
  if (state.connection !== "connected") return;
  const requests = [
    api.datasets().then(v => state.datasets = safeArray(v)).catch(() => state.datasets = []),
    api.curricula().then(v => state.curricula = safeArray(v)).catch(() => state.curricula = []),
    api.architectures().then(v => state.registeredArchitectures = safeArray(v)).catch(() => state.registeredArchitectures = []),
    api.runs().then(v => state.runs = safeArray(v)).catch(() => state.runs = []),
    api.checkpoints().then(v => state.checkpoints = safeArray(v)).catch(() => state.checkpoints = []),
    api.backupStatus().then(v => state.backup = v).catch(error => state.backup = {status: "unavailable", reason: error.message}),
  ];
  await Promise.all(requests);
}

function setView(view) {
  state.view = view;
  dom.nav.forEach(button => button.classList.toggle("active", button.dataset.view === view));
  render();
  dom.main.focus({preventScroll: true});
}

function render() {
  const renderer = {
    system: renderSystem,
    architect: renderArchitect,
    data: renderData,
    train: renderTrain,
    runs: renderRuns,
    inspect: renderInspect,
    inference: renderInference,
    checkpoints: renderCheckpoints,
    backup: renderBackup,
  }[state.view] || renderSystem;

  dom.main.innerHTML = renderer();
  bindViewEvents();
}

function disconnectedBlock() {
  if (state.connection === "loading") {
    return `<section class="panel state-panel"><div class="spinner"></div><div><h2>Connecting to trainer backend</h2><p>Axon Lab is waiting for <code>${esc(api.baseUrl)}</code>. No controls are assumed available until capabilities arrive.</p></div></section>`;
  }
  if (state.connection !== "connected") {
    return `<section class="panel state-panel state-error">
      <div class="state-icon">!</div>
      <div><h2>Trainer backend unavailable</h2>
      <p>${esc(state.connectionError?.message || "No capability response received.")}</p>
      <p class="muted">Local graph edits are preserved. Axon Lab will not fabricate components, providers, metrics, or run state while disconnected.</p>
      <button class="button" data-action="retry">Retry connection</button></div>
    </section>`;
  }
  return "";
}

function renderSystem() {
  if (state.connection !== "connected") return page("System", "Connection and readiness", disconnectedBlock());

  const caps = state.capabilities || {};
  const components = safeArray(caps.components);
  const devices = safeArray(caps.devices);
  const providers = safeArray(caps.providers);
  const checks = safeArray(state.readiness?.checks);

  return page("System", "Backend capabilities, explicit preflight and start-readiness evidence", `
    <section class="metric-grid">
      ${metricCard("Components", components.length, "Advertised by backend")}
      ${metricCard("Devices", devices.filter(x => x.status === "available").length, `${devices.length} reported`)}
      ${metricCard("Providers", providers.filter(x => x.status === "available").length, `${providers.length} reported`)}
      ${metricCard("Schema", caps.schema_version || "unknown", caps.backend_version || "backend version unavailable")}
    </section>

    ${renderPreflightPanel(checks)}

    <section class="panel">
      <div class="panel-header"><div><h2>Training readiness</h2><p>Foundation checks are evidence only; the backend must explicitly authorize training.</p></div>
        <span class="tag ${state.readiness?.training_authorized ? "good" : "bad"}">${state.readiness?.training_authorized ? "Authorized" : "Not authorized"}</span>
      </div>
      ${checks.length ? `<div class="check-list">${checks.map(check => `
        <div class="check-row">
          <span class="dot ${statusClass(check.status)}"></span>
          <div><strong>${esc(check.name || check.id || "check")}</strong><div class="muted">${esc(check.reason || "")}</div></div>
          <span class="tag ${statusClass(check.status)}">${esc(check.status || "unknown")}</span>
        </div>`).join("")}</div>` : `<div class="notice">No higher-level readiness checks were returned.</div>`}
    </section>

    <section class="two-column">
      ${capabilityPanel("Components", components)}
      ${capabilityPanel("Devices", devices)}
    </section>
    <section class="panel">
      <div class="panel-header"><div><h2>Providers</h2><p>Only backend-advertised integrations are shown.</p></div></div>
      ${entityRows(providers, "No providers reported.")}
    </section>
  `);
}

function renderPreflightPanel(checks) {
  const supported = safeArray(state.capabilities?.supported_actions).includes("preflight");
  const busy = isPreflightBusy(state.preflight.status);
  const current = state.preflight.operation;
  const historical = checks.find(check => check.id === "foundation");
  const result = current?.result;
  const resultChecks = safeArray(result?.checks);
  const cached = state.preflight.cachedResult;

  return `<section class="panel preflight-panel">
    <div class="panel-header split">
      <div><h2>Device preflight</h2><p>Runs a real selected-device foundation check. Each new check gets a command ID that is retained for safe retry.</p></div>
      <span class="tag ${statusClass(state.preflight.status)}">${esc(state.preflight.status || "idle")}</span>
    </div>

    <div class="preflight-buttons">
      ${[
        ["auto", "Auto"],
        ["cpu", "CPU"],
        ["cuda:0", "CUDA 0"],
      ].map(([device, label]) => `<button class="button ${state.preflight.device === device ? "" : "button-secondary"}" data-action="start-preflight" data-device="${device}" ${!supported || busy ? "disabled" : ""}>${label}</button>`).join("")}
      <button class="button button-secondary" data-action="retry-preflight" ${!state.preflight.commandId || busy ? "disabled" : ""}>Recover/retry same command</button>
      <button class="button button-secondary" data-action="poll-preflight" ${!state.preflight.operationId || busy ? "disabled" : ""}>Refresh operation</button>
    </div>

    <div class="evidence-grid">
      <article class="evidence-card">
        <div class="evidence-label">Current operation</div>
        ${current || state.preflight.commandId ? `
          <dl class="mini-details">
            ${detail("Device requested", state.preflight.device)}
            ${detail("Command ID", state.preflight.commandId)}
            ${detail("Operation ID", state.preflight.operationId)}
            ${detail("Status", current?.status || state.preflight.status)}
            ${detail("Started", current?.started_at)}
            ${detail("Finished", current?.finished_at)}
          </dl>
          ${state.preflight.error ? `<div class="validation-row bad">${esc(state.preflight.error)}</div>` : ""}
          ${state.preflight.evidenceRefreshError ? `<div class="validation-row warn">${esc(state.preflight.evidenceRefreshError)}</div>` : ""}
          ${resultChecks.length ? `<div class="check-list compact">${resultChecks.map(check => `
            <div class="check-row"><span class="dot ${statusClass(check.status)}"></span><div><strong>${esc(check.name || check.id)}</strong><div class="muted">${esc(check.message || check.reason || "")}</div></div><span class="tag ${statusClass(check.status)}">${esc(check.status)}</span></div>`).join("")}</div>` : ""}
          ${result ? `<details><summary>Raw completed result</summary><pre class="json-view">${esc(JSON.stringify(result, null, 2))}</pre></details>` : ""}
        ` : `<div class="notice">No check is currently selected.</div>`}
      </article>

      <article class="evidence-card historical-evidence">
        <div class="evidence-label">Latest completed server evidence · historical</div>
        ${historical ? `
          <dl class="mini-details">
            ${detail("Status", historical.status)}
            ${detail("Evidence operation", historical.evidence)}
            ${detail("Updated", historical.updated_at)}
            ${detail("Reason", historical.reason)}
          </dl>
          <p class="muted">This is cached completed evidence reported by the server. It is not a claim that a new check is running now.</p>
        ` : `<div class="notice">The server has no completed foundation evidence yet.</div>`}
        ${cached && cached.operation_id !== historical?.evidence ? `<details><summary>Browser-cached earlier result</summary><pre class="json-view">${esc(JSON.stringify(cached, null, 2))}</pre></details>` : ""}
      </article>
    </div>
  </section>`;
}

function renderArchitect() {
  const components = safeArray(state.capabilities?.components);
  const disconnected = state.connection !== "connected";
  const nodes = state.architecture.nodes;

  return page("Architect", "Build experiment graphs only from backend-supplied component contracts", `
    ${disconnected ? disconnectedBlock() : ""}
    <section class="architect-layout">
      <aside class="panel palette">
        <div class="panel-header"><div><h2>Components</h2><p>Blocks remain disabled until a real configuration contract and execution adapter are advertised.</p></div></div>
        <div class="palette-list">
          ${components.length ? components.map(component => {
            const hasContract = Boolean(component.configuration_schema) && Array.isArray(component.ports);
            const enabled = hasContract && component.execution_eligible === true;
            const reason = enabled ? (component.version || "") : (component.reason || (!hasContract ? "Component contract not supplied." : "Execution adapter unavailable."));
            return `<button class="palette-item" data-action="add-node" data-component-id="${esc(component.id)}" ${enabled ? "" : "disabled"}>
              <span><strong>${esc(component.name || component.id)}</strong><small>${esc(reason)}</small></span>
              <span class="tag ${statusClass(component.status)}">${esc(component.status || "unknown")}</span>
            </button>`;
          }).join("") : `<div class="notice">Connect to a backend with component capabilities to populate the palette.</div>`}
        </div>
      </aside>

      <section class="panel graph-panel">
        <div class="panel-header split">
          <div>
            <h2>Experiment graph</h2>
            <p>Draft and selection persist locally across refreshes and backend errors. No draft is executable until backend validation says so.</p>
          </div>
          <div class="button-row wrap">
            <button class="button button-secondary" data-action="clear-graph" ${nodes.length ? "" : "disabled"}>Clear</button>
            <button class="button button-secondary" data-action="validate-architecture" ${disconnected || !nodes.length ? "disabled" : ""}>Validate</button>
            <button class="button" data-action="register-architecture" ${disconnected || !nodes.length || !state.architectureValidation?.valid ? "disabled" : ""}>Register</button>
          </div>
        </div>
        <label class="field">
          <span>Architecture name</span>
          <input id="architecture-name" value="${esc(state.architecture.name)}" autocomplete="off">
        </label>
        <div class="node-canvas">
          ${nodes.length ? nodes.map((node, index) => renderNode(node, index)).join("") : `
            <div class="canvas-empty"><strong>No executable component contracts yet</strong><span>The palette will activate only when Core publishes real schemas, ports and adapters.</span></div>`}
        </div>

        <div class="architect-detail-grid">
          <section class="architect-detail">
            <div class="subheader"><h3>Selected component properties</h3><span class="muted">Schema-driven</span></div>
            ${renderSelectedNodeProperties()}
          </section>
          <section class="architect-detail">
            <div class="subheader"><h3>Explicit port connections</h3><span class="muted">No implicit adapters</span></div>
            ${renderEdgeEditor()}
          </section>
        </div>

        ${renderValidation()}
        ${renderRegistrationStatus()}
      </section>
    </section>
  `);
}

function renderNode(node, index) {
  const selected = state.architectUi.selectedNodeId === node.node_id;
  const component = componentForNode(node);
  const outputs = portsForNode(node, "output").length;
  const inputs = portsForNode(node, "input").length;
  return `<article class="node-card ${selected ? "selected-node" : ""}">
    <button class="node-select" data-action="select-node" data-node-id="${esc(node.node_id)}">
      <div class="node-index">${index + 1}</div>
      <div class="node-body">
        <strong>${esc(node.display_name || component?.name || node.component_type)}</strong>
        <span class="muted">${esc(node.component_type)} · ${esc(node.component_version || "version unknown")}</span>
        <code>${esc(node.node_id)}</code>
        <small>${inputs} input · ${outputs} output port(s)</small>
      </div>
    </button>
    <button class="icon-button" data-action="remove-node" data-node-id="${esc(node.node_id)}" title="Remove component">×</button>
  </article>`;
}

function componentForNode(node) {
  return safeArray(state.capabilities?.components).find(component => component.id === node?.component_type) || null;
}

function portsForNode(node, direction) {
  const component = componentForNode(node);
  const ports = Array.isArray(component?.ports) ? component.ports : (Array.isArray(node?.ports) ? node.ports : []);
  return ports.filter(port => port.direction === direction);
}

function renderSelectedNodeProperties() {
  const node = state.architecture.nodes.find(item => item.node_id === state.architectUi.selectedNodeId);
  if (!node) return `<div class="notice">Select a component to inspect its real configuration contract.</div>`;
  const component = componentForNode(node);
  const schema = component?.configuration_schema;
  if (!schema) return `<div class="notice">No configuration schema has been supplied by the Core owner for this component.</div>`;

  const ports = Array.isArray(component.ports) ? component.ports : [];
  return `
    <div class="selected-component-heading"><strong>${esc(component.name || component.id)}</strong><span class="tag ${statusClass(component.status)}">${esc(component.status)}</span></div>
    <div class="schema-form">${renderSchemaFields(schema, node.config || {}, "")}</div>
    <div class="subheader"><h3>Contract ports</h3><span class="muted">${ports.length} declared</span></div>
    ${ports.length ? `<div class="port-list">${ports.map(port => `
      <div class="port-row"><span class="tag neutral">${esc(port.direction)}</span><strong>${esc(port.port_id)}</strong><small>${esc(port.meaning || "")}</small><code>${esc(port.dtype || "?")} ${esc(JSON.stringify(port.shape ?? port.width ?? "?"))}</code><span class="tag ${port.surface === "substrate-exact" ? "good" : "neutral"}">${esc(port.surface || "surface unset")}</span></div>`).join("")}</div>` : `<div class="notice">No ports supplied.</div>`}
  `;
}

function renderSchemaFields(schema, config, prefix) {
  const properties = schema?.properties || {};
  const required = new Set(schema?.required || []);
  if (!Object.keys(properties).length) return `<div class="notice">This component has no editable configuration fields.</div>`;
  return Object.entries(properties).map(([name, field]) => {
    const path = prefix ? `${prefix}.${name}` : name;
    const value = getAtPath(config, path);
    const label = `${name}${required.has(name) ? " *" : ""}`;
    if (field.type === "object" || field.properties) {
      return `<fieldset class="schema-group"><legend>${esc(label)}</legend>${renderSchemaFields(field, config, path)}</fieldset>`;
    }
    if (Array.isArray(field.enum)) {
      return `<label class="field"><span>${esc(label)}</span><select data-config-path="${esc(path)}" data-config-type="enum">${field.enum.map(option => `<option value="${esc(option)}" ${String(value) === String(option) ? "selected" : ""}>${esc(option)}</option>`).join("")}</select>${field.description ? `<small>${esc(field.description)}</small>` : ""}</label>`;
    }
    if (field.type === "boolean") {
      return `<label class="field"><span>${esc(label)}</span><select data-config-path="${esc(path)}" data-config-type="boolean"><option value="true" ${value === true ? "selected" : ""}>true</option><option value="false" ${value === false ? "selected" : ""}>false</option></select>${field.description ? `<small>${esc(field.description)}</small>` : ""}</label>`;
    }
    if (field.type === "integer" || field.type === "number") {
      return `<label class="field"><span>${esc(label)}</span><input data-config-path="${esc(path)}" data-config-type="${field.type}" type="number" value="${esc(value ?? "")}" ${field.minimum !== undefined ? `min="${esc(field.minimum)}"` : ""} ${field.maximum !== undefined ? `max="${esc(field.maximum)}"` : ""} step="${field.type === "integer" ? "1" : "any"}">${field.description ? `<small>${esc(field.description)}</small>` : ""}</label>`;
    }
    if (field.type === "array") {
      return `<label class="field"><span>${esc(label)} · JSON array</span><textarea data-config-path="${esc(path)}" data-config-type="json" rows="3">${esc(JSON.stringify(value ?? []))}</textarea>${field.description ? `<small>${esc(field.description)}</small>` : ""}</label>`;
    }
    return `<label class="field"><span>${esc(label)}</span><input data-config-path="${esc(path)}" data-config-type="string" type="text" value="${esc(value ?? "")}">${field.description ? `<small>${esc(field.description)}</small>` : ""}</label>`;
  }).join("");
}

function renderEdgeEditor() {
  const nodes = state.architecture.nodes;
  if (!nodes.length) return `<div class="notice">Edges become available after real component contracts are added.</div>`;
  const draft = state.architectUi.edgeDraft;
  const sourceNode = nodes.find(node => node.node_id === draft.sourceNode);
  const destinationNode = nodes.find(node => node.node_id === draft.destinationNode);
  const allSourcePorts = sourceNode ? portsForNode(sourceNode, "output") : [];
  const allDestinationPorts = destinationNode ? portsForNode(destinationNode, "input") : [];
  const selectedSource = allSourcePorts.find(port => port.port_id === draft.sourcePort) || null;
  const selectedDestination = allDestinationPorts.find(port => port.port_id === draft.destinationPort) || null;

  const destinationPorts = selectedSource
    ? allDestinationPorts.filter(port => portCompatibility(selectedSource, port).compatible)
    : allDestinationPorts;
  const sourcePorts = selectedDestination
    ? allSourcePorts.filter(port => portCompatibility(port, selectedDestination).compatible)
    : allSourcePorts;

  const compatibility = selectedSource && selectedDestination
    ? portCompatibility(selectedSource, selectedDestination)
    : null;
  const canAdd = Boolean(
    draft.sourceNode &&
    draft.sourcePort &&
    draft.destinationNode &&
    draft.destinationPort &&
    compatibility?.compatible
  );

  return `
    <div class="edge-form">
      <label class="field"><span>Source component</span><select data-edge-field="sourceNode"><option value="">Select…</option>${nodes.filter(node => portsForNode(node, "output").length).map(node => `<option value="${esc(node.node_id)}" ${draft.sourceNode === node.node_id ? "selected" : ""}>${esc(node.display_name || node.component_type)}</option>`).join("")}</select></label>
      <label class="field"><span>Source port</span><select data-edge-field="sourcePort"><option value="">Select…</option>${sourcePorts.map(port => `<option value="${esc(port.port_id)}" ${draft.sourcePort === port.port_id ? "selected" : ""}>${esc(port.port_id)} · ${esc(port.surface || "surface unset")} · ${esc(port.meaning || "")}</option>`).join("")}</select></label>
      <label class="field"><span>Destination component</span><select data-edge-field="destinationNode"><option value="">Select…</option>${nodes.filter(node => portsForNode(node, "input").length).map(node => `<option value="${esc(node.node_id)}" ${draft.destinationNode === node.node_id ? "selected" : ""}>${esc(node.display_name || node.component_type)}</option>`).join("")}</select></label>
      <label class="field"><span>Destination port</span><select data-edge-field="destinationPort"><option value="">Select…</option>${destinationPorts.map(port => `<option value="${esc(port.port_id)}" ${draft.destinationPort === port.port_id ? "selected" : ""}>${esc(port.port_id)} · ${esc(port.surface || "surface unset")} · ${esc(port.meaning || "")}</option>`).join("")}</select></label>
      <button class="button button-secondary" data-action="add-edge" ${canAdd ? "" : "disabled"}>Add explicit edge</button>
    </div>
    ${compatibility && !compatibility.compatible ? `<div class="validation validation-bad"><strong>Ports are incompatible</strong>${compatibility.reasons.map(reason => `<div class="validation-row bad">${esc(reason)}</div>`).join("")}</div>` : ""}
    <div class="edge-list">
      ${state.architecture.edges.length ? state.architecture.edges.map((edge, index) => `
        <div class="edge-row"><code>${esc(edge.source?.node_id)}:${esc(edge.source?.port_id)}</code><span>→</span><code>${esc(edge.destination?.node_id)}:${esc(edge.destination?.port_id)}</code><button class="icon-button" data-action="remove-edge" data-edge-index="${index}">×</button></div>`).join("") : `<div class="notice">No edges. Component card order never implies a connection.</div>`}
    </div>
  `;
}

function renderValidation() {
  const v = state.architectureValidation;
  if (!v) return `<div class="notice">Validation has not run for this draft.</div>`;
  const errors = safeArray(v.errors);
  const warnings = safeArray(v.warnings);
  return `<div class="validation ${v.valid ? "validation-good" : "validation-bad"}">
    <strong>${v.valid ? (v.execution_eligible ? "Graph valid and execution-eligible" : "Graph structurally valid, not executable") : "Graph not valid"}</strong>
    ${errors.map(item => `<div class="validation-row bad">${esc(item.path ? item.path + ": " : "")}${esc(item.message || item)}</div>`).join("")}
    ${warnings.map(item => `<div class="validation-row warn">${esc(item.path ? item.path + ": " : "")}${esc(item.message || item)}</div>`).join("")}
    ${v.resource_estimates ? `<pre>${esc(JSON.stringify(v.resource_estimates, null, 2))}</pre>` : ""}
  </div>`;
}

function renderRegistrationStatus() {
  const r = state.architectureRegistration;
  if (!r.commandId) return "";
  return `<div class="registration-status">
    <div class="split"><strong>Registration retry identity</strong><span class="tag ${statusClass(r.status)}">${esc(r.status)}</span></div>
    <code>${esc(r.commandId)}</code>
    ${r.error ? `<div class="validation-row bad">${esc(r.error)}</div>` : ""}
    <p class="muted">The same command ID is reused only while the architecture payload is unchanged. Editing the graph creates a new retry identity.</p>
  </div>`;
}

function renderData() {
  return page("Data", "Provenance-bearing datasets and curricula", `
    ${state.connection !== "connected" ? disconnectedBlock() : ""}
    <section class="two-column">
      <div class="panel">
        <div class="panel-header"><div><h2>Datasets</h2><p>No source is inferred from local files by the frontend.</p></div></div>
        ${manifestCards(state.datasets, "dataset_id")}
      </div>
      <div class="panel">
        <div class="panel-header"><div><h2>Curricula</h2><p>Versions and stages as advertised by the trainer backend.</p></div></div>
        ${manifestCards(state.curricula, "curriculum_id")}
      </div>
    </section>
  `);
}

function readinessReasons(readiness = state.readiness) {
  if (Array.isArray(readiness?.reasons)) return readiness.reasons;
  return safeArray(readiness?.checks)
    .filter(check => !["passed", "pass", "ready", "available"].includes(check.status))
    .map(check => check.reason || `${check.name || check.id}: ${check.status || "unavailable"}`);
}

function architectureCurrent(architecture) {
  const catalog = new Map(safeArray(state.capabilities?.components).map(component => [component.id, component]));
  const nodes = safeArray(architecture?.nodes);
  if (!nodes.length) return {current: false, reason: "No nodes in registered architecture."};
  for (const node of nodes) {
    const component = catalog.get(node.component_type);
    if (!component) return {current: false, reason: `${node.component_type} is not in the live catalog.`};
    if (String(node.component_version) !== String(component.version)) {
      return {current: false, reason: `${node.component_type} is ${node.component_version}; live catalog requires ${component.version}. Rebuild/migrate explicitly.`};
    }
  }
  return {current: true, reason: "Matches live component versions."};
}

function architectureSelectField() {
  const items = state.registeredArchitectures;
  return `<label class="field"><span>Registered architecture</span><select name="architecture"><option value="">Select…</option>${items.map(item => {
    const status = architectureCurrent(item);
    return `<option value="${esc(item.architecture_id)}" ${status.current ? "" : "disabled"}>${esc(item.name || item.architecture_id)} · v${esc(item.version || "?")}${status.current ? "" : " · REBUILD REQUIRED"}</option>`;
  }).join("")}</select><small>Stored graphs are never silently upgraded when component versions change.</small></label>`;
}

function renderMutationStatus(slot, title) {
  if (!slot?.commandId) return "";
  return `<section class="operation-card">
    <div class="split"><strong>${esc(title)}</strong><span class="tag ${statusClass(slot.status)}">${esc(slot.status || "idle")}</span></div>
    <dl class="mini-details">
      ${detail("Command ID", slot.commandId)}
      ${detail("Operation ID", slot.operationId)}
      ${detail("Run", slot.runId)}
      ${detail("Operation", slot.operation?.kind)}
      ${detail("Finished", slot.operation?.finished_at)}
    </dl>
    ${slot.error ? `<div class="validation-row bad">${esc(slot.error)}</div>` : ""}
    ${slot.refreshError ? `<div class="validation-row warn">${esc(slot.refreshError)}</div>` : ""}
    ${slot.status === "uncertain" ? `<button class="button button-secondary" data-action="recover-run-mutation" data-kind="${slot === state.runMutation.creation ? "creation" : "command"}">Recover with same command ID</button>` : ""}
    ${slot.operation?.error ? `<pre class="json-view">${esc(JSON.stringify(slot.operation.error, null, 2))}</pre>` : ""}
  </section>`;
}

function renderTrain() {
  const devices = safeArray(state.capabilities?.devices).filter(x => x.status === "available");
  const providers = safeArray(state.capabilities?.providers).filter(x => x.status === "available");
  const trainingAuthorized = Boolean(state.readiness?.training_authorized);
  const preparationAvailable = Boolean(state.capabilities?.feature_flags?.run_preparation);
  const creationBusy = isRunMutationBusy(state.runMutation.creation.status);
  const reasons = readinessReasons();
  return page("Train", "Prepare a pinned E0 run now; execution stays blocked until acceptance gates clear", `
    ${state.connection !== "connected" ? disconnectedBlock() : ""}
    <section class="panel">
      <div class="panel-header"><div><h2>Prepare run</h2><p>Preparation validates and pins architecture, frozen data, device, seed and training settings. It does not start training.</p></div>
        <span class="tag ${trainingAuthorized ? "good" : preparationAvailable ? "warn" : "bad"}">${trainingAuthorized ? "Execution authorized" : preparationAvailable ? "Preparation available · Start blocked" : "Run preparation unavailable"}</span>
      </div>
      ${!trainingAuthorized && reasons.length ? `<div class="readiness-reasons"><strong>Execution remains blocked because:</strong>${reasons.map(reason => `<div class="validation-row warn">${esc(reason)}</div>`).join("")}</div>` : ""}
      <form id="create-run-form" class="form-grid">
        ${selectField("Dataset", "dataset", state.datasets, "dataset_id")}
        ${selectField("Curriculum", "curriculum", state.curricula, "curriculum_id")}
        ${selectField("Device", "device", devices, "id")}
        ${selectField("Provider", "provider", providers, "id")}
        <label class="field"><span>Seed</span><input name="seed" type="number" value="1" min="0" step="1"></label>
        <label class="field"><span>Epochs</span><input name="epochs" type="number" value="1" min="1" max="100" step="1"></label>
        <label class="field"><span>Learning rate</span><input name="learning_rate" type="number" value="0.001" min="0.0000001" max="0.1" step="any"></label>
        ${architectureSelectField()}
        <div class="form-actions">
          <button class="button" type="submit" ${state.connection === "connected" && preparationAvailable && !creationBusy ? "" : "disabled"}>Prepare run</button>
          <span class="muted">Preparation only. Start remains server-gated and disabled until readiness authorizes execution.</span>
        </div>
      </form>
      ${renderMutationStatus(state.runMutation.creation, "Run preparation operation")}
    </section>
  `);
}

function renderRuns() {
  return page("Runs", "Prepared work, blocked execution, lifecycle commands and durable ordered events", `
    ${state.connection !== "connected" ? disconnectedBlock() : ""}
    <section class="runs-layout">
      <aside class="panel">
        <div class="panel-header split"><div><h2>Runs</h2><p>${state.runs.length} reported</p></div><button class="button button-secondary" data-action="refresh-runs">Refresh</button></div>
        <div class="list">
          ${state.runs.length ? state.runs.map(run => `
            <button class="list-item ${state.selectedRun?.run_id === run.run_id ? "selected" : ""}" data-action="select-run" data-run-id="${esc(run.run_id)}">
              <span><strong>${esc(run.run_id)}</strong><small>${esc(run.architecture_id || "")}</small></span>
              <span class="tag ${statusClass(run.lifecycle_state)}">${esc(run.lifecycle_state || "unknown")}</span>
            </button>`).join("") : `<div class="notice">No prepared runs reported.</div>`}
        </div>
      </aside>
      <section class="panel">
        ${state.selectedRun ? renderSelectedRun(state.selectedRun) : `<div class="canvas-empty"><strong>Select a run</strong><span>Lifecycle, readiness, real episode results and durable events will appear here.</span></div>`}
      </section>
    </section>
  `);
}

function renderLatestResult(result) {
  if (!result) return `<div class="notice">No completed episode result is available. Loss is unavailable from the current loop.</div>`;
  const metrics = result.metrics || {};
  return `<div class="result-card">
    <div class="split"><strong>Latest real episode result</strong><span class="tag neutral">${esc(result.family || "episode")}</span></div>
    <dl class="detail-grid">
      ${detail("Episode", result.episode_id)}
      ${detail("Split", result.split)}
      ${detail("Exact", metrics.exact)}
      ${detail("Per-char accuracy", metrics.per_char_accuracy)}
      ${detail("Control correct", metrics.control_correct)}
      ${detail("Binding error", metrics.binding_error)}
      ${detail("Obsolete error", metrics.obsolete_error)}
      ${detail("Invalid content", metrics.invalid_content)}
      ${detail("Loss", "Unavailable — loop does not expose loss yet")}
    </dl>
    <details><summary>Result row</summary><pre class="json-view">${esc(JSON.stringify(result, null, 2))}</pre></details>
  </div>`;
}

function commandLabel(command) {
  if (command === "pause") return "Pause after current episode";
  if (command === "stop") return "Stop after current episode";
  if (command === "checkpoint") return "Checkpoint at episode boundary";
  return command[0].toUpperCase() + command.slice(1);
}

function renderSelectedRun(run) {
  const allowed = safeArray(run.allowed_actions);
  const commands = ["start", "pause", "resume", "stop", "checkpoint"];
  const executionFeature = Boolean(state.capabilities?.feature_flags?.training);
  const globalAuthorized = Boolean(state.readiness?.training_authorized);
  const runReasons = readinessReasons(run.readiness);
  const commandBusy = isRunMutationBusy(state.runMutation.command.status);
  const cursor = run.execution_cursor?.position || {};
  return `
    <div class="panel-header split"><div><h2>${esc(run.run_id)}</h2><p>${esc(run.architecture_id || "architecture unavailable")}</p></div><span class="tag ${statusClass(run.lifecycle_state)}">${esc(run.lifecycle_state || "unknown")}</span></div>
    ${!run.readiness?.authorized && runReasons.length ? `<div class="readiness-reasons"><strong>Execution blocked:</strong>${runReasons.map(reason => `<div class="validation-row warn">${esc(reason)}</div>`).join("")}</div>` : ""}
    <div class="button-row wrap">
      ${commands.map(command => {
        const executionCommand = command === "start" || command === "resume";
        const enabled = allowed.includes(command) && !commandBusy && (!executionCommand || (executionFeature && globalAuthorized && run.readiness?.authorized));
        return `<button class="button ${command === "stop" ? "button-danger" : "button-secondary"}" data-action="run-command" data-command="${command}" ${enabled ? "" : "disabled"}>${esc(commandLabel(command))}</button>`;
      }).join("")}
    </div>
    <p class="muted">Pause and stop are episode-boundary operations: they take effect <strong>after the current episode</strong>, never mid-episode.</p>
    <dl class="detail-grid">
      ${detail("Optimizer steps", run.step)}
      ${detail("Epoch", run.epoch)}
      ${detail("Next episode", run.next_episode)}
      ${detail("Device", run.device?.name || run.device || null)}
      ${detail("Provider", run.provider?.name || run.provider || null)}
      ${detail("Snapshot", run.snapshot_id)}
      ${detail("Latest checkpoint", run.latest_checkpoint_id)}
      ${detail("Failure", run.failure_reason)}
      ${detail("Cursor phase", cursor.phase)}
      ${detail("Cursor episode index", cursor.episode_index)}
      ${detail("Cursor optimizer step", cursor.optimizer_step)}
    </dl>
    ${renderLatestResult(run.latest_result)}
    ${renderMutationStatus(state.runMutation.command, "Lifecycle operation")}
    <div class="subheader"><h3>Durable ordered events</h3><span class="muted">${state.eventMode} · sequence ${state.runEventCursor} · ${state.runEvents.length} loaded</span></div>
    <div class="event-log">${state.runEvents.length ? state.runEvents.slice(-120).map(event => `
      <div class="event-row"><code>${esc(event.sequence ?? "-")}</code><span>${esc(event.type || "event")}</span><small>${esc(event.timestamp || "")}</small></div>`).join("") : `<div class="notice">No durable events received yet.</div>`}</div>
  `;
}

function renderInspect() {
  const snapshot = state.runSnapshot;
  return page("Inspect", "Bounded observational access to coherent episode-boundary snapshots", `
    ${state.connection !== "connected" ? disconnectedBlock() : ""}
    <section class="two-column">
      <div class="panel">
        <div class="panel-header"><div><h2>Observable tensors</h2><p>Tensor values are bounded to 256 per request and pinned to one snapshot ID.</p></div></div>
        ${state.selectedRun ? `<button class="button button-secondary" data-action="load-tensors">Load current episode-boundary snapshot</button>` : `<div class="notice">No run selected.</div>`}
        ${state.tensorError ? `<div class="validation-row bad">${esc(state.tensorError)}</div>` : ""}
        ${snapshot ? `<dl class="mini-details">${detail("Snapshot", snapshot.snapshot_id)}${detail("Last event sequence", snapshot.last_observed_sequence)}${detail("Private draft", snapshot.draft)}${detail("Committed response", snapshot.committed_response)}</dl>` : ""}
        <div class="list compact">
          ${state.tensors.map(tensor => `<button class="list-item" data-action="select-tensor" data-tensor-id="${esc(tensor.tensor_id)}"><span><strong>${esc(tensor.name || tensor.tensor_id)}</strong><small>${esc(tensor.semantic_role || "")}</small></span><code>${esc((tensor.shape || []).join("×"))}</code></button>`).join("")}
        </div>
      </div>
      <div class="panel tensor-detail">
        ${state.selectedTensor ? renderTensor(state.selectedTensor) : `<div class="canvas-empty"><strong>No tensor slice</strong><span>Reload the tensor list when a snapshot expires; stale snapshot slices are rejected explicitly.</span></div>`}
      </div>
    </section>
  `);
}

function renderTensor(t) {
  const stats = t.statistics || t.stats || t;
  const slice = t.values || t.value_slice || t.slice || [];
  return `
    <div class="panel-header"><div><h2>${esc(t.name || t.tensor_id)}</h2><p>${esc(t.semantic_role || "role unavailable")}</p></div></div>
    <dl class="detail-grid">
      ${detail("Shape", Array.isArray(t.shape) ? t.shape.join(" × ") : t.shape)}
      ${detail("Dtype", t.dtype)}
      ${detail("Device", t.device)}
      ${detail("Snapshot", t.snapshot_id)}
      ${detail("Step", t.step)}
      ${detail("Sampled", t.sampled === true ? "yes" : t.sampled === false ? "no" : null)}
      ${detail("Min", stats.min)}
      ${detail("Max", stats.max)}
      ${detail("Mean", stats.mean)}
      ${detail("Std", stats.std)}
      ${detail("Norm", stats.norm)}
    </dl>
    <div class="subheader"><h3>Bounded value slice</h3><span class="muted">Never interpreted as exact English without a decoder contract.</span></div>
    <pre class="tensor-values">${esc(JSON.stringify(slice, null, 2))}</pre>
  `;
}

function renderInference() {
  const enabled = Boolean(state.capabilities?.feature_flags?.inference);
  return page("Inference", "Selected-checkpoint inference remains unavailable until the backend exposes the real same-path session adapter", `
    ${state.connection !== "connected" ? disconnectedBlock() : ""}
    <section class="panel">
      <div class="panel-header split"><div><h2>Inference</h2><p>No generic restore or UI-only inference bypass is permitted.</p></div><span class="tag ${enabled ? "good" : "bad"}">${enabled ? "Available" : "Not integrated"}</span></div>
      ${enabled ? `<div class="notice">The backend now advertises inference; refresh this frontend slice before using it so the exact contract can be exercised.</div>` : `<div class="notice">Inference controls are intentionally disabled. Current checkpoints are observable artifacts only; generic restore/session endpoints remain unavailable.</div>`}
    </section>
  `);
}

function renderCheckpoints() {
  return page("Checkpoints", "Real episode-boundary artifacts; generic restore is intentionally unavailable", `
    ${state.connection !== "connected" ? disconnectedBlock() : ""}
    <section class="panel">
      <div class="panel-header"><div><h2>Checkpoint registry</h2><p>These artifacts come from the real E0 run adapter. Read-only display does not imply generic restore is available.</p></div></div>
      ${state.checkpoints.length ? `<div class="card-grid">${state.checkpoints.map(cp => `
        <article class="manifest-card">
          <div class="split"><strong>${esc(cp.checkpoint_id)}</strong><span class="tag ${cp.completeness === true ? "good" : "neutral"}">${cp.completeness === true ? "complete" : "unverified"}</span></div>
          <dl class="mini-details">
            ${detail("Run", cp.run_id)}
            ${detail("Step", cp.step)}
            ${detail("Parent", cp.parent_id)}
            ${detail("Boundary", cp.boundary)}
            ${detail("Mid-episode resume", cp.mid_episode_resume)}
            ${detail("Backup", cp.backup_state?.status || cp.backup_state)}
            ${detail("Restore verification", cp.restore_verification?.status || cp.restore_verification)}
            ${detail("Created", cp.created_at)}
          </dl>
          <div class="notice">Generic checkpoint restore is not integrated. Resume is only through the run lifecycle when the server allows it.</div>
        </article>`).join("")}</div>` : `<div class="notice">No checkpoints reported.</div>`}
    </section>
  `);
}

function renderBackup() {
  const b = state.backup;
  return page("Backup", "Valuable-artifact backup and restore evidence", `
    ${state.connection !== "connected" ? disconnectedBlock() : ""}
    <section class="panel">
      <div class="panel-header split"><div><h2>Artifact protection</h2><p>Configured destination is not proof of a completed backup or restore drill.</p></div><span class="tag ${statusClass(b?.status)}">${esc(b?.status || "unknown")}</span></div>
      <dl class="detail-grid">
        ${detail("Destination", b?.destination || b?.configured_destination)}
        ${detail("Last attempt", b?.last_attempt)}
        ${detail("Verified artifact", b?.verified_artifact)}
        ${detail("Restore drill", b?.restore_drill?.status || b?.restore_drill)}
        ${detail("Reason", b?.reason)}
      </dl>
    </section>
  `);
}

function page(title, subtitle, body) {
  return `<section class="view-heading"><div><div class="eyebrow">AXON LAB</div><h2>${esc(title)}</h2><p>${esc(subtitle)}</p></div><code class="api-base">${esc(api.baseUrl)}</code></section>${body}`;
}

function metricCard(label, value, hint) {
  return `<article class="metric-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(hint)}</small></article>`;
}

function capabilityPanel(title, items) {
  return `<div class="panel"><div class="panel-header"><div><h2>${esc(title)}</h2><p>Capability-driven inventory</p></div></div>${entityRows(items, `No ${title.toLowerCase()} reported.`)}</div>`;
}

function entityRows(items, emptyText) {
  if (!items.length) return `<div class="notice">${esc(emptyText)}</div>`;
  return `<div class="entity-list">${items.map(item => `
    <div class="entity-row"><div><strong>${esc(item.name || item.id)}</strong><small>${esc(item.reason || item.version || "")}</small></div><span class="tag ${statusClass(item.status)}">${esc(item.status || "unknown")}</span></div>`).join("")}</div>`;
}

function manifestCards(items, key) {
  if (!items.length) return `<div class="notice">Nothing reported by the backend.</div>`;
  return `<div class="card-grid">${items.map(item => `
    <article class="manifest-card">
      <strong>${esc(item.name || item[key] || item.id)}</strong>
      <small>${esc(item.version || item.stage || "")}</small>
      <dl class="mini-details">
        ${detail("ID", item[key] || item.id)}
        ${detail("Native-95", item.native95_status)}
        ${detail("Conversion", item.conversion_status)}
        ${detail("Held-out", item.heldout_policy)}
        ${detail("Dataset hash", item.dataset_hash)}
        ${detail("Curriculum hash", item.curriculum_hash)}
        ${detail("Manifest hash", item.manifest_hash)}
        ${detail("Splits", item.splits)}
        ${detail("Stages", item.stages)}
      </dl>
    </article>`).join("")}</div>`;
}

function selectField(label, name, items, key, optional = false) {
  return `<label class="field"><span>${esc(label)}</span><select name="${esc(name)}"><option value="">${optional ? "None" : "Select…"}</option>${items.map(item => `<option value="${esc(item[key] || item.id)}">${esc(item.name || item[key] || item.id)}</option>`).join("")}</select></label>`;
}

function detail(label, value) {
  const display = value === null || value === undefined || value === "" ? "Unavailable" : (typeof value === "object" ? JSON.stringify(value) : String(value));
  return `<div><dt>${esc(label)}</dt><dd>${esc(display)}</dd></div>`;
}

function invalidateArchitectureDraft() {
  state.architectureValidation = null;
  state.architecture.architecture_id = null;
  state.architecture.architecture_hash = null;
  state.architectureRegistration = {commandId: null, fingerprint: null, status: "idle", result: null, error: null};
  persistUiState();
}

function addNode(componentId) {
  const component = safeArray(state.capabilities?.components).find(x => x.id === componentId);
  if (!component || component.execution_eligible !== true || !component.configuration_schema || !Array.isArray(component.ports)) return;
  const node = {
    node_id: `node-${id()}`,
    component_type: component.id,
    component_version: component.version,
    display_name: component.name || component.id,
    config: defaultsFromSchema(component.configuration_schema) || {},
    ports: structuredClone(component.ports),
  };
  state.architecture.nodes.push(node);
  state.architectUi.selectedNodeId = node.node_id;
  invalidateArchitectureDraft();
  render();
}

function removeNode(nodeId) {
  state.architecture.nodes = state.architecture.nodes.filter(node => node.node_id !== nodeId);
  state.architecture.edges = state.architecture.edges.filter(edge => edge.source?.node_id !== nodeId && edge.destination?.node_id !== nodeId);
  if (state.architectUi.selectedNodeId === nodeId) state.architectUi.selectedNodeId = null;
  for (const field of ["sourceNode", "destinationNode"]) {
    if (state.architectUi.edgeDraft[field] === nodeId) state.architectUi.edgeDraft[field] = "";
  }
  invalidateArchitectureDraft();
  render();
}

function addEdge() {
  const draft = state.architectUi.edgeDraft;
  if (!draft.sourceNode || !draft.sourcePort || !draft.destinationNode || !draft.destinationPort) return;
  state.architecture.edges.push({
    source: {node_id: draft.sourceNode, port_id: draft.sourcePort},
    destination: {node_id: draft.destinationNode, port_id: draft.destinationPort},
  });
  state.architectUi.edgeDraft = {sourceNode: "", sourcePort: "", destinationNode: "", destinationPort: ""};
  invalidateArchitectureDraft();
  render();
}

async function validateArchitecture() {
  state.architecture.name = document.querySelector("#architecture-name")?.value || state.architecture.name;
  persistUiState();
  try {
    state.architectureValidation = await api.validateArchitecture(registrationPayload(state.architecture));
  } catch (error) {
    state.architectureValidation = {valid: false, execution_eligible: false, errors: [{message: error.message}], warnings: []};
  }
  render();
}

async function registerArchitecture() {
  state.architecture.name = document.querySelector("#architecture-name")?.value || state.architecture.name;
  if (!state.architectureValidation?.valid) {
    alert("Validate the draft successfully before registration.");
    return;
  }
  const payload = registrationPayload(state.architecture);
  const fingerprint = stableFingerprint(payload);
  if (state.architectureRegistration.fingerprint !== fingerprint || !state.architectureRegistration.commandId) {
    state.architectureRegistration = {
      commandId: id(),
      fingerprint,
      status: "ready",
      result: null,
      error: null,
    };
  }
  state.architectureRegistration.status = "submitting";
  state.architectureRegistration.error = null;
  persistUiState();
  render();
  try {
    const registered = await api.registerArchitecture({...payload, command_id: state.architectureRegistration.commandId});
    state.architecture.architecture_id = registered.architecture_id || null;
    state.architecture.version = registered.version || state.architecture.version;
    state.architecture.architecture_hash = registered.architecture_hash || registered.hash || null;
    state.architectureRegistration.status = "completed";
    state.architectureRegistration.result = registered;
    state.registeredArchitectures = safeArray(await api.architectures());
  } catch (error) {
    state.architectureRegistration.status = "error";
    state.architectureRegistration.error = error.message;
  }
  persistUiState();
  render();
}

function ensurePreflightController() {
  if (!preflightController) {
    preflightController = new PreflightController({
      api,
      preflight: state.preflight,
      makeId: id,
      persist: persistUiState,
      onChange: () => {
        if (state.view === "system") render();
      },
      refreshEvidence,
    });
  }
  return preflightController;
}

async function startPreflight(device, {reuseCommand = false} = {}) {
  if (state.connection !== "connected") return;
  return ensurePreflightController().start(device, {reuseCommand});
}

async function pollPreflightOperation() {
  if (state.connection !== "connected") return;
  return ensurePreflightController().poll();
}

function resumePreflightPolling() {
  if (state.connection !== "connected") return null;
  return ensurePreflightController().resume();
}

function ensureRunMutationController() {
  if (!runMutationController) {
    runMutationController = new RunMutationController({
      api,
      state: state.runMutation,
      makeId: id,
      persist: persistUiState,
      onChange: () => {
        if (state.view === "train" || state.view === "runs") render();
      },
      refreshRuns: refreshRunsData,
    });
  }
  return runMutationController;
}

async function refreshRunsData() {
  const [runs, checkpoints] = await Promise.all([api.runs(), api.checkpoints()]);
  state.runs = safeArray(runs);
  state.checkpoints = safeArray(checkpoints);
  if (state.selectedRun) state.selectedRun = await api.run(state.selectedRun.run_id);
}

async function refreshRuns() {
  try {
    await refreshRunsData();
  } catch (error) {
    state.connectionError = error;
  }
  render();
}

function appendRunEvents(events) {
  for (const event of safeArray(events)) {
    const sequence = Number(event.sequence || 0);
    if (!Number.isFinite(sequence) || sequence <= state.runEventCursor) continue;
    state.runEvents.push(event);
    state.runEventCursor = sequence;
  }
  if (state.runEvents.length > 500) state.runEvents.splice(0, state.runEvents.length - 500);
}

async function pollRunEvents(runId) {
  if (state.selectedRun?.run_id !== runId) return;
  try {
    const payload = await api.runEvents(runId, state.runEventCursor);
    appendRunEvents(payload);
    state.selectedRun = await api.run(runId);
    state.eventMode = "json-poll";
    if (state.view === "runs") render();
    const terminal = ["completed", "stopped", "failed"].includes(state.selectedRun.lifecycle_state);
    if (!terminal) {
      state.eventPollTimer = setTimeout(() => pollRunEvents(runId), 900);
    }
  } catch {
    state.eventMode = "json-retry";
    if (state.view === "runs") render();
    state.eventPollTimer = setTimeout(() => pollRunEvents(runId), 1500);
  }
}

async function selectRun(runId) {
  closeEventSource();
  try {
    state.selectedRun = await api.run(runId);
    state.runEvents = [];
    state.runEventCursor = 0;
    state.eventMode = "loading-history";
    const history = await api.runEvents(runId, 0);
    appendRunEvents(history);
    subscribeRunEvents(runId);
  } catch (error) {
    state.connectionError = error;
  }
  render();
}

function subscribeRunEvents(runId) {
  try {
    const source = api.eventSourceForRun(runId, state.runEventCursor);
    state.eventSource = source;
    state.eventMode = "sse";
    source.onmessage = event => {
      try {
        const payload = JSON.parse(event.data);
        appendRunEvents([payload]);
        if (["episode_result", "snapshot", "checkpoint", "lifecycle"].includes(payload.type)) {
          void api.run(runId).then(run => {
            if (state.selectedRun?.run_id === runId) {
              state.selectedRun = run;
              if (state.view === "runs") render();
            }
          }).catch(() => {});
        }
        if (state.view === "runs") render();
      } catch {}
    };
    source.onerror = () => {
      if (state.selectedRun?.run_id !== runId) return;
      source.close();
      if (state.eventSource === source) state.eventSource = null;
      state.eventMode = "json-poll";
      if (state.eventPollTimer) clearTimeout(state.eventPollTimer);
      state.eventPollTimer = setTimeout(() => pollRunEvents(runId), 150);
      if (state.view === "runs") render();
    };
  } catch {
    state.eventMode = "json-poll";
    state.eventPollTimer = setTimeout(() => pollRunEvents(runId), 150);
  }
}

function closeEventSource() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  if (state.eventPollTimer) {
    clearTimeout(state.eventPollTimer);
    state.eventPollTimer = null;
  }
  state.eventMode = "idle";
}

async function runCommand(command) {
  if (!state.selectedRun) return;
  await ensureRunMutationController().command(state.selectedRun.run_id, command);
}

async function loadTensors() {
  if (!state.selectedRun) return;
  state.tensorError = null;
  state.selectedTensor = null;
  try {
    const [tensors, snapshot] = await Promise.all([
      api.runTensors(state.selectedRun.run_id),
      api.runSnapshot(state.selectedRun.run_id),
    ]);
    state.tensors = safeArray(tensors);
    state.runSnapshot = snapshot;
  } catch (error) {
    state.tensors = [];
    state.runSnapshot = null;
    state.tensorError = error.message;
  }
  render();
}

async function selectTensor(tensorId) {
  if (!state.selectedRun) return;
  state.tensorError = null;
  const metadata = state.tensors.find(item => item.tensor_id === tensorId);
  const snapshotId = metadata?.snapshot_id || state.runSnapshot?.snapshot_id || null;
  try {
    state.selectedTensor = await api.tensor(
      state.selectedRun.run_id,
      tensorId,
      {offset: 0, count: 256, ...(snapshotId ? {snapshot_id: snapshotId} : {})}
    );
  } catch (error) {
    state.selectedTensor = null;
    state.tensorError = error.code === "snapshot_expired"
      ? "Snapshot expired. Reload the current episode-boundary tensor list before requesting values."
      : error.message;
  }
  render();
}

async function createRun(form) {
  const data = new FormData(form);
  const architectureId = data.get("architecture") || null;
  const registered = state.registeredArchitectures.find(item => item.architecture_id === architectureId) || {};
  const spec = {
    architecture_id: architectureId,
    architecture_version: registered.version || null,
    architecture_hash: registered.architecture_hash || registered.hash || null,
    dataset_id: data.get("dataset") || null,
    curriculum_id: data.get("curriculum") || null,
    device: data.get("device") || null,
    provider: data.get("provider") || null,
    split: "train",
    seed: Number(data.get("seed") || 1),
    training_settings: {
      epochs: Number(data.get("epochs") || 1),
      learning_rate: Number(data.get("learning_rate") || 0.001),
    },
  };
  await ensureRunMutationController().prepare(spec);
}

async function restoreCheckpoint(checkpointId) {
  try {
    const op = await api.restoreCheckpoint(checkpointId, {command_id: id()});
    alert(`Restore request accepted as operation ${op.operation_id || "unknown"}. Verify completion before use.`);
  } catch (error) {
    alert(`Restore failed: ${error.message}`);
  }
}

async function createInference(form) {
  const data = new FormData(form);
  try {
    const response = await api.createInferenceSession({checkpoint_id: data.get("checkpoint"), command_id: id()});
    state.inferenceSession = response;
  } catch (error) {
    alert(`Inference session failed: ${error.message}`);
  }
  render();
}

async function sendInference(command) {
  const session = state.inferenceSession;
  if (!session) return;
  const textarea = document.querySelector("#inference-input");
  const text = textarea?.value || "";
  if (command === "input") {
    const check = analyzeNativeText(text, state.capabilities?.native_alphabet);
    if (!check.valid) {
      updateInputValidity(text);
      return;
    }
  }
  const body = {command, command_id: id()};
  if (command === "input") body.input = text;
  try {
    await api.inferenceCommand(session.session_id, body);
    setTimeout(async () => {
      try {
        state.inferenceSession = await api.inferenceSession(session.session_id);
        render();
      } catch {}
    }, 300);
  } catch (error) {
    alert(`Inference command failed: ${error.message}`);
  }
}

function updateInputValidity(text) {
  const el = document.querySelector("#input-validity");
  if (!el) return;
  const alphabet = state.capabilities?.native_alphabet;
  if (typeof alphabet !== "string") {
    el.textContent = "The backend has not supplied native_alphabet; input cannot be locally validated.";
    el.className = "validation-text warn";
    return;
  }
  const result = analyzeNativeText(text, alphabet);
  if (!result.valid) {
    const preview = result.invalid.slice(0, 8).map(item => `${displayCharacter(item.character)} @ ${item.offset}`).join(", ");
    el.textContent = `Rejected by client check: ${result.invalid.length} character(s) are outside native_alphabet: ${preview}${result.invalid.length > 8 ? "…" : ""}`;
    el.className = "validation-text bad";
  } else {
    el.textContent = "Valid against the exact backend native_alphabet. Newline is accepted when present in that alphabet.";
    el.className = "validation-text good";
  }
}

function updateConfigField(element) {
  const node = state.architecture.nodes.find(item => item.node_id === state.architectUi.selectedNodeId);
  if (!node) return;
  const path = element.dataset.configPath;
  const type = element.dataset.configType;
  let value = element.value;
  try {
    if (type === "integer") value = value === "" ? null : Number.parseInt(value, 10);
    else if (type === "number") value = value === "" ? null : Number(value);
    else if (type === "boolean") value = value === "true";
    else if (type === "json") value = JSON.parse(value);
  } catch {
    element.setCustomValidity("Enter valid JSON.");
    element.reportValidity();
    return;
  }
  element.setCustomValidity("");
  setAtPath(node.config, path, value);
  invalidateArchitectureDraft();
}

function bindViewEvents() {
  dom.main.querySelectorAll("[data-action='retry']").forEach(el => el.addEventListener("click", connect));

  dom.main.querySelectorAll("[data-action='start-preflight']").forEach(el => el.addEventListener("click", () => startPreflight(el.dataset.device)));
  dom.main.querySelector("[data-action='retry-preflight']")?.addEventListener("click", () => startPreflight(state.preflight.device, {reuseCommand: true}));
  dom.main.querySelector("[data-action='poll-preflight']")?.addEventListener("click", pollPreflightOperation);

  dom.main.querySelectorAll("[data-action='add-node']").forEach(el => el.addEventListener("click", () => addNode(el.dataset.componentId)));
  dom.main.querySelectorAll("[data-action='select-node']").forEach(el => el.addEventListener("click", () => {
    state.architectUi.selectedNodeId = el.dataset.nodeId;
    persistUiState();
    render();
  }));
  dom.main.querySelectorAll("[data-action='remove-node']").forEach(el => el.addEventListener("click", () => removeNode(el.dataset.nodeId)));
  dom.main.querySelector("[data-action='clear-graph']")?.addEventListener("click", () => {
    state.architecture = {...freshArchitecture(), name: state.architecture.name, version: state.architecture.version};
    state.architectUi.selectedNodeId = null;
    state.architectUi.edgeDraft = {sourceNode: "", sourcePort: "", destinationNode: "", destinationPort: ""};
    invalidateArchitectureDraft();
    render();
  });
  dom.main.querySelector("[data-action='validate-architecture']")?.addEventListener("click", validateArchitecture);
  dom.main.querySelector("[data-action='register-architecture']")?.addEventListener("click", registerArchitecture);
  dom.main.querySelector("#architecture-name")?.addEventListener("input", event => {
    state.architecture.name = event.target.value;
    invalidateArchitectureDraft();
    const button = dom.main.querySelector("[data-action='register-architecture']");
    if (button) button.disabled = true;
  });

  dom.main.querySelectorAll("[data-config-path]").forEach(el => {
    el.addEventListener("change", () => updateConfigField(el));
  });

  dom.main.querySelectorAll("[data-edge-field]").forEach(el => {
    el.addEventListener("change", () => {
      const field = el.dataset.edgeField;
      state.architectUi.edgeDraft[field] = el.value;
      if (field === "sourceNode") state.architectUi.edgeDraft.sourcePort = "";
      if (field === "destinationNode") state.architectUi.edgeDraft.destinationPort = "";
      persistUiState();
      render();
    });
  });
  dom.main.querySelector("[data-action='add-edge']")?.addEventListener("click", addEdge);
  dom.main.querySelectorAll("[data-action='remove-edge']").forEach(el => el.addEventListener("click", () => {
    state.architecture.edges.splice(Number(el.dataset.edgeIndex), 1);
    invalidateArchitectureDraft();
    render();
  }));

  dom.main.querySelector("#create-run-form")?.addEventListener("submit", event => { event.preventDefault(); createRun(event.currentTarget); });
  dom.main.querySelectorAll("[data-action='recover-run-mutation']").forEach(el => el.addEventListener("click", () => ensureRunMutationController().recover(el.dataset.kind)));
  dom.main.querySelector("[data-action='refresh-runs']")?.addEventListener("click", refreshRuns);
  dom.main.querySelectorAll("[data-action='select-run']").forEach(el => el.addEventListener("click", () => selectRun(el.dataset.runId)));
  dom.main.querySelectorAll("[data-action='run-command']").forEach(el => el.addEventListener("click", () => runCommand(el.dataset.command)));

  dom.main.querySelector("[data-action='load-tensors']")?.addEventListener("click", loadTensors);
  dom.main.querySelectorAll("[data-action='select-tensor']").forEach(el => el.addEventListener("click", () => selectTensor(el.dataset.tensorId)));

  dom.main.querySelector("#inference-input")?.addEventListener("input", event => updateInputValidity(event.target.value));
}

dom.nav.forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
dom.retry.addEventListener("click", connect);
window.addEventListener("beforeunload", () => {
  persistUiState();
  closeEventSource();
  preflightController?.dispose();
  runMutationController?.dispose();
});

persistUiState();
render();
connect();
