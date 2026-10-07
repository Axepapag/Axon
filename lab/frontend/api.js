export class ApiError extends Error {
  constructor(message, {status = 0, body = null, code = null, retryable = false} = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.code = code;
    this.retryable = retryable;
  }
}

export function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const configured = params.get("api");
  if (configured) return configured.replace(/\/$/, "");
  return "/api/v1";
}

export class AxonApi {
  constructor(baseUrl = resolveApiBase()) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async request(path, {method = "GET", body, signal, headers = {}} = {}) {
    const options = {
      method,
      signal,
      headers: {
        Accept: "application/json",
        ...headers,
      },
    };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }

    let response;
    try {
      response = await fetch(this.baseUrl + path, options);
    } catch (error) {
      throw new ApiError("Backend is unreachable.", {body: String(error), retryable: true});
    }

    let payload = null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try { payload = await response.json(); } catch { payload = null; }
    } else {
      try { payload = await response.text(); } catch { payload = null; }
    }

    if (!response.ok) {
      const message = payload?.message || payload?.error?.message || `Request failed (${response.status})`;
      throw new ApiError(message, {
        status: response.status,
        body: payload,
        code: payload?.code || payload?.error?.code || null,
        retryable: Boolean(payload?.retryable || payload?.error?.retryable),
      });
    }
    return payload;
  }

  capabilities(signal) { return this.request("/capabilities", {signal}); }
  readiness(signal) { return this.request("/readiness", {signal}); }
  preflight(spec) { return this.request("/preflight", {method: "POST", body: spec}); }
  datasets(signal) { return this.request("/datasets", {signal}); }
  curricula(signal) { return this.request("/curricula", {signal}); }
  architectures(signal) { return this.request("/architectures", {signal}); }
  validateArchitecture(definition) {
    return this.request("/architectures/validate", {method: "POST", body: definition});
  }
  registerArchitecture(definition) {
    return this.request("/architectures", {method: "POST", body: definition});
  }
  runs(signal) { return this.request("/runs", {signal}); }
  run(id, signal) { return this.request(`/runs/${encodeURIComponent(id)}`, {signal}); }
  createRun(spec) { return this.request("/runs", {method: "POST", body: spec}); }
  commandRun(id, command) {
    return this.request(`/runs/${encodeURIComponent(id)}/commands`, {method: "POST", body: command});
  }
  runEvents(id, afterSequence = 0, signal) {
    const query = new URLSearchParams({after_sequence: String(afterSequence)}).toString();
    return this.request(`/runs/${encodeURIComponent(id)}/events?${query}`, {signal});
  }
  runSnapshot(id, signal) {
    return this.request(`/runs/${encodeURIComponent(id)}/snapshot`, {signal});
  }
  runTensors(id, signal) {
    return this.request(`/runs/${encodeURIComponent(id)}/tensors`, {signal});
  }
  tensor(id, tensorId, params = {}, signal) {
    const query = new URLSearchParams(params).toString();
    const suffix = query ? `?${query}` : "";
    return this.request(`/runs/${encodeURIComponent(id)}/tensors/${encodeURIComponent(tensorId)}${suffix}`, {signal});
  }
  checkpoints(signal) { return this.request("/checkpoints", {signal}); }
  checkpoint(id, signal) { return this.request(`/checkpoints/${encodeURIComponent(id)}`, {signal}); }
  restoreCheckpoint(id, command) {
    return this.request(`/checkpoints/${encodeURIComponent(id)}/restore`, {method: "POST", body: command});
  }
  createInferenceSession(spec) {
    return this.request("/inference/sessions", {method: "POST", body: spec});
  }
  inferenceSession(id, signal) {
    return this.request(`/inference/sessions/${encodeURIComponent(id)}`, {signal});
  }
  inferenceCommand(id, command) {
    return this.request(`/inference/sessions/${encodeURIComponent(id)}/commands`, {method: "POST", body: command});
  }
  backupStatus(signal) { return this.request("/backup/status", {signal}); }
  operation(id, signal) { return this.request(`/operations/${encodeURIComponent(id)}`, {signal}); }

  eventSourceForRun(runId, afterSequence = null) {
    const params = new URLSearchParams();
    if (afterSequence !== null) params.set("after_sequence", String(afterSequence));
    const suffix = params.toString() ? `?${params}` : "";
    return new EventSource(this.baseUrl + `/runs/${encodeURIComponent(runId)}/events${suffix}`);
  }
}
