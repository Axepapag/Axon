const TERMINAL = new Set(["completed", "failed", "interrupted"]);
const ACTIVE = new Set(["submitting", "queued", "running"]);

function clone(value) {
  return value == null ? value : structuredClone(value);
}

function normalized(value) {
  if (Array.isArray(value)) return value.map(normalized);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, normalized(value[key])]));
  }
  return value;
}

function fingerprint(value) {
  return JSON.stringify(normalized(value));
}

export function isRunMutationTerminal(status) {
  return TERMINAL.has(status);
}

export function isRunMutationBusy(status) {
  return ACTIVE.has(status);
}

export function uncertainSubmission(slot) {
  return Boolean(slot?.commandId && !slot?.operationId && ["submitting", "uncertain"].includes(slot?.status));
}

export function defaultRunMutationState() {
  return {
    creation: {
      generation: 0,
      commandId: null,
      fingerprint: null,
      payload: null,
      status: "idle",
      operationId: null,
      runId: null,
      operation: null,
      error: null,
      refreshError: null,
    },
    command: {
      generation: 0,
      commandId: null,
      runId: null,
      command: null,
      status: "idle",
      operationId: null,
      operation: null,
      error: null,
      refreshError: null,
    },
  };
}

export class RunMutationController {
  constructor({
    api,
    state,
    makeId,
    persist = () => {},
    onChange = () => {},
    refreshRuns = async () => {},
    schedule = (fn, delay) => setTimeout(fn, delay),
    cancel = handle => clearTimeout(handle),
  }) {
    this.api = api;
    this.state = state;
    this.makeId = makeId;
    this.persist = persist;
    this.onChange = onChange;
    this.refreshRuns = refreshRuns;
    this.schedule = schedule;
    this.cancel = cancel;
    this.timers = new Map();

    const defaults = defaultRunMutationState();
    this.state.creation = {...defaults.creation, ...(this.state.creation || {})};
    this.state.command = {...defaults.command, ...(this.state.command || {})};
  }

  emit() {
    this.persist();
    this.onChange();
  }

  dispose() {
    for (const timer of this.timers.values()) this.cancel(timer);
    this.timers.clear();
  }

  current(kind, selection) {
    const slot = this.state[kind];
    if (!slot) return false;
    if (slot.generation !== selection.generation) return false;
    if (slot.commandId !== selection.commandId) return false;
    if (selection.operationId !== undefined && slot.operationId !== selection.operationId) return false;
    return true;
  }

  newCreation(payload) {
    const slot = this.state.creation;
    slot.generation += 1;
    slot.commandId = this.makeId();
    slot.fingerprint = fingerprint(payload);
    slot.payload = clone(payload);
    slot.status = "idle";
    slot.operationId = null;
    slot.runId = null;
    slot.operation = null;
    slot.error = null;
    slot.refreshError = null;
  }

  async prepare(payload, {reuseCommand = false} = {}) {
    const slot = this.state.creation;
    const samePayload = slot.fingerprint === fingerprint(payload);
    const recovering = reuseCommand && samePayload && uncertainSubmission(slot);

    if (isRunMutationBusy(slot.status) && !recovering) return {ignored: true, reason: "busy"};
    if (!reuseCommand || !slot.commandId || !samePayload) this.newCreation(payload);

    const selection = {generation: slot.generation, commandId: slot.commandId};
    slot.status = "submitting";
    slot.error = null;
    slot.refreshError = null;
    this.emit();

    let ack;
    try {
      ack = await this.api.createRun({...slot.payload, command_id: slot.commandId});
    } catch (error) {
      if (this.current("creation", selection) && !slot.operationId) {
        if (error?.status) {
          slot.status = "failed";
          slot.error = error.message;
          slot.operation = {status: "failed", error: {code: error.code, message: error.message}, details: error.body?.details};
        } else {
          slot.status = "uncertain";
          slot.error = `Run preparation acknowledgement was not received: ${error.message}. Recover with the same command ID.`;
        }
        this.emit();
      }
      return {uncertain: !error?.status, error};
    }

    if (!this.current("creation", selection)) return {stale: true};
    slot.operationId = ack.operation_id || null;
    slot.runId = ack.run_id || null;
    slot.operation = ack;
    slot.status = ack.status || "queued";
    slot.error = null;
    this.emit();

    if (isRunMutationTerminal(slot.status)) {
      await this.afterTerminal("creation", selection);
    } else if (slot.operationId) {
      this.schedulePoll("creation", {...selection, operationId: slot.operationId}, 150);
    }
    return ack;
  }

  newCommand(runId, command) {
    const slot = this.state.command;
    slot.generation += 1;
    slot.commandId = this.makeId();
    slot.runId = runId;
    slot.command = command;
    slot.status = "idle";
    slot.operationId = null;
    slot.operation = null;
    slot.error = null;
    slot.refreshError = null;
  }

  async command(runId, command, {reuseCommand = false} = {}) {
    const slot = this.state.command;
    const same = slot.runId === runId && slot.command === command;
    const recovering = reuseCommand && same && uncertainSubmission(slot);

    if (isRunMutationBusy(slot.status) && !recovering) return {ignored: true, reason: "busy"};
    if (!reuseCommand || !slot.commandId || !same) this.newCommand(runId, command);

    const selection = {generation: slot.generation, commandId: slot.commandId};
    slot.status = "submitting";
    slot.error = null;
    slot.refreshError = null;
    this.emit();

    let ack;
    try {
      ack = await this.api.commandRun(runId, {command, command_id: slot.commandId});
    } catch (error) {
      if (this.current("command", selection) && !slot.operationId) {
        // A definitive server rejection is not an uncertain submission. Network
        // failure is; the same command ID must be used to discover the truth.
        if (error?.status) {
          slot.status = "failed";
          slot.error = error.message;
          slot.operation = {status: "failed", error: {code: error.code, message: error.message}, details: error.body?.details};
        } else {
          slot.status = "uncertain";
          slot.error = `Command acknowledgement was not received: ${error.message}. Recover with the same command ID.`;
        }
        this.emit();
      }
      return {uncertain: !error?.status, error};
    }

    if (!this.current("command", selection)) return {stale: true};
    slot.operationId = ack.operation_id || null;
    slot.operation = ack;
    slot.status = ack.status || "queued";
    slot.error = null;
    this.emit();

    if (isRunMutationTerminal(slot.status)) {
      await this.afterTerminal("command", selection);
    } else if (slot.operationId) {
      this.schedulePoll("command", {...selection, operationId: slot.operationId}, 150);
    }
    return ack;
  }

  async recover(kind) {
    const slot = this.state[kind];
    if (!uncertainSubmission(slot)) return {ignored: true};
    if (kind === "creation") return this.prepare(slot.payload, {reuseCommand: true});
    return this.command(slot.runId, slot.command, {reuseCommand: true});
  }

  resume() {
    for (const kind of ["creation", "command"]) {
      const slot = this.state[kind];
      if (uncertainSubmission(slot)) {
        void this.recover(kind);
      } else if (slot.operationId && !isRunMutationTerminal(slot.status)) {
        this.schedulePoll(kind, {
          generation: slot.generation,
          commandId: slot.commandId,
          operationId: slot.operationId,
        }, 100);
      }
    }
  }

  schedulePoll(kind, selection, delay = 700) {
    const old = this.timers.get(kind);
    if (old !== undefined) this.cancel(old);
    const timer = this.schedule(() => {
      this.timers.delete(kind);
      void this.poll(kind, selection);
    }, delay);
    this.timers.set(kind, timer);
  }

  async poll(kind, selection = null) {
    const slot = this.state[kind];
    const chosen = selection || {
      generation: slot.generation,
      commandId: slot.commandId,
      operationId: slot.operationId,
    };
    if (!chosen.operationId) return {ignored: true};

    let operation;
    try {
      operation = await this.api.operation(chosen.operationId);
    } catch (error) {
      if (this.current(kind, chosen)) {
        slot.status = "poll_error";
        slot.error = error.message;
        this.emit();
      }
      return {error};
    }

    if (!this.current(kind, chosen)) return {stale: true};
    slot.operation = operation;
    slot.status = operation.status || "unknown";
    slot.error = operation.error?.message || null;
    if (operation.run_id) slot.runId = operation.run_id;
    this.emit();

    if (isRunMutationTerminal(slot.status)) {
      await this.afterTerminal(kind, chosen);
    } else {
      this.schedulePoll(kind, chosen, 700);
    }
    return operation;
  }

  async afterTerminal(kind, selection) {
    const slot = this.state[kind];
    // Terminal operation evidence is authoritative even if refresh fails.
    slot.refreshError = null;
    this.emit();
    try {
      await this.refreshRuns();
      if (this.current(kind, selection)) this.emit();
    } catch (error) {
      if (this.current(kind, selection)) {
        slot.refreshError = `Operation status is preserved, but run refresh failed: ${error.message}`;
        this.emit();
      }
    }
  }
}
