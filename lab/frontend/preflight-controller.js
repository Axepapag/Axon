const TERMINAL_STATUSES = new Set(["completed", "failed", "interrupted"]);
const ACTIVE_STATUSES = new Set(["submitting", "queued", "running"]);

export function isPreflightTerminal(status) {
  return TERMINAL_STATUSES.has(status);
}

export function isPreflightBusy(status) {
  return ACTIVE_STATUSES.has(status);
}

export function canRecoverUnacknowledged(preflight) {
  return Boolean(
    preflight?.commandId &&
    !preflight?.operationId &&
    ["submitting", "uncertain", "error"].includes(preflight?.status)
  );
}

export class PreflightController {
  constructor({
    api,
    preflight,
    makeId,
    persist = () => {},
    onChange = () => {},
    refreshEvidence = async () => {},
    schedule = (fn, delay) => setTimeout(fn, delay),
    cancel = handle => clearTimeout(handle),
  }) {
    this.api = api;
    this.preflight = preflight;
    this.makeId = makeId;
    this.persist = persist;
    this.onChange = onChange;
    this.refreshEvidence = refreshEvidence;
    this.schedule = schedule;
    this.cancel = cancel;
    this.pollTimer = null;

    if (!Number.isInteger(this.preflight.generation)) this.preflight.generation = 0;
    if (!Object.prototype.hasOwnProperty.call(this.preflight, "evidenceRefreshError")) {
      this.preflight.evidenceRefreshError = null;
    }
  }

  dispose() {
    if (this.pollTimer !== null) {
      this.cancel(this.pollTimer);
      this.pollTimer = null;
    }
  }

  emit() {
    this.persist();
    this.onChange();
  }

  isCurrent({generation, commandId, operationId = undefined}) {
    if (this.preflight.generation !== generation) return false;
    if (this.preflight.commandId !== commandId) return false;
    if (operationId !== undefined && this.preflight.operationId !== operationId) return false;
    return true;
  }

  beginNewSelection(device) {
    this.dispose();
    this.preflight.generation += 1;
    this.preflight.device = device;
    this.preflight.commandId = this.makeId();
    this.preflight.operationId = null;
    this.preflight.operation = null;
    this.preflight.status = "idle";
    this.preflight.error = null;
    this.preflight.evidenceRefreshError = null;
  }

  async start(device, {reuseCommand = false} = {}) {
    const recoveringUnacknowledged =
      reuseCommand &&
      this.preflight.device === device &&
      canRecoverUnacknowledged(this.preflight);

    if (isPreflightBusy(this.preflight.status) && !recoveringUnacknowledged) {
      return {ignored: true, reason: "busy"};
    }

    if (!reuseCommand || !this.preflight.commandId || this.preflight.device !== device) {
      this.beginNewSelection(device);
    }

    const selection = {
      generation: this.preflight.generation,
      commandId: this.preflight.commandId,
    };

    this.preflight.device = device;
    this.preflight.status = "submitting";
    this.preflight.error = null;
    this.preflight.evidenceRefreshError = null;
    this.emit();

    let ack;
    try {
      ack = await this.api.preflight({
        command_id: selection.commandId,
        device,
      });
    } catch (error) {
      if (this.isCurrent(selection) && !this.preflight.operationId) {
        this.preflight.status = "uncertain";
        this.preflight.error =
          `Preflight acknowledgement was not received: ${error.message}. Recover with the same command ID; do not create a new request.`;
        this.emit();
      }
      return {uncertain: true, error};
    }

    if (!this.isCurrent(selection)) return {stale: true};

    this.preflight.operationId = ack.operation_id;
    this.preflight.operation = ack;
    this.preflight.status = ack.status || "queued";
    this.preflight.error = null;
    this.emit();

    this.schedulePoll(150, {
      ...selection,
      operationId: ack.operation_id,
    });
    return ack;
  }

  async recoverUnacknowledged() {
    if (!canRecoverUnacknowledged(this.preflight)) return {ignored: true};
    return this.start(this.preflight.device || "auto", {reuseCommand: true});
  }

  resume() {
    if (canRecoverUnacknowledged(this.preflight)) {
      return this.recoverUnacknowledged();
    }
    if (
      this.preflight.operationId &&
      !isPreflightTerminal(this.preflight.status)
    ) {
      this.schedulePoll(100, {
        generation: this.preflight.generation,
        commandId: this.preflight.commandId,
        operationId: this.preflight.operationId,
      });
    }
    return null;
  }

  schedulePoll(delay = 700, selection = null) {
    if (this.pollTimer !== null) this.cancel(this.pollTimer);
    const chosen = selection || {
      generation: this.preflight.generation,
      commandId: this.preflight.commandId,
      operationId: this.preflight.operationId,
    };
    this.pollTimer = this.schedule(() => {
      this.pollTimer = null;
      void this.poll(chosen);
    }, delay);
  }

  async poll(selection = null) {
    const chosen = selection || {
      generation: this.preflight.generation,
      commandId: this.preflight.commandId,
      operationId: this.preflight.operationId,
    };
    if (!chosen.operationId) return {ignored: true};

    let operation;
    try {
      operation = await this.api.operation(chosen.operationId);
    } catch (error) {
      if (this.isCurrent(chosen)) {
        this.preflight.status = "poll_error";
        this.preflight.error = error.message;
        this.emit();
      }
      return {error};
    }

    if (!this.isCurrent(chosen)) return {stale: true};

    this.preflight.operation = operation;
    this.preflight.status = operation.status || "unknown";
    this.preflight.error = operation.error?.message || null;

    if (isPreflightTerminal(operation.status)) {
      this.preflight.cachedResult = structuredClone(operation);
      this.preflight.evidenceRefreshError = null;

      // Persist the authoritative terminal probe before any auxiliary refresh.
      this.emit();

      try {
        await this.refreshEvidence();
      } catch (error) {
        if (this.isCurrent(chosen)) {
          this.preflight.evidenceRefreshError =
            `Terminal operation is preserved, but readiness/capability refresh failed: ${error.message}`;
          this.emit();
        }
        return operation;
      }

      if (this.isCurrent(chosen)) this.emit();
      return operation;
    }

    this.emit();
    this.schedulePoll(700, chosen);
    return operation;
  }
}
