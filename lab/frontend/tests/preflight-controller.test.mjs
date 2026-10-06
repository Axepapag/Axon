import test from "node:test";
import assert from "node:assert/strict";
import {PreflightController} from "../preflight-controller.js";

function baseState(overrides = {}) {
  return {
    device: "auto",
    commandId: null,
    operationId: null,
    status: "idle",
    operation: null,
    error: null,
    cachedResult: null,
    generation: 0,
    evidenceRefreshError: null,
    ...overrides,
  };
}

function noSchedule() {
  return 1;
}

test("accepted-but-ack-lost recovers with the same command_id", async () => {
  const server = new Map();
  let calls = 0;
  const api = {
    async preflight(body) {
      calls += 1;
      if (!server.has(body.command_id)) {
        server.set(body.command_id, {
          operation_id: "op-accepted",
          status: "queued",
        });
      }
      if (calls === 1) throw new Error("ack lost after accept");
      return server.get(body.command_id);
    },
    async operation() {
      throw new Error("not used");
    },
  };

  const preflight = baseState();
  const controller = new PreflightController({
    api,
    preflight,
    makeId: () => "cmd-stable",
    schedule: noSchedule,
  });

  await controller.start("auto");
  assert.equal(preflight.status, "uncertain");
  assert.equal(preflight.commandId, "cmd-stable");
  assert.equal(preflight.operationId, null);

  await controller.resume();
  assert.equal(calls, 2);
  assert.equal(preflight.commandId, "cmd-stable");
  assert.equal(preflight.operationId, "op-accepted");
  assert.equal(preflight.status, "queued");
});

test("never-reached-server recovers by replaying the same command_id", async () => {
  let calls = 0;
  const seen = [];
  const api = {
    async preflight(body) {
      calls += 1;
      seen.push(body.command_id);
      if (calls === 1) throw new Error("network failed before delivery");
      return {operation_id: "op-created-on-retry", status: "queued"};
    },
    async operation() {
      throw new Error("not used");
    },
  };

  const preflight = baseState();
  const controller = new PreflightController({
    api,
    preflight,
    makeId: () => "cmd-never-reached",
    schedule: noSchedule,
  });

  await controller.start("cpu");
  assert.equal(preflight.status, "uncertain");
  await controller.resume();

  assert.deepEqual(seen, ["cmd-never-reached", "cmd-never-reached"]);
  assert.equal(preflight.operationId, "op-created-on-retry");
});

test("reload from persisted submitting state automatically replays same command_id", async () => {
  const seen = [];
  const api = {
    async preflight(body) {
      seen.push(body);
      return {operation_id: "op-after-reload", status: "running"};
    },
    async operation() {
      throw new Error("not used");
    },
  };

  const preflight = baseState({
    device: "cuda:0",
    commandId: "cmd-persisted",
    operationId: null,
    status: "submitting",
    generation: 4,
  });

  const controller = new PreflightController({
    api,
    preflight,
    makeId: () => "must-not-be-used",
    schedule: noSchedule,
  });

  await controller.resume();

  assert.equal(seen.length, 1);
  assert.equal(seen[0].command_id, "cmd-persisted");
  assert.equal(preflight.commandId, "cmd-persisted");
  assert.equal(preflight.operationId, "op-after-reload");
  assert.equal(preflight.generation, 4);
});

test("terminal operation survives auxiliary evidence refresh failure", async () => {
  const preflight = baseState({
    device: "cuda:0",
    commandId: "cmd-terminal",
    operationId: "op-terminal",
    status: "running",
    generation: 2,
  });

  const api = {
    async preflight() {
      throw new Error("not used");
    },
    async operation(id) {
      assert.equal(id, "op-terminal");
      return {
        operation_id: id,
        status: "completed",
        result: {foundation_passed: true, training_authorized: false},
      };
    },
  };

  const controller = new PreflightController({
    api,
    preflight,
    makeId: () => "not-used",
    refreshEvidence: async () => {
      throw new Error("readiness endpoint temporarily unavailable");
    },
    schedule: noSchedule,
  });

  await controller.poll();

  assert.equal(preflight.status, "completed");
  assert.equal(preflight.cachedResult.operation_id, "op-terminal");
  assert.match(preflight.evidenceRefreshError, /temporarily unavailable/);
});

test("stale old poll cannot overwrite a newer selected operation", async () => {
  let resolveOld;
  const oldPromise = new Promise(resolve => {
    resolveOld = resolve;
  });

  const api = {
    async preflight(body) {
      return {
        operation_id: "op-new",
        status: "queued",
        echoed: body.command_id,
      };
    },
    async operation(id) {
      if (id === "op-old") return oldPromise;
      throw new Error("unexpected operation");
    },
  };

  const preflight = baseState({
    device: "auto",
    commandId: "cmd-old",
    operationId: "op-old",
    status: "poll_error",
    generation: 1,
  });

  let nextId = 0;
  const controller = new PreflightController({
    api,
    preflight,
    makeId: () => `cmd-new-${++nextId}`,
    schedule: noSchedule,
  });

  const oldPoll = controller.poll({
    generation: 1,
    commandId: "cmd-old",
    operationId: "op-old",
  });

  await controller.start("cpu");
  assert.equal(preflight.operationId, "op-new");
  assert.equal(preflight.generation, 2);

  resolveOld({
    operation_id: "op-old",
    status: "completed",
    result: {foundation_passed: false},
  });
  const oldResult = await oldPoll;

  assert.equal(oldResult.stale, true);
  assert.equal(preflight.operationId, "op-new");
  assert.equal(preflight.status, "queued");
});
