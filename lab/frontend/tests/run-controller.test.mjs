import test from "node:test";
import assert from "node:assert/strict";
import {RunMutationController, defaultRunMutationState} from "../run-controller.js";

function noSchedule() { return 1; }

test("run preparation accepted-but-ack-lost reuses the same command_id", async () => {
  const server = new Map();
  const seen = [];
  let calls = 0;
  const api = {
    async createRun(body) {
      calls += 1;
      seen.push(body.command_id);
      if (!server.has(body.command_id)) {
        server.set(body.command_id, {operation_id: "op-create", run_id: "run-1", status: "completed"});
      }
      if (calls === 1) throw new Error("ack lost after accept");
      return server.get(body.command_id);
    },
    async operation() { throw new Error("not used"); },
  };
  const state = defaultRunMutationState();
  const controller = new RunMutationController({
    api, state, makeId: () => "cmd-create", schedule: noSchedule,
    refreshRuns: async () => {},
  });
  const payload = {architecture_id: "a", dataset_id: "d", curriculum_id: "c"};

  await controller.prepare(payload);
  assert.equal(state.creation.status, "uncertain");
  assert.equal(state.creation.commandId, "cmd-create");

  await controller.resume();
  assert.deepEqual(seen, ["cmd-create", "cmd-create"]);
  assert.equal(state.creation.runId, "run-1");
  assert.equal(state.creation.status, "completed");
});

test("run preparation never-reached-server recovery keeps command identity", async () => {
  let calls = 0;
  const seen = [];
  const api = {
    async createRun(body) {
      calls += 1;
      seen.push(body.command_id);
      if (calls === 1) throw new Error("network failed before delivery");
      return {operation_id: "op-created-retry", run_id: "run-retry", status: "completed"};
    },
    async operation() { throw new Error("not used"); },
  };
  const state = defaultRunMutationState();
  const controller = new RunMutationController({
    api, state, makeId: () => "cmd-never", schedule: noSchedule,
    refreshRuns: async () => {},
  });
  const payload = {architecture_id: "a"};

  await controller.prepare(payload);
  await controller.resume();

  assert.deepEqual(seen, ["cmd-never", "cmd-never"]);
  assert.equal(state.creation.runId, "run-retry");
});

test("lifecycle command accepted-but-ack-lost recovers with same command_id", async () => {
  let calls = 0;
  const seen = [];
  const api = {
    async commandRun(runId, body) {
      calls += 1;
      seen.push([runId, body.command, body.command_id]);
      if (calls === 1) throw new Error("connection dropped");
      return {operation_id: "op-pause", run_id: runId, command: body.command, status: "queued"};
    },
    async operation(id) {
      return {operation_id: id, run_id: "run-1", command: "pause", status: "completed"};
    },
  };
  const state = defaultRunMutationState();
  const controller = new RunMutationController({
    api, state, makeId: () => "cmd-pause", schedule: noSchedule,
    refreshRuns: async () => {},
  });

  await controller.command("run-1", "pause");
  assert.equal(state.command.status, "uncertain");
  await controller.resume();

  assert.deepEqual(seen, [
    ["run-1", "pause", "cmd-pause"],
    ["run-1", "pause", "cmd-pause"],
  ]);
  assert.equal(state.command.operationId, "op-pause");
});

test("definitive execution_blocked response is failed, not uncertain", async () => {
  const error = new Error("Execution acceptance and backup gates have not cleared.");
  error.status = 409;
  error.code = "execution_blocked";
  error.body = {details: {authorized: false, reasons: ["backup restore pending"]}};

  const api = {
    async commandRun() { throw error; },
    async operation() { throw new Error("not used"); },
  };
  const state = defaultRunMutationState();
  const controller = new RunMutationController({
    api, state, makeId: () => "cmd-start", schedule: noSchedule,
    refreshRuns: async () => {},
  });

  const result = await controller.command("run-1", "start");

  assert.equal(result.uncertain, false);
  assert.equal(state.command.status, "failed");
  assert.equal(state.command.operation.error.code, "execution_blocked");
  assert.equal(state.command.operation.details.authorized, false);
});

test("terminal operation survives run refresh failure", async () => {
  const api = {
    async createRun() {
      return {operation_id: "op-create", run_id: "run-1", status: "queued"};
    },
    async operation() {
      return {operation_id: "op-create", run_id: "run-1", status: "completed"};
    },
  };
  const state = defaultRunMutationState();
  const controller = new RunMutationController({
    api, state, makeId: () => "cmd-create", schedule: noSchedule,
    refreshRuns: async () => { throw new Error("catalog temporarily unavailable"); },
  });

  await controller.prepare({architecture_id: "a"});
  await controller.poll("creation");

  assert.equal(state.creation.status, "completed");
  assert.equal(state.creation.operation.operation_id, "op-create");
  assert.match(state.creation.refreshError, /temporarily unavailable/);
});

test("late old operation poll cannot overwrite newer lifecycle command", async () => {
  let resolveOld;
  const old = new Promise(resolve => { resolveOld = resolve; });
  let ids = 0;
  const api = {
    async commandRun(runId, body) {
      return {operation_id: body.command === "pause" ? "op-new" : "op-other", run_id: runId, command: body.command, status: "queued"};
    },
    async operation(id) {
      if (id === "op-old") return old;
      throw new Error("unexpected operation");
    },
  };
  const state = defaultRunMutationState();
  Object.assign(state.command, {
    generation: 1,
    commandId: "cmd-old",
    runId: "run-1",
    command: "checkpoint",
    operationId: "op-old",
    status: "poll_error",
  });
  const controller = new RunMutationController({
    api, state, makeId: () => `cmd-new-${++ids}`, schedule: noSchedule,
    refreshRuns: async () => {},
  });

  const oldPoll = controller.poll("command", {
    generation: 1, commandId: "cmd-old", operationId: "op-old",
  });
  await controller.command("run-1", "pause");

  resolveOld({operation_id: "op-old", run_id: "run-1", command: "checkpoint", status: "completed"});
  const result = await oldPoll;

  assert.equal(result.stale, true);
  assert.equal(state.command.operationId, "op-new");
  assert.equal(state.command.command, "pause");
  assert.equal(state.command.status, "queued");
});
