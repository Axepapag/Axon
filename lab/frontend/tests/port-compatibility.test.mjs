import test from "node:test";
import assert from "node:assert/strict";
import {portCompatibility} from "../ui-utils.js";

function port(overrides = {}) {
  return {
    port_id: "p",
    direction: "output",
    meaning: "frozen-d16-cell-surface",
    dtype: "float32",
    shape: [32, 16],
    authority: "frozen-substrate",
    surface: "substrate-exact",
    constraints: {codebook: "native95+empty95", tolerance: 0.0},
    ...overrides,
  };
}

test("matching substrate-exact ports are compatible", () => {
  const source = port();
  const destination = port({direction: "input"});
  assert.deepEqual(portCompatibility(source, destination), {compatible: true, reasons: []});
});

test("substrate-exact cannot connect directly to free", () => {
  const source = port();
  const destination = port({
    direction: "input",
    surface: "free",
    meaning: "recurrent-reading",
    shape: [512],
    authority: "core-internal",
    constraints: {},
  });
  const result = portCompatibility(source, destination);
  assert.equal(result.compatible, false);
  assert.ok(result.reasons.some(reason => reason.includes("surface mismatch")));
  assert.ok(result.reasons.some(reason => reason.includes("explicit adapter")));
});

test("substrate-exact codebooks must match", () => {
  const source = port();
  const destination = port({
    direction: "input",
    constraints: {codebook: "other-codebook", tolerance: 0.0},
  });
  const result = portCompatibility(source, destination);
  assert.equal(result.compatible, false);
  assert.ok(result.reasons.some(reason => reason.includes("codebook mismatch")));
});

test("free hidden tensor ports still require exact trusted semantics", () => {
  const source = port({
    meaning: "recurrent-reading",
    shape: [512],
    authority: "core-internal",
    surface: "free",
    constraints: {},
  });
  const destination = {
    ...source,
    direction: "input",
  };
  assert.equal(portCompatibility(source, destination).compatible, true);

  destination.shape = [256];
  const mismatch = portCompatibility(source, destination);
  assert.equal(mismatch.compatible, false);
  assert.ok(mismatch.reasons.some(reason => reason.includes("shape mismatch")));
});
