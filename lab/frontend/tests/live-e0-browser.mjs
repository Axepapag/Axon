import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";

const chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const port = 9337;
const profile = path.join(os.tmpdir(), `axon-lab-e0-${Date.now()}`);
fs.mkdirSync(profile, {recursive: true});

const child = spawn(chrome, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  "https://axon.gliksbot.com/",
], {stdio: "ignore"});

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function waitForPage() {
  for (let i = 0; i < 80; i += 1) {
    try {
      const pages = await fetch(`http://127.0.0.1:${port}/json`).then(r => r.json());
      const page = pages.find(item => item.type === "page" && item.url.includes("axon.gliksbot.com"));
      if (page) return page;
    } catch {}
    await sleep(100);
  }
  throw new Error("Chrome DevTools page did not appear");
}

const page = await waitForPage();
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener("open", resolve, {once: true});
  ws.addEventListener("error", reject, {once: true});
});

let seq = 0;
const pending = new Map();
ws.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  if (!message.id) return;
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
  else waiter.resolve(message.result);
});

function cdp(method, params = {}) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
}

async function evaluate(expression) {
  const result = await cdp("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}

async function setSelect(selector, value) {
  return evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return false;
    el.value = ${JSON.stringify(value)};
    el.dispatchEvent(new Event("change", {bubbles:true}));
    return true;
  })()`);
}

try {
  await cdp("Runtime.enable");
  await cdp("Page.enable");
  await sleep(5000);

  assert.equal(await evaluate('document.querySelector("#connection-pill")?.textContent'), "Backend connected");

  await evaluate('document.querySelector("[data-view=architect]").click()');
  await sleep(250);

  const availability = await evaluate(`(() => Object.fromEntries(
    ["axon.substrate_input","axon.core_reasoning_gru","axon.response_state","gru"].map(id => {
      const el = document.querySelector('[data-component-id="'+id+'"]');
      return [id, el ? {disabled:el.disabled, text:el.innerText} : null];
    })
  ))()`);
  for (const id of ["axon.substrate_input","axon.core_reasoning_gru","axon.response_state"]) {
    assert.ok(availability[id], `${id} card missing`);
    assert.equal(availability[id].disabled, false, `${id} should be enabled`);
  }
  assert.equal(availability.gru?.disabled, true, "generic placeholder GRU must remain disabled");

  for (const id of ["axon.substrate_input","axon.core_reasoning_gru","axon.response_state"]) {
    await evaluate(`document.querySelector('[data-component-id="${id}"]').click()`);
    await sleep(75);
  }

  const nodes = await evaluate(`(() => [...document.querySelectorAll(".node-card")].map(card => ({
    id: card.querySelector('[data-action="select-node"]')?.dataset.nodeId,
    text: card.innerText
  })))()`);
  assert.equal(nodes.length, 3);

  const substrate = nodes.find(node => node.text.includes("Substrate Input"));
  const reasoning = nodes.find(node => node.text.includes("Reasoning/Memory"));
  const response = nodes.find(node => node.text.includes("English Response"));
  assert.ok(substrate && reasoning && response);

  await evaluate(`document.querySelector('[data-action="select-node"][data-node-id="${substrate.id}"]').click()`);
  await sleep(75);
  assert.ok(await evaluate('Boolean(document.querySelector("[data-config-path=width]"))'));
  const substratePortText = await evaluate('[...document.querySelectorAll(".port-row")].map(x=>x.innerText).join("\\n")');
  assert.match(substratePortText, /char_ids_in/);
  assert.match(substratePortText, /cells_out/);
  assert.match(substratePortText, /substrate-exact/);

  await evaluate(`document.querySelector('[data-action="select-node"][data-node-id="${response.id}"]').click()`);
  await sleep(75);
  assert.ok(await evaluate('Boolean(document.querySelector("[data-config-path=max_chunk]"))'));
  const responsePortText = await evaluate('[...document.querySelectorAll(".port-row")].map(x=>x.innerText).join("\\n")');
  assert.match(responsePortText, /char_ids_out/);
  assert.match(responsePortText, /control_out/);
  assert.match(responsePortText, /substrate-exact/);
  assert.match(responsePortText, /free/);

  // Impossible exact-substrate -> free-hidden connection is filtered before POST validation.
  await setSelect('[data-edge-field="sourceNode"]', substrate.id);
  await sleep(50);
  await setSelect('[data-edge-field="sourcePort"]', "cells_out");
  await sleep(50);
  await setSelect('[data-edge-field="destinationNode"]', response.id);
  await sleep(50);
  assert.equal(await evaluate('[...document.querySelectorAll(\'[data-edge-field="destinationPort"] option\')].some(o=>o.value==="reading")'), false);

  await setSelect('[data-edge-field="destinationNode"]', "");
  await sleep(50);

  await setSelect('[data-edge-field="sourceNode"]', substrate.id);
  await sleep(50);
  await setSelect('[data-edge-field="sourcePort"]', "cells_out");
  await sleep(50);
  await setSelect('[data-edge-field="destinationNode"]', reasoning.id);
  await sleep(50);
  await setSelect('[data-edge-field="destinationPort"]', "observation");
  await sleep(50);
  assert.equal(await evaluate('document.querySelector("[data-action=add-edge]").disabled'), false);
  await evaluate('document.querySelector("[data-action=add-edge]").click()');
  await sleep(75);

  await setSelect('[data-edge-field="sourceNode"]', reasoning.id);
  await sleep(50);
  await setSelect('[data-edge-field="sourcePort"]', "state_reading");
  await sleep(50);
  await setSelect('[data-edge-field="destinationNode"]', response.id);
  await sleep(50);
  await setSelect('[data-edge-field="destinationPort"]', "reading");
  await sleep(50);
  assert.equal(await evaluate('document.querySelector("[data-action=add-edge]").disabled'), false);
  await evaluate('document.querySelector("[data-action=add-edge]").click()');
  await sleep(75);

  assert.equal(await evaluate('document.querySelectorAll(".edge-row").length'), 2);
  await evaluate('document.querySelector("[data-action=validate-architecture]").click()');
  await sleep(800);
  const validationText = await evaluate('document.querySelector(".validation")?.innerText || ""');
  assert.match(validationText, /Graph valid and execution-eligible/);

  console.log("LIVE_E0_UI_OK", JSON.stringify({
    components: Object.keys(availability).filter(id => availability[id] && !availability[id].disabled),
    nodes: nodes.length,
    edges: 2,
    validation: validationText.split("\n")[0],
  }));
} finally {
  try { ws.close(); } catch {}
  child.kill();
  await sleep(350);
  try {
    fs.rmSync(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 100});
  } catch {
    // Chrome Crashpad can briefly hold a profile file after browser exit.
    // Cleanup failure must not turn successful browser assertions into a product failure.
  }
}
