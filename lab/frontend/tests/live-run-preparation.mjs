import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";

const chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const port = 9341;
const profile = path.join(os.tmpdir(), `axon-lab-runprep-${Date.now()}`);
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
  for (let i = 0; i < 100; i += 1) {
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
  ws.addEventListener("open", resolve, {once:true});
  ws.addEventListener("error", reject, {once:true});
});

let seq = 0;
const pending = new Map();
ws.addEventListener("message", event => {
  const msg = JSON.parse(event.data);
  if (!msg.id) return;
  const waiter = pending.get(msg.id);
  if (!waiter) return;
  pending.delete(msg.id);
  if (msg.error) waiter.reject(new Error(JSON.stringify(msg.error)));
  else waiter.resolve(msg.result);
});

function cdp(method, params = {}) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
  });
}

async function evaluate(expression) {
  const result = await cdp("Runtime.evaluate", {expression, returnByValue:true, awaitPromise:true});
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}

async function waitUntil(expression, timeoutMs = 12000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return true;
    await sleep(150);
  }
  return false;
}

async function setSelect(selector, value = null) {
  return evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return {ok:false, reason:"missing"};
    const desired = ${JSON.stringify(value)};
    const option = desired == null ? [...el.options].find(o => o.value && !o.disabled) : [...el.options].find(o => o.value === desired && !o.disabled);
    if (!option) return {ok:false, reason:"no-option", options:[...el.options].map(o=>({v:o.value,d:o.disabled,t:o.textContent}))};
    el.value = option.value;
    el.dispatchEvent(new Event("change",{bubbles:true}));
    return {ok:true,value:el.value,text:option.textContent};
  })()`);
}

try {
  await cdp("Runtime.enable");
  await cdp("Page.enable");
  await sleep(5000);

  assert.equal(await evaluate('document.querySelector("#connection-pill")?.textContent'), "Backend connected");

  await evaluate('document.querySelector("[data-view=train]").click()');
  await sleep(250);

  const trainText = await evaluate('document.querySelector("#main-view").innerText');
  assert.match(trainText, /Preparation available/);
  assert.match(trainText, /Start blocked/);
  assert.match(trainText, /Prepare run/);

  for (const [selector, value] of [
    ['select[name="dataset"]', null],
    ['select[name="curriculum"]', null],
    ['select[name="device"]', "cpu"],
    ['select[name="provider"]', "local"],
    ['select[name="architecture"]', null],
  ]) {
    assert.equal(
      await waitUntil(`document.querySelector(${JSON.stringify(selector)})?.options.length > 1`, 7000),
      true,
      `catalog options did not load for ${selector}`
    );
    const result = await setSelect(selector, value);
    assert.equal(result.ok, true, `${selector}: ${JSON.stringify(result)}`);
  }

  await evaluate(`document.querySelector('input[name="seed"]').value="4242"; document.querySelector('input[name="epochs"]').value="1"; document.querySelector('input[name="learning_rate"]').value="0.001";`);
  const prepareDisabled = await evaluate('document.querySelector("#create-run-form button[type=submit]").disabled');
  assert.equal(prepareDisabled, false);

  let runId = process.env.AXON_EXISTING_RUN_ID || null;
  if (!runId) {
    // PREPARATION ONLY. This does not click Start or any run lifecycle command.
    await evaluate('document.querySelector("#create-run-form button[type=submit]").click()');

    assert.equal(await waitUntil('document.querySelector(".operation-card .tag")?.textContent === "completed"', 12000), true);
    const operationText = await evaluate('document.querySelector(".operation-card").innerText');
    const match = operationText.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/ig);
    assert.ok(match?.length, operationText);
    runId = match.at(-1);
  }

  await evaluate('document.querySelector("[data-view=runs]").click()');
  assert.equal(await waitUntil(`[...document.querySelectorAll('[data-action="select-run"]')].some(x => x.dataset.runId === ${JSON.stringify(runId)})`, 5000), true);
  await evaluate(`[...document.querySelectorAll('[data-action="select-run"]')].find(x => x.dataset.runId === ${JSON.stringify(runId)}).click()`);
  assert.equal(await waitUntil(`document.querySelector(".runs-layout > section.panel")?.innerText.includes(${JSON.stringify(runId)})`, 7000), true);

  const runPanel = await evaluate('document.querySelector(".runs-layout > section.panel").innerText');
  assert.match(runPanel, /created/i);
  assert.match(runPanel, /Execution blocked/i);
  assert.match(runPanel, /Pause after current episode/);
  assert.match(runPanel, /Stop after current episode/);

  const startDisabled = await evaluate(`document.querySelector('[data-action="run-command"][data-command="start"]').disabled`);
  assert.equal(startDisabled, true);

  const stopDisabled = await evaluate(`document.querySelector('[data-action="run-command"][data-command="stop"]').disabled`);
  assert.equal(stopDisabled, false);

  console.log("PUBLIC_PREPARATION_OK", JSON.stringify({
    run_id: runId,
    preparation: "completed",
    lifecycle: "created",
    start_disabled: startDisabled,
    training_started: false,
  }));
} finally {
  try { ws.close(); } catch {}
  child.kill();
  await sleep(350);
  try { fs.rmSync(profile, {recursive:true, force:true, maxRetries:5, retryDelay:100}); } catch {}
}
