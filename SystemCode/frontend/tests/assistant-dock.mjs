// App-9 addendum A: the one assistant panel in a real browser. The production build is served by a tiny Node server whose
// /api answers from the recorded fixtures (two cases) and keeps the assistant's questions and conversations in memory, like
// the SQLite store; /ask is slow on purpose so an in-flight request can be minimized. Headless Chrome drives the page over
// the DevTools protocol. "Same instance" = the very same <input> element object survives (window.__inp === the input).
// With no Chrome/Chromium it reports SKIPPED rather than passing silently.
import { spawn } from "node:child_process";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "..", "dist");
const fx = (n) => JSON.parse(readFileSync(join(here, "fixtures", n), "utf8"));
const A = fx("known_fault.json"), B = fx("unknown.json"), trace = fx("trace_known_fault.json");
const CASES = { [A.record.case_id]: A, [B.record.case_id]: B };
const chrome = [process.env.CHROME, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"].find((p) => p && existsSync(p));
if (!existsSync(join(dist, "index.html"))) { console.error("✗ no production build (run vite build first)"); process.exit(1); }
if (!chrome || typeof WebSocket === "undefined") { console.log("⚠ assistant dock check SKIPPED (no Chrome/Chromium or no WebSocket)"); process.exit(0); }

// the in-memory store: questions (id, case, conversation) and each case's current conversation
const store = { questions: [], conv: {}, nextConv: 1, posts: [] };
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".ttf": "font/ttf", ".json": "application/json" };
const json = (res, body, code = 200) => { res.writeHead(code, { "content-type": "application/json" }); res.end(JSON.stringify(body)); };
const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (!url.startsWith("/api/")) {
    const file = join(dist, url === "/" ? "index.html" : url);
    if (!file.startsWith(dist) || !existsSync(file)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" });
    return res.end(readFileSync(file));
  }
  if (req.method === "POST") store.posts.push(url);
  const m = /^\/api\/cases\/([^/]+)(?:\/(\w+))?$/.exec(url);
  if (url === "/api/cases") return json(res, { cases: [] });
  if (!m) return json(res, { status: "ok", limitations: [] });
  const [, id, sub] = m, f = CASES[id];
  if (!f) return json(res, { detail: "no case" }, 404);
  const conv = store.conv[id] || 0;
  if (!sub) return json(res, f.record);
  if (sub === "trace") return json(res, { ...trace, case_id: id });
  if (sub === "explanation") return json(res, { ...f.explanation, case_id: id });
  if (["report", "activity", "channels", "signals"].includes(sub)) return json(res, { ...f[sub], case_id: id });
  if (sub === "questions") {
    const qs = store.questions.filter((q) => q.case_id === id);
    return json(res, { case_id: id, questions: qs, current_conversation_id: conv, conversation: qs.filter((q) => q.conversation_id === conv).map((q) => q.id) });
  }
  if (sub === "conversations" && req.method === "POST") { store.conv[id] = store.nextConv++; return json(res, { case_id: id, conversation_id: store.conv[id] }); }
  if (sub === "ask" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => { body += c; });
    return req.on("end", () => setTimeout(() => {
      const q = JSON.parse(body).question;
      const guidance = q === "What should I inspect next?";
      const sensor = q === "Which sensor changed most?";
      const ans = { id: store.questions.length + 1, case_id: id, conversation_id: conv, mode: "conversation", question: q,
        answer: `Answer to: ${q}`,
        claims: sensor ? [{ text: "The sensor changed.", refs: ["evidence.largest_change.channel"] }]
          : guidance ? [{ text: "Inspect the bearing.", refs: ["guidance.items.0"] }] : [],
        citations: guidance ? [{ id: "bearing-defect-frequencies", kind: "inspection" }] : [],
        actions: guidance ? [{ tool: "lookup_guidance", summary: "inspection steps" }]
          : sensor ? [{ tool: "get_evidence", summary: "changed channels" }] : [],
        review: { status: "passed", failed: [] }, source: "agent" };
      store.questions.push(ans);
      json(res, ans);
    }, 1500));
  }
  return json(res, { detail: "not served by this test" }, 404);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}/`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9300 + Math.floor(Math.random() * 400);
const profile = join(tmpdir(), `ms-dock-${port}`);
const proc = spawn(chrome, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
let failed = 0;
const check = (ok, name, detail = "") => { if (ok) console.log(`✓ dock: ${name}`); else { failed++; console.error(`✗ dock: ${name} ${detail}`); } };
try {
  let wsUrl;
  for (let i = 0; i < 100 && !wsUrl; i++) {
    try { wsUrl = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page")?.webSocketDebuggerUrl; }
    catch { await sleep(100); }
  }
  const ws = new WebSocket(wsUrl);
  await new Promise((r) => (ws.onopen = r));
  let n = 0;
  const pending = new Map();
  ws.onmessage = (msg) => { const d = JSON.parse(msg.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result); pending.delete(d.id); } };
  const send = (method, params = {}) => new Promise((r) => { const i = ++n; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => (await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true })).result?.value;
  const until = async (expr, ms = 8000) => { for (let t = 0; t < ms; t += 100) { if (await ev(expr)) return true; await sleep(100); } return false; };
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  const go = async (hash) => { await ev(`(location.hash = ${JSON.stringify(hash)}, true)`); await sleep(400); };
  const INPUT = `document.querySelector('#assistant-panel input[aria-label="Ask about this recording"]')`;
  const visible = `(() => { const p = document.getElementById('assistant-panel'); return !!p && !p.hidden && p.getBoundingClientRect().width > 0; })()`;
  const typeDraft = (t) => ev(`(() => { const i = ${INPUT}; const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    set.call(i, ${JSON.stringify(t)}); i.dispatchEvent(new Event('input', { bubbles: true })); window.__inp = i; return true; })()`);
  const same = `(${INPUT} === window.__inp)`;
  const click = (sel) => ev(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return false; e.click(); return true; })()`);
  const caseA = A.record.case_id, caseB = B.record.case_id;

  // 1. direct /assistant link → dashboard with the panel open; the URL becomes /dashboard
  await send("Page.navigate", { url: `${base}#/case/${caseA}/assistant` });
  check(await until(`${visible} && !!${INPUT}`), "the /assistant deep link opens the panel on the dashboard");
  check(await ev(`location.hash === '#/case/${caseA}/dashboard' && !!document.querySelector('.rd-root.has-assistant')`),
    "the URL is canonicalised to /dashboard and the dashboard makes room");
  check(await until(`document.activeElement === ${INPUT}`), "focus goes to the question box on open");

  // The left edge resizes the panel and the dashboard together; keyboard adjustment works too.
  const grip = await ev(`(() => { const r = document.querySelector('.as-resize-handle').getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
  await send("Input.dispatchMouseEvent", { type: "mousePressed", x: grip.x, y: grip.y, button: "left", buttons: 1, clickCount: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: grip.x - 160, y: grip.y, button: "left", buttons: 1 });
  await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: grip.x - 160, y: grip.y, button: "left", buttons: 0, clickCount: 1 });
  await sleep(100);
  const afterDrag = await ev(`(() => { const p = document.getElementById('assistant-panel').getBoundingClientRect();
    const d = document.querySelector('.rd-root').getBoundingClientRect(); return { width: p.width, left: p.left, dashboardRight: d.right }; })()`);
  check(Math.abs(afterDrag.width - 580) < 3 && Math.abs(afterDrag.dashboardRight - afterDrag.left) < 3,
    "dragging the edge widens the panel and reflows the dashboard", JSON.stringify(afterDrag));
  await ev(`(() => { const h = document.querySelector('.as-resize-handle'); h.focus();
    h.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })); return true; })()`);
  await sleep(100);
  const afterKey = await ev(`(() => ({ width: document.getElementById('assistant-panel').getBoundingClientRect().width,
    value: document.querySelector('.as-resize-handle').getAttribute('aria-valuenow') }))()`);
  check(Math.abs(afterKey.width - 600) < 3 && afterKey.value === '600',
    "the resize handle also supports keyboard adjustment", JSON.stringify(afterKey));

  // A cited inspection answer opens the Mechanic checks even if Developer is showing; a sensor answer opens the
  // Developer sensor card. In both cases the destination receives a visible highlight without closing the panel.
  await ev(`(() => { const b = [...document.querySelectorAll('[role=radio]')].find((x) => x.textContent.trim() === 'Developer'); b.click(); return true; })()`);
  await typeDraft("What should I inspect next?");
  await ev("(document.querySelector('#assistant-panel form').requestSubmit(), true)");
  check(await until(`document.querySelector('#assistant-panel').textContent.includes('Answer to: What should I inspect next?')`),
    "an inspection answer is shown");
  await ev(`(() => { [...document.querySelectorAll('#assistant-panel a')].find((a) => a.textContent.includes('See suggested checks'))?.click(); return true; })()`);
  check(await until(`!!document.querySelector('#rd-suggested-checks.rd-focus-highlight')`) && await ev(`
    document.querySelector('[role=radio][aria-checked=true]')?.textContent.trim() === 'Mechanic' && ${visible}`),
    "inspection link switches to Mechanic and highlights Suggested checks");
  await typeDraft("Which sensor changed most?");
  await ev("(document.querySelector('#assistant-panel form').requestSubmit(), true)");
  check(await until(`document.querySelector('#assistant-panel').textContent.includes('Answer to: Which sensor changed most?')`),
    "a sensor answer is shown");
  await ev(`(() => { [...document.querySelectorAll('#assistant-panel a')].find((a) => a.textContent.includes('See the sensor comparison'))?.click(); return true; })()`);
  check(await until(`!!document.querySelector('#rd-sensor-channels.rd-focus-highlight')`) && await ev(`
    document.querySelector('[role=radio][aria-checked=true]')?.textContent.trim() === 'Developer' && ${visible}`),
    "sensor link switches to Developer and highlights the sensor card");
  await ev(`(() => { const b = [...document.querySelectorAll('[role=radio]')].find((x) => x.textContent.trim() === 'Mechanic'); b.click(); return true; })()`);

  // 2. open → minimize → restore with the draft and the same instance
  await typeDraft("my unsent draft");
  await click('button[aria-label="Minimize the assistant"]');
  await sleep(200);
  check(await ev(`!${visible} && !!document.querySelector('.as-restore') && document.activeElement === document.querySelector('.as-restore')`),
    "minimize hides the panel, shows the restore button and focuses it");
  await click(".as-restore");
  await sleep(200);
  check(await ev(`${visible} && ${same} && ${INPUT}.value === 'my unsent draft'`), "restore shows the same panel with the draft");

  // 3. pin → Developer → Mechanic → Pipeline → dashboard: one continuous instance
  await click('button[aria-label="Pin the assistant"]');
  check(await ev(`document.querySelector('.as-pin').getAttribute('aria-pressed') === 'true'`), "Pin sets aria-pressed");
  await ev(`(() => { const b = [...document.querySelectorAll('[role=radio]')].find((x) => x.textContent.trim() === 'Developer'); b?.click(); return !!b; })()`);
  await sleep(300);
  check(await ev(`${visible} && ${same} && ${INPUT}.value === 'my unsent draft'`), "pinned: Mechanic → Developer keeps the instance and draft");
  await go(`#/case/${caseA}/pipeline`);
  check(await until(`!!document.querySelector('.pl-root')`) && await ev(`${visible} && ${same} && ${INPUT}.value === 'my unsent draft'
    && !!document.querySelector('.pl-root.has-assistant') && Math.abs(document.getElementById('assistant-panel').getBoundingClientRect().width - 600) < 3
    && Math.abs(document.querySelector('.pl-root').getBoundingClientRect().right - document.getElementById('assistant-panel').getBoundingClientRect().left) < 3`),
    "pinned: the pipeline keeps the same instance, draft and resized width");
  await ev("(history.back(), true)");
  await sleep(500);
  check(await ev(`location.hash === '#/case/${caseA}/dashboard' && ${visible} && ${same}`), "browser Back to the dashboard keeps the pinned panel");
  await ev("(history.forward(), true)");
  await sleep(500);
  check(await ev(`location.hash === '#/case/${caseA}/pipeline' && ${visible} && ${same}`), "browser Forward to the pipeline keeps it too");
  await go(`#/case/${caseA}/dashboard`);

  // 4. an in-flight request survives minimize
  await ev(`(() => { const f = document.querySelector('#assistant-panel form'); f.requestSubmit(); return true; })()`);
  await sleep(100);
  await click('button[aria-label="Minimize the assistant"]');
  await sleep(2000);
  await click(".as-restore");
  check(await until(`document.querySelector('#assistant-panel').textContent.includes('Answer to: my unsent draft')`),
    "a question sent before minimizing is answered and shown after restore");

  // 5. minimized + pinned across navigation
  await click('button[aria-label="Minimize the assistant"]');
  await go(`#/case/${caseA}/pipeline`);
  check(await ev(`!${visible} && !!document.querySelector('.as-restore') && !!document.getElementById('assistant-panel')`),
    "minimized + pinned stays minimized on the pipeline with its restore button");
  await go(`#/case/${caseA}/dashboard`);
  await click(".as-restore");
  await sleep(200);

  // 6. Escape minimizes without navigating
  await ev(`(${INPUT}.focus(), ${INPUT}.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })), true)`);
  await sleep(200);
  check(await ev(`!${visible} && location.hash === '#/case/${caseA}/dashboard'`), "Escape minimizes and stays on the case");
  await click(".as-restore");
  await sleep(200);

  // 7. unpinned → the pipeline closes the panel; the pipeline's own control opens it
  await click('button[aria-label="Unpin the assistant"]');
  await go(`#/case/${caseA}/pipeline`);
  check(await until(`!!document.querySelector('.pl-root')`) && await ev(`!document.getElementById('assistant-panel')`), "unpinned: going to the pipeline closes the panel");
  check(await click(".pl-ask") && await until(`${visible} && !!${INPUT}`), "the pipeline's Ask the assistant control opens it");
  check(await ev(`${INPUT}.value === ''`), "a reopened panel is a new session (no old draft)");

  // 8. Home ends the session; saved history stays; the next open starts a new conversation (this one has a turn)
  await typeDraft("a question in this session");
  await ev(`(document.querySelector('#assistant-panel form').requestSubmit(), true)`);
  check(await until(`document.querySelector('#assistant-panel').textContent.includes('Answer to: a question in this session')`),
    "a question in the new session is answered");
  const before = store.questions.length, convPosts = store.posts.filter((p) => p.endsWith("/conversations")).length;
  await go("#/");
  check(await until(`!document.getElementById('assistant-panel') && !document.querySelector('.as-restore')`), "Home closes the panel (no restore button)");
  await go(`#/case/${caseA}/assistant`);
  check(await until(`${visible} && !!${INPUT}`), "the panel opens again for the case");
  await sleep(500);
  check(store.questions.length === before && store.posts.filter((p) => p.endsWith("/conversations")).length > convPosts
    && await until(`document.querySelector('#assistant-panel').textContent.includes('Earlier questions about this recording')`),
    "after Home a new conversation starts; the saved questions stay and are listed as earlier");

  // 9. another case never inherits the panel or the draft
  await typeDraft("draft for case A");
  await go(`#/case/${caseB}/dashboard`);
  check(await until(`!!document.querySelector('.rd-root')`) && await ev(`!document.getElementById('assistant-panel')`), "opening another case closes the previous panel");
  await go(`#/case/${caseB}/assistant`);
  check(await until(`${visible} && !!${INPUT}`) && await ev(`${INPUT}.value === '' && !document.querySelector('#assistant-panel').textContent.includes('my unsent draft')`),
    "the other case's panel starts empty (no draft or conversation carried over)");

  // 10. Close ends the session; history says it stays
  check(await ev(`document.querySelector('#assistant-panel').textContent.includes('Saved questions about this recording stay available')`), "the panel says saved history stays available");
  await click('button[aria-label="Close the assistant and end this session"]');
  await sleep(200);
  check(await ev(`!document.getElementById('assistant-panel') && location.hash === '#/case/${caseB}/dashboard'`), "Close removes the panel and stays on the case");

  // 11. 390 px: a full-width overlay and a visible restore button
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await go(`#/case/${caseB}/assistant`);
  check(await until(`${visible}`) && await ev(`Math.round(document.getElementById('assistant-panel').getBoundingClientRect().width) === innerWidth`),
    "390 px: the panel is a full-width overlay");
  check(await ev(`getComputedStyle(document.querySelector('.as-resize-handle')).display === 'none'`),
    "390 px: the resize handle is hidden on the full-width overlay");
  await click('button[aria-label="Minimize the assistant"]');
  await sleep(200);
  check(await ev(`(() => { const r = document.querySelector('.as-restore').getBoundingClientRect(); return r.width > 0 && r.right <= innerWidth && r.bottom <= innerHeight; })()`),
    "390 px: minimized shows the restore button inside the screen");
  ws.close();
} finally {
  proc.kill();
  server.close();
  await sleep(200);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
}
if (failed) { console.error(`${failed} assistant dock check(s) failed`); process.exit(1); }
console.log("assistant dock check passed");
