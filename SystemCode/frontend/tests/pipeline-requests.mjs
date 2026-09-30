// App-4.3: opening #/case/<id>/pipeline performs exactly one GET /trace and triggers no inference (no /diagnose, no
// /rerun). The production build is served by a tiny Node http server (no dependency) whose /api answers from the
// fixture and logs every request; headless Chrome loads the page over the DevTools protocol (Node >= 22 WebSocket).
// With no Chrome/Chromium it reports SKIPPED rather than passing silently.
import { spawn } from "node:child_process";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "..", "dist");
const fixture = JSON.parse(readFileSync(join(here, "fixtures", "trace_known_fault.json"), "utf8"));
const signals = JSON.parse(readFileSync(join(here, "fixtures", "known_fault.json"), "utf8")).signals;  // App-4.4 previews
const chrome = [process.env.CHROME, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"].find((p) => p && existsSync(p));
if (!existsSync(join(dist, "index.html"))) { console.error("✗ no production build (run vite build first)"); process.exit(1); }
if (!chrome || typeof WebSocket === "undefined") { console.log("⚠ pipeline request check SKIPPED (no Chrome/Chromium or no WebSocket)"); process.exit(0); }

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".woff2": "font/woff2", ".json": "application/json" };
const api = [];
const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url.startsWith("/api/")) {
    api.push(`${req.method} ${url}`);
    if (req.method === "GET" && url === `/api/cases/${fixture.case_id}/trace`) {
      res.writeHead(200, { "content-type": "application/json" });
      return res.end(JSON.stringify(fixture));
    }
    if (req.method === "GET" && url === `/api/cases/${fixture.case_id}/signals`) {
      res.writeHead(200, { "content-type": "application/json" });
      return res.end(JSON.stringify({ ...signals, case_id: fixture.case_id }));
    }
    res.writeHead(404, { "content-type": "application/json" });
    return res.end(JSON.stringify({ detail: "not served by this test" }));
  }
  const file = join(dist, url === "/" ? "index.html" : url);
  if (!file.startsWith(dist) || !existsSync(file)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const httpPort = server.address().port;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9800 + Math.floor(Math.random() * 150);
const profile = join(tmpdir(), `ms-requests-${port}`);
const proc = spawn(chrome, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
let failed = 0;
try {
  let wsUrl;
  for (let i = 0; i < 100 && !wsUrl; i++) {
    try { wsUrl = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page")?.webSocketDebuggerUrl; }
    catch { await sleep(100); }
  }
  const ws = new WebSocket(wsUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const pending = new Map();
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result); pending.delete(d.id); } };
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, returnByValue: true })).result.value;
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `http://127.0.0.1:${httpPort}/#/case/${fixture.case_id}/pipeline` });
  let drawn = false;
  for (let i = 0; i < 100 && !drawn; i++) { await sleep(100); drawn = await evaluate("!!document.querySelector('svg.pl-canvas')"); }
  await sleep(1500);  // anything the page would still request
  const verdict = await evaluate("document.querySelector('[data-verdict]')?.textContent || ''");
  const onLoad = [...api];
  const traceGets = api.filter((r) => r === `GET /api/cases/${fixture.case_id}/trace`).length;
  const inference = api.filter((r) => /\/diagnose|\/rerun/.test(r));
  const check = (ok, msg) => { if (ok) console.log(`✓ ${msg}`); else { failed++; console.error(`✗ ${msg}`); } };
  check(drawn, "the pipeline page drew the diagram");
  check(traceGets === 1 && onLoad.length === 1, `page load: exactly one API request, GET /trace, and no /signals (saw: ${JSON.stringify(onLoad)})`);
  check(inference.length === 0, "no /diagnose or /rerun request");
  const want = fixture.blocks.find((b) => b.id === "decision").output.display_label.value;
  check(verdict === want, `opened in "This diagnosis" mode showing the verdict "${want}"`);
  // App-4.4: a preview is fetched only by the explicit load action, once; opening another preview or reopening does not refetch
  const click = (sel) => evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (e) e.click(); return !!e; })()`);
  const loadBtn = "(() => { const b = [...document.querySelectorAll('button')].find((x) => /Load signal preview/.test(x.textContent)); if (b) b.click(); return !!b; })()";
  await click("[data-block=preparation]"); await sleep(300);
  check(api.filter((r) => r.endsWith("/signals")).length === 0, "opening the step alone does not fetch /signals");
  check(await evaluate(loadBtn), "the drawer offers an explicit load button");
  for (let i = 0; i < 30 && !(await evaluate("!!document.querySelector('aside [data-preview=computed] path')")); i++) await sleep(100);
  const sigGets = () => api.filter((r) => r === `GET /api/cases/${fixture.case_id}/signals`).length;
  check(sigGets() === 1, "loading the preview issues exactly one GET /signals");
  await click("[data-block=order_spectrogram]"); await sleep(400);
  const specDrawn = await evaluate("!!document.querySelector('aside [data-preview=computed] path')");
  await click("[data-block=preparation]"); await sleep(400);
  check(specDrawn && sigGets() === 1, "opening another preview and reopening do not refetch (still one /signals)");
  check(api.every((r) => !/\/diagnose|\/rerun/.test(r)), "no /diagnose or /rerun at any point");
  ws.close();
} finally {
  proc.kill();
  server.close();
  await sleep(200);
  rmSync(profile, { recursive: true, force: true });
}
if (failed) process.exit(1);
console.log("pipeline request check passed");
