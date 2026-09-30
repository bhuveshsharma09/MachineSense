// App-2a.3: the small-screen pages written by the smoke test (tests/.ssr/mobile/*.html, the real markup with the built
// CSS) are loaded in headless Chrome emulating a 390 × 844 screen; the document must be no wider than the viewport
// (no horizontal page scroll). Uses the Chrome DevTools protocol over Node's built-in WebSocket (Node ≥ 22); with no
// Chrome/Chromium or no WebSocket it reports SKIPPED rather than passing silently.
import { spawn } from "node:child_process";
import { existsSync, readdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

const W = 390, H = 844;
const dir = join(dirname(fileURLToPath(import.meta.url)), ".ssr", "mobile");
const chrome = [process.env.CHROME, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"].find((p) => p && existsSync(p));
const pages = existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".html")) : [];
if (!pages.length) { console.error("✗ no small-screen pages found (run the smoke test first)"); process.exit(1); }
// App-4.2: the pipeline architecture page must be among them
if (!pages.includes("pipeline_390.html")) { console.error("✗ the 390 px pipeline page is missing (tests/smoke.jsx writes it)"); process.exit(1); }
if (!chrome || typeof WebSocket === "undefined") {
  console.log(`⚠ 390 px width check SKIPPED (${!chrome ? "no Chrome/Chromium found; set CHROME" : "Node without WebSocket"})`);
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9400 + Math.floor(Math.random() * 400);
const profile = join(tmpdir(), `ms-width-${port}`);
const proc = spawn(chrome, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
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
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: true });
  for (const f of pages) {
    await send("Page.navigate", { url: "file://" + join(dir, f) });
    await sleep(700);
    const r = await send("Runtime.evaluate", { expression: "JSON.stringify([document.documentElement.scrollWidth, window.innerWidth])", returnByValue: true });
    const [sw, vw] = JSON.parse(r.result.value);
    if (sw > W || vw !== W) { failed++; console.error(`✗ ${f}: document width ${sw} px > viewport ${vw} px`); }
    else console.log(`✓ ${f}: document width ${sw} px ≤ viewport ${vw} px`);
  }
  ws.close();
} finally {
  proc.kill();
  await sleep(200);
  rmSync(profile, { recursive: true, force: true });
}
if (failed) process.exit(1);
console.log("390 px width check passed");
