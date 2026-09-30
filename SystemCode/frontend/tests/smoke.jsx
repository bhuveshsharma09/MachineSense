// Frontend smoke test: the production build passes (npm run build) and the main screens render, server-side, from
// recorded real API responses (tests/fixtures, made by tests/make_fixtures.py). No browser and no new dependency.
import { renderToString } from "react-dom/server";
import Workspace from "../src/components/Workspace.jsx";
import { qualityFlags, unusualSensors } from "../src/lib/model.js";
import common from "./fixtures/common.json";
import knownFault from "./fixtures/known_fault.json";
import unknown from "./fixtures/unknown.json";
import caution from "./fixtures/caution.json";
import poor from "./fixtures/poor_quality.json";
import healthy from "./fixtures/healthy.json";
import err422 from "./fixtures/error_422.json";
import disagree from "./fixtures/disagree.json";
import narrow from "./fixtures/narrow.json";
import plainHealthy from "./fixtures/plain_healthy.json";

const base = { recordings: common.recordings, showcase: common.showcase.showcase, limitations: common.health.limitations,
  userName: "Test User", needsName: false, status: "idle", noop: () => {} };
const sel = (f) => ({ kind: "dataset", rec_id: f.record.input.rec_id, label: f.record.input.rec_id });
const done = (f, extra = {}) => ({ ...base, selection: sel(f), status: "done", record: f.record, channels: f.channels,
  signals: f.signals, activity: f.activity, report: f.report, explanation: { status: "ready", data: f.explanation },
  initialPanelView: "developer", ...extra });
const mech = (f, extra = {}) => done(f, { initialPanelView: "mechanic", ...extra });

const screens = {
  "no recording": [base, ["No recording selected", "Select a recording"]],
  "name prompt": [{ ...base, userName: "", needsName: true }, ["Display name", "stored in this browser only"]],
  running: [{ ...base, selection: sel(knownFault), status: "running" }, ["Checking recording", "Analysing signals", "Preparing findings", "Investigating…"]],
  "known fault": [done(knownFault, { initialUi: { sel: "out_brg", evCh: "over_radial", evView: "envelope", signal: true } }),
    ["Known fault", "Overhang bearing · cage fault", "Dataset label: Overhang bearing · cage fault", "above healthy limit — consistent with a fault",
      "Location according to the dataset label (model)", "outer bearing housing", "CNN-6 · main model", "Secondary logistic regression", "> 0.9999", "margin",
      "Decision rule: provisional research setting", "Excess kurt.",
      "Model scores are not the probability that the diagnosis is correct.", "This recording was used to build the model",
      "Envelope spectrum", "BPFO", "Measured", "Checks against calibrated thresholds"]],
  "known fault, report tab": [done(knownFault, { tab: "report" }), ["What the readings suggest", "Supporting evidence",
    "Conflicting or missing evidence", "What to inspect next", "guidance library not yet connected"]],
  "known fault, activity tab": [done(knownFault, { tab: "activity" }), ["Bundle verified", "deterministic step", "Planned"]],
  "further inspection (no banner)": [done(unknown), ["Further inspection needed", "Candidate locations", "fail"], ["Checks disagree"]],
  "caution with a narrow margin (no banner)": [done(caution), ["Healthy — caution", "Narrowly above the speed-aware baseline (B1s) threshold",
    "above healthy limit — caution", "the difference comes from a narrow margin"], ["Checks disagree"]],
  "checks disagree (rule variants, not narrow)": [done(disagree), ["Checks disagree — recheck recommended.", "alternative decision rules reach different verdicts"]],
  "exact tie at a threshold": [done(narrow), ["Exactly at the familiarity (Mahalanobis distance) threshold"], ["Checks disagree"]],
  "plain healthy: no change marker": [done(plainHealthy), ["Healthy", "within what the healthy checks accept"],
    [], () => unusualSensors(plainHealthy.channels, qualityFlags(plainHealthy.record)).length === 0 && plainHealthy.channels.marked.length === 0],
  "poor quality": [done(poor, { initialUi: { sel: "s_tach" } }), ["Warnings · measured speed and signal quality", "glitched tacho intervals", "glitched pulse intervals · measured"]],
  // ---- App-3 Mechanic view
  "mechanic · known fault": [mech(knownFault), ["Mechanic", "Developer", "Possible bearing cage fault",
    "Outer bearing housing — according to the dataset label (model prediction)", "Why it was flagged", "template explanation",
    "Supporting checks", "the secondary model agrees", "What to inspect next", "Inspection guidance not yet connected.",
    "Research demonstration; not validated on real machines.", "not evidence of accuracy", "View investigation activity",
    "Reading details"], ["Checks against calibrated thresholds", "CNN-6 · main model", "Investigation agents"]],
  "rig part card": [mech(knownFault, { initialUi: { sel: "out_brg" } }), ["What this part does", "What was observed", "View signal",
    "compared with a healthy example"], ["Excess kurt."]],
  "rig sensor card": [mech(knownFault, { initialUi: { sel: "s_out" } }), ["What it measures", "View signal"]],
  "mechanic · preparing explanation": [mech(knownFault, { explanation: { status: "loading" } }), ["Preparing explanation…"]],
  "mechanic · AI-written explanation": [mech(knownFault, { explanation: { status: "ready", data: { ...knownFault.explanation, source: "llm",
    badge: "AI-written explanation of the evidence", text: { ...knownFault.explanation.text, finding: "The pattern resembles a cage fault." } } } }),
    ["AI-written explanation of the evidence", "The pattern resembles a cage fault."]],
  "mechanic · healthy with caution": [mech(caution), ["No fault detected — a check disagrees; repeat the reading or inspect further"]],
  "mechanic · unknown": [mech(unknown), ["Further inspection needed", "Candidate parts:"]],
  "mechanic · signal quality": [mech(poor), ["Warnings that affect this result", "The speed sensor signal had glitches.", "The rotation speed was not steady"]],
  "mechanic · plain healthy": [mech(plainHealthy), ["No fault detected in this recording"]],
  "reading details drawer": [done(knownFault, { initialUi: { drawer: true } }), ["Rotation speed", "Overall vibration level (RMS)",
    "Compared with a healthy example (reference ratio)", "Different from a healthy example is not the same as a confirmed fault."]],
  "healthy with calibration-speed warning": [done(healthy), ["Healthy", "healthy thresholds were not calibrated at this speed"]],
  "invalid upload": [{ ...base, upload: { name: "bad.csv", size: 12 }, status: "error", error: { status: err422.status, message: err422.detail } },
    ["Input not accepted", "got 3", "server response 422"]],
  "reserved upload refused": [{ ...base, upload: { name: "r.csv", size: 17280033 }, status: "error", error: { status: 403, message: "this appears to be a reserved test recording" } },
    ["Upload refused", "This appears to be a reserved test recording", "Server response · 403"]],
  "static fallback": [done(knownFault, { forceStatic: true }), ["Static view · rendered frame of the 3D rig", "rig-fallback.png"]],
  "running names the recording (App-2a.3)": [{ ...base, selection: sel(knownFault), status: "running", runningFor: sel(knownFault).label },
    [`Diagnosing ${sel(knownFault).label}`, "Investigating…"]],
  "reopened upload without its stored array (App-2a.3)": [done(knownFault, { selection: null,
    upload: { name: "uploaded recording (case abcdef12)", size: 0, stored: true, canRerun: false, note: "re-upload the file to run again" } }),
    ["re-upload the file to run again", "Run again"], [], () => true],
};
const always = ["Limitations", "Available when the investigation assistant is connected", "Illustrative 3D geometry",
  "Orange glow is a visual highlight, not a measured temperature."];
const never = ["Simulated", "Classifier A", "Classifier B", "Demo state", "remaining life", "urgency", "Stage 3D", "App-2b",
  "B_with_caution", "1.0000", "100.0 %", "(Recommended)", "Largest change vs healthy reference"];

let failed = 0;
for (const [name, [props, expect, absent = [], extra]] of Object.entries(screens)) {
  const p = { onPick: base.noop, onFile: base.noop, onClearUpload: base.noop, onStart: base.noop, onSaveName: base.noop, onChangeName: base.noop, ...props };
  let html;
  try { html = renderToString(<Workspace {...p} />); } catch (e) { console.error(`✗ ${name}: render threw ${e.stack}`); failed++; continue; }
  const text = html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/\s+/g, " ");
  const miss = [...expect, ...always].filter((s) => !text.includes(s) && !html.includes(s));
  const bad = [...never, ...absent].filter((s) => text.toLowerCase().includes(s.toLowerCase()));
  if (extra && !extra()) miss.push("extra check");
  if (miss.length || bad.length) { failed++; console.error(`✗ ${name}: missing ${JSON.stringify(miss)} forbidden ${JSON.stringify(bad)}`); }
  else console.log(`✓ ${name}`);
}
// App-3: the Mechanic panel (rendered alone) contains no percentage and no raw model score
import { PanelContent } from "../src/components/Aside.jsx";
for (const [name, f] of Object.entries({ knownFault, unknown, caution, poor, healthy, plainHealthy, disagree, narrow })) {
  const html = renderToString(<PanelContent record={f.record} status="done" report={f.report} activity={f.activity} view="mechanic"
    explanation={{ status: "ready", data: f.explanation }} onView={() => {}} />);
  const text = html.replace(/<[^>]+>/g, " ").replace(/&[a-z#0-9]+;/g, " ");
  const s = f.record.diagnosis.scores, raw = [s.msp.toFixed(4), s.maha.toFixed(2), s.ae_error.toFixed(4), s.b1s.toFixed(4)];
  const bad = [...(text.match(/\d\s*%/g) || []), ...(text.match(/\d\.\d{3,}/g) || []), ...raw.filter((r) => text.includes(r))];
  if (bad.length || !text.includes("Why it was flagged")) { failed++; console.error(`✗ mechanic panel ${name}: ${JSON.stringify(bad)}`); }
  else console.log(`✓ mechanic panel ${name}: no percentage, no raw score`);
}

// The disabled "Run again" of a reopened upload whose array is gone (App-2a.3)
{
  const html = renderToString(<Workspace {...{ ...done(knownFault, { selection: null,
    upload: { name: "uploaded recording", size: 0, stored: true, canRerun: false } }), onPick() {}, onFile() {}, onStart() {} }} />);
  const btn = html.match(/<button[^>]*class="start-btn"[^>]*>/)?.[0] || "";
  if (!/disabled=""/.test(btn) || !/title="re-upload the file to run again"/.test(btn)) { failed++; console.error("✗ Run again not disabled with the tooltip"); }
  else console.log("✓ Run again disabled with the tooltip when the stored upload is missing");
}

// Small screens (App-2a.3): the stacked layout at 390 px, and no horizontal scroll (document width ≤ viewport width).
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const MOBILE = { w: 390, h: 844 };
const mobileScreens = {
  "390 px · no recording": { ...base },
  "390 px · known fault": done(knownFault),
  "390 px · running": { ...base, selection: sel(knownFault), status: "running", runningFor: sel(knownFault).label },
};
const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");  // tests/.ssr → frontend/
const cssDir = join(root, "dist", "assets");
const css = existsSync(cssDir) ? readdirSync(cssDir).filter((f) => f.endsWith(".css")).map((f) => readFileSync(join(cssDir, f), "utf8")).join("\n") : "";
for (const [name, props] of Object.entries(mobileScreens)) {
  const p = { onPick() {}, onFile() {}, onClearUpload() {}, onStart() {}, onSaveName() {}, onChangeName() {}, ...props, initialViewport: MOBILE };
  const html = renderToString(<Workspace {...p} />);
  const text = html.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ");
  const miss = ["Best viewed on a desktop screen.", "Diagnosis", "Mechanic", "Measured change vs healthy reference", "Rotor test rig",
    "Limitations", "rig-fallback.png"].filter((s) => !text.includes(s) && !html.includes(s));
  if (!html.includes('class="ms-mobile"') || html.includes("<canvas")) miss.push("stacked layout without 3D");
  if (miss.length) { failed++; console.error(`✗ ${name}: missing ${JSON.stringify(miss)}`); continue; }
  // Written for tests/mobile-width.mjs, which measures them in headless Chrome at 390 × 844 (device-metrics emulation).
  const out = join(here, "mobile");
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, name.replace(/[^a-z0-9]+/gi, "_") + ".html"),
    `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>${css}</style></head><body><div id="root">${html}</div></body></html>`);
  console.log(`✓ ${name} rendered (stacked layout, no 3D)`);
}

// App-4.2: the pipeline architecture page, from a real /trace response. Structure only: every fixture edge is drawn and
// nothing else is; lr never connects to the decision; no case value appears; a bad trace draws nothing.
import Pipeline, { PipelineView } from "../src/components/Pipeline.jsx";
import traceFx from "./fixtures/trace.json";

let pipelineChecks = 0;
const pcheck = (name, ok, detail = "") => {
  pipelineChecks++;
  if (ok) console.log(`✓ pipeline: ${name}`);
  else { failed++; console.error(`✗ pipeline: ${name} ${detail}`); }
};
const textOf = (h) => h.replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/\s+/g, " ");
const flat = (bs) => bs.flatMap((b) => [b, ...flat(b.children || [])]);
const DESK = { w: 1440, h: 900 };
{
  const dev = renderToString(<PipelineView trace={traceFx} initialLevel="developer" initialSize={DESK} />);
  const drawn = [...dev.matchAll(/data-edge="([^"]+)"/g)].map((m) => m[1].replace(/&gt;/g, ">"));
  const want = traceFx.edges.map((e) => `${e.from}->${e.to}`);
  const paths = (dev.match(/<g data-edge="[^"]+"[^>]*><path /g) || []).length;
  pcheck("every trace edge has a rendered path", want.every((w) => drawn.includes(w)) && paths === want.length,
    JSON.stringify(want.filter((w) => !drawn.includes(w))));
  pcheck("no rendered path without a trace edge", drawn.every((d) => want.includes(d)) && drawn.length === new Set(drawn).size,
    JSON.stringify(drawn.filter((d) => !want.includes(d))));
  pcheck("lr has no edge to decision", !drawn.some((d) => /^lr\./.test(d) && /->decision\./.test(d)) && drawn.some((d) => d.startsWith("lr.")));
  const blocks = flat(traceFx.blocks);
  pcheck("every trace block is drawn (developer titles)", blocks.every((b) => dev.includes(`data-block="${b.id}"`)) &&
    blocks.every((b) => textOf(dev).includes(b.title_developer.slice(0, 20))));
  pcheck("header says it is the general architecture, not a diagnosis", textOf(dev).includes("General architecture view") &&
    textOf(dev).includes("not the result of a specific diagnosis"));
  // no case value: case id, record / activity / explanation numbers, statuses, hashes
  const vals = [];
  const walk = (x) => {
    if (Array.isArray(x)) return x.forEach(walk);
    if (!x || typeof x !== "object") return;
    if (typeof x.from === "string" && "value" in x) { vals.push(x.value); return; }
    if (typeof x.timing_ms === "number") vals.push(x.timing_ms);
    Object.values(x).forEach(walk);
  };
  walk(traceFx.blocks);
  const numbers = vals.flatMap((v) => (typeof v === "number" ? [v] : v && typeof v === "object" ? Object.values(v).filter((y) => typeof y === "number") : []))
    .filter((v) => !Number.isInteger(v)).map(String).filter((v) => v.length >= 5);
  const hashes = [traceFx.case_id, ...vals.filter((v) => typeof v === "string" && /^[0-9a-f]{12,}$/.test(v))];
  const leaked = [...numbers, ...hashes].filter((v) => dev.includes(v));
  pcheck(`no case value rendered (${numbers.length} numbers, ${hashes.length} ids and hashes checked)`, numbers.length > 20 && !leaked.length, JSON.stringify(leaked.slice(0, 5)));
  pcheck("no status rendered", !/(\bnot_run\b|\bcompleted\b|"warning")/.test(dev));

  const mech = renderToString(<PipelineView trace={traceFx} initialLevel="mechanic" initialSize={DESK} />);
  pcheck("mechanic titles", blocks.every((b) => textOf(mech).includes(b.title_mechanic.slice(0, 20))) && !textOf(mech).includes(traceFx.blocks[6].title_developer));

  const dec = traceFx.blocks.find((b) => b.id === "decision");
  const aside = (h) => textOf((/<aside[\s\S]*<\/aside>/.exec(h) || [""])[0]);  // the drawer only
  const drawerDev = aside(renderToString(<PipelineView trace={traceFx} initialLevel="developer" initialSelected="decision" initialSize={DESK} />));
  pcheck("drawer (developer): purpose, operation, ports, connections with the kind in words", drawerDev.includes(dec.purpose) &&
    drawerDev.includes(dec.operation.slice(0, 60)) && drawerDev.includes(dec.ports_in[0].label) && drawerDev.includes("carries a model output (Model)"));
  const drawerMech = aside(renderToString(<PipelineView trace={traceFx} initialLevel="mechanic" initialSelected="decision" initialSize={DESK} />));
  pcheck("drawer (mechanic): purpose only", drawerMech.includes(dec.purpose) && !drawerMech.includes(dec.operation.slice(0, 60)) && !drawerMech.includes("How it works") && !drawerMech.includes(dec.ports_in[0].label));

  const loading = renderToString(<Pipeline caseId="x" fetchTrace={() => new Promise(() => {})} />);
  pcheck("loading: plain state, no diagram", loading.includes("Loading the pipeline") && !loading.includes("pl-canvas"));
  const broken = { ...traceFx, edges: [...traceFx.edges, { from: "lr.probs", to: "nowhere.port", kind: "model", label: "" }] };
  const bad = renderToString(<PipelineView trace={broken} initialSize={DESK} />);
  pcheck("a trace that cannot be drawn: empty state, never a half-drawn diagram", bad.includes("could not be drawn") && !bad.includes("pl-canvas"));
  const warned = [];
  const extra = { ...traceFx, blocks: [...traceFx.blocks, { id: "future_step", title_mechanic: "Future step", title_developer: "Future step",
    purpose: "", operation: "", ports_in: [], ports_out: [] }] };
  const up = renderToString(<PipelineView trace={extra} initialSize={DESK} warn={(m) => warned.push(m)} />);
  pcheck("a block with no layout entry is drawn in the unplaced lane, with a warning", up.includes('data-block="future_step"') &&
    up.includes("UNPLACED") && warned.length === 1 && warned[0].includes("future_step"));

  // 390 px page for tests/mobile-width.mjs
  const small = renderToString(<PipelineView trace={traceFx} initialLevel="mechanic" initialSelected="decision" initialSize={MOBILE} />);
  const out = join(here, "mobile");
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, "pipeline_390.html"), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>${css}</style></head><body><div id="root">${small}</div></body></html>`);
  pcheck("390 px pipeline page written", small.includes("pl-root"));
}

// App-4.2.1: legibility of the rebalanced layout, labels on demand, and the way in and out of the page.
import { countCrossings, edgeThroughBlock, GEOM, layoutPipeline, titleWidth } from "../src/lib/pipelineLayout.js";
import { fitView, LEGIBLE } from "../src/components/PipelineCanvas.jsx";
{
  const L = layoutPipeline(traceFx);
  const through = edgeThroughBlock(L);
  pcheck(`no edge passes through a block (${countCrossings(L)} line crossings)`, through.length === 0, JSON.stringify(through));
  const f1440 = fitView(L.bounds, { w: 1440, h: 900 }), f1280 = fitView(L.bounds, { w: 1280, h: 800 });
  const whole1440 = Math.min((1440 - 32) / L.bounds.w, (900 - 112) / L.bounds.h);
  pcheck(`fit to screen at 1440×900 shows the whole graph at scale ${f1440.scale.toFixed(3)} ≥ ${LEGIBLE}`, f1440.scale >= LEGIBLE && Math.abs(f1440.scale - whole1440) < 1e-9);
  pcheck(`fit to screen at 1280×800 keeps a legible scale (${f1280.scale.toFixed(3)}; width fitted, vertical pan)`, f1280.scale >= LEGIBLE && f1280.scale * L.bounds.w <= 1280 - 32 + 1e-6);
  const short = Object.values(L.nodes).filter((n) => n.w < titleWidth(n.block.title_developer) + 2 * GEOM.PAD_X);
  pcheck("no block is narrower than its Developer title", short.length === 0, JSON.stringify(short.map((n) => n.id)));
  const none = renderToString(<PipelineView trace={traceFx} initialLevel="developer" initialSize={DESK} />);
  const sel = renderToString(<PipelineView trace={traceFx} initialLevel="developer" initialSelected="decision" initialSize={DESK} />);
  const labelled = (h) => [...h.matchAll(/<g data-edge="([^"]+)"[^>]*>(?:(?!<\/g>)[\s\S])*?class="pl-edge-label"/g)].map((m) => m[1].replace(/&gt;/g, ">"));
  const decisionEdges = traceFx.edges.filter((e) => e.from.startsWith("decision.") || e.to.startsWith("decision."));
  pcheck("edge labels by default only on the selected block's edges", labelled(none).length === 0 && labelled(sel).length > 0 &&
    labelled(sel).every((id) => /(^|>)decision\./.test(id)) && labelled(sel).length <= decisionEdges.length, JSON.stringify(labelled(sel)));
  const back = renderToString(<PipelineView trace={traceFx} caseId={traceFx.case_id} initialSize={DESK} />);
  pcheck("back link to the case in the header (navigation only; the id is not shown as text)",
    back.includes(`href="#/case/${traceFx.case_id}"`) && !textOf(back).includes(traceFx.case_id));
  const panel = renderToString(<PanelContent record={knownFault.record} status="done" report={knownFault.report} activity={knownFault.activity} view="mechanic" onView={() => {}} />);
  pcheck("the case view links to the pipeline page", panel.includes(`href="#/case/${knownFault.record.case_id}/pipeline"`) && textOf(panel).includes("How this diagnosis was made"));
}

if (failed) { console.error(`${failed} screen(s) failed`); process.exit(1); }
console.log(`all ${Object.keys(screens).length + Object.keys(mobileScreens).length + 1} screens rendered; ${pipelineChecks} pipeline checks passed`);
