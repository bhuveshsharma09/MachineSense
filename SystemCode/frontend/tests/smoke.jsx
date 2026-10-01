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
import { fitView, INSET, LEGIBLE } from "../src/components/PipelineCanvas.jsx";
{
  const L = layoutPipeline(traceFx);
  const through = edgeThroughBlock(L);
  pcheck(`no edge passes through a block (${countCrossings(L)} line crossings)`, through.length === 0, JSON.stringify(through));
  const f1440 = fitView(L.bounds, { w: 1440, h: 900 }), f1280 = fitView(L.bounds, { w: 1280, h: 800 });
  const whole1440 = Math.min((1440 - 2 * INSET.side) / L.bounds.w, (900 - INSET.top - INSET.bottom) / L.bounds.h);
  pcheck(`fit to screen at 1440×900 shows the whole graph at scale ${f1440.scale.toFixed(3)} ≥ ${LEGIBLE}`, f1440.scale >= LEGIBLE && Math.abs(f1440.scale - whole1440) < 1e-9);
  pcheck(`fit to screen at 1280×800 keeps a legible scale (${f1280.scale.toFixed(3)}; width fitted, vertical pan)`, f1280.scale >= LEGIBLE && f1280.scale * L.bounds.w <= 1280 - 2 * INSET.side + 1e-6);
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

// App-4.3: the case-populated page ("This diagnosis"). Every displayed value is checked against the fixture trace.
import { walkPage } from "./pipeline_walk.js";
import fxHealthy from "./fixtures/trace_healthy.json";
import fxCaution from "./fixtures/trace_caution.json";
import fxKnown from "./fixtures/trace_known_fault.json";
import fxUnknown from "./fixtures/trace_unknown.json";
import fxUpload from "./fixtures/trace_upload.json";
import fxNoExpl from "./fixtures/trace_no_explanation.json";
{
  const CASES = { healthy: fxHealthy, caution: fxCaution, known_fault: fxKnown, unknown: fxUnknown, upload: fxUpload, no_explanation: fxNoExpl };
  const page = (fx, level, sel = null, mode = "case") => renderToString(<PipelineView trace={fx} initialMode={mode} initialLevel={level}
    initialSelected={sel} initialSize={DESK} />);
  const svgOf = (h) => (/<svg class="pl-canvas"[\s\S]*?<\/svg>/.exec(h) || [""])[0];
  const drawerOf = (h) => (/<aside[\s\S]*<\/aside>/.exec(h) || [""])[0];
  const verdicts = (h) => [...h.matchAll(/data-verdict="1"[^>]*>([^<]*)</g)].map((m) => m[1].replace(/&amp;/g, "&"));
  let values = 0;
  for (const [name, fx] of Object.entries(CASES)) {
    const dec = fx.blocks.find((b) => b.id === "decision").output;
    const label = dec.display_label.value;
    const errs = [];
    const seen = {};
    for (const level of ["mechanic", "developer"]) {
      const pages = [page(fx, level), ...flat(fx.blocks).map((b) => page(fx, level, b.id))];
      for (const h of pages) { const w = walkPage(h, fx); values += w.checked; errs.push(...w.errors); }
      seen[level] = new Set(pages.flatMap(verdicts));
      // per-channel autoencoder errors: never on the diagram, never in Mechanic, only in the Developer drawer of AE-6
      for (const h of pages) {
        if (svgOf(h).includes("ae_channel_errors")) errs.push(`${level}: per-channel AE errors on the diagram`);
        const inDrawer = drawerOf(h).includes(".technical.ae_channel_errors");
        if (inDrawer && !(level === "developer" && /About: AE-6/.test(h))) errs.push(`${level}: per-channel AE errors outside the AE-6 Developer drawer`);
      }
      const ae = page(fx, level, "ae6");
      const caveat = fx.blocks.find((b) => b.id === "models").children.find((c) => c.id === "ae6").technical.ae_channel_errors_caveat;
      if (level === "developer" && !(drawerOf(ae).includes(".technical.ae_channel_errors") && textOf(drawerOf(ae)).includes(caveat)))
        errs.push("developer AE-6 drawer lacks the per-channel errors with their caveat");
      if (level === "mechanic" && drawerOf(ae).includes("ae_channel_errors")) errs.push("mechanic AE-6 drawer shows per-channel errors");
    }
    pcheck(`case ${name}: every shown value equals the fixture (both levels, every drawer)`, errs.length === 0, JSON.stringify(errs.slice(0, 4)));
    pcheck(`case ${name}: Mechanic and Developer show the same verdict "${label}"`, seen.mechanic.size === 1 && seen.developer.size === 1 &&
      seen.mechanic.has(label) && seen.developer.has(label), JSON.stringify([...seen.mechanic, ...seen.developer]));
    if (dec.decision.value === "known_fault") pcheck(`case ${name}: a predicted fault is labelled a model prediction`, textOf(page(fx, "mechanic")).includes("model prediction"));
  }
  pcheck(`case pages: ${values} displayed values checked`, values > 2000);

  // the walker is not trivially passing: a doctored copy renders a value the fixture does not have
  const doctored = JSON.parse(JSON.stringify(fxKnown));
  doctored.blocks.find((b) => b.id === "models").children.find((c) => c.id === "cnn6").output.msp.value = 0.9512;
  const bad = walkPage(page(doctored, "developer"), fxKnown);
  pcheck("walker catches a doctored value (MSP changed in the rendered copy)", bad.errors.some((e) => e.includes("output.msp")), JSON.stringify(bad.errors.slice(0, 2)));
  pcheck("walker catches a bare number that is not in the trace", walkPage("<div><span>distance 0.4213</span></div>", fxKnown).errors.length === 1);

  // abstention: "further inspection needed" exactly; no "new fault" / "unknown fault detected" wording (the trace's own
  // disclaimer "it does not mean a new fault was found" is the one allowed mention, and it is a negation)
  const disclaimers = [...new Set(flat(fxUnknown.blocks).flatMap((b) => [b.operation, b.output?.unknown_meaning]).filter((x) => typeof x === "string" && /new fault/.test(x)))];
  const leaks = [];
  for (const level of ["mechanic", "developer"]) for (const sel of [null, ...flat(fxUnknown.blocks).map((b) => b.id)]) {
    let t = textOf(page(fxUnknown, level, sel));
    for (const d of disclaimers) t = t.split(d).join(" ");
    if (/new fault|unknown fault|fault detected/i.test(t)) leaks.push(`${level}/${sel}`);
  }
  pcheck("abstention: no \"new fault\" or \"unknown fault detected\" wording; the verdict reads \"further inspection needed\"",
    !leaks.length && textOf(page(fxUnknown, "mechanic")).includes("further inspection needed"), JSON.stringify(leaks));

  const decPage = textOf(page(fxKnown, "developer", "decision"));
  pcheck("rule comparison is marked as a comparison of alternative rules, not the verdict", decPage.includes("comparison only, not the verdict"));
  const drawnCase = [...page(fxKnown, "developer").matchAll(/data-edge="([^"]+)"/g)].map((m) => m[1].replace(/&gt;/g, ">"));
  pcheck("case mode: lr still has no edge to decision", drawnCase.length === fxKnown.edges.length && !drawnCase.some((d) => /^lr\./.test(d) && /->decision\./.test(d)));
  pcheck("case mode: a not-recorded value shows the trace's reason", textOf(page(fxKnown, "developer")).includes("not recorded: the number of R1 feature columns"));
  const other = { ...fxKnown, bundle_matches_record: { ...fxKnown.bundle_matches_record, value: false },
    bundle_note: "this case was produced by a different bundle" };
  pcheck("different bundle: a visible banner in case mode", textOf(page(other, "mechanic")).includes("produced by a different bundle than the one now loaded"));
  pcheck("architecture mode of a case trace shows no case value", !page(fxKnown, "developer", null, "architecture").includes("data-p=") &&
    !page(fxKnown, "developer", null, "architecture").includes(fxKnown.case_id));
  const small = renderToString(<PipelineView trace={fxUnknown} initialMode="case" initialLevel="mechanic" initialSelected="decision" initialSize={MOBILE} />);
  writeFileSync(join(here, "mobile", "pipeline_case_390.html"), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>${css}</style></head><body><div id="root">${small}</div></body></html>`);
}

const drawerOf = (h) => (/<aside[\s\S]*<\/aside>/.exec(h) || [""])[0];
// App-4.4: signal previews. /signals-derived previews are labelled "computed now", the saved envelope spectrum is not;
// a case without its stored array shows the API's reason; architecture mode shows none.
import { previewSections, stripPreviews } from "./pipeline_walk.js";
import fxMissing from "./fixtures/trace_upload_missing.json";
import { layoutPipeline as layoutFor } from "../src/lib/pipelineLayout.js";
import { PREVIEW_BLOCKS, THUMB_EXTRA } from "../src/components/PipelinePreview.jsx";
{
  const sig = knownFault.signals, savedRec = knownFault.record;
  const pv = (fx, sel = null, extra = {}) => renderToString(<PipelineView trace={fx} initialMode="case" initialLevel="developer" initialSelected={sel}
    initialSize={DESK} initialSignals={sig} initialSaved={savedRec} {...extra} />);
  const sections = (h, kind) => previewSections(h).filter((x) => x.kind === kind).map((x) => x.html);
  const loaded = pv(fxKnown);
  const comp = sections(loaded, "computed"), saved = sections(loaded, "saved");
  pcheck("block thumbnails: two computed-now previews and one saved preview, each drawn", comp.length === 2 && saved.length === 1 &&
    comp.every((x) => /computed now/i.test(textOf(x)) && x.includes("<path")) && saved.every((x) => x.includes("<path")));
  pcheck("the saved envelope spectrum is never labelled computed now", saved.every((x) => !/computed now/i.test(textOf(x)) && /saved/i.test(textOf(x))));
  const dPrep = drawerOf(pv(fxKnown, "preparation")), dSpec = drawerOf(pv(fxKnown, "order_spectrogram")), dRec = drawerOf(pv(fxKnown, "evidence_record"));
  pcheck("drawer previews: waveform and order spectrum say computed now and not part of the saved diagnosis; the envelope says saved evidence",
    textOf(dPrep).includes("Computed now for this view · not part of the saved diagnosis") && dPrep.includes("<path") &&
    textOf(dSpec).includes("Computed now for this view · not part of the saved diagnosis") && dSpec.includes("<path") &&
    textOf(dRec).includes("Saved evidence · part of the evidence record") && !/computed now/i.test(textOf(dRec)) && dRec.includes("<path"));
  pcheck("drawer preview: channel choice defaults to under_radial and names units from the response",
    dPrep.includes('aria-checked="true"') && textOf(dPrep).includes("Inner bearing, radial") && textOf(dPrep).includes("time (s)") &&
    textOf(dSpec).includes("order (multiples of the measured shaft speed"));
  const idle = drawerOf(renderToString(<PipelineView trace={fxKnown} initialMode="case" initialSelected="preparation" initialSize={DESK} />));
  pcheck("before loading: an explicit load button, no chart", textOf(idle).includes("Load signal preview (computed now from the stored recording)") && !sections(idle, "computed").some((x) => x.includes("<path")));
  const missing = drawerOf(renderToString(<PipelineView trace={fxMissing} initialMode="case" initialSelected="preparation" initialSize={DESK} />));
  const reason = fxMissing.blocks.find((b) => b.id === "presentation").output.views.find((v) => v.endpoint.endsWith("/signals")).reason;
  pcheck(`no stored array: the reason is shown instead of a chart ("${reason}")`, textOf(missing).includes(`Signal not available: ${reason}`) &&
    !textOf(missing).includes("Load signal preview") && !sections(missing, "computed").some((x) => x.includes("<path")));
  const arch = renderToString(<PipelineView trace={fxKnown} initialMode="architecture" initialSelected="preparation" initialSize={DESK} initialSignals={sig} initialSaved={savedRec} />);
  pcheck("architecture mode shows no previews", !arch.includes("data-preview") && !/computed now/i.test(textOf(arch)));
  const walked = walkPage(stripPreviews(pv(fxKnown, "decision")), fxKnown);
  pcheck(`value walker still passes with previews loaded (preview sections excluded; ${walked.checked} values)`, walked.errors.length === 0 && walked.checked > 50, JSON.stringify(walked.errors.slice(0, 3)));
  pcheck("previews carry no predicted fault", comp.concat(saved).every((x) => !/known fault|prediction|overhang cage/i.test(textOf(x))));
  const Lc = layoutFor(fxKnown, { extraWidth: Object.fromEntries(Object.keys(PREVIEW_BLOCKS).map((k) => [k, THUMB_EXTRA])) });
  const fc = fitView(Lc.bounds, { w: 1440, h: 900 });
  pcheck(`case mode with thumbnail slots: fit at 1440×900 is ${fc.scale.toFixed(3)} (whole graph) and no edge passes through a block`,
    fc.scale >= LEGIBLE && edgeThroughBlock(Lc).length === 0);
  const smallCase = renderToString(<PipelineView trace={fxKnown} initialMode="case" initialLevel="mechanic" initialSelected="preparation" initialSize={MOBILE} initialSignals={sig} initialSaved={savedRec} />);
  writeFileSync(join(here, "mobile", "pipeline_preview_390.html"), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>${css}</style></head><body><div id="root">${smallCase}</div></body></html>`);
}

// App-4.5: phone-width treatment, error states, and no new motion.
{
  const PHONE = { w: 390, h: 844 };
  const CASES = { healthy: fxHealthy, caution: fxCaution, known_fault: fxKnown, unknown: fxUnknown, upload: fxUpload, no_explanation: fxNoExpl };
  const errs = [];
  let walked = 0;
  for (const [name, fx] of Object.entries(CASES)) for (const level of ["mechanic", "developer"]) {
    const label = fx.blocks.find((b) => b.id === "decision").output.display_label.value;
    const opened = flat(fx.blocks).map((b) => renderToString(<PipelineView trace={fx} initialMode="case" initialLevel={level} initialSelected={b.id} initialSize={PHONE} />));
    const h = opened[0];
    if (h.includes('class="pl-canvas"') || !h.includes('class="pl-steps"')) errs.push(`${name}/${level}: not the step list`);
    for (const b of flat(fx.blocks)) if (!h.includes(`data-block="${b.id}"`)) errs.push(`${name}/${level}: step ${b.id} missing`);
    const v = new Set(opened.flatMap((x) => [...x.matchAll(/data-verdict="1"[^>]*>([^<]*)</g)].map((m) => m[1].replace(/&amp;/g, "&"))));
    if (!(v.size === 1 && v.has(label))) errs.push(`${name}/${level}: verdict ${[...v]}`);
    for (const x of opened) { const w = walkPage(stripPreviews(x), fx); walked += w.checked; errs.push(...w.errors.map((e) => `${name}/${level}: ${e}`)); }
  }
  pcheck(`phone width (390 px): a step list, every step, the desktop verdict, every value equals the trace (${walked} values)`, !errs.length && walked > 2000, JSON.stringify(errs.slice(0, 4)));
  const phoneOpen = renderToString(<PipelineView trace={fxKnown} initialMode="case" initialLevel="developer" initialSelected="evidence_record" initialSize={PHONE}
    initialSignals={knownFault.signals} initialSaved={knownFault.record} />);
  const phonePrev = previewSections(phoneOpen);
  pcheck("phone width keeps the computed-now and saved labelling", phonePrev.some((x) => x.kind === "computed" && /computed now/i.test(textOf(x.html))) &&
    phonePrev.filter((x) => x.kind === "saved").every((x) => !/computed now/i.test(textOf(x.html)) && /saved/i.test(textOf(x.html))) &&
    textOf(phoneOpen).includes("Saved evidence · part of the evidence record"));
  pcheck("phone width, architecture mode: the step list, no case values, no previews",
    (() => { const a = renderToString(<PipelineView trace={fxKnown} initialMode="architecture" initialSize={PHONE} />); return a.includes('class="pl-steps"') && !a.includes("data-p=") && !a.includes("data-preview"); })());

  const decisionless = JSON.parse(JSON.stringify(fxKnown));
  decisionless.blocks.find((b) => b.id === "decision").output = null;
  const broken = { "HTML instead of JSON": "<html><body>502 Bad Gateway</body></html>", "empty object": {}, "blocks not a list": { blocks: "x", edges: [], edge_kinds: {} },
    "a block without an id": { blocks: [{ title_mechanic: "x" }], edges: [], edge_kinds: {} }, "an edge without ends": { ...fxKnown, edges: [{ kind: "signal" }] },
    "case mode without a decision": decisionless };
  const bad = [];
  for (const [what, t] of Object.entries(broken)) {
    let h = "";
    try { h = renderToString(<PipelineView trace={t} initialMode="case" initialSize={DESK} />); } catch (e) { bad.push(`${what}: threw ${e.message}`); continue; }
    if (h.includes("pl-canvas") || h.includes("pl-steps") || !textOf(h).includes("could not be drawn")) bad.push(what);
  }
  pcheck(`malformed traces show the plain error state, never a partial diagram (${Object.keys(broken).length} kinds)`, !bad.length, JSON.stringify(bad));
  const cssText = readFileSync(join(root, "src", "index.css"), "utf8");
  const moving = (cssText.match(/\.pl-[^{]*\{[^}]*\b(transition|animation)\b[^}]*\}/g) || []);
  pcheck("no pipeline rule animates or transitions; the global reduced-motion rule is in place",
    !moving.length && /prefers-reduced-motion: reduce\)[^{]*\{[^}]*animation: none !important; transition: none !important/.test(cssText), JSON.stringify(moving));
}

// the main screen's way to the pipeline page: a button once a diagnosis is on screen (desktop top bar and phone header)
{
  const P = { onPick() {}, onFile() {}, onClearUpload() {}, onStart() {}, onSaveName() {}, onChangeName() {} };
  const href = `href="#/case/${knownFault.record.case_id}/pipeline"`;
  const desk = renderToString(<Workspace {...P} {...done(knownFault)} />);
  const phone = renderToString(<Workspace {...P} {...done(knownFault)} initialViewport={MOBILE} />);
  const running = renderToString(<Workspace {...P} {...base} selection={sel(knownFault)} status="running" runningFor="x" />);
  pcheck("main screen: a \"Pipeline view\" button to this case's pipeline page (desktop and phone), none while running",
    /class="pl-open-btn"[^>]*>/.test(desk) && desk.includes(href) && textOf(desk).includes("Pipeline view") &&
    phone.includes('class="pl-open-btn"') && phone.includes(href) && !running.includes("pl-open-btn"));
  pcheck("pipeline page: a \"← Back to the case\" button in the header",
    textOf(renderToString(<PipelineView trace={fxKnown} caseId={fxKnown.case_id} initialMode="case" initialSize={DESK} />)).includes("← Back to the case"));
}

// App-6: "Explore an unseen fault", rendered from the REAL committed selection (backend/unseen.json); every displayed number
// must equal the backend's value for that example, the required copy must be on the page itself, and the steps stay hidden
// until revealed.
import unseenFile from "../../backend/unseen.json";
import { UnseenView, pair, verdictText } from "../src/components/Unseen.jsx";
import { UnseenLink } from "../src/components/TopBar.jsx";
{
  const summary = { ...unseenFile, built: true, live: { available: false, reason: "saved" },
    runs: Object.fromEntries(Object.entries(unseenFile.runs).map(([c, r]) => [c, { ...r, examples: r.examples.map(({ result, ...e }) => e) }])) };
  const exOf = (c, cat, kind) => unseenFile.runs[c].examples.find((e) => e.category === cat && (!kind || e.failure_kind === kind));
  const page = (c, e, extra = {}) => {
    const ordered = [...unseenFile.runs[c].examples.filter((x) => x.role === "withheld"), ...unseenFile.runs[c].examples.filter((x) => x.role === "familiar")];
    return renderToString(<UnseenView summary={summary} initialHidden={c} initialExample={ordered.indexOf(e)} initialResult={e.result}
      initialStep={4} {...extra} />);
  };
  const start = renderToString(<UnseenView summary={summary} initialLevel="mechanic" />);
  const t0 = textOf(start);
  pcheck("unseen page: title, the simulated-fault statement and the easier-part statement are in the page copy",
    t0.includes("What happens when a machine fault appears that this model was never taught?") && t0.includes(unseenFile.simulated)
    && t0.includes(unseenFile.easier_part.replace(/"/g, "\"")));
  pcheck("unseen page: defaults to imbalance with the required copy, and lists the nine learned classes and the hidden one",
    t0.includes("This experimental model was trained without imbalance recordings.") && /Imbalance\s*— never seen/.test(t0)
    && (start.match(/<li>/g) || []).length === 9);
  pcheck("unseen page: nothing is shown before Run investigation (no verdict, no reveal)", !start.includes('data-v="verdict"') && !t0.includes("Reveal actual condition"));
  let bad = [];
  for (const [c, r] of Object.entries(unseenFile.runs)) for (const e of r.examples) for (const level of ["mechanic", "developer"]) {
    const h = page(c, e, { initialLevel: level }), t = textOf(h), res = e.result;
    const conf = pair(res.checks.confidence.value, res.checks.confidence.threshold, 100, " %");
    const fam = pair(res.checks.familiarity.value, res.checks.familiarity.threshold);
    if (!t.includes(conf[0]) || !t.includes(conf[1]) || !t.includes(fam[0]) || !t.includes(fam[1])) bad.push(`${c}/${e.rec_id}: values`);
    if (!t.includes(verdictText(res.decision))) bad.push(`${c}/${e.rec_id}: verdict`);
    const passTxt = [...h.matchAll(/class="un-chip (ok|bad)"/g)].map((m) => m[1]);
    if (passTxt.join() !== [res.checks.confidence.passed ? "ok" : "bad", res.checks.familiarity.passed ? "ok" : "bad"].join()) bad.push(`${c}/${e.rec_id}: pass/fail`);
    if (res.decision.decision === "unknown" && !t.includes(unseenFile.further_inspection)) bad.push(`${c}/${e.rec_id}: further-inspection copy`);
    if (level === "mechanic" && /MSP|Mahalanobis|τ_/.test(t.replace(/Experiment details.*$/, ""))) bad.push(`${c}/${e.rec_id}: technical names in Mechanic view`);
    if (level === "developer" && !(t.includes("MSP") && t.includes("Mahalanobis"))) bad.push(`${c}/${e.rec_id}: developer names missing`);
    if (t.includes("Actual condition")) bad.push(`${c}/${e.rec_id}: truth shown before Reveal`);
  }
  pcheck("unseen page: every curated example shows the backend's values, pass/fail and verdict; Mechanic has no technical names", !bad.length, bad.slice(0, 5).join("; "));
  const fail = exOf("imbalance", "honest failure");
  const rev = textOf(page("imbalance", fail, { initialRevealed: true }));
  pcheck("unseen page: Reveal actual condition shows the true label, marked hidden during fitting, and names the honest failure",
    rev.includes("Actual condition (dataset label)") && rev.includes("Hidden from this model during fitting") && rev.includes("honest failure"));
  const partial = page("imbalance", fail, { initialStep: 1 });
  pcheck("unseen page: steps are revealed one at a time (at step 2 the confidence card is shown, familiarity and the verdict not yet)",
    /class="un-card un-check is-shown" aria-hidden="false" data-check="MSP/.test(partial)
    && /class="un-card un-check" aria-hidden="true" data-check="Mahalanobis/.test(partial)
    && partial.includes('class="un-card glass-strong un-verdict"'));
  const ob = textOf(renderToString(<UnseenView summary={summary} initialHidden="overhang_ball" />));
  pcheck("unseen page: a class without an honest failure says so, with the reason, instead of substituting one",
    ob.includes("No honest failure among the eligible recordings: every one of the 69 eligible withheld recordings was rejected"));
  const P = { onPick() {}, onFile() {}, onClearUpload() {}, onStart() {}, onSaveName() {}, onChangeName() {} };
  const desk = renderToString(<Workspace {...P} {...base} />), phone = renderToString(<Workspace {...P} {...base} initialViewport={MOBILE} />);
  // for tests/mobile-width.mjs: the page at 390 px, fully revealed (the widest state)
  const full = page("horizontal_misalignment", exOf("horizontal_misalignment", "honest failure", "called healthy"),
    { initialRevealed: true, initialLevel: "developer" });
  writeFileSync(join(here, "mobile", "unseen_390.html"), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>${css}</style></head><body><div id="root">${full}</div></body></html>`);
  pcheck("navigation: \"Explore an unseen fault\" in the desktop top bar and the phone header",
    desk.includes('href="#/unseen"') && textOf(desk).includes("Explore an unseen fault") && phone.includes('href="#/unseen"'));
}

if (failed) { console.error(`${failed} screen(s) failed`); process.exit(1); }
console.log(`all ${Object.keys(screens).length + Object.keys(mobileScreens).length + 1} screens rendered; ${pipelineChecks} pipeline checks passed`);
