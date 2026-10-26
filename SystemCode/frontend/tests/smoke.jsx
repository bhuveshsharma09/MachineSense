// Frontend smoke test: the production build passes (npm run build) and the main screens render, server-side, from
// recorded real API responses (tests/fixtures, made by tests/make_fixtures.py). No browser and no new dependency.
import { renderToString } from "react-dom/server";
import Workspace from "../src/components/Workspace.jsx";
import InvestigationDialog from "../src/components/InvestigationDialog.jsx";
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
  "no recording": [base, ["Understand what your machine is telling you", "Start investigation"]],
  "name prompt": [{ ...base, userName: "", needsName: true }, ["Display name", "stored in this browser only"]],
  running: [{ ...base, selection: sel(knownFault), status: "running" }, ["Start investigation"]],
  "known fault": [done(knownFault, { initialUi: { sel: "out_brg", evCh: "over_radial", evView: "envelope", signal: true } }),
    ["Known fault", "Overhang bearing · cage fault", "Dataset label: Overhang bearing · cage fault", "above healthy limit — consistent with a fault",
      "Location according to the dataset label (model)", "outer bearing housing", "CNN-6 · main model", "Secondary logistic regression", "> 0.9999", "margin",
      "Decision rule: provisional research setting", "Excess kurt.",
      "Model scores are not the probability that the diagnosis is correct.", "This recording was used to build the model",
      "Envelope spectrum", "BPFO", "Measured", "Checks against calibrated thresholds"]],
  "known fault, report tab": [done(knownFault, { tab: "report" }), ["What the readings suggest", "Supporting evidence",
    "Conflicting or missing evidence", "What to inspect next", "source: bearing-defect-frequencies",
    "Citation not yet checked against the full text", "curated guidance library"], ["guidance library not yet connected"]],
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
    "Overhang bearing housing — according to the dataset label (model prediction)", "Why it was flagged", "template explanation",
    "Supporting checks", "A second, simpler model reaches the same answer", "What to inspect next", "(source: bearing-defect-frequencies; Citation not yet checked against the full text)",
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
    ["Start investigation"]],
  "reserved upload refused": [{ ...base, upload: { name: "r.csv", size: 17280033 }, status: "error", error: { status: 403, message: "this appears to be a reserved test recording" } },
    ["Start investigation"]],
  "static fallback": [done(knownFault, { forceStatic: true }), ["Static view · rendered frame of the 3D rig", "rig-fallback.png"]],
  "running names the recording (App-2a.3)": [{ ...base, selection: sel(knownFault), status: "running", runningFor: sel(knownFault).label },
    ["Start investigation"]],
  "reopened upload without its stored array (App-2a.3)": [done(knownFault, { selection: null,
    upload: { name: "uploaded recording (case abcdef12)", size: 0, stored: true, canRerun: false, note: "re-upload the file to run again" } }),
    ["New investigation"], [], () => true],
};
const always = ["MachineSense"];
const never = ["not yet connected", "Simulated", "Classifier A", "Classifier B", "Demo state", "remaining life", "urgency", "Stage 3D", "App-2b",
  "B_with_caution", "1.0000", "100.0 %", "(Recommended)", "Largest change vs healthy reference"];

let failed = 0;
for (const [name, [props, expect, absent = [], extra]] of Object.entries(screens)) {
  const p = { onPick: base.noop, onFile: base.noop, onClearUpload: base.noop, onStart: base.noop, onSaveName: base.noop, onChangeName: base.noop, ...props };
  let html;
  // App-8: the diagnosis panel lives on the phone layout (desktop: a summary card linking the dashboard), so a report screen
  // is checked on both widths together.
  try { html = renderToString(<Workspace {...p} />) + (p.record && p.status === "done" ? renderToString(<Workspace {...p} initialViewport={{ w: 390, h: 844 }} />) : ""); } catch (e) { console.error(`✗ ${name}: render threw ${e.stack}`); failed++; continue; }
  const text = html.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/\s+/g, " ");
  const miss = [...expect, ...always].filter((s) => !text.includes(s) && !html.includes(s));
  const bad = [...never, ...absent].filter((s) => text.toLowerCase().includes(s.toLowerCase()));
  if (extra && !extra()) miss.push("extra check");
  if (miss.length || bad.length) { failed++; console.error(`✗ ${name}: missing ${JSON.stringify(miss)} forbidden ${JSON.stringify(bad)}`); }
  else console.log(`✓ ${name}`);
}
// App-5: no guidance placeholder may come back anywhere in the frontend source (the rendered screens are covered by `never`).
{
  const walk = (d) => readdirSync(d).flatMap((f) => statSync(`${d}/${f}`).isDirectory() ? walk(`${d}/${f}`) : [`${d}/${f}`]);
  const hits = walk("src").filter((f) => /not yet connected/i.test(readFileSync(f, "utf8")));
  if (hits.length) { failed++; console.error(`✗ guidance placeholder reappeared in ${hits.join(", ")}`); }
  else console.log("✓ no guidance placeholder in the frontend source");
}
// App-7: the assistant. Answers render the answer, citations (title on hover), actions and the review badge, and nothing
// else: a reasoning field in the payload is never shown.
import AgentChat, { AnswerView, REVIEW_BADGE, UNAVAILABLE } from "../src/components/AgentChat.jsx";
{
  const strip = (h) => h.replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/\s+/g, " ");
  const secret = "PRIVATE-REASONING-TEXT";
  const agentAnswer = {
    question: "What should I inspect next?", answer: "The verdict is a known fault; the cited guidance lists envelope checks.",
    claims: [{ text: "Known fault.", refs: ["diagnosis.decision"] }],
    citations: [{ id: "bearing-defect-frequencies", title: "Check a bearing verdict against the envelope defect lines" }],
    actions: [{ tool: "diagnose", arguments: {}, summary: "verdict: known fault: overhang cage; 2 of 4 checks passed" },
              { tool: "lookup_guidance", arguments: {}, summary: "7 inspection step(s) from bearing-defect-frequencies" }],
    review: { status: "passed", failed: [] }, source: "agent", prompt_version: "x", model: "m", latency_ms: 1, tool_calls_used: 2,
    reasoning: secret, reasoning_content: secret,
  };
  const fallback = { ...agentAnswer, source: "fallback", reason: "evidence review failed", citations: [], actions: [],
    answer: "The assistant could not give an answer supported by the case record. The fixed report's findings and inspection steps are below.",
    review: { status: "fallback", failed: [{ check: "numbers", detail: "9.9 is not a value returned by a tool in this turn" }] },
    report_sections: [{ n: "04", title: "What to inspect next", items: [{ tag: "Guidance", text: "Check the comb. (source: bearing-defect-frequencies)" }] }] };
  const h1 = renderToString(<AnswerView answer={agentAnswer} />);
  const t1 = strip(h1);
  const ok1 = t1.includes(REVIEW_BADGE.passed) && t1.includes("bearing-defect-frequencies") && h1.includes('title="Check a bearing verdict') &&
    t1.includes("Actions (2)") && t1.includes("lookup_guidance") && t1.includes("7 inspection step(s)") && !h1.includes(secret);
  if (!ok1) { failed++; console.error("✗ assistant answer: badge, citation chip with title, actions, no reasoning"); }
  else console.log("✓ assistant answer: badge, citation chip with title, actions, no reasoning");
  const h2 = renderToString(<AnswerView answer={fallback} />);
  const t2 = strip(h2);
  const ok2 = t2.includes(REVIEW_BADGE.fallback) && t2.includes("could not give an answer supported") && t2.includes("Check the comb.") &&
    !t2.includes("9.9") && !h2.includes(secret);
  if (!ok2) { failed++; console.error("✗ assistant fallback: badge and report items, no rejected detail, no reasoning"); }
  else console.log("✓ assistant fallback: badge and report items, no rejected detail, no reasoning");
  const h3 = strip(renderToString(<AnswerView answer={{ ...fallback, reason: "language model disabled" }} />));
  if (!h3.includes(UNAVAILABLE) || h3.includes(REVIEW_BADGE.fallback)) { failed++; console.error("✗ assistant unavailable line"); }
  else console.log("✓ assistant unavailable: one line pointing to the report");
  const chat = strip(renderToString(<AgentChat caseId="c1" open onOpen={() => {}} onClose={() => {}}
    initialEntries={[{ question: "What should I inspect next?", status: "done", answer: agentAnswer }]} />));
  const none = strip(renderToString(<AgentChat caseId={null} open onOpen={() => {}} onClose={() => {}} />));
  const ok4 = chat.includes("What should I inspect next?") && chat.includes(REVIEW_BADGE.passed) && !/not connected|preview|draft only/i.test(chat) &&
    none.includes("Start an investigation") && !chat.includes(secret);
  if (!ok4) { failed++; console.error("✗ assistant panel: a working chat for the case; the no-case state kept"); }
  else console.log("✓ assistant panel: a working chat for the case; the no-case state kept");
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

// A reopened upload without its stored array cannot be run from the new investigation dialog.
{
  const html = renderToString(<InvestigationDialog open recordings={common.recordings} showcase={common.showcase.showcase}
    upload={{ name: "uploaded recording", stored: true, canRerun: false, note: "re-upload the file to run again" }}
    status="idle" onPick={() => {}} onFile={() => {}} onClearUpload={() => {}} onStart={() => {}} onClose={() => {}} />);
  const btn = html.match(/<button[^>]*class="start-btn"[^>]*>/)?.[0] || "";
  if (!/disabled=""/.test(btn) || !html.includes("re-upload the file to run again")) { failed++; console.error("✗ unavailable stored upload was not blocked"); }
  else console.log("✓ unavailable stored upload blocked in the investigation dialog");
}

// Small screens (App-2a.3): the stacked layout at 390 px, and no horizontal scroll (document width ≤ viewport width).
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
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
  const expected = props.status === "done" ? ["Diagnosis", "Mechanic", "Measured change vs healthy reference", "Rotor test rig", "Research limitations", "rig-fallback.png"]
    : ["Understand what your machine is telling you", "Start investigation", "Rotor test rig", "rig-fallback.png"];
  const miss = expected.filter((s) => !text.includes(s) && !html.includes(s));
  if (!html.includes('class="ms-mobile ') || html.includes("<canvas")) miss.push("stacked layout without 3D");
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
import { countCrossings, edgeThroughBlock, layoutPipeline, titleFits } from "../src/lib/pipelineLayout.js";
import { fitView, INSET, LEGIBLE } from "../src/components/PipelineCanvas.jsx";
{
  const L = layoutPipeline(traceFx);
  const through = edgeThroughBlock(L);
  pcheck(`no edge passes through a block (${countCrossings(L)} line crossings)`, through.length === 0, JSON.stringify(through));
  const f1440 = fitView(L.bounds, { w: 1440, h: 900 }), f1280 = fitView(L.bounds, { w: 1280, h: 800 });
  const whole1440 = Math.min((1440 - 2 * INSET.side) / L.bounds.w, (900 - INSET.top - INSET.bottom) / L.bounds.h);
  pcheck(`fit to screen at 1440×900 shows the whole graph at scale ${f1440.scale.toFixed(3)} ≥ ${LEGIBLE}`, f1440.scale >= LEGIBLE && Math.abs(f1440.scale - whole1440) < 1e-9);
  // App-8 (left to right): when the whole graph is not legible, the height is fitted and the page pans horizontally
  pcheck(`fit to screen at 1280×800 keeps a legible scale (${f1280.scale.toFixed(3)}; whole graph, or height fitted with horizontal pan)`,
    f1280.scale >= LEGIBLE && (f1280.scale * L.bounds.h <= 800 - INSET.top - INSET.bottom + 1e-6));
  // App-8: block titles wrap to at most two lines (columns have fixed widths), replacing the one-line width check
  const short = Object.values(L.nodes).filter((n) => !titleFits(n));
  pcheck("every Developer title fits its block in at most two lines", short.length === 0, JSON.stringify(short.map((n) => n.id)));
  const none = renderToString(<PipelineView trace={traceFx} initialLevel="developer" initialSize={DESK} />);
  const sel = renderToString(<PipelineView trace={traceFx} initialLevel="developer" initialSelected="decision" initialSize={DESK} />);
  const labelled = (h) => [...h.matchAll(/<g data-edge="([^"]+)"[^>]*>(?:(?!<\/g>)[\s\S])*?class="pl-edge-label"/g)].map((m) => m[1].replace(/&gt;/g, ">"));
  const decisionEdges = traceFx.edges.filter((e) => e.from.startsWith("decision.") || e.to.startsWith("decision."));
  pcheck("edge labels by default only on the selected block's edges", labelled(none).length === 0 && labelled(sel).length > 0 &&
    labelled(sel).every((id) => /(^|>)decision\./.test(id)) && labelled(sel).length <= decisionEdges.length, JSON.stringify(labelled(sel)));
  const back = renderToString(<PipelineView trace={traceFx} caseId={traceFx.case_id} initialSize={DESK} />);
  pcheck("back link to the case in the header (navigation only; the id is not shown as text)",
    back.includes(`href="#/case/${traceFx.case_id}/dashboard"`) && !textOf(back).includes(traceFx.case_id));
  const panelProps = { record: knownFault.record, status: "done", report: knownFault.report, activity: knownFault.activity, onView: () => {} };
  const devPanel = renderToString(<PanelContent {...panelProps} view="developer" />);
  const mechPanel = renderToString(<PanelContent {...panelProps} view="mechanic" />);
  pcheck("the pipeline button appears only in Developer view", devPanel.includes(`href="#/case/${knownFault.record.case_id}/pipeline"`) &&
    textOf(devPanel).includes("Pipeline view") && !mechPanel.includes(`href="#/case/${knownFault.record.case_id}/pipeline"`));
}

// App-4.3: the case-populated page ("This diagnosis"). Every displayed value is checked against the fixture trace.
import { walkPage } from "./pipeline_walk.js";
import { at, blockPaths } from "../src/lib/pipelineCase.js";
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
  // App-8: the decision panel's relation signs follow each check's outcome (a value above τ_e reads ">", never "≤")
  {
    const h = page(fxKnown, "developer"), dP = blockPaths(fxKnown).decision;
    const rows = h.split('data-check="').slice(1).map((seg) => [seg.slice(0, seg.indexOf('"')), textOf(seg.split(/pl-dec-row"|pl-dec-verdict/)[0]).replace(/&gt;/g, ">").replace(/&lt;/g, "<")]);
    const ok = rows.length === 4 && rows.every(([k, t]) => {
      const c = at(fxKnown, `${dP}.output.checks`).find((x) => x.check.startsWith(k));
      const sign = k === "msp" ? (c.passed.value ? "≥" : "<") : c.passed.value ? "≤" : ">";
      return t.includes(` ${sign} `);
    });
    pcheck("decision panel: each check row states the relation its outcome implies", ok, JSON.stringify(rows));
  }
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
    dPrep.includes('aria-checked="true"') && textOf(dPrep).includes("Underhang bearing, radial") && textOf(dPrep).includes("time (s)") &&
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
  const Lc = layoutFor(fxKnown, { extraHeight: Object.fromEntries(Object.keys(PREVIEW_BLOCKS).map((k) => [k, THUMB_EXTRA])) });
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

// The pipeline entry: on the desktop report page in the summary card (App-8, both views); on the phone in the Developer panel.
{
  const P = { onPick() {}, onFile() {}, onClearUpload() {}, onStart() {}, onSaveName() {}, onChangeName() {} };
  const href = `href="#/case/${knownFault.record.case_id}/pipeline"`;
  const desk = renderToString(<Workspace {...P} {...done(knownFault)} />);
  const phone = renderToString(<Workspace {...P} {...done(knownFault)} initialViewport={MOBILE} />);
  const mechDesk = renderToString(<Workspace {...P} {...mech(knownFault)} />);
  const mechPhone = renderToString(<Workspace {...P} {...mech(knownFault)} initialViewport={MOBILE} />);
  const running = renderToString(<Workspace {...P} {...base} selection={sel(knownFault)} status="running" runningFor="x" />);
  pcheck("main screen: Pipeline view in the desktop summary card and the phone Developer panel, hidden from the phone Mechanic panel",
    /class="pl-open-btn"[^>]*>/.test(desk) && desk.includes(href) && textOf(desk).includes("Pipeline view") && mechDesk.includes(href) &&
    phone.includes('class="pl-open-btn"') && phone.includes(href) &&
    !mechPhone.includes(href) && !running.includes("pl-open-btn"));
  const dash = `href="#/case/${knownFault.record.case_id}/dashboard"`, asst = `href="#/case/${knownFault.record.case_id}/assistant"`;
  pcheck("App-8: the report page links the dashboard and the assistant page on desktop and phone (no side panel on desktop)",
    [desk, mechDesk, phone, mechPhone].every((h) => h.includes(dash) && h.includes(asst)) && !desk.includes("Checks against calibrated thresholds"));
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
    const conf = pair(res.checks.confidence.value, res.checks.confidence.threshold, 100, " %", 1);
    if (res.checks.confidence.value < 1 && /^100\.0*\s%/.test(conf[0])) bad.push(`${c}/${e.rec_id}: a probability below 1 shown as 100 %`);
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
  // App-8.5: the wizard's page 3 reveals the real result row by row (at step 2 the confidence row is shown, familiarity and the
  // verdict not yet, and no result page); page 4 is reached only after every row is shown
  const partial = page("imbalance", fail, { initialStep: 1 });
  const row = (s, on) => new RegExp(`data-step="${s}" class="uw-run-row${on ? " is-shown" : ""}"`).test(partial);
  pcheck("unseen page: steps are revealed one at a time (at step 2 the confidence row is shown, familiarity and the verdict not yet)",
    row("label", true) && row("confidence", true) && row("familiarity", false) && row("verdict", false) && !partial.includes('data-v="verdict"')
    && textOf(partial).includes("Step 3 of 4"));
  const wiz = textOf(start);
  pcheck("unseen page: the wizard opens on step 1 with the four steps, the nine faults in their groups and the About box",
    wiz.includes("Step 1 of 4") && ["Hide a fault", "Choose a recording", "Run the checks", "See the result", "Rotor", "Alignment",
      "Underhang bearing", "Overhang bearing", "Next: choose a recording", "Research demonstration; not validated on real machines."].every((x) => wiz.includes(x))
    && (start.match(/class="uw-fault[ "]/g) || []).length === 9, JSON.stringify([...["Step 1 of 4", "Hide a fault", "Choose a recording", "Run the checks", "See the result", "Rotor", "Alignment",
      "Underhang bearing", "Overhang bearing", "Next: choose a recording", "Research demonstration; not validated on real machines."].filter((x) => !wiz.includes(x))]));
  const cards = textOf(renderToString(<UnseenView summary={summary} initialPage={2} />));
  pcheck("unseen page: recording cards do not reveal whether the example is a rejection or an honest failure before it is run",
    cards.includes("Step 2 of 4") && !/successful rejection|honest failure|failure_kind|called healthy/i.test(cards)
    && unseenFile.runs.imbalance.examples.every((e) => cards.includes(e.rec_id)));
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

// App-8: the dashboard and the assistant page, from the same recorded responses. Every fixture in both views renders; the
// honesty lines are on the page; the designer's sample values never appear; the Mechanic view shows no score or threshold.
import { DashboardView } from "../src/components/Dashboard.jsx";
import DeveloperView from "../src/components/DeveloperView.jsx";
import { STAGES } from "../src/lib/pipelineLayout.js";
import { valueOf } from "../src/lib/pipelineCase.js";
import { AssistantView, TOOLS } from "../src/components/Assistant.jsx";
import { decisionSummary, healthyCheckPhrase } from "../src/lib/decisionSummary.js";
import { guidanceSteps, showMeTarget } from "../src/lib/dashboard.js";
import { fmtCheck as fmtC, IN_SAMPLE_NOTICE as INS, SCORES_NOTE as SN, human, shownWarnings } from "../src/lib/model.js";
{
  const cases = [knownFault, unknown, caution, plainHealthy];
  pcheck("App-8.1 shared decision summary uses the saved checks for each outcome", cases.every((f) => {
    const d = f.record.diagnosis;
    const s = decisionSummary({ topClass: d.top_class, decision: d.decision, caution: d.caution, checks: f.report.assessment.checks });
    const c = Object.fromEntries(f.report.assessment.checks.map((x) => [x.key, x.passed]));
    return s.checks.msp === c.msp && s.checks.maha === c.maha && s.checks.ae === c.ae && s.checks.b1s === c.b1s
      && s.healthy === healthyCheckPhrase(f.report.assessment.checks, d.decision) && s.lines.length === 3;
  }));
  pcheck("App-8.2 dashboard check outcomes match the pipeline Decision box", [
    [knownFault, fxKnown], [unknown, fxUnknown], [caution, fxCaution], [plainHealthy, fxHealthy],
  ].every(([f, trace]) => {
    const report = decisionSummary({ topClass: f.record.diagnosis.top_class, decision: f.record.diagnosis.decision,
      caution: f.record.diagnosis.caution, checks: f.report.assessment.checks });
    const pipeline = decisionSummary({ topClass: f.record.diagnosis.top_class, decision: f.record.diagnosis.decision,
      caution: f.record.diagnosis.caution, checks: trace.blocks.find((b) => b.id === "decision").output.checks });
    return ["msp", "maha", "ae", "b1s"].every((key) => report.checks[key] === pipeline.checks[key])
      && report.healthy === pipeline.healthy && report.outcome === pipeline.outcome;
  }));
  const SAMPLE = ["6.82", "0.9981", "0.9412", "GL-BRG", "c_8f3a21", "36.454", "Sam Okafor", "xp-prompt v3", "7e2d9a1", "look_up_guidance"];
  const fx = { knownFault, unknown, caution, poor, healthy, disagree, narrow, plainHealthy };
  const dash = (f, view, extra = {}) => renderToString(<DashboardView caseId={f.record.case_id} record={f.record} report={f.report} channels={f.channels}
    signals={f.signals} activity={f.activity} explanation={{ status: "ready", data: f.explanation }} recent={[unknown.record, f.record]}
    initialView={view} initialWidth={1440} {...extra} />);
  const bad = [];
  for (const [n, f] of Object.entries(fx)) for (const view of ["mechanic", "developer"]) {
    let h; try { h = dash(f, view); } catch (e) { bad.push(`${n}/${view}: threw ${e.message}`); continue; }
    const t = textOf(h);
    const verdictLine = f.record.diagnosis.decision === "known_fault" ? `Verdict · ${human(f.record.diagnosis.top_class)} (model prediction)`
      : f.record.diagnosis.decision === "unknown" ? "Verdict · Further inspection needed" : "Verdict · Healthy";
    const need = ["Sensor channels vs a healthy recording", verdictLine,
      ...(view === "mechanic" ? ["Evidence", "Measurements", "Back to home",
        ...(f.record.diagnosis.decision === "healthy" ? [] : ["Suggested checks"]),
        ...(shownWarnings(f.record).length ? [] : ["No signal-quality warnings detected."])]
        : ["Research demonstration; not validated on real machines.", SN, "What to inspect next", "No healthy-reference spectrum is stored with the case.", "This is not a measure of damage severity."]),
      ...(f.record.source === "dataset" && view === "developer" ? [INS] : []),
      ...(f.record.diagnosis.decision === "known_fault" ? [view === "mechanic" ? "Mapped from the model’s predicted class" : "According to the dataset label (model prediction)"] : [])];
    const miss = need.filter((x) => !t.includes(x));
    const forb = [...never, ...SAMPLE].filter((x) => t.toLowerCase().includes(x.toLowerCase()));
    if (view === "mechanic") {
      const d = f.record.diagnosis;
      forb.push(...["τ_", "MSP", "Mahalanobis", fmtC(d.scores.b1s), fmtC(d.thresholds.tau_b), "Traceability", "Decision checks"].filter((x) => t.includes(x)));
      // App-8.4: the design's order (finding, rig, measurements, then what to check)
      const ordered = ['aria-label="Finding"', 'aria-label="Interactive test rig"', 'aria-label="Explore measurements"',
        ...(f.record.diagnosis.decision === "healthy" ? [] : ['aria-label="Suggested checks"'])];
      if (ordered.some((x, i) => h.indexOf(x) < 0 || i > 0 && h.indexOf(x) < h.indexOf(ordered[i - 1]))) miss.push("mechanic section order");
    } else {
      miss.push(...["Decision checks", "only add a caution to a healthy verdict", "CNN attribution · exploratory", "Not evidence of location"].filter((x) => !t.includes(x)));
    }
    if (miss.length || forb.length) bad.push(`${n}/${view}: missing ${JSON.stringify(miss)} forbidden ${JSON.stringify(forb)}`);
  }
  pcheck(`App-8 dashboard: ${Object.keys(fx).length} fixtures × 2 views render with the required lines, no sample values, no scores in Mechanic`, !bad.length, bad.join("\n"));
  const steps = guidanceSteps(knownFault.report.sections.find((s) => s.n === "04").items);
  const tk = textOf(dash(knownFault, "mechanic"));
  pcheck("App-8 dashboard: all inspection steps retain section 04 text and the guidance note, without raw ids in Mechanic",
    steps.length > 0 && steps.every((x) => x.src && knownFault.report.sections.find((s) => s.n === "04").items.some((it) => it.text.startsWith(x.text)))
    && steps.every((x) => tk.includes(x.text)) && !tk.includes("source: bearing-defect-frequencies")
    && tk.includes("Citation not yet checked against the full text"));
  const hp = textOf(dash(plainHealthy, "mechanic"));
  pcheck("App-8 dashboard: a healthy verdict hides suggested checks and fault location",
    !hp.includes("Suggested checks") && !hp.includes("Suggested location"));
  pcheck("App-8.2 further inspection does not name a candidate fault area as a next action",
    !textOf(dash(unknown, "mechanic")).includes("Consider the")
    && textOf(dash(unknown, "mechanic")).includes("inspect the readings before choosing a fault area."));
  const mechanicFault = dash(knownFault, "mechanic");
  const mechanicFaultText = textOf(mechanicFault);
  const developerFault = dash(knownFault, "developer");
  const developerFinding = developerFault.match(/<section[^>]*aria-label="Finding"[\s\S]*?<\/section>/)?.[0] || "";
  // App-8.3: the Developer view is one page of seven numbered sections under a jump bar, on the case's real data
  const SEVEN = ["verdict", "machine", "data", "models", "decision", "pipeline", "report"];
  const title = (html) => html.match(/<h1 class="rd-m-title">([^<]+)<\/h1>/)?.[1];
  const devBad = [];
  for (const [n, f] of Object.entries(fx)) {
    const html = dash(f, "developer"), t = textOf(html), d = f.record.diagnosis;
    const s = decisionSummary({ topClass: d.top_class, decision: d.decision, caution: d.caution, checks: f.report.assessment.checks });
    const ids = SEVEN.map((id) => html.indexOf(`id="dvw-${id}"`));
    const errs = [];
    if (ids.some((x, i) => x < 0 || (i > 0 && x < ids[i - 1]))) errs.push("seven sections in order");
    if (!["01", "07", "Verdict", "Machine", "Data", "Models", "Decision", "Pipeline", "Report"].every((x) => html.includes(x))) errs.push("jump bar");
    if (title(html) !== title(dash(f, "mechanic"))) errs.push("same finding title as Mechanic");
    if (!t.includes("Rule in use · classifier first") || !t.includes("caution only") || !t.includes(s.outcome)) errs.push("rule in use");
    if ((html.match(/class="dvw-check"/g) || []).length !== f.report.assessment.checks.length) errs.push("one card per check");
    if (!f.report.assessment.variants.every((v) => t.includes(v.name))) errs.push("alternative rules");
    if (t.includes("4 of 4 checks support") || /Rule A · used/.test(t) || t.includes("B_with_caution")) errs.push("design sample rule text");
    if (!t.includes("Full-screen diagram") || !html.includes(`href="#/case/${f.record.case_id}/pipeline"`)) errs.push("pipeline link");
    if (d.decision === "known_fault" && !t.includes("Mapped from the model's predicted class on this test rig; the sensors do not pinpoint the fault.")) errs.push("mapped caveat");
    if (errs.length) devBad.push(`${n}: ${errs.join(", ")}`);
  }
  pcheck(`App-8.3 Developer: ${Object.keys(fx).length} fixtures render seven sections, the Mechanic finding title, the rule in use and every check and variant`,
    !devBad.length, devBad.join("\n"));
  // App-8.3: section 06 embeds the pipeline diagram from the case's own trace (every value a trace path, the rule in use)
  const pipeBad = [];
  for (const [n, tr] of Object.entries({ fxKnown, fxUnknown, fxCaution, fxHealthy })) {
    const ids = STAGES.flatMap((s) => s.ids).filter((id) => blockPaths(tr)[id]);
    const html = renderToString(<DeveloperView caseId={tr.case_id} record={knownFault.record} report={knownFault.report} channels={knownFault.channels}
      signals={knownFault.signals} explanation={{ status: "ready", data: knownFault.explanation }} view="developer" onView={() => {}} trace={tr}
      initialWidth={1440} />);
    const pipe = html.slice(html.indexOf('id="dvw-pipeline"'), html.indexOf('id="dvw-report"'));
    const t = textOf(pipe), dec = blockPaths(tr).decision;
    const errs = [];
    if (!ids.every((id) => pipe.includes(`data-pn="${id}"`))) errs.push("a step card is missing");
    if (!STAGES.every((s) => t.includes(s.title))) errs.push("a stage heading is missing");
    const paths = [...pipe.matchAll(/data-p="([^"]+)"/g)].map((m) => m[1]);
    if (!paths.length || paths.some((p) => at(tr, p) === undefined)) errs.push("a value without a trace path");
    if ((pipe.match(/data-cell="/g) || []).length !== 4) errs.push("four decision checks");
    if (!t.includes(String(valueOf(at(tr, `${dec}.output.display_label`))))) errs.push("verdict");
    if (!t.includes("caution only") || t.includes("B_with_caution") || !t.includes("Expand all") || !t.includes("Connections carry")) errs.push("rule wording / controls");
    if (!pipe.includes(`href="#/case/${tr.case_id}/pipeline"`)) errs.push("full-screen link");
    if (errs.length) pipeBad.push(`${n}: ${errs.join(", ")}`);
  }
  pcheck("App-8.3 Developer section 06 draws every pipeline step from the trace, the four checks and the verdict", !pipeBad.length, pipeBad.join("\n"));
  // App-8.4: the Mechanic view is four numbered sections; no score, threshold or percentage; the location keeps its label
  const mechBad = [];
  for (const [n, f] of Object.entries(fx)) {
    const html = dash(f, "mechanic"), t = textOf(html), d = f.record.diagnosis, errs = [];
    const ids = ["rd-finding", "mvw-machine", "rd-measurements", "rd-suggested-checks"].map((id) => html.indexOf(`id="${id}"`));
    if (ids.some((x, i) => x < 0 || (i > 0 && x < ids[i - 1]))) errs.push("four sections in order");
    if (!["Verdict", "Machine", "Sensors", "What to check"].every((x) => t.includes(x))) errs.push("jump bar");
    if (/\d+(?:\.\d+)?\s?%/.test(t.replace(SN, "")) || /\b0\.\d{3,}/.test(t) || t.includes("τ_")) errs.push("a score, threshold or percentage");
    if (d.decision === "known_fault" && !(t.includes("According to the dataset label (model prediction)") && t.includes("(model prediction)"))) errs.push("location label");
    if (!t.includes("Research demonstration; not validated on real machines.") || !t.includes(SN)) errs.push("honesty footer");
    if (/\b[↑↓]\s?×\d/.test(t)) errs.push("a change factor");
    if (d.decision === "healthy" && (t.includes("Suggested checks") || t.includes("on the rig</button>"))) errs.push("healthy shows checks");
    if (errs.length) mechBad.push(`${n}: ${errs.join(", ")}`);
  }
  pcheck(`App-8.4 Mechanic: ${Object.keys(fx).length} fixtures render four sections with the location label and honesty footer, no score, threshold, percentage or factor`,
    !mechBad.length, mechBad.join("\n"));
  const mapped = (src) => {
    const report = { ...knownFault.report, sections: knownFault.report.sections.map((s) => s.n !== "04" ? s : { ...s, items: [
      { tag: "Guidance", text: `Check the 2× to 1× ratio. (source: ${src}; AI-drafted, not yet checked against a published source)`,
        hover: "Compare shaft orders — internal" },
    ] }) };
    return dash({ ...knownFault, report }, "mechanic");
  };
  pcheck("App-8.2 Show me appears only for the two declared first-step mappings", !!showMeTarget({ src: "misalignment-2x-axial" }, 0)
    && !!showMeTarget({ src: "imbalance-1x" }, 0) && !showMeTarget({ src: "bearing-defect-frequencies" }, 0)
    && !showMeTarget({ src: "misalignment-2x-axial" }, 1) && mapped("misalignment-2x-axial").includes("Show me")
    && mapped("imbalance-1x").includes("Show me") && !mapped("bearing-defect-frequencies").includes("Show me"));
  const caveatLines = [
    "Mapped from the model’s predicted class on this test rig; the sensors do not pinpoint the fault.",
    "Highlights do not confirm a fault.", "They show where the signal changed, not the fault location.",
  ];
  const spectrumText = textOf(dash(knownFault, "mechanic", { initialUi: { measure: "spectrum" } }));
  pcheck("App-8.2 Mechanic caveats remain visible word for word", caveatLines.every((line) => mechanicFaultText.includes(line))
    && mechanicFaultText.includes(knownFault.explanation.badge)
    && spectrumText.includes("No healthy-reference spectrum is stored with the case.")
    && spectrumText.includes("Without a healthy spectrum, compare the 2× peak with the 1× peak in this recording."),
  caveatLines.filter((line) => !mechanicFaultText.includes(line)).join(" | "));
  pcheck("App-8.2 new Mechanic blocks use the existing font scale without monospace labels", (() => {
    const blocks = mechanicFault.match(/<main class="rd-m-main"[\s\S]*?<\/main>/)?.[0] || "";
    const css = readFileSync("src/redesign.css", "utf8");
    return !/class="[^"]*mono[^"]*rd-m-(?:label|caveat|gloss|citation|show|why|sensor)/.test(blocks)
      && !/\.rd-m-(?:label|caveat|gloss|citation|show|why|sensor)[^\n]*font-family:\s*"Fragment Mono"/.test(css)
      && !/font-size:\s*(?:10px|14px|16px|17px|18px|19px|20px)/.test(css.slice(css.indexOf("/* The answer and its reasoning lead"), css.indexOf(".rd-m-measure {")));
  })());
  pcheck("App-8 dashboard: no demo badge or in-sample line in the Developer finding; both views link home",
    !mechanicFault.includes("Research demo</span>") && !developerFault.includes("Research demo</span>")
    && !developerFinding.includes(INS) && developerFault.includes('Back to home</a>'));
  pcheck("Dashboard uses the main rig without a separate part card in either view",
    ["mechanic", "developer"].every((view) =>
      [knownFault, plainHealthy, unknown].every((fx) =>
        !dash(fx, view).includes('aria-label="Part view"')
        && !dash(fx, view, { initialSel: "rotor" }).includes('aria-label="Part view"'))));
  pcheck("App-8 dashboard: Mechanic links home and to the assistant; Developer links to the pipeline",
    dash(knownFault, "mechanic").includes('Back to home</a>')
    && dash(knownFault, "mechanic").includes(`href="#/case/${knownFault.record.case_id}/assistant"`)
    && dash(knownFault, "developer").includes(`href="#/case/${knownFault.record.case_id}/pipeline"`));
  // App-9 addendum A: the dashboard no longer renders the panel (one dock above the routes does); it only makes room
  const withAssistant = dash(knownFault, "mechanic", { assistantOpen: true });
  pcheck("Assistant: the dashboard makes room for the expanded panel and renders no panel of its own",
    withAssistant.includes('aria-label="Finding"') && withAssistant.includes("has-assistant")
    && !withAssistant.includes('aria-label="Investigation assistant"') && !withAssistant.includes('class="as-aside"'));
  const narrowHtml = dash(knownFault, "developer", { initialWidth: 390 });
  pcheck("App-8 dashboard: below 1024 px Developer stays one column without the 3-D rig", narrowHtml.includes("is-narrow") && !narrowHtml.includes("Rotor test rig · illustrative"));
  pcheck("Mechanic keeps its full-width rig at narrow widths", dash(knownFault, "mechanic", { initialWidth: 390 }).includes('aria-label="Interactive test rig"'));
  writeFileSync(join(here, "mobile", "dashboard_390.html"), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>${css}</style></head><body><div id="root">${narrowHtml}</div></body></html>`);

  const secret = "PRIVATE-REASONING-TEXT";
  const ans = { question: "What should I inspect next?", answer: "First paragraph.\n\nSecond paragraph.", claims: [],
    citations: [{ id: "bearing-defect-frequencies", title: "Check a bearing verdict against the envelope defect lines" }],
    actions: [{ tool: "lookup_guidance", arguments: {}, summary: "7 inspection step(s) from bearing-defect-frequencies" }],
    review: { status: "revised", failed: [] }, source: "agent", reasoning: secret, reasoning_content: secret };
  const off = { ...ans, source: "fallback", reason: "language model unreachable (timeout)", review: { status: "fallback", failed: [] } };
  const av = (entries) => renderToString(<AssistantView caseId={knownFault.record.case_id} record={knownFault.record} report={knownFault.report}
    channels={knownFault.channels} initialEntries={entries} />);
  const a0 = textOf(av([])), a1h = av([{ question: ans.question, status: "done", answer: ans }]), a1 = textOf(a1h), a2 = textOf(av([{ question: "q", status: "done", answer: off }]));
  pcheck("App-8 assistant page: suggestions, the four real tool names, context from the case, honesty lines",
    a0.includes("Ask about this recording") && a0.includes("Why was this recording flagged?") && TOOLS.every((x) => a0.includes(x))
    && a0.includes("according to the dataset label") && a0.includes(SN) && a0.includes("Research demonstration; not validated on real machines.")
    && a0.includes("inspection step") && !SAMPLE.some((x) => a0.includes(x)));
  pcheck("App-8 assistant page: an answer shows its paragraphs, review badge, actions count and citation (title on hover), never reasoning",
    a1.includes("First paragraph.") && a1.includes("Second paragraph.") && a1.includes(REVIEW_BADGE.revised) && a1.includes("What I checked · 1")
    && a1.includes("Inspection guidance") && a1.includes("lookup_guidance") && a1.includes("7 inspection step(s)")
    && a1h.includes('title="Check a bearing verdict') && !a1h.includes(secret));
  pcheck("App-8 assistant page: an unavailable model gives the one-line pointer to the report, no badge", a2.includes(UNAVAILABLE) && !a2.includes(REVIEW_BADGE.fallback));
}

// App-9: conversation mode in the assistant. On reload only the current conversation's turns are shown (and used as context
// on the server); earlier questions stay listed as "not used in this conversation"; New conversation is offered once a
// conversation has turns; the empty state still offers the suggestions.
import { splitConversation } from "../src/components/Assistant.jsx";
{
  const turn = (id, q, extra = {}) => ({ id, question: q, answer: `answer ${id}`, source: "agent", review: { status: "passed", failed: [] },
    citations: [], actions: [], ...extra });
  const data = { questions: [turn(1, "old one"), turn(2, "old two"), turn(5, "current one")], current_conversation_id: 3, conversation: [5] };
  const sp = splitConversation(data);
  pcheck("App-9: reload splits the stored questions into the current conversation and earlier ones",
    sp.current.map((x) => x.id).join() === "5" && sp.earlier.map((x) => x.id).join() === "1,2" && splitConversation(null).current.length === 0);
  const view = (entries, earlier = []) => renderToString(<AssistantView caseId={knownFault.record.case_id} record={knownFault.record}
    report={knownFault.report} channels={knownFault.channels} initialEntries={entries} initialEarlier={earlier} />);
  const withConv = textOf(view(sp.current.map((a) => ({ question: a.question, status: "done", answer: a })), sp.earlier));
  pcheck("App-9: the current conversation, the earlier questions marked not used, and New conversation are shown",
    withConv.includes("current one") && withConv.includes("Earlier questions about this recording (2) · not used in this conversation")
    && withConv.includes("New conversation") && withConv.includes("remembers this conversation about this recording only"));
  const empty = textOf(view([]));
  pcheck("App-9: an empty conversation shows the suggestions and no New conversation button or earlier block",
    empty.includes("Why was this recording flagged?") && !empty.includes("New conversation") && !empty.includes("Earlier questions"));
}

// App-9 addendum A: the panel lifecycle (pure reducer) and the dock's controls. The browser test (tests/assistant-dock.mjs)
// checks the same rules with real navigation, focus and one continuous React instance.
import { CLOSED, dockReducer, expanded as dockExpanded } from "../src/lib/assistantDock.js";
import AssistantDock, { SESSION_NOTE } from "../src/components/AssistantDock.jsx";
{
  const run = (...acts) => acts.reduce((st, a) => dockReducer(st, a), CLOSED);
  const R = (id, page) => ({ type: "route", route: id ? { id, page } : null });
  const A = "case-a", B = "case-b";
  const opened = run(R(A, "dashboard"), { type: "open", caseId: A });
  pcheck("dock: open → minimize → restore keeps the same session (so the same mounted assistant and draft)",
    (() => { const m = dockReducer(opened, { type: "minimize" }), r = dockReducer(m, { type: "restore" });
      return m.open && m.minimized && r.open && !r.minimized && r.session === opened.session && !r.pinned; })());
  const pinned = dockReducer(opened, { type: "togglePin" });
  const walk = [R(A, "dashboard"), R(A, "pipeline"), R(A, "dashboard"), R(A, "pipeline")].reduce((st, a) => dockReducer(st, a), pinned);
  pcheck("dock: a pinned panel stays open with the same session across the dashboard and the same case's pipeline",
    walk.open && walk.pinned && !walk.minimized && walk.session === pinned.session && walk.caseId === A);
  const minPin = [R(A, "pipeline"), R(A, "dashboard")].reduce((st, a) => dockReducer(st, a), dockReducer(pinned, { type: "minimize" }));
  pcheck("dock: minimized + pinned stays minimized (restore available) across navigation", minPin.open && minPin.minimized && minPin.session === pinned.session);
  const toPipe = dockReducer(opened, R(A, "pipeline"));
  pcheck("dock: an unpinned panel closes on going to the pipeline (session ended) and stays open on dashboard routes",
    !toPipe.open && toPipe.session === opened.session + 1 && dockReducer(opened, R(A, "dashboard")).open);
  const home = dockReducer(pinned, R(null));
  pcheck("dock: Home closes, unpins and ends the session", !home.open && !home.pinned && home.caseId === null && home.session === pinned.session + 1);
  const other = dockReducer(pinned, R(B, "dashboard"));
  pcheck("dock: another case closes the panel and carries nothing over", !other.open && !other.pinned && other.caseId === B && other.session === pinned.session + 1);
  const deep = run(R(A, "assistant"));
  pcheck("dock: the /assistant deep link opens the panel on the dashboard; later dashboard routes keep it",
    deep.open && deep.caseId === A && dockReducer(deep, R(A, "dashboard")).open && dockExpanded(deep, A) && !dockExpanded(deep, B));
  const closed = dockReducer(pinned, { type: "close" }), again = dockReducer(closed, { type: "open", caseId: A });
  pcheck("dock: Close ends the session and a reopen is a new session (a new conversation, nothing deleted)",
    !closed.open && !closed.pinned && again.open && again.session === pinned.session + 1);
  const data = { status: "ok", record: knownFault.record, report: knownFault.report, channels: knownFault.channels };
  const dockHtml = (st) => renderToString(<AssistantDock state={st} dispatch={() => {}} initialData={data} />);
  const open1 = dockHtml(pinned);
  pcheck("dock: the panel header has Minimize, Pin (aria-pressed) and Close, and says saved history stays available",
    open1.includes('aria-label="Minimize the assistant"') && /aria-pressed="true"[^>]*aria-label="Unpin the assistant"/.test(open1)
    && open1.includes('aria-label="Close the assistant and end this session"') && textOf(open1).includes(SESSION_NOTE)
    && open1.includes('aria-label="Ask about this recording"') && !/<aside[^>]*hidden/.test(open1));
  const min1 = dockHtml(dockReducer(pinned, { type: "minimize" }));
  pcheck("dock: minimized keeps the panel mounted (hidden) and shows a visible Ask the assistant restore button",
    /<aside[^>]*hidden=""/.test(min1) && min1.includes('aria-label="Ask about this recording"') && min1.includes('class="as-restore"')
    && textOf(min1).includes("Ask the assistant"));
  pcheck("dock: a closed dock renders nothing", dockHtml(CLOSED) === "");
}

// App-9 slice 2: the Notes & status view keeps the model prediction, the technician status and observations apart, says
// the actions are a local demo with an unverified name, and shows edit and status history.
import CaseNotes, { KIND, STATUS_TEXT } from "../src/components/CaseNotes.jsx";
{
  const actor = { name: "Sam", verified: false, method: "local-demo" };
  const base = { case_id: "c1", local_demo: "Local demo action: this app does not sign in technicians; the name is the display name set in this browser and is not verified.",
    model_prediction: { kind: "model_prediction", decision: "known_fault", fault_class: "underhang_outer", display_label: "known fault: underhang outer",
      note: "Model prediction from the recording; not a physical confirmation." },
    parts: { in_brg: "Underhang bearing housing", out_brg: "Overhang bearing housing" }, fault_classes: ["underhang_outer", "underhang_ball"] };
  const none = { ...base, technician: { status: "not_confirmed", latest: null, agrees_with_model: null }, events: [] };
  const ev = { id: 2, case_id: "c1", created_at: "2026-10-26T10:00:00+00:00", status: "confirmed", fault_class: "underhang_ball", part: "in_brg", note: "Pitting on a ball", actor };
  const conf = { ...base, technician: { status: "technician_confirmed", latest: ev, agrees_with_model: false },
    events: [{ ...ev, id: 1, status: "rejected", fault_class: null, part: null }, ev] };
  const obs = [{ id: 7, kind: "technician_observation", created_at: "2026-10-26T09:00:00+00:00", updated_at: "2026-10-26T09:30:00+00:00", actor,
    text: "Grinding noise after 10 min.", edits: [{ edited_at: "2026-10-26T09:30:00+00:00", actor, previous_text: "Grinding noise.", new_text: "Grinding noise after 10 min." }] }];
  const t0 = textOf(renderToString(<CaseNotes caseId="c1" initialStatus={none} initialObservations={[]} />));
  pcheck("notes: model prediction, technician status and observations are separately labelled; local demo stated",
    t0.includes(KIND.model) && t0.includes("Known fault · Underhang bearing · outer race") && t0.includes("not a physical confirmation")
    && t0.includes(STATUS_TEXT.not_confirmed) && t0.includes("Technician observations") && t0.includes("No observations yet.")
    && t0.includes("does not sign in technicians") && t0.includes("not part of the recording's measurements") && !t0.includes("Withdraw"));
  const t1 = textOf(renderToString(<CaseNotes caseId="c1" initialStatus={conf} initialObservations={obs} />));
  pcheck("notes: a technician-confirmed fault shows class, part, unverified name and that it differs from the model, beside the unchanged prediction",
    t1.includes(STATUS_TEXT.technician_confirmed) && t1.includes("Underhang bearing · ball fault at the underhang bearing housing")
    && t1.includes("Sam (name not verified)") && t1.includes("differs from the model prediction") && t1.includes("Known fault · Underhang bearing · outer race")
    && t1.includes("Status history (2 events)") && t1.includes("Withdraw"));
  pcheck("notes: an edited observation shows its edit count and history", t1.includes("edited 1×") && t1.includes("was: “Grinding noise.”"));
  const dk = renderToString(<AssistantDock state={{ caseId: "c1", open: true, minimized: false, pinned: false, session: 0 }} dispatch={() => {}}
    initialData={{ status: "ok", record: knownFault.record, report: knownFault.report, channels: knownFault.channels }}
    notesProps={{ initialStatus: none, initialObservations: [] }} />);
  pcheck("dock: Ask and Notes & status tabs; both views mounted, the inactive one hidden",
    /role="tab"[^>]*aria-selected="true"[^>]*>Ask</.test(dk) && dk.includes(">Notes &amp; status<") && /id="as-view-notes"[^>]*hidden=""/.test(dk)
    && dk.includes('aria-label="Ask about this recording"') && dk.includes(KIND.model));
}

// App-9 addendum C: conversation answers keep the kinds of fact apart (labelled sections when the model uses them; the
// source kinds from the claim refs), show each citation's title and verification status, render a code-built status answer
// even when the model is unavailable, and the Notes view shows the procedure gate apart from inspection guidance.
import { answerKinds, sections, SUGGESTIONS as SUGG } from "../src/components/Assistant.jsx";
{
  const sec = sections("A technician recorded a confirmed fault.\n\nWhat the recording shows: over_radial is furthest outside its healthy range.\n\n**What to inspect next:** Check the comb.");
  pcheck("App-9 answers: labelled paragraphs become sections, unlabelled ones stay plain",
    sec.length === 3 && sec[0].label === null && sec[1].label === "What the recording shows" && sec[1].text.startsWith("over_radial")
    && sec[2].label === "What to inspect next" && sec[2].text === "Check the comb.");
  const a = { question: "Why?", answer: "What the model predicts: a known fault.\n\nWhat the technician recorded: no confirmation is recorded.",
    claims: [{ text: "x", refs: ["diagnosis.decision"] }, { text: "y", refs: ["notes.status.current"] }, { text: "z", refs: ["evidence.over_radial.rank"] }],
    citations: [{ id: "bearing-defect-frequencies", title: "Check a bearing verdict against the envelope defect lines", kind: "inspection guidance",
      source: "Randall and Antoni", verification: "ai_drafted" }], actions: [], review: { status: "passed", failed: [] }, source: "agent" };
  pcheck("App-9 answers: the source kinds come from the claim refs and citations", answerKinds(a).join() === "measured,model,technician,guidance");
  const v = (entries) => textOf(renderToString(<AssistantView caseId={knownFault.record.case_id} record={knownFault.record} report={knownFault.report}
    channels={knownFault.channels} initialEntries={entries} />));
  const t = v([{ question: a.question, status: "done", answer: a }]);
  pcheck("App-9 answers: sections, source-kind chips and the citation's title and AI-drafted status are shown",
    t.includes("What the model predicts") && t.includes("What the technician recorded") && t.includes("Measured (recording)")
    && t.includes("Technician record (local demo)") && t.includes("Check a bearing verdict against the envelope defect lines · AI-drafted, not yet checked against a published source"));
  const quick = { question: "Has a technician confirmed this?", answer: "No technician confirmation is recorded for this case. The model prediction is unchanged.",
    claims: [], citations: [], actions: [], review: { status: "fallback", failed: [] }, source: "fallback", reason: "language model disabled", fallback_kind: "status" };
  const tq = v([{ question: quick.question, status: "done", answer: quick }]);
  pcheck("App-9 answers: a code-built status answer is shown (not the 'unavailable' line) with its own badge",
    tq.includes("No technician confirmation is recorded for this case.") && tq.includes("From the stored case records") && !tq.includes(UNAVAILABLE));
  pcheck("App-9 answers: the status question is a suggestion", SUGG.includes("Has a technician confirmed this?"));
  const closed = { open: false, message: "No verified procedure is available.", reason: "no technician confirmation has been recorded", procedures: [] };
  const synth = { open: true, message: null, reason: "a technician confirmation is recorded and a reviewed entry applies", procedures: [{ id: "synthetic", title: "SYNTHETIC procedure",
    reviewer: "Test Reviewer", reviewed_on: "2026-10-26", source: "SYNTHETIC source", version: "1.0", safety_stop: ["SYNTHETIC stop"], steps: ["SYNTHETIC step"] }] };
  const st0 = { case_id: "c1", local_demo: "Local demo action.", model_prediction: { kind: "model_prediction", decision: "known_fault", fault_class: "imbalance", note: "Model prediction." },
    technician: { status: "not_confirmed", latest: null, agrees_with_model: null }, events: [], parts: { rotor: "Rotor discs" }, fault_classes: ["imbalance"] };
  const g0 = textOf(renderToString(<CaseNotes caseId="c1" initialStatus={st0} initialObservations={[]} initialProcedures={closed} />));
  const g1 = textOf(renderToString(<CaseNotes caseId="c1" initialStatus={st0} initialObservations={[]} initialProcedures={synth} />));
  pcheck("App-9 notes: the procedure gate says no verified procedure and why, or lists a reviewed one with reviewer, source and stop conditions",
    g0.includes("Post-confirmation procedure") && g0.includes("No verified procedure is available.") && g0.includes("Why: no technician confirmation has been recorded.")
    && g1.includes("SYNTHETIC procedure") && g1.includes("Reviewed by Test Reviewer on 2026-10-26") && g1.includes("Stop if: SYNTHETIC stop")
    && g0.includes("Inspection suggestions are separate guidance, not procedures."));
}

if (failed) { console.error(`${failed} screen(s) failed`); process.exit(1); }
console.log(`all ${Object.keys(screens).length + Object.keys(mobileScreens).length + 1} screens rendered; ${pipelineChecks} pipeline checks passed`);
