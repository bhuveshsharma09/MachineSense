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
  signals: f.signals, activity: f.activity, report: f.report, ...extra });

const screens = {
  "no recording": [base, ["No recording selected", "Select a recording"]],
  "name prompt": [{ ...base, userName: "", needsName: true }, ["Display name", "stored in this browser only"]],
  running: [{ ...base, selection: sel(knownFault), status: "running" }, ["Running the deterministic pipeline", "Investigating…"]],
  "known fault": [done(knownFault, { initialUi: { sel: "out_brg", evCh: "over_radial", evView: "envelope" } }),
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
  "healthy with calibration-speed warning": [done(healthy), ["Healthy", "healthy thresholds were not calibrated at this speed"]],
  "invalid upload": [{ ...base, upload: { name: "bad.csv", size: 12 }, status: "error", error: { status: err422.status, message: err422.detail } },
    ["Input not accepted", "got 3", "server response 422"]],
  "reserved upload refused": [{ ...base, upload: { name: "r.csv", size: 17280033 }, status: "error", error: { status: 403, message: "this appears to be a reserved test recording" } },
    ["Upload refused", "This appears to be a reserved test recording", "Server response · 403"]],
  "static fallback": [done(knownFault, { forceStatic: true }), ["Static view · rendered frame of the 3D rig", "rig-fallback.png"]],
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
if (failed) { console.error(`${failed} screen(s) failed`); process.exit(1); }
console.log(`all ${Object.keys(screens).length} screens rendered`);
