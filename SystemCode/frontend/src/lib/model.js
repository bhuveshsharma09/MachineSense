// View model: every value shown is taken from a real API response (evidence record, /channels, /signals,
// /activity, /report). Nothing here invents a number; functions only select, map and format.

export const CALIBRATION_SPEED_WARNING =
  "healthy thresholds were not calibrated at this speed; a healthy verdict is less reliable";
export const IN_SAMPLE_NOTICE = "This recording was used to build the model; the result is not evidence of accuracy.";
export const SCORES_NOTE = "Model scores are not the probability that the diagnosis is correct.";
export const ASK_DISABLED = "Available when the investigation assistant is connected";

// Backend channel ids (ml PreprocessConfig.channels). Underhang = inner bearing housing, overhang = outer.
export const CHANNELS = [
  { id: "tacho", name: "Tachometer", short: "Tach", sensor: "s_tach" },
  { id: "under_axial", name: "Inner bearing · axial", short: "In · ax", sensor: "s_in", dir: "Axial" },
  { id: "under_radial", name: "Inner bearing · radial", short: "In · rad", sensor: "s_in", dir: "Radial" },
  { id: "under_tangential", name: "Inner bearing · tangential", short: "In · tan", sensor: "s_in", dir: "Tangential" },
  { id: "over_axial", name: "Outer bearing · axial", short: "Out · ax", sensor: "s_out", dir: "Axial" },
  { id: "over_radial", name: "Outer bearing · radial", short: "Out · rad", sensor: "s_out", dir: "Radial" },
  { id: "over_tangential", name: "Outer bearing · tangential", short: "Out · tan", sensor: "s_out", dir: "Tangential" },
  { id: "mic", name: "Microphone", short: "Mic", sensor: "s_mic" },
];
export const CH = Object.fromEntries(CHANNELS.map((c) => [c.id, c]));
const UNDER = ["under_axial", "under_radial", "under_tangential"];
const OVER = ["over_axial", "over_radial", "over_tangential"];

export const POS = {
  approx: "A mounted unit is visible on this bearing housing in the reference photo. Exact position and axis orientation are not verified.",
  illus: "Not identifiable in the reference photo. Shown for orientation only.",
};
export const PARTS = [
  { id: "motor", kind: "component", label: "Motor", desc: "Drives the shaft through a flexible coupling.", ch: ["tacho"] },
  { id: "coupling", kind: "component", label: "Coupling", desc: "Connects the motor to the shaft. Horizontal and vertical misalignment are introduced here in the dataset.", ch: ["under_axial", "under_radial"] },
  { id: "in_brg", kind: "component", label: "Inner bearing housing", desc: "Bearing housing on the motor side of the rotor. Called “underhang” in the MAFAULDA dataset.", ch: UNDER },
  { id: "rotor", kind: "component", label: "Rotor discs", desc: "Discs on the shaft. Test masses fitted here create imbalance in the dataset.", ch: ["under_radial", "over_radial"] },
  { id: "shaft", kind: "component", label: "Shaft", desc: "Carries the rotor discs between the two bearing housings.", ch: ["tacho"] },
  { id: "out_brg", kind: "component", label: "Outer bearing housing", desc: "Bearing housing on the far side of the rotor. Called “overhang” in the MAFAULDA dataset.", ch: OVER },
  { id: "s_in", kind: "sensor", pos: "approx", label: "Accelerometer", full: "Accelerometer · inner bearing", desc: "Triaxial accelerometer: axial, radial and tangential channels.", ch: UNDER },
  { id: "s_out", kind: "sensor", pos: "approx", label: "Accelerometer", full: "Accelerometer · outer bearing", desc: "Triaxial accelerometer: axial, radial and tangential channels.", ch: OVER },
  { id: "s_tach", kind: "sensor", pos: "illus", label: "Tachometer", full: "Tachometer", desc: "Optical once-per-revolution pulse sensor; used to measure shaft speed.", ch: ["tacho"] },
  { id: "s_mic", kind: "sensor", pos: "illus", label: "Microphone", full: "Microphone", desc: "Records sound near the rotor.", ch: ["mic"] },
];
export const PART = Object.fromEntries(PARTS.map((p) => [p.id, p]));

// Class → rig part (Stage App-2a mapping). Location is according to the dataset label (model output).
export function partsForClass(cls) {
  if (!cls || cls === "normal") return [];
  if (cls.startsWith("underhang")) return ["in_brg"];
  if (cls.startsWith("overhang")) return ["out_brg"];
  if (cls === "imbalance") return ["rotor"];
  if (cls.endsWith("misalignment")) return ["coupling", "shaft"];
  return [];
}
export const PART_NAME = { in_brg: "inner bearing housing", out_brg: "outer bearing housing", rotor: "rotor discs", coupling: "coupling", shaft: "shaft" };

export const CLASS_LABEL = {
  normal: "Normal", imbalance: "Imbalance", horizontal_misalignment: "Horizontal misalignment",
  vertical_misalignment: "Vertical misalignment", underhang_ball: "Underhang bearing · ball fault",
  underhang_cage: "Underhang bearing · cage fault", underhang_outer: "Underhang bearing · outer race",
  overhang_ball: "Overhang bearing · ball fault", overhang_cage: "Overhang bearing · cage fault",
  overhang_outer: "Overhang bearing · outer race",
};
export const human = (c) => CLASS_LABEL[c] || String(c || "").replace(/_/g, " ");
export const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
export const nf = (n) => Math.round(n).toLocaleString("en-US");

export function topClasses(probs, n = 2) {
  return Object.entries(probs || {}).sort((a, b) => b[1] - a[1]).slice(0, n).map(([c]) => c);
}

// Where to highlight on the rig.
export function locations(record) {
  if (!record) return { fault: [], cand: [] };
  const d = record.diagnosis;
  if (d.decision === "known_fault") return { fault: partsForClass(d.fault_class), cand: [] };
  if (d.decision === "unknown") {
    const cand = [...new Set(topClasses(d.class10_probabilities, 2).flatMap(partsForClass))];
    return { fault: [], cand };
  }
  return { fault: [], cand: [] };
}

// Sensors with a measured signal-quality flag (amber), each with its reason; only flags that carry a warning.
export function qualityFlags(record) {
  const out = {};
  if (!record) return out;
  const q = record.signal_quality;
  for (const ch of q.clipping_channels || []) if (CH[ch]) out[CH[ch].sensor] = "Possible clipping · measured";
  const fb = (q.window_speed_fallback || []).filter(Boolean).length;
  if (q.tacho_glitches < 0) out.s_tach = "No usable pulse train · measured";
  else if (q.tacho_glitches > 0) out.s_tach = `${q.tacho_glitches} glitched pulse intervals · measured`;
  else if (fb) out.s_tach = `Pulses unusable in ${fb} windows · measured`;
  if (q.mic_bursts_removed) out.s_mic = `${q.mic_bursts_removed} bursts removed · measured`;
  return out;
}

// Sensors of the channels changed beyond typical healthy variation (lime; at most 2), excluding amber ones.
export function unusualSensors(channels, flags = {}) {
  if (!channels) return [];
  return [...new Set((channels.marked || []).map((c) => CH[c]?.sensor).filter(Boolean))].filter((s) => !flags[s]);
}

// Warnings to show: every signal-quality warning, except the calibration-speed warning on a non-healthy verdict.
export function shownWarnings(record) {
  if (!record) return [];
  const healthy = record.diagnosis.decision === "healthy";
  return record.signal_quality.warnings.filter((w) => healthy || w !== CALIBRATION_SPEED_WARNING);
}

// A model score in [0, 1] without implying certainty: never "1.0000" or "0.0000" (same rule as backend.checks.fmt_score).
export function fmtScore(v, digits = 4) {
  const lim = 10 ** -digits;
  if (v > 1 - lim) return "> 0." + "9".repeat(digits);
  if (v < lim) return "< 0." + "0".repeat(digits - 1) + "1";
  return v.toFixed(digits);
}

// A check value or threshold as the report and main panel show it (backend.checks._num: value_text / threshold_text):
// scores in [0, 1] as fmtScore with 6 digits; other values with 4 decimals below 10, else 2 decimals.
export function fmtCheck(v) {
  return v >= 0 && v <= 1 ? fmtScore(v, 6) : Math.abs(v) < 10 ? v.toFixed(4) : v.toFixed(2);
}

// User-facing text without internal rule codes (limitations come from the bundle manifest).
export const cleanText = (x) => x.replace(/\s*\((?:rule )?B_with_caution\)/g, "").replace("provisional demo default", "provisional research setting");

export function verdict(record) {
  const d = record.diagnosis;
  if (d.decision === "known_fault") return { kind: "fault", chip: "Known fault" };
  if (d.decision === "unknown") return { kind: "further", chip: "Further inspection needed" };
  if (d.caution) return { kind: "caution", chip: "Healthy — caution" };
  return { kind: "healthy", chip: "Healthy" };
}

// The verdict heading: the category for healthy / caution / unknown; the fault class for a known fault (the category
// then appears once, as the chip).
export function verdictTitle(record) {
  const d = record.diagnosis;
  return d.decision === "known_fault" ? human(d.fault_class) : verdict(record).chip;
}

export function recordingTitle(rec) {
  if (!rec) return "";
  const bearing = /^(underhang|overhang)_/.test(rec.class10);
  const sev = rec.class10 === "normal" ? "" : ` · ${rec.severity_unit === "mm" ? rec.severity.toFixed(2) + " mm" : rec.severity + " g"}${bearing ? " added mass" : ""}`;
  return human(rec.class10) + sev;
}

export const STEP_CODE = { bundle_verified: "BV", signal_quality: "SQ", models: "MD", decision_rule: "DR", physics: "PH", explanations: "XP", report: "RP" };
export const PLANNED_AGENTS = [
  { code: "CO", name: "Coordinator", role: "Selects the next investigation step." },
  { code: "SQ", name: "Signal-quality agent", role: "Checks whether the recording is usable." },
  { code: "DX", name: "Diagnostic agent", role: "Runs the numerical models and checks agreement." },
  { code: "MG", name: "Maintenance-guidance agent", role: "Retrieves relevant inspection guidance." },
  { code: "ER", name: "Evidence-review agent", role: "Checks that conclusions are supported." },
];
export const PIPELINE = ["Bundle verified", "Signal quality", "Models scored", "Decision rule", "Physics cursors and envelope spectrum", "Explanations", "Report"];
