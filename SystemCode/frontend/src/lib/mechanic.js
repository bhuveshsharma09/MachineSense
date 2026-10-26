// Mechanic-level wording (Stage App-3), mirroring backend/explain/plain.py: plain words, no percentages, no raw scores.
// Everything is derived from the evidence record, the check assessment and the measured channel evidence.
import { healthyCheckPhrase } from "./decisionSummary.js";
import { calibrationWarning } from "./model.js";

export const LIMITATION = "Research demonstration; not validated on real machines.";
export const IN_SAMPLE = "This development recording was used for training or threshold calibration; its result is not an independent accuracy test.";

export const FAULT_PLAIN = {
  imbalance: "imbalance", horizontal_misalignment: "horizontal misalignment", vertical_misalignment: "vertical misalignment",
  underhang_ball: "bearing ball fault", underhang_cage: "bearing cage fault", underhang_outer: "bearing outer-race fault",
  overhang_ball: "bearing ball fault", overhang_cage: "bearing cage fault", overhang_outer: "bearing outer-race fault",
  normal: "normal operation",
};
// "What to inspect next" = section 04 of the server's report (backend.report.guidance_items, the one selection), shown as
// given with each step's citation and verification note; never re-selected here. null until the report has arrived.
export function guidanceItems(report) {
  return report?.sections?.find((s) => s.n === "04")?.items ?? null;
}
export const PART_PLAIN = { in_brg: "Underhang bearing housing", out_brg: "Overhang bearing housing", rotor: "Rotor discs", coupling: "Coupling", shaft: "Shaft" };

// Mechanic-friendly labels with the technical term in brackets
export const MEASURE_LABEL = {
  rms: "Overall vibration level (RMS)", order_1x: "Vibration once per rotation (1×)", order_2x: "Vibration twice per rotation (2×)",
  kurtosis: "Impulsiveness (excess kurtosis)", env_bpfo1: "Repeated impacts, outer-race rate (envelope spectrum)",
  env_bpfi1: "Repeated impacts, inner-race rate (envelope spectrum)", env_bsf2: "Repeated impacts, ball rate (envelope spectrum)",
  env_ftf1: "Repeated impacts, cage rate (envelope spectrum)",
};
export const CHANNEL_LABEL = {
  tacho: "Speed sensor (tachometer)", under_axial: "Underhang bearing, along the shaft (axial)", under_radial: "Underhang bearing, radial",
  under_tangential: "Underhang bearing, sideways (tangential)", over_axial: "Overhang bearing, along the shaft (axial)",
  over_radial: "Overhang bearing, radial", over_tangential: "Overhang bearing, sideways (tangential)", mic: "Microphone",
};
export const REFERENCE_LABEL = "Compared with a healthy example (reference ratio)";
export const DIFFERENCE_NOTE = "Different from a healthy example is not the same as a confirmed fault.";

const partsFor = (c) => !c || c === "normal" ? [] : c.startsWith("underhang") ? ["in_brg"] : c.startsWith("overhang") ? ["out_brg"]
  : c === "imbalance" ? ["rotor"] : c.endsWith("misalignment") ? ["coupling", "shaft"] : [];

export function finding(record) {
  const d = record.diagnosis;
  if (d.decision === "known_fault") return `Possible ${FAULT_PLAIN[d.fault_class] || d.fault_class.replace(/_/g, " ")}`;
  if (d.decision === "unknown") return "Further inspection needed";
  if (d.caution) return "No fault detected — a check disagrees; repeat the reading or inspect further";
  return "No fault detected in this recording";
}

export function findingKind(record) {
  const d = record.diagnosis;
  return d.decision === "known_fault" ? "fault" : d.decision === "unknown" ? "further" : d.caution ? "caution" : "healthy";
}

export function location(record) {
  const d = record.diagnosis;
  if (d.decision === "known_fault") {
    const parts = partsFor(d.fault_class);
    return { kind: "predicted", parts, text: parts.map((p) => PART_PLAIN[p]).join(" and ") + " — according to the dataset label (model prediction)" };
  }
  return { kind: "none", parts: [], text: null };
}

const WARN = [
  [/speed outside the range the model was trained on/, "The rotation speed is outside the range the model was trained on."],
  [/healthy thresholds were not calibrated at this speed/, "The shaft speed is outside the calibrated range. The healthy checks are less reliable at this speed."],
  [/shaft speed varies/, "The rotation speed was not steady during the recording."],
  [/windows use the recording speed|no usable tacho/, "The speed sensor signal was partly unusable."],
  [/glitched tacho intervals/, "The speed sensor signal had glitches."],
  [/microphone bursts removed/, "Short bursts were removed from the microphone signal."],
  [/possible clipping/, "A vibration sensor may have been overloaded (clipping)."],
  [/missing \(NaN\) samples/, "Some samples were missing."],
];

// One signal-quality warning in plain words
export const plainWarning = (w) => (WARN.find(([re]) => re.test(w)) || [null, "A recording-quality warning applies."])[1];

// App-8.6 item 4: the calibration-speed warning on every verdict, with the record's calibrated range
export function warnings(record) {
  const out = [];
  for (const w of record.signal_quality.warnings) {
    const p = /not calibrated at this speed/.test(w) ? calibrationWarning(record) : plainWarning(w);
    if (!out.includes(p)) out.push(p);
  }
  return out;
}

export function supporting(record, assessment) {
  if (!assessment) return null;
  const d = record.diagnosis;
  const base = healthyCheckPhrase(assessment.checks, d.decision);
  const plain = base === "Unlike the healthy recordings" ? "Both healthy checks find this recording unlike the healthy recordings" : base;
  return `${plain}. A second, simpler model ${d.top_class === d.lr_secondary.top_class ? "reaches the same answer." : "points to a different condition."}`;
}

export const rpm = (hz) => Math.round(60 * hz).toLocaleString("en-US");
export const speedText = (hz) => `Rotation speed ${rpm(hz)} rpm (shaft speed ${hz.toFixed(1)} Hz)`;

// App-8.6 item 8: a Mechanic-view rewrite of model jargon in a stored explanation (explanations written before the
// prompt and guard change may still use it); the Developer view keeps the technical names.
const JARGON = [
  [/\bclassifier confidence(?: \(MSP\))?(?=\W|$)/gi, "confidence"],
  [/\bMSP\b/g, "confidence score"],
  [/\bfamiliarity(?: \(Mahalanobis distance\))?(?=\W|$)/gi, "pattern-match"],
  [/\bMahalanobis distance\b/gi, "pattern distance"],
  [/\bspeed-aware baseline(?: \(B1s\))? check(?=\W|$)/gi, "speed-matched healthy-pattern check"],
  [/\bspeed-aware baseline(?: \(B1s\))?(?=\W|$)/gi, "speed-matched healthy-pattern score"],
  [/\bautoencoder error\b/gi, "healthy-pattern score"],
  [/\bautoencoder\b/gi, "healthy-pattern check"],
];
// Older saved explanations can predate the canonical sensor names. Keep the measured axis and bearing side intact.
export function sensorNames(t) {
  return String(t || "")
    .replace(/\b(?:inner|underhang) bearing sensor,?\s*radial\b,?/gi, "underhang bearing, radial sensor")
    .replace(/\b(?:outer|overhang) bearing sensor,?\s*radial\b,?/gi, "overhang bearing, radial sensor")
    .replace(/\b(?:inner|underhang) bearing sensor,?\s*(?:sideways|tangential)\b,?/gi, "underhang bearing, tangential sensor")
    .replace(/\b(?:outer|overhang) bearing sensor,?\s*(?:sideways|tangential)\b,?/gi, "overhang bearing, tangential sensor")
    .replace(/\b(?:inner|underhang) bearing sensor,?\s*(?:along the shaft|axial)\b,?/gi, "underhang bearing, axial sensor")
    .replace(/\b(?:outer|overhang) bearing sensor,?\s*(?:along the shaft|axial)\b,?/gi, "overhang bearing, axial sensor")
    .replace(/\binner bearing sensor\b/gi, "underhang bearing sensor")
    .replace(/\bouter bearing sensor\b/gi, "overhang bearing sensor");
}
export const plainExplain = (t) => JARGON.reduce((s, [re, by]) => s.replace(re, by), sensorNames(t))
  .replace(/\b(the)\s+the\b/gi, "$1")
  .replace(/\bthe confidence (?:check )?failed its check\b/gi, (m) => `${m[0] === "T" ? "The" : "the"} model's confidence check did not pass`)
  .replace(/\bthe confidence (?:check )?passed its check\b/gi, (m) => `${m[0] === "T" ? "The" : "the"} model's confidence check passed`)
  .replace(/\bcheck (passed|failed) its check\b/gi, "check $1");
export const JARGON_WORDS = /classifier confidence|familiarity|autoencoder|speed-aware baseline|\bMSP\b|Mahalanobis/i;

// App-8.6 item 16: the in-sample statement is shown once (the verdict card already carries it)
export const notInSample = (line) => !/used (?:to build the model|for training or threshold calibration)/i.test(line);

// App-8.6 item 13: bearing steps that name only other bearing fault types are left out of the Mechanic view for a bearing
// verdict (display only: the report's section 04 and the assistant keep every step); analyst terms get a short gloss.
const FAULT_TERMS = { inner: /inner[- ]race|\bBPFI\b/i, outer: /outer[- ]race|\bBPFO\b/i, ball: /\bball\b|\bBSF\b/i, cage: /\bcage\b|\bFTF\b/i };
export function bearingKind(faultClass) {
  const m = /^(?:underhang|overhang)_(ball|cage|outer)$/.exec(faultClass || "");
  return m ? m[1] : null;
}
export function fitBearingSteps(steps, faultClass) {
  const kind = bearingKind(faultClass);
  if (!kind) return { steps, hidden: 0 };
  const keep = steps.filter((s) => {
    const named = Object.entries(FAULT_TERMS).filter(([, re]) => re.test(s.text)).map(([k]) => k);
    return !named.length || named.includes(kind);
  });
  return { steps: keep, hidden: steps.length - keep.length };
}
export const ENVELOPE_GLOSS = "(envelope spectrum: Sensors → Spectrum → Envelope)";
export const needsEnvelopeGloss = (t) => /envelope|\bcomb\b|defect line|sidebands?/i.test(t);
