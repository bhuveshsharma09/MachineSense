// Mechanic-level wording (Stage App-3), mirroring backend/explain/plain.py: plain words, no percentages, no raw scores.
// Everything is derived from the evidence record, the check assessment and the measured channel evidence.

export const GUIDANCE_TEXT = "Inspection guidance not yet connected.";
export const LIMITATION = "Research demonstration; not validated on real machines.";
export const IN_SAMPLE = "This recording was used to build the model, so the result is not evidence of accuracy.";

export const FAULT_PLAIN = {
  imbalance: "imbalance", horizontal_misalignment: "horizontal misalignment", vertical_misalignment: "vertical misalignment",
  underhang_ball: "bearing ball fault", underhang_cage: "bearing cage fault", underhang_outer: "bearing outer-race fault",
  overhang_ball: "bearing ball fault", overhang_cage: "bearing cage fault", overhang_outer: "bearing outer-race fault",
  normal: "normal operation",
};
export const PART_PLAIN = { in_brg: "Inner bearing housing", out_brg: "Outer bearing housing", rotor: "Rotor discs", coupling: "Coupling", shaft: "Shaft" };

// Mechanic-friendly labels with the technical term in brackets
export const MEASURE_LABEL = {
  rms: "Overall vibration level (RMS)", order_1x: "Vibration once per rotation (1×)", order_2x: "Vibration twice per rotation (2×)",
  kurtosis: "Impulsiveness (excess kurtosis)", env_bpfo1: "Repeated impacts, outer-race rate (envelope spectrum)",
  env_bpfi1: "Repeated impacts, inner-race rate (envelope spectrum)", env_bsf2: "Repeated impacts, ball rate (envelope spectrum)",
  env_ftf1: "Repeated impacts, cage rate (envelope spectrum)",
};
export const CHANNEL_LABEL = {
  tacho: "Speed sensor (tachometer)", under_axial: "Inner bearing, along the shaft (axial)", under_radial: "Inner bearing, radial",
  under_tangential: "Inner bearing, sideways (tangential)", over_axial: "Outer bearing, along the shaft (axial)",
  over_radial: "Outer bearing, radial", over_tangential: "Outer bearing, sideways (tangential)", mic: "Microphone",
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
  if (d.decision === "unknown") {
    const top = Object.entries(d.class10_probabilities).sort((a, b) => b[1] - a[1]).slice(0, 2);
    const parts = [...new Set(top.flatMap(([c]) => partsFor(c)))];
    return { kind: "candidate", parts, text: parts.length ? "Candidate parts: " + parts.map((p) => PART_PLAIN[p].toLowerCase()).join(", ")
      + " — from the two most likely fault types (model prediction)" : "No candidate part: the most likely types include normal operation" };
  }
  return { kind: "none", parts: [], text: null };
}

const WARN = [
  [/speed outside the range the model was trained on/, "The rotation speed is outside the range the model was trained on."],
  [/healthy thresholds were not calibrated at this speed/, "The healthy checks were not calibrated at this rotation speed."],
  [/shaft speed varies/, "The rotation speed was not steady during the recording."],
  [/windows use the recording speed|no usable tacho/, "The speed sensor signal was partly unusable."],
  [/glitched tacho intervals/, "The speed sensor signal had glitches."],
  [/microphone bursts removed/, "Short bursts were removed from the microphone signal."],
  [/possible clipping/, "A vibration sensor may have been overloaded (clipping)."],
  [/missing \(NaN\) samples/, "Some samples were missing."],
];

export function warnings(record) {
  const healthy = record.diagnosis.decision === "healthy";
  const out = [];
  for (const w of record.signal_quality.warnings) {
    if (/not calibrated at this speed/.test(w) && !healthy) continue;
    const p = (WARN.find(([re]) => re.test(w)) || [null, "A recording-quality warning applies."])[1];
    if (!out.includes(p)) out.push(p);
  }
  return out;
}

export function supporting(record, assessment) {
  if (!assessment) return null;
  const d = record.diagnosis;
  const by = Object.fromEntries(assessment.checks.map((c) => [c.key, c]));
  const baseOk = by.ae.passed && by.b1s.passed;
  const base = d.decision === "healthy"
    ? (baseOk ? "Both healthy-baseline checks agree it looks like the healthy recordings" : "One or both healthy-baseline checks see a difference from the healthy recordings")
    : (baseOk ? "The healthy-baseline checks do not see a difference from the healthy recordings" : "The healthy-baseline checks also see a difference from the healthy recordings");
  return `${base}; the secondary model ${d.top_class === d.lr_secondary.top_class ? "agrees." : "points to a different condition."}`;
}

export const rpm = (hz) => Math.round(60 * hz).toLocaleString("en-US");
export const speedText = (hz) => `Rotation speed ${rpm(hz)} rpm (shaft speed ${hz.toFixed(1)} Hz)`;
