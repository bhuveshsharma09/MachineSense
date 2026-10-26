// Dashboard view model (Stage App-8): pure functions over one case's API responses (evidence record, /report,
// /channels, /signals, /activity, /explanation, /cases). They select, map and format; nothing here invents a value.
import { CH, PART, fmtCheck, fmtScore, human, partsForClass, qualityFlags, topClasses } from "./model.js";
import { PART_PLAIN } from "./mechanic.js";

// The seven measurement channels compared with a healthy recording (the tachometer only gives the speed).
export const COMPARED = ["under_axial", "under_radial", "under_tangential", "over_axial", "over_radial", "over_tangential", "mic"];
export const METRIC_NAME = { rms: "RMS", order_1x: "1×", order_2x: "2×", env_bpfo1: "env BPFO", env_bpfi1: "env BPFI", env_bsf2: "env 2×BSF", env_ftf1: "env FTF" };
export const METRIC_PLAIN = { rms: "overall vibration", order_1x: "once per rotation", order_2x: "twice per rotation",
  env_bpfo1: "outer-race impact rate", env_bpfi1: "inner-race impact rate", env_bsf2: "ball impact rate", env_ftf1: "cage rate" };

// "(source: <id>; <note>)" at the end of a section 04 item (backend.report.guidance_items writes it into the text).
const SOURCE = /\s*\(source: ([^;)]+)(?:; ([^)]*))?\)\s*$/;
export function guidanceSteps(items) {
  return (items || []).map((it) => {
    const m = SOURCE.exec(it.text || "");
    return { text: m ? it.text.slice(0, m.index) : it.text, src: m ? m[1].trim() : it.cite || null, flag: m?.[2]?.trim() || null,
      title: it.hover || "", tag: it.tag };
  });
}

// Only the first step of these cards names a single measured channel and an order-spectrum comparison.
export function showMeTarget(step, index) {
  if (index !== 0) return null;
  if (step.src === "misalignment-2x-axial") return { channel: "under_axial", view: "spectrum", mode: "order" };
  if (step.src === "imbalance-1x") return { channel: "under_radial", view: "spectrum", mode: "order" };
  return null;
}

// The speed-aware healthy check (B1s) against its threshold τ_b: where both sit on one scale. A healthy-baseline check,
// not a severity measure.
export function gauge(record) {
  const v = record.diagnosis.scores.b1s, t = record.diagnosis.thresholds.tau_b;
  const max = Math.max(2 * t, v * 1.15);
  return { value: v, threshold: t, max, f: Math.min(1, v / max), t: t / max, beyond: v > t };
}

// One SVG arc gauge: background arc, value arc, threshold marker and ticks (values in [0, 1] of the scale).
export function arcPoint(f, r, cx = 120, cy = 118) {
  const a = Math.PI * (1 - f);
  return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
}
export function arcPath(f0, f1, r) {
  const [x0, y0] = arcPoint(f0, r), [x1, y1] = arcPoint(f1, r);
  return `M${x0.toFixed(1)},${y0.toFixed(1)} A${r},${r} 0 0 1 ${x1.toFixed(1)},${y1.toFixed(1)}`;
}

// Log-scale position of a ratio on the channel axis (×1/8 … ×8), in % of the track.
export const CH_MIN = 1 / 8, CH_MAX = 8;
export const logPos = (x) => (Math.min(Math.max(Math.log2(x), Math.log2(CH_MIN)), Math.log2(CH_MAX)) - Math.log2(CH_MIN))
  / (Math.log2(CH_MAX) - Math.log2(CH_MIN)) * 100;
// App-8.6 item 6: a ratio beyond ×8 / ÷8 is drawn at the axis edge with an overflow marker and its value.
const big = (x) => (x >= 10 ? Math.round(x) : Math.round(x * 10) / 10);
export function overflow(ratio) {
  if (!(ratio > 0)) return null;
  if (ratio > CH_MAX) return { side: "right", text: `×${big(ratio)} ▸` };
  if (ratio < CH_MIN) return { side: "left", text: `◂ ÷${big(1 / ratio)}` };
  return null;
}
export const CH_TICKS = [[1 / 8, "÷8"], [1 / 4, "÷4"], [1 / 2, "÷2"], [1, "×1"], [2, "×2"], [4, "×4"], [8, "×8"]];

// Each compared channel: its change metric's ratio to the healthy reference, the typical healthy band, and whether it is
// beyond the band (the /channels "exceeding" list). Channels without a comparable metric have no ratio.
export function channelRows(channels) {
  if (!channels) return [];
  const by = Object.fromEntries((channels.channels || []).map((c) => [c.channel, c]));
  const exceeding = new Set(channels.exceeding || []), marked = new Set(channels.marked || []);
  return COMPARED.map((id) => {
    const c = by[id];
    if (!c || !c.change_metric) return { id, name: CH[id].name, sensor: CH[id].sensor, comparable: false };
    return { id, name: CH[id].name, sensor: CH[id].sensor, comparable: true, metric: c.change_metric, ratio: c.change,
      factor: c.change_factor, direction: c.direction, band: c.band, beyond: exceeding.has(id), marked: marked.has(id) };
  });
}
export const changeText = (r) => `${r.direction === "up" ? "↑" : "↓"} ×${r.factor.toFixed(1)}`;

// The part shown in the part view: the selection, else the predicted location, a candidate, a changed sensor, or the rotor.
export function focusPart(sel, loc, unusual) {
  return sel || loc.fault[0] || loc.cand[0] || unusual[0] || "rotor";
}

// The three most probable classes of CNN-6 with the logistic regression's score for the same class.
export function candidates(record) {
  const d = record.diagnosis, lr = d.lr_secondary;
  return topClasses(d.class10_probabilities, 3).map((c, i) => {
    const parts = partsForClass(c);
    return { rank: i + 1, cls: c, name: human(c), cnn: fmtScore(d.class10_probabilities[c]), lr: fmtScore(lr.probabilities?.[c] ?? 0),
      loc: parts.length ? parts.map((p) => PART_PLAIN[p]).join(" and ") + " · according to the dataset label" : "No fault location (normal operation)" };
  });
}

// Signal quality as checks, from the record's signal_quality fields (measured; no model).
export function qualityChecks(record) {
  const q = record.signal_quality, [lo, hi] = q.train_speed_range_hz || [null, null];
  const fb = (q.window_speed_fallback || []).filter(Boolean).length;
  const inRange = lo == null || (q.shaft_speed_hz >= lo && q.shaft_speed_hz <= hi);
  return [
    { ok: q.speed_stable, text: q.speed_stable ? "Speed steady during the recording" : "Speed not steady during the recording" },
    { ok: inRange, text: inRange ? "Speed inside the range the model was trained on" : "Speed outside the range the model was trained on" },
    { ok: q.tacho_glitches === 0 && !fb, text: q.tacho_glitches < 0 ? "No usable tachometer pulses" : q.tacho_glitches > 0
      ? "Tachometer pulses had glitches" : fb ? `Tachometer pulses unusable in ${fb} windows` : "Tachometer pulses clean" },
    { ok: !q.mic_bursts_removed, text: q.mic_bursts_removed ? "Microphone bursts removed" : "No microphone bursts" },
    { ok: !(q.clipping_channels || []).length, text: (q.clipping_channels || []).length ? "Possible clipping on a vibration sensor" : "No clipping" },
    { ok: !q.nan_count, text: q.nan_count ? "Some samples missing" : "No missing samples" },
  ];
}

export function qualityRows(record) {
  const q = record.signal_quality, r = (x) => x.map((v) => v.toFixed(2)).join("–") + " Hz";
  return [
    ["Shaft speed", q.shaft_speed_hz.toFixed(2) + " Hz"], ["Speed variation", q.speed_range_pct.toFixed(2) + " %"],
    ["Training speed range", q.train_speed_range_hz ? r(q.train_speed_range_hz) : "—"],
    ["Calibration speed range", q.calibration_normal_speed_range_hz ? r(q.calibration_normal_speed_range_hz) : "—"],
    ["Tacho glitches", String(q.tacho_glitches)], ["Mic bursts removed", String(q.mic_bursts_removed)],
    ["Clipping channels", (q.clipping_channels || []).join(", ") || "none"], ["NaN samples", String(q.nan_count)],
  ];
}

// Sensors flagged amber (quality) and parts highlighted, as on the rig.
export { qualityFlags };

// Part view facts: the shaft speed and the part's largest measured change (Mechanic: words only).
export function partFacts(partId, record, rows, dev) {
  const part = PART[partId];
  const facts = [{ k: "Shaft speed", v: Math.round(60 * record.signal_quality.shaft_speed_hz).toLocaleString("en-US") + " rpm" }];
  const mine = rows.filter((r) => r.comparable && part.ch.includes(r.id));
  if (mine.length) {
    const r = mine.reduce((a, b) => (b.factor / b.band > a.factor / a.band ? b : a));
    facts.push({ k: "Measured change · " + CH[r.id].name.split(" · ").pop().toLowerCase(), v: dev ? changeText(r) : r.beyond ? "Beyond typical" : "Within typical" });
  }
  return facts;
}

// Earlier cases: the speed-aware healthy check of the most recent stored cases (separate recordings, not one machine).
export function trendBars(cases, current) {
  const xs = (cases || []).filter((c) => c?.diagnosis?.scores && Number.isFinite(c.diagnosis.scores.b1s));
  const t = current.diagnosis.thresholds.tau_b;
  const max = Math.max(2 * t, ...xs.map((c) => c.diagnosis.scores.b1s)) * 1.1;
  return { threshold: t, line: (t / max) * 100, bars: xs.map((c) => ({ id: c.case_id, current: c.case_id === current.case_id,
    h: (c.diagnosis.scores.b1s / max) * 100, value: c.diagnosis.scores.b1s, beyond: c.diagnosis.scores.b1s > t,
    label: (c.created_at || "").slice(5, 16).replace("T", " "), decision: c.diagnosis.decision, caution: !!c.diagnosis.caution })) };
}

export const verdictWord = (decision, caution) => decision === "known_fault" ? "Known fault" : decision === "unknown" ? "Inspect further"
  : caution ? "Healthy · caution" : "Healthy";

export const BEARING_NOTE = "Bearing evidence: look for peaks at the FTF, BPFO, BSF and BPFI lines. Lines are drawn at the measured speed.";
// App-8.6 item 2: a bearing verdict opens the spectrum on the predicted bearing's radial channel, envelope view
export function bearingSpectrum(record) {
  const d = record?.diagnosis, c = d?.decision === "known_fault" ? d.fault_class || "" : "";
  if (!/^(underhang|overhang)_/.test(c)) return null;
  return { ch: c.startsWith("underhang") ? "under_radial" : "over_radial", sp: "envelope" };
}

export const checkExpr = (c) => `${c.value_text} ${c.relation} ${c.threshold_text}`;
export { fmtCheck };
