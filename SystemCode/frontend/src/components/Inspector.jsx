import { useState } from "react";
import { CHANNEL_LABEL, MEASURE_LABEL, speedText } from "../lib/mechanic.js";
import { K, al } from "../lib/theme.js";
import { CH, POS, nf } from "../lib/model.js";
import { Svg, bandPath, linePath, rangeMax } from "./Charts.jsx";

const MAXO = 8, ENV_MAXO = 12;
const f3 = (v) => (v == null || !Number.isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
const ENV_CURSORS = [["bpfo", "BPFO"], ["bpfi", "BPFI"], ["bsf", "2×BSF"], ["ftf", "FTF"]];

function chartFor(view, chId, record, signals) {
  const fr = record.signal_quality.shaft_speed_hz;
  if (view === "envelope") {
    const env = record.physics.envelope_spectrum, ys = env.channels[chId];
    if (!ys) return null;
    const ymax = rangeMax(env.orders, ys, 0.3, ENV_MAXO) * 1.15;
    const p = linePath(env.orders, ys, 0, ENV_MAXO, ymax);
    const cur = record.physics.cursors_orders;
    const markers = ENV_CURSORS.map(([k, lab]) => {
      // comb_lines lists 2×BSF ± FTF and BPFI with its ±1× sidebands: the defect line is the middle of the first triple
      const o = k === "bsf" || k === "bpfi" ? cur[k][1] : cur[k][0];
      return { left: (o / ENV_MAXO) * 100 + "%", label: lab, color: K.orangeT };
    });
    return { el: <Svg paths={[{ d: p.area, fill: al(K.violet, 0.1) }, { d: p.line, stroke: K.violet, w: 1.3 }]} />, markers,
      ticks: [0, 3, 6, 9, 12].map((o) => ({ left: (o / ENV_MAXO) * 100 + "%", label: o ? o + "×" : "0" })), ymax: f3(ymax),
      caption: `Envelope spectrum (${env.band_hz[0] / 1000}–${env.band_hz[1] / 1000} kHz band) · orders of the measured speed (1× = ${fr.toFixed(2)} Hz) · bearing cursors at the measured speed` };
  }
  const s = signals?.channels?.[chId];
  if (!s) return null;
  if (view === "waveform") {
    const w = s.waveform, m = Math.max(...w.max.map(Math.abs), ...w.min.map(Math.abs)) * 1.15 || 1;
    const dur = w.dt_s * w.max.length;
    return { el: <Svg paths={[{ d: bandPath(w.min, w.max, m), fill: al(K.violet, 0.55), stroke: K.violet, w: 0.6 }]} />, markers: [],
      ticks: [0, dur / 2, dur].map((t) => ({ left: (t / dur) * 100 + "%", label: (w.t0_s + t).toFixed(2) + " s" })), ymax: "±" + f3(m),
      caption: `Waveform · ${dur.toFixed(2)} s excerpt of the cleaned recording (V; min/max per bucket)` };
  }
  const ymax = rangeMax(signals.orders, s.spectrum, 0.5, MAXO) * 1.15;
  const p = linePath(signals.orders, s.spectrum, 0, MAXO, ymax);
  return { el: <Svg paths={[{ d: p.area, fill: al(K.violet, 0.1) }, { d: p.line, stroke: K.violet, w: 1.3 }]} />,
    markers: [1, 2, 3].map((o) => ({ left: (o / MAXO) * 100 + "%", label: o + "×", color: K.ink3 })),
    ticks: [0, 2, 4, 6, 8].map((o) => ({ left: (o / MAXO) * 100 + "%", label: o ? o + "×" : "0" })), ymax: f3(ymax),
    caption: `Order spectrum · multiples of the measured shaft speed (1× = ${fr.toFixed(2)} Hz) · V` };
}

function statsFor(chId, record, channels) {
  const q = record.signal_quality;
  if (chId === "tacho") {
    const fb = (q.window_speed_fallback || []).filter(Boolean).length;
    return {
      stats: [
        { k: "Speed", v: q.shaft_speed_hz.toFixed(2) + " Hz" }, { k: "rpm", v: nf(60 * q.shaft_speed_hz) },
        { k: "Variation", v: q.speed_range_pct.toFixed(2) + " %", c: q.speed_stable ? K.ink : K.amberT },
        { k: "Glitches", v: q.tacho_glitches < 0 ? "no pulses" : String(q.tacho_glitches), c: q.tacho_glitches ? K.amberT : K.ink },
      ],
      note: `Shaft speed ${q.shaft_speed_hz.toFixed(2)} Hz measured from the tachometer; per-window speeds vary by ${q.speed_range_pct.toFixed(2)} %` +
        (fb ? `; ${fb} windows fall back to the recording speed.` : "."),
    };
  }
  const row = channels?.channels?.find((c) => c.channel === chId);
  if (!row) return { stats: [], note: "Measured values are loading." };
  const v = row.values, r = row.reference_values, ref = channels.reference;
  const METRIC = { rms: "RMS", order_1x: "1× amplitude", order_2x: "2× amplitude", env_bpfo1: "envelope BPFO", env_bpfi1: "envelope BPFI", env_bsf2: "envelope 2×BSF", env_ftf1: "envelope FTF" };
  const change = row.change_metric
    ? `${METRIC[row.change_metric] || row.change_metric} ${changeText(row)} vs the reference (typical healthy variation ×${row.band.toFixed(1)}; ${row.exceeds ? "beyond it" : "within it"})`
    : "no comparable metric";
  const refTxt = `healthy training recording ${ref.rec_id} (${ref.speed_hz.toFixed(2)} Hz)`;
  if (chId === "mic") {
    return {
      stats: [{ k: "RMS", v: f3(v.rms) }, { k: "Ref. RMS", v: f3(r.rms) }, { k: "Change", v: row.change_metric ? changeText(row) : "—", c: row.exceeds ? K.limeT : K.ink },
        { k: "Bursts", v: String(record.signal_quality.mic_bursts_removed), c: record.signal_quality.mic_bursts_removed ? K.amberT : K.ink }],
      note: `Cleaned microphone RMS ${f3(v.rms)} V; ${refTxt}: ${f3(r.rms)} V. Change: ${change}.`,
    };
  }
  let note = `RMS ${f3(v.rms)} V, 1× ${f3(v.order_1x)} V, 2× ${f3(v.order_2x)} V, excess kurtosis ${v.kurtosis.toFixed(2)} (normal ≈ 0); ${refTxt}: RMS ${f3(r.rms)}, 1× ${f3(r.order_1x)}, 2× ${f3(r.order_2x)}, excess kurtosis ${r.kurtosis.toFixed(2)}. Largest change: ${change}.`;
  if (v.env_bpfo1 != null) note += ` Envelope lines (ratio to the envelope median): BPFO ${f3(v.env_bpfo1)}, BPFI ${f3(v.env_bpfi1)}, 2×BSF ${f3(v.env_bsf2)}, FTF ${f3(v.env_ftf1)} (reference ${f3(r.env_bpfo1)}, ${f3(r.env_bpfi1)}, ${f3(r.env_bsf2)}, ${f3(r.env_ftf1)}).`;
  return {
    stats: [{ k: "RMS", v: f3(v.rms) }, { k: "1×", v: f3(v.order_1x) }, { k: "Change", v: row.change_metric ? changeText(row) : "—", c: row.exceeds ? K.limeT : K.ink },
      { k: "Excess kurt.", v: v.kurtosis.toFixed(2) }],
    note,
  };
}

// "↑ ×2.1" / "↓ ×3.3": the direction and the change factor max(ratio, 1/ratio).
export function changeText(row) {
  return `${row.direction === "up" ? "↑" : "↓"} ×${row.change_factor.toFixed(1)}`;
}

// What was observed on this part's channels, in plain words (App-3 part card).
function observed(part, record, channels) {
  if (!record) return "No recording investigated yet.";
  if (part.ch.includes("tacho")) return speedText(record.signal_quality.shaft_speed_hz) + ".";
  if (!channels) return "Readings are loading.";
  const rows = channels.channels.filter((c) => part.ch.includes(c.channel) && c.change_metric);
  if (!rows.length) return "No comparable reading.";
  const r = rows.reduce((a, b) => ((b.beyond_band || 0) > (a.beyond_band || 0) ? b : a));
  return `${MEASURE_LABEL[r.change_metric] || r.change_metric} on the ${CHANNEL_LABEL[r.channel].toLowerCase()}: ${changeText(r)} `
    + `compared with a healthy example — ${r.exceeds ? "beyond typical healthy variation (different from a healthy example, not a confirmed fault)"
      : "within typical healthy variation"}.`;
}

export default function Inspector({ part, status, chId, onChannel, view, onView, record, channels, signals, onClose, maxHeight, initialShowSignal }) {
  const [showSignal, setShowSignal] = useState(!!initialShowSignal);
  const chips = part.ch.length > 1 ? part.ch : [];
  const hasEnv = !!record?.physics?.envelope_spectrum?.channels?.[chId];
  const views = [["spectrum", "Spectrum"], ...(hasEnv ? [["envelope", "Envelope"]] : []), ["waveform", "Waveform"]];
  const v = view === "envelope" && !hasEnv ? "spectrum" : view;
  const chart = record ? chartFor(v, chId, record, signals) : null;
  const st = record ? statsFor(chId, record, channels) : null;
  return (
    <div role="dialog" aria-label={part.full || part.label} className="glass-strong"
      style={{ width: 316, maxHeight: maxHeight || "calc(100vh - 340px)", overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, padding: "16px 16px 14px", borderRadius: 18 }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
          <span className="mono kicker">{part.kind === "sensor" ? "Sensor" : "Component"}</span>
          <span style={{ fontSize: 20, letterSpacing: "-0.012em", lineHeight: 1.2 }}>{part.full || part.label}</span>
        </div>
        <button onClick={onClose} aria-label="Close details" className="icon-btn">×</button>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: status.color }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: status.dot }} />{status.text}
      </div>
      <div className="card-row"><span className="mono kicker">{part.kind === "sensor" ? "What it measures" : "What this part does"}</span>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.5, color: K.ink2 }}>{part.desc}</p></div>
      <div className="card-row"><span className="mono kicker">What was observed</span>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.5 }}>{observed(part, record, channels)}</p></div>
      <button className="ctl" style={{ pointerEvents: "auto", alignSelf: "flex-start" }} onClick={() => setShowSignal(!showSignal)} aria-expanded={showSignal}>
        {showSignal ? "Hide signal" : "View signal"}
      </button>
      {!showSignal ? null : record && chart ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 10, borderTop: "1px solid oklch(0.23 0.006 260 / 0.08)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            {chips.map((c) => {
              const on = c === chId;
              return (
                <button key={c} onClick={() => onChannel(c)} className="chip"
                  style={{ borderColor: on ? al(K.violet, 0.5) : "oklch(0.23 0.006 260 / 0.12)", background: on ? K.white : "transparent", color: on ? K.ink : K.ink2 }}>
                  {CH[c].dir || CH[c].name}
                </button>
              );
            })}
            <div className="seg" style={{ marginLeft: "auto" }}>
              {views.map(([k, lab]) => (
                <button key={k} onClick={() => onView(k)} className={k === v ? "on" : ""}>{lab}</button>
              ))}
            </div>
          </div>
          <div className="chart" style={v === "envelope" ? { marginTop: 34 } : undefined}>
            {chart.el}
            {chart.markers.map((m, i) => (
              <span key={m.label + m.left}>
                <span style={{ position: "absolute", left: m.left, top: 0, bottom: 0, borderLeft: "1px dashed " + m.color, opacity: 0.8 }} />
                <span className="mono" style={{ position: "absolute", left: m.left, top: v === "envelope" ? -14 - (i % 3) * 10 : -14, transform: "translateX(-50%)", fontSize: 9.5, color: m.color, whiteSpace: "nowrap" }}>{m.label}</span>
              </span>
            ))}
            {chart.ticks.map((t) => (
              <span key={t.label} className="mono" style={{ position: "absolute", left: t.left, bottom: -16, transform: "translateX(-50%)", fontSize: 9.5, color: K.ink3 }}>{t.label}</span>
            ))}
            <span className="mono" style={{ position: "absolute", left: -42, top: -5, width: 38, textAlign: "right", fontSize: 9.5, color: K.ink3 }}>{chart.ymax}</span>
          </div>
          <span style={{ fontSize: 11, color: "oklch(0.48 0.008 260)" }}>{chart.caption}</span>
          {st.stats.length > 0 && (
            <div className="stat-grid">
              {st.stats.map((s) => (
                <div key={s.k}><span>{s.k}</span><span className="mono" style={{ color: s.c || K.ink }}>{s.v}</span></div>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
            <span className="tag" style={{ color: K.ink, background: "oklch(0.23 0.006 260 / 0.07)" }}>Measured</span>
            <span style={{ fontSize: 12.5, lineHeight: 1.45, color: "oklch(0.32 0.007 260)" }}>{st.note}</span>
          </div>
        </div>
      ) : (
        <p style={{ margin: 0, fontSize: 12.5, color: "oklch(0.48 0.008 260)" }}>
          {record ? "Readings are loading." : `Run an investigation to see readings for this ${part.kind === "sensor" ? "sensor" : "component"}.`}
        </p>
      )}
      {showSignal && <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "9px 11px", borderRadius: 10, border: "1px dashed oklch(0.23 0.006 260 / 0.2)" }}>
        <span className="mono" style={{ fontSize: 9.5, letterSpacing: "0.08em", textTransform: "uppercase", color: K.amberT }}>
          {part.kind === "sensor" ? (part.pos === "approx" ? "Approximate position" : "Illustrative position") : "Illustrative geometry"}
        </span>
        <span style={{ fontSize: 11.5, lineHeight: 1.4, color: K.ink2 }}>
          {part.kind === "sensor" ? POS[part.pos] : "Shape and position approximated from the reference photo. Not a CAD model."}
        </span>
      </div>}
      <button disabled title={"Ask about this part — " + "available when the investigation assistant is connected"} className="ghost-btn" style={{ height: 36 }}>
        Ask about this part · not connected
      </button>
    </div>
  );
}
