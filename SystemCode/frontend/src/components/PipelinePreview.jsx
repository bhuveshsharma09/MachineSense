import { K, al } from "../lib/theme.js";
import { CHANNEL_LABEL } from "../lib/mechanic.js";
import { Svg, W, H, bandPath, linePath, rangeMax } from "./Charts.jsx";

// App-4.4: signal previews on the pipeline page (case mode only). Two sources, kept visibly apart:
//  - GET /api/cases/{id}/signals: the cleaned waveform and order spectrum, COMPUTED ON DEMAND for this view; never part
//    of the saved diagnosis. Drawn in a dashed amber frame labelled "computed now".
//  - physics.envelope_spectrum of the saved evidence record (GET /api/cases/{id}): SAVED EVIDENCE, plain frame.
// Nothing is fetched until the user asks in the drawer; a preview is evidence about the signal, not about fault
// location (no predicted fault is drawn; the default channel does not depend on the verdict).
export const PREVIEW_BLOCKS = { preparation: "waveform", order_spectrogram: "spectrum", evidence_record: "envelope" };
export const THUMB_EXTRA = 110;               // extra block width for the thumbnail slot (layout extraWidth)
export const DEFAULT_CHANNEL = "under_radial";
const MAXO = 8, ENV_MAXO = 12;
const COMPUTED = "Computed now for this view · not part of the saved diagnosis";
const SAVED = "Saved evidence · part of the evidence record";
export const sourceOf = (kind) => (kind === "envelope" ? "saved" : "computed");

// the chart for one kind and channel, from the fetched response (signals) or the saved record; null if not loaded
export function chart(kind, ch, sig, saved) {
  if (kind === "envelope") {
    const env = saved?.physics?.envelope_spectrum;
    const ys = env?.channels?.[ch];
    if (!ys) return null;
    const ymax = rangeMax(env.orders, ys, 0.3, ENV_MAXO) * 1.15;
    const p = linePath(env.orders, ys, 0, ENV_MAXO, ymax);
    const cur = saved.physics.cursors_orders || {};
    // the four bearing lines at the measured speed, always all four (a physics reference, not a finding)
    const lines = [["bpfo", 0, "BPFO"], ["bpfi", 1, "BPFI"], ["bsf", 1, "2×BSF"], ["ftf", 0, "FTF"]]
      .filter(([k, i]) => cur[k]?.[i] !== undefined).map(([k, i, lab]) => ({ o: cur[k][i], lab }));
    return { paths: [{ d: p.area, fill: al(K.violet, 0.1) }, { d: p.line, stroke: K.violet, w: 1.3 }],
      cursors: lines.map((l) => ({ x: l.o / ENV_MAXO, lab: l.lab })), xmax: ENV_MAXO,
      xlabel: `order (multiples of the measured shaft speed) · envelope band ${env.band_hz.join("–")} Hz`, note: env.note };
  }
  const s = sig?.channels?.[ch];
  if (!s) return null;
  if (kind === "waveform") {
    const w = s.waveform, m = Math.max(...w.max.map(Math.abs), ...w.min.map(Math.abs)) * 1.15 || 1;
    return { paths: [{ d: bandPath(w.min, w.max, m), fill: al(K.ink3, 0.35), stroke: K.ink2, w: 0.6 }], cursors: [],
      xlabel: `time (s) from ${w.t0_s} s, ${w.max.length} buckets of ${w.dt_s} s (min / max per bucket) · amplitude: the response states no unit`,
      note: sig.note };
  }
  const ymax = rangeMax(sig.orders, s.spectrum, 0.5, MAXO) * 1.15;
  const p = linePath(sig.orders, s.spectrum, 0, MAXO, ymax);
  const shaft = (sig.cursors_orders?.shaft || []).filter((o) => o <= MAXO);
  return { paths: [{ d: p.area, fill: al(K.ink3, 0.18) }, { d: p.line, stroke: K.ink2, w: 1.2 }],
    cursors: shaft.map((o) => ({ x: o / MAXO, lab: `${o}×` })), xmax: MAXO,
    xlabel: `order (multiples of the measured shaft speed, 1× = ${sig.shaft_speed_hz.toFixed(2)} Hz); shaft-order cursors from the response · amplitude: the response states no unit`,
    note: sig.note };
}

function Plot({ c, small }) {
  const paths = [...c.paths, ...c.cursors.map((m) => ({ d: `M${(m.x * W).toFixed(1)},0L${(m.x * W).toFixed(1)},${H}`, stroke: al(K.ink3, 0.6), w: 0.8, dash: "3 3" }))];
  return (
    <div className="pl-plot" style={{ height: small ? 30 : 150 }}>
      <Svg paths={paths} />
      {!small && c.cursors.map((m, i) => <span key={i} className="pl-cursor" style={{ left: `${m.x * 100}%` }}>{m.lab}</span>)}
    </div>
  );
}

// the thumbnail slot on the block
export function Thumb({ kind, st }) {
  const src = sourceOf(kind);
  const load = src === "saved" ? st.saved : st.sig;
  const c = load.status === "ok" ? chart(kind, DEFAULT_CHANNEL, st.sig.data, st.saved.data) : null;
  let body;
  if (src === "computed" && st.unavailable) body = <span className="pl-thumb-msg">Signal not available: {st.unavailable}</span>;
  else if (load.status === "error") body = <span className="pl-thumb-msg">Preview not available: {load.error}</span>;
  else if (load.status === "loading") body = <span className="pl-thumb-msg">Loading the preview</span>;
  else if (c) body = <Plot c={c} small />;
  else body = <span className="pl-thumb-msg">Preview: open this step to load it</span>;
  return (
    <div className={`pl-prev pl-thumb is-${src}`} data-preview={src} title={src === "saved" ? SAVED : COMPUTED}>
      <span className="pl-prev-badge">{src === "saved" ? "saved" : "computed now"}</span>
      {body}
    </div>
  );
}

// the drawer section: explicit load button, channel choice, full-size chart with its source label
export function PreviewSection({ kind, st, channel, onChannel, onLoad }) {
  const src = sourceOf(kind);
  const load = src === "saved" ? st.saved : st.sig;
  const channels = kind === "envelope" ? Object.keys(st.saved.data?.physics?.envelope_spectrum?.channels || {})
    : Object.keys(st.sig.data?.channels || {});
  const ch = channels.includes(channel) ? channel : channels.includes(DEFAULT_CHANNEL) ? DEFAULT_CHANNEL : channels[0];
  const c = load.status === "ok" ? chart(kind, ch, st.sig.data, st.saved.data) : null;
  const title = kind === "waveform" ? "Cleaned waveform" : kind === "spectrum" ? "Order spectrum" : "Envelope spectrum";
  return (
    <section className={`pl-prev pl-prev-full is-${src}`} data-preview={src}>
      <div className="pl-prev-head"><h3 className="mono kicker">{title}</h3><span className="pl-prev-badge">{src === "saved" ? SAVED : COMPUTED}</span></div>
      {src === "computed" && st.unavailable ? <p className="pl-thumb-msg">Signal not available: {st.unavailable}</p>
        : load.status === "error" ? <p className="pl-thumb-msg">Preview not available: {load.error}</p>
        : load.status === "loading" ? <p className="pl-thumb-msg">Loading the preview.</p>
        : load.status !== "ok" ? (
          <button className="ctl pl-load" onClick={onLoad}>
            {src === "saved" ? "Load the saved envelope spectrum (from the evidence record)" : "Load signal preview (computed now from the stored recording)"}
          </button>)
        : (
          <>
            <div role="radiogroup" aria-label="Channel" className="seg pl-channels">
              {channels.map((k) => <button key={k} role="radio" aria-checked={k === ch} className={k === ch ? "on" : ""} onClick={() => onChannel(k)}>{CHANNEL_LABEL[k] || k}</button>)}
            </div>
            {c ? <Plot c={c} /> : <p className="pl-thumb-msg">No data for this channel in the response.</p>}
            {c && <p className="pl-muted">{c.xlabel}</p>}
            {c?.note && <p className="pl-muted">{c.note}</p>}
          </>
        )}
    </section>
  );
}
