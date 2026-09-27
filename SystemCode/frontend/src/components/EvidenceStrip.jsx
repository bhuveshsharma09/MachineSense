import { K, al } from "../lib/theme.js";
import { CHANNELS, nf, shownWarnings } from "../lib/model.js";
import { Svg, linePath, rangeMax } from "./Charts.jsx";
import { changeText } from "./Inspector.jsx";

// Bottom strip: recording facts and one tile per channel with its measured change vs the healthy training reference.
// The bar's tick is the typical healthy variation of that channel's metric; lime = beyond it (at most 2 channels).
export default function EvidenceStrip({ record, channels, signals, flags, selCh, onTile, facts, upload }) {
  const rows = Object.fromEntries((channels?.channels || []).map((c) => [c.channel, c]));
  const marked = new Set(channels?.marked || []);
  const flagged = new Set(Object.keys(flags || {}));
  const note = !record ? (upload ? "Available after a successful upload" : "Select a recording to begin")
    : !channels ? "Loading measured values…"
    : `vs ${channels.reference.rec_id} (healthy training, ${channels.reference.speed_hz.toFixed(1)} Hz) · tick = typical healthy variation${channels.healthy_note ? " · " + channels.healthy_note : ""} · descriptive, not a calibrated anomaly test`;
  return (
    <section aria-label="Evidence" className="glass strip" style={{ position: "absolute", left: 16, bottom: 16, zIndex: 4, display: "flex", gap: 14, padding: "12px 14px", borderRadius: 18 }}>
      <div className="facts" style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
        <span className="mono kicker">Recording details</span>
        {facts.map((f) => (
          <div key={f.k} style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, paddingBottom: 4, borderBottom: "1px solid oklch(0.23 0.006 260 / 0.07)" }}>
            <span style={{ fontSize: 12, color: "oklch(0.45 0.008 260)" }}>{f.k}</span>
            <span className="fact-v" style={{ color: f.color || K.ink, whiteSpace: "nowrap" }}>{f.v}</span>
          </div>
        ))}
      </div>
      <span style={{ width: 1, background: "oklch(0.23 0.006 260 / 0.08)" }} />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline", minWidth: 0 }}>
          <span className="mono kicker" style={{ whiteSpace: "nowrap" }}>Measured change vs healthy reference</span>
          <span style={{ fontSize: 11, color: "oklch(0.48 0.008 260)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", minWidth: 0 }} title={note}>{note}</span>
        </div>
        <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: "repeat(8, minmax(0,1fr))", gap: 5 }}>
          {CHANNELS.map((c) => {
            const row = rows[c.id], s = signals?.channels?.[c.id], avail = !!record;
            const lime = marked.has(c.id), bad = flagged.has(c.sensor) && (c.id === "tacho" || c.id === "mic" || (record?.signal_quality.clipping_channels || []).includes(c.id));
            const score = c.id === "tacho" ? (record ? record.signal_quality.shaft_speed_hz.toFixed(1) + " Hz" : "—")
              : row?.change_metric ? changeText(row) : "—";
            let el = null;
            if (s && signals) {
              const ymax = rangeMax(signals.orders, s.spectrum, 0.5, 8) * 1.1;
              const p = linePath(signals.orders, s.spectrum, 0, 8, ymax);
              el = <Svg paths={[{ d: p.area, fill: lime ? al(K.lime, 0.35) : bad ? al(K.amber, 0.25) : "oklch(0.23 0.006 260 / 0.07)" },
                { d: p.line, stroke: lime ? K.limeT : bad ? K.amberT : "oklch(0.23 0.006 260 / 0.7)", w: 1 }]} />;
            }
            const w = row?.beyond_band ? Math.min(1, row.beyond_band / 2) : 0;
            const selected = selCh === c.id;
            return (
              <button key={c.id} disabled={!avail} onClick={() => onTile(c)} className="tile"
                title={row?.change_metric ? `${c.name}: ${row.change_metric.replace(/_/g, " ")} ${changeText(row)} vs the healthy reference; typical healthy variation ×${row.band.toFixed(1)}${row.exceeds ? (channels.healthy_note ? ` (beyond it; ${channels.healthy_note})` : " (beyond it)") : ""}. Note: ${channels.multiplicity_note}.` : c.name}
                aria-label={`${c.name}${row?.change_metric ? `, ${row.change_metric.replace(/_/g, " ")} ${changeText(row)} of the healthy reference${row.exceeds ? ", beyond typical healthy variation" : ""}` : ""}`}
                style={{ borderColor: selected ? al(K.violet, 0.65) : "oklch(0.23 0.006 260 / 0.08)", background: selected ? "oklch(1 0 0 / 0.85)" : "oklch(1 0 0 / 0.35)",
                  boxShadow: selected ? "0 0 0 3px " + al(K.violet, 0.12) : "none", cursor: avail ? "pointer" : "default" }}>
                <span className="tile-name">{c.short}</span>
                <span className="mono tile-v" style={{ color: bad ? K.amberT : lime ? K.limeT : K.ink3 }}>{score}</span>
                <span style={{ position: "relative", flex: 1, minHeight: 18, display: "block" }}>{el}</span>
                <span style={{ height: 3, borderRadius: 2, background: "oklch(0.23 0.006 260 / 0.08)", position: "relative", display: "block" }}>
                  <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: (bad ? 100 : w * 100) + "%", borderRadius: 2, background: bad ? K.amber : lime ? K.lime : "oklch(0.23 0.006 260 / 0.35)" }} />
                  {c.id !== "tacho" && <span style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "oklch(0.23 0.006 260 / 0.35)" }} />}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}

export function factsFor(record, upload, error) {
  if (record) {
    const q = record.signal_quality, n = shownWarnings(record).length;
    return [
      { k: "Duration", v: "5.00 s · 50 kHz" },
      { k: "Shaft speed", v: `${q.shaft_speed_hz.toFixed(1)} Hz · ${nf(60 * q.shaft_speed_hz)} rpm` },
      { k: "Signal quality", v: n ? `${n} warning${n > 1 ? "s" : ""}` : "No warnings", color: n ? K.amberT : K.ink },
    ];
  }
  if (upload) {
    return [
      { k: "File", v: upload.name.length > 18 ? upload.name.slice(0, 17) + "…" : upload.name },
      { k: "Size", v: upload.stored ? "stored case" : upload.size >= 1e6 ? (upload.size / 1e6).toFixed(1) + " MB" : (upload.size / 1e3).toFixed(1) + " kB" },
      { k: "File check", v: error ? (error.status === 403 ? "Refused" : "Not accepted") : "Not run yet", color: error ? K.amberT : K.ink3 },
    ];
  }
  return [{ k: "Duration", v: "—", color: K.ink3 }, { k: "Shaft speed", v: "—", color: K.ink3 }, { k: "Signal quality", v: "—", color: K.ink3 }];
}
