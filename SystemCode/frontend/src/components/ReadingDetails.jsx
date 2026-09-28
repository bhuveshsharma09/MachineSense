import { K } from "../lib/theme.js";
import { CHANNELS } from "../lib/model.js";
import { CHANNEL_LABEL, DIFFERENCE_NOTE, MEASURE_LABEL, REFERENCE_LABEL, speedText } from "../lib/mechanic.js";
import EvidenceStrip from "./EvidenceStrip.jsx";
import { changeText } from "./Inspector.jsx";

// App-3: the bottom strip becomes a floating "Reading details" button that opens a drawer with the recording details,
// per-channel readings (mechanic-friendly labels, technical term in brackets) and the spectrum tiles.
function Icon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M1 12 L4 7 L7 10 L10 3 L13 8 L15 6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function ReadingDetails({ open, onToggle, record, channels, signals, flags, selCh, upload, facts, onTile }) {
  const rows = Object.fromEntries((channels?.channels || []).map((c) => [c.channel, c]));
  return (
    <>
      <button className="reading-btn glass" onClick={onToggle} aria-expanded={open} aria-controls="reading-drawer">
        <Icon /> {open ? "Hide reading details" : "Reading details"}
      </button>
      {open && (
        <section id="reading-drawer" aria-label="Reading details" className="glass reading-drawer">
          <div className="rd-head">
            <span className="mono kicker">Reading details</span>
            <button onClick={onToggle} aria-label="Close reading details" className="icon-btn">×</button>
          </div>
          <div className="rd-body">
            <div className="rd-facts">
              {record ? <span>{speedText(record.signal_quality.shaft_speed_hz)} · 5.00 s recording at 50 kHz (8 channels)</span>
                : <span className="mv-muted">{upload ? "Available after a successful upload." : "Select a recording and start the investigation."}</span>}
            </div>
            {record && channels && (
              <table className="rd-table">
                <thead><tr><th>Sensor channel</th><th>Largest difference</th><th>{REFERENCE_LABEL}</th><th>vs typical healthy variation</th></tr></thead>
                <tbody>
                  {CHANNELS.filter((c) => c.id !== "tacho").map((c) => {
                    const r = rows[c.id];
                    return (
                      <tr key={c.id} className={channels.marked?.includes(c.id) ? "rd-marked" : ""}>
                        <td>{CHANNEL_LABEL[c.id]}</td>
                        <td>{r?.change_metric ? MEASURE_LABEL[r.change_metric] || r.change_metric : "—"}</td>
                        <td className="mono">{r?.change_metric ? changeText(r) : "—"}</td>
                        <td>{r?.change_metric ? (r.exceeds ? "beyond it — different from a healthy example" : "within it") : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            {record && channels && (
              <p className="m-note" style={{ color: K.ink2 }}>
                {DIFFERENCE_NOTE} Reference: {channels.reference.rec_id} (a healthy training recording at {channels.reference.speed_hz.toFixed(1)} Hz).
                {channels.healthy_note ? ` The differences here are ${channels.healthy_note}.` : ""}
              </p>
            )}
            <EvidenceStrip embedded record={record} channels={channels} signals={signals} flags={flags} selCh={selCh} upload={upload}
              facts={facts} onTile={onTile} />
          </div>
        </section>
      )}
    </>
  );
}
