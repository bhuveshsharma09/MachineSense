import { useEffect, useState } from "react";
import { K } from "../lib/theme.js";
import { CHANNELS, PARTS, cleanText, locations, qualityFlags, recordingTitle, unusualSensors } from "../lib/model.js";
import { Diagnosis, Report } from "./Aside.jsx";
import { FALLBACK, markerFor } from "./Scene.jsx";
import { changeText } from "./Inspector.jsx";
import { Logo, groupRecordings } from "./TopBar.jsx";
import NameModal from "./NameModal.jsx";
import { human } from "../lib/model.js";

// Below 1024 px (App-2a.3): a stacked, readable layout with no horizontal scroll and no 3D — top bar, diagnosis card
// (with warnings), report, the evidence strip as a list and a static rig image with markers.
function StaticRig({ state }) {
  const [anchors, setAnchors] = useState({});
  useEffect(() => {
    fetch(FALLBACK.anchors).then((r) => r.json()).then(setAnchors).catch(() => setAnchors({}));
  }, []);
  const marks = PARTS.map((h) => markerFor(h, state)).filter((m) => anchors[m.id]);
  const tagged = PARTS.map((h) => markerFor(h, state)).filter((m) => m.tag);
  return (
    <div>
      <div className="m-rig">
        <img src={FALLBACK.img} alt="Rendered frame of the illustrative 3D rig" />
        {marks.map((m) => (
          <span key={m.id} className="m-dot" title={m.label + (m.tag ? " · " + m.tag : "")}
            style={{ left: anchors[m.id][0] * 100 + "%", top: anchors[m.id][1] * 100 + "%", width: m.size, height: m.size,
              background: m.fill, border: m.border, boxShadow: m.ring ? "0 0 0 4px " + m.ring.bg : undefined }} />
        ))}
      </div>
      {tagged.length > 0 && (
        <ul className="m-list">
          {tagged.map((m) => (
            <li key={m.id}><span className="m-dot-inline" style={{ background: m.fill, border: m.border }} />
              <span><b>{m.label}</b> · <span style={{ color: m.tagColor }}>{m.tag}</span></span></li>
          ))}
        </ul>
      )}
      <p className="m-note">Illustrative geometry (a rendered frame of the 3D rig). Orange is a visual highlight, not a measured temperature.</p>
    </div>
  );
}

function EvidenceList({ record, channels }) {
  if (!record) return <p className="m-note">Measured values appear after an investigation.</p>;
  if (!channels) return <p className="m-note">Loading measured values…</p>;
  const rows = Object.fromEntries(channels.channels.map((c) => [c.channel, c]));
  const marked = new Set(channels.marked || []);
  return (
    <>
      <p className="m-note">vs {channels.reference.rec_id} (healthy training, {channels.reference.speed_hz.toFixed(1)} Hz)
        {channels.healthy_note ? ` · ${channels.healthy_note}` : ""} · descriptive, not a calibrated anomaly test</p>
      <ul className="m-list">
        {CHANNELS.map((c) => {
          const r = rows[c.id];
          const v = c.id === "tacho" ? `${record.signal_quality.shaft_speed_hz.toFixed(1)} Hz` : r?.change_metric ? changeText(r) : "—";
          return (
            <li key={c.id} style={{ color: marked.has(c.id) ? K.limeT : undefined }}>
              <span>{c.name}</span>
              <span className="mono">{v}{r?.change_metric ? ` ${r.change_metric.replace(/_/g, " ")}` : ""}{r?.band ? ` (typical ×${r.band.toFixed(1)})` : ""}</span>
            </li>
          );
        })}
      </ul>
    </>
  );
}

export default function MobileLayout(props) {
  const { recordings, showcase, selection, upload, status, error, record, channels, report, limitations, userName, needsName,
    onSaveName, onPick, onFile, onClearUpload, onStart, runningFor } = props;
  const flags = qualityFlags(record);
  const loc = locations(record);
  const state = { sel: null, hover: null, fault: loc.fault, cand: loc.cand, unusual: unusualSensors(channels, flags), flags, labelsAll: false };
  const running = status === "running";
  const blocked = upload?.stored && upload.canRerun === false;
  const canStart = !!(selection || upload) && !running && !blocked;
  const rec = selection ? recordings?.find((r) => r.rec_id === selection.rec_id) : null;
  const groups = groupRecordings(recordings);
  return (
    <div className="ms-mobile">
      <header className="m-card m-top">
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}><Logo /><b style={{ fontWeight: 500 }}>MachineSense</b>
          <span className="pill-outline mono" style={{ marginLeft: "auto" }}>Research demo</span></div>
        <select value={selection?.rec_id || ""} onChange={(e) => onPick(e.target.value)} aria-label="Development recording" className="rec-select">
          <option value="" disabled>Select a development recording…</option>
          {showcase?.length > 0 && (
            <optgroup label="Showcase">{showcase.map((s) => <option key={"sc" + s.state} value={s.rec_id}>{s.state} · {s.rec_id}</option>)}</optgroup>
          )}
          {Object.entries(groups).map(([cls, rs]) => (
            <optgroup key={cls} label={human(cls)}>{rs.map((r) => <option key={r.rec_id} value={r.rec_id}>{r.rec_id}</option>)}</optgroup>
          ))}
        </select>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <label className="upload-label" style={{ flex: "1 1 auto", justifyContent: "center" }}>Upload CSV
            <input type="file" accept=".csv,text/csv" aria-label="Upload a CSV recording"
              onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) onFile(f); }}
              style={{ position: "absolute", inset: 0, opacity: 0, width: "100%" }} />
          </label>
          <button onClick={onStart} disabled={!canStart} className="start-btn" title={blocked ? "re-upload the file to run again" : undefined}
            style={{ flex: "1 1 auto", justifyContent: "center", background: canStart || running ? K.ink : "oklch(0.23 0.006 260 / 0.1)", color: canStart || running ? K.white : K.ink3 }}>
            {running && <span className="spinner" />}{running ? "Investigating…" : record || upload?.stored ? "Run again" : "Start investigation"}
          </button>
        </div>
        {upload && (
          <div className="m-note" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span style={{ overflowWrap: "anywhere" }}>{upload.name}</span>
            <button onClick={onClearUpload} aria-label="Remove uploaded file" style={{ border: 0, background: "transparent" }}>×</button>
          </div>
        )}
        {blocked && <p className="m-note" style={{ color: K.amberT }}>{upload.note || "re-upload the file to run again"}</p>}
        <p className="m-note">Best viewed on a desktop screen.</p>
      </header>
      {rec && (
        <section className="m-card">
          <span className="mono kicker">Development recording · used for {rec.used_for}</span>
          <div style={{ fontSize: 18, overflowWrap: "anywhere" }}>{rec.rec_id}</div>
          <div className="m-note">Dataset label: {recordingTitle(rec)}</div>
        </section>
      )}
      <section className="m-card"><span className="mono kicker">Diagnosis</span>
        <Diagnosis record={record} status={status} error={error} selection={selection} assessment={report?.assessment} runningFor={runningFor} />
      </section>
      <section className="m-card"><span className="mono kicker">Report</span><Report status={status} report={report} /></section>
      <section className="m-card"><span className="mono kicker">Measured change vs healthy reference</span>
        <EvidenceList record={record} channels={channels} /></section>
      <section className="m-card"><span className="mono kicker">Rotor test rig</span><StaticRig state={state} /></section>
      <section className="m-card m-note">
        <span className="mono kicker">Limitations</span>
        <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>{(limitations || []).map((x) => <li key={x}>{cleanText(x)}</li>)}</ul>
      </section>
      {needsName && <NameModal initial={userName} onSave={onSaveName} />}
    </div>
  );
}
