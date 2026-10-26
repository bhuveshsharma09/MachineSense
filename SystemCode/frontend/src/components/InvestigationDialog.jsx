import { useEffect, useRef, useState } from "react";
import { human, recordingTitle } from "../lib/model.js";
import { groupRecordings } from "./TopBar.jsx";

// App-8.6 items 20–21: the recording list's label, a word filter (every word must match the class, setting, speed or id),
// and a showcase entry labelled like the rest of the list with its file id after it
export const recLabel = (r) => `${recordingTitle(r)} · ${r.speed_hz.toFixed(1)} Hz · ${r.used_for}`;
export function filterRecordings(recordings, q) {
  const words = String(q || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return recordings || [];
  return (recordings || []).filter((r) => words.every((w) => `${human(r.class10)} ${recLabel(r)} ${r.rec_id}`.toLowerCase().includes(w)));
}
export function showcaseLabel(s, recordings) {
  const r = (recordings || []).find((x) => x.rec_id === s.rec_id);
  return r ? `${s.state} · ${recordingTitle(r)} · ${r.speed_hz.toFixed(1)} Hz (${s.rec_id})` : `${s.state} · ${s.rec_id}`;
}

export default function InvestigationDialog({ open, recordings, showcase, selection, upload, onPick, onFile,
  onClearUpload, onStart, onClose, status, error, runningFor }) {
  const ref = useRef(null);
  const [q, setQ] = useState("");  // App-8.6 item 20: filter by fault, setting or speed
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === "Escape" && status !== "running") close.current(); };
    window.addEventListener("keydown", onKey);
    ref.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, status]);
  if (!open) return null;
  const words = q.trim().split(/\s+/).filter(Boolean);
  const filtered = filterRecordings(recordings, q);
  const groups = groupRecordings(filtered);
  const showLabel = (s) => showcaseLabel(s, recordings);
  const shown = (showcase || []).filter((s) => !words.length || words.every((w) => showLabel(s).toLowerCase().includes(w.toLowerCase())));
  const nMatch = filtered.length;
  const running = status === "running";
  const blocked = upload?.stored && upload.canRerun === false;
  const canStart = !!(selection || upload) && !running && !blocked;
  return <div className="invest-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget && !running) onClose(); }}>
    <section className="invest-dialog glass-strong" role="dialog" aria-modal="true" aria-labelledby="invest-title" tabIndex={-1} ref={ref}>
      <div className="invest-head"><div><span className="mono kicker">New investigation</span><h2 id="invest-title">Choose a machine reading</h2></div>
        {!running && <button type="button" className="icon-btn" onClick={onClose} aria-label="Close dialog">×</button>}</div>
      {running ? <div className="invest-processing" role="status">
        <div className="invest-spinner" /><h3>Investigating the recording</h3>
        <p>{runningFor || selection?.label || upload?.name}</p>
        <ol><li>Checking the signal</li><li>Preparing features and spectrograms</li><li>Running model checks and building the report</li></ol>
        <div className="progress"><div /></div>
      </div> : <>
        <p className="invest-intro">Select a development recording or upload a five-second CSV. The results will open in the dashboard.</p>
        <label className="invest-field">Find a recording
          <input type="search" className="rec-select invest-filter" value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Filter by fault, setting or speed, e.g. bearing 20g or 35 Hz" aria-label="Filter the recordings" />
          {words.length > 0 && <small className="invest-count">{`${nMatch} recording${nMatch === 1 ? "" : "s"} match`}</small>}
        </label>
        <label className="invest-field">Development recording
          <select value={selection?.rec_id || ""} onChange={(e) => onPick(e.target.value)} aria-label="Development recording" className="rec-select">
            <option value="">Select a recording…</option>
            {shown.length > 0 && <optgroup label="Examples">{shown.map((s) => <option key={s.state + s.rec_id} value={s.rec_id}>{showLabel(s)}</option>)}</optgroup>}
            {Object.entries(groups).map(([cls, rs]) => <optgroup key={cls} label={human(cls) + ` (${rs.length})`}>
              {rs.map((r) => <option key={r.rec_id} value={r.rec_id}>{recLabel(r)}</option>)}
            </optgroup>)}
          </select>
        </label>
        <div className="invest-or">or</div>
        <label className="invest-upload">Upload CSV
          <small>8 numeric columns · 250,000 rows · 5 seconds at 50 kHz</small>
          <input type="file" accept=".csv,text/csv" aria-label="Upload a CSV recording" onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) onFile(f); }} />
        </label>
        {upload && <div className="invest-file"><span>{upload.name}</span><button type="button" onClick={onClearUpload} aria-label="Remove uploaded file">Remove</button></div>}
        {blocked && <p className="invest-error">{upload.note || "Re-upload the file to run again."}</p>}
        {error && <p className="invest-error" role="alert">{error.message}</p>}
        <div className="invest-actions"><button type="button" className="ctl" onClick={onClose}>Cancel</button>
          <button type="button" className="start-btn" disabled={!canStart} onClick={onStart}>Start investigation →</button></div>
      </>}
    </section>
  </div>;
}
