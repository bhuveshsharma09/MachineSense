import { useEffect, useRef } from "react";
import { human, recordingTitle } from "../lib/model.js";
import { groupRecordings } from "./TopBar.jsx";

export default function InvestigationDialog({ open, recordings, showcase, selection, upload, onPick, onFile,
  onClearUpload, onStart, onClose, status, error, runningFor }) {
  const ref = useRef(null);
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
  const groups = groupRecordings(recordings);
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
        <p className="invest-intro">Select a development recording or upload a five-second CSV. The result will open as a saved report.</p>
        <label className="invest-field">Development recording
          <select value={selection?.rec_id || ""} onChange={(e) => onPick(e.target.value)} aria-label="Development recording" className="rec-select">
            <option value="">Select a recording…</option>
            {showcase?.length > 0 && <optgroup label="Examples">{showcase.map((s) => <option key={s.state + s.rec_id} value={s.rec_id}>{s.state} · {s.rec_id}</option>)}</optgroup>}
            {Object.entries(groups).map(([cls, rs]) => <optgroup key={cls} label={human(cls) + ` (${rs.length})`}>
              {rs.map((r) => <option key={r.rec_id} value={r.rec_id}>{recordingTitle(r)} · {r.speed_hz.toFixed(1)} Hz · {r.used_for}</option>)}
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
