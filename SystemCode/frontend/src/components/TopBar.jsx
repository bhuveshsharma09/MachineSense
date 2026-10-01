import { K } from "../lib/theme.js";
import { human, recordingTitle } from "../lib/model.js";

// The official MachineSense icon (frontend/public/icon-*.png); decorative beside the "MachineSense" wordmark.
export function Logo({ size = 22 }) {
  return <img src="/icon-64.png" width={size} height={size} alt="" aria-hidden="true" style={{ display: "block", flex: "none" }} />;
}

// App-6: the navigation entry to the "Explore an unseen fault" research demonstration page (#/unseen)
export function UnseenLink({ style, compact }) {
  return (
    <a className="un-nav" href="#/unseen" style={style}
      title="Research demonstration: a model trained without one fault class meets a recording of that class">
      <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" strokeWidth="1.4" strokeDasharray="3 2" />
        <path d="M8 5.2v3.4M8 10.6v.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
      {compact ? "Unseen fault" : "Explore an unseen fault"}
    </a>
  );
}

// A short, readable name for the selected recording, kept beside the Start button (App-3).
export function shortName(id) {
  if (!id) return "";
  return id.length > 28 ? id.slice(0, 26) + "…" : id;
}

export function groupRecordings(recordings) {
  const g = {};
  for (const r of recordings || []) (g[r.class10] ||= []).push(r);
  return g;
}

// the way to the pipeline page of the case on screen (App-4.5): a button, shown once a diagnosis is on screen
export function PipelineButton({ caseId, style }) {
  if (!caseId) return null;
  return (
    <a className="pl-open-btn" href={`#/case/${encodeURIComponent(caseId)}/pipeline`} title="How this diagnosis was made: every step, from the saved case" style={style}>
      <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><circle cx="3" cy="3" r="2" fill="currentColor" /><circle cx="13" cy="8" r="2" fill="currentColor" />
        <circle cx="3" cy="13" r="2" fill="currentColor" /><path d="M5 3h3v10H5M8 8h3" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
      Pipeline view
    </a>
  );
}

export default function TopBar({ recordings, showcase, value, onPick, onFile, upload, onClearUpload, onStart, status, hasRecord, userName, onChangeName, runningFor, caseId }) {
  const groups = groupRecordings(recordings);
  const byId = Object.fromEntries((recordings || []).map((r) => [r.rec_id, r]));
  const running = status === "running";
  const blocked = upload?.stored && upload.canRerun === false;  // reopened upload whose stored array is gone
  const canStart = !!(value || upload) && !running && !blocked;
  const initials = (userName || "Guest").trim().split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase() || "G";
  return (
    <header className="glass" style={{ position: "absolute", top: 16, left: 16, right: 16, height: 60, zIndex: 10, display: "flex", alignItems: "center", gap: 14, padding: "0 10px 0 18px", borderRadius: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}><Logo /><span style={{ fontSize: 17, fontWeight: 500, letterSpacing: "-0.01em" }}>MachineSense</span></div>
      <span className="pill-outline mono">Research demo</span>
      <span style={{ width: 1, height: 26, background: "oklch(0.23 0.006 260 / 0.1)" }} />
      <div style={{ position: "relative", width: 340 }}>
        <select value={value || ""} onChange={(e) => onPick(e.target.value)} aria-label="Development recording" className="rec-select">
          <option value="" disabled>Select a development recording…</option>
          {showcase?.length > 0 && (
            <optgroup label="Showcase · real development recordings">
              {showcase.map((s) => (
                <option key={"sc" + s.state} value={s.rec_id}>{s.state} · {s.rec_id}</option>
              ))}
            </optgroup>
          )}
          {Object.entries(groups).map(([cls, rs]) => (
            <optgroup key={cls} label={human(cls) + ` (${rs.length})`}>
              {rs.map((r) => (
                <option key={r.rec_id} value={r.rec_id}>{recordingTitle(r)} · {r.speed_hz.toFixed(1)} Hz · {r.used_for}</option>
              ))}
            </optgroup>
          ))}
        </select>
        <span className="chev" />
      </div>
      <label className="upload-label">
        Upload CSV
        <input type="file" accept=".csv,text/csv" onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) onFile(f); }}
          aria-label="Upload a CSV recording" style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer", width: "100%" }} />
      </label>
      {upload && (
        <span className="mono" style={{ height: 32, display: "inline-flex", alignItems: "center", gap: 4, padding: "0 4px 0 10px", borderRadius: 8, background: "oklch(0.23 0.006 260 / 0.06)", fontSize: 11.5, maxWidth: 200 }}>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{upload.name}</span>
          <button onClick={onClearUpload} aria-label="Remove uploaded file" style={{ width: 26, height: 26, border: 0, background: "transparent", cursor: "pointer", fontSize: 15 }}>×</button>
        </span>
      )}
      {value && byId[value] && !upload && <span style={{ fontSize: 12, color: K.ink3, whiteSpace: "nowrap" }}>used for {byId[value].used_for}</span>}
      <div style={{ flex: 1 }} />
      {running && runningFor && (
        <span className="running-for" title={runningFor}>Diagnosing <b>{runningFor}</b></span>
      )}
      {blocked && <span className="running-for" style={{ color: K.amberT }}>re-upload the file to run again</span>}
      {!running && (value || upload) && (
        <span className="rec-short" title={value || upload?.name}>{shortName(value || upload?.name)}</span>
      )}
      <UnseenLink compact={!!caseId} />
      <PipelineButton caseId={caseId} />
      <button onClick={onStart} disabled={!canStart} className="start-btn"
        title={blocked ? "re-upload the file to run again" : undefined} aria-label={blocked ? "Run again (re-upload the file to run again)" : undefined}
        style={{ background: canStart || running ? K.ink : "oklch(0.23 0.006 260 / 0.1)", color: canStart || running ? K.white : K.ink3, cursor: canStart ? "pointer" : "default" }}>
        {running && <span className="spinner" />}
        {running ? "Investigating…" : hasRecord || upload?.stored ? "Run again" : "Start investigation"}
      </button>
      <span style={{ width: 1, height: 26, background: "oklch(0.23 0.006 260 / 0.1)" }} />
      <button onClick={onChangeName} title="Change display name" style={{ display: "flex", alignItems: "center", gap: 9, height: 40, padding: "0 12px 0 5px", border: 0, background: "transparent", borderRadius: 999, cursor: "pointer" }}>
        <span style={{ width: 30, height: 30, borderRadius: "50%", background: K.ink, color: "oklch(0.98 0.002 85)", display: "grid", placeItems: "center", fontSize: 11.5, fontWeight: 500 }}>{initials}</span>
        <span style={{ fontSize: 13.5, whiteSpace: "nowrap" }}>{userName || "Guest"}</span>
      </button>
    </header>
  );
}
