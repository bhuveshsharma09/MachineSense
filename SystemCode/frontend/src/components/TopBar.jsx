import { K } from "../lib/theme.js";

export function Logo({ size = 22 }) {
  return <img src="/icon-64.png" width={size} height={size} alt="" aria-hidden="true" style={{ display: "block", flex: "none" }} />;
}

export function UnseenLink({ style, compact }) {
  return <a className="un-nav" href="#/unseen" style={style}
    title="Research demonstration: a model trained without one fault class meets a recording of that class">
    <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" strokeWidth="1.4" strokeDasharray="3 2" />
      <path d="M8 5.2v3.4M8 10.6v.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
    {compact ? "Unseen fault" : "Explore an unseen fault"}
  </a>;
}

export function shortName(id) {
  if (!id) return "";
  return id.length > 28 ? id.slice(0, 26) + "…" : id;
}

export function groupRecordings(recordings) {
  const g = {};
  for (const r of recordings || []) (g[r.class10] ||= []).push(r);
  return g;
}

export function PipelineButton({ caseId, style }) {
  if (!caseId) return null;
  return <a className="pl-open-btn" href={`#/case/${encodeURIComponent(caseId)}/pipeline`}
    title="How this diagnosis was made: every step, from the saved case" style={style}>
    <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><circle cx="3" cy="3" r="2" fill="currentColor" /><circle cx="13" cy="8" r="2" fill="currentColor" />
      <circle cx="3" cy="13" r="2" fill="currentColor" /><path d="M5 3h3v10H5M8 8h3" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
    Pipeline view
  </a>;
}

export default function TopBar({ onHome, onInvestigate, status, userName, onChangeName, reportPage }) {
  const initials = (userName || "Guest").trim().split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase() || "G";
  return <header className="glass ms-header">
    <a className="ms-brand" href="#/" onClick={onHome} aria-label="MachineSense home"><Logo /><span>MachineSense</span></a>
    <div className="ms-header-spacer" />
    <UnseenLink compact={reportPage} />
    <button type="button" onClick={onInvestigate} className="start-btn ms-header-cta" disabled={status === "running"}>
      {status === "running" && <span className="spinner" />}{status === "running" ? "Investigating…" : reportPage ? "New investigation" : "Start investigation"}
    </button>
    <button onClick={onChangeName} title="Change display name" className="ms-account">
      <span className="ms-avatar">{initials}</span><span>{userName || "Guest"}</span>
    </button>
  </header>;
}
