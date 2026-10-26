import { createContext, useContext, useEffect, useRef, useState } from "react";
import CaseNotes from "./CaseNotes.jsx";
import { useCaseData } from "./Dashboard.jsx";
import { AssistantView } from "./Assistant.jsx";
import "../redesign.css";

// App-9 addendum A: ONE assistant panel, mounted by App above the dashboard / pipeline routes, so pinning keeps the same
// React instance (draft, shown turns, in-flight request) across Mechanic / Developer and the same case's Pipeline. Its
// state comes from lib/assistantDock.js; pages read it through DockContext to open the panel and to make room for it.
export const DockContext = createContext(null);
export const useDock = () => useContext(DockContext);

export const SESSION_NOTE = "Closing ends this session. Saved questions about this recording stay available.";
export const MIN_PANEL_WIDTH = 320;
export const MAX_PANEL_WIDTH = 900;
const MIN_PAGE_WIDTH = 400;

export function clampPanelWidth(width, viewportWidth) {
  const max = Math.max(MIN_PANEL_WIDTH, Math.min(MAX_PANEL_WIDTH, viewportWidth - MIN_PAGE_WIDTH));
  return Math.max(MIN_PANEL_WIDTH, Math.min(max, Math.round(width)));
}

export default function AssistantDock({ state, dispatch, triggerRef, width = 420, onWidthChange = () => {}, onShowEvidence, AssistantComponent = AssistantView, initialData = null, initialTab = "ask", notesProps = {} }) {
  const { caseId, open, minimized, pinned, session } = state;
  if (!open || !caseId) return null;
  return <DockPanel key={`${caseId}:${session}`} caseId={caseId} minimized={minimized} pinned={pinned} dispatch={dispatch}
    triggerRef={triggerRef} width={width} onWidthChange={onWidthChange} onShowEvidence={onShowEvidence} AssistantComponent={AssistantComponent}
    initialData={initialData} initialTab={initialTab} notesProps={notesProps} />;
}

function DockPanel({ caseId, minimized, pinned, dispatch, triggerRef, width, onWidthChange, onShowEvidence, AssistantComponent, initialData, initialTab, notesProps }) {
  const [tab, setTab] = useState(initialTab);  // both views stay mounted, so switching keeps the draft
  const [loaded] = useCaseData(caseId, { withExplanation: false, skip: !!initialData });
  const st = initialData || loaded;
  const panel = useRef(null), restoreBtn = useRef(null), minBtn = useRef(null);
  const drag = useRef(null);
  const wasMin = useRef(minimized);
  // focus: the question box on open / restore; the restore button on minimize
  useEffect(() => {
    if (minimized) { if (!wasMin.current) restoreBtn.current?.focus(); }
    else requestAnimationFrame(() => panel.current?.querySelector('input[aria-label="Ask about this recording"]')?.focus());
    wasMin.current = minimized;
  }, [minimized, st.status]);
  useEffect(() => () => document.body.classList.remove("as-resizing"), []);
  const maxWidth = clampPanelWidth(MAX_PANEL_WIDTH, typeof window === "undefined" ? 1440 : window.innerWidth);
  const resizeTo = (value) => onWidthChange(clampPanelWidth(value, window.innerWidth));
  const stopResize = () => { drag.current = null; document.body.classList.remove("as-resizing"); };
  const startResize = (e) => {
    if (e.button !== 0) return;
    drag.current = { pointerId: e.pointerId, x: e.clientX, width: panel.current?.getBoundingClientRect().width || width };
    e.currentTarget.setPointerCapture(e.pointerId);
    document.body.classList.add("as-resizing");
    e.preventDefault();
  };
  const moveResize = (e) => {
    if (drag.current?.pointerId === e.pointerId) resizeTo(drag.current.width + drag.current.x - e.clientX);
  };
  const endResize = (e) => {
    if (drag.current?.pointerId !== e.pointerId) return;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    stopResize();
  };
  const keyResize = (e) => {
    const step = e.shiftKey ? 60 : 20;
    const next = e.key === "ArrowLeft" ? width + step : e.key === "ArrowRight" ? width - step
      : e.key === "Home" ? MIN_PANEL_WIDTH : e.key === "End" ? maxWidth : null;
    if (next == null) return;
    e.preventDefault();
    resizeTo(next);
  };
  // Escape inside the panel minimizes it (never navigates away from the case)
  const onKeyDown = (e) => { if (e.key === "Escape") { e.stopPropagation(); dispatch({ type: "minimize" }); } };
  const close = () => {
    const back = triggerRef?.current;
    dispatch({ type: "close" });
    requestAnimationFrame(() => { if (back?.isConnected) back.focus(); });
  };
  return <>
    <aside ref={panel} id="assistant-panel" className="as-panel" role="complementary" aria-label="Investigation assistant"
      hidden={minimized} onKeyDown={onKeyDown}>
      <div className="as-resize-handle" role="separator" aria-label="Resize assistant panel" aria-orientation="vertical"
        aria-valuemin={MIN_PANEL_WIDTH} aria-valuemax={maxWidth} aria-valuenow={Math.min(width, maxWidth)} tabIndex={0}
        title="Drag to resize the assistant panel" onPointerDown={startResize} onPointerMove={moveResize}
        onPointerUp={endResize} onPointerCancel={endResize} onLostPointerCapture={stopResize} onKeyDown={keyResize} />
      <div className="as-panel-head">
        <div className="as-panel-title"><span className="rd-kicker">This investigation</span><h2>Ask the assistant</h2>
          <span className="as-panel-note">{SESSION_NOTE}</span></div>
        <div className="as-panel-ctl" role="group" aria-label="Assistant panel controls">
          <button ref={minBtn} type="button" className="as-ctl" onClick={() => dispatch({ type: "minimize" })} aria-label="Minimize the assistant"
            title="Minimize (keeps your draft and this conversation)">–</button>
          <button type="button" className={"as-ctl as-pin" + (pinned ? " is-on" : "")} aria-pressed={pinned} onClick={() => dispatch({ type: "togglePin" })}
            aria-label={pinned ? "Unpin the assistant" : "Pin the assistant"} title={pinned ? "Pinned: stays open on this case's pages. Click to unpin." : "Pin: keep the assistant open on this case's dashboard and pipeline"}>
            <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 1.5h4l-.6 4.2 2.6 2.3H4l2.6-2.3zM8 8v6.5" fill={pinned ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" /></svg>
            <span>{pinned ? "Pinned" : "Pin"}</span></button>
          <button type="button" className="as-ctl" onClick={close} aria-label="Close the assistant and end this session"
            title={SESSION_NOTE}>×</button>
        </div>
      </div>
      <div className="as-tabs" role="tablist" aria-label="Assistant panel views">
        {[["ask", "Ask"], ["notes", "Notes & status"]].map(([k, lab]) => <button key={k} type="button" role="tab" id={`as-tab-${k}`}
          aria-selected={tab === k} aria-controls={`as-view-${k}`} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{lab}</button>)}
      </div>
      <div id="as-view-ask" role="tabpanel" aria-labelledby="as-tab-ask" className="as-view" hidden={tab !== "ask"}>
        {st.status === "ok" ? <AssistantComponent panel newSession caseId={caseId} record={st.record} report={st.report}
          channels={st.channels} onShowEvidence={onShowEvidence} />
          : <p className="as-panel-state" role="status">{st.status === "error" ? st.message : "Loading the case…"}</p>}
      </div>
      <div id="as-view-notes" role="tabpanel" aria-labelledby="as-tab-notes" className="as-view as-view-notes" hidden={tab !== "notes"}>
        <CaseNotes caseId={caseId} {...notesProps} />
      </div>
    </aside>
    {minimized && <button ref={restoreBtn} type="button" className="as-restore" aria-controls="assistant-panel" aria-expanded="false"
      onClick={() => dispatch({ type: "restore" })} title={pinned ? "Restore the pinned assistant" : "Restore the assistant"}>
      <span className="rd-dot" aria-hidden="true" />Ask the assistant{pinned && <span className="as-restore-pin"> · pinned</span>}</button>}
  </>;
}

// A page's "Ask the assistant" control: opens the one panel for this case (a deep link when no dock is mounted, e.g. in
// server-side render tests).
export function AskButton({ caseId, className = "rd-ask", children }) {
  const dock = useDock();
  const label = children || <><span className="rd-dot" aria-hidden="true" />Ask the assistant</>;
  if (!dock) return <a className={className} href={`#/case/${encodeURIComponent(caseId)}/assistant`}>{label}</a>;
  const st = dock.state;
  if (st.open && st.caseId === caseId && !st.minimized) return null;  // already open: the panel has its own controls
  return <button type="button" className={className} onClick={(e) => dock.open(caseId, e.currentTarget)}
    aria-controls="assistant-panel" aria-expanded={st.open && st.caseId === caseId && !st.minimized}>{label}</button>;
}
