import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import {
  diagnoseRecording, diagnoseUpload, getActivity, getCase, getCaseMeta, getChannels, getExplanation, getHealth, getRecordings,
  getReport, getShowcase, getSignals, postExplanation, rerunCase,
} from "./api.js";
import { createSession } from "./lib/session.js";
import AssistantDock, { DockContext } from "./components/AssistantDock.jsx";
import { CLOSED, dockReducer, expanded } from "./lib/assistantDock.js";
import Dashboard from "./components/Dashboard.jsx";
import Pipeline from "./components/Pipeline.jsx";
import Unseen from "./components/Unseen.jsx";
import Scene from "./components/Scene.jsx";
import Workspace from "./components/Workspace.jsx";

const NAME_KEY = "ms_display_name";
const readName = () => { try { return localStorage.getItem(NAME_KEY) || ""; } catch { return ""; } };

// Hash routes: #/ ; #/case/<case_id>/dashboard (opens a stored case) ; #/diagnose/<rec_id> (selects and runs a recording) ;
// #/case/<case_id>/pipeline (App-4.2: the pipeline architecture, drawn from that case's trace; no inference) ;
// #/case/<case_id>/assistant (a deep link: opens the assistant panel on the dashboard, then becomes /dashboard; App-9
// addendum A: the panel's state is not in the URL). Bare #/case/<id> links redirect to the dashboard.
// #/unseen (App-6: "Explore an unseen fault", a leave-one-class-out research demonstration).
// Optional query for linking to a view: ?sel=<part>&ch=<channel>&view=spectrum|envelope|waveform&tab=activity|report&static=1
function query() {
  const q = new URLSearchParams(window.location.search);
  const ui = {};
  if (q.get("sel")) ui.sel = q.get("sel");
  if (q.get("ch")) ui.evCh = q.get("ch");
  if (q.get("view")) ui.evView = q.get("view");
  if (q.get("labels")) ui.labels = true;
  if (q.get("exploded")) ui.exploded = true;
  return { ui, tab: q.get("tab"), forceStatic: q.get("static") === "1", capture: q.get("capture") === "1" };
}

// ?capture=1 renders only the 3D scene (no panels, markers invisible) to record the static fallback frame and anchors.
function Capture() {
  const state = { sel: null, hover: null, exploded: false, fault: [], cand: [], unusual: [], flags: {}, labelsAll: false };
  return <div className="ms-root capture"><Scene state={state} onSelect={() => {}} onHover={() => {}} captureMode /></div>;
}

const isUnseen = () => /^#\/unseen\/?$/.test(window.location.hash);
const casePage = () => { const m = /^#\/case\/([^/]+)(?:\/(pipeline|dashboard|assistant))?\/?$/.exec(window.location.hash); return m ? { id: decodeURIComponent(m[1]), page: m[2] || "dashboard" } : null; };
const canonicalizeCaseHash = () => {
  const m = /^#\/case\/([^/]+)(?:\/assistant)?\/?$/.exec(window.location.hash);
  if (m) window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#/case/${m[1]}/dashboard`);
};

export default function App() {
  const capture = useRef(query()).current.capture;
  // the route as read from the URL (an /assistant deep link is reported once, then the URL says /dashboard)
  const [page, setPage] = useState(casePage);
  const [unseen, setUnseen] = useState(isUnseen);
  const [dock, dispatch] = useReducer(dockReducer, CLOSED, (s) => dockReducer(s, { type: "route", route: casePage() }));
  const [dockWidth, setDockWidth] = useState(420);
  const [dashboardFocus, setDashboardFocus] = useState(null);
  const trigger = useRef(null);
  useEffect(() => {
    canonicalizeCaseHash();
    const on = () => {
      const r = casePage();
      dispatch({ type: "route", route: r });
      setDashboardFocus((focus) => r && focus && r.id === focus.caseId && r.page !== "pipeline" ? focus : null);
      canonicalizeCaseHash();
      setPage(r && r.page === "assistant" ? { ...r, page: "dashboard" } : r);
      setUnseen(isUnseen());
      window.scrollTo?.(0, 0);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const ctx = useMemo(() => ({
    state: dock,
    open: (caseId, el) => { trigger.current = el || null; dispatch({ type: "open", caseId }); },
    expanded: (caseId) => expanded(dock, caseId),
  }), [dock]);
  const showEvidence = (caseId, target) => {
    setDashboardFocus({ caseId, target });
    window.location.hash = `#/case/${encodeURIComponent(caseId)}/dashboard`;
  };
  if (capture) return <Capture />;
  let route;
  if (unseen) route = <Unseen key="unseen" />;
  else if (page?.page === "pipeline") route = <Pipeline key={`p:${page.id}`} caseId={page.id} />;
  else if (page) route = <Dashboard key={`d:${page.id}`} caseId={page.id} assistantOpen={expanded(dock, page.id)}
    focusRequest={dashboardFocus?.caseId === page.id ? dashboardFocus : null} />;
  else route = <Main key="main" />;
  // ONE assistant panel above the routes: pinning keeps this instance across the dashboard and the pipeline
  return (
    <DockContext.Provider value={ctx}>
      <div style={{ "--assistant-width": `${dockWidth}px` }}>
        {route}
        <AssistantDock key="assistant-dock" state={dock} dispatch={dispatch} triggerRef={trigger}
          width={dockWidth} onWidthChange={setDockWidth} onShowEvidence={showEvidence} />
      </div>
    </DockContext.Provider>
  );
}

let uploadCounter = 0;

function Main() {
  const init = useRef(query()).current;
  const [health, setHealth] = useState(null);
  const [recordings, setRecordings] = useState([]);
  const [showcase, setShowcase] = useState([]);
  const [name, setName] = useState(readName());
  const [editingName, setEditingName] = useState(false);
  const [s, setS] = useState(null);
  const [tab, setTab] = useState(init.tab || "activity");
  const session = useRef(null);
  if (!session.current) {
    session.current = createSession({
      diagnoseRecording, diagnoseUpload, rerunCase, getCase, getCaseMeta, getActivity, getReport, getChannels, getSignals,
      getExplanation, postExplanation,
      onCase: (id) => { window.location.hash = `#/case/${encodeURIComponent(id)}/dashboard`; return true; },
    }, setS);
  }
  const st = s || session.current.get();

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getRecordings().then(setRecordings).catch(() => {});
    getShowcase().then((x) => setShowcase(x.showcase || [])).catch(() => {});
    const [, kind, id] = window.location.hash.replace(/^#/, "").split("/");
    if (kind === "diagnose" && id) {
      const r = decodeURIComponent(id);
      session.current.select({ kind: "dataset", rec_id: r, label: r });
      session.current.run();
    }
    return () => session.current.abort();
  }, []);

  // tab follows the run: activity while running, report when done (unless a tab was linked)
  useEffect(() => { if (st.status === "running") setTab("activity"); if (st.status === "done") setTab(init.tab || "report"); },
    [st.status, st.record?.case_id]);

  const input = st.input;
  const selection = input?.kind === "dataset" ? input : null;
  const upload = input?.kind === "upload" ? { name: input.file.name, size: input.file.size }
    : input?.kind === "stored" ? { name: input.label, size: 0, stored: true, canRerun: input.canRerun, note: input.note } : null;
  const onPick = (rec_id) => session.current.select({ kind: "dataset", rec_id, label: rec_id });
  const onFile = (f) => session.current.select({ kind: "upload", file: f, uploadId: `upload-${Date.now().toString(36)}-${++uploadCounter}`, label: f.name });

  return (
    <Workspace recordings={recordings} showcase={showcase} selection={selection} upload={upload} status={st.status} error={st.error}
      record={st.record} channels={st.ev.channels} signals={st.ev.signals} activity={st.ev.activity} report={st.ev.report}
      runningFor={st.runningFor} explanation={st.ev.explanation} onRegenerate={() => session.current.regenerate()} limitations={st.record?.limitations || health?.limitations} tab={tab} onTab={setTab}
      userName={name} needsName={!name || editingName} onChangeName={() => setEditingName(true)}
      onSaveName={(n) => { try { localStorage.setItem(NAME_KEY, n); } catch { /* browser-local only */ } setName(n); setEditingName(false); }}
      onPick={onPick} onFile={onFile} onClearUpload={() => session.current.select(null)} onStart={() => session.current.run()}
      onHome={() => { session.current.select(null); window.location.hash = "#/"; }}
      forceStatic={init.forceStatic} initialUi={init.ui} />
  );
}
