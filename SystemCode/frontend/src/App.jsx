import { useEffect, useRef, useState } from "react";
import {
  diagnoseRecording, diagnoseUpload, getActivity, getCase, getCaseMeta, getChannels, getHealth, getRecordings, getReport,
  getShowcase, getSignals, rerunCase,
} from "./api.js";
import { createSession } from "./lib/session.js";
import Scene from "./components/Scene.jsx";
import Workspace from "./components/Workspace.jsx";

const NAME_KEY = "ms_display_name";
const readName = () => { try { return localStorage.getItem(NAME_KEY) || ""; } catch { return ""; } };

// Hash routes: #/ ; #/case/<case_id> (reopens a stored case) ; #/diagnose/<rec_id> (selects and runs a recording).
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

export default function App() {
  return useRef(query()).current.capture ? <Capture /> : <Main />;
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
      onCase: (id) => window.history.replaceState(null, "", window.location.search + `#/case/${id}`),
    }, setS);
  }
  const st = s || session.current.get();

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getRecordings().then(setRecordings).catch(() => {});
    getShowcase().then((x) => setShowcase(x.showcase || [])).catch(() => {});
    const [, kind, id] = window.location.hash.replace(/^#/, "").split("/");
    if (kind === "case" && id) session.current.open(id);
    else if (kind === "diagnose" && id) {
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
      runningFor={st.runningFor} limitations={st.record?.limitations || health?.limitations} tab={tab} onTab={setTab}
      userName={name} needsName={!name || editingName} onChangeName={() => setEditingName(true)}
      onSaveName={(n) => { try { localStorage.setItem(NAME_KEY, n); } catch { /* browser-local only */ } setName(n); setEditingName(false); }}
      onPick={onPick} onFile={onFile} onClearUpload={() => session.current.select(null)} onStart={() => session.current.run()}
      forceStatic={init.forceStatic} initialUi={init.ui} />
  );
}
