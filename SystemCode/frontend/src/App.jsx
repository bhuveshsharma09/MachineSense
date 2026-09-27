import { useEffect, useRef, useState } from "react";
import {
  diagnoseRecording, diagnoseUpload, errorInfo, getActivity, getCase, getChannels, getHealth, getRecordings, getReport,
  getShowcase, getSignals,
} from "./api.js";
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

function Main() {
  const init = useRef(query()).current;
  const [health, setHealth] = useState(null);
  const [recordings, setRecordings] = useState([]);
  const [showcase, setShowcase] = useState([]);
  const [name, setName] = useState(readName());
  const [editingName, setEditingName] = useState(false);
  const [selection, setSelection] = useState(null);
  const [upload, setUpload] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [record, setRecord] = useState(null);
  const [ev, setEv] = useState({ channels: null, signals: null, activity: null, report: null });
  const [tab, setTab] = useState(init.tab || "activity");

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(errorInfo(e)));
    getRecordings().then(setRecordings).catch(() => {});
    getShowcase().then((s) => setShowcase(s.showcase || [])).catch(() => {});
  }, []);

  const reset = () => { setRecord(null); setError(null); setEv({ channels: null, signals: null, activity: null, report: null }); setStatus("idle"); };

  async function loadCase(r) {
    setRecord(r);
    const id = r.case_id;
    const [activity, report] = await Promise.all([getActivity(id), getReport(id)]);
    setEv((e) => ({ ...e, activity, report }));
    setStatus("done");
    setTab(init.tab || "report");
    const [channels, signals] = await Promise.all([getChannels(id), getSignals(id)]);
    setEv((e) => ({ ...e, channels, signals }));
  }

  async function run(sel = selection, file = upload) {
    reset();
    setStatus("running");
    setTab("activity");
    try {
      const r = file ? await diagnoseUpload(file) : await diagnoseRecording(sel.rec_id);
      window.history.replaceState(null, "", window.location.search + `#/case/${r.case_id}`);
      await loadCase(r);
    } catch (e) {
      setError(errorInfo(e));
      setStatus("error");
      setTab("activity");
    }
  }

  useEffect(() => {
    const [, kind, id] = window.location.hash.replace(/^#/, "").split("/");
    if (kind === "case" && id) {
      getCase(id).then((r) => {
        if (r.source === "dataset") setSelection({ kind: "dataset", rec_id: r.input.rec_id, label: r.input.rec_id });
        else setUpload({ name: "uploaded recording", size: 0, stored: true });
        return loadCase(r);
      }).catch((e) => { setError(errorInfo(e)); setStatus("error"); });
    } else if (kind === "diagnose" && id) {
      const sel = { kind: "dataset", rec_id: decodeURIComponent(id), label: decodeURIComponent(id) };
      setSelection(sel);
      run(sel, null);
    }
  }, []);

  const onPick = (rec_id) => { setUpload(null); setSelection({ kind: "dataset", rec_id, label: rec_id }); reset(); };
  const onFile = (f) => { setSelection(null); setUpload(f); reset(); };
  const onStart = () => { if (upload?.stored) return; run(); };

  return (
    <Workspace recordings={recordings} showcase={showcase} selection={selection} upload={upload} status={status} error={error}
      record={record} channels={ev.channels} signals={ev.signals} activity={ev.activity} report={ev.report}
      limitations={record?.limitations || health?.limitations} tab={tab} onTab={setTab}
      userName={name} needsName={!name || editingName} onChangeName={() => setEditingName(true)}
      onSaveName={(n) => { try { localStorage.setItem(NAME_KEY, n); } catch { /* browser-local only */ } setName(n); setEditingName(false); }}
      onPick={onPick} onFile={onFile} onClearUpload={() => { setUpload(null); reset(); }} onStart={onStart}
      forceStatic={init.forceStatic} initialUi={init.ui} />
  );
}
