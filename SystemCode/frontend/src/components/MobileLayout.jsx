import { useEffect, useState } from "react";
import { K } from "../lib/theme.js";
import { CHANNELS, PARTS, cleanText, locations, qualityFlags, unusualSensors } from "../lib/model.js";
import { PanelContent } from "./Aside.jsx";
import { FALLBACK, markerFor } from "./Scene.jsx";
import { changeText } from "./Inspector.jsx";
import { Logo, UnseenLink } from "./TopBar.jsx";
import InvestigationDialog from "./InvestigationDialog.jsx";
import NameModal from "./NameModal.jsx";
import AgentChat from "./AgentChat.jsx";
import { CaseLinks } from "./CaseCard.jsx";

// Below 1024 px (App-2a.3): a stacked, readable layout with no horizontal scroll and no 3D — top bar, diagnosis card
// (with warnings), report, the evidence strip as a list and a static rig image with markers.
function StaticRig({ state }) {
  const [anchors, setAnchors] = useState({});
  useEffect(() => {
    fetch(FALLBACK.anchors).then((r) => r.json()).then(setAnchors).catch(() => setAnchors({}));
  }, []);
  const marks = PARTS.map((h) => markerFor(h, state)).filter((m) => anchors[m.id]);
  return (
    <div>
      <div className="m-rig">
        <img src={FALLBACK.img} alt="Rendered frame of the illustrative 3D rig" />
        {marks.map((m) => (
          <span key={m.id} className="m-dot" title={m.label}
            style={{ left: anchors[m.id][0] * 100 + "%", top: anchors[m.id][1] * 100 + "%", width: m.size, height: m.size,
              background: m.fill, border: m.border, boxShadow: m.ring ? "0 0 0 4px " + m.ring.bg : undefined }} />
        ))}
      </div>
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
    onSaveName, onPick, onFile, onClearUpload, onStart, runningFor, onHome } = props;
  const [dialogOpen, setDialogOpen] = useState(false);
  useEffect(() => {
    if (status === "running" || status === "error") setDialogOpen(true);
    if (status === "done" && record) setDialogOpen(false);
  }, [status, record?.case_id]);
  const flags = qualityFlags(record);
  const loc = locations(record);
  const state = { sel: null, hover: null, fault: loc.fault, cand: loc.cand, unusual: unusualSensors(channels, flags), flags, labelsAll: false };
  const reportPage = !!record && status === "done";
  return (
    <div className={"ms-mobile " + (reportPage ? "mobile-report" : "mobile-home")}>
      <header className="m-card m-top mobile-header">
        <a href="#/" onClick={onHome} className="brand-link"><Logo /><b>MachineSense</b></a>
        <UnseenLink style={{ alignSelf: "flex-start" }} />
      </header>
      {!reportPage && <section className="m-card mobile-intro">
        <span className="mono kicker">Rotor fault test rig</span>
        <h1>Understand what your machine is telling you</h1>
        <p>Investigate a recording, inspect the evidence, and decide what needs attention.</p>
        <button type="button" className="start-btn" onClick={() => setDialogOpen(true)}>Start investigation →</button>
      </section>}
      {reportPage && <CaseLinks caseId={record.case_id} />}
      {reportPage && <section className="m-card m-panel">
        <PanelContent record={record} status={status} error={error} selection={selection} report={report} activity={props.activity}
          runningFor={runningFor} explanation={props.explanation} onRegenerate={props.onRegenerate} view={props.view} onView={props.onView} />
      </section>}
      <section className="m-card"><span className="mono kicker">Rotor test rig</span><StaticRig state={state} /></section>
      {reportPage && <section className="m-card"><span className="mono kicker">Measured change vs healthy reference</span>
        <EvidenceList record={record} channels={channels} /></section>}
      {reportPage && <section className="m-card m-note"><details><summary>Research limitations</summary>
        <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>{(limitations || []).map((x) => <li key={x}>{cleanText(x)}</li>)}</ul>
      </details></section>}
      {reportPage && <button type="button" className="start-btn mobile-new" onClick={() => setDialogOpen(true)}>New investigation</button>}
      <InvestigationDialog open={dialogOpen} recordings={recordings} showcase={showcase} selection={selection} upload={upload}
        onPick={onPick} onFile={onFile} onClearUpload={onClearUpload} onStart={onStart} onClose={() => setDialogOpen(false)}
        status={status} error={error} runningFor={runningFor} />
      {needsName && <NameModal initial={userName} onSave={onSaveName} />}
      <AgentChat caseId={record?.case_id} open={props.chatOpen} onOpen={props.onOpenChat} onClose={props.onCloseChat} />
    </div>
  );
}
