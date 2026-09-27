import { useState } from "react";
import { K, al } from "../lib/theme.js";
import { PART, locations, nf, qualityFlags, recordingTitle, unusualSensors } from "../lib/model.js";
import Aside from "./Aside.jsx";
import EvidenceStrip, { factsFor } from "./EvidenceStrip.jsx";
import Inspector from "./Inspector.jsx";
import LimitationsBanner from "./LimitationsBanner.jsx";
import NameModal from "./NameModal.jsx";
import Scene, { markerFor } from "./Scene.jsx";
import TopBar from "./TopBar.jsx";

function Legend() {
  const row = (dot, text) => (
    <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, color: "oklch(0.30 0.006 260)" }}>
      <span style={{ width: 14, height: 14, flex: "none", display: "grid", placeItems: "center" }}>{dot}</span>{text}
    </div>
  );
  return (
    <div className="glass-soft" style={{ position: "absolute", top: 140, right: 428, zIndex: 1, width: 250, display: "flex", flexDirection: "column", gap: 8, padding: "13px 15px", borderRadius: 14 }}>
      {row(<span style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid " + K.violet }} />, "Selected part")}
      {row(<span style={{ width: 14, height: 14, borderRadius: "50%", background: `radial-gradient(circle, ${K.orange} 0 35%, ${al(K.orange, 0.25)} 60%, transparent 72%)` }} />, "Location according to the dataset label (model)")}
      {row(<span style={{ width: 10, height: 10, borderRadius: "50%", background: K.lime, border: "2px solid " + K.ink }} />, "Largest change vs healthy reference · measured")}
      {row(<span style={{ width: 14, height: 14, borderRadius: "50%", border: "1.5px dashed " + K.amberT, background: al(K.amber, 0.25) }} />, "Candidate location or signal-quality flag")}
      <p style={{ margin: "2px 0 0", fontSize: 11, lineHeight: 1.4, color: "oklch(0.48 0.008 260)" }}>Measured change is descriptive, not a calibrated anomaly test. Orange glow is a visual highlight, not a measured temperature.</p>
    </div>
  );
}

export default function Workspace(props) {
  const { recordings, showcase, selection, upload, status, error, record, channels, signals, activity, report, limitations,
    userName, needsName, onSaveName, onChangeName, onPick, onFile, onClearUpload, onStart, forceStatic, initialUi } = props;
  const [ui, setUi] = useState({ sel: null, hover: null, exploded: false, labels: false, evCh: null, evView: "spectrum", tab: "activity", resetTick: 0, ...initialUi });
  const set = (p) => setUi((u) => ({ ...u, ...p }));
  const tab = props.tab ?? ui.tab;
  const onTab = props.onTab ?? ((t) => set({ tab: t }));

  const flags = qualityFlags(record);
  const loc = locations(record);
  const unusual = unusualSensors(channels, flags);
  const sceneState = { ...ui, fault: loc.fault, cand: loc.cand, unusual, flags, labelsAll: ui.labels };

  const part = ui.sel ? PART[ui.sel] : null;
  const chId = part ? (part.ch.includes(ui.evCh) ? ui.evCh : part.ch[0]) : ui.evCh;
  let inspector = null;
  if (part) {
    const m = markerFor(part, sceneState);
    const status2 = m.tag ? { text: m.tag, color: m.tagColor, dot: m.fill }
      : { text: record ? "Nothing flagged for this part" : "No recording investigated", color: K.ink3, dot: "oklch(0.23 0.006 260 / 0.25)" };
    inspector = <Inspector part={part} status={status2} chId={chId} onChannel={(c) => set({ evCh: c })} view={ui.evView} onView={(v) => set({ evView: v })}
      record={record} channels={channels} signals={signals} onClose={() => set({ sel: null })} />;
  }

  const rec = selection?.kind === "dataset" ? recordings?.find((r) => r.rec_id === selection.rec_id) : null;
  const fr = record?.signal_quality.shaft_speed_hz;
  const hero = rec
    ? { k: `Recording ${rec.rec_id} · dataset label`, t: recordingTitle(rec), s: `${(fr ?? rec.speed_hz).toFixed(1)} Hz · ${nf(60 * (fr ?? rec.speed_hz))} rpm · 5.00 s · 8 channels · ${rec.used_for}` }
    : upload ? { k: "Uploaded file", t: upload.name, s: record ? `${fr.toFixed(1)} Hz measured · 5.00 s · 8 channels` : error ? (error.status === 403 ? "Upload refused" : "File check failed") : "Not checked yet" }
    : { k: "Rotor fault test rig", t: "Select a recording", s: "Choose a development recording or upload a CSV file." };

  return (
    <div className="ms-root">
      <Scene state={sceneState} onSelect={(id) => set({ sel: id, evCh: id && PART[id] && !PART[id].ch.includes(ui.evCh) ? PART[id].ch[0] : ui.evCh })}
        onHover={(id) => { if (id !== ui.hover) set({ hover: id }); }} inspector={inspector} forceStatic={forceStatic} />
      <div style={{ position: "absolute", left: 36, top: 150, zIndex: 1, pointerEvents: "none", display: "flex", flexDirection: "column", gap: 6, maxWidth: 540, opacity: part ? 0 : 1, transition: "opacity 0.35s ease" }}>
        <span className="mono" style={{ fontSize: 10.5, letterSpacing: "0.1em", textTransform: "uppercase", color: "oklch(0.48 0.008 260)" }}>{hero.k}</span>
        <span style={{ fontSize: 46, fontWeight: 300, letterSpacing: "-0.028em", lineHeight: 1.04 }}>{hero.t}</span>
        <span style={{ fontSize: 14.5, color: K.ink2 }}>{hero.s}</span>
      </div>
      <Legend />
      <TopBar recordings={recordings} showcase={showcase} value={selection?.kind === "dataset" ? selection.rec_id : ""} onPick={onPick}
        onFile={onFile} upload={upload} onClearUpload={onClearUpload} onStart={onStart} status={status} hasRecord={!!record}
        userName={userName} onChangeName={onChangeName} />
      <LimitationsBanner items={limitations} />
      <div style={{ position: "absolute", left: 16, bottom: 168, right: 412, zIndex: 4, display: "flex", alignItems: "center", gap: 8, pointerEvents: "none" }}>
        <button className="ctl" onClick={() => set({ sel: null, resetTick: ui.resetTick + 1 })}>Reset view</button>
        <button className="ctl" aria-pressed={ui.exploded} onClick={() => set({ exploded: !ui.exploded })}
          style={ui.exploded ? { background: K.white, borderColor: al(K.violet, 0.55), color: K.violet } : undefined}>Exploded view</button>
        <button className="ctl" onClick={() => set({ labels: !ui.labels })}>{ui.labels ? "Labels: all" : "Labels: minimal"}</button>
        <span style={{ marginLeft: "auto", fontSize: 11.5, color: "oklch(0.45 0.008 260)", textAlign: "right", lineHeight: 1.35 }}>
          Illustrative 3D geometry, approximated from the reference photo.<br />Drag to orbit · scroll to zoom · click a part
        </span>
      </div>
      <EvidenceStrip record={record} channels={channels} signals={signals} flags={flags} selCh={part ? chId : null} upload={upload}
        facts={factsFor(record, upload, error)} onTile={(c) => set({ sel: c.sensor, evCh: c.id })} />
      <Aside record={record} status={status} error={error} selection={selection} activity={activity} report={report} tab={tab} onTab={onTab} />
      {needsName && <NameModal initial={userName} onSave={onSaveName} />}
    </div>
  );
}
