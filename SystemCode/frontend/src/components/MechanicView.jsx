import { useEffect, useRef, useState } from "react";
import { K, al } from "../lib/theme.js";
import * as M from "../lib/mechanic.js";
import * as D from "../lib/dashboard.js";
import { CH, IN_SAMPLE_NOTICE, PART, SCORES_NOTE, cleanText, human, locations, shownWarnings, unusualSensors } from "../lib/model.js";
import Scene, { markerFor } from "./Scene.jsx";
import { Caveat, ChannelCard, Header, Seg, SpectrumCard, WaveformCard } from "./Dashboard.jsx";
import "../mechanic.css";

// Stage App-8.4: the dashboard's Mechanic view as four numbered sections under a jump bar (designer's "MachineSense
// Mechanic"). No score, threshold or percentage is shown. The location stays "according to the dataset label (model
// prediction)"; "Show me" exists only for App-8.2's two declared mappings and opens their spectrum in section 03.
// Section ids rd-finding / rd-measurements / rd-suggested-checks are the targets the assistant's links scroll to.
export const MECH_SECTIONS = [["rd-finding", "Verdict"], ["mvw-machine", "Machine"], ["rd-measurements", "Sensors"], ["rd-suggested-checks", "What to check"]];
const FAULT = "oklch(0.50 0.16 40)";

export default function MechanicView({ caseId, record, report, channels, signals, explanation, view, onView, userName,
  assistantOpen = false, narrow = false, initialSel = null, initialUi = {} }) {
  const [ui, setUi] = useState({ sel: initialSel, hover: null, exploded: false, labels: false, resetTick: 0, ch: null, sp: "order",
    measure: "channels", lim: false, ...initialUi });
  const set = (p) => setUi((u) => ({ ...u, ...p }));
  const [rigFocus, setRigFocus] = useState(null);
  const [current, setCurrent] = useState(MECH_SECTIONS[0][0]);
  const rigRef = useRef(null);

  const d = record.diagnosis;
  const e = explanation?.status === "ready" ? explanation.data : null;
  const flags = D.qualityFlags(record);
  const loc = locations(record);
  const unusual = unusualSensors(channels, flags);
  const rows = D.channelRows(channels);
  const mloc = M.location(record);
  const hasFaultArea = d.decision === "known_fault" && loc.fault.length > 0;
  const warn = shownWarnings(record);
  const steps = D.guidanceSteps(M.guidanceItems(report));
  const showChecks = d.decision !== "healthy";
  const healthyLine = (report?.sections?.find((s) => s.n === "04")?.items || []).find((it) => it.tag === "Model")?.text;
  const limits = (record.limitations || []).map(cleanText);

  const partId = D.focusPart(ui.sel, loc, unusual);
  const part = PART[partId];
  const sceneState = { ...ui, fault: loc.fault, cand: hasFaultArea ? loc.cand : [], unusual, flags, labelsAll: ui.labels };
  const evidenceChannel = rows.filter((r) => r.comparable && r.beyond && r.metric === "order_2x").sort((a, b) => b.factor / b.band - a.factor / a.band)[0]?.id
    || rows.find((r) => r.comparable && r.beyond)?.id;
  const chId = ui.ch || evidenceChannel || part.ch[0];
  const measuredPart = PART[CH[chId]?.sensor] || part;
  const measuredOptions = measuredPart.ch.filter((c) => signals?.channels?.[c]);
  const select = (id) => set({ sel: id, ch: id && PART[id] && !PART[id].ch.includes(ui.ch) ? PART[id].ch[0] : ui.ch });
  const scrollTo = (id) => requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  const focusRigPart = (id) => { select(id); setRigFocus({ id }); scrollTo("mvw-machine"); };
  const showMeasurement = (target) => { set({ measure: "spectrum", sp: "order", ch: target.channel, sel: CH[target.channel].sensor }); scrollTo("rd-measurements"); };

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return undefined;
    const io = new IntersectionObserver((es) => {
      const vis = es.filter((x) => x.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (vis) setCurrent(vis.target.id);
    }, { rootMargin: "-120px 0px -60% 0px" });
    MECH_SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);

  const verdictLabel = d.decision === "known_fault" ? `Verdict · ${human(d.top_class)} (model prediction)`
    : d.decision === "unknown" ? "Verdict · Further inspection needed" : "Verdict · Healthy";
  const verdictColor = d.decision === "known_fault" ? FAULT : d.decision === "unknown" ? K.amberT : K.greenT;
  const marker = markerFor(part, sceneState);
  const related = rows.filter((r) => part.ch.includes(r.id));

  return (
    <div className={"rd-root rd-mechanic mvw-root" + (narrow ? " is-narrow" : "") + (assistantOpen ? " has-assistant" : "")}>
      <Header caseId={caseId} record={record} view={view} onView={onView} userName={userName} back={{ href: "#/", label: "Back to home" }} />
      <nav className="mvw-jump" aria-label="Mechanic view sections">
        <div>{MECH_SECTIONS.map(([id, label], i) => <button key={id} type="button" className={current === id ? "on" : ""} aria-current={current === id ? "true" : undefined}
          onClick={() => { setCurrent(id); scrollTo(id); }}><span>{String(i + 1).padStart(2, "0")}</span>{label}</button>)}</div>
      </nav>
      <main className="rd-m-main mvw-main">
        {/* 01 Verdict */}
        <section id="rd-finding" tabIndex={-1} className="mvw-verdict" aria-label="Finding">
          <div className="mvw-col">
            <span className="mvw-verdict-label" style={{ color: verdictColor }}>{verdictLabel}</span>
            <h1 className="rd-m-title">{M.finding(record)}</h1>
            {d.caution && <span className="rd-sub">A healthy-baseline check disagrees; consider a recheck.</span>}
            {hasFaultArea && mloc.parts.length > 0 && <p className="mvw-next">Start with the {mloc.parts.map((p) => M.PART_PLAIN[p].toLowerCase()).join(" and ")}.</p>}
            {d.decision === "unknown" && <p className="mvw-next">The checks did not support a known condition; inspect the readings before choosing a fault area.</p>}
            <span className="mvw-speed">Shaft speed {record.signal_quality.shaft_speed_hz.toFixed(1)} Hz ({Math.round(record.physics.shaft_speed_rpm).toLocaleString("en-US")} rpm)</span>
            {hasFaultArea && mloc.parts.length > 0 && <div className="mvw-location">
              <strong>{mloc.parts.map((p) => M.PART_PLAIN[p]).join(" and ")}</strong>
              <span className="mvw-loc-label">According to the dataset label (model prediction)</span>
              <Caveat>Mapped from the model’s predicted class on this test rig; the sensors do not pinpoint the fault.</Caveat>
              <button type="button" className="rd-dark-btn" onClick={() => focusRigPart(mloc.parts[0])}>Show {M.PART_PLAIN[mloc.parts[0]].toLowerCase()} on the rig</button>
            </div>}
            <div className="mvw-quality">
              {warn.length ? <div className="rd-warn" role="alert">{M.warnings(record).map((w) => <div key={w}><span className="mvw-tick is-warn" aria-hidden="true">!</span>{w}</div>)}</div>
                : <span className="rd-ok"><span className="mvw-tick" aria-hidden="true">✓</span>No signal-quality warnings detected.</span>}
              {warn.length > 0 && <details><summary>Signal quality details</summary><div className="rd-checklist">{D.qualityChecks(record).map((q) => <div key={q.text}>
                <span className={"mvw-tick" + (q.ok ? "" : " is-warn")} aria-hidden="true">{q.ok ? "✓" : "!"}</span>{q.text}</div>)}</div></details>}
              {record.source === "dataset" && <span className="mvw-insample">{IN_SAMPLE_NOTICE}</span>}
            </div>
          </div>
          <span className="mvw-rule-v" aria-hidden="true" />
          <div className="mvw-col" aria-label="Evidence">
            <h2>Evidence</h2>
            {e ? <>
              <Caveat>{e.badge}</Caveat>
              <ul className="mvw-bullets">{e.text.why_flagged.map((x) => <li key={x}>{x}</li>)}</ul>
            </> : <p className="rd-muted">{explanation?.status === "error" ? "The explanation is unavailable. Review the measured channels below." : "Preparing the explanation…"}</p>}
            <div className="mvw-box"><span className="rd-m-label">Healthy-baseline check</span><span>{M.supporting(record, report?.assessment) || "Checking…"}</span></div>
            {e?.text.uncertainty?.length > 0 && <div className="mvw-unc"><span className="rd-m-label">Uncertainty</span>
              <ul>{e.text.uncertainty.map((x) => <li key={x}>{x}</li>)}</ul></div>}
          </div>
        </section>

        {/* 02 Machine */}
        <section id="mvw-machine" className="mvw-sec">
          <div className="mvw-sec-head"><span className="mvw-n">02</span><h2>Machine</h2>
            <span className="mvw-sub">Where the model points and where the signal changed. Select a part or sensor to see its readings.</span></div>
          <div className={"mvw-machine" + (narrow ? " is-narrow" : "")}>
            <div ref={rigRef} className="rd-rig mvw-rig" aria-label="Interactive test rig" role="region">
              <Scene state={sceneState} insets={{ top: 62, right: 20, bottom: 50, left: 20 }} zoomOut={0.92} wheelZoomRequiresModifier focusOnSelect={false}
                focusRequest={rigFocus} onSelect={select} onHover={(id) => { if (id !== ui.hover) set({ hover: id }); }} />
              <div className="mvw-rig-btns">
                <button className="rd-btn" onClick={() => set({ sel: null, resetTick: ui.resetTick + 1 })}>Reset view</button>
                <button className={"rd-btn" + (ui.exploded ? " is-on" : "")} aria-pressed={ui.exploded} onClick={() => set({ exploded: !ui.exploded })}>Exploded view</button>
                <button className="rd-btn" aria-pressed={ui.labels} onClick={() => set({ labels: !ui.labels })}>{ui.labels ? "Labels: all" : "Labels: minimal"}</button>
              </div>
              <div className="mvw-rig-legend">
                <span><i className="lg-sel" />Selected</span>
                {hasFaultArea && <span><i className="lg-fault" />Predicted location · model</span>}
                <span><i className="lg-change" />Measured change</span><span><i className="lg-flag" />Inspect · quality flag</span>
                <Caveat>Highlights do not confirm a fault.</Caveat>
              </div>
            </div>
            <aside className="mvw-insp" aria-label="Selected part">
              <span className="mvw-tag" style={marker.tag ? { color: marker.tagColor, background: al(marker.fill, 0.14) } : undefined}>{marker.tag || "Nothing flagged"}</span>
              <b className="mvw-insp-name">{M.PART_PLAIN[part.id] || part.full || part.label}</b>
              <p>{part.desc}</p>
              {related.length > 0 && <div className="mvw-col">
                <span className="rd-m-label">Related channels</span>
                {related.map((r) => <button key={r.id} type="button" className={"mvw-insp-ch" + (r.id === chId ? " on" : "")}
                  onClick={() => { set({ ch: r.id, sel: r.sensor, measure: "channels" }); scrollTo("rd-measurements"); }}>
                  <span><i style={{ background: r.beyond ? K.lime : "oklch(0.23 0.006 260 / 0.2)", border: r.beyond ? "1.5px solid " + K.ink : "none" }} />{r.name}</span>
                  <em style={{ color: r.beyond ? K.limeT : K.ink2 }}>{!r.comparable ? "No comparable reading" : r.beyond ? "Changed" : "Typical"}</em></button>)}
              </div>}
              <span className="mvw-insp-hint">{ui.sel ? "Selected on the rig. Its readings are below." : "Showing the predicted location or a changed sensor. Select a part or sensor on the rig."}</span>
            </aside>
          </div>
        </section>

        {/* 03 Sensors */}
        <section id="rd-measurements" tabIndex={-1} className="mvw-sec" aria-label="Explore measurements">
          <div className="mvw-sec-head"><span className="mvw-n">03</span><h2>Sensors</h2>
            <span className="mvw-sub">Measurements: measured readings only. No model is involved in this section.</span>
            <Seg label="Measurement view" value={ui.measure} onChange={(measure) => set({ measure })}
              options={[["channels", "Sensors"], ["spectrum", "Spectrum"], ["waveform", "Waveform"]]} /></div>
          {ui.measure === "channels" && <ChannelCard channels={channels} rows={rows} dev={false} sel={ui.sel} onSelect={(r) => set({ sel: r.sensor, ch: r.id })} />}
          {ui.measure === "spectrum" && <SpectrumCard record={record} signals={signals} part={measuredPart} chId={chId} chOptions={measuredOptions} dev={false}
            mode={ui.sp} onMode={(sp) => set({ sp })} onChannel={(ch) => set({ ch })} />}
          {ui.measure === "waveform" && <WaveformCard signals={signals} chId={chId} />}
        </section>

        {/* 04 What to check */}
        <section id="rd-suggested-checks" tabIndex={-1} className="mvw-sec" aria-label={showChecks ? "Suggested checks" : "What to check"}>
          <div className="mvw-sec-head"><span className="mvw-n">04</span><h2>What to check</h2>
            <span className="mvw-sub">Steps from the curated guidance library, in order.</span></div>
          <div className="mvw-checks">
            {!showChecks ? <p className="mvw-healthy">{healthyLine || "No inspection steps are offered for a healthy verdict."}</p>
              : !report ? <p className="rd-muted">Suggested checks appear with the report.</p> : <>
                <div className="mvw-col"><h3>Suggested checks</h3><span className="mvw-sub">From curated guidance · for {M.finding(record).replace(/^./, (c) => c.toLowerCase())}</span></div>
                <ol className="mvw-steps">{steps.map((s, i) => { const show = D.showMeTarget(s, i);
                  return <li key={i}>
                    <span className="rd-step-n">{i + 1}</span>
                    <div className="mvw-col">
                      <span className="mvw-step-t">{s.text}</span>
                      {/\b(?:1×|2×|order)\b/i.test(s.text) && <span className="rd-m-gloss">1× = once per shaft turn; 2× = twice per turn.</span>}
                      {s.src && <span className="mvw-cite" title={`${s.title} (source: ${s.src})`}>{s.title.split(" — ")[0]}</span>}
                      {s.flag && <Caveat>{s.flag}</Caveat>}
                      {show && <button type="button" className="rd-m-show" onClick={() => showMeasurement(show)}>Show me</button>}
                    </div>
                  </li>; })}</ol>
                <Caveat>Work on the rig only when it is stopped and isolated. These steps suggest where to look; they do not confirm a fault.</Caveat>
              </>}
          </div>
        </section>

        <details className="rd-m-details"><summary>Recording details</summary>
          <div className="rd-kv">
            <div><span>Recording</span><span className="mono">{record.input?.rec_id || "uploaded recording"}</span></div>
            <div><span>Shaft speed</span><span>{Math.round(record.physics.shaft_speed_rpm).toLocaleString("en-US")} rpm</span></div>
            <div><span>Duration and channels</span><span>5.00 s · 8 channels</span></div>
            <div><span>Case</span><span className="mono">{caseId.slice(0, 12)}…</span></div>
          </div>
        </details>

        <footer className="mvw-foot">
          <div><b>{M.LIMITATION}</b><span>{SCORES_NOTE}</span>
            {limits.length > 0 && <button type="button" className="rd-ghost" aria-expanded={ui.lim} onClick={() => set({ lim: !ui.lim })}>{ui.lim ? "Hide research limitations" : "Research limitations"}</button>}</div>
          {ui.lim && <ul>{limits.map((l) => <li key={l}>{l}</li>)}</ul>}
        </footer>
      </main>
    </div>
  );
}
