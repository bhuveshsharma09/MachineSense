import { useEffect, useRef, useState } from "react";
import { K, TAG, al } from "../lib/theme.js";
import * as M from "../lib/mechanic.js";
import * as D from "../lib/dashboard.js";
import { decisionSummary } from "../lib/decisionSummary.js";
import { CH, IN_SAMPLE_NOTICE, PART, RULE_LINE, SCORES_NOTE, cleanText, fmtScore, human, locations, shownWarnings,
  unusualSensors, verdictTitle } from "../lib/model.js";
import { Svg, bandPath, linePath, rangeMax } from "./Charts.jsx";
import Scene, { markerFor } from "./Scene.jsx";
import { Header } from "./Dashboard.jsx";
import DevPipeline from "./DevPipeline.jsx";
import "../developer.css";

// Stage App-8.3: the dashboard's Developer view as one page of seven numbered sections (designer's "MachineSense
// Developer"). Every value comes from the case's existing responses. The decision section shows the rule in use: the
// classifier-first rule decides with the confidence and familiarity checks and the top class; the autoencoder and B1s
// checks only add a caution to a healthy verdict.
export const SECTIONS = [["verdict", "Verdict"], ["machine", "Machine"], ["data", "Data"], ["models", "Models"],
  ["decision", "Decision"], ["pipeline", "Pipeline"], ["report", "Report"]];
const PRIMARY = { msp: "Confidence", maha: "Familiarity", ae: "Healthy pattern", b1s: "Healthy at this speed" };
const TECH = { msp: "MSP · CNN-6", maha: "Mahalanobis distance", ae: "autoencoder error", b1s: "B1s speed-aware baseline" };
const f3 = (v) => (v == null || !Number.isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));

function Section({ id, n, title, sub, children, wide = false, anchor }) {
  return (
    <section id={`dvw-${id}`} className={"dvw-sec" + (wide ? " is-wide" : "")} aria-labelledby={`dvw-${id}-h`}>
      <div id={anchor} tabIndex={anchor ? -1 : undefined} className="dvw-sec-head"><span className="dvw-n">{n}</span><h2 id={`dvw-${id}-h`}>{title}</h2>{sub && <span className="dvw-sub">{sub}</span>}</div>
      {children}
    </section>
  );
}

function Panel({ id, title, kicker, children, className = "", span }) {
  return (
    <div id={id} tabIndex={id ? -1 : undefined} className={"dvw-panel " + className} style={span ? { gridColumn: `span ${span}` } : undefined}>
      {(title || kicker) && <div className="dvw-panel-head">{title && <h3>{title}</h3>}{kicker}</div>}
      {children}
    </div>
  );
}

export default function DeveloperView({ caseId, record, report, channels, signals, activity, explanation, onRegenerate,
  view, onView, userName, assistantOpen = false, narrow = false, initialSel = null, trace = null }) {
  const [sel, setSel] = useState(initialSel);
  const [hover, setHover] = useState(null);
  const [rig, setRig] = useState({ exploded: false, labels: false, resetTick: 0 });
  const [ch, setCh] = useState(() => D.bearingSpectrum(record)?.ch || null);
  const [sp, setSp] = useState(() => D.bearingSpectrum(record)?.sp || "order");
  const [current, setCurrent] = useState("verdict");
  const machineRef = useRef(null);

  const d = record.diagnosis;
  const flags = D.qualityFlags(record);
  const loc = locations(record);
  const unusual = unusualSensors(channels, flags);
  const rows = D.channelRows(channels);
  const partId = D.focusPart(sel, loc, unusual);
  const part = PART[partId];
  const chOptions = part.ch.filter((c) => signals?.channels?.[c]);
  const chId = chOptions.includes(ch) ? ch : chOptions[0] || part.ch[0];
  const sceneState = { sel, hover, exploded: rig.exploded, labels: rig.labels, resetTick: rig.resetTick, fault: loc.fault, cand: loc.cand,
    unusual, flags, labelsAll: rig.labels };
  const mloc = M.location(record);
  const checks = report?.assessment?.checks;
  const summary = decisionSummary({ topClass: d.top_class, decision: d.decision, caution: d.caution, checks });
  const warn = shownWarnings(record);
  const e = explanation?.status === "ready" ? explanation.data : null;
  const variants = report?.assessment?.variants || [];
  const agree = variants.filter((x) => x.verdict === variants.find((v) => /in use/.test(v.name))?.verdict).length;
  const select = (id) => { setSel(id); if (id && PART[id] && !PART[id].ch.includes(ch)) setCh(PART[id].ch[0]); };
  const showOnRig = (id) => { select(id); machineRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" }); };

  // the jump bar follows the section in view
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return undefined;
    const io = new IntersectionObserver((es) => {
      const vis = es.filter((x) => x.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (vis) setCurrent(vis.target.id.replace("dvw-", ""));
    }, { rootMargin: "-120px 0px -60% 0px" });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(`dvw-${id}`); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  const jump = (id) => { setCurrent(id); document.getElementById(`dvw-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }); };

  const lrTop = d.lr_secondary?.top_class;
  const keyNums = [
    ["CNN-6 · MSP", fmtScore(d.scores.msp)],
    ["Second opinion · LR", lrTop ? fmtScore(d.lr_secondary.probabilities[lrTop]) : "—"],
    ["Confidence · familiarity", checks ? `${summary.checks.msp ? "pass" : "fail"} · ${summary.checks.maha ? "pass" : "fail"}` : "…"],
    ["Rule variants", variants.length ? `${agree} of ${variants.length} agree` : "…"],
  ];

  return (
    <div className={"rd-root dvw-root" + (narrow ? " is-narrow" : "") + (assistantOpen ? " has-assistant" : "")}>
      <Header caseId={caseId} record={record} view={view} onView={onView} userName={userName} back={{ href: "#/", label: "Back to home" }} />
      <nav className="dvw-jump" aria-label="Developer view sections">
        <div>{SECTIONS.map(([id, label], i) => <button key={id} type="button" aria-current={current === id ? "true" : undefined}
          className={current === id ? "on" : ""} onClick={() => jump(id)}><span>{String(i + 1).padStart(2, "0")}</span>{label}</button>)}</div>
      </nav>
      <main className="dvw-main">
        {/* 01 Verdict */}
        <section id="dvw-verdict" className="dvw-verdict" aria-label="Finding">
          <div className="dvw-col">
            <span className="dvw-kick">{`Verdict · ${verdictTitle(record)}${d.decision === "known_fault" ? " (model prediction)" : ""}`}</span>
            <h1 className="rd-m-title">{M.finding(record)}</h1>
            <span className="dvw-sub">{`Shaft speed ${record.signal_quality.shaft_speed_hz.toFixed(2)} Hz (${Math.round(record.physics.shaft_speed_rpm).toLocaleString("en-US")} rpm)`}</span>
            {mloc.parts.length > 0 && <div className="dvw-loc">
              <b>{mloc.parts.map((p) => M.PART_PLAIN[p]).join(" and ")}</b>
              <span className="dvw-caveat">{mloc.kind === "candidate" ? "Candidate parts from the two most likely fault types (model prediction)."
                : "According to the dataset label (model prediction). Mapped from the model's predicted class on this test rig; the sensors do not pinpoint the fault."}</span>
              {!narrow && <button type="button" className="dvw-dark" onClick={() => showOnRig(mloc.parts[0])}>{`Show ${M.PART_PLAIN[mloc.parts[0]].toLowerCase()} on the rig`}</button>}
            </div>}
            <div className="dvw-keys">{keyNums.map(([k, v]) => <div key={k}><span>{k}</span><b className="mono">{v}</b></div>)}</div>
            <span className="dvw-note">{SCORES_NOTE}</span>
            {warn.length ? <div className="dvw-warn" role="alert">{warn.map((w) => <div key={w}>{w}</div>)}</div>
              : <span className="dvw-ok">No signal-quality warnings detected.</span>}
          </div>
          <span className="dvw-rule-v" aria-hidden="true" />
          <div className="dvw-col">
            <h2 className="dvw-h">Evidence</h2>
            {e ? <>
              <span className="dvw-caveat">{e.badge}</span>
              <ul className="dvw-bullets">{e.text.why_flagged.map((x) => <li key={x}>{x}</li>)}</ul>
              {e.text.uncertainty?.length > 0 && <div className="dvw-unc"><span className="dvw-kick-s">Uncertainty</span>{e.text.uncertainty.map((x) => <div key={x}>{x}</div>)}</div>}
            </> : <p className="dvw-muted">{explanation?.status === "error" ? "The explanation could not be prepared; the report below still applies." : "Preparing explanation…"}</p>}
            <div className="dvw-box"><b>Healthy-baseline check</b><span>{M.supporting(record, report?.assessment) || "Checking…"}</span></div>
          </div>
        </section>

        {/* 02 Machine */}
        {!narrow && <div ref={machineRef}><Section id="machine" n="02" title="Machine" sub="Where the model points and where the signal changed. Select a part to filter the data below.">
          <div className="dvw-machine">
            <Scene state={sceneState} insets={{ top: 60, right: 360, bottom: 52, left: 20 }} onSelect={select}
              onHover={(id) => { if (id !== hover) setHover(id); }} />
            <div className="dvw-rig-btns">
              <button type="button" onClick={() => { setSel(null); setRig((r) => ({ ...r, resetTick: r.resetTick + 1 })); }}>Reset view</button>
              <button type="button" aria-pressed={rig.exploded} className={rig.exploded ? "on" : ""} onClick={() => setRig((r) => ({ ...r, exploded: !r.exploded }))}>Exploded view</button>
              <button type="button" aria-pressed={rig.labels} onClick={() => setRig((r) => ({ ...r, labels: !r.labels }))}>{rig.labels ? "Labels: all" : "Labels: minimal"}</button>
            </div>
            <Inspector part={part} sceneState={sceneState} rows={rows} sel={sel} chId={chId} onChannel={(c) => setCh(c)} />
            <div className="dvw-rig-legend"><span><i className="lg-sel" />Selected</span><span><i className="lg-fault" />Predicted location · model</span>
              <span><i className="lg-change" />Measured change</span><span><i className="lg-flag" />Inspect · quality flag</span>
              <span className="dvw-rig-note">Highlights suggest where to inspect; they do not confirm a fault.</span></div>
          </div>
        </Section></div>}

        {/* 03 Data */}
        <Section id="data" n="03" title="What the data shows" sub="Measured values only. No model is involved in this section.">
          <div className="dvw-grid3">
            <Panel id="rd-signal-quality" title="Signal quality" kicker={<span className={"dvw-pill" + (warn.length ? " is-warn" : "")}>
              {warn.length ? `${warn.length} warning${warn.length > 1 ? "s" : ""}` : "No warnings"}</span>}>
              <div className="dvw-kv">{D.qualityRows(record).map(([k, v]) => <div key={k}><span>{k}</span><b className="mono">{v}</b></div>)}</div>
              <span className="dvw-muted">{warn.length ? warn.join(" · ") : "Warnings: none."}</span>
              {record.signal_quality.warnings.filter((w) => !warn.includes(w)).map((w) => <span key={w} className="dvw-muted">{`Note, applies to a healthy verdict only: ${w}`}</span>)}
            </Panel>
            <Channels channels={channels} rows={rows} sel={sel} chId={chId} span={2} onRow={(r) => { setSel(r.sensor); setCh(r.id); }} />
            <Spectrum record={record} signals={signals} part={part} chId={chId} chOptions={chOptions} mode={sp} onMode={setSp} onChannel={setCh} span={2} />
            <Waveform signals={signals} chId={chId} />
          </div>
        </Section>

        {/* 04 Models */}
        <Section id="models" n="04" title="What the models saw" sub={SCORES_NOTE}>
          <div className="dvw-grid3">
            <Panel title="Top-3 classes" kicker={<span className="dvw-legend2"><i className="is-cnn" />CNN-6<i className="is-lr" />LR</span>}>
              {D.candidates(record).map((c) => (
                <div key={c.cls} className="dvw-top">
                  <span>{c.name}</span>
                  <div className="dvw-bar2"><span className="dvw-track"><i className="is-cnn" style={{ width: pct(d.class10_probabilities[c.cls]) }} /></span><b className="mono">{c.cnn}</b></div>
                  <div className="dvw-bar2"><span className="dvw-track"><i className="is-lr" style={{ width: pct(d.lr_secondary.probabilities?.[c.cls] ?? 0) }} /></span><b className="mono">{c.lr}</b></div>
                </div>))}
              <span className="dvw-muted">{`${d.top_class === lrTop ? "Both rank the same class first." : "They rank different classes first."} The logistic regression is a second opinion and not an input to the decision.`}</span>
            </Panel>
            <LrPanel record={record} />
            <AttributionPanel record={record} />
            <div className="dvw-xpmeta" style={{ gridColumn: "1 / -1" }}>
              <b>Written explanation</b>
              {e ? <>
                <span><em>Source</em>{e.source === "llm" ? "language model" : "fixed template"}</span>
                <span><em>Prompt</em><span className="mono">{e.prompt_version}</span></span>
                <span><em>Latency</em><span className="mono">{`${Math.round(e.latency_ms).toLocaleString("en-US")} ms`}</span></span>
                <span><em>Fallback</em>{e.fallback_reason || "none"}</span>
              </> : <span className="dvw-muted">Preparing explanation…</span>}
              {onRegenerate && <button type="button" className="dvw-ghost" onClick={onRegenerate}>Regenerate explanation</button>}
            </div>
          </div>
        </Section>

        {/* 05 Decision */}
        <Section id="decision" n="05" title="Decision" sub={<span title={report?.assessment?.rule_tooltip} className="dvw-dotted">{report?.assessment?.rule_line || RULE_LINE}</span>}>
          <div className="dvw-dec">
            <span className="dvw-kick-s">Decision checks</span>
            {checks ? <div id="rd-decision-checks" tabIndex={-1} className="dvw-checks">{checks.map((c) => <CheckCard key={c.key} c={c} />)}</div> : <p className="dvw-muted">Loading the checks…</p>}
            <span className="dvw-muted">Each scale marks the check's threshold (bar) and this recording's value (dot). This is not a measure of damage severity.</span>
            <div className="dvw-ruleblock">
              <div className="dvw-col">
                <span className="dvw-kick-s">Rule in use · classifier first</span>
                <span className="mono dvw-formula">verdict ⇐ top class, if confidence ∧ familiarity; otherwise further inspection</span>
                <span className="dvw-muted">The verdict rests on the confidence and familiarity checks and the top class; the two healthy-baseline checks only add a caution to a healthy verdict.</span>
                <b className="dvw-verdict-line">{summary.outcome}{d.decision === "known_fault" ? ` · ${human(d.fault_class)}` : ""}</b>
                <span className="dvw-muted">{`${summary.confidence}. ${summary.familiarity}. ${summary.healthy}.`}{report?.assessment?.narrow_notes?.length ? " " + report.assessment.narrow_notes.join(" ") : " No narrow margins."}</span>
              </div>
              <div className="dvw-col">
                <span className="dvw-kick-s">{report?.assessment?.variants_summary || "Alternative rules"}</span>
                <div className="dvw-variants">{variants.map((v) => <div key={v.name} title={v.name}>
                  <span>{v.name}</span><b className="mono">{v.verdict === "unknown" ? "further inspection" : v.verdict === "healthy" ? "healthy" : human(v.verdict)}</b></div>)}</div>
                <span className="dvw-muted">Comparison only; the rule in use decides the verdict.</span>
              </div>
            </div>
          </div>
        </Section>

        {/* 06 Pipeline */}
        <Section id="pipeline" n="06" title="Pipeline" sub="How this diagnosis was made, read from the case's saved trace; nothing is re-run.">
          <DevPipeline caseId={caseId} initialTrace={trace} />
        </Section>

        {/* 07 Report */}
        <Section id="report" n="07" anchor="rd-fixed-report" title="Report" sub="Fixed template from the evidence record; no language model.">
          <div className="dvw-grid2">
            {report ? report.sections.map((sec) => (
              <Panel key={sec.n} title={`${sec.n} · ${sec.title}`}>
                {sec.items.map((it, i) => <div key={i} className="dvw-item" title={it.hover}>
                  <span className="tag" style={{ color: TAG[it.tag].c, background: TAG[it.tag].b }}>{it.tag}</span><span>{it.text}</span></div>)}
              </Panel>)) : <p className="dvw-muted">Loading the report…</p>}
            <Panel title="Traceability" span={2}>
              <div className="dvw-kv dvw-kv3">{[
                ["Recording", record.input?.rec_id || "uploaded recording"], ["Case", record.case_id], ["Created", record.created_at],
                ["Bundle manifest sha256", record.provenance.manifest_sha256.slice(0, 16) + "…"], ["Bundle git sha", String(record.provenance.bundle_git_sha).slice(0, 12)],
                ["Preprocess config sha256", record.provenance.preprocess_config_sha256.slice(0, 16) + "…"], ["Split hash", record.provenance.split_hash.slice(0, 16) + "…"],
                ["Schema version", record.provenance.schema_version], ["Diagnosis time", `${Math.round(record.timing_ms?.total || 0)} ms`],
              ].map(([k, v]) => <div key={k}><span>{k}</span><b className="mono" title={v}>{v}</b></div>)}</div>
            </Panel>
          </div>
        </Section>

        <footer className="dvw-foot">
          {record.source === "dataset" && <span className="dvw-insample">{IN_SAMPLE_NOTICE}</span>}
          <b>{M.LIMITATION}</b><span>{SCORES_NOTE}</span>
          {(record.limitations || []).length > 0 && <details><summary>Research limitations</summary>
            <ul>{record.limitations.map(cleanText).map((l) => <li key={l}>{l}</li>)}</ul></details>}
        </footer>
      </main>
    </div>
  );
}

const pct = (v) => `${Math.max(0, Math.min(1, v || 0)) * 100}%`;

function Inspector({ part, sceneState, rows, sel, chId, onChannel }) {
  const m = markerFor(part, sceneState);
  const mine = rows.filter((r) => part.ch.includes(r.id));
  return (
    <aside className="dvw-insp" aria-label="Selected part">
      <span className="dvw-pill" style={m.tag ? { color: m.tagColor, background: al(m.fill, 0.14) } : undefined}>{m.tag || "Nothing flagged"}</span>
      <b className="dvw-insp-name">{part.full || part.label}</b>
      <p>{part.desc}</p>
      {mine.length > 0 && <div className="dvw-col">
        <span className="dvw-kick-s">Related channels</span>
        {mine.map((r) => <button key={r.id} type="button" className={"dvw-insp-ch" + (r.id === chId ? " on" : "")} onClick={() => onChannel(r.id)}>
          <span><i style={{ background: r.beyond ? K.lime : "oklch(0.23 0.006 260 / 0.2)" }} />{r.name}</span>
          <b className="mono">{r.comparable ? `${D.METRIC_NAME[r.metric] || r.metric} ${D.changeText(r)}` : "no comparable metric"}</b></button>)}
      </div>}
      <span className="dvw-muted dvw-insp-hint">{sel ? "Selected on the rig. The data below follows this part." : "Showing the model's location (or a changed sensor). Select a part or sensor on the rig."}</span>
    </aside>
  );
}

function Channels({ channels, rows, sel, chId, span, onRow }) {
  if (!channels) return <Panel title="Sensor channels vs a healthy recording" span={span}><p className="dvw-muted">Loading the channel comparison…</p></Panel>;
  const n = rows.filter((r) => r.beyond).length;
  return (
    <Panel id="rd-sensor-channels" title="Sensor channels vs a healthy recording" span={span} kicker={<span className="dvw-kick-s">{`${n} of ${rows.length} beyond typical variation`}</span>}>
      <span className="dvw-muted">{`Reference: healthy training recording ${channels.reference.rec_id} at ${channels.reference.speed_hz.toFixed(2)} Hz · each channel's most changed measurement`}</span>
      <div className="dvw-chrows">{rows.map((r) => {
        if (!r.comparable) return <div key={r.id} className="dvw-chrow is-off"><span>{r.name}</span><span className="dvw-muted">no comparable reading</span><span /></div>;
        const v = D.logPos(r.ratio), one = D.logPos(1), b0 = D.logPos(1 / r.band), b1 = D.logPos(r.band);
        return <button key={r.id} type="button" className={"dvw-chrow" + (r.id === chId || sel === r.sensor ? " on" : "")} onClick={() => onRow(r)}
          aria-label={`${r.name}: ${r.beyond ? "beyond" : "within"} typical healthy variation`}>
          <span className="dvw-chname"><i style={{ background: r.beyond ? K.lime : "oklch(0.23 0.006 260 / 0.2)" }} />{r.name}</span>
          <span className="dvw-chtrack"><span className="dvw-axis" /><span className="dvw-band" style={{ left: b0 + "%", width: b1 - b0 + "%" }} />
            <span className="dvw-one" style={{ left: one + "%" }} />
            <span className="dvw-bar" style={{ left: Math.min(one, v) + "%", width: Math.abs(v - one) + "%", background: r.beyond ? K.lime : "oklch(0.23 0.006 260 / 0.25)" }} />
            <span className="dvw-knob" style={{ left: v + "%", background: r.beyond ? K.lime : K.white }} />
            {D.overflow(r.ratio) && <span className={"mono dvw-over is-" + D.overflow(r.ratio).side} title="Beyond the axis">{D.overflow(r.ratio).text}</span>}</span>
          <b className="mono" style={{ color: r.beyond ? K.limeT : K.ink2 }}>{`${D.changeText(r)} ${D.METRIC_NAME[r.metric] || r.metric}`}</b>
        </button>;
      })}</div>
      <div className="dvw-chrow dvw-chfoot"><span className="dvw-muted"><i className="dvw-band-key" />Typical healthy variation</span>
        <span className="dvw-chtrack">{D.CH_TICKS.map(([x, l]) => <span key={l} className="mono dvw-tick" style={{ left: D.logPos(x) + "%" }}>{l}</span>)}</span><span /></div>
      <span className="dvw-muted">Measured differences from a healthy recording at a similar speed: descriptive, not a calibrated anomaly test. They show where the signal changed, not the fault location.</span>
    </Panel>
  );
}

const MAXO = 10, ENV_MAXO = 12;
function Spectrum({ record, signals, part, chId, chOptions, mode, onMode, onChannel, span }) {
  const fr = record.signal_quality.shaft_speed_hz;
  const env = record.physics.envelope_spectrum;
  const hasEnv = !!env?.channels?.[chId];
  const m = mode === "envelope" && hasEnv ? "envelope" : "order";
  let chart = null;
  if (m === "envelope") {
    const ys = env.channels[chId], ymax = rangeMax(env.orders, ys, 0.3, ENV_MAXO) * 1.15, p = linePath(env.orders, ys, 0, ENV_MAXO, ymax);
    const cur = record.physics.cursors_orders;
    const cs = [["ftf", "FTF", cur.ftf[0]], ["shaft", "1×", 1], ["bpfo", "BPFO", cur.bpfo[0]], ["bsf", "2×BSF", cur.bsf[1]], ["bpfi", "BPFI", cur.bpfi[1]]];
    chart = { p, ymax, max: ENV_MAXO, ticks: [0, 3, 6, 9, 12], cursors: cs.map(([k, l, o]) => ({ k, l, left: (o / ENV_MAXO) * 100, c: k === "shaft" ? K.ink3 : K.orangeT })),
      caption: `${CH[chId].name} · envelope spectrum (${env.band_hz[0] / 1000}–${env.band_hz[1] / 1000} kHz band) · 1× = ${fr.toFixed(2)} Hz`, note: D.BEARING_NOTE };
  } else if (signals?.channels?.[chId]) {
    const s = signals.channels[chId], ymax = rangeMax(signals.orders, s.spectrum, 0.5, MAXO) * 1.15, p = linePath(signals.orders, s.spectrum, 0, MAXO, ymax);
    chart = { p, ymax, max: MAXO, ticks: [0, 2, 4, 6, 8, 10], cursors: [1, 2, 3].map((o) => ({ k: o, l: `${o}×`, left: (o / MAXO) * 100, c: K.ink3 })),
      caption: `${CH[chId].name} · order spectrum · 1× = ${fr.toFixed(2)} Hz · V`, note: "Shaft orders are evidence for imbalance and misalignment." };
  }
  return (
    <Panel title="Frequency spectrum" span={span} kicker={<div className="dvw-row">
      {chOptions.length > 1 && chOptions.map((c) => <button key={c} type="button" className={"dvw-chip" + (c === chId ? " on" : "")} onClick={() => onChannel(c)}>{CH[c].name.split(" · ").pop()}</button>)}
      <div className="dvw-seg">{[["order", "Order"], ["envelope", "Envelope"]].map(([k, l]) => <button key={k} type="button" disabled={k === "envelope" && !hasEnv}
        title={k === "envelope" && !hasEnv ? "The saved envelope spectrum covers the radial channels" : undefined} className={m === k ? "on" : ""} onClick={() => onMode(k)}>{l}</button>)}</div></div>}>
      {chart ? <>
        <span className="dvw-muted">{chart.caption}</span>
        <div className="dvw-plot"><Svg paths={[{ d: chart.p.area, fill: al(K.orange, 0.1) }, { d: chart.p.line, stroke: K.orange, w: 1.4 }]} />
          {chart.cursors.map((c, i) => <span key={c.k}><span className="dvw-cur" style={{ left: c.left + "%", borderColor: c.c }} />
            <span className="mono dvw-curl" style={{ left: c.left + "%", color: c.c, top: -15 - (m === "envelope" ? (i % 2) * 11 : 0) }}>{c.l}</span></span>)}
          {chart.ticks.map((t) => <span key={t} className="mono dvw-xt" style={{ left: (t / chart.max) * 100 + "%" }}>{t ? `${t}×` : "0"}</span>)}
          <span className="mono dvw-yt" style={{ top: -6 }}>{f3(chart.ymax)}</span><span className="mono dvw-yt" style={{ bottom: -6 }}>0</span></div>
        <span className="dvw-muted">{`This recording only. No healthy-reference spectrum is stored with the case. ${chart.note}`}</span>
      </> : <p className="dvw-muted">{part.ch.includes(chId) ? "The stored signal is not available for this channel." : "Loading the stored signal…"}</p>}
    </Panel>
  );
}

function Waveform({ signals, chId }) {
  const s = signals?.channels?.[chId];
  if (!s) return <Panel title="Time waveform"><p className="dvw-muted">Loading the stored signal…</p></Panel>;
  const w = s.waveform, mx = Math.max(...w.max.map(Math.abs), ...w.min.map(Math.abs)) * 1.15 || 1, dur = w.dt_s * w.max.length;
  return (
    <Panel title="Time waveform">
      <span className="dvw-muted">{`${CH[chId].name} · ${dur.toFixed(2)} s excerpt of the cleaned recording (V)`}</span>
      <div className="dvw-plot"><Svg paths={[{ d: bandPath(w.min, w.max, mx), fill: al(K.ink, 0.5), stroke: K.ink, w: 0.6 }]} />
        {[0, dur / 2, dur].map((t) => <span key={t} className="mono dvw-xt" style={{ left: (t / dur) * 100 + "%" }}>{`${(w.t0_s + t).toFixed(2)} s`}</span>)}
        <span className="mono dvw-yt" style={{ top: -6 }}>{`+${f3(mx)}`}</span><span className="mono dvw-yt" style={{ bottom: -6 }}>{`−${f3(mx)}`}</span></div>
      <span className="dvw-muted">Minimum and maximum per time bucket of the stored, cleaned signal.</span>
    </Panel>
  );
}

function LrPanel({ record }) {
  const lr = record.xai?.lr;
  if (!lr) return <Panel title="Feature contributions"><p className="dvw-muted">Not available for this case.</p></Panel>;
  const feats = lr.top_features.slice(0, 5), mx = Math.max(1e-9, ...feats.map((f) => Math.abs(f.contribution)));
  return (
    <Panel title="Feature contributions" kicker={<span className="dvw-kick-s">{`Logistic regression · ${human(lr.class)}`}</span>}>
      <div className="dvw-stack">{feats.map((f) => {
        const w = (Math.abs(f.contribution) / mx) * 50;
        return <div key={f.feature} className="dvw-feat" title={f.definition || f.feature}>
          <div className="dvw-row-b"><span className="mono">{f.feature}</span><b className="mono">{`${f.contribution >= 0 ? "+" : "−"}${Math.abs(f.contribution).toFixed(2)}`}</b></div>
          <span className="dvw-div"><i /><span style={{ left: f.contribution >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`, background: f.contribution >= 0 ? K.violet : "oklch(0.23 0.006 260 / 0.3)" }} /></span>
        </div>;
      })}</div>
      <span className="dvw-muted">Top five contributions toward the logistic regression's top class; a second opinion, not an input to the decision.</span>
    </Panel>
  );
}

function AttributionPanel({ record }) {
  const cnn = record.xai?.cnn;
  if (!cnn) return <Panel title="CNN attribution"><p className="dvw-muted">Not available for this case.</p></Panel>;
  const mx = Math.max(1e-9, ...cnn.top_channels.map((c) => c.share));
  return (
    <Panel title="CNN attribution · exploratory">
      <div className="dvw-stack">{cnn.top_channels.map((c) => <div key={c.channel} className="dvw-attr">
        <span>{CH[c.channel]?.name || c.channel}</span><span className="dvw-track"><i style={{ width: `${(c.share / mx) * 100}%` }} /></span><b className="mono">{c.share.toFixed(2)}</b></div>)}</div>
      <span className="dvw-muted">{`Strongest order bands: ${cnn.top_order_bands.map((b) => `${b.orders[0].toFixed(1)}–${b.orders[1].toFixed(1)}×`).join(", ")}.`}</span>
      <span className="dvw-muted">{`${cnn.label}. Shows what the network weighted. Not evidence of location.`}</span>
    </Panel>
  );
}

// One check: its value against its threshold on a simple scale. Confidence and familiarity decide (pass / does not pass);
// the two healthy-baseline checks only describe the recording and add a caution to a healthy verdict.
function CheckCard({ c }) {
  const healthy = c.key === "ae" || c.key === "b1s";
  const status = healthy ? (c.passed ? "Like healthy" : "Unlike healthy") : c.passed ? "Passes" : "Does not pass";
  const tone = healthy ? "is-info" : c.passed ? "is-ok" : "is-warn";
  const lo = Math.min(c.value, c.threshold), hi = Math.max(c.value, c.threshold), span = hi - lo || Math.abs(hi) || 1;
  const a = lo - span * 0.6, b = hi + span * 0.6, pos = (x) => ((x - a) / (b - a)) * 100;
  const plain = { msp: c.passed ? "The classifier strongly prefers its best match." : "The classifier is not confident enough in its best match.",
    maha: c.passed ? "The recording resembles the training data." : "The recording is unlike the training data.",
    ae: c.passed ? "Like the healthy recordings." : "Unlike the healthy recordings.",
    b1s: c.passed ? "Like healthy recordings at this speed." : "Unlike healthy recordings at this speed." }[c.key];
  return (
    <div className="dvw-check">
      <div className="dvw-row-b"><div className="dvw-col"><b>{PRIMARY[c.key]}</b><span className="dvw-muted">{TECH[c.key]}</span></div>
        <span className={"dvw-status " + tone}>{status}</span></div>
      <span className="mono dvw-expr">{D.checkExpr(c)}</span>
      <div className="dvw-scale" aria-hidden="true"><span className="dvw-axis" /><span className="dvw-thr" style={{ left: pos(c.threshold) + "%" }} />
        <span className="dvw-val" style={{ left: pos(c.value) + "%" }} /></div>
      <span className="dvw-plain">{plain}</span>
      <span className="mono dvw-muted">{`${c.passed ? "inside" : "outside"} the threshold by ${Math.abs(c.margin_abs).toPrecision(3)}${c.narrow ? " · narrow margin" : ""}`}{healthy ? " · caution only" : ""}</span>
    </div>
  );
}
