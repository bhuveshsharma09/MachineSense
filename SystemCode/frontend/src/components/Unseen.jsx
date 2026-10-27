// App-6: "Explore an unseen fault" — a saved leave-one-class-out CNN-6 run on a recording of the class it never trained
// on. A SIMULATED unseen fault inside MAFAULDA. Every number comes from the backend (GET /api/unseen, /api/unseen/run):
// the checks are ml.rejection.failed_checks and the verdict ml.arbitrator.decide; nothing is computed here. Stage App-8.5:
// laid out as the designer's four-step wizard (hide a fault, choose a recording, run the checks, see the result).
import { useEffect, useRef, useState } from "react";
import { getUnseen, runUnseen } from "../api.js";
import { fmtCheck, human, partsForClass } from "../lib/model.js";
import { PART_PLAIN } from "../lib/mechanic.js";
import { usePanelView } from "./Aside.jsx";
import { Logo } from "./TopBar.jsx";
import { PartView } from "./Dashboard.jsx";
import "../unseen.css";

export const STEPS = ["label", "confidence", "familiarity", "checks", "verdict"];
const ICON = { pass: "✓", fail: "✗", na: "–", info: "•" };
const STEP_MS = 750;
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const name = (c) => (c === "normal" ? "healthy (normal)" : human(c));
// a class name inside a sentence ("trained without imbalance recordings")
export const low = (c) => { const h = human(c); return h.charAt(0).toLowerCase() + h.slice(1); };

// Stage App-8.7 items 6, 7, 10: Mechanic shows a check as passed / failed with a value-vs-limit bar and no numbers;
// Developer shows the value and its limit in the dashboard's check format (fmtCheck: 6 decimals in [0, 1], never 1).
export const LIMITS_NOTE = "Each experimental model has its own limits, calibrated without the hidden fault.";
const ENDS = { confidence: ["less sure", "more sure"], familiarity: ["close to learned patterns", "far from learned patterns"] };

// Stage App-8.7 items 8, 9: the two healthy-machine checks add a caution to a healthy verdict only; for any other verdict
// they are not applied. Plain names as the pipeline page uses them.
export const HEALTHY = [["ae", "Healthy-pattern check", "AE-6 reconstruction error", "τ_e"], ["b1s", "Speed-aware healthy check", "B1s speed-aware baseline", "τ_b"]];
export const healthyState = (res, k) => (res.decision.decision !== "healthy" ? "na" : res.checks[k].passed ? "pass" : "fail");
export function cautionText(caution, level, gatesNote) {
  if (level === "developer") return `A healthy-baseline check disagrees (${caution.replace("_", " ")}). ${gatesNote}`;
  return caution === "both" ? "Both healthy-machine checks disagree, so this healthy result carries a caution."
    : "One of the two healthy-machine checks disagrees, so this healthy result carries a caution.";
}

function severityText(r) {
  if (r.true_class === "normal") return "";
  return r.true_class.includes("misalignment") ? `${r.severity} mm` : `${r.severity} g`;
}

export function exampleLabel(e, i, hidden) {
  if (e.role === "familiar") return `Familiar comparison · ${human(e.true_class)}`;
  return `Recording ${i + 1} of the hidden fault (${low(hidden)})`;
}

// Stage App-8.7 item 1: the recording's role for the main (ten-class) model and for this experimental model. The backend's
// used_for is the main model's role (split_closed); the experimental model never saw the hidden class, and its fit set is
// closed train within the development pool minus that class (§12a), so a familiar example (never one of its calibration
// recordings, by may_show) was used for its training.
export function rolesText(res) {
  const main = res.used_for === "threshold calibration" ? "used for threshold calibration" : "used for training";
  return `Main model: ${main}. This experimental model: ${res.role === "withheld" ? "never saw this class" : main}.`;
}

// Stage App-8.7 item 4: the share of this hidden fault's eligible development recordings the checks stopped, from the same
// saved counts as Experiment details (a share below 100 % never rounds to 100 %).
export function rateLine(k) {
  const n = k.withheld_recordings, s = k.rejected, w = k.accepted_as_a_wrong_fault, h = k.accepted_as_healthy;
  const pct = s === n ? 100 : s === 0 ? 0 : Math.min(99, Math.max(1, Math.round((100 * s) / n)));
  return `For this hidden fault, the checks stopped ${s} of ${n} eligible development recordings (${pct} %). `
    + `${w} got a wrong familiar label; ${h} ${h === 1 ? "was" : "were"} called healthy.`;
}

// Stage App-8.7 item 15: the file name's nominal speed and the measured speed, each labelled
export const speedText = (id, measured) => [Number.isFinite(nominalHz(id)) && `nominal ${nominalHz(id).toFixed(1)} Hz (from the file name)`,
  Number.isFinite(measured) && `measured ${measured.toFixed(1)} Hz`].filter(Boolean).join(" · ");

// Stage App-8.7 item 14: the rule variants under the dashboard's names (backend.checks.VARIANT_NAMES); B is the rule in use
export const VARIANT_NAMES = {
  A: "classifier first, with an autoencoder override",
  B: "classifier first (the rule in use)",
  C: "strict: healthy also needs the autoencoder check",
  D: "strict: healthy also needs the speed-aware check",
};
const variantVerdict = (v) => (v === "unknown" ? "further inspection" : v === "healthy" ? "healthy" : human(v));

export function verdictText(d) {
  if (d.decision === "unknown") return "Further inspection needed";
  if (d.decision === "healthy") return d.caution ? "Healthy — with a caution" : "Healthy";
  return `A known condition: ${human(d.fault_class)}`;
}

// Stage App-8.5: the designer's four-step wizard. Groups and part names follow the app's labels; every number below is the
// backend's (the design's generated values are not used).
const GROUPS = [
  ["Rotor", ["imbalance"]],
  ["Alignment", ["horizontal_misalignment", "vertical_misalignment"]],
  ["Underhang bearing", ["underhang_ball", "underhang_cage", "underhang_outer"]],
  ["Overhang bearing", ["overhang_ball", "overhang_cage", "overhang_outer"]],
];
const SHORT = { imbalance: "Imbalance", horizontal_misalignment: "Horizontal misalignment", vertical_misalignment: "Vertical misalignment",
  underhang_ball: "Ball fault", underhang_cage: "Cage fault", underhang_outer: "Outer race", overhang_ball: "Ball fault",
  overhang_cage: "Cage fault", overhang_outer: "Outer race" };
const partNames = (c) => partsForClass(c).map((p, i) => (i ? PART_PLAIN[p].toLowerCase() : PART_PLAIN[p])).join(" and ");
// nominal shaft speed from the recording id (the file name is the nominal speed in Hz)
const nominalHz = (id) => Number(String(id).split("__").pop());
const PAGES = [["Hide a fault", "Pick one fault. An experimental model was trained without it, then shown recordings of it."],
  ["Choose a recording", "Start with a recording of the hidden fault. Try the familiar example afterward to compare."],
  ["Run the checks", "The model proposes a familiar label; two checks decide whether to accept it."],
  ["See the result", "What the model proposed, what the checks said, and what the recording really was."]];

const accepted = (d) => d.decision !== "unknown";

// Outcome words from the role and the verdict only (App-8.5 item 2).
export function outcome(res) {
  const d = res.decision, withheld = res.role === "withheld";
  if (withheld) {
    if (!accepted(d)) return { word: "Wrong familiar label stopped", good: true, text: `The checks stopped a wrong familiar label. The verdict does not name ${low(res.true_class)}.` };
    if (d.decision === "healthy") return { word: "Hidden fault called healthy", good: false, text: "The model accepted a healthy verdict for a fault it never saw: an honest failure of the checks." };
    return { word: "Wrong familiar label accepted", good: false, text: "The model accepted a familiar label for a fault it never saw: an honest failure of the checks." };
  }
  return accepted(d) ? { word: "Correct familiar label accepted", good: true, text: "The model learned this condition, and both checks support its label." }
    : { word: "Familiar label not accepted", good: false, text: "A check did not support the label of a condition this model learned." };
}

// value-vs-limit bar: the shaded zone is the accepted side, the bar the limit, the dot this recording (no numbers)
function Scale({ kind, raw, rawT, passed, small }) {
  const atLeast = kind === "confidence";
  const lo = Math.min(raw, rawT), hi = Math.max(raw, rawT), span = hi - lo || Math.abs(hi) || 1;
  const a = lo - span * 0.8, b = hi + span * 0.8, pos = (x) => ((x - a) / (b - a)) * 100;
  return <div className={"uw-scale-box" + (small ? " is-small" : "")} data-scale={kind}>
    <div className="uw-scale" aria-hidden="true"><span className="uw-axis" />
      <span className={"uw-zone" + (atLeast ? " is-right" : " is-left")} style={atLeast ? { left: pos(rawT) + "%" } : { width: pos(rawT) + "%" }} />
      <span className="uw-thr" style={{ left: pos(rawT) + "%" }} /><span className={"uw-dot" + (passed ? "" : " is-bad")} style={{ left: pos(raw) + "%" }} /></div>
    <div className="uw-ends"><span>{ENDS[kind][0]}</span><span>{ENDS[kind][1]}</span></div>
  </div>;
}

function CheckCard({ kind, title, plain, tech, check, rule, level }) {
  const atLeast = kind === "confidence";
  return (
    <section className="uw-check" data-check={kind}>
      <div className="uw-check-head">
        <div className="uw-col"><span className="uw-kick">{level === "developer" ? tech : title}</span><b>{plain}</b></div>
        <span className={"un-chip " + (check.passed ? "ok" : "bad")}>{check.passed ? "passed" : "failed"}</span>
      </div>
      <Scale kind={kind} raw={check.value} rawT={check.threshold} passed={check.passed} />
      {level === "developer" ? <>
        <div className="uw-measure">
          <span><em>measured</em><b className="mono" data-v="value">{fmtCheck(check.value)}</b></span>
          <span><em>{atLeast ? "required at least" : "allowed at most"}</em><b className="mono" data-v="threshold">{fmtCheck(check.threshold)}</b></span>
        </div>
        <span className="uw-small mono">{rule}</span></>
        : <span className="uw-small">{`The bar marks the limit; the dot is this recording. ${check.passed ? (atLeast ? "At or above the limit." : "Within the limit.") : (atLeast ? "Below the limit." : "Beyond the limit.")}`}</span>}
    </section>
  );
}

export function UnseenView({ summary, initialHidden, initialLevel, initialResult = null, initialStep = -1,
  initialExample = 0, initialPage = null, initialHistory = [], run }) {
  const runs = summary.runs || {};
  const classes = Object.keys(runs);
  const [hidden, setHidden] = useState(initialHidden || summary.default_hidden_class || classes[0]);
  const [ex, setEx] = useState(initialExample);
  const [level, setLevel] = usePanelView(initialLevel);
  const [result, setResult] = useState(initialResult);
  const [step, setStep] = useState(initialStep);
  const startPage = initialPage || (initialResult ? (initialStep >= STEPS.length - 1 ? 4 : 3) : 1);
  const [page, setPage] = useState(startPage);
  // App-8.7 item 3: step 4's result stays "Ready" in the rail until the user opens step 4
  const [opened, setOpened] = useState(startPage === 4);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState(initialHistory);
  const timer = useRef(null);
  const r = runs[hidden] || { examples: [], classes_learned: [] };
  const examples = r.examples || [];
  const withheldEx = examples.filter((e) => e.role === "withheld");
  const familiarEx = examples.filter((e) => e.role === "familiar");
  const ordered = [...withheldEx, ...familiarEx];
  const chosen = ordered[ex] || ordered[0];
  const missing = r.categories_without_an_eligible_recording || [];

  useEffect(() => () => clearInterval(timer.current), []);
  useEffect(() => { if (page === 4) setOpened(true); }, [page]);
  const reset = () => { clearInterval(timer.current); setResult(null); setStep(-1); setOpened(false); setError(null); };
  const go = async (e, i) => {
    if (!e || !run) return;
    reset();
    if (i !== undefined) setEx(i);
    setPage(3);
    setBusy(true);
    try {
      const res = await run(hidden, e.rec_id, e.role);
      setResult(res);
      setHistory((h) => [...h.filter((x) => !(x.hidden === hidden && x.rec_id === e.rec_id)), { hidden, rec_id: e.rec_id, role: e.role, res }]);
      setStep(0);
      let k = 0;
      timer.current = setInterval(() => { k += 1; setStep(k); if (k >= STEPS.length - 1) clearInterval(timer.current); }, STEP_MS);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "the example could not be run");
    } finally { setBusy(false); }
  };
  // the healthy-machine row is revealed with the checks row
  const shown = (s) => result && step >= STEPS.indexOf(s === "healthy" ? "checks" : s);
  const c = result?.checks;
  const d = result?.decision;
  const isWithheld = result?.role === "withheld";
  const o = result && outcome(result);
  const done = result && step >= STEPS.length - 1;
  const parts = partsForClass(hidden);
  const dev = level === "developer";
  const healthyRow = result?.proposed_label === "normal";
  const exLabel = (e) => exampleLabel(e, e.role === "withheld" ? withheldEx.indexOf(e) : ordered.indexOf(e), hidden);

  const rail = [
    [`${cap(human(hidden))} hidden`, true],
    [page >= 2 && chosen ? exLabel(chosen) : "Not chosen yet", page >= 2],
    [result ? (done ? "Checks run" : "Running…") : busy ? "Running…" : "Waiting for a recording", page >= 3],
    [done ? (opened ? verdictText(d) : "Ready") : "Not run yet", done],
  ];
  const nav = {
    1: { label: "Next: choose a recording", on: () => setPage(2), ok: true },
    2: { label: busy ? "Running…" : "Run the checks", on: () => go(chosen), ok: !!chosen && !busy && !!run },
    3: { label: "See the result", on: () => setPage(4), ok: !!done },
  }[page];
  // App-8.7 item 5: each row's mark shows its state (pass ✓, fail ✗, not applicable –, information •)
  const both = c && c.confidence.passed && c.familiarity.passed;
  const hs = result ? HEALTHY.map(([k]) => healthyState(result, k)) : [];
  const runRows = result ? [
    ["label", "Proposed label", cap(name(result.proposed_label)), `Its best answer among the ${result.classes_learned.length} conditions it learned.`, "info"],
    ["confidence", "Confidence check", c.confidence.passed ? "Passed" : "Failed", dev
      ? `MSP ${fmtCheck(c.confidence.value)}, ${c.confidence.passed ? "at least" : "below"} τ_p ${fmtCheck(c.confidence.threshold)}`
      : <Scale kind="confidence" raw={c.confidence.value} rawT={c.confidence.threshold} passed={c.confidence.passed} small />, c.confidence.passed ? "pass" : "fail"],
    ["familiarity", "Familiarity check", c.familiarity.passed ? "Passed" : "Failed", dev
      ? `Mahalanobis ${fmtCheck(c.familiarity.value)}, ${c.familiarity.passed ? "within" : "above"} τ_d ${fmtCheck(c.familiarity.threshold)}`
      : <Scale kind="familiarity" raw={c.familiarity.value} rawT={c.familiarity.threshold} passed={c.familiarity.passed} small />, c.familiarity.passed ? "pass" : "fail"],
    ["checks", "Both checks", both ? "Both passed: the proposed label is accepted" : "Not both passed: the proposed label is not accepted", null, both ? "pass" : "fail"],
    ...(healthyRow ? [["healthy", "Healthy-machine checks (caution only)",
      d.decision !== "healthy" ? "Not applied: the verdict is not healthy" : hs.every((x) => x === "pass") ? "Both agree: no caution"
        : `${hs.every((x) => x === "fail") ? "Both disagree" : "One disagrees"}: the healthy verdict carries a caution`,
      <span className="uw-hc">{HEALTHY.map(([k, plain, tech, tau], i) => <span key={k} data-hc={k} data-state={hs[i]}>
        <i className={"uw-ico is-" + hs[i]} aria-hidden="true">{ICON[hs[i]]}</i>
        {`${dev ? tech : plain}: ${{ pass: "passed", fail: "failed", na: "not applicable" }[hs[i]]}`}
        {dev && ` (${fmtCheck(c[k].value)} vs ${tau} ${fmtCheck(c[k].threshold)})`}</span>)}</span>,
      d.decision !== "healthy" ? "na" : hs.includes("fail") ? "fail" : "pass"]] : []),
    ["verdict", "Verdict", verdictText(d), null, "info"],
  ] : [];

  return (
    <div className="uw-root">
      <header className="uw-header">
        <a className="pl-brand ms-brand" href="#/" aria-label="MachineSense home"><Logo /><span style={{ fontSize: 17, fontWeight: 500 }}>MachineSense</span></a>
        <a className="uw-back" href="#/"><span aria-hidden="true">‹</span>Back to the app</a>
        <div style={{ flex: 1 }} />
        <div role="radiogroup" aria-label="Reading level" className="seg">
          {[["mechanic", "Mechanic"], ["developer", "Developer"]].map(([k, lab]) => (
            <button key={k} role="radio" aria-checked={level === k} className={level === k ? "on" : ""} onClick={() => setLevel(k)}>{lab}</button>
          ))}
        </div>
      </header>

      <div className="uw-title">
        <span className="uw-eyebrow">Explore an unseen fault</span>
        <h1>What happens when a machine fault appears that this model was never taught?</h1>
        <p>Hide one fault from training, choose a recording, and see whether the model accepts a familiar label or asks for further inspection.</p>
        <button type="button" className="uw-about-link" onClick={() => document.getElementById("uw-about")?.scrollIntoView({ behavior: "smooth", block: "start" })}>
          Research demonstration · about this page ↓</button>
      </div>

      <div className="uw-grid">
        <aside className="uw-left">
          <ol className="uw-rail">{PAGES.map(([t], i) => {
            const n = i + 1, reach = n === 1 || rail[i - 1][1] && (n !== 4 || done) && (n !== 3 || result);
            return <li key={t} className={(page === n ? "on" : "") + (reach ? "" : " is-off")}>
              <button type="button" disabled={!reach} aria-current={page === n ? "step" : undefined} onClick={() => setPage(n)}>
                <span className="uw-rail-n">{n}</span><span className="uw-col"><b>{t}</b><span>{rail[i][0]}</span></span>
              </button></li>;
          })}</ol>
          <section className="uw-about" id="uw-about">
            <span className="uw-kick">About this demonstration</span>
            <p className="un-simulated" data-copy="simulated">{summary.simulated}</p>
            <p className="un-easier" data-copy="easier">{summary.easier_part}</p>
            <b>Research demonstration; not validated on real machines.</b>
          </section>
        </aside>

        <section className="uw-main" aria-live="polite">
          <div className="uw-main-head">
            <span className="uw-kick">{`Step ${page} of 4`}</span>
            <h2>{PAGES[page - 1][0]}</h2>
            <p>{PAGES[page - 1][1]}</p>
          </div>

          {page === 1 && <div className="uw-p1">
            <div className="uw-col">
              {GROUPS.map(([g, ids]) => <div key={g} className="uw-group"><span className="uw-kick">{g}</span>
                <div className="uw-faults">{ids.filter((k) => runs[k]).map((k) => <button key={k} type="button" aria-pressed={hidden === k}
                  className={"uw-fault" + (hidden === k ? " on" : "")} onClick={() => { setHidden(k); setEx(0); reset(); }}>
                  <b>{SHORT[k]}</b><span>{partNames(k)}</span>{hidden === k && <em>Hidden from training</em>}</button>)}</div>
              </div>)}
              <p className="un-copy" data-copy="trained-without">This experimental model was trained without {low(hidden)} recordings.</p>
              {missing.map((m) => <p key={m} className="uw-small" data-copy="missing">No {m} among the eligible recordings: {r.why_missing?.[m] || "none met the conditions"}.</p>)}
            </div>
            <div className="uw-col">
              <div className="uw-rig">
                <PartView key={parts.join()} partId={parts[0]} sceneState={{ fault: parts, cand: [], unusual: [], flags: {} }} height={250}>
                  <div className="uw-rig-label"><span>Where this fault occurs (dataset label)</span><b>{partNames(hidden)}</b></div>
                  <span className="uw-rig-note">Illustrative rig</span>
                </PartView>
              </div>
              <div className="uw-learned-box">
                <b>{`This model learned ${(r.classes_learned || []).length} conditions`}</b>
                <ul className="uw-learned">
                  {(r.classes_learned || []).map((k) => <li key={k}>{name(k)}</li>)}
                  <li className="un-not-learned">{human(hidden)} — never seen</li>
                </ul>
              </div>
            </div>
          </div>}

          {page === 2 && <div className="uw-p2">
            <div className="uw-recs" role="radiogroup" aria-label="Example recording">
              {ordered.map((e, i) => { const prev = history.find((x) => x.hidden === hidden && x.rec_id === e.rec_id);
                return <button key={e.rec_id + e.role} type="button" role="radio" aria-checked={ex === i} className={"uw-rec" + (ex === i ? " on" : "")}
                  onClick={() => { setEx(i); reset(); }}>
                  <span className={"uw-tag" + (e.role === "withheld" ? " is-hidden" : "")}>{e.role === "withheld" ? "Hidden fault" : "Familiar condition"}</span>
                  <b>{exLabel(e)}</b>
                  <span>{e.role === "withheld" ? `A recording of ${low(hidden)}, which this model never saw.`
                    : e.true_class === "normal" ? "A healthy recording (a condition this model learned)." : `A recording of ${low(e.true_class)} (a condition this model learned).`}</span>
                  <span className="uw-rec-meta"><span className="mono">{e.rec_id}</span>
                    <span data-v="speed">{speedText(e.rec_id, prev?.res.speed_hz)}</span>
                    {prev && <span className="uw-done">{`✓ Already run · ${verdictText(prev.res.decision)}`}</span>}</span>
                </button>; })}
            </div>
            {missing.map((m) => <p key={m} className="uw-small" data-copy="missing">No {m} among the eligible recordings: {r.why_missing?.[m] || "none met the conditions"}.</p>)}
          </div>}

          {page === 3 && <div className="uw-p3">
            {error && <p className="un-error" role="alert">{error}</p>}
            <div className="uw-col"><ol className="uw-run">{(result ? runRows : STEPS.map((s) => [s, cap(s), null, null, null])).map(([s, title, res, detail, state]) => {
              const on = shown(s);
              return <li key={s} data-step={s} className={"uw-run-row" + (on ? " is-shown" : "")}>
                <span className={"uw-run-mark" + (on ? " is-" + state : busy || result ? " is-wait" : "")} data-state={on ? state : undefined}
                  aria-hidden="true">{on ? ICON[state] : ""}</span>
                <div className="uw-col"><b>{title}</b>{on && <><span className="uw-run-res">{res}</span>{detail && <span className="uw-small">{detail}</span>}</>}</div>
              </li>; })}</ol>
            <p className="uw-small" data-copy="limits">{LIMITS_NOTE}</p></div>
            <div className="uw-facts"><span className="uw-kick">This run</span>
              <div><span>Hidden fault</span><b>{human(hidden)}</b></div>
              <div><span>Recording</span><b className="mono">{chosen?.rec_id || "—"}</b></div>
              <div><span>Model</span><b>{level === "developer" ? `leave-one-class-out CNN-6, seed ${summary.seed}` : `trained without ${low(hidden)}`}</b></div>
            </div>
          </div>}

          {page === 4 && result && <div className="uw-p4">
            <div className="uw-res-top">
              <section className={"uw-verdict" + (d.decision === "unknown" ? " is-unknown" : "")}>
                <span className="uw-kick">Verdict</span>
                <h3 data-v="verdict">{verdictText(d)}</h3>
                <p>{d.decision === "unknown" ? "At least one check did not support the proposed familiar label."
                  : d.decision === "healthy" ? "The decision rule accepted the healthy label." : "The decision rule accepted a learned condition."}</p>
                {d.decision === "unknown" && <p className="un-copy" data-copy="further">{summary.further_inspection}</p>}
                {d.caution && <p className="uw-small" data-copy="caution">{cautionText(d.caution, level, summary.gates_note)}</p>}
                {level === "developer" && <div className="uw-rules" data-v="rules"><span className="uw-small">Decision rules (comparison only; the rule in use decides)</span>
                  {Object.entries(result.rule_comparison).filter(([k]) => VARIANT_NAMES[k]).map(([k, v]) => <span key={k} className={"uw-small" + (k === "B" ? " is-used" : "")}
                    title={k === "B" ? d.rule_variant : k}>{`${VARIANT_NAMES[k]}: ${variantVerdict(v)}`}{k === "B" && <b> · in use</b>}</span>)}</div>}
              </section>
              {/* App-8.7 item 2 (choice b): step 1 names the hidden fault, so the actual condition is shown directly */}
              <section className="uw-truth is-open">
                <div data-v="truth" className="uw-col">
                  <span className="uw-kick">Actual condition (dataset label)</span>
                  <b className="uw-truth-name">{cap(name(result.true_class))}</b>
                  <span className="uw-small">{isWithheld ? "Hidden from this model during fitting: it never saw this condition." : "A condition this model learned."}</span>
                  <span className={"uw-outcome" + (o.good ? " is-good" : " is-bad")}><b>{o.word}</b><span className="un-copy">{o.text}</span></span>
                </div>
              </section>
            </div>
            <div className="uw-res-mid">
              <section className="uw-proposed">
                <span className="uw-kick">{level === "developer" ? "CNN-6 top class (leave-one-class-out model)" : "The label the model proposes"}</span>
                <h3 data-v="proposed">{cap(name(result.proposed_label))}</h3>
                <span className="uw-small">Its best answer among the {result.classes_learned.length} conditions it learned.</span>
                {level === "developer" && <ol className="un-top mono">{result.top_probabilities.map(([k, p]) => <li key={k}>{k}: {fmtCheck(p)}</li>)}</ol>}
                <p className="uw-sum" data-v="checks">{c.confidence.passed && c.familiarity.passed ? "Both checks passed, so the proposed label is accepted."
                  : `${cap([!c.confidence.passed && "the confidence check", !c.familiarity.passed && "the familiarity check"].filter(Boolean).join(" and "))} failed, so the proposed label is not accepted.`}</p>
              </section>
              <div className="uw-col">
                <div className="uw-checks">
                  <CheckCard kind="confidence" title="Confidence" plain="How sure is its best answer?" tech="MSP (maximum softmax probability)"
                    check={c.confidence} rule="MSP ≥ τ_p" level={level} />
                  <CheckCard kind="familiarity" title="Familiarity" plain="How close is this to learned patterns?" tech="Mahalanobis distance (shared Ledoit–Wolf)"
                    check={c.familiarity} rule="distance ≤ τ_d" level={level} />
                </div>
                <p className="uw-small" data-copy="limits">{LIMITS_NOTE}</p>
              </div>
            </div>
            {r.counts && <section className="uw-group-stat" aria-label="Hidden-fault group statistic">
              <span className="uw-kick">{`Group statistic · all ${r.counts.withheld_recordings} eligible ${low(hidden)} recordings · not this recording`}</span>
              <p className="uw-rate" data-v="rate">{rateLine(r.counts)}</p>
              <span className="uw-small" data-v="rate-scope">{isWithheld
                ? "This recording is one of them; its own result is shown above."
                : `This recording is a familiar ${low(result.true_class)} comparison and is not part of this count; its own result is shown above.`}</span>
            </section>}
            <div className="uw-actions">
              {familiarEx[0] && isWithheld && <button type="button" className="uw-dark" disabled={busy}
                onClick={() => go(familiarEx[0], ordered.indexOf(familiarEx[0]))}>Compare with a familiar recording</button>}
              <button type="button" className="uw-ghost" onClick={() => { reset(); setPage(2); }}>Try another recording</button>
              <button type="button" className="uw-ghost" onClick={() => { reset(); setPage(1); }}>Hide a different fault</button>
            </div>
            {history.length > 0 && <section className="uw-history">
              <div className="uw-hbar"><span className="uw-kick">Your runs</span>
                <button type="button" className="uw-ghost uw-clear" onClick={() => setHistory([])}>Clear runs</button></div>
              <div className="uw-hrow uw-hhead"><span>Hidden fault</span><span>Recording</span><span>Proposed label</span><span>Confidence</span><span>Familiarity</span><span>Outcome</span></div>
              {history.map((x) => { const oo = outcome(x.res);
                return <button key={x.hidden + x.rec_id} type="button" className="uw-hrow" onClick={() => {
                  setHidden(x.hidden); setEx(Math.max(0, [...runs[x.hidden].examples.filter((e) => e.role === "withheld"), ...runs[x.hidden].examples.filter((e) => e.role === "familiar")].findIndex((e) => e.rec_id === x.rec_id)));
                  setResult(x.res); setStep(STEPS.length - 1); setPage(4); }}>
                  <span data-v="hidden">{human(x.hidden)}</span><span className="mono">{x.rec_id}</span><span>{cap(name(x.res.proposed_label))}</span>
                  <span>{x.res.checks.confidence.passed ? "passed" : "failed"}</span><span>{x.res.checks.familiarity.passed ? "passed" : "failed"}</span>
                  <span className={oo.good ? "is-good" : "is-bad"}>{oo.word}</span></button>; })}
            </section>}
            <details className="uw-fold"><summary>How this decision works</summary>
              <div className="un-flow">
                <span className="un-node">Recording of the withheld fault</span><span className="un-arrow">↓</span>
                <span className="un-node">CNN trained without that fault</span><span className="un-arrow">↓</span>
                <span className="un-node">proposed familiar label</span><span className="un-arrow">↓</span>
                <div className="un-split"><span className="un-node">confidence check</span><span className="un-node">familiarity check</span></div>
                <div className="un-out"><span><b>both pass</b> → accept the proposed label</span><span><b>either fails</b> → further inspection needed</span></div>
                {healthyRow && <><span className="un-arrow">↓</span>
                  <span className="un-node" data-v="healthy-node">{`if the accepted label is healthy: healthy-machine checks (caution only) — ${HEALTHY.map(([k, plain, tech], i) =>
                    `${dev ? tech : plain.toLowerCase()} ${{ pass: "passed", fail: "failed", na: "not applicable" }[hs[i]]}`).join(", ")}`}</span>
                  <div className="un-out"><span><b>both pass</b> → healthy</span><span><b>either fails</b> → healthy, with a caution</span></div></>}
              </div>
            </details>
            <details className="uw-fold un-details"><summary>Experiment details</summary>
              <dl>
                <dt>Withheld class</dt><dd>{human(hidden)}</dd>
                <dt>Run</dt><dd>leave-one-class-out CNN-6, seed {summary.seed}</dd>
                <dt>Recording</dt><dd className="mono">{chosen?.rec_id || "—"}</dd>
                <dt>Recording details</dt><dd data-v="speed">{[severityText(result), speedText(result.rec_id, result.speed_hz)].filter(Boolean).join(" · ")}</dd>
                <dt>Role in training</dt><dd data-v="roles">{rolesText(result)}</dd>
                <dt>Example type</dt><dd>{chosen?.category}{chosen?.failure_kind ? ` (${chosen.failure_kind})` : ""}</dd>
                {r.counts && <><dt>This model on its hidden class</dt><dd data-v="counts">{r.counts.withheld_recordings} eligible recordings: {r.counts.rejected} not accepted,
                  {" "}{r.counts.accepted_as_a_wrong_fault} accepted as a wrong fault, {r.counts.accepted_as_healthy} accepted as healthy</dd></>}
                <dt>Values</dt><dd>{result.source === "live" ? "computed now from the saved models" : "saved values (the model weights are not present here)"}</dd>
                <dt>Final manifest</dt><dd className="mono" title={summary.final_manifest_sha256}>{summary.final_manifest_sha256?.slice(0, 16)}…</dd>
                {level === "developer" && <><dt>Healthy-baseline checks</dt><dd className="mono">AE {fmtCheck(result.scores.ae)} / τ_e {fmtCheck(result.thresholds.tau_e)} · B1s {fmtCheck(result.scores.b1s)} / τ_b {fmtCheck(result.thresholds.tau_b)} — {summary.gates_note}</dd></>}
                <dt>Categories without an eligible recording</dt><dd>{missing.length ? missing.map((m) => `${m} (${r.why_missing?.[m] || "none"})`).join("; ") : "none"}</dd>
              </dl>
            </details>
          </div>}

          <footer className="uw-foot">
            {page > 1 ? <button type="button" className="uw-ghost" onClick={() => setPage(page - 1)}>‹ Back</button> : <span />}
            <div className="uw-dots" aria-hidden="true">{[1, 2, 3, 4].map((n) => <i key={n} className={n === page ? "on" : n < page ? "past" : ""} />)}</div>
            {nav ? <button type="button" className="uw-dark" disabled={!nav.ok} onClick={nav.on}>{nav.label}</button> : <span />}
          </footer>
        </section>
      </div>
    </div>
  );
}

export default function Unseen() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { getUnseen().then(setSummary).catch((e) => setError(e?.message || "could not load")); }, []);
  if (error) return <div className="un-root"><p className="un-error" role="alert">The unseen-fault page could not be loaded ({error}).</p></div>;
  if (!summary) return <div className="un-root"><p className="un-placeholder">Loading…</p></div>;
  if (!summary.built) return <div className="un-root"><p className="un-error" role="alert">{summary.note}</p></div>;
  return <UnseenView summary={summary} run={runUnseen} />;
}
