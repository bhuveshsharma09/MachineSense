// App-6: "Explore an unseen fault" — a saved leave-one-class-out CNN-6 run on a recording of the class it never trained
// on. A SIMULATED unseen fault inside MAFAULDA. Every number comes from the backend (GET /api/unseen, /api/unseen/run):
// the checks are ml.rejection.failed_checks and the verdict ml.arbitrator.decide; nothing is computed here.
import { useEffect, useRef, useState } from "react";
import { getUnseen, runUnseen } from "../api.js";
import { K } from "../lib/theme.js";
import { human } from "../lib/model.js";
import { usePanelView } from "./Aside.jsx";
import { Logo } from "./TopBar.jsx";

export const STEPS = ["label", "confidence", "familiarity", "checks", "verdict"];
const STEP_MS = 750;
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const name = (c) => (c === "normal" ? "healthy (normal)" : human(c));
// a class name inside a sentence ("trained without imbalance recordings")
export const low = (c) => { const h = human(c); return h.charAt(0).toLowerCase() + h.slice(1); };

// enough decimals that a value and its threshold read differently whenever they differ, and that a value below the
// maximum never reads as the maximum (a top probability of 0.99999 is not "100.00 %"); 2 to 8 decimals
export function pair(v, t, scale = 1, unit = "", max = null) {
  const atMax = (x, s) => max !== null && x < max && Number(s) >= max * scale;
  for (let d = 2; d <= 8; d++) {
    const a = (v * scale).toFixed(d), b = (t * scale).toFixed(d);
    if ((a !== b && !atMax(v, a) && !atMax(t, b)) || d === 8) return [a + unit, b + unit];
  }
}

function severityText(r) {
  if (r.true_class === "normal") return "";
  return r.true_class.includes("misalignment") ? `${r.severity} mm` : `${r.severity} g`;
}

export function exampleLabel(e, i, hidden) {
  if (e.role === "familiar") return `Familiar comparison · ${human(e.true_class)}`;
  return `Recording ${i + 1} of the hidden fault (${low(hidden)})`;
}

function CheckCard({ title, plain, tech, value, threshold, passed, rule, level, shown }) {
  return (
    <section className={"un-card un-check" + (shown ? " is-shown" : "")} aria-hidden={!shown} data-check={tech}>
      <div className="kicker">{level === "developer" ? tech : title}</div>
      <h3>{plain}</h3>
      <div className="un-measure">
        <div><span className="un-small">measured</span><b className="mono" data-v="value">{value}</b></div>
        <div><span className="un-small">{rule.includes("≥") ? "required at least" : "allowed at most"}</span><b className="mono" data-v="threshold">{threshold}</b></div>
        <span className={"un-chip " + (passed ? "ok" : "bad")}>{passed ? "passed" : "failed"}</span>
      </div>
      {level === "developer" && <p className="un-small mono">{rule}</p>}
    </section>
  );
}

function RuleDiagram() {
  return (
    <figure className="un-card un-diagram" aria-label="How the decision is made">
      <figcaption className="kicker">How the decision is made</figcaption>
      <div className="un-flow">
        <span className="un-node">Recording of the withheld fault</span><span className="un-arrow">↓</span>
        <span className="un-node">CNN trained without that fault</span><span className="un-arrow">↓</span>
        <span className="un-node">proposed familiar label</span><span className="un-arrow">↓</span>
        <div className="un-split"><span className="un-node">confidence check</span><span className="un-node">familiarity check</span></div>
        <div className="un-out"><span><b>both pass</b> → accept the proposed label</span><span><b>either fails</b> → further inspection needed</span></div>
      </div>
    </figure>
  );
}

export function verdictText(d) {
  if (d.decision === "unknown") return "Further inspection needed";
  if (d.decision === "healthy") return d.caution ? "Healthy — with a caution" : "Healthy";
  return `A known condition: ${human(d.fault_class)}`;
}

export function UnseenView({ summary, initialHidden, initialLevel, initialResult = null, initialStep = -1, initialRevealed = false,
  initialExample = 0, run }) {
  const runs = summary.runs || {};
  const classes = Object.keys(runs);
  const [hidden, setHidden] = useState(initialHidden || summary.default_hidden_class || classes[0]);
  const [ex, setEx] = useState(initialExample);
  const [level, setLevel] = usePanelView(initialLevel);
  const [result, setResult] = useState(initialResult);
  const [step, setStep] = useState(initialStep);
  const [revealed, setRevealed] = useState(initialRevealed);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);
  const r = runs[hidden] || { examples: [], classes_learned: [] };
  const examples = r.examples || [];
  const withheldEx = examples.filter((e) => e.role === "withheld");
  const familiarEx = examples.filter((e) => e.role === "familiar");
  const ordered = [...withheldEx, ...familiarEx];
  const chosen = ordered[ex] || ordered[0];

  useEffect(() => () => clearInterval(timer.current), []);
  const reset = () => { clearInterval(timer.current); setResult(null); setStep(-1); setRevealed(false); setError(null); };
  const go = async (e) => {
    if (!e || !run) return;
    reset();
    setBusy(true);
    try {
      const res = await run(hidden, e.rec_id, e.role);
      setResult(res);
      setStep(0);
      let k = 0;
      timer.current = setInterval(() => { k += 1; setStep(k); if (k >= STEPS.length - 1) clearInterval(timer.current); }, STEP_MS);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "the example could not be run");
    } finally { setBusy(false); }
  };
  const shown = (s) => result && step >= STEPS.indexOf(s);
  const c = result?.checks;
  const conf = c && pair(c.confidence.value, c.confidence.threshold, 100, " %", 1);
  const fam = c && pair(c.familiarity.value, c.familiarity.threshold);
  const d = result?.decision;
  const missing = r.categories_without_an_eligible_recording || [];
  const isWithheld = result?.role === "withheld";

  return (
    <div className="un-root">
      <header className="glass un-header">
        <a className="pl-brand ms-brand" href="#/" aria-label="MachineSense home"><Logo /><span style={{ fontSize: 17, fontWeight: 500 }}>MachineSense</span></a>
        <a className="pl-link un-back" href="#/">← Back to the app</a>
        <div style={{ flex: 1 }} />
        <div role="radiogroup" aria-label="Reading level" className="seg">
          {[["mechanic", "Mechanic"], ["developer", "Developer"]].map(([k, lab]) => (
            <button key={k} role="radio" aria-checked={level === k} className={level === k ? "on" : ""} onClick={() => setLevel(k)}>{lab}</button>
          ))}
        </div>
      </header>

      <div className="un-title">
        <h1>What happens when a machine fault appears that this model was never taught?</h1>
        <p className="un-intro">Hide one fault from training, choose a recording, and see whether the model accepts a familiar label or asks for further inspection.</p>
        <details className="un-context"><summary>About this research demonstration</summary>
          <p className="un-simulated" data-copy="simulated">{summary.simulated}</p>
          <p className="un-easier" data-copy="easier">{summary.easier_part}</p>
        </details>
      </div>

      <div className="un-grid">
        <aside className="un-left">
          <section className="un-card glass-soft">
            <h2 className="un-step-title"><span>1</span> Hide a fault</h2>
            <label className="kicker" htmlFor="un-hidden">Fault to hide</label>
            <select id="un-hidden" className="rec-select un-select" value={hidden}
              onChange={(e) => { setHidden(e.target.value); setEx(0); reset(); }}>
              {classes.map((k) => <option key={k} value={k}>{cap(human(k))}</option>)}
            </select>
            <p className="un-copy" data-copy="trained-without">This experimental model was trained without {low(hidden)} recordings.</p>
            <details className="un-learned-details"><summary>See the conditions it learned</summary>
              <div className="kicker">What this model learned</div>
              <ul className="un-learned">
                {(r.classes_learned || []).map((k) => <li key={k}>{name(k)}</li>)}
                <li className="un-not-learned">{human(hidden)} — never seen</li>
              </ul>
            </details>
          </section>
          <section className="un-card glass-soft">
            <h2 className="un-step-title"><span>2</span> Choose a recording</h2>
            <p className="un-copy">Start with a recording of the hidden fault. Try the familiar example afterward to compare.</p>
            <div role="radiogroup" aria-label="Example recording" className="un-examples">
              {ordered.map((e, i) => (
                <button key={e.rec_id + e.role} role="radio" aria-checked={ex === i} className={"un-ex" + (ex === i ? " on" : "")}
                  onClick={() => { setEx(i); reset(); }}>
                  <span>{exampleLabel(e, e.role === "withheld" ? withheldEx.indexOf(e) : i, hidden)}</span>
                </button>
              ))}
            </div>
            <button className="un-run" disabled={!chosen || busy} onClick={() => go(chosen)}>{busy ? "Running…" : "3. Run investigation →"}</button>
            {familiarEx[0] && chosen?.role === "withheld" && (
              <button className="un-compare" disabled={busy} onClick={() => { const i = ordered.indexOf(familiarEx[0]); setEx(i); go(familiarEx[0]); }}>
                Compare with a familiar recording
              </button>
            )}
            {missing.map((m) => (
              <p key={m} className="un-small" data-copy="missing">No {m} among the eligible recordings: {r.why_missing?.[m] || "none met the conditions"}.</p>
            ))}
          </section>
        </aside>

        <div className="un-result-column">
        <main className="un-centre" aria-live="polite">
          {!result && !error && <div className="un-placeholder"><h2>What you will see</h2><p>Choose a recording on the left and press <b>Run investigation</b>. We will show the model’s proposed label, its two checks, and the final verdict.</p></div>}
          {error && <p className="un-error" role="alert">{error}</p>}
          {result && (
            <>
              <section className={"un-card un-label" + (shown("label") ? " is-shown" : "")}>
                <div className="kicker">{level === "developer" ? "CNN-6 top class (leave-one-class-out model)" : "The label the model proposes"}</div>
                <h2 data-v="proposed">{cap(name(result.proposed_label))}</h2>
                <p className="un-small">Its best answer among the {result.classes_learned.length} conditions it learned.</p>
                {level === "developer" && (
                  <ol className="un-top mono">{result.top_probabilities.map(([k, p]) => <li key={k}>{k}: {p.toFixed(6)}</li>)}</ol>
                )}
              </section>
              <div className="un-checks">
                <CheckCard title="Confidence" plain="How sure is its best answer?" tech="MSP (maximum softmax probability)"
                  value={conf[0]} threshold={conf[1]} passed={c.confidence.passed} rule="MSP ≥ τ_p" level={level} shown={shown("confidence")} />
                <CheckCard title="Familiarity" plain="How close is this to learned patterns?" tech="Mahalanobis distance (shared Ledoit–Wolf)"
                  value={fam[0]} threshold={fam[1]} passed={c.familiarity.passed} rule="distance ≤ τ_d" level={level} shown={shown("familiarity")} />
              </div>
              <p className={"un-card un-checksum" + (shown("checks") ? " is-shown" : "")} data-v="checks">
                {c.confidence.passed && c.familiarity.passed ? "Both checks passed, so the proposed label is accepted."
                  : `${[!c.confidence.passed && "The confidence check", !c.familiarity.passed && "the familiarity check"].filter(Boolean).join(" and ")} failed, so the proposed label is not accepted.`}
              </p>
            </>
          )}
          <details className="un-rule"><summary>How this decision works</summary><RuleDiagram /></details>
        </main>

        <aside className="un-right">
          {result && (
            <section className={"un-card glass-strong un-verdict" + (shown("verdict") ? " is-shown" : "")}>
              <div className="kicker">Verdict</div>
              <h2 data-v="verdict" className={d.decision === "unknown" ? "is-unknown" : ""}>{verdictText(d)}</h2>
              <p className="un-verdict-reason">{d.decision === "unknown" ? "At least one check did not support the proposed familiar label." : "The decision rule accepted a learned condition."}</p>
              {d.decision === "unknown" && <p className="un-copy" data-copy="further">{summary.further_inspection}</p>}
              {d.caution && <p className="un-small">A healthy-baseline check disagrees ({d.caution.replace("_", " ")}). {summary.gates_note}</p>}
              {level === "developer" && (
                <p className="un-small mono">rule {d.rule_variant}; {Object.entries(result.rule_comparison).map(([k, v]) => `${k}: ${v}`).join(" · ")}</p>
              )}
              {shown("verdict") && !revealed && <button className="un-reveal" onClick={() => setRevealed(true)}>Reveal actual condition</button>}
              {revealed && (
                <div className="un-truth" data-v="truth">
                  <div className="kicker">Actual condition (dataset label)</div>
                  <b>{cap(name(result.true_class))}</b>
                  <p className="un-small">{isWithheld ? "Hidden from this model during fitting: it never saw this condition."
                    : "A condition this model learned."}</p>
                  {isWithheld && d.decision !== "unknown" && (
                    <p className="un-copy">The model accepted a familiar label for a fault it never saw: an honest failure of the checks.</p>
                  )}
                  {isWithheld && d.decision === "unknown" && (
                    <p className="un-copy">The checks stopped a wrong familiar label. The verdict does not name {low(result.true_class)}.</p>
                  )}
                </div>
              )}
            </section>
          )}
          <details className="un-card glass-soft un-details">
            <summary>Experiment details</summary>
            <dl>
              <dt>Withheld class</dt><dd>{human(hidden)}</dd>
              <dt>Run</dt><dd>leave-one-class-out CNN-6, seed {summary.seed}</dd>
              <dt>Recording</dt><dd className="mono">{chosen?.rec_id || "—"}</dd>
              {result && <><dt>Recording details</dt><dd>{[severityText(result), `${result.speed_hz.toFixed(1)} Hz`, `used for ${result.used_for}`].filter(Boolean).join(" · ")}</dd></>}
              {result && <><dt>Example type</dt><dd>{chosen?.category}{chosen?.failure_kind ? ` (${chosen.failure_kind})` : ""}</dd></>}
              {r.counts && <><dt>This model on its hidden class</dt><dd data-v="counts">{r.counts.withheld_recordings} eligible recordings: {r.counts.rejected} not accepted,
                {" "}{r.counts.accepted_as_a_wrong_fault} accepted as a wrong fault, {r.counts.accepted_as_healthy} accepted as healthy</dd></>}
              <dt>Values</dt><dd>{result ? (result.source === "live" ? "computed now from the saved models" : "saved values (the model weights are not present here)") : (summary.live?.available ? "computed live" : "saved values")}</dd>
              <dt>Final manifest</dt><dd className="mono" title={summary.final_manifest_sha256}>{summary.final_manifest_sha256?.slice(0, 16)}…</dd>
              {result && level === "developer" && <><dt>Healthy-baseline checks</dt><dd className="mono">AE {result.scores.ae.toFixed(4)} / τ_e {result.thresholds.tau_e.toFixed(4)} · B1s {result.scores.b1s.toFixed(4)} / τ_b {result.thresholds.tau_b.toFixed(4)} — {summary.gates_note}</dd></>}
              <dt>Categories without an eligible recording</dt><dd>{missing.length ? missing.map((m) => `${m} (${r.why_missing?.[m] || "none"})`).join("; ") : "none"}</dd>
            </dl>
          </details>
        </aside>
        </div>
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
