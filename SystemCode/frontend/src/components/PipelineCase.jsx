import { K } from "../lib/theme.js";
import { FMT, STATUS_WORD, at, blockPaths, caseLine, isDerived, mechanicDetail, missing, statusReason, valueOf } from "../lib/pipelineCase.js";

// App-4.3: case-mode rendering. Every value is a <span data-p="<trace path>" data-fmt="<formatter>"> so the tests can
// check it against the fixture; a missing value reads "not recorded: <the trace's reason>", never 0, blank or a dash.
const NONE_FORMATS = new Set(["none", "noneText", "rerun", "recid", "full"]);  // formatters that word a null themselves
// App-4.5 (WCAG 1.4.11, 3:1 against the block): the text-safe amber / orange / grey tones
export const STATUS_COLOUR = { completed: K.violet, warning: K.amberT, failed: K.orangeT, not_run: K.ink3, unavailable: K.ink3 };

export function V({ trace, p, f = "full" }) {
  const n = at(trace, p);
  if (missing(n)) return <span className="pl-nr" data-p={p} data-nr="1">not recorded: {n.reason}</span>;
  const v = valueOf(n);
  // a null where the field has no "none" meaning reads "not recorded" (App-4.5: never 0, blank, "null" or a crash)
  if (v === undefined || (v === null && !NONE_FORMATS.has(f))) return <span className="pl-nr" data-p={p} data-nr="1">not recorded</span>;
  return <span className="pl-val" data-p={p} data-fmt={f}>{FMT[f](v)}</span>;
}

export function Verdict({ trace, p }) {
  return <span className="pl-verdict" data-verdict="1" data-p={p} data-fmt="text">{FMT.text(valueOf(at(trace, p)))}</span>;
}

// Alternative decision rules: read-only comparison evidence, visually subordinate to the rule in use, never the verdict.
export function RuleComparison({ trace, p }) {
  const comp = valueOf(at(trace, p)) || {};
  return (
    <div className="pl-comparison">
      <span className="mono kicker">Alternative rules · comparison only, not the verdict</span>
      <ul className="pl-list">{Object.keys(comp).map((k) => <li key={k}>Rule {k}: <V trace={trace} p={`${p}.value.${k}`} f="rule" /></li>)}</ul>
    </div>
  );
}

export function Parts({ trace, parts }) {
  return parts.map((x, i) => {
    if (typeof x === "string") return <span key={i}>{x}</span>;
    if (x.verdict) return <Verdict key={i} trace={trace} p={x.verdict} />;
    if (x.tag) return <span key={i} className="tag pl-tag">{x.tag}</span>;
    if (x.comparison) return <RuleComparison key={i} trace={trace} p={x.comparison} />;
    return <V key={i} trace={trace} p={x.p} f={x.f} />;
  });
}

export function StatusWord({ status }) {
  return (
    <span className="pl-status" data-status={status}>
      <svg width="8" height="8" aria-hidden="true"><circle cx="4" cy="4" r="3.5" fill={STATUS_COLOUR[status] || K.ink3} /></svg>
      <b>{STATUS_WORD[status] || status}</b>
    </span>
  );
}

// the block's second line in case mode: status word, then the case summary
export function CaseLine({ trace, id, level }) {
  const b = at(trace, blockPaths(trace)[id]);
  const parts = caseLine(trace, id, level);
  return (
    <div className="pl-sub pl-case">
      <StatusWord status={b.status} />{parts.length > 0 && <span> · </span>}<Parts trace={trace} parts={parts} />
    </div>
  );
}

// ------------------------------------------------------------------------------------------------------------ drawer
const FMT_BY_KEY = { display_label: "verdict", decision: "decision", caution: "noneText", fault_class: "noneText", rerun_of: "noneText",
  rec_id: "recid", source: "text", fallback_reason: "noneText" };
const label = (k) => k.replace(/_/g, " ");

function Leaf({ trace, p, k, v }) {
  if (k === "display_label") return <Verdict trace={trace} p={p} />;
  return <V trace={trace} p={p} f={FMT_BY_KEY[k] || (typeof v === "boolean" ? "yesno" : "full")} />;
}

// every field of an output / technical object, from its trace path, at full precision
export function Fields({ trace, path, obj }) {
  const rows = Object.entries(obj || {}).map(([k, n]) => {
    const p = `${path}.${k}`;
    let body;
    if (k === "rule_comparison") body = <RuleComparison trace={trace} p={p} />;
    else if (missing(n)) body = <V trace={trace} p={p} />;
    else if (n !== null && typeof n === "object" && !Array.isArray(n) && "value" in n) {
      const v = n.value;
      if (Array.isArray(v)) body = <Nested trace={trace} p={`${p}.value`} v={v} k={k} />;
      else if (v !== null && typeof v === "object") body = <Nested trace={trace} p={`${p}.value`} v={v} k={k} />;
      else body = <Leaf trace={trace} p={p} k={k} v={v} />;
      if (isDerived(n)) body = <>{body} <span className="pl-derived">computed for this view</span></>;
    } else if (typeof n === "string") body = <span className="pl-static">{n}</span>;
    else if (Array.isArray(n)) body = <ul className="pl-list">{n.map((item, i) => <li key={i}><Fields trace={trace} path={`${p}.${i}`} obj={item} /></li>)}</ul>;
    else if (n !== null && typeof n === "object") body = <Fields trace={trace} path={p} obj={n} />;
    else body = <Leaf trace={trace} p={p} k={k} v={n} />;
    return <div key={k} className="pl-field"><span className="pl-key">{label(k)}</span><span className="pl-fv">{body}</span></div>;
  });
  return <div className="pl-fields">{rows}</div>;
}

function Nested({ trace, p, v, k }) {
  if (Array.isArray(v)) {
    return <span className="pl-inline">{v.map((x, i) => (
      <span key={i}>{i > 0 && ", "}{Array.isArray(x) ? <>[<Nested trace={trace} p={`${p}.${i}`} v={x} k={k} />]</>
        : x && typeof x === "object" ? <Nested trace={trace} p={`${p}.${i}`} v={x} k={k} />
        : <Leaf trace={trace} p={`${p}.${i}`} k={k} v={x} />}</span>))}</span>;
  }
  return <div className="pl-fields">{Object.entries(v).map(([kk, x]) => (
    <div key={kk} className="pl-field"><span className="pl-key">{kk}</span><span className="pl-fv">
      {x !== null && typeof x === "object" ? <Nested trace={trace} p={`${p}.${kk}`} v={x} k={kk} /> : <Leaf trace={trace} p={`${p}.${kk}`} k={kk} v={x} />}
    </span></div>))}</div>;
}

export function CaseSection({ trace, block, level }) {
  const P = blockPaths(trace)[block.id];
  const dev = level === "developer";
  const reasons = statusReason(trace, block.id, level);
  const detail = dev ? null : mechanicDetail(trace, block.id);
  return (
    <div className="pl-case-drawer">
      <h3 className="mono kicker">This diagnosis</h3>
      <div className="pl-status-row"><StatusWord status={block.status} /></div>
      {reasons && reasons.length > 0 && <ul className="pl-list">{reasons.map((r, i) => <li key={i}><Parts trace={trace} parts={r} /></li>)}</ul>}
      {!dev && detail.length > 0 && <ul className="pl-list">{detail.map((r, i) => <li key={i}><Parts trace={trace} parts={r} /></li>)}</ul>}
      {dev && block.operation_case && <p className="pl-case-note">{block.operation_case}</p>}
      {dev && (
        <>
          <div className="pl-field"><span className="pl-key">recorded step time</span><span className="pl-fv">
            {block.timing_ms === null || block.timing_ms === undefined
              ? <span className="pl-nr" data-p={`${P}.timing_ms`} data-nr="1">not recorded</span>
              : <V trace={trace} p={`${P}.timing_ms`} f="ms" />}</span></div>
          {block.derived && <p className="pl-muted">Values marked “computed for this view” were worked out for this page from the saved case; the rest were saved by the diagnosis run.</p>}
          {block.output && typeof block.output === "object" && (<><h3 className="mono kicker">Values</h3><Fields trace={trace} path={`${P}.output`} obj={block.output} /></>)}
          {block.technical && typeof block.technical === "object" && (<><h3 className="mono kicker">Technical</h3><Fields trace={trace} path={`${P}.technical`} obj={block.technical} /></>)}
        </>
      )}
    </div>
  );
}

// the page's statement of the verdict in case mode (both reading levels show the same string)
export function CaseHeading({ trace }) {
  const dec = blockPaths(trace).decision;
  const inp = blockPaths(trace).input;
  const known = valueOf(at(trace, `${dec}.output.decision`)) === "known_fault";
  return (
    <p className="pl-case-heading">
      <span>This diagnosis: </span><Verdict trace={trace} p={`${dec}.output.display_label`} />
      {known && <span className="tag pl-tag">model prediction</span>}
      <span className="pl-muted"> · recorded </span><V trace={trace} p={`${inp}.output.created_at`} f="text" />
      <span className="pl-muted"> · read from the saved case; nothing is re-run</span>
    </p>
  );
}

// App-8: the decision panel on the canvas (the designer's "Decision" column). One row per check of the rule, each beside
// its input port, then the verdict. Under the rule in use the confidence and familiarity checks decide the verdict
// (with the top class); the two healthy-baseline checks only add a caution to a healthy verdict, and the rows say so.
const DEC_KEYS = ["msp", "maha", "ae", "b1s"];
const DEC_ROW = {
  msp: { name: "Confidence", pass: "The classifier is confident in its best match", fail: "The classifier is not confident enough", arch: "Passes when the best-match score is at least τ_p" },
  maha: { name: "Familiarity", pass: "Looks like recordings the model has seen", fail: "Unlike the recordings the model has seen", arch: "Passes when the distance is at most τ_d" },
  ae: { name: "Healthy pattern", pass: "Looks like the healthy recordings", fail: "Differs from the healthy recordings", arch: "Above τ_e: differs from healthy; adds a caution only" },
  b1s: { name: "Healthy at this speed", pass: "Looks like healthy recordings at this speed", fail: "Differs from healthy recordings at this speed", arch: "Above τ_b: differs from healthy; adds a caution only" },
};
export function DecisionPanel({ trace, block, level, geom }) {
  const dev = level === "developer";
  const P = trace ? blockPaths(trace).decision : null;
  const checks = trace ? at(trace, `${P}.output.checks`) || [] : [];
  const idx = (k) => checks.findIndex((c) => c.check.startsWith(k));
  const d = trace ? valueOf(at(trace, `${P}.output.decision`)) : null;
  const tone = (k, ok) => (k === "ae" || k === "b1s" ? (ok ? "is-ok" : "is-info") : ok ? "is-ok" : "is-warn");
  return (
    <div className="pl-dec">
      <div className="pl-dec-head" style={{ height: geom.HEAD }}>
        <div className="pl-title" style={{ textAlign: "left" }}>{block.title_mechanic && !dev ? block.title_mechanic : block.title_developer}</div>
        <div className="pl-dec-intro">{!trace ? "Combines the checks with a fixed rule. The second opinion is not used."
          : dev ? "MSP ≥ τ_p and distance ≤ τ_d: the top class; otherwise further inspection. The healthy-baseline checks add a caution only."
          : "The verdict needs a confident, familiar match. The healthy checks can only add a caution."}</div>
      </div>
      {DEC_KEYS.map((k, r) => {
        const i = idx(k), c = i >= 0 ? `${P}.output.checks.${i}` : null, ok = c ? valueOf(at(trace, `${c}.passed`)) : null;
        const base = k === "ae" || k === "b1s";
        return (
          <div key={k} className="pl-dec-row" data-check={k} style={{ top: geom.HEAD + r * (geom.ROW + geom.ROW_GAP), height: geom.ROW }}>
            <div className="pl-dec-row-text">
              <b>{DEC_ROW[k].name}</b>
              <span>{!c ? DEC_ROW[k].arch : dev ? <><V trace={trace} p={`${c}.value`} f="check" /> {k === "msp" ? (ok ? "≥" : "<") : ok ? "≤" : ">"} <V trace={trace} p={`${c}.threshold`} f="check" />{base ? " · caution only" : ""}</>
                : ok ? DEC_ROW[k].pass : DEC_ROW[k].fail}</span>
            </div>
            {c && <span className={"pl-dec-vote " + tone(k, ok)}>{base ? (ok ? "✓ Healthy" : "Differs") : ok ? "✓ Passes" : "✗ Fails"}</span>}
          </div>
        );
      })}
      <div className={"pl-dec-verdict" + (trace ? " is-" + (d === "known_fault" ? "fault" : d === "unknown" ? "further" : "healthy") : "")}
        style={{ bottom: 42, height: geom.VERDICT }}>
        <div className="pl-dec-vhead"><span className="mono kicker">{trace ? "Verdict" : "Possible outcomes"}</span>
          {trace && <span className="pl-dec-pips">{DEC_KEYS.map((k) => { const i = idx(k); const ok = i >= 0 && valueOf(at(trace, `${P}.output.checks.${i}.passed`));
            return <i key={k} className={ok ? "is-ok" : k === "ae" || k === "b1s" ? "is-info" : "is-warn"} />; })}</span>}</div>
        {trace ? <div className="pl-dec-vtitle"><Verdict trace={trace} p={`${P}.output.display_label`} /></div>
          : <div className="pl-dec-vtitle" style={{ fontSize: 13.5 }}>Healthy · Healthy (caution) · Known fault · Further inspection needed</div>}
        <div className="pl-dec-vsub">{!trace ? "A recording the checks do not support gets further inspection, not a guess."
          : d === "known_fault" ? "Model prediction, according to the dataset label." : d === "unknown" ? "Further inspection needed: the checks did not support a known condition."
          : "No fault detected by the checks in use."}</div>
      </div>
      <span className="pl-dec-rule" title="The decision rule is a provisional setting for this research demonstration.">Decision rule: provisional research setting</span>
    </div>
  );
}
