import { useEffect, useId, useRef, useState } from "react";
import { askQuestion, getQuestions } from "../api.js";

const SUGGESTIONS = [
  "Why was this recording flagged?",
  "Which sensor changed most?",
  "What should I inspect next?",
];

// A silver gear with a "?" in its hub: the assistant's icon. Colours are the rig's own materials (rig/rigScene.js:
// polished steel #eef0f2, brushed aluminium #d6d9dd, motor paint #a7b1bb, black anodised #3a3d43).
const GEAR_PATH = (() => {
  const teeth = 10, R = 47, r = 38, c = 50, step = (2 * Math.PI) / teeth;
  const pt = (rad, a) => `${(c + rad * Math.cos(a)).toFixed(2)} ${(c + rad * Math.sin(a)).toFixed(2)}`;
  let d = "";
  for (let i = 0; i < teeth; i++) {
    const a = i * step - Math.PI / 2;
    d += `${i ? "L" : "M"}${pt(r, a - step * 0.3)} L${pt(R, a - step * 0.17)} L${pt(R, a + step * 0.17)} L${pt(r, a + step * 0.3)} `;
    d += `A${r} ${r} 0 0 1 ${pt(r, a + step * 0.7)} `;
  }
  return d + "Z";
})();

export function AssistantIcon({ size = 44, spin = false }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg className={"assistant-gear" + (spin ? " spin" : "")} width={size} height={size} viewBox="0 0 100 100" aria-hidden="true">
      <defs>
        <linearGradient id={`${id}-metal`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#eef0f2" /><stop offset="0.45" stopColor="#d6d9dd" /><stop offset="1" stopColor="#a7b1bb" />
        </linearGradient>
        <linearGradient id={`${id}-hub`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#c4c8cd" /><stop offset="1" stopColor="#f6f7f8" />
        </linearGradient>
      </defs>
      <g className="assistant-gear-body">
        <path d={GEAR_PATH} fill={`url(#${id}-metal)`} stroke="#8d969f" strokeWidth="1.2" strokeLinejoin="round" />
        <circle cx="50" cy="50" r="29" fill="none" stroke="#a7b1bb" strokeWidth="1.2" opacity="0.8" />
      </g>
      <circle cx="50" cy="50" r="24" fill={`url(#${id}-hub)`} stroke="#8d969f" strokeWidth="1.2" />
    </svg>
  );
}

// Badge per review outcome (App-7). "off" is the D7 pilot setting, never offered in the UI: no badge.
export const REVIEW_BADGE = { passed: "Checked against the case record", revised: "Revised once", fallback: "Could not answer with evidence" };
export const UNAVAILABLE = "The assistant is not available right now; the report has the full result.";
const unavailable = (a) => a.source === "fallback" && /^language model (disabled|unreachable)/.test(a.reason || "");

// One answer, rendered only from the fields listed here: the answer, its citations, the actions taken and the review
// badge (and, for a fallback, the fixed report items). Model reasoning is never part of the response or rendered.
export function AnswerView({ answer: a }) {
  if (unavailable(a)) return <p className="agent-answer-unavailable">{UNAVAILABLE}</p>;
  const badge = REVIEW_BADGE[a.review?.status];
  return (
    <div className="agent-answer">
      {badge && <span className={"agent-badge agent-badge-" + a.review.status}>{badge}</span>}
      <p>{a.answer}</p>
      {a.source === "fallback" && (a.report_sections || []).map((sec) => (
        <div key={sec.n} className="agent-fallback-sec"><span className="mono kicker">{sec.title}</span>
          <ul>{sec.items.map((it) => <li key={it.text} title={it.hover}>{it.text}</li>)}</ul></div>))}
      {a.citations?.length > 0 && <div className="agent-cites" aria-label="Cited guidance">
        {a.citations.map((c) => <span key={c.id} className="agent-cite mono" title={c.title || c.id}>{c.id}</span>)}</div>}
      {a.actions?.length > 0 && <details className="agent-actions"><summary>{`Actions (${a.actions.length})`}</summary>
        <ol>{a.actions.map((x, i) => <li key={i}><span className="mono">{x.tool}</span> — {x.summary}</li>)}</ol></details>}
    </div>
  );
}

export default function AgentChat({ caseId, open, onOpen, onClose, initialEntries = [] }) {
  const [draft, setDraft] = useState("");
  const [entries, setEntries] = useState(initialEntries);  // [{question, status: loading | done | error, answer}]
  const inputRef = useRef(null);
  const triggerRef = useRef(null);
  const caseRef = useRef(caseId);
  const close = () => {
    onClose();
    requestAnimationFrame(() => triggerRef.current?.focus());
  };

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
    const onKeyDown = (event) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  // the questions already asked about this case (stored with it); a different case starts a different list
  useEffect(() => {
    caseRef.current = caseId;
    setEntries([]);
    if (!caseId || !open) return;
    const ctl = new AbortController();
    getQuestions(caseId, { signal: ctl.signal })
      .then((qs) => { if (caseRef.current === caseId) setEntries(qs.map((q) => ({ question: q.question, status: "done", answer: q }))); })
      .catch(() => {});
    return () => ctl.abort();
  }, [caseId, open]);

  const busy = entries.some((e) => e.status === "loading");
  const send = (event) => {
    event?.preventDefault();
    const q = draft.trim();
    if (!caseId || !q || busy) return;
    const forCase = caseId;
    setDraft("");
    setEntries((xs) => [...xs, { question: q, status: "loading" }]);
    const settle = (patch) => {
      if (caseRef.current !== forCase) return;
      setEntries((xs) => xs.map((e, i) => (i === xs.length - 1 ? { ...e, ...patch } : e)));
    };
    askQuestion(forCase, q).then((answer) => settle({ status: "done", answer })).catch(() => settle({ status: "error" }));
  };

  return (
    <>
      {!open && <button ref={triggerRef} type="button" className="agent-launcher" onClick={onOpen}
        aria-label="Open MachineSense assistant" aria-expanded={false} aria-controls="agent-chat-panel" title="Open MachineSense assistant">
        <AssistantIcon size={76} spin />
        <span className="agent-launcher-tip">Ask MachineSense</span>
      </button>}
      {open && <>
        <div className="agent-chat-shade" onClick={close} aria-hidden="true" />
        <section id="agent-chat-panel" className="agent-chat-panel" role="dialog" aria-modal="true" aria-label="MachineSense assistant">
          <header className="agent-chat-head">
            <AssistantIcon size={40} />
            <div><strong>MachineSense assistant</strong><span>Answers checked against this case</span></div>
            <button type="button" onClick={close} className="agent-chat-close" aria-label="Close assistant">×</button>
          </header>
          <div className="agent-chat-body" aria-live="polite">
            {!caseId ? <>
              <h2>Ask about the machine</h2>
              <p>Start an investigation to give the assistant a recording to discuss.</p>
            </> : entries.length === 0 ? <>
              <h2>Ask about this recording</h2>
              <p>The assistant reads this case's diagnosis, signal quality, channel evidence and cited inspection steps. Each answer is checked against the case record before it is shown.</p>
              <div className="agent-chat-suggestions" aria-label="Suggested questions">
                {SUGGESTIONS.map((question) => <button type="button" key={question} onClick={() => { setDraft(question); inputRef.current?.focus(); }}>{question}</button>)}
              </div>
            </> : entries.map((e, i) => (
              <div key={i} className="agent-turn">
                <p className="agent-question">{e.question}</p>
                {e.status === "loading" ? <p className="mv-muted">Investigating…</p>
                  : e.status === "error" ? <p className="agent-answer-unavailable">{UNAVAILABLE}</p>
                  : <AnswerView answer={e.answer} />}
              </div>))}
          </div>
          <form className="agent-chat-compose" onSubmit={send}>
            <label htmlFor="agent-chat-draft" className="mono kicker">Your question</label>
            <textarea id="agent-chat-draft" ref={inputRef} value={draft} onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) send(event); }} maxLength={500}
              placeholder={caseId ? "Ask a question about this recording…" : "Start an investigation first"} rows={2} disabled={!caseId} />
            <div className="agent-chat-actions"><span>Uses only this case's record and the curated guidance</span>
              <button type="submit" disabled={!caseId || !draft.trim() || busy}>Send</button></div>
          </form>
        </section>
      </>}
    </>
  );
}
