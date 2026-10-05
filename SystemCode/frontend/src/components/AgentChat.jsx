import { useEffect, useRef, useState } from "react";
import { askQuestion, getQuestions } from "../api.js";

const SUGGESTIONS = [
  "Why was this recording flagged?",
  "Which sensor changed most?",
  "What should I inspect next?",
];

export function RobotIcon({ size = 90 }) {
  return <img className="robot-render" src="/assets/robot-camera-3d.png" alt="" width={size} height={size} draggable="false" />;
}

function RobotWalker() {
  return (
    <span className="robot-walker" aria-hidden="true">
      <img className="robot-walker-piece robot-walker-body" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
      <img className="robot-walker-piece robot-walker-left" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
      <img className="robot-walker-piece robot-walker-middle" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
      <img className="robot-walker-piece robot-walker-right" src="/assets/robot-camera-3d.png" alt="" draggable="false" />
    </span>
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
        <RobotWalker />
        <span className="agent-launcher-tip">Ask MachineSense</span>
      </button>}
      {open && <>
        <div className="agent-chat-shade" onClick={close} aria-hidden="true" />
        <section id="agent-chat-panel" className="agent-chat-panel" role="dialog" aria-modal="true" aria-label="MachineSense assistant">
          <header className="agent-chat-head">
            <RobotIcon size={44} />
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
