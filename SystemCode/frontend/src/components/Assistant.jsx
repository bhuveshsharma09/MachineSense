import { useEffect, useRef, useState } from "react";
import { askConversation, getConversation, newConversation } from "../api.js";
import { K, TAG, al } from "../lib/theme.js";
import * as M from "../lib/mechanic.js";
import * as D from "../lib/dashboard.js";
import { PART, SCORES_NOTE, human, locations, qualityFlags, unusualSensors, verdict, verdictTitle } from "../lib/model.js";
import { AssistantIcon, REVIEW_BADGE, UNAVAILABLE } from "./AgentChat.jsx";
import { Header, PageState, PartView, VKIND, useCaseData } from "./Dashboard.jsx";
import "../redesign.css";

// Stage App-8: #/case/<id>/assistant — the investigation assistant (App-7) on its own page. Same endpoints as the chat
// panel (POST /ask, GET /questions); the page shows the answer, its citations, the actions taken and the review outcome,
// never model reasoning. Context on the left is read from the case record.
export const SUGGESTIONS = ["Why was this recording flagged?", "Which sensor changed most?", "What should I inspect next?",
  "Has a technician confirmed this?"];
export const TOOLS = ["diagnose", "check_signal", "get_evidence", "lookup_guidance", "get_case_notes", "get_procedures"];

// App-9 addendum C: what an answer rests on, by kind, from its claim refs and citations (never from the answer text)
export const SOURCE_KIND = { measured: "Measured (recording)", model: "Model prediction", technician: "Technician record (local demo)",
  guidance: "Inspection guidance", procedure: "Verified procedure" };
export function refKind(ref, citations = []) {
  if (/^(?:evidence|signal_quality|physics)\./.test(ref)) return "measured";
  if (/^(?:diagnosis|assessment)\./.test(ref)) return "model";
  if (/^notes\./.test(ref)) return "technician";
  if (/^procedures\./.test(ref) || citations.some((c) => c.id === ref && c.kind === "procedure")) return "procedure";
  return "guidance";
}
export function answerKinds(a) {
  const kinds = new Set((a.claims || []).flatMap((c) => c.refs || []).map((r) => refKind(r, a.citations || [])));
  for (const c of a.citations || []) kinds.add(c.kind === "procedure" ? "procedure" : "guidance");
  return Object.keys(SOURCE_KIND).filter((k) => kinds.has(k));
}
// the four labelled paragraphs the conversation prompt may use; other paragraphs stay plain
export const SECTION_LABELS = ["What the recording shows", "What the model predicts", "What the technician recorded", "What to inspect next"];
export function sections(text) {
  return String(text || "").split(/\n\s*\n/).filter(Boolean).map((p) => {
    const m = new RegExp(`^\\s*(?:\\*\\*)?(${SECTION_LABELS.join("|")}):?(?:\\*\\*)?:?\\s*`).exec(p);
    return m ? { label: m[1], text: p.slice(m[0].length) } : { label: null, text: p };
  });
}
export const VERIFICATION_LABEL = { internal: "project protocol", open_verified: "checked against an open source",
  citation_unverified: "citation not yet checked against the full text", ai_drafted: "AI-drafted, not yet checked against a published source" };
const BADGE_STYLE = {
  passed: { bg: al(K.green, 0.12), color: "oklch(0.42 0.09 160)", dot: K.green },
  revised: { bg: al(K.violet, 0.09), color: "oklch(0.44 0.15 277)", dot: K.violet },
  fallback: { bg: al(K.amber, 0.2), color: "oklch(0.45 0.10 62)", dot: "oklch(0.72 0.14 70)" },
};
const unavailable = (a) => a?.source === "fallback" && /^language model (disabled|unreachable)/.test(a.reason || "");

export default function Assistant({ caseId, initialPart }) {
  const [st] = useCaseData(caseId, { withExplanation: false });
  if (st.status !== "ok") return <PageState message={st.status === "loading" ? "Loading the case…" : st.message} />;
  return <AssistantView {...st} caseId={caseId} initialPart={initialPart} />;
}

// The context column's summary of each source the tools read, from the same responses the dashboard uses.
export function contextRows(record, report, channels) {
  const v = verdict(record), q = record.signal_quality;
  const checks = report?.assessment?.checks?.length;
  const rows = D.channelRows(channels);
  const steps = D.guidanceSteps(M.guidanceItems(report)).filter((s) => s.src);
  return [
    { tag: "Model", title: "Diagnosis", text: `${v.chip}${record.diagnosis.decision === "known_fault" ? " · " + human(record.diagnosis.fault_class) : ""}${checks ? `; ${checks} decision checks` : ""}` },
    { tag: "Measured", title: "Signal quality", text: `${q.warnings.length ? q.warnings.length + " warning" + (q.warnings.length > 1 ? "s" : "") : "No warnings"} · speed ${q.speed_stable ? "steady" : "not steady"} at ${Math.round(60 * q.shaft_speed_hz).toLocaleString("en-US")} rpm` },
    { tag: "Measured", title: "Channel comparison", text: channels ? `${rows.filter((r) => r.beyond).length} of ${rows.length} channels beyond typical healthy variation vs ${channels.reference.rec_id}` : "Loading…" },
    { tag: "Guidance", title: "Guidance library", text: report ? (steps.length ? `${steps.length} inspection step${steps.length > 1 ? "s" : ""} with sources` : "No cited inspection steps for this verdict") : "Loading…" },
  ];
}

// App-9: the current conversation of the case is shown and used as context; earlier questions stay stored and listed, but
// are not context. "New conversation" starts a new one; nothing is deleted.
export function splitConversation(data) {
  const cur = new Set(data?.conversation || []);
  const qs = data?.questions || [];
  return { current: qs.filter((q) => cur.has(q.id)), earlier: qs.filter((q) => !cur.has(q.id)) };
}

export function AssistantView({ caseId, record, report, channels, initialEntries = null, initialEarlier = [], initialPart = null,
  ask = askConversation, load = getConversation, restart = newConversation, panel = false, newSession = false }) {
  const [entries, setEntries] = useState(initialEntries || []);
  const [earlier, setEarlier] = useState(initialEarlier);
  const [draft, setDraft] = useState("");
  const [open, setOpen] = useState({});
  const scroll = useRef(null);
  const caseRef = useRef(caseId);
  useEffect(() => {
    caseRef.current = caseId;
    if (initialEntries) return undefined;
    const ac = new AbortController();
    setEntries([]); setEarlier([]);
    load(caseId, { signal: ac.signal }).then(async (data) => {
      if (caseRef.current !== caseId || ac.signal.aborted) return;
      // App-9 addendum A: a new panel session starts a new conversation (only if the current one has turns; nothing deleted)
      if (newSession && (data?.conversation || []).length) { await restart(caseId); data = { ...data, conversation: [] }; }
      if (caseRef.current !== caseId || ac.signal.aborted) return;
      const { current, earlier: old } = splitConversation(data);
      setEntries(current.map((a) => ({ question: a.question, status: "done", answer: a })));
      setEarlier(old);
    }).catch(() => {});
    return () => ac.abort();
  }, [caseId]);
  useEffect(() => { const el = scroll.current; if (el?.scrollTo) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" }); }, [entries.length, entries.at?.(-1)?.status]);

  const busy = entries.some((e) => e.status === "loading");
  const fresh = () => {
    if (busy) return;
    const forCase = caseId;
    restart(forCase).then(() => {
      if (caseRef.current !== forCase) return;
      setEarlier((xs) => [...xs, ...entries.filter((e) => e.status === "done").map((e) => e.answer)]);
      setEntries([]);
    }).catch(() => {});
  };
  const send = (q) => {
    q = (q || "").trim();
    if (!q || busy) return;
    const forCase = caseId;
    setDraft("");
    setEntries((xs) => [...xs, { question: q, status: "loading" }]);
    const settle = (p) => { if (caseRef.current === forCase) setEntries((xs) => xs.map((e, i) => (i === xs.length - 1 ? { ...e, ...p } : e))); };
    ask(forCase, q).then((answer) => settle({ status: "done", answer })).catch(() => settle({ status: "error" }));
  };

  const v = verdict(record), vs = VKIND[v.kind];
  const loc = locations(record), flags = qualityFlags(record), unusual = unusualSensors(channels, flags);
  const partId = initialPart && PART[initialPart] ? initialPart : D.focusPart(null, loc, unusual);
  const mloc = M.location(record);
  const href = `#/case/${encodeURIComponent(caseId)}/dashboard`;

  return (
    <div className={panel ? "as-panel-content" : "rd-root as-root"}>
      {!panel && <Header caseId={caseId} record={record} back={{ href, label: "Back to dashboard" }}
        title={<span className="as-title">Investigation assistant</span>} userName={(() => { try { return localStorage.getItem("ms_display_name") || ""; } catch { return ""; } })()} />}
      <div className="as-grid">
        {!panel && <aside className="as-aside" aria-label="Context from the dashboard">
          <section className="rd-card">
            <div className="rd-card-head"><span className="rd-kicker">Context from the dashboard</span>
              <span className="rd-verdict" style={{ color: vs.color, fontSize: 12 }}><span className="rd-dot" style={{ background: vs.dot }} />{v.chip}</span></div>
            <div className="rd-stack-4"><span className="as-ctx-title">{verdictTitle(record)}</span>
              {mloc.text && <span className="rd-sub-s" style={{ fontSize: 12.5 }}>{mloc.text}</span>}</div>
            <PartView partId={partId} sceneState={{ fault: loc.fault, cand: loc.cand, unusual, flags }} height={190}>
              <div className="rd-iso-facts"><div><span>Shown here</span><b style={{ fontSize: 15 }}>{PART[partId].full || PART[partId].label}</b></div></div>
            </PartView>
            <span className="mono rd-sub-s">Case {caseId.slice(0, 8)} · {record.input?.rec_id || "uploaded recording"}</span>
          </section>
          <section className="rd-card" style={{ gap: 10 }}>
            <h2 style={{ fontSize: 14 }}>What the assistant can read</h2>
            {contextRows(record, report, channels).map((c) => (
              <div key={c.title} className="as-ctx-row">
                <span className="tag" style={{ color: TAG[c.tag].c, background: TAG[c.tag].b }}>{c.tag}</span>
                <div className="rd-col" style={{ gap: 1 }}><span className="rd-strong">{c.title}</span><span className="rd-sub-s" style={{ fontSize: 12 }}>{c.text}</span></div>
              </div>))}
          </section>
          <section className="rd-card as-tools">
            <span className="rd-kicker">Read-only tools</span>
            <div className="rd-row-wrap" style={{ gap: 6 }}>{TOOLS.map((t) => <span key={t} className="mono as-tool">{t}</span>)}</div>
            <span className="rd-note">{M.LIMITATION}</span>
          </section>
        </aside>}

        <main className="as-main" aria-label="Conversation">
          <div ref={scroll} className="as-scroll" aria-live="polite">
            <div className="as-inner">
              <div className="rd-stack-6">
                {!panel && <span className="rd-kicker">Investigation assistant</span>}
                <h1 className="as-h1">Ask about this recording</h1>
                <p className="rd-p" style={{ fontSize: 14, maxWidth: 560 }}>Answers come from the case record and the guidance library, and are checked against the case record before they are shown.</p>
                <p className="rd-note" style={{ maxWidth: 560 }}>The assistant remembers this conversation about this recording only. Earlier answers help it follow your question; every fact is checked against the case record again.</p>
              </div>
              {earlier.length > 0 && <details className="as-earlier"><summary>{`Earlier questions about this recording (${earlier.length}) · not used in this conversation`}</summary>
                {earlier.map((a, i) => <div key={a.id ?? i} className="as-earlier-q"><b>{a.question}</b><span>{a.source === "fallback" ? "Could not answer with evidence" : a.answer}</span></div>)}
              </details>}
              {entries.length === 0 && <div className="as-suggest">{SUGGESTIONS.map((q) => (
                <button key={q} disabled={busy} onClick={() => send(q)}><span>{q}</span><span className="mono rd-sub-s">Suggested question</span></button>))}</div>}
              {entries.map((e, i) => <Turn key={i} e={e} open={!!open[i]} onToggle={() => setOpen((o) => ({ ...o, [i]: !o[i] }))} href={href} />)}
            </div>
          </div>
          <div className="as-compose">
            <div className="as-inner-s">
              {entries.length > 0 && <div className="rd-row-wrap" style={{ gap: 6 }}>{SUGGESTIONS.map((q) => <button key={q} className="as-chipq" disabled={busy} onClick={() => send(q)}>{q}</button>)}
                <button type="button" className="as-chipq as-new" disabled={busy} onClick={fresh} title="Start a new conversation about this recording. Nothing is deleted.">New conversation</button></div>}
              <form className="as-form" onSubmit={(ev) => { ev.preventDefault(); send(draft); }}>
                <input value={draft} onChange={(ev) => setDraft(ev.target.value)} maxLength={500} disabled={busy} aria-label="Ask about this recording"
                  placeholder="Ask about the evidence, the sensors or what to inspect…" />
                <button type="submit" disabled={busy || !draft.trim()} className={busy || !draft.trim() ? "" : "is-on"}>Ask</button>
              </form>
              <span className="rd-note" style={{ textAlign: "center" }}>{SCORES_NOTE} Answers do not replace an inspection.</span>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

function Turn({ e, open, onToggle, href }) {
  const a = e.answer;
  const quick = !!a?.fallback_kind;  // a short answer built from the stored records (no model)
  const off = e.status === "error" || (unavailable(a) && !quick);
  const badge = a && !off ? (quick ? "From the stored case records" : REVIEW_BADGE[a.review?.status]) : null;
  const bs = badge ? (quick ? BADGE_STYLE.passed : BADGE_STYLE[a.review.status]) : null;
  const kinds = a && !off ? answerKinds(a) : [];
  const acts = a?.actions || [];
  return <>
    <div className="as-user">{e.question}</div>
    <article className="as-bot">
      <div className="rd-row-wrap" style={{ gap: 10 }}>
        <AssistantIcon size={28} /><span className="rd-strong">Assistant</span>
        {badge && <span className="as-badge" style={{ background: bs.bg, color: bs.color }}><span className="rd-dot" style={{ width: 6, height: 6, background: bs.dot }} />{badge}</span>}
      </div>
      {e.status === "loading" ? <div className="as-acts"><span className="as-acts-head"><span className="as-spin" />Checking the case record…</span></div>
        : off ? <p className="as-para">{UNAVAILABLE}</p> : <>
          {acts.length > 0 && <div className="as-acts">
            <button className="as-acts-head" onClick={onToggle} aria-expanded={open}><span className="rd-kicker">Actions · {acts.length}</span><span className="rd-sub-s" style={{ marginLeft: "auto" }}>{open ? "Hide" : "Show"}</span></button>
            {open && acts.map((x, i) => <div key={i} className="as-act"><span className="mono as-tool">{x.tool}</span><span>{x.summary}</span></div>)}
          </div>}
          <div className="rd-stack-8">{sections(a.answer).map((p, i) => p.label
            ? <section key={i} className="as-sec"><h4>{p.label}</h4><p className="as-para">{p.text}</p></section>
            : <p key={i} className="as-para">{p.text}</p>)}</div>
          {kinds.length > 0 && <div className="as-kinds" aria-label="What this answer rests on">{kinds.map((k) => <span key={k} className={"as-kind is-" + k}>{SOURCE_KIND[k]}</span>)}</div>}
          {a.source === "fallback" && (a.report_sections || []).map((sec) => (
            <div key={sec.n} className="as-fallback"><span className="rd-kicker">{sec.title}</span>
              {sec.items.map((it, i) => <div key={i} className="rd-row-top" title={it.hover}><span className="tag" style={{ color: TAG[it.tag].c, background: TAG[it.tag].b }}>{it.tag}</span><span className="rd-report-t">{it.text}</span></div>)}
            </div>))}
          <div className="rd-row-wrap" style={{ gap: 6 }}>
            {(a.citations || []).map((c) => <span key={c.id} className="as-cite-full" title={[c.title, c.source].filter(Boolean).join(" — ")}>
              <span className="mono as-cite"><span className="rd-dot" style={{ width: 5, height: 5, background: "oklch(0.70 0.15 125)" }} />{c.id}</span>
              {(c.title || c.verification) && <span className="as-cite-meta">{[c.title, c.kind === "procedure" ? "reviewed procedure" : VERIFICATION_LABEL[c.verification]].filter(Boolean).join(" · ")}</span>}
            </span>)}
            <a href={href} style={{ marginLeft: "auto", fontSize: 12.5 }}>See the evidence on the dashboard ›</a>
          </div>
        </>}
    </article>
  </>;
}
