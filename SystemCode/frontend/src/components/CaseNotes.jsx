import { useEffect, useState } from "react";
import { addObservation, addStatusEvent, editObservation, getFaultStatus, getObservations } from "../api.js";
import { human } from "../lib/model.js";

// App-9 slice 2 (addendum B): the case's human records in the assistant panel, kept visibly apart from the model:
// the model prediction (read from the record, never changed), the technician's fault status (an append-only event log)
// and technician observations with their edit history. The app does not sign technicians in: the name is the browser's
// display name and every record says it is a local demo action. Nothing here is measured evidence.
export const KIND = { model: "Model prediction", observation: "Technician observation", confirmed: "Technician-confirmed fault" };
export const STATUS_TEXT = { technician_confirmed: "Technician-confirmed fault", technician_rejected: "Technician rejected the model prediction",
  not_confirmed: "Not confirmed by a technician" };
const readName = () => { try { return localStorage.getItem("ms_display_name") || ""; } catch { return ""; } };
const when = (iso) => { try { return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch { return iso; } };

export default function CaseNotes({ caseId, initialStatus = null, initialObservations = null, api = {} }) {
  const A = { getObservations, addObservation, editObservation, getFaultStatus, addStatusEvent, ...api };
  const [status, setStatus] = useState(initialStatus);
  const [obs, setObs] = useState(initialObservations);
  const [err, setErr] = useState(null);
  useEffect(() => {
    if (initialStatus && initialObservations) return undefined;
    const ac = new AbortController();
    Promise.all([A.getFaultStatus(caseId, { signal: ac.signal }), A.getObservations(caseId, { signal: ac.signal })])
      .then(([s, o]) => { setStatus(s); setObs(o.observations); }).catch(() => { if (!ac.signal.aborted) setErr("The case notes could not be loaded."); });
    return () => ac.abort();
  }, [caseId]);
  if (err) return <p className="cn-state" role="status">{err}</p>;
  if (!status || !obs) return <p className="cn-state" role="status">Loading the case notes…</p>;
  const author = readName();
  const fail = (e) => setErr(e?.response?.data?.detail || "The change was not saved.");
  return (
    <div className="cn" aria-label="Case notes and fault status">
      <p className="cn-demo">{status.local_demo}</p>
      <ModelPrediction mp={status.model_prediction} />
      <TechnicianStatus status={status} author={author} onEvent={(body) => A.addStatusEvent(caseId, { ...body, author })
        .then((r) => setStatus(r)).catch(fail)} />
      <Observations obs={obs} author={author} onAdd={(t) => A.addObservation(caseId, t, author).then((o) => setObs((xs) => [...xs, o])).catch(fail)}
        onEdit={(id, t) => A.editObservation(caseId, id, t, author).then((o) => setObs((xs) => xs.map((x) => (x.id === id ? o : x)))).catch(fail)} />
    </div>
  );
}

function ModelPrediction({ mp }) {
  const label = mp.decision === "known_fault" ? `Known fault · ${human(mp.fault_class)}` : mp.decision === "unknown" ? "Further inspection needed" : "Healthy";
  return (
    <section className="cn-card cn-model" aria-label={KIND.model}>
      <span className="cn-kind">{KIND.model}</span>
      <b>{label}</b>
      <span className="cn-sub">{mp.note}</span>
    </section>
  );
}

function TechnicianStatus({ status, author, onEvent }) {
  const t = status.technician, latest = t.latest;
  const [form, setForm] = useState(null);  // null | "confirm" | "reject"
  const [fc, setFc] = useState(status.model_prediction.fault_class && status.fault_classes.includes(status.model_prediction.fault_class)
    ? status.model_prediction.fault_class : status.fault_classes[0]);
  const [part, setPart] = useState(Object.keys(status.parts)[0]);
  const [note, setNote] = useState("");
  const submit = (ev) => {
    ev.preventDefault();
    onEvent(form === "confirm" ? { status: "confirmed", fault_class: fc, part, note } : { status: "rejected", note });
    setForm(null); setNote("");
  };
  return (
    <section className="cn-card" aria-label="Technician fault status">
      <span className="cn-kind">Technician status</span>
      <b className={"cn-status is-" + t.status}>{STATUS_TEXT[t.status]}</b>
      {latest && latest.status === "confirmed" && <span className="cn-sub">
        {`${human(latest.fault_class)} at the ${status.parts[latest.part].toLowerCase()} · recorded ${when(latest.created_at)} by ${latest.actor.name} (name not verified)`}
        {t.agrees_with_model === false && " · differs from the model prediction"}</span>}
      {latest && latest.status !== "confirmed" && <span className="cn-sub">{`Last change ${when(latest.created_at)} by ${latest.actor.name} (name not verified)`}</span>}
      {latest?.note && <span className="cn-note">“{latest.note}”</span>}
      {!form && <div className="cn-actions">
        <button type="button" className="as-chipq" onClick={() => setForm("confirm")}>Record a confirmed fault…</button>
        <button type="button" className="as-chipq" onClick={() => setForm("reject")}>Record a rejection…</button>
        {t.status !== "not_confirmed" && <button type="button" className="as-chipq" onClick={() => onEvent({ status: "withdrawn", note: "" })}>Withdraw</button>}
      </div>}
      {form && <form className="cn-form" onSubmit={submit}>
        {form === "confirm" && <>
          <label>Fault found<select value={fc} onChange={(e) => setFc(e.target.value)}>
            {status.fault_classes.map((c) => <option key={c} value={c}>{human(c)}</option>)}</select></label>
          <label>Part<select value={part} onChange={(e) => setPart(e.target.value)}>
            {Object.entries(status.parts).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>
        </>}
        <label>Note<textarea value={note} maxLength={1000} rows={2} onChange={(e) => setNote(e.target.value)}
          placeholder={form === "confirm" ? "What you found, e.g. where and how you checked" : "Why the prediction does not match"} /></label>
        <span className="cn-sub">{`Recorded as ${author || "an unnamed user"} on this device. The model's prediction is kept unchanged beside it.`}</span>
        <div className="cn-actions"><button type="submit" className="as-chipq is-primary">{form === "confirm" ? "Record confirmation" : "Record rejection"}</button>
          <button type="button" className="as-chipq" onClick={() => setForm(null)}>Cancel</button></div>
      </form>}
      {status.events.length > 1 && <details className="cn-hist"><summary>{`Status history (${status.events.length} events)`}</summary>
        {status.events.map((e) => <div key={e.id}>{`${when(e.created_at)} · ${e.status}${e.fault_class ? ` · ${human(e.fault_class)}` : ""} · ${e.actor.name}`}</div>)}</details>}
    </section>
  );
}

function Observations({ obs, onAdd, onEdit }) {
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(null);  // {id, text}
  return (
    <section className="cn-card" aria-label="Technician observations">
      <span className="cn-kind">{`${KIND.observation}s`}</span>
      <span className="cn-sub">What you saw, heard or measured by hand. Kept beside the case; not part of the recording's measurements.</span>
      {obs.length === 0 && <span className="cn-empty">No observations yet.</span>}
      {obs.map((o) => <div key={o.id} className="cn-obs">
        {editing?.id === o.id
          ? <form className="cn-form" onSubmit={(e) => { e.preventDefault(); onEdit(o.id, editing.text); setEditing(null); }}>
              <textarea value={editing.text} maxLength={1000} rows={2} onChange={(e) => setEditing({ id: o.id, text: e.target.value })} aria-label="Edit observation" />
              <div className="cn-actions"><button type="submit" className="as-chipq is-primary">Save edit</button><button type="button" className="as-chipq" onClick={() => setEditing(null)}>Cancel</button></div></form>
          : <><p>{o.text}</p>
            <span className="cn-sub">{`${when(o.created_at)} · ${o.actor.name} (name not verified)${o.edits.length ? ` · edited ${o.edits.length}×` : ""}`}
              <button type="button" className="cn-link" onClick={() => setEditing({ id: o.id, text: o.text })}>Edit</button></span>
            {o.edits.length > 0 && <details className="cn-hist"><summary>Edit history</summary>
              {o.edits.map((e, i) => <div key={i}>{`${when(e.edited_at)} · was: “${e.previous_text}”`}</div>)}</details>}</>}
      </div>)}
      <form className="cn-form" onSubmit={(e) => { e.preventDefault(); if (draft.trim()) { onAdd(draft.trim()); setDraft(""); } }}>
        <textarea value={draft} maxLength={1000} rows={2} onChange={(e) => setDraft(e.target.value)} aria-label="New observation"
          placeholder="Add an observation, e.g. what you heard or found when you checked" />
        <div className="cn-actions"><button type="submit" className="as-chipq is-primary" disabled={!draft.trim()}>Add observation</button></div>
      </form>
    </section>
  );
}
