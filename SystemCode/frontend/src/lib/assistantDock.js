// App-9 addendum A: the lifecycle of the one assistant panel, as a pure reducer (testable without a browser).
// State is scoped to ONE case: open, minimized, pinned, and a session number. Every panel session mounts a new assistant,
// which starts a new App-9 conversation if the current one already has turns; nothing stored is deleted. Close, Home and
// another case end the session. The URL never carries the panel state: #/case/<id>/assistant only opens it (App canonicalises it to
// /dashboard), so Back / Forward move between routes and the panel follows these rules.

export const CLOSED = { caseId: null, open: false, minimized: false, pinned: false, session: 0 };

// route: { id, page } for a case page (page: dashboard | pipeline | assistant), or null for Home / other pages
export function dockReducer(st, a) {
  switch (a.type) {
    case "route": {
      const r = a.route;
      if (!r) return endSession(st);                                      // Home (or a non-case page): end the session
      if (st.caseId !== null && r.id !== st.caseId) st = endSession(st);  // another case: never carry the panel over
      if (r.page === "assistant") return openFor(st, r.id);               // deep link: open on the dashboard
      if (r.page === "pipeline" && st.open && !st.pinned) return endSession(st, r.id);  // unpinned: closes on the pipeline
      return { ...st, caseId: st.caseId ?? r.id };
    }
    case "open": return openFor(st, a.caseId);
    case "minimize": return st.open ? { ...st, minimized: true } : st;
    case "restore": return st.open ? { ...st, minimized: false } : st;
    case "togglePin": return st.open ? { ...st, pinned: !st.pinned } : st;
    case "close": return endSession(st, st.caseId);
    default: return st;
  }
}

function openFor(st, caseId) {
  if (st.open && st.caseId === caseId) return { ...st, minimized: false };
  const base = st.caseId === caseId ? st : endSession(st, caseId);
  return { ...base, caseId, open: true, minimized: false };
}

// Close / Home / another case: closed, unpinned, a new session number (the next open is a new conversation)
function endSession(st, caseId = null) {
  return { ...CLOSED, caseId, session: st.session + (st.open ? 1 : 0) };
}

export const expanded = (st, caseId) => st.open && !st.minimized && st.caseId === caseId;
