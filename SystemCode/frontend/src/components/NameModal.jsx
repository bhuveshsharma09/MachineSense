import { useState } from "react";
import { K } from "../lib/theme.js";

// Display name, stored in this browser only (localStorage key ms_display_name).
export default function NameModal({ initial, onSave }) {
  const [draft, setDraft] = useState(initial || "");
  const invalid = !draft.trim();
  return (
    <div style={{ position: "absolute", inset: 0, zIndex: 20, display: "grid", placeItems: "center", padding: 20, background: "oklch(0.93 0.004 85 / 0.45)", backdropFilter: "blur(6px)", WebkitBackdropFilter: "blur(6px)" }}>
      <form onSubmit={(e) => { e.preventDefault(); if (!invalid) onSave(draft.trim()); }} className="glass-strong"
        style={{ width: "100%", maxWidth: 440, display: "flex", flexDirection: "column", gap: 22, padding: 30, borderRadius: 22 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ width: 22, height: 22, borderRadius: "50%", border: "1.5px solid " + K.ink, display: "grid", placeItems: "center" }}><span style={{ width: 7, height: 7, borderRadius: "50%", background: K.orange }} /></span>
          <span style={{ fontSize: 17, fontWeight: 500 }}>MachineSense</span>
          <span className="mono kicker" style={{ marginLeft: "auto" }}>Research demo</span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <h1 style={{ margin: 0, fontSize: 30, fontWeight: 300, letterSpacing: "-0.022em", lineHeight: 1.12 }}>Investigate a machine recording</h1>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.5, color: K.ink2 }}>Check sensor recordings from a rotor test rig and follow how each conclusion was reached.</p>
        </div>
        <label style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <span className="mono kicker">Display name</span>
          <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="e.g. Sam Okafor" autoFocus
            style={{ height: 46, padding: "0 14px", borderRadius: 11, border: "1px solid oklch(0.23 0.006 260 / 0.16)", background: "oklch(1 0 0 / 0.85)", fontSize: 15 }} />
        </label>
        <button type="submit" disabled={invalid} style={{ height: 46, border: 0, borderRadius: 11, background: invalid ? "oklch(0.23 0.006 260 / 0.1)" : K.ink, color: invalid ? K.ink3 : K.white, fontSize: 15, fontWeight: 500, cursor: "pointer" }}>Continue</button>
        <p style={{ margin: 0, fontSize: 12, lineHeight: 1.5, color: "oklch(0.48 0.008 260)" }}>No account needed. Your name is stored in this browser only.</p>
      </form>
    </div>
  );
}
