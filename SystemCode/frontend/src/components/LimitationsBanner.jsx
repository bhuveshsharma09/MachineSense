import { useState } from "react";
import { K } from "../lib/theme.js";

// Always visible: the first limitations in one line; the full list (from the bundle manifest) on demand.
export default function LimitationsBanner({ items }) {
  const [open, setOpen] = useState(false);
  const list = items?.length ? items : ["Research demonstration; development results; not validated for real machines."];
  return (
    <div role="note" aria-label="Limitations" style={{ position: "absolute", top: 84, left: 16, right: 16, zIndex: 9, display: "flex", flexDirection: "column", gap: 6, padding: "6px 14px", borderRadius: 12, background: "oklch(0.80 0.14 75 / 0.16)", border: "1px solid oklch(0.70 0.13 70 / 0.35)", fontSize: 12.5, color: "oklch(0.34 0.05 60)", backdropFilter: "blur(14px)", WebkitBackdropFilter: "blur(14px)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span className="mono" style={{ fontSize: 9.5, letterSpacing: "0.08em", textTransform: "uppercase", color: K.amberT, flex: "none" }}>Limitations</span>
        <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{list[0]} {list[1] || ""}</span>
        <button onClick={() => setOpen(!open)} aria-expanded={open} style={{ border: 0, background: "transparent", cursor: "pointer", fontSize: 12, color: K.amberT, flex: "none" }}>{open ? "Hide" : `All ${list.length}`}</button>
      </div>
      {open && (
        <ul style={{ margin: "0 0 4px", paddingLeft: 18, display: "flex", flexDirection: "column", gap: 3, lineHeight: 1.4 }}>
          {list.map((x) => <li key={x}>{x}</li>)}
        </ul>
      )}
    </div>
  );
}
