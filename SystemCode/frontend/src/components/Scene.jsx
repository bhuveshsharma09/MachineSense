import { useEffect, useRef, useState } from "react";
import { K, al } from "../lib/theme.js";
import { PARTS } from "../lib/model.js";

export const INSETS = { top: 124, right: 412, bottom: 190, left: 16 };
export const FALLBACK = { img: "rig-fallback.png", anchors: "rig-fallback.json" };

// One marker per part/sensor. Colours follow the design: orange = location according to the dataset label (model),
// amber dashed = candidate, lime = largest measured change vs the healthy reference, amber sensor = quality flag.
export function markerFor(h, { fault, cand, unusual, flags, sel, hover, labelsAll }) {
  const isS = h.kind === "sensor", illus = h.pos === "illus";
  let fill = isS ? (illus ? "oklch(0.995 0.002 85 / 0.92)" : K.ink) : K.white;
  let border = isS ? (illus ? "1.5px dashed " + K.ink : "2px solid " + K.white) : "2px solid " + K.ink;
  let tag = "", tagColor = K.ink3, labelBorder = "oklch(1 0 0 / 0.9)", ring = null, pulse = null;
  if (isS && flags[h.id]) {
    fill = K.amber; border = "2px solid " + K.white; tag = flags[h.id]; tagColor = K.amberT; pulse = al(K.amber, 0.6); labelBorder = al(K.amber, 0.6);
  } else if (isS && unusual.includes(h.id)) {
    fill = K.lime; border = "2px solid " + K.ink; tag = "Change beyond typical healthy variation · measured"; tagColor = K.limeT; pulse = al(K.lime, 0.75); labelBorder = al(K.limeT, 0.45);
  }
  if (!isS && fault.includes(h.id)) {
    fill = K.orange; border = "2px solid " + K.white; ring = { b: "1.5px solid " + K.orange, bg: al(K.orange, 0.14) };
    tag = "Location according to the dataset label (model)"; tagColor = K.orangeT; labelBorder = al(K.orange, 0.55);
  } else if (!isS && cand.includes(h.id)) {
    fill = K.amber; border = "2px solid " + K.white; ring = { b: "1.5px dashed " + K.amberT, bg: al(K.amber, 0.16) };
    tag = "Candidate location · top-2 classes (model)"; tagColor = K.amberT; labelBorder = al(K.amber, 0.6);
  }
  const selected = sel === h.id;
  return { id: h.id, fill, border, size: isS ? 10 : 11, ring, pulse, selected, tag, tagColor,
    showLabel: isS && !!labelsAll, label: h.full || h.label,
    labelBorder: selected ? al(K.violet, 0.55) : labelBorder };
}

export function Marker({ m, onClick, onEnter, onLeave, style }) {
  return (
    <div data-anchor={m.id} style={{ position: "absolute", left: 0, top: 0, width: 0, height: 0, willChange: "transform", ...style }}>
      <button onClick={onClick} onMouseEnter={onEnter} onMouseLeave={onLeave} aria-pressed={m.selected}
        aria-label={`${m.label}${m.tag ? ", " + m.tag : ""}. Show details.`}
        style={{ position: "absolute", left: -16, top: -16, width: 32, height: 32, padding: 0, border: 0, background: "transparent", borderRadius: "50%", pointerEvents: "auto", cursor: "pointer", display: "grid", placeItems: "center" }}>
        {m.ring && <span style={{ position: "absolute", inset: -2, borderRadius: "50%", border: m.ring.b, background: m.ring.bg }} />}
        {m.pulse && <span style={{ position: "absolute", left: "50%", top: "50%", width: 12, height: 12, margin: "-6px 0 0 -6px", borderRadius: "50%", background: m.pulse, animation: "ms-pulse 1.9s ease-out infinite" }} />}
        {m.selected && <span style={{ position: "absolute", left: "50%", top: "50%", width: 24, height: 24, margin: "-12px 0 0 -12px", borderRadius: "50%", border: "2px solid " + K.violet, boxShadow: "0 0 0 4px " + al(K.violet, 0.16) }} />}
        <span style={{ position: "relative", width: m.size, height: m.size, borderRadius: "50%", background: m.fill, border: m.border, boxShadow: "0 1px 4px oklch(0.3 0.01 260 / 0.35)" }} />
      </button>
      {m.showLabel && (
        <div className="ms-label" style={{ borderColor: m.labelBorder }}>
          <span>{m.label}</span>
        </div>
      )}
    </div>
  );
}

export default function Scene({ state, onSelect, onHover, inspector, forceStatic = false, captureMode = false, insets = INSETS, zoomOut = 1 }) {
  const sceneRef = useRef(null), overlayRef = useRef(null), rig = useRef(null);
  const [mode, setMode] = useState(forceStatic ? "static" : "loading");
  const [anchors, setAnchors] = useState(null);
  const cb = useRef({ onSelect, onHover, sel: state.sel });
  cb.current = { onSelect, onHover, sel: state.sel };

  useEffect(() => {
    if (forceStatic) return undefined;
    let dead = false;
    import("../rig/rigScene.js").then((mod) => {
      if (dead || !sceneRef.current) return;
      try {
        const rm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");
        rig.current = mod.createRig(sceneRef.current, {
          overlay: overlayRef.current, reducedMotion: !!(rm && rm.matches),
          onPick: (id) => cb.current.onSelect(id && id === cb.current.sel ? null : id),
          onHover: (id) => cb.current.onHover(id),
        });
        setMode("3d");
        if (captureMode) window.__rigReady = true;
      } catch (e) {
        console.warn("3D view unavailable", e);
        setMode("static");
      }
    }).catch((e) => { console.warn("3D view unavailable", e); setMode("static"); });
    return () => { dead = true; if (rig.current) rig.current.dispose(); rig.current = null; };
  }, [forceStatic, captureMode]);

  useEffect(() => {
    if (mode !== "static" || anchors) return;
    fetch(FALLBACK.anchors).then((r) => r.json()).then(setAnchors).catch(() => setAnchors({}));
  }, [mode, anchors]);

  useEffect(() => {
    if (!rig.current || mode !== "3d") return;
    rig.current.update({
      sel: state.sel, hover: state.hover, exploded: state.exploded, fault: state.fault, cand: state.cand,
      unusual: state.unusual, unusable: Object.keys(state.flags), insets: captureMode ? { top: 0, right: 0, bottom: 0, left: 0 } : insets, zoomOut,
    });
  });

  useEffect(() => { if (state.resetTick && rig.current) rig.current.resetView(); }, [state.resetTick]);

  const markers = PARTS.map((h) => markerFor(h, state));
  const handlers = (m) => ({
    onClick: () => onSelect(state.sel === m.id ? null : m.id),
    onEnter: () => onHover(m.id), onLeave: () => onHover(null),
  });

  if (mode === "static") {
    const a = anchors || {};
    return (
      <>
        <div style={{ position: "absolute", left: insets.left, right: insets.right, top: insets.top, bottom: insets.bottom, zIndex: 0, display: "grid", placeItems: "center" }}>
          <div style={{ position: "relative", width: "min(100%, calc((100vh - 314px) * 1.6))", aspectRatio: "1.6" }}>
            <img src={FALLBACK.img} alt="Rendered frame of the illustrative 3D rig" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", mixBlendMode: "multiply", WebkitMaskImage: "radial-gradient(ellipse 50% 50% at 50% 50%, #000 70%, transparent 100%)", maskImage: "radial-gradient(ellipse 50% 50% at 50% 50%, #000 70%, transparent 100%)" }} />
            <div style={{ position: "absolute", inset: 0, zIndex: 3, pointerEvents: "none" }}>
              {markers.filter((m) => a[m.id]).map((m) => (
                <Marker key={m.id} m={m} {...handlers(m)} style={{ left: a[m.id][0] * 100 + "%", top: a[m.id][1] * 100 + "%" }} />
              ))}
            </div>
          </div>
          <span className="mono kicker" style={{ position: "absolute", bottom: 4 }}>Static view · rendered frame of the 3D rig (WebGL unavailable)</span>
        </div>
        {inspector && <div style={{ position: "absolute", left: 36, top: insets.top + 8, zIndex: 6 }}>{inspector}</div>}
      </>
    );
  }
  return (
    <>
      <div ref={sceneRef} style={{ position: "absolute", inset: 0, zIndex: 0 }} />
      {mode === "loading" && (
        <div className="mono kicker" style={{ position: "absolute", left: insets.left, right: insets.right, top: insets.top, bottom: insets.bottom, zIndex: 0, display: "grid", placeItems: "center", pointerEvents: "none" }}>Loading 3D rig…</div>
      )}
      <div ref={overlayRef} className="overlay" style={{ position: "absolute", inset: 0, zIndex: 3, pointerEvents: "none" }}>
        {markers.map((m) => <Marker key={m.id} m={m} {...handlers(m)} style={{ visibility: "hidden" }} />)}
        {inspector && <div data-follow="1" style={{ position: "absolute", left: 0, top: 0, opacity: 0, transition: "opacity 0.2s ease", pointerEvents: "auto" }}>{inspector}</div>}
      </div>
    </>
  );
}
