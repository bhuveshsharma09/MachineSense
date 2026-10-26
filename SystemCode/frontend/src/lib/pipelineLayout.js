// Pipeline canvas layout (App-8: left to right, after the designer's "How this diagnosis was made"). The single source of
// truth for block positions and edge routes, derived from the trace's blocks, ports and edges; STAGES only says which
// column a block id sits in. A block with no stage is drawn in an "unplaced" lane at the bottom (with a console warning),
// never dropped.
//
// Shape: six stage columns read left to right (Recording · Preparation and quality · Features · Models · Decision ·
// Outputs). Inputs enter a block on its left side and outputs leave on its right; a port whose edges only link a block
// to the one directly below it sits on the bottom / top edge instead, and a port whose edges stay inside one column sits
// on the right edge. Edges run in lanes in the gutters between columns; an edge that skips columns crosses them at a
// height where they are empty (a corridor above or below a column's blocks).

export const GEOM = {
  H: 100,            // block height (title up to two lines and the case line)
  PAD_X: 16,         // block side padding
  TITLE_FIT: 1.08,   // estimated title width × this ≥ the measured width of every title (Switzer 16px)
  TITLE_PX: 15,      // block title size on this layout
  GAP_Y: 18,         // vertical space between blocks of one column
  FRAME_PAD: 14, FRAME_TOP: 30, FRAME_BOTTOM: 14,
  STAGE_H: 60,       // the numbered stage heading above each column
  GUTTER_MIN: 64,    // smallest gap between two columns
  LANE: 8,           // spacing of parallel vertical runs in a gutter
  TRACK: 8,          // spacing of parallel horizontal runs in a corridor
  CLEAR: 14,         // clearance of a corridor from the blocks it passes
  PORT_STEP: 22,     // minimum distance between two ports on one side
  RADIUS: 8,         // rounded corners of edge routes
  MARGIN: 4,         // around the whole graph
};

// The decision panel: four check rows, each with its input port, and the verdict box with the output port.
export const DEC = { W: 256, HEAD: 92, ROW: 52, ROW_GAP: 6, VERDICT: 96, PAD: 14 };
DEC.H = DEC.HEAD + 4 * DEC.ROW + 3 * DEC.ROW_GAP + 12 + DEC.VERDICT + 30;

export const STAGES = [
  { title: "Recording", desc: "The raw sensor file", arch: "A 5-second, 8-channel file", w: 184, ids: ["input", "validation"] },
  { title: "Preparation and quality", desc: "Is the recording usable?", arch: "Speed, cleaning and quality checks", w: 214, ids: ["preparation", "signal_quality"] },
  { title: "Features", desc: "Signals turned into pictures and numbers", arch: "Model inputs", w: 204, ids: ["order_spectrogram", "physics_features"] },
  { title: "Models", desc: "Compared with what the models learned", arch: "Classifier, familiarity and healthy checks", w: 214,
    ids: ["cnn6", "mahalanobis", "class6", "ae6", "b1s", "lr"] },
  { title: "Decision", desc: "Do the checks agree?", arch: "A fixed rule over the checks", w: DEC.W, ids: ["decision"] },
  { title: "Outputs", desc: "What is saved and shown to you", arch: "Evidence record, explanation, views", w: 204,
    ids: ["evidence_record", "explanation", "presentation"] },
];
export const PLACEMENT = Object.fromEntries(STAGES.flatMap((s, c) => s.ids.map((id, i) => [id, { col: c, i }])));

// App-8.6 item 24: plain Developer names for steps whose trace title carries a code; the code stays visible (devCode)
export const DEV_PLAIN = { signal_quality: "Signal quality", decision: "Decision rule", explanation: "Written explanation",
  physics_features: "Physics features", order_spectrogram: "Order spectrogram" };
export const devTitle = (b) => DEV_PLAIN[b.id] || b.title_developer || b.id;
export const devCode = (b) => (DEV_PLAIN[b.id] && b.title_developer && b.title_developer !== DEV_PLAIN[b.id] ? b.title_developer : null);
const title = (b, level) => (level === "developer" ? devTitle(b) : b.title_mechanic) || b.id;

// Width of a title at 16px, by character class (see GEOM.TITLE_FIT; check against Switzer rendering)
const charW = (ch) => (/[ilj.,:;'!|()[\] ]/.test(ch) ? 4.4 : /[ftrI\-_]/.test(ch) ? 5.8 : /[mwMW]/.test(ch) ? 12.2
  : /[A-Z]/.test(ch) ? 9.9 : /[0-9]/.test(ch) ? 8.6 : 8.3);
export const titleWidth = (s) => [...(s || "")].reduce((a, c) => a + charW(c), 0) * GEOM.TITLE_FIT;
// a title fits its block if it takes at most two lines at the block's title size
export const titleFits = (n) => titleWidth(devTitle(n.block)) * (GEOM.TITLE_PX / 16) <= 2 * (n.w - 2 * GEOM.PAD_X);

function flatten(blocks, parent = null, out = []) {
  for (const b of blocks || []) {
    out.push({ block: b, parent });
    if (b.children?.length) flatten(b.children, b.id, out);
  }
  return out;
}

export class LayoutError extends Error {}

// ------------------------------------------------------------------------------------------------------------ layout
// extraHeight: {block id: px} added to a block's height (case mode: the preview thumbnail under the text).
export function layoutPipeline(trace, { warn = (m) => console.warn(m), extraHeight = {} } = {}) {
  if (!trace || !Array.isArray(trace.blocks) || !Array.isArray(trace.edges)) throw new LayoutError("not a trace");
  const all = flatten(trace.blocks);
  const groups = all.filter((x) => x.block.children?.length);
  const leaves = all.filter((x) => !x.block.children?.length);
  const placed = leaves.filter((x) => PLACEMENT[x.block.id]);
  const unplaced = leaves.filter((x) => !PLACEMENT[x.block.id]);
  if (unplaced.length) warn(`pipeline layout: no placement for ${unplaced.map((x) => x.block.id).join(", ")}; drawn in the unplaced lane`);
  const nodes = {};
  for (const x of leaves) {
    const pl = PLACEMENT[x.block.id];
    const w = pl ? STAGES[pl.col].w : 214;
    nodes[x.block.id] = { id: x.block.id, block: x.block, parent: x.parent, col: pl ? pl.col : null, i: pl ? pl.i : null, w };
  }
  const groupOf = {};
  for (const g of groups) for (const c of g.block.children) groupOf[c.id] = g.block.id;
  for (const e of trace.edges) {
    const [a, b] = [e.from.split(".")[0], e.to.split(".")[0]];
    if (!nodes[a] || !nodes[b]) throw new LayoutError(`edge ${e.from} → ${e.to} names a block the trace does not declare`);
  }
  const portDecl = (key, side) => {
    const [bid, pid] = key.split(".");
    return (side === "out" ? nodes[bid].block.ports_out : nodes[bid].block.ports_in || []).find((p) => p.id === pid);
  };
  for (const e of trace.edges) {
    if (!portDecl(e.from, "out") || !portDecl(e.to, "in")) throw new LayoutError(`edge ${e.from} → ${e.to} names a port the trace does not declare`);
  }

  // 1. column order (placed blocks), ports' sides
  const cols = STAGES.map((_, c) => Object.values(nodes).filter((n) => n.col === c).sort((a, b) => a.i - b.i));
  const below = (a, b) => a.col !== null && a.col === b.col && cols[a.col].indexOf(b) === cols[a.col].indexOf(a) + 1;
  const ends = (e) => [nodes[e.from.split(".")[0]], nodes[e.to.split(".")[0]]];
  const outSide = {}, inSide = {};
  for (const n of Object.values(nodes)) for (const p of n.block.ports_out || []) {
    const es = trace.edges.filter((e) => e.from === `${n.id}.${p.id}`);
    outSide[`${n.id}.${p.id}`] = es.length && es.every((e) => below(n, ends(e)[1])) ? "bottom" : "right";
  }
  for (const n of Object.values(nodes)) for (const p of n.block.ports_in || []) {
    const key = `${n.id}.${p.id}`, es = trace.edges.filter((e) => e.to === key);
    const src = es.map((e) => ends(e)[0]);
    inSide[key] = es.length && es.every((e) => outSide[e.from] === "bottom" && below(ends(e)[0], n)) ? "top"
      : es.length && n.col !== null && src.every((s) => s.col === n.col) ? "right" : "left";
  }

  // 2. heights and y: each column stacked, centred on the tallest; frames add their caption band
  const sideCount = (n, side) => [...(n.block.ports_in || []).filter((p) => inSide[`${n.id}.${p.id}`] === side),
    ...(n.block.ports_out || []).filter((p) => outSide[`${n.id}.${p.id}`] === side)].length;
  for (const n of Object.values(nodes)) {
    n.h = n.id === "decision" ? DEC.H
      : Math.max(GEOM.H, (sideCount(n, "left") + 1) * GEOM.PORT_STEP, (sideCount(n, "right") + 1) * GEOM.PORT_STEP) + (extraHeight[n.id] || 0);
  }
  const colH = cols.map((ns) => {
    let h = 0, prevG = null;
    ns.forEach((n, k) => {
      const g = groupOf[n.id] || null;
      if (g !== prevG && prevG) h += GEOM.FRAME_BOTTOM;
      if (g !== prevG && g) h += (k ? GEOM.GAP_Y : 0) + GEOM.FRAME_TOP; else if (k) h += GEOM.GAP_Y;
      h += n.h; prevG = g;
    });
    return h + (prevG ? GEOM.FRAME_BOTTOM : 0);
  });
  const top = GEOM.STAGE_H, H = Math.max(...colH);
  cols.forEach((ns, c) => {
    let y = top + (H - colH[c]) / 2, prevG = null;
    ns.forEach((n, k) => {
      const g = groupOf[n.id] || null;
      if (g !== prevG && prevG) y += GEOM.FRAME_BOTTOM;
      if (g !== prevG && g) y += (k ? GEOM.GAP_Y : 0) + GEOM.FRAME_TOP; else if (k) y += GEOM.GAP_Y;
      n.y = y; y += n.h; prevG = g;
    });
  });
  const blocksBottom = top + H;

  // 3. ports' y (sides) and order: left inputs ordered by where their sources sit, to avoid crossings
  const ports = {};
  for (const n of Object.values(nodes)) {
    const list = [...(n.block.ports_in || []).map((p) => ({ p, side: inSide[`${n.id}.${p.id}`], dir: "in" })),
      ...(n.block.ports_out || []).map((p) => ({ p, side: outSide[`${n.id}.${p.id}`], dir: "out" }))];
    for (const side of ["left", "right", "top", "bottom"]) {
      const xs = list.filter((x) => x.side === side);
      if (n.id === "decision" && side === "left") {
        const order = ["top_msp", "maha", "ae", "b1s"];
        xs.sort((a, b) => order.indexOf(a.p.id) - order.indexOf(b.p.id));
        xs.forEach((x, i) => { ports[`${n.id}.${x.p.id}`] = { ...x.p, node: n.id, side, dir: x.dir, y: n.y + DEC.HEAD + i * (DEC.ROW + DEC.ROW_GAP) + DEC.ROW / 2 }; });
        continue;
      }
      if (n.id === "decision" && side === "right") {
        xs.forEach((x) => { ports[`${n.id}.${x.p.id}`] = { ...x.p, node: n.id, side, dir: x.dir, y: n.y + n.h - 42 - DEC.VERDICT / 2 }; });
        continue;
      }
      if (side === "left") {
        const srcY = (x) => { const s = trace.edges.filter((e) => e.to === `${n.id}.${x.p.id}`).map((e) => nodes[e.from.split(".")[0]]);
          return s.length ? s.reduce((a, m) => a + m.y + m.h / 2, 0) / s.length : n.y; };
        xs.sort((a, b) => srcY(a) - srcY(b));
      }
      xs.forEach((x, i) => {
        const k = `${n.id}.${x.p.id}`;
        ports[k] = { ...x.p, node: n.id, side, dir: x.dir, i, n: xs.length };
        if (side === "left" || side === "right") ports[k].y = n.y + Math.min(n.h, (xs.length + 1) * GEOM.PORT_STEP + 8) * (i + 1) / (xs.length + 1) - 4;
      });
    }
  }

  // 4. route topology (y only): kinds, gutter lanes and corridors
  const routes = trace.edges.map((e) => {
    const A = ports[e.from], B = ports[e.to], a = nodes[A.node], b = nodes[B.node];
    const r = { edge: e, from: e.from, to: e.to, A, B, a, b };
    if (A.side === "bottom" && B.side === "top") return { ...r, kind: "vertical" };
    if (a.col === null || b.col === null) return { ...r, kind: "unplaced" };
    if (A.side === "right" && (B.side === "right" || B.side === "top")) {
      return { ...r, kind: "loop", lanes: [{ g: a.col, key: "s:" + e.from, y0: A.y, y1: B.side === "top" ? b.y - GEOM.GAP_Y / 2 : B.y }] };
    }
    if (b.col === a.col + 1) return { ...r, kind: "next", lanes: [{ g: a.col, key: "s:" + e.from, y0: A.y, y1: B.y }] };
    if (b.col > a.col + 1) return { ...r, kind: "skip" };
    throw new LayoutError(`edge ${e.from} → ${e.to} runs backwards`);
  });
  // a skip edge crosses columns a.col+1 … b.col−1 at one height: its source's, its target's, or a corridor above or
  // below every block of those columns (the nearer one); corridors stack in tracks
  const busy = (c, y) => cols[c].some((n) => y > n.y - GEOM.CLEAR && y < n.y + n.h + GEOM.CLEAR)
    || groups.some((g) => cols[c].some((n) => groupOf[n.id] === g.block.id) && (() => {
      const kids = cols[c].filter((n) => groupOf[n.id] === g.block.id);
      return y > kids[0].y - GEOM.FRAME_TOP - 4 && y < kids[kids.length - 1].y + kids[kids.length - 1].h + GEOM.FRAME_BOTTOM + 4;
    })());
  const tracks = { above: [], below: [] };
  for (const r of routes.filter((x) => x.kind === "skip")) {
    const mid = [];
    for (let c = r.a.col + 1; c < r.b.col; c++) mid.push(c);
    const clear = (y) => mid.every((c) => !busy(c, y));
    if (clear(r.A.y)) { r.cy = r.A.y; r.lanes = [{ g: r.b.col - 1, key: "t:" + r.to, y0: r.A.y, y1: r.B.y }]; r.kind = "straight"; continue; }
    if (clear(r.B.y)) { r.cy = r.B.y; r.lanes = [{ g: r.a.col, key: "s:" + r.from, y0: r.A.y, y1: r.B.y }]; r.kind = "straight"; continue; }
    const hi = Math.min(...mid.flatMap((c) => cols[c].map((n) => n.y - (groupOf[n.id] ? GEOM.FRAME_TOP : 0))));
    const lo = Math.max(...mid.flatMap((c) => cols[c].map((n) => n.y + n.h + (groupOf[n.id] ? GEOM.FRAME_BOTTOM : 0))));
    const side = Math.abs((r.A.y + r.B.y) / 2 - hi) < Math.abs((r.A.y + r.B.y) / 2 - lo) && hi - GEOM.CLEAR > top ? "above" : "below";
    r.corridor = side;
    r.base = side === "above" ? hi - GEOM.CLEAR : lo + GEOM.CLEAR;
    r.span = [r.a.col, r.b.col];
    tracks[side].push(r);
  }
  // corridor tracks: edges sharing a source port share a track; others stack outwards when their spans overlap
  for (const side of ["above", "below"]) {
    const placedT = [];
    const rs = tracks[side].sort((p, q) => p.span[0] - q.span[0] || p.A.y - q.A.y);
    for (const r of rs) {
      const same = placedT.find((t) => t.from === r.from);
      let k = same ? same.k : 0;
      if (!same) while (placedT.some((t) => t.k === k && t.from !== r.from && t.span[0] < r.span[1] && r.span[0] < t.span[1]
        && Math.abs(t.base - r.base) < 1e-6)) k++;
      placedT.push({ from: r.from, k, span: r.span, base: r.base });
      r.cy = r.base + (side === "above" ? -1 : 1) * k * GEOM.TRACK;
      r.lanes = [{ g: r.a.col, key: "s:" + r.from, y0: r.A.y, y1: r.cy }, { g: r.b.col - 1, key: "t:" + r.to, y0: r.cy, y1: r.B.y }];
      r.kind = "corridor";
    }
  }

  // 5. lanes per gutter: one lane per source port (fan-out) or target port (fan-in); packed so that overlapping vertical
  // spans never share a slot; the gutter is as wide as its slots need
  const lanes = {};
  for (const r of routes) for (const l of r.lanes || []) {
    const id = `${l.g}|${l.key}`;
    const L = (lanes[id] ||= { g: l.g, key: l.key, y0: Infinity, y1: -Infinity, anchor: l.key.startsWith("s:") ? "src" : "dst" });
    L.y0 = Math.min(L.y0, l.y0, l.y1); L.y1 = Math.max(L.y1, l.y0, l.y1);
  }
  const byGutter = {};
  for (const L of Object.values(lanes)) (byGutter[L.g] ||= []).push(L);
  const slots = {};
  for (const [g, ls] of Object.entries(byGutter)) {
    // source lanes sit near the source column (left), target lanes near the target column (right); within each, the
    // longer spans outward
    const order = ls.sort((p, q) => (p.anchor === q.anchor ? (q.y1 - q.y0) - (p.y1 - p.y0) : p.anchor === "src" ? -1 : 1));
    const used = [];
    for (const L of order) {
      let k = 0;
      while (used.some((u) => u.k === k && u.y0 < L.y1 + 6 && L.y0 < u.y1 + 6)) k++;
      L.k = k; used.push(L);
    }
    slots[g] = Math.max(0, ...used.map((u) => u.k + 1));
  }
  const gutterW = (g) => (g === STAGES.length - 1 ? 16 + (slots[g] || 0) * GEOM.LANE
    : Math.max(GEOM.GUTTER_MIN, 2 * GEOM.FRAME_PAD + 24 + (slots[g] || 0) * GEOM.LANE));

  // 6. x: columns left to right; a frame's padding sits inside the gutter
  const colX = [];
  let x = 0;
  STAGES.forEach((s, c) => { colX[c] = x; x += s.w + gutterW(c); });
  for (const n of Object.values(nodes)) if (n.col !== null) n.x = colX[n.col];
  const laneX = (g, k, n = slots[g] || 1) => colX[g] + STAGES[g].w + gutterW(g) / 2 + (k - (n - 1) / 2) * GEOM.LANE;
  for (const L of Object.values(lanes)) L.x = laneX(L.g, L.k);

  // unplaced lane: below everything, in a row
  const corridorBottom = Math.max(blocksBottom, ...tracks.below.map((r) => r.cy)) + GEOM.CLEAR;
  const unplacedY = unplaced.length ? corridorBottom + 44 : null;
  let ux = 0;
  for (const u of unplaced) { const n = nodes[u.block.id]; n.x = ux; n.y = unplacedY; n.h = GEOM.H + (extraHeight[n.id] || 0); ux += n.w + GEOM.GUTTER_MIN; }

  for (const [k, p] of Object.entries(ports)) {
    const n = nodes[p.node];
    if (p.side === "left") p.x = n.x;
    else if (p.side === "right") p.x = n.x + n.w;
    else { p.x = n.x + (n.w * (p.i + 1)) / (p.n + 1); p.y = p.side === "top" ? n.y : n.y + n.h; }
    if (p.y === undefined) p.y = n.y + n.h / 2;
    // micro-label: outside the block, beside the port
    const w = (p.short || "").length * 6.4;
    p.lx = p.side === "left" ? p.x - 7 : p.side === "right" ? p.x + 7 : p.x + 6;
    p.ly = p.side === "top" ? p.y - 5 : p.side === "bottom" ? p.y + 13 : p.y - 6;
    p.anchor = p.side === "left" ? "end" : "start";
    p.box = { x0: p.anchor === "end" ? p.lx - w - 2 : p.lx - 2, x1: p.anchor === "end" ? p.lx + 2 : p.lx + w + 2, y0: p.ly - 10, y1: p.ly + 3 };
    void k;
  }

  // 7. points
  const laneOf = (g, key) => lanes[`${g}|${key}`].x;
  const edges = routes.map((r) => {
    const { A, B } = r;
    let pts;
    if (r.kind === "vertical") {
      const my = (A.y + B.y) / 2;
      pts = Math.abs(A.x - B.x) < 0.5 ? [[A.x, A.y], [B.x, B.y]] : [[A.x, A.y], [A.x, my], [B.x, my], [B.x, B.y]];
    } else if (r.kind === "unplaced") {
      const my = Math.max(A.y, B.y) + 20;
      pts = [[A.x, A.y], [A.x + 12, A.y], [A.x + 12, my], [B.x - 12, my], [B.x - 12, B.y], [B.x, B.y]];
    } else if (r.kind === "loop") {
      const lx = laneOf(r.a.col, "s:" + r.from);
      pts = B.side === "top" ? [[A.x, A.y], [lx, A.y], [lx, r.b.y - GEOM.GAP_Y / 2], [B.x, r.b.y - GEOM.GAP_Y / 2], [B.x, B.y]]
        : [[A.x, A.y], [lx, A.y], [lx, B.y], [B.x, B.y]];
    } else if (r.kind === "next") {
      const lx = laneOf(r.a.col, "s:" + r.from);
      pts = [[A.x, A.y], [lx, A.y], [lx, B.y], [B.x, B.y]];
    } else if (r.kind === "straight") {
      const g = r.cy === A.y ? r.b.col - 1 : r.a.col, key = r.cy === A.y ? "t:" + r.to : "s:" + r.from;
      const lx = laneOf(g, key);
      pts = [[A.x, A.y], [lx, A.y], [lx, B.y], [B.x, B.y]];
    } else {
      const l1 = laneOf(r.a.col, "s:" + r.from), l2 = laneOf(r.b.col - 1, "t:" + r.to);
      pts = [[A.x, A.y], [l1, A.y], [l1, r.cy], [l2, r.cy], [l2, B.y], [B.x, B.y]];
    }
    pts = pts.filter((p, i) => i === 0 || Math.abs(p[0] - pts[i - 1][0]) > 1e-6 || Math.abs(p[1] - pts[i - 1][1]) > 1e-6);
    return { ...r.edge, id: `${r.from}->${r.to}`, points: pts, d: roundedPath(pts) };
  });

  // frames around each group's blocks, caption on the top edge
  const frames = [];
  for (const g of groups) {
    const kids = g.block.children.map((c) => nodes[c.id]).filter(Boolean);
    if (!kids.length) continue;
    const fx0 = Math.min(...kids.map((k) => k.x)) - GEOM.FRAME_PAD, fx1 = Math.max(...kids.map((k) => k.x + k.w)) + GEOM.FRAME_PAD;
    const fy0 = Math.min(...kids.map((k) => k.y)) - GEOM.FRAME_TOP + 6, fy1 = Math.max(...kids.map((k) => k.y + k.h)) + GEOM.FRAME_BOTTOM;
    const capW = Math.max(titleWidth(g.block.title_mechanic), titleWidth(g.block.title_developer)) * (14 / 16) + 20;
    frames.push({ id: g.block.id, block: g.block, x: fx0, y: fy0, w: fx1 - fx0, h: fy1 - fy0, members: new Set(kids.map((k) => k.id)),
      captionX: fx0 + 10, captionW: Math.min(capW, fx1 - fx0 - 20) });
  }

  const colourKind = {};
  for (const e of edges) { colourKind[e.from] ??= e.kind; colourKind[e.to] ??= e.kind; }
  for (const [k, p] of Object.entries(ports)) p.kind = colourKind[k] || null;

  const stages = STAGES.map((s, c) => ({ ...s, n: c + 1, x: colX[c], y: 0, w: s.w }));
  const xs = [...Object.values(nodes).flatMap((n) => [n.x, n.x + n.w]), ...frames.flatMap((f) => [f.x, f.x + f.w]),
    ...edges.flatMap((e) => e.points.map((p) => p[0])), ...Object.values(ports).flatMap((p) => [p.box.x0, p.box.x1])];
  const ys = [0, ...Object.values(nodes).flatMap((n) => [n.y, n.y + n.h + 16]), ...frames.flatMap((f) => [f.y - 12, f.y + f.h]),
    ...edges.flatMap((e) => e.points.map((p) => p[1]))];
  const bounds = { x: Math.min(...xs) - GEOM.MARGIN, y: Math.min(...ys) - GEOM.MARGIN };
  bounds.w = Math.max(...xs) + GEOM.MARGIN - bounds.x;
  bounds.h = Math.max(...ys) + GEOM.MARGIN - bounds.y;

  const layout = { nodes, frames, ports, edges, bounds, stages, order: all.map((x) => x.block.id), unplaced: unplaced.map((x) => x.block.id),
    unplacedY, title, direction: "lr" };
  layout.labels = (visible) => placeLabels(layout, visible);
  return layout;
}

export function roundedPath(pts, r = GEOM.RADIUS) {
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i], [ax, ay] = pts[i - 1], [bx, by] = pts[i + 1];
    const l1 = Math.hypot(px - ax, py - ay), l2 = Math.hypot(bx - px, by - py);
    const k = Math.min(r, l1 / 2, l2 / 2);
    const s = [px - ((px - ax) / l1) * k, py - ((py - ay) / l1) * k], t = [px + ((bx - px) / l2) * k, py + ((by - py) / l2) * k];
    d += ` L${s[0]},${s[1]} Q${px},${py} ${t[0]},${t[1]}`;
  }
  const z = pts[pts.length - 1];
  return d + ` L${z[0]},${z[1]}`;
}

// ------------------------------------------------------------------------------------------------------------ labels
// Edge labels (10px) beside a vertical run, or above a horizontal run, of their edge, only where they collide with
// nothing: no block, port label, frame caption, other shown label or other edge's line. `visible` is the set of edge ids
// whose labels are shown (the hovered or selected block's edges, or all); returns {edge id: {x, y, anchor}}.
export const LABEL = { CHAR_W: 5.1, H: 12, GAP: 6, MIN_RUN: 30 };
const hits = (a, b) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;

function placeLabels(layout, visible) {
  const { nodes, ports, frames, edges } = layout;
  const boxes = Object.values(nodes).map((n) => ({ x0: n.x - 4, x1: n.x + n.w + 4, y0: n.y - 4, y1: n.y + n.h + 4 }));
  for (const f of frames) boxes.push({ x0: f.captionX - 4, x1: f.captionX + f.captionW + 4, y0: f.y - 13, y1: f.y + 13 });
  for (const p of Object.values(ports)) boxes.push(p.box);
  const segs = edges.map((e) => e.points.slice(1).map((q, i) => {
    const [x0, y0] = e.points[i], [x1, y1] = q;
    return { x0: Math.min(x0, x1) - 2, x1: Math.max(x0, x1) + 2, y0: Math.min(y0, y1) - 2, y1: Math.max(y0, y1) + 2 };
  }));
  const placed = [];
  const out = {};
  edges.forEach((e, ei) => {
    if (!e.label || !visible.has(e.id)) return;
    const w = e.label.length * LABEL.CHAR_W;
    const runs = e.points.slice(1).map((q, i) => [e.points[i], q]).filter(([a, b]) => Math.hypot(b[0] - a[0], b[1] - a[1]) >= LABEL.MIN_RUN)
      .sort((r, t) => Math.hypot(t[1][0] - t[0][0], t[1][1] - t[0][1]) - Math.hypot(r[1][0] - r[0][0], r[1][1] - r[0][1]));
    // first choice: a chip just before the target port (drawn with a background halo, so it may sit over other lines)
    const z = e.points[e.points.length - 1], z0 = e.points[e.points.length - 2];
    // (shortened with an ellipsis to the free space before the port; the full label is the line's tooltip and in the drawer)
    if (z0 && z0[1] === z[1]) {
      for (const dy of [-9, 13, -21, 25]) {
        const y0 = z[1] + dy - LABEL.H / 2 - 1, y1 = z[1] + dy + LABEL.H / 2 + 1, right = z[0] - 10;
        const left = Math.max(-Infinity, ...[...boxes, ...placed].filter((o) => o.y0 < y1 && o.y1 > y0 && o.x1 <= right + 2).map((o) => o.x1));
        const chars = Math.floor((right - left - 6) / LABEL.CHAR_W);
        if (chars < 8) continue;
        const cut = e.label.length > chars;
        const tw = (cut ? chars : e.label.length) * LABEL.CHAR_W;
        const box = { x0: right - tw - 2, x1: right + 2, y0, y1 };
        if (!boxes.some((o) => hits(box, o)) && !placed.some((o) => hits(box, o))) {
          out[e.id] = { x: right - tw, y: z[1] + dy + 3.5, anchor: "start", ...(cut ? { clipW: tw - LABEL.CHAR_W } : {}) };
          placed.push(box);
          return;
        }
      }
    }
    for (const [a, b] of runs) {
      const vertical = a[0] === b[0];
      for (const f of [0.5, 0.3, 0.7]) {
        const cands = vertical
          ? [1, -1].map((side) => ({ x: side > 0 ? a[0] + LABEL.GAP : a[0] - LABEL.GAP - w, y: a[1] + (b[1] - a[1]) * f }))
          : [-1, 1].map((side) => ({ x: a[0] + (b[0] - a[0]) * f - w / 2, y: a[1] + side * (LABEL.H / 2 + 3) }));
        for (const c of cands) {
          const box = { x0: c.x - 2, x1: c.x + w + 2, y0: c.y - LABEL.H / 2 - 1, y1: c.y + LABEL.H / 2 + 1 };
          const clash = boxes.some((o) => hits(box, o)) || placed.some((o) => hits(box, o))
            || segs.some((ss, j) => j !== ei && ss.some((o) => hits(box, o)));
          if (!clash) {
            out[e.id] = { x: c.x, y: c.y + 3.5, anchor: "start" };
            placed.push(box);
            return;
          }
        }
      }
    }
  });
  return out;
}

// ------------------------------------------------------------------------------------------------------------ checks
// Geometry checks used by the tests and the report: an edge segment inside a block, and perpendicular line crossings.
export function edgeThroughBlock(layout) {
  const bad = [];
  for (const e of layout.edges) for (let i = 1; i < e.points.length; i++) {
    const [x0, y0] = e.points[i - 1], [x1, y1] = e.points[i];
    for (const n of Object.values(layout.nodes)) {
      if (Math.max(x0, x1) > n.x + 1 && Math.min(x0, x1) < n.x + n.w - 1 && Math.max(y0, y1) > n.y + 1 && Math.min(y0, y1) < n.y + n.h - 1) bad.push(`${e.id} through ${n.id}`);
    }
  }
  return bad;
}

export function countCrossings(layout) {
  const segs = layout.edges.flatMap((e) => e.points.slice(1).map((q, i) => ({ e: e.id, a: e.points[i], b: q })));
  let n = 0;
  for (let i = 0; i < segs.length; i++) for (let j = i + 1; j < segs.length; j++) {
    const s = segs[i], t = segs[j];
    if (s.e === t.e) continue;
    const sv = s.a[0] === s.b[0], tv = t.a[0] === t.b[0];
    if (sv === tv) continue;
    const [v, h] = sv ? [s, t] : [t, s];
    const vx = v.a[0], hy = h.a[1];
    if (vx > Math.min(h.a[0], h.b[0]) + 0.5 && vx < Math.max(h.a[0], h.b[0]) - 0.5 && hy > Math.min(v.a[1], v.b[1]) + 0.5 && hy < Math.max(v.a[1], v.b[1]) - 0.5) n++;
  }
  return n;
}
