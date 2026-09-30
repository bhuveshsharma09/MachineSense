// Pipeline canvas layout (App-4.2): the single source of truth for block positions and edge routes. Everything is derived
// from the trace's blocks, ports and edges; PLACEMENT only says where a block id sits (grid units). A block with no
// placement is drawn in an "unplaced" lane at the bottom (with a console warning), never dropped.

export const GEOM = {
  BH: 76,            // block height: 16px 22px padding, one title line, one subtitle line
  MIN_W: 208, MAX_W: 340, CHAR_W: 8.1, PAD_X: 22,
  GAP_X: 64,         // horizontal space between neighbouring blocks
  GAP_Y: 150,        // vertical space between rows (edge channels run here)
  FRAME_PAD: 28, FRAME_TOP: 30, FRAME_BOTTOM: 30,
  TRACK: 9,          // spacing of parallel horizontal jogs in one channel
  LANE: 8,           // spacing of parallel vertical runs in one corridor
  CLEAR: 22,         // clearance of a vertical run from a block
  RADIUS: 9,         // rounded corners of edge routes
  MARGIN: 80,        // around the whole graph
};

// Where each block sits: row (top to bottom, trace order) and column (in units of the widest block + GAP_X; 0 = spine).
export const PLACEMENT = {
  input: { row: 0, col: 0 },
  validation: { row: 1, col: 0 },
  signal_quality: { row: 2, col: -1.2 },     // off the spine: its output goes to the evidence record only
  preparation: { row: 3, col: 0 },
  order_spectrogram: { row: 4, col: -0.55 },
  physics_features: { row: 4, col: 0.55 },
  cnn6: { row: 5, col: -1.6 },
  ae6: { row: 5, col: -0.55 },
  b1s: { row: 5, col: 0.55 },
  lr: { row: 5, col: 2.05 },                  // off the spine: it does not feed the decision
  class6: { row: 6, col: -2.25 },             // second tier of the models group: both read CNN-6's outputs
  mahalanobis: { row: 6, col: -1.1 },
  decision: { row: 7, col: 0 },
  evidence_record: { row: 8, col: 0 },
  explanation: { row: 9, col: 0 },
  presentation: { row: 10, col: 0 },
};

const title = (b, level) => (level === "developer" ? b.title_developer : b.title_mechanic) || b.id;

function blockWidth(b) {
  const longest = Math.max((b.title_mechanic || "").length, (b.title_developer || "").length);
  const ports = Math.max(b.ports_in?.length || 0, b.ports_out?.length || 0);
  const w = Math.max(GEOM.MIN_W, longest * GEOM.CHAR_W + 2 * GEOM.PAD_X, (ports + 1) * 46);
  return Math.round(Math.min(GEOM.MAX_W, w));
}

function flatten(blocks, parent = null, out = []) {
  for (const b of blocks || []) {
    out.push({ block: b, parent });
    if (b.children?.length) flatten(b.children, b.id, out);
  }
  return out;
}

export class LayoutError extends Error {}

// ------------------------------------------------------------------------------------------------------------ layout
export function layoutPipeline(trace, { warn = (m) => console.warn(m) } = {}) {
  if (!trace || !Array.isArray(trace.blocks) || !Array.isArray(trace.edges)) throw new LayoutError("not a trace");
  const all = flatten(trace.blocks);
  const groups = all.filter((x) => x.block.children?.length);
  const leaves = all.filter((x) => !x.block.children?.length);
  const U = Math.max(...leaves.map((x) => blockWidth(x.block))) + GEOM.GAP_X;
  const pitch = GEOM.BH + GEOM.GAP_Y;

  const placed = leaves.filter((x) => PLACEMENT[x.block.id]);
  const unplaced = leaves.filter((x) => !PLACEMENT[x.block.id]);
  const lastRow = Math.max(0, ...placed.map((x) => PLACEMENT[x.block.id].row));
  if (unplaced.length) warn(`pipeline layout: no placement for ${unplaced.map((x) => x.block.id).join(", ")}; drawn in the unplaced lane`);

  const nodes = {};
  const put = (x, row, cx) => {
    const w = blockWidth(x.block);
    nodes[x.block.id] = { id: x.block.id, block: x.block, parent: x.parent, row, w, h: GEOM.BH, x: cx - w / 2, y: row * pitch, cx };
  };
  for (const x of placed) put(x, PLACEMENT[x.block.id].row, PLACEMENT[x.block.id].col * U);
  const unplacedRow = lastRow + 1;
  unplaced.forEach((x, i) => put(x, unplacedRow, (i - (unplaced.length - 1) / 2) * U));

  // group frames around their children (dashed, no fill): the parent is drawn as a grouping, not a block
  const frames = [];
  for (const g of groups) {
    const kids = g.block.children.map((c) => nodes[c.id]).filter(Boolean);
    if (!kids.length) continue;
    const x0 = Math.min(...kids.map((k) => k.x)) - GEOM.FRAME_PAD, x1 = Math.max(...kids.map((k) => k.x + k.w)) + GEOM.FRAME_PAD;
    const y0 = Math.min(...kids.map((k) => k.y)) - GEOM.FRAME_TOP, y1 = Math.max(...kids.map((k) => k.y + k.h)) + GEOM.FRAME_BOTTOM;
    frames.push({ id: g.block.id, block: g.block, x: x0, y: y0, w: x1 - x0, h: y1 - y0, rows: [...new Set(kids.map((k) => k.row))],
      members: new Set(kids.map((k) => k.id)) });
  }

  // ports on the border line: inputs evenly on the top edge, outputs on the bottom edge
  const ports = {};
  for (const n of Object.values(nodes)) {
    const ins = n.block.ports_in || [], outs = n.block.ports_out || [];
    ins.forEach((p, i) => { ports[`${n.id}.${p.id}`] = { ...p, node: n.id, side: "in", x: n.x + (n.w * (i + 1)) / (ins.length + 1), y: n.y }; });
    outs.forEach((p, i) => { ports[`${n.id}.${p.id}`] = { ...p, node: n.id, side: "out", x: n.x + (n.w * (i + 1)) / (outs.length + 1), y: n.y + n.h }; });
  }

  // input ports ordered left to right by where their sources sit (fewer crossings); which port connects where is unchanged
  for (const n of Object.values(nodes)) {
    const ins = n.block.ports_in || [];
    if (ins.length < 2) continue;
    const at = (p) => {
      const src = trace.edges.filter((e) => e.to === `${n.id}.${p.id}`).map((e) => ports[e.from]?.x).filter((x) => x !== undefined);
      return src.length ? src.reduce((a, b) => a + b, 0) / src.length : ports[`${n.id}.${p.id}`].x;
    };
    const order = ins.map((p, i) => ({ p, i, x: at(p) })).sort((a, b) => a.x - b.x || a.i - b.i);
    order.forEach(({ p }, i) => { ports[`${n.id}.${p.id}`].x = n.x + (n.w * (i + 1)) / (ins.length + 1); });
  }
  const edges = routeEdges(trace.edges, nodes, frames, ports, pitch);
  placeLabels(edges, nodes, ports, frames);
  const colourKind = {};
  for (const e of edges) { colourKind[e.from] ??= e.kind; colourKind[e.to] ??= e.kind; }
  for (const [k, p] of Object.entries(ports)) p.kind = colourKind[k] || null;  // connected ports take their edge's colour

  const xs = [...Object.values(nodes).flatMap((n) => [n.x, n.x + n.w]), ...frames.flatMap((f) => [f.x, f.x + f.w]),
    ...edges.flatMap((e) => e.points.map((p) => p[0]))];
  const ys = [...Object.values(nodes).flatMap((n) => [n.y, n.y + n.h]), ...frames.flatMap((f) => [f.y, f.y + f.h]),
    ...edges.flatMap((e) => e.points.map((p) => p[1]))];
  const bounds = { x: Math.min(...xs) - GEOM.MARGIN, y: Math.min(...ys) - GEOM.MARGIN };
  bounds.w = Math.max(...xs) + GEOM.MARGIN - bounds.x;
  bounds.h = Math.max(...ys) + GEOM.MARGIN - bounds.y;

  const order = all.map((x) => x.block.id);  // trace order (tab order)
  return { nodes, frames, ports, edges, bounds, order, unplaced: unplaced.map((x) => x.block.id), unplacedRow: unplaced.length ? unplacedRow : null,
    pitch, title };
}

// ------------------------------------------------------------------------------------------------------------ routes
function rowSpan(n) { return [n.y, n.y + n.h]; }

// obstacles of the rows [r0, r1] for an edge: blocks and the group frames the edge is not inside. A downward run starts
// below its source and ends above its target, so those two are skipped; an upward run passes their rows (withEnds).
function obstacles(rows, nodes, frames, a, b, withEnds = false) {
  const out = [];
  for (const n of Object.values(nodes)) {
    if ((!withEnds && (n.id === a.id || n.id === b.id)) || n.row < rows[0] || n.row > rows[1]) continue;
    out.push([n.x - GEOM.CLEAR, n.x + n.w + GEOM.CLEAR]);
  }
  for (const f of frames) {
    if (f.members.has(a.id) || f.members.has(b.id)) continue;
    if (f.rows.some((r) => r >= rows[0] && r <= rows[1])) out.push([f.x - 12, f.x + f.w + 12]);
  }
  out.sort((p, q) => p[0] - q[0]);
  const merged = [];
  for (const o of out) {
    if (merged.length && o[0] <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], o[1]);
    else merged.push([...o]);
  }
  return merged;
}

function corridor(ax, bx, blocked, used, yRange) {
  const free = [];
  let lo = -Infinity;
  for (const [s, e] of blocked) { free.push([lo, s]); lo = e; }
  free.push([lo, Infinity]);
  let best = null;
  for (const [s, e] of free) {
    for (const want of [ax, bx, (ax + bx) / 2]) {
      const x = Math.min(Math.max(want, s + 1), e - 1);
      if (!(x > s && x < e)) continue;
      const cost = Math.abs(x - ax) + Math.abs(x - bx);
      if (!best || cost < best.cost) best = { x, cost, s, e };
    }
  }
  // keep parallel vertical runs apart
  let x = best.x;
  const clash = (v) => used.some((u) => Math.abs(u.x - v) < GEOM.LANE - 0.5 && u.y0 < yRange[1] && u.y1 > yRange[0]);
  for (let k = 1; clash(x) && k < 40; k++) {
    const cand = best.x + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * GEOM.LANE;
    if (cand > best.s && cand < best.e) x = cand;
  }
  used.push({ x, y0: yRange[0], y1: yRange[1] });
  return x;
}

function routeEdges(list, nodes, frames, ports, pitch) {
  const channel = (row) => row * pitch + GEOM.BH + GEOM.GAP_Y / 2;  // middle of the gap below `row`
  const tracks = {};  // channel row → [{x0, x1, i}]
  const track = (row, x0, x1) => {
    const lst = (tracks[row] ||= []);
    const [lo, hi] = [Math.min(x0, x1) - 6, Math.max(x0, x1) + 6];
    let i = 0;
    while (lst.some((t) => t.i === i && t.lo < hi && t.hi > lo)) i++;
    lst.push({ lo, hi, i });
    return i;
  };
  const trackY = (row, i) => channel(row) + (i % 2 ? 1 : -1) * Math.ceil(i / 2) * GEOM.TRACK;
  const used = [];
  const out = [];
  for (const e of list) {
    const A = ports[e.from], B = ports[e.to];
    if (!A || !B) throw new LayoutError(`edge ${e.from} → ${e.to} names a port the trace does not declare`);
    const a = nodes[A.node], b = nodes[B.node];
    let pts;
    if (b.row === a.row + 1) {                     // adjacent rows: down, one jog, down
      const y = trackY(a.row, track(a.row, A.x, B.x));
      pts = [[A.x, A.y], [A.x, y], [B.x, y], [B.x, B.y]];
    } else if (b.row > a.row + 1) {                // skips rows: down, jog to a clear corridor, down, jog, down
      const y1 = trackY(a.row, 0), y2 = trackY(b.row - 1, 0);
      const x = corridor(A.x, B.x, obstacles([a.row + 1, b.row - 1], nodes, frames, a, b), used, [y1, y2]);
      const t1 = trackY(a.row, track(a.row, A.x, x)), t2 = trackY(b.row - 1, track(b.row - 1, x, B.x));
      pts = [[A.x, A.y], [A.x, t1], [x, t1], [x, t2], [B.x, t2], [B.x, B.y]];
    } else {                                       // same row or upwards: around, through a clear corridor
      const lo = Math.min(a.row, b.row), hi = Math.max(a.row, b.row);
      const x = corridor(A.x, B.x, obstacles([lo, hi], nodes, frames, a, b, true), used, [b.row * pitch - GEOM.GAP_Y / 2, channel(a.row)]);
      const t1 = trackY(a.row, track(a.row, A.x, x)), t2 = trackY(b.row - 1, track(b.row - 1, x, B.x));
      pts = [[A.x, A.y], [A.x, t1], [x, t1], [x, t2], [B.x, t2], [B.x, B.y]];
    }
    pts = pts.filter((p, i) => i === 0 || p[0] !== pts[i - 1][0] || p[1] !== pts[i - 1][1]);
    out.push({ ...e, id: `${e.from}->${e.to}`, points: pts, d: roundedPath(pts) });
  }
  return out;
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

// Edge labels (10px) beside a vertical run of their edge, only where they collide with nothing: no block, port label,
// other label or other edge's line. An edge whose label finds no clean spot keeps it in its hover title and the drawer
// (labelAt = null). Deterministic: the same trace always gives the same placement.
export const LABEL = { CHAR_W: 5.1, H: 12, GAP: 6, MIN_RUN: 56 };
const hits = (a, b) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;

function placeLabels(edges, nodes, ports, frames) {
  const boxes = Object.values(nodes).map((n) => ({ x0: n.x - 4, x1: n.x + n.w + 4, y0: n.y - 4, y1: n.y + n.h + 4 }));
  for (const f of frames) {  // the frame caption (both reading levels: the longer title)
    const len = Math.max((f.block.title_mechanic || "").length, (f.block.title_developer || "").length);
    boxes.push({ x0: f.x + 6, x1: f.x + 14 + len * 6.6 + 18, y0: f.y - 13, y1: f.y + 13 });
  }
  for (const p of Object.values(ports)) {
    const w = (p.short || "").length * 6.4 + 4, y = p.side === "out" ? p.y + 4 : p.y - 14;
    boxes.push({ x0: p.x + 3, x1: p.x + 5 + w, y0: y, y1: y + 12 });
  }
  const segs = edges.map((e) => e.points.slice(1).map((q, i) => {
    const [x0, y0] = e.points[i], [x1, y1] = q;
    return { x0: Math.min(x0, x1) - 2, x1: Math.max(x0, x1) + 2, y0: Math.min(y0, y1) - 2, y1: Math.max(y0, y1) + 2 };
  }));
  const placed = [];
  edges.forEach((e, ei) => {
    e.labelAt = null;
    if (!e.label) return;
    const w = e.label.length * LABEL.CHAR_W;
    const runs = e.points.slice(1).map((q, i) => [e.points[i], q]).filter(([a, b]) => a[0] === b[0] && Math.abs(b[1] - a[1]) >= LABEL.MIN_RUN)
      .sort((r, t) => Math.abs(t[1][1] - t[0][1]) - Math.abs(r[1][1] - r[0][1]));
    for (const [a, b] of runs) {
      for (const f of [0.5, 0.3, 0.7]) {
        const y = a[1] + (b[1] - a[1]) * f;
        for (const side of [1, -1]) {
          const x = side > 0 ? a[0] + LABEL.GAP : a[0] - LABEL.GAP - w;
          const box = { x0: x - 2, x1: x + w + 2, y0: y - LABEL.H / 2 - 1, y1: y + LABEL.H / 2 + 1 };
          const clash = boxes.some((o) => hits(box, o)) || placed.some((o) => hits(box, o))
            || segs.some((ss, j) => j !== ei && ss.some((o) => hits(box, o)));
          if (!clash) {
            e.labelAt = { x: side > 0 ? x : x + w, y: y + 3.5, anchor: side > 0 ? "start" : "end" };
            placed.push(box);
            return;
          }
        }
      }
    }
  });
}
