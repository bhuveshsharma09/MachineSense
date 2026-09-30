// Pipeline canvas layout (App-4.2, rebalanced in App-4.2.1): the single source of truth for block positions and edge
// routes. Everything is derived from the trace's blocks, ports and edges; PLACEMENT only says where a block id sits. A
// block with no placement is drawn in an "unplaced" lane at the bottom (with a console warning), never dropped.
//
// Shape (App-4.2.1): seven rows, wide rather than deep, so the whole graph fits a 16:9 screen at a legible scale.
// Channel heights between rows follow the number of edge tracks each one carries instead of a fixed gap.

export const GEOM = {
  BH: 72,            // block height: 16px 22px padding, a title line and a subtitle line
  MIN_W: 190, MAX_W: 460, PAD_X: 22,
  TITLE_FIT: 1.08,   // estimated title width × this ≥ the measured width of every title (Outfit 16px; App-4.2.1)
  GAP_X: 56,         // minimum horizontal space between neighbouring blocks in a row
  OUT_BAND: 18,      // below a row: room for the output-port labels before the first edge track
  IN_BAND: 16,       // above a row: room for the input-port labels after the last edge track
  CH_MIN: 40,        // smallest channel between two rows
  FRAME_PAD: 20, FRAME_TOP: 20, FRAME_BOTTOM: 18,
  TRACK: 6,          // spacing of parallel horizontal jogs in one channel
  LANE: 6,           // spacing of parallel vertical runs in one corridor
  CLEAR: 16,         // clearance of a vertical run from a block
  RADIUS: 8,         // rounded corners of edge routes
  MARGIN: 12,        // around the whole graph
};

// Where each block sits: row (top to bottom, in trace order) and x, the block's centre in px (0 = the spine). Rows are
// resolved left to right with at least GAP_X between blocks, so a wider title never makes two blocks overlap.
export const PLACEMENT = {
  input: { row: 0, x: 0 },
  validation: { row: 1, x: 0 },
  signal_quality: { row: 2, x: -400 },        // beside preparation: its output goes to the evidence record only
  preparation: { row: 2, x: 0 },
  order_spectrogram: { row: 3, x: -300 },
  physics_features: { row: 3, x: 280 },
  cnn6: { row: 4, x: -559 },
  ae6: { row: 4, x: -226 },
  b1s: { row: 4, x: 160 },
  lr: { row: 4, x: 527 },                     // off the spine, to the side: it does not feed the decision
  class6: { row: 5, x: -600 },                // second tier of the models group: both read CNN-6's outputs
  mahalanobis: { row: 5, x: -230 },
  decision: { row: 6, x: -330 },
  evidence_record: { row: 6, x: 20 },        // the record, explanation and views run along the bottom row
  explanation: { row: 6, x: 350 },
  presentation: { row: 6, x: 600 },
};

const title = (b, level) => (level === "developer" ? b.title_developer : b.title_mechanic) || b.id;

// Width of a title in Outfit 16px, by character class (calibrated on the measured titles; see GEOM.TITLE_FIT)
const charW = (ch) => (/[ilj.,:;'!|()[\] ]/.test(ch) ? 4.4 : /[ftrI\-_]/.test(ch) ? 5.8 : /[mwMW]/.test(ch) ? 12.2
  : /[A-Z]/.test(ch) ? 9.9 : /[0-9]/.test(ch) ? 8.6 : 8.3);
export const titleWidth = (s) => [...(s || "")].reduce((a, c) => a + charW(c), 0) * GEOM.TITLE_FIT;

function blockWidth(b) {
  const text = Math.max(titleWidth(b.title_mechanic), titleWidth(b.title_developer));
  const ports = Math.max(b.ports_in?.length || 0, b.ports_out?.length || 0);
  return Math.round(Math.min(GEOM.MAX_W, Math.max(GEOM.MIN_W, text + 2 * GEOM.PAD_X + 4, (ports + 1) * 46)));
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
  const placed = leaves.filter((x) => PLACEMENT[x.block.id]);
  const unplaced = leaves.filter((x) => !PLACEMENT[x.block.id]);
  const lastRow = Math.max(0, ...placed.map((x) => PLACEMENT[x.block.id].row));
  if (unplaced.length) warn(`pipeline layout: no placement for ${unplaced.map((x) => x.block.id).join(", ")}; drawn in the unplaced lane`);

  // 1. x: blocks per row, resolved left to right so neighbours keep GAP_X
  const nodes = {};
  const put = (x, row, cx) => {
    const w = blockWidth(x.block);
    nodes[x.block.id] = { id: x.block.id, block: x.block, parent: x.parent, row, w, h: GEOM.BH, x: cx - w / 2 };
  };
  for (const x of placed) put(x, PLACEMENT[x.block.id].row, PLACEMENT[x.block.id].x);
  const unplacedRow = unplaced.length ? lastRow + 1 : null;
  let ux = 0;
  for (const x of unplaced) { const w = blockWidth(x.block); put(x, unplacedRow, ux + w / 2); ux += w + GEOM.GAP_X; }
  const nRows = (unplacedRow ?? lastRow) + 1;
  for (let r = 0; r < nRows; r++) {
    const row = Object.values(nodes).filter((n) => n.row === r).sort((a, b) => a.x - b.x);
    for (let i = 1; i < row.length; i++) {
      const min = row[i - 1].x + row[i - 1].w + GEOM.GAP_X;
      if (row[i].x < min) row[i].x = min;
    }
  }

  // group frames (x now; y after the rows are placed)
  const frames = [];
  for (const g of groups) {
    const kids = g.block.children.map((c) => nodes[c.id]).filter(Boolean);
    if (!kids.length) continue;
    const x0 = Math.min(...kids.map((k) => k.x)) - GEOM.FRAME_PAD, x1 = Math.max(...kids.map((k) => k.x + k.w)) + GEOM.FRAME_PAD;
    const rows = [...new Set(kids.map((k) => k.row))];
    frames.push({ id: g.block.id, block: g.block, x: x0, w: x1 - x0, rows, r0: Math.min(...rows), r1: Math.max(...rows),
      members: new Set(kids.map((k) => k.id)) });
  }

  // 2. ports (x): inputs evenly on the top edge, outputs on the bottom edge; inputs ordered by where their sources sit
  const ports = {};
  for (const n of Object.values(nodes)) {
    const ins = n.block.ports_in || [], outs = n.block.ports_out || [];
    ins.forEach((p, i) => { ports[`${n.id}.${p.id}`] = { ...p, node: n.id, side: "in", i, n: ins.length }; });
    outs.forEach((p, i) => { ports[`${n.id}.${p.id}`] = { ...p, node: n.id, side: "out", i, n: outs.length }; });
  }
  const setX = (key) => { const p = ports[key], n = nodes[p.node]; p.x = n.x + (n.w * (p.i + 1)) / (p.n + 1); };
  Object.keys(ports).forEach(setX);
  for (const n of Object.values(nodes)) {
    const ins = n.block.ports_in || [];
    if (ins.length < 2) continue;
    const at = (p) => {
      const src = trace.edges.filter((e) => e.to === `${n.id}.${p.id}`).map((e) => ports[e.from]?.x).filter((x) => x !== undefined);
      return src.length ? src.reduce((a, b) => a + b, 0) / src.length : ports[`${n.id}.${p.id}`].x;
    };
    ins.map((p, i) => ({ p, i, x: at(p) })).sort((a, b) => a.x - b.x || a.i - b.i)
      .forEach(({ p }, i) => { const q = ports[`${n.id}.${p.id}`]; q.i = i; setX(`${n.id}.${p.id}`); });
  }

  // 3. routes in row / channel / track terms (independent of the rows' heights)
  const routes = planRoutes(trace.edges, nodes, frames, ports);

  // 4. rows' y from the channels' needs
  const tracksIn = {};
  for (const r of routes) for (const h of r.jogs) tracksIn[h.ch] = Math.max(tracksIn[h.ch] || 0, h.track + 1);
  const frameEnds = new Set(frames.map((f) => f.r1)), frameStarts = new Set(frames.map((f) => f.r0));
  const outBand = (k) => GEOM.OUT_BAND + (frameEnds.has(k) ? GEOM.FRAME_BOTTOM : 0);
  const inBand = (k) => GEOM.IN_BAND + (frameStarts.has(k + 1) ? GEOM.FRAME_TOP + 4 : 0);
  const rowY = [0];
  const channelH = {};
  for (let k = 0; k < nRows; k++) {
    channelH[k] = tracksIn[k] ? Math.max(GEOM.CH_MIN, outBand(k) + tracksIn[k] * GEOM.TRACK + inBand(k)) : (k < nRows - 1 ? GEOM.CH_MIN : 0);
    rowY[k + 1] = rowY[k] + GEOM.BH + channelH[k];
  }
  for (const n of Object.values(nodes)) n.y = rowY[n.row];
  for (const p of Object.values(ports)) { const n = nodes[p.node]; p.y = p.side === "in" ? n.y : n.y + n.h; }
  for (const f of frames) { f.y = rowY[f.r0] - GEOM.FRAME_TOP; f.h = rowY[f.r1] + GEOM.BH + GEOM.FRAME_BOTTOM - f.y; }
  const trackY = (ch, i) => rowY[ch] + GEOM.BH + outBand(ch) + (i + 0.5) * GEOM.TRACK;

  // 5. points and paths
  const edges = routes.map((r) => {
    const A = ports[r.from], B = ports[r.to];
    let pts;
    if (r.kind === "adjacent") { const t = trackY(r.jogs[0].ch, r.jogs[0].track); pts = [[A.x, A.y], [A.x, t], [B.x, t], [B.x, B.y]]; }
    else {
      const [j1, j2] = r.jogs;
      const t1 = trackY(j1.ch, j1.track), t2 = trackY(j2.ch, j2.track);
      pts = [[A.x, A.y], [A.x, t1], [r.x, t1], [r.x, t2], [B.x, t2], [B.x, B.y]];
    }
    pts = pts.filter((p, i) => i === 0 || p[0] !== pts[i - 1][0] || p[1] !== pts[i - 1][1]);
    return { ...r.edge, id: `${r.from}->${r.to}`, points: pts, d: roundedPath(pts) };
  });
  // frame captions sit on the frame's top edge, at the first stretch clear of input-port labels and entering lines
  for (const f of frames) {
    const len = Math.max((f.block.title_mechanic || "").length, (f.block.title_developer || "").length);
    const w = len * 6.6 + 18;
    const busy = Object.values(ports).filter((p) => p.side === "in" && f.members.has(p.node) && nodes[p.node].row === f.r0)
      .map((p) => [p.x - 3, p.x + 7 + (p.short || "").length * 6.4]);
    for (const e of edges) for (let i = 1; i < e.points.length; i++) {
      const [x0, y0] = e.points[i - 1], [x1, y1] = e.points[i];
      if (x0 === x1 && Math.min(y0, y1) < f.y && Math.max(y0, y1) > f.y) busy.push([x0 - 3, x0 + 3]);
    }
    busy.sort((a, b) => a[0] - b[0]);
    let x = f.x + 10;
    for (const [a, b] of busy) if (x + w > a && x < b) x = b + 6;
    f.captionX = x + w <= f.x + f.w - 10 ? x : f.x + 10;
    f.captionW = w;
  }
  const colourKind = {};
  for (const e of edges) { colourKind[e.from] ??= e.kind; colourKind[e.to] ??= e.kind; }
  for (const [k, p] of Object.entries(ports)) p.kind = colourKind[k] || null;  // connected ports take their edge's colour

  const xs = [...Object.values(nodes).flatMap((n) => [n.x, n.x + n.w]), ...frames.flatMap((f) => [f.x, f.x + f.w]),
    ...edges.flatMap((e) => e.points.map((p) => p[0]))];
  const ys = [...Object.values(nodes).flatMap((n) => [n.y, n.y + n.h + 16]), ...frames.flatMap((f) => [f.y - 12, f.y + f.h]),
    ...edges.flatMap((e) => e.points.map((p) => p[1]))];
  const bounds = { x: Math.min(...xs) - GEOM.MARGIN, y: Math.min(...ys) - GEOM.MARGIN };
  bounds.w = Math.max(...xs) + GEOM.MARGIN - bounds.x;
  bounds.h = Math.max(...ys) + GEOM.MARGIN - bounds.y;

  const layout = { nodes, frames, ports, edges, bounds, order: all.map((x) => x.block.id), unplaced: unplaced.map((x) => x.block.id),
    unplacedY: unplacedRow === null ? null : rowY[unplacedRow], title };
  layout.labels = (visible) => placeLabels(layout, visible);
  return layout;
}

// ------------------------------------------------------------------------------------------------------------ routes
// obstacles of the rows [r0, r1] for a vertical run: blocks and the group frames the edge is not inside. A downward run
// starts below its source and ends above its target, so those two are skipped; an upward run passes their rows (withEnds).
function obstacles(rows, nodes, frames, a, b, withEnds = false) {
  const out = [];
  for (const n of Object.values(nodes)) {
    if ((!withEnds && (n.id === a.id || n.id === b.id)) || n.row < rows[0] || n.row > rows[1]) continue;
    out.push([n.x - GEOM.CLEAR, n.x + n.w + GEOM.CLEAR]);
  }
  for (const f of frames) {
    if (f.members.has(a.id) || f.members.has(b.id)) continue;
    if (f.rows.some((r) => r >= rows[0] && r <= rows[1])) out.push([f.x - 10, f.x + f.w + 10]);
  }
  out.sort((p, q) => p[0] - q[0]);
  const merged = [];
  for (const o of out) {
    if (merged.length && o[0] <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], o[1]);
    else merged.push([...o]);
  }
  return merged;
}

function corridor(ax, bx, blocked, used, span) {
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
  let x = best.x;  // keep parallel vertical runs apart
  const clash = (v) => used.some((u) => Math.abs(u.x - v) < GEOM.LANE - 0.5 && u.s0 < span[1] && u.s1 > span[0]);
  for (let k = 1; clash(x) && k < 40; k++) {
    const cand = best.x + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * GEOM.LANE;
    if (cand > best.s && cand < best.e) x = cand;
  }
  used.push({ x, s0: span[0], s1: span[1] });
  return x;
}

// Each jog is a horizontal segment in a channel between rows; its two ends continue vertically, above ("up") or below
// ("down") the track. Tracks are assigned per channel after every route is known (orderTracks), so that the order of
// overlapping jogs minimises crossings.
function planRoutes(list, nodes, frames, ports) {
  const used = [];
  const routes = list.map((e) => {
    const A = ports[e.from], B = ports[e.to];
    if (!A || !B) throw new LayoutError(`edge ${e.from} → ${e.to} names a port the trace does not declare`);
    const a = nodes[A.node], b = nodes[B.node];
    if (b.row === a.row + 1) {                      // adjacent rows: down, one jog, down
      return { edge: e, from: e.from, to: e.to, kind: "adjacent", jogs: [{ ch: a.row, x0: A.x, d0: "up", x1: B.x, d1: "down" }] };
    }
    let x, upward = false;
    if (b.row > a.row + 1) {                        // skips rows: down, jog to a clear corridor, down, jog, down
      x = corridor(A.x, B.x, obstacles([a.row + 1, b.row - 1], nodes, frames, a, b), used, [a.row + 0.5, b.row - 0.5]);
    } else {                                        // same row or upwards: around, through a clear corridor
      const lo = Math.min(a.row, b.row), hi = Math.max(a.row, b.row);
      x = corridor(A.x, B.x, obstacles([lo, hi], nodes, frames, a, b, true), used, [b.row - 0.5, a.row + 0.5]);
      upward = true;
    }
    return { edge: e, from: e.from, to: e.to, kind: "corridor", x,
      jogs: [{ ch: a.row, x0: A.x, d0: "up", x1: x, d1: upward ? "up" : "down" },
             { ch: b.row - 1, x0: x, d0: upward ? "down" : "up", x1: B.x, d1: "down" }] };
  });
  const byCh = {};
  for (const r of routes) for (const j of r.jogs) (byCh[j.ch] ||= []).push(j);
  for (const js of Object.values(byCh)) orderTracks(js);
  return routes;
}

const span = (j) => [Math.min(j.x0, j.x1) - 5, Math.max(j.x0, j.x1) + 5];
const overlaps = (j, k) => { const [a0, a1] = span(j), [b0, b1] = span(k); return a0 < b1 && b0 < a1; };
const inside = (x, j) => { const [a0, a1] = span(j); return x > a0 + 5.5 && x < a1 - 5.5; };
// crossings if jog j runs above jog k: k's upward ends pass j's line, j's downward ends pass k's line
const costAbove = (j, k) => [[k.x0, k.d0], [k.x1, k.d1]].filter(([x, d]) => d === "up" && inside(x, j)).length
  + [[j.x0, j.d0], [j.x1, j.d1]].filter(([x, d]) => d === "down" && inside(x, k)).length;

function orderTracks(js) {
  const total = (order) => {
    let c = 0;
    for (let i = 0; i < order.length; i++) for (let k = i + 1; k < order.length; k++) if (overlaps(order[i], order[k])) c += costAbove(order[i], order[k]);
    return c;
  };
  let order = [...js];
  let best = total(order);
  for (let pass = 0; pass < 60; pass++) {         // adjacent swaps and single moves while the count improves
    let improved = false;
    for (let i = 0; i < order.length; i++) for (let k = 0; k < order.length; k++) {
      if (i === k) continue;
      const cand = [...order];
      cand.splice(k, 0, cand.splice(i, 1)[0]);
      const c = total(cand);
      if (c < best) { best = c; order = cand; improved = true; }
    }
    if (!improved) break;
  }
  // stack in that order: a jog goes one track below every earlier jog it overlaps (non-overlapping jogs share tracks)
  order.forEach((j, i) => { j.track = Math.max(-1, ...order.slice(0, i).filter((k) => overlaps(k, j)).map((k) => k.track)) + 1; });
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
// Edge labels (10px) beside a vertical run of their edge, only where they collide with nothing: no block, port label,
// frame caption, other shown label or other edge's line. `visible` is the set of edge ids whose labels are shown (App-4.2.1:
// the hovered or selected block's edges, or all); returns {edge id: {x, y, anchor}} for those that found a clean spot.
export const LABEL = { CHAR_W: 5.1, H: 12, GAP: 6, MIN_RUN: 30 };
const hits = (a, b) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;

function placeLabels(layout, visible) {
  const { nodes, ports, frames, edges } = layout;
  const boxes = Object.values(nodes).map((n) => ({ x0: n.x - 4, x1: n.x + n.w + 4, y0: n.y - 4, y1: n.y + n.h + 4 }));
  for (const f of frames) boxes.push({ x0: f.captionX - 4, x1: f.captionX + f.captionW + 4, y0: f.y - 13, y1: f.y + 13 });
  for (const p of Object.values(ports)) {
    const w = (p.short || "").length * 6.4 + 4, y = p.side === "out" ? p.y + 4 : p.y - 14;
    boxes.push({ x0: p.x + 3, x1: p.x + 5 + w, y0: y, y1: y + 12 });
  }
  const segs = edges.map((e) => e.points.slice(1).map((q, i) => {
    const [x0, y0] = e.points[i], [x1, y1] = q;
    return { x0: Math.min(x0, x1) - 2, x1: Math.max(x0, x1) + 2, y0: Math.min(y0, y1) - 2, y1: Math.max(y0, y1) + 2 };
  }));
  const placed = [];
  const out = {};
  edges.forEach((e, ei) => {
    if (!e.label || !visible.has(e.id)) return;
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
            out[e.id] = { x: side > 0 ? x : x + w, y: y + 3.5, anchor: side > 0 ? "start" : "end" };
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
