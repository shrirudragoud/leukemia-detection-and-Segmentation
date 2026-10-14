// IR -> positioned scene.
//
// Source code is structured, so its control flow is series-parallel. That lets us lay it out with a
// deterministic recursive box model instead of a general graph-layout engine:
//
//   Block = { nodes, edges, regions, L, R, H, entry, exits }
//     - every block is laid out around a vertical axis x = 0 (L/R = extent left/right of it), top y = 0
//     - `entry` is the node that receives the incoming edge, `exits` are dangling ends that the next
//       sibling (or the enclosing construct) connects to
//   sequence  -> stack blocks vertically on the same axis
//   if/else   -> question on the axis, `yes` column left, `no` column right, a free corridor between
//   loop      -> question + body below it inside a tinted region; back-edge climbs a right-hand lane
//   switch    -> question with one column per case
//
// Edges are logical here (from/to ids + kind); route.js turns them into orthogonal polylines once
// absolute coordinates are known.
//
// `mode: 'kids'` writes plain-English text on the blocks and shows the real code underneath;
// `mode: 'code'` writes the technical text only.

import { kindMeta, CATEGORY_EMOJI } from '../core/kinds.js';
import { EXPLAIN } from '../profiles/plain.js';
import { wrap, truncate } from '../core/text.js';
import { routeEdges } from './route.js';

export const G = {
  CARD_W: 300,
  GAP_Y: 46,
  GAP_X: 44,
  BRANCH_GAP: 74,
  LANE_M: 28,
  REGION_PAD: 20,
  PANEL_PAD: 44,
  PANEL_GAP: 70,
  PANEL_HEADER: 92,
  PANEL_MIN_W: 360,
};

let uid = 0;
let opts = { mode: 'kids', showCode: true };
const nid = () => `n${++uid}`;
const kids = () => opts.mode === 'kids';

// ---------------------------------------------------------------------------------------------
// node factories (positions are relative until the lane is assembled; x = centre, y = top)
// ---------------------------------------------------------------------------------------------

const oneLine = (t) => truncate(String(t || '').replace(/;\s*$/, ''), 120);
const explainFor = (kind, category) => EXPLAIN[`${kind}:${category}`] || EXPLAIN[kind] || '';

function cardNode(s, width = G.CARD_W) {
  const meta = kindMeta(s.kind);
  const category = s.category || meta.label;
  const title = kids() ? s.friendly || s.title : s.title;
  const sub = kids() ? (s.sub && s.sub.startsWith('#') ? s.sub : '') : s.sub;
  const textW = width - 68 - 14;
  const chars = Math.floor(textW / 8.3);
  const monoChars = Math.floor((textW - 16) / 6.9);
  const titleLines = wrap(title, chars, 3);
  const subLines = sub ? wrap(sub, Math.floor(textW / 6.6), 1) : [];
  const code = kids() && opts.showCode && s.text && !s.lines ? truncate(oneLine(s.text), monoChars) : null;
  const raw = (kids() ? s.friendlyLines : null) || s.lines || [];
  const lines = raw.length > 6 ? [...raw.slice(0, 6).map((l) => wrap(l, chars + 4, 1)[0]), `+ ${raw.length - 6} more`] : raw.map((l) => wrap(l, chars + 4, 1)[0]);
  const h = Math.max(66, 12 + 13 + 6 + titleLines.length * 19 + (subLines.length ? 17 : 0) + (code ? 32 : 0) + (lines.length ? lines.length * 18 + 6 : 0) + 14);
  return {
    id: nid(), kind: s.kind, shape: 'card', x: 0, y: 0, w: width, h,
    category: String(category).toUpperCase(), emoji: CATEGORY_EMOJI[category] || meta.emoji,
    titleLines, subLines, code, lines, explain: s.explain || explainFor(s.kind, category),
    loc: s.loc, warn: s.warn, error: s.error, ref: s.ref || null, text: s.text, labelW: 0,
  };
}

function hexNode(kind, title, sub, loc, text) {
  const meta = kindMeta(kind);
  const titleLines = wrap(title, 24, 3);
  const subLines = sub ? [truncate(sub, 34)] : [];
  const h = Math.max(70, 26 + titleLines.length * 19 + (subLines.length ? 17 : 0) + 18);
  return {
    id: nid(), kind, shape: 'hex', x: 0, y: 0, w: 300, h, emoji: meta.emoji, titleLines, subLines, loc, text,
    category: meta.label.toUpperCase(), explain: explainFor(kind), labelW: 0, lines: [], code: null,
  };
}

function pillNode(kind, title, loc) {
  const meta = kindMeta(kind);
  return {
    id: nid(), kind, shape: 'pill', x: 0, y: 0, w: Math.min(260, Math.max(150, title.length * 8.4 + 76)), h: 44, emoji: meta.emoji,
    titleLines: [truncate(title, 28)], subLines: [], lines: [], code: null, category: meta.label.toUpperCase(), explain: explainFor(kind), loc, labelW: 0,
  };
}

function hatNode(title, emoji) {
  const lines = wrap(title, 24, 2);
  return {
    id: nid(), kind: 'start', shape: 'hat', x: 0, y: 0, w: 280, h: 28 + lines.length * 19 + 14, emoji, titleLines: lines, subLines: [], lines: [], code: null,
    category: 'START', explain: EXPLAIN.start, labelW: 0,
  };
}

// ---------------------------------------------------------------------------------------------
// block algebra
// ---------------------------------------------------------------------------------------------

const emptyBlock = () => ({ nodes: [], edges: [], regions: [], L: 0, R: 0, H: 0, entry: null, exits: [], empty: true });

function single(node) {
  return { nodes: [node], edges: [], regions: [], empty: false, entry: node.id, exits: [{ id: node.id }], H: node.h, L: node.w / 2, R: node.w / 2 };
}

const terminal = (node) => ({ ...single(node), exits: [] });

function shift(b, dx, dy) {
  for (const n of b.nodes) { n.x += dx; n.y += dy; }
  for (const r of b.regions) { r.x += dx; r.y += dy; }
  for (const e of b.edges) {
    if (e.laneX != null) e.laneX += dx;
    if (e.srcLaneX != null) e.srcLaneX += dx;
    if (e.busY != null) e.busY += dy;
  }
  for (const x of b.exits) if (x.laneX != null) x.laneX += dx;
}

const tone = (label) => (label === 'yes' ? 'yes' : label === 'no' ? 'no' : null);

const connect = (exits, to, edges) => {
  for (const x of exits) edges.push({ from: x.id, fromPort: x.port || 'bottom', laneX: x.laneX, label: x.label || '', tone: tone(x.label), to, kind: 'flow' });
};

/** Stack blocks vertically; stops at the first block that cannot fall through (return/break/...). */
function stack(blocks) {
  const out = emptyBlock();
  for (const b of blocks) {
    if (b.empty) continue;
    if (!out.empty && out.exits.length === 0) break; // unreachable
    const y = out.empty ? 0 : out.H + G.GAP_Y;
    if (!out.empty) connect(out.exits, b.entry, out.edges);
    shift(b, 0, y);
    out.nodes.push(...b.nodes);
    out.edges.push(...b.edges);
    out.regions.push(...b.regions);
    out.L = Math.max(out.L, b.L);
    out.R = Math.max(out.R, b.R);
    out.H = y + b.H;
    if (out.empty) out.entry = b.entry;
    out.exits = b.exits;
    out.empty = false;
  }
  return out;
}

function seq(list, ctx) {
  // In kids mode a recognised counting loop hides its `i = 0` / `i++` plumbing: "Repeat 3 times" says it all.
  const visible = kids() ? list.filter((s, i) => !(s.role === 'loop-init' && list[i + 1]?.t === 'loop' && list[i + 1].count)) : list;
  return stack(visible.map((s) => stmt(s, ctx)));
}

const branchEdge = (from, to, label) => ({ from, to, fromPort: 'bottom', label, tone: tone(label), kind: 'branch' });

function place(out, b, dx, dy) {
  shift(b, dx, dy);
  out.nodes.push(...b.nodes);
  out.edges.push(...b.edges);
  out.regions.push(...b.regions);
}

// ---------------------------------------------------------------------------------------------
// statements
// ---------------------------------------------------------------------------------------------

function stmt(s, ctx) {
  switch (s.t) {
    case 'action': return single(cardNode(s));
    case 'return': return terminal(pillNode('return', kids() ? (s.value ? `Give back ${s.value}` : 'Finish here') : s.value ? `return ${s.value}` : 'return', s.loc));
    case 'break': {
      const n = pillNode('break', kids() ? 'Stop the loop' : 'break', s.loc);
      ctx.breaks?.push({ id: n.id });
      return terminal(n);
    }
    case 'continue': {
      const n = pillNode('continue', kids() ? 'Skip to next round' : 'continue', s.loc);
      ctx.continues?.push({ id: n.id });
      return terminal(n);
    }
    case 'if': return ifBlock(s, ctx);
    case 'loop': return loopBlock(s, ctx);
    case 'switch': return switchBlock(s, ctx);
    default: return emptyBlock();
  }
}

function ifBlock(s, ctx) {
  const D = hexNode('decision', kids() ? s.friendly : s.cond, null, s.headLoc, s.cond);
  const yes = seq(s.then, ctx);
  const no = seq(s.else, ctx);
  const out = { ...emptyBlock(), empty: false, nodes: [D], entry: D.id };
  const half = G.GAP_X / 2;
  const top = D.h + G.BRANCH_GAP;
  let L = D.w / 2;
  let R = D.w / 2;
  let bottom = D.h;
  const exits = [];

  if (!yes.empty) {
    const x = -(half + yes.R);
    place(out, yes, x, top);
    out.edges.push(branchEdge(D.id, yes.entry, 'yes'));
    exits.push(...yes.exits);
    L = Math.max(L, -x + yes.L);
    bottom = Math.max(bottom, top + yes.H);
  } else exits.push({ id: D.id, label: 'yes' });

  if (!no.empty) {
    const x = half + no.L;
    place(out, no, x, top);
    out.edges.push(branchEdge(D.id, no.entry, 'no'));
    exits.push(...no.exits);
    R = Math.max(R, x + no.R);
    bottom = Math.max(bottom, top + no.H);
  } else exits.push({ id: D.id, label: 'no' });

  return Object.assign(out, { L, R, H: bottom, exits });
}

function loopBlock(s, ctx) {
  const forever = s.form === 'for' && s.cond === 'forever';
  const infinite = forever || (s.form !== 'do' && /^(true|1)$/.test(s.cond));
  const sub = kids()
    ? s.friendlySub || (s.form === 'while' && !infinite ? 'checks first, then repeats' : '')
    : { while: 'while loop', for: 'for loop', do: 'do-while', range: 'for-each' }[s.form];
  const D = hexNode('loop', kids() ? s.friendly : forever ? 'forever' : s.cond, sub, s.headLoc, s.cond);
  const bctx = { breaks: [], continues: [] };
  const body = kids() && s.count ? seq(s.body.filter((b) => b.role !== 'loop-step'), bctx) : seq(s.body, bctx);
  const out = { ...emptyBlock(), empty: false, nodes: [D], regions: [] };
  const lane = G.LANE_M;
  const pad = G.REGION_PAD;
  const region = (x1, x2, y1, y2) => ({ kind: 'loop', x: x1, y: y1, w: x2 - x1, h: y2 - y1 });

  if (body.empty) return Object.assign(out, { entry: D.id, L: D.w / 2, R: D.w / 2, H: D.h, exits: [{ id: D.id }] });

  if (s.form === 'do') {
    place(out, body, 0, 0);
    const dy = body.H + G.GAP_Y;
    D.y = dy;
    connect([...body.exits, ...bctx.continues], D.id, out.edges);
    const laneR = Math.max(body.R, D.w / 2) + lane;
    const left = Math.max(body.L, D.w / 2);
    out.edges.push({ from: D.id, fromPort: 'right', to: body.entry, kind: 'back', laneX: laneR, label: 'again', tone: null });
    out.regions.unshift(region(-left - pad, laneR + pad, -16, dy + D.h + 26));
    return Object.assign(out, {
      entry: body.entry, L: left + pad + 2, R: laneR + pad + 2, H: dy + D.h,
      exits: [{ id: D.id, label: 'no' }, ...bctx.breaks],
    });
  }

  const bodyTop = D.h + 60;
  place(out, body, 0, bodyTop);
  out.edges.push(branchEdge(D.id, body.entry, 'yes'));
  const laneR = Math.max(body.R, D.w / 2) + lane;
  const laneL = -(Math.max(body.L, D.w / 2) + lane);
  const H = bodyTop + body.H;
  for (const src of [...body.exits, ...bctx.continues]) {
    out.edges.push({ from: src.id, fromPort: src.port || 'bottom', srcLaneX: src.laneX, to: D.id, kind: 'back', laneX: laneR, busY: H + 12, label: '', tone: null });
  }
  out.regions.unshift(region(laneL - pad, laneR + pad, -16, H + 26));
  return Object.assign(out, {
    entry: D.id, L: -laneL + pad + 2, R: laneR + pad + 2, H,
    exits: [...(infinite ? [] : [{ id: D.id, port: 'left', laneX: laneL, label: 'no' }]), ...bctx.breaks],
  });
}

function switchBlock(s, ctx) {
  const D = hexNode('switch', kids() ? s.friendly : s.expr, kids() ? '' : 'switch', s.headLoc, s.expr);
  const sctx = { breaks: [], continues: ctx.continues };
  const cols = s.cases.map((c) => ({ c, b: seq(c.body, sctx) }));
  const filled = cols.filter((c) => !c.b.empty);
  const caseLabel = (c) => truncate(kids() ? c.friendlyLabel : c.labels.join(', '), 22);
  const straight = [];
  if (!s.cases.some((c) => c.isDefault)) straight.push(kids() ? 'anything else' : 'other');
  for (const c of cols) if (c.b.empty) straight.push(caseLabel(c.c));

  const out = { ...emptyBlock(), empty: false, nodes: [D], entry: D.id };
  const half = G.GAP_X / 2;
  const top = D.h + G.BRANCH_GAP;
  const nLeft = Math.ceil(filled.length / 2);
  const left = filled.slice(0, nLeft);
  const right = filled.slice(nLeft);
  let L = D.w / 2;
  let R = D.w / 2;
  let bottom = D.h;
  const exits = [];

  const attach = (col, x) => {
    place(out, col.b, x, top);
    out.edges.push(branchEdge(D.id, col.b.entry, caseLabel(col.c)));
    exits.push(...col.b.exits);
    L = Math.max(L, -(x - col.b.L));
    R = Math.max(R, x + col.b.R);
    bottom = Math.max(bottom, top + col.b.H);
  };
  let cur = -half;
  for (const col of [...left].reverse()) { const x = cur - col.b.R; attach(col, x); cur = x - col.b.L - G.GAP_X; }
  cur = half;
  for (const col of right) { const x = cur + col.b.L; attach(col, x); cur = x + col.b.R + G.GAP_X; }

  if (straight.length) exits.push({ id: D.id, label: truncate(straight.join(', '), 22) });
  exits.push(...sctx.breaks);
  return Object.assign(out, { L, R, H: bottom, exits });
}

// ---------------------------------------------------------------------------------------------
// functions -> lanes -> scene
// ---------------------------------------------------------------------------------------------

function functionBlock(fn) {
  const start = hatNode(fn.startLabel || 'Start', fn.startEmoji || '🏁');
  const body = seq(fn.body, { breaks: null, continues: null });
  let top = stack([single(start), body]);
  if (fn.repeats && !body.empty) {
    const laneR = top.R + G.LANE_M;
    for (const src of top.exits) {
      top.edges.push({ from: src.id, fromPort: src.port || 'bottom', to: body.entry, kind: 'back', laneX: laneR, busY: top.H + 12, label: 'again', tone: null });
    }
    top.exits = [];
    top.R = laneR + 6;
  } else if (top.exits.length) {
    top = stack([top, single(pillNode('end', fn.endTitle || (kids() ? 'The end' : 'End'), fn.loc))]);
  }
  return top;
}

const ROLE_RANK = { setup: 0, loop: 1, isr: 2, callback: 3, user: 4 };
const shortName = (fn) => `${fn.name}(${fn.params.map((p) => p.split(/\s+/).pop().replace(/[*&[\]]/g, '')).filter(Boolean).join(', ')})`;

export function buildScene(program, options = {}) {
  uid = 0;
  opts = { mode: 'kids', showCode: true, ...options };
  const lanes = [];
  const blocks = [];

  if (program.globals.length) {
    const lines = program.globals.map((g) => (kids() ? g.friendly || g.text : g.text));
    const node = cardNode({
      kind: 'globals', category: kids() ? 'What the program knows' : 'Globals', title: kids() ? 'Names & tools' : `${program.globals.length} declarations`,
      lines, friendlyLines: lines, loc: { start: program.globals[0].line, end: program.globals[program.globals.length - 1].line },
    }, 340);
    node.lines = lines.length > 14 ? [...lines.slice(0, 13).map((l) => truncate(l, 40)), `+ ${lines.length - 13} more`] : lines.map((l) => truncate(l, 40));
    node.h = 12 + 13 + 6 + node.titleLines.length * 19 + node.lines.length * 18 + 6 + 14;
    lanes.push({ id: 'globals', name: kids() ? 'Before we start' : 'Globals', description: kids() ? 'Libraries, names for pins, and variables' : 'Includes, defines, variables', role: 'globals' });
    blocks.push(single(node));
  }

  const fns = program.functions
    .map((f, i) => ({ f, i }))
    .sort((a, b) => (ROLE_RANK[a.f.role] ?? 9) - (ROLE_RANK[b.f.role] ?? 9) || a.i - b.i)
    .map((x) => x.f);

  for (const fn of fns) {
    lanes.push({ id: `fn:${fn.name}`, name: kids() ? shortName(fn) : fn.signature, fn: fn.name, description: fn.description, role: fn.role, loc: fn.loc });
    blocks.push(functionBlock(fn));
  }

  const nodes = [];
  const edges = [];
  const regions = [];
  const contentTop = G.PANEL_HEADER + 22;
  let x = 0;
  let maxH = 0;
  blocks.forEach((b, i) => {
    const lane = lanes[i];
    const natural = b.L + b.R + 2 * G.PANEL_PAD;
    const w = Math.max(natural, G.PANEL_MIN_W);
    lane.x = x;
    lane.y = 0;
    lane.w = w;
    const axis = x + (w - natural) / 2 + G.PANEL_PAD + b.L;
    shift(b, axis, contentTop);
    for (const n of b.nodes) n.lane = lane.id;
    nodes.push(...b.nodes);
    edges.push(...b.edges);
    regions.push(...b.regions);
    maxH = Math.max(maxH, contentTop + b.H + G.PANEL_PAD + 20);
    x += w + G.PANEL_GAP;
  });
  for (const l of lanes) l.h = maxH;

  edges.forEach((e, i) => { e.id = `e${i + 1}`; });
  routeEdges(nodes, edges);

  const byFn = new Map(lanes.filter((l) => l.fn).map((l) => [l.fn, l]));
  const links = nodes
    .filter((n) => n.ref && byFn.has(n.ref))
    .map((n) => ({ from: n.id, toLane: byFn.get(n.ref).id, kind: n.kind }));

  return {
    version: 2,
    mode: opts.mode,
    language: program.language,
    lanes,
    nodes,
    edges,
    regions,
    links,
    bounds: { w: Math.max(0, x - G.PANEL_GAP), h: maxH },
  };
}
