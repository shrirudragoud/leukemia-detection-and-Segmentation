// scene -> SVG markup. Pure string building, so it works in the browser, in Node (tests, SSR,
// server-side thumbnails) and for "export as .svg".
//
// Look: Scratch-style chunky blocks (solid colour, 3D lip, emoji badge) on soft pastel lanes.

import { kindMeta } from '../core/kinds.js';

const FONT = "Nunito, 'Baloo 2', ui-rounded, 'SF Pro Rounded', 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif";
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace";
const EMOJI = "'Apple Color Emoji','Segoe UI Emoji','Noto Color Emoji',sans-serif";

const ROLE = {
  setup: { label: 'RUNS ONCE', color: '#1FB58F', tint: '#E8F8F3' },
  loop: { label: 'REPEATS FOREVER', color: '#7C5CFF', tint: '#F1EEFF' },
  isr: { label: 'DOORBELL', color: '#F5B700', tint: '#FFF8E1' },
  callback: { label: 'CALLBACK', color: '#F5B700', tint: '#FFF8E1' },
  user: { label: 'MY BLOCK', color: '#FF6680', tint: '#FFEEF1' },
  globals: { label: 'NAMES', color: '#6B7A99', tint: '#F0F3F8' },
};

const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const n1 = (v) => Math.round(v * 10) / 10;

function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const f = (c) => Math.max(0, Math.min(255, Math.round(c * (1 - amt))));
  return `#${[(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => f(c).toString(16).padStart(2, '0')).join('')}`;
}
const isLight = (ink) => ink !== '#fff';

export const CSS = `
.cv-root{font-family:${FONT}}
.cv-root text{font-family:${FONT};pointer-events:none}
.cv-root .emoji{font-family:${EMOJI}}
.cv-root .node{cursor:pointer;transition:transform .12s ease}
.cv-root .node:hover{transform:translateY(-2px)}
.cv-root .node.sel .blk{stroke:#25324D;stroke-width:3.5}
.cv-root .node.run .blk{stroke:#fff;stroke-width:4}
.cv-root .node.run{filter:drop-shadow(0 0 12px rgba(31,181,143,.95))}
.cv-root .node.err .blk{stroke:#E5483D;stroke-width:3;stroke-dasharray:6 4}
.cv-root .node.dim,.cv-root .edge.dim{opacity:.3}
.cv-root .edge path.line{fill:none;stroke:#98A5BA;stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round}
.cv-root .edge.tone-yes path.line{stroke:#2FBF71}
.cv-root .edge.tone-no path.line{stroke:#FF7070}
.cv-root .edge.back path.line{stroke:#9B7CFF}
.cv-root .edge.hi path.line{stroke-width:4}
.cv-root .edge.run path.line{stroke:#1FB58F;stroke-width:4.5}
.cv-root .lane-panel{fill:#fff;fill-opacity:.78;stroke:#E1E7F0;stroke-width:1.5}
.cv-root .lane.hi .lane-panel{stroke:#1FB58F;stroke-width:3}
.cv-root .t-cat{font-size:10px;font-weight:800;letter-spacing:.09em}
.cv-root .t-title{font-size:15px;font-weight:800}
.cv-root .t-sub{font-size:12px;font-weight:700}
.cv-root .t-line{font-size:13px;font-weight:700}
.cv-root .t-code{font-family:${MONO};font-size:11.5px;font-weight:600}
.cv-root .t-lane{font-size:19px;font-weight:900;fill:#25324D;font-family:${MONO}}
.cv-root .t-lane-sub{font-size:13px;font-weight:700;fill:#7A879C}
.cv-root .t-edge{font-size:11.5px;font-weight:800}
.cv-root .link{fill:none;stroke:#F59E0B;stroke-width:3;stroke-dasharray:3 8;stroke-linecap:round;pointer-events:none}
`;

function roundedPath(pts, r = 12) {
  if (pts.length < 2) return '';
  let d = `M${n1(pts[0][0])} ${n1(pts[0][1])}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1];
    const [cx, cy] = pts[i];
    const [nx, ny] = pts[i + 1];
    const l1 = Math.hypot(cx - px, cy - py);
    const l2 = Math.hypot(nx - cx, ny - cy);
    const rr = Math.min(r, l1 / 2, l2 / 2);
    d += ` L${n1(cx - ((cx - px) / l1) * rr)} ${n1(cy - ((cy - py) / l1) * rr)} Q${n1(cx)} ${n1(cy)} ${n1(cx + ((nx - cx) / l2) * rr)} ${n1(cy + ((ny - cy) / l2) * rr)}`;
  }
  const last = pts[pts.length - 1];
  return d + ` L${n1(last[0])} ${n1(last[1])}`;
}

const trunc = (s, n) => (s.length > n ? s.slice(0, Math.max(1, n - 1)) + '…' : s);

// ---- lanes & regions -------------------------------------------------------------------------

function laneSvg(l) {
  const role = ROLE[l.role] || ROLE.user;
  const cw = role.label.length * 7.4 + 22;
  return `<g class="lane" data-lane="${esc(l.id)}">
<rect class="lane-panel" x="${n1(l.x)}" y="${n1(l.y)}" width="${n1(l.w)}" height="${n1(l.h)}" rx="30"/>
<path d="M${n1(l.x + 34)} ${n1(l.y + 2)}H${n1(l.x + l.w - 34)}" stroke="${role.color}" stroke-width="5" stroke-linecap="round"/>
<text class="t-lane" x="${n1(l.x + 28)}" y="${n1(l.y + 44)}">${esc(trunc(l.name, Math.floor((l.w - 56) / 11.5)))}</text>
<text class="t-lane-sub" x="${n1(l.x + 28)}" y="${n1(l.y + 68)}">${esc(trunc(l.description || '', Math.floor((l.w - 56) / 7.2)))}</text>
<g transform="translate(${n1(l.x + l.w - 24)} ${n1(l.y + 38)})"><rect x="${-cw}" y="-14" width="${cw}" height="24" rx="12" fill="${role.color}" fill-opacity=".16"/><text x="${-cw / 2}" y="3" text-anchor="middle" style="font-size:10.5px;font-weight:900;letter-spacing:.08em" fill="${shade(role.color, 0.25)}">${role.label}</text></g>
</g>`;
}

const regionSvg = (r) =>
  `<rect x="${n1(r.x)}" y="${n1(r.y)}" width="${n1(r.w)}" height="${n1(r.h)}" rx="30" fill="#FF8C1A" fill-opacity=".07" stroke="#FF8C1A" stroke-opacity=".4" stroke-width="2.5" stroke-dasharray="1 9" stroke-linecap="round"/>`;

// ---- edges -----------------------------------------------------------------------------------

function edgeSvg(e) {
  if (!e.points || e.points.length < 2) return '';
  const tone = e.tone || (e.kind === 'back' ? 'back' : 'plain');
  const marker = { yes: 'cv-arrow-yes', no: 'cv-arrow-no', back: 'cv-arrow-back', plain: 'cv-arrow' }[tone];
  let label = '';
  if (e.labelAt && e.label) {
    const text = e.tone === 'yes' ? '✔ yes' : e.tone === 'no' ? '✖ no' : e.label === 'again' ? '🔁 again' : e.label;
    const w = text.length * 7 + 20;
    const style = { yes: ['#E4F8EC', '#16914F'], no: ['#FFE9E9', '#E04848'], back: ['#F0EBFF', '#6D4AE0'], plain: ['#fff', '#4B5A73'] }[e.label === 'again' ? 'back' : tone];
    label = `<rect x="${n1(e.labelAt[0] - w / 2)}" y="${n1(e.labelAt[1] - 11)}" width="${n1(w)}" height="22" rx="11" fill="${style[0]}" stroke="${style[1]}" stroke-opacity=".35"/><text class="t-edge" x="${n1(e.labelAt[0])}" y="${n1(e.labelAt[1] + 4.2)}" text-anchor="middle" fill="${style[1]}">${esc(text)}</text>`;
  }
  return `<g class="edge ${e.kind} tone-${tone}" data-edge="${e.id}" data-from="${e.from}" data-to="${e.to}"><path class="line" d="${roundedPath(e.points)}" marker-end="url(#${marker})"/>${label}</g>`;
}

// ---- nodes -----------------------------------------------------------------------------------

function badge(n, left) {
  if (!n.warn && !n.error) return '';
  return `<g transform="translate(${n1(left + n.w - 6)} ${n1(n.y + 6)})"><circle r="11" fill="#E5483D" stroke="#fff" stroke-width="3"/><text y="4.5" text-anchor="middle" style="font-size:13px;font-weight:900" fill="#fff">${n.error ? '×' : '!'}</text></g>`;
}

function textBlock(n, textX, top, ink) {
  let y = top;
  let out = `<text class="t-cat" x="${n1(textX)}" y="${n1(y)}" fill="${ink}" fill-opacity=".72">${esc(n.category)}</text>`;
  y += 4;
  for (const t of n.titleLines) { y += 19; out += `<text class="t-title" x="${n1(textX)}" y="${n1(y)}" fill="${ink}">${esc(t)}</text>`; }
  for (const t of n.subLines) { y += 17; out += `<text class="t-sub" x="${n1(textX)}" y="${n1(y)}" fill="${ink}" fill-opacity=".85">${esc(t)}</text>`; }
  if (n.lines.length) {
    y += 4;
    for (const t of n.lines) { y += 18; out += `<text class="t-line" x="${n1(textX)}" y="${n1(y)}" fill="${ink}" fill-opacity=".95">• ${esc(t)}</text>`; }
  }
  if (n.code) {
    const w = n.w - 68 - 14;
    y += 10;
    out += `<rect x="${n1(textX)}" y="${n1(y)}" width="${n1(w)}" height="22" rx="8" fill="${isLight(ink) ? '#fff' : '#000'}" fill-opacity="${isLight(ink) ? 0.55 : 0.22}"/>`;
    out += `<text class="t-code" x="${n1(textX + 8)}" y="${n1(y + 15)}" fill="${isLight(ink) ? '#3B2A00' : '#fff'}">${esc(n.code)}</text>`;
  }
  return out;
}

function emojiBubble(cx, cy, r, emoji) {
  return `<circle cx="${n1(cx)}" cy="${n1(cy)}" r="${r}" fill="#fff" fill-opacity=".92"/><text class="emoji" x="${n1(cx)}" y="${n1(cy + r * 0.36)}" text-anchor="middle" style="font-size:${n1(r * 1.1)}px">${emoji}</text>`;
}

function nodeSvg(n) {
  const meta = kindMeta(n.kind);
  const color = meta.color;
  const ink = meta.ink;
  const dark = shade(color, 0.22);
  const cls = ['node', n.error ? 'err' : ''].filter(Boolean).join(' ');
  const open = `<g class="${cls}" data-node="${n.id}" data-kind="${n.kind}"${n.loc ? ` data-start="${n.loc.start}" data-end="${n.loc.end}"` : ''}>`;
  const tip = [n.text && esc(n.text), n.warn && `⚠ ${esc(n.warn)}`].filter(Boolean).join('\n');
  const title = tip ? `<title>${tip}</title>` : '';
  const left = n.x - n.w / 2;
  const cy = n.y + n.h / 2;
  let body = '';

  if (n.shape === 'card') {
    body += `<rect x="${n1(left)}" y="${n1(n.y + 5)}" width="${n.w}" height="${n.h}" rx="20" fill="${dark}" style="filter:url(#cv-shadow)"/>`;
    body += `<rect class="blk" x="${n1(left)}" y="${n1(n.y)}" width="${n.w}" height="${n.h}" rx="20" fill="${color}"/>`;
    body += `<rect x="${n1(left + 14)}" y="${n1(n.y + 3)}" width="${n.w - 28}" height="3" rx="1.5" fill="#fff" fill-opacity=".38"/>`;
    body += emojiBubble(left + 14 + 20, n.h > 78 ? n.y + 14 + 20 : cy, 20, n.emoji);
    body += textBlock(n, left + 68, n.y + 22, ink);
    body += badge(n, left);
  } else if (n.shape === 'hex') {
    const l = left;
    const r = left + n.w;
    const pts = (dy) => `${n1(l)},${n1(cy + dy)} ${n1(l + 28)},${n1(n.y + dy)} ${n1(r - 28)},${n1(n.y + dy)} ${n1(r)},${n1(cy + dy)} ${n1(r - 28)},${n1(n.y + n.h + dy)} ${n1(l + 28)},${n1(n.y + n.h + dy)}`;
    body += `<polygon points="${pts(5)}" fill="${dark}" stroke="${dark}" stroke-width="10" stroke-linejoin="round" style="filter:url(#cv-shadow)"/>`;
    body += `<polygon class="blk" points="${pts(0)}" fill="${color}" stroke="${color}" stroke-width="10" stroke-linejoin="round"/>`;
    body += emojiBubble(l + 52, cy, 19, n.emoji);
    const textX = l + 80;
    const total = n.titleLines.length * 19 + n.subLines.length * 17;
    let y = cy - total / 2 + 14;
    for (const t of n.titleLines) { body += `<text class="t-title" x="${n1(textX)}" y="${n1(y)}" fill="${ink}">${esc(t)}</text>`; y += 19; }
    for (const t of n.subLines) { body += `<text class="t-sub" x="${n1(textX)}" y="${n1(y - 1)}" fill="${ink}" fill-opacity=".8">${esc(t)}</text>`; y += 17; }
  } else if (n.shape === 'hat') {
    body += `<path d="M${n1(left + 18)} ${n1(n.y + 8)}C${n1(left + 34)} ${n1(n.y - 18)} ${n1(left + 100)} ${n1(n.y - 18)} ${n1(left + 126)} ${n1(n.y + 8)}Z" fill="${dark}"/>`;
    body += `<rect x="${n1(left)}" y="${n1(n.y + 5)}" width="${n.w}" height="${n.h}" rx="20" fill="${dark}" style="filter:url(#cv-shadow)"/>`;
    body += `<path class="blk" d="M${n1(left + 18)} ${n1(n.y + 8)}C${n1(left + 34)} ${n1(n.y - 18)} ${n1(left + 100)} ${n1(n.y - 18)} ${n1(left + 126)} ${n1(n.y + 8)}Z" fill="${color}"/>`;
    body += `<rect class="blk" x="${n1(left)}" y="${n1(n.y)}" width="${n.w}" height="${n.h}" rx="20" fill="${color}"/>`;
    body += emojiBubble(left + 14 + 18, cy + 2, 18, n.emoji);
    const total = n.titleLines.length * 19;
    let y = cy - total / 2 + 16;
    for (const t of n.titleLines) { body += `<text class="t-title" x="${n1(left + 66)}" y="${n1(y)}" fill="${ink}">${esc(t)}</text>`; y += 19; }
  } else {
    body += `<rect x="${n1(left)}" y="${n1(n.y + 4)}" width="${n.w}" height="${n.h}" rx="22" fill="${dark}" style="filter:url(#cv-shadow)"/>`;
    body += `<rect class="blk" x="${n1(left)}" y="${n1(n.y)}" width="${n.w}" height="${n.h}" rx="22" fill="${color}"/>`;
    body += `<text class="emoji" x="${n1(left + 20)}" y="${n1(cy + 7)}" style="font-size:20px">${n.emoji}</text>`;
    body += `<text class="t-title" x="${n1(left + 50)}" y="${n1(cy + 5.5)}" fill="${ink}">${esc(n.titleLines[0])}</text>`;
  }
  return open + title + body + '</g>';
}

/** Inner markup (no <svg> wrapper) so a viewer can place it inside its own pan/zoom group. */
export function renderContent(scene) {
  return [
    ...scene.lanes.map(laneSvg),
    `<g class="regions">${(scene.regions || []).map(regionSvg).join('')}</g>`,
    `<g class="edges">${scene.edges.map(edgeSvg).join('')}</g>`,
    `<g class="nodes">${scene.nodes.map(nodeSvg).join('')}</g>`,
    '<g class="links"></g>',
  ].join('\n');
}

const arrow = (id, color) => `<marker id="${id}" viewBox="0 0 10 10" refX="7.6" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M1 1L9 5L1 9z" fill="${color}" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/></marker>`;

export const DEFS = `<defs>
<filter id="cv-shadow" x="-10%" y="-20%" width="120%" height="160%"><feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#25324d" flood-opacity=".16"/></filter>
${arrow('cv-arrow', '#98A5BA')}${arrow('cv-arrow-yes', '#2FBF71')}${arrow('cv-arrow-no', '#FF7070')}${arrow('cv-arrow-back', '#9B7CFF')}
</defs>`;

/** Standalone SVG document (export / SSR / tests). */
export function renderSvg(scene, { padding = 48, background = '#F2F5FB' } = {}) {
  const w = scene.bounds.w + padding * 2;
  const h = scene.bounds.h + padding * 2;
  return `<svg xmlns="http://www.w3.org/2000/svg" class="cv-root" width="${n1(w)}" height="${n1(h)}" viewBox="${-padding} ${-padding} ${n1(w)} ${n1(h)}">
<style>${CSS}</style>${DEFS}
<rect x="${-padding}" y="${-padding}" width="${n1(w)}" height="${n1(h)}" fill="${background}"/>
${renderContent(scene)}
</svg>`;
}
