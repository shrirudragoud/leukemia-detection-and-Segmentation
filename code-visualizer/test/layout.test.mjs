import test from 'node:test';
import assert from 'node:assert/strict';
import { configureNode, analyze, renderSvg } from '../src/node.js';
import { SAMPLES } from '../demo/samples.js';
import { TORTURE, SNIPPETS } from './fixtures.mjs';

configureNode();

const all = { ...SAMPLES, torture: TORTURE, ...SNIPPETS };

function checkScene(name, scene) {
  const byId = new Map(scene.nodes.map((n) => [n.id, n]));
  assert.equal(byId.size, scene.nodes.length, `${name}: duplicate node ids`);

  // 1. nodes inside one lane never overlap
  const byLane = new Map();
  for (const n of scene.nodes) (byLane.get(n.lane) ?? byLane.set(n.lane, []).get(n.lane)).push(n);
  for (const [lane, ns] of byLane) {
    for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
      const a = ns[i]; const b = ns[j];
      const ox = Math.min(a.x + a.w / 2, b.x + b.w / 2) - Math.max(a.x - a.w / 2, b.x - b.w / 2);
      const oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      assert.ok(!(ox > 0 && oy > 0), `${name}: nodes ${a.id} and ${b.id} overlap in ${lane}`);
    }
  }

  // 2. every node sits inside its lane panel
  for (const n of scene.nodes) {
    const l = scene.lanes.find((x) => x.id === n.lane);
    assert.ok(n.x - n.w / 2 >= l.x && n.x + n.w / 2 <= l.x + l.w && n.y >= l.y && n.y + n.h <= l.y + l.h, `${name}: ${n.id} escapes its lane`);
  }

  // 3. edges: known endpoints, orthogonal, drawn downwards except back edges
  for (const e of scene.edges) {
    assert.ok(byId.has(e.from) && byId.has(e.to), `${name}: edge ${e.id} dangling`);
    assert.ok(e.points.length >= 2, `${name}: edge ${e.id} has no geometry`);
    for (let i = 1; i < e.points.length; i++) {
      const [a, b] = [e.points[i - 1], e.points[i]];
      assert.ok(a[0] === b[0] || a[1] === b[1], `${name}: edge ${e.id} has a diagonal segment`);
    }
    if (e.kind !== 'back') {
      const last = e.points[e.points.length - 1];
      const first = e.points[0];
      assert.ok(last[1] >= first[1], `${name}: forward edge ${e.id} runs upwards`);
    }
  }

  // 3b. no edge segment cuts through a node it is not attached to
  for (const e of scene.edges) {
    for (let i = 1; i < e.points.length; i++) {
      const [[x1, y1], [x2, y2]] = [e.points[i - 1], e.points[i]];
      for (const n of scene.nodes) {
        if (n.id === e.from || n.id === e.to) continue;
        const l = n.x - n.w / 2 + 1; const r = n.x + n.w / 2 - 1; const t = n.y + 1; const b = n.y + n.h - 1;
        const hit = Math.max(Math.min(x1, x2), l) <= Math.min(Math.max(x1, x2), r) && Math.max(Math.min(y1, y2), t) <= Math.min(Math.max(y1, y2), b);
        assert.ok(!hit, `${name}: edge ${e.id} (${e.from}->${e.to}, ${e.kind}) passes through node ${n.id} (${n.kind})`);
      }
    }
  }

  // 4. everything except a lane's start node is reachable (has an incoming edge)
  const incoming = new Set(scene.edges.map((e) => e.to));
  for (const n of scene.nodes) {
    if (n.kind === 'start' || n.kind === 'globals') continue;
    assert.ok(incoming.has(n.id), `${name}: node ${n.id} (${n.kind}) has no incoming edge`);
  }
}

for (const mode of ['kids', 'code']) {
  for (const [name, src] of Object.entries(all)) {
    test(`layout invariants [${mode}]: ${name}`, async () => {
      const { scene } = await analyze(src, { mode });
      checkScene(`${name}/${mode}`, scene);
      const svg = renderSvg(scene);
      assert.ok(svg.startsWith('<svg') && svg.endsWith('</svg>'));
    });
  }
}

test('analysis is deterministic', async () => {
  const a = await analyze(TORTURE);
  const b = await analyze(TORTURE);
  assert.equal(JSON.stringify(a.scene), JSON.stringify(b.scene));
});
