// Interactive pan / zoom / select viewer for a scene. Browser only; framework-agnostic.
//
//   const viewer = createViewer(el, { onSelect: (node) => editor.highlight(node.loc) });
//   viewer.setScene(scene);
//   viewer.selectByLine(12);   // editor -> graph
//   viewer.fit();

import { renderContent, DEFS, CSS } from './svg.js';

const SVGNS = 'http://www.w3.org/2000/svg';

export function createViewer(container, { onSelect, minZoom = 0.08, maxZoom = 2.5 } = {}) {
  container.classList.add('cv-viewer');
  container.innerHTML = `<style>${CSS}
.cv-viewer{position:relative;overflow:hidden;touch-action:none;background-color:#F2F5FB;background-image:radial-gradient(#D3DBEA 1.4px,transparent 1.4px);background-size:24px 24px;cursor:grab}
.cv-vp{transform-origin:0 0}
.cv-viewer.panning{cursor:grabbing}
.cv-viewer>svg{width:100%;height:100%;display:block}
.cv-ctl{position:absolute;right:18px;bottom:18px;display:flex;flex-direction:column;gap:8px}
.cv-ctl button{width:44px;height:44px;border-radius:50%;border:0;background:#fff;color:#25324D;font:800 20px system-ui;box-shadow:0 3px 10px rgba(37,50,77,.16);cursor:pointer}
.cv-ctl button:hover{background:#F0F3F5}
.cv-empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#7A8591;font:14px system-ui}
</style>
<svg class="cv-root" xmlns="${SVGNS}">${DEFS}<g class="cv-vp"></g></svg>
<div class="cv-ctl"><button data-a="out" title="Zoom out">−</button><button data-a="in" title="Zoom in">+</button><button data-a="fit" title="Fit to view">⤢</button></div>
<div class="cv-empty">Nothing to draw yet</div>`;

  const svg = container.querySelector('svg');
  const vp = container.querySelector('.cv-vp');
  const empty = container.querySelector('.cv-empty');
  let scene = null;
  let view = { x: 0, y: 0, k: 1 };
  let selectedId = null;
  let userMoved = false;

  const apply = () => { vp.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`; };
  const size = () => container.getBoundingClientRect();

  // `readable`: never start below 0.55x on big graphs (anchor top-left and let the user pan);
  // the toolbar "Fit" button always shows everything.
  function fit(pad = 40, readable = false, top = 104) {
    if (!scene) return;
    const r = size();
    const { w, h } = scene.bounds;
    const full = Math.min(1, (r.width - pad * 2) / Math.max(w, 1), (r.height - top - pad) / Math.max(h, 1));
    const k = Math.max(minZoom, readable ? Math.max(full, 0.55) : full);
    const overflow = w * k > r.width - pad * 2;
    view = { k, x: overflow ? pad : (r.width - w * k) / 2, y: h * k > r.height - top - pad ? top : Math.max(top, (r.height - h * k) / 2) };
    apply();
  }

  function zoomAt(factor, cx, cy) {
    const k = Math.min(maxZoom, Math.max(minZoom, view.k * factor));
    const f = k / view.k;
    view = { k, x: cx - (cx - view.x) * f, y: cy - (cy - view.y) * f };
    userMoved = true;
    apply();
  }

  function centerOn(x, y, k = Math.max(view.k, 0.9)) {
    const r = size();
    view = { k, x: r.width / 2 - x * k, y: r.height / 2 - y * k };
    userMoved = true;
    apply();
  }

  // ---- pan / zoom input ----
  let drag = null;
  container.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.cv-ctl') || e.target.closest('[data-node]')) return;
    drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    vp.style.transition = 'none';
    container.classList.add('panning');
    container.setPointerCapture(e.pointerId);
  });
  container.addEventListener('pointermove', (e) => {
    if (!drag) return;
    view.x = drag.vx + e.clientX - drag.x;
    view.y = drag.vy + e.clientY - drag.y;
    userMoved = true;
    apply();
  });
  const endDrag = () => { drag = null; container.classList.remove('panning'); };
  container.addEventListener('pointerup', endDrag);
  container.addEventListener('pointercancel', endDrag);
  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = size();
    if (e.ctrlKey || e.metaKey) zoomAt(Math.exp(-e.deltaY * 0.01), e.clientX - r.left, e.clientY - r.top);
    else { view.x -= e.deltaX; view.y -= e.deltaY; userMoved = true; apply(); }
  }, { passive: false });
  container.querySelector('.cv-ctl').addEventListener('click', (e) => {
    const a = e.target.closest('button')?.dataset.a;
    const r = size();
    if (a === 'fit') { userMoved = false; fit(); }
    else if (a) zoomAt(a === 'in' ? 1.25 : 0.8, r.width / 2, r.height / 2);
  });
  new ResizeObserver(() => { if (!userMoved) fit(40, true); }).observe(container);

  // ---- selection ----
  svg.addEventListener('click', (e) => {
    const g = e.target.closest('[data-node]');
    if (g) select(g.dataset.node, { source: 'graph' });
  });
  svg.addEventListener('dblclick', (e) => {
    const g = e.target.closest('[data-node]');
    const n = g && scene.nodes.find((x) => x.id === g.dataset.node);
    const link = n && scene.links.find((l) => l.from === n.id);
    if (link) focusLane(link.toLane);
  });

  function focusLane(id) {
    const l = scene.lanes.find((x) => x.id === id);
    if (l) centerOn(l.x + l.w / 2, l.y + Math.min(l.h / 2, 300 / view.k + 80), view.k);
  }

  function clearLinks() {
    vp.querySelector('.links').innerHTML = '';
    vp.querySelectorAll('.lane.hi').forEach((el) => el.classList.remove('hi'));
  }

  function select(id, { pan = false, source = 'api' } = {}) {
    vp.querySelectorAll('.node.sel').forEach((el) => el.classList.remove('sel'));
    vp.querySelectorAll('.edge.hi').forEach((el) => el.classList.remove('hi'));
    clearLinks();
    selectedId = id;
    const node = scene?.nodes.find((n) => n.id === id);
    if (!node) return;
    vp.querySelector(`[data-node="${id}"]`)?.classList.add('sel');
    vp.querySelectorAll(`.edge[data-from="${id}"], .edge[data-to="${id}"]`).forEach((el) => el.classList.add('hi'));
    const link = scene.links.find((l) => l.from === id);
    if (link) {
      const lane = scene.lanes.find((l) => l.id === link.toLane);
      vp.querySelector(`[data-lane="${CSS_ESC(lane.id)}"]`)?.classList.add('hi');
      const x1 = node.x + node.w / 2;
      const y1 = node.y + node.h / 2;
      const x2 = lane.x;
      const y2 = lane.y + 40;
      const path = document.createElementNS(SVGNS, 'path');
      path.setAttribute('class', 'link');
      path.setAttribute('d', `M${x1} ${y1} C${x1 + 120} ${y1}, ${x2 - 120} ${y2}, ${x2} ${y2}`);
      path.setAttribute('marker-end', 'url(#cv-arrow)');
      vp.querySelector('.links').appendChild(path);
    }
    if (pan) {
      const r = size();
      const sx = node.x * view.k + view.x;
      const sy = node.y * view.k + view.y;
      if (sx < 40 || sy < 40 || sx > r.width - 40 || sy > r.height - 40) centerOn(node.x, node.y);
    }
    if (source !== 'api' || pan) onSelect?.(node, { source });
  }

  function selectByLine(line, opts = {}) {
    if (!scene) return null;
    let best = null;
    for (const n of scene.nodes) {
      if (!n.loc || line < n.loc.start || line > n.loc.end) continue;
      if (!best || n.loc.end - n.loc.start < best.loc.end - best.loc.start) best = n;
    }
    if (best) select(best.id, { pan: true, ...opts, source: 'editor' });
    return best;
  }

  // ---- guided run ----
  let runTimer = null;
  function clearRun() {
    vp.querySelectorAll('.node.run, .edge.run').forEach((el) => el.classList.remove('run'));
  }
  function stop() {
    clearTimeout(runTimer);
    runTimer = null;
    clearRun();
    vp.style.transition = 'none';
  }
  function play(steps, { interval = 1100, onStep, onEnd } = {}) {
    stop();
    let i = 0;
    const tick = () => {
      clearRun();
      if (i >= steps.length) { stop(); onEnd?.(); return; }
      const { node, via } = steps[i++];
      vp.querySelector(`[data-node="${node.id}"]`)?.classList.add('run');
      if (via) vp.querySelector(`[data-edge="${via.id}"]`)?.classList.add('run');
      const r = size();
      vp.style.transition = 'transform .6s ease';
      const k = Math.min(1, Math.max(view.k, 0.75));
      view = { k, x: r.width / 2 - node.x * k, y: r.height * 0.42 - (node.y + node.h / 2) * k };
      userMoved = true;
      apply();
      onStep?.(node, i - 1, steps.length);
      runTimer = setTimeout(tick, interval);
    };
    tick();
    return { stop };
  }

  return {
    play,
    stop,
    isPlaying: () => runTimer != null,
    setScene(next) {
      scene = next;
      const first = !vp.querySelector('.nodes');
      empty.style.display = scene.nodes.length ? 'none' : 'flex';
      vp.innerHTML = renderContent(scene);
      if (first || !userMoved) fit(40, true);
      if (selectedId && scene.nodes.some((n) => n.id === selectedId)) select(selectedId);
    },
    fit: () => { userMoved = false; fit(); },
    select,
    selectByLine,
    focusLane,
    zoomBy: (f) => { const r = size(); zoomAt(f, r.width / 2, r.height / 2); },
    getSceneSvgElement: () => svg,
    destroy: () => { stop(); container.innerHTML = ''; },
  };
}

const CSS_ESC = (s) => (window.CSS?.escape ? window.CSS.escape(s) : s.replace(/"/g, '\\"'));
