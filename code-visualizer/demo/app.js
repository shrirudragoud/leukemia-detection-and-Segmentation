import { configure, analyze, renderSvg, buildTour } from '../src/index.js';
import { createViewer } from '../src/render/viewer.js';
import { SAMPLES, LABELS } from './samples.js';

const $ = (id) => document.getElementById(id);
const src = $('src');
const status = $('status');

configure({
  Parser: window.TreeSitter,
  runtimeWasm: '/vendor/tree-sitter.wasm',
  grammars: { cpp: '/vendor/grammars/tree-sitter-cpp.wasm', c: '/vendor/grammars/tree-sitter-c.wasm' },
});

const state = { mode: 'kids', showCode: true };
let last = null;
let seq = 0;
let runner = null;

const viewer = createViewer($('viewer'), {
  onSelect: (node, { source }) => {
    explain(node);
    if (source === 'graph' && node.loc) highlight(node.loc, true);
  },
});

// ---- explanation card ----
const escapeHtml = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const HINT = { emoji: '👆', title: 'Click a block!', text: 'Click any block (or a line of code) to learn what it does — or press “Watch it run”.' };

function explain(node) {
  const d = node
    ? { emoji: node.emoji, title: node.titleLines.join(' '), text: node.explain || '', code: node.text, warn: node.warn }
    : HINT;
  $('explain').innerHTML = `<div class="em">${d.emoji}</div><div><h3>${escapeHtml(d.title)}</h3><p>${escapeHtml(d.text)}</p>${
    d.code ? `<code>${escapeHtml(d.code.replace(/\s+/g, ' ').slice(0, 160))}</code>` : ''}${d.warn ? `<div class="warn">⚠️ ${escapeHtml(d.warn)}</div>` : ''}</div>`;
}
explain(null);

// ---- analysis ----
async function run() {
  const mine = ++seq;
  stopRun();
  status.className = 'chip';
  status.textContent = 'thinking…';
  try {
    const res = await analyze(src.value, { language: 'arduino', mode: state.mode, showCode: state.showCode });
    if (mine !== seq) return;
    last = res;
    viewer.setScene(res.scene);
    const errs = res.diagnostics.length;
    status.className = `chip ${errs ? 'bad' : 'ok'}`;
    status.textContent = errs
      ? `${errs} typo${errs > 1 ? 's' : ''} to fix — the rest is still shown`
      : `${res.stats.nodes} blocks · ${res.stats.ms} ms`;
    $('diag').innerHTML = res.diagnostics.map((d) => `<div data-line="${d.line}">Line ${d.line}: ${escapeHtml(d.message)}</div>`).join('');
  } catch (err) {
    console.error(err);
    status.className = 'chip bad';
    status.textContent = String(err.message || err);
  }
}

// ---- guided run ----
function stopRun() {
  viewer.stop();
  runner = null;
  $('play').textContent = '▶ Watch it run';
  $('play').classList.remove('stop');
}
$('play').addEventListener('click', () => {
  if (runner) { stopRun(); return; }
  if (!last) return;
  const steps = buildTour(last.scene);
  if (!steps.length) return;
  $('play').textContent = '⏹ Stop';
  $('play').classList.add('stop');
  runner = viewer.play(steps, {
    interval: 1300,
    onStep: (node) => { explain(node); if (node.loc) highlight(node.loc, true); },
    onEnd: () => { stopRun(); explain({ emoji: '🎉', titleLines: ['That is one full run!'], explain: 'The board does this over and over — thousands of times every second. Try changing the code and watch the blocks change.' }); },
  });
});

// ---- editor ----
const gutter = $('gutter');
const hl = $('hl');
let hlRange = null;
const LH = 21;
const PAD = 12;

function updateGutter() {
  const n = src.value.split('\n').length;
  gutter.textContent = Array.from({ length: n }, (_, i) => i + 1).join('\n');
  gutter.style.paddingTop = `${PAD}px`;
  syncScroll();
}
function syncScroll() { gutter.scrollTop = src.scrollTop; if (hlRange) placeHl(); }
function placeHl() {
  hl.style.display = 'block';
  hl.style.top = `${PAD + (hlRange.start - 1) * LH - src.scrollTop}px`;
  hl.style.height = `${(hlRange.end - hlRange.start + 1) * LH}px`;
}
function highlight(range, scrollIntoView) {
  hlRange = range;
  placeHl();
  if (scrollIntoView) {
    const top = PAD + (range.start - 1) * LH;
    if (top < src.scrollTop || top > src.scrollTop + src.clientHeight - 2 * LH) src.scrollTop = Math.max(0, top - src.clientHeight / 3);
  }
}
const currentLine = () => src.value.slice(0, src.selectionStart).split('\n').length;
function onCaret() {
  if (runner) return;
  const n = viewer.selectByLine(currentLine());
  if (n?.loc) { highlight(n.loc, false); explain(n); }
}

let timer;
src.addEventListener('input', () => { updateGutter(); clearTimeout(timer); timer = setTimeout(run, 250); });
src.addEventListener('scroll', syncScroll);
src.addEventListener('click', onCaret);
src.addEventListener('keyup', (e) => { if (e.key.startsWith('Arrow') || e.key === 'Home' || e.key === 'End') onCaret(); });
$('diag').addEventListener('click', (e) => {
  const line = Number(e.target.closest('[data-line]')?.dataset.line);
  if (!line) return;
  const pos = src.value.split('\n').slice(0, line - 1).reduce((a, l) => a + l.length + 1, 0);
  src.focus();
  src.setSelectionRange(pos, pos);
  highlight({ start: line, end: line }, true);
});

// ---- samples / files ----
const sel = $('sample');
for (const name of Object.keys(SAMPLES)) sel.add(new Option(LABELS[name] || name, name));
sel.addEventListener('change', () => load(SAMPLES[sel.value]));
$('open').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', async (e) => { const f = e.target.files[0]; if (f) load(await f.text()); });
src.addEventListener('dragover', (e) => e.preventDefault());
src.addEventListener('drop', async (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) load(await f.text()); });

function load(text) {
  src.value = text;
  hlRange = null;
  hl.style.display = 'none';
  src.scrollTop = 0;
  explain(null);
  updateGutter();
  run();
}

// ---- toolbar ----
$('fit').addEventListener('click', () => viewer.fit());
$('mode').addEventListener('click', (e) => {
  const m = e.target.closest('button')?.dataset.m;
  if (!m || m === state.mode) return;
  state.mode = m;
  for (const b of $('mode').children) b.classList.toggle('on', b.dataset.m === m);
  $('showcode').disabled = m === 'code';
  run();
});
$('showcode').addEventListener('change', (e) => { state.showCode = e.target.checked; run(); });
$('svg').addEventListener('click', () => {
  if (!last) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([renderSvg(last.scene)], { type: 'image/svg+xml' }));
  a.download = 'my-blocks.svg';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

window.__cv = { viewer, run, state, get last() { return last; } };
sel.value = 'Blink';
load(SAMPLES.Blink);
