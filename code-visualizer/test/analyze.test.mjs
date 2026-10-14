import test from 'node:test';
import assert from 'node:assert/strict';
import { configureNode, analyze } from '../src/node.js';
import { SAMPLES } from '../demo/samples.js';
import { walkStmts } from '../src/core/ir.js';

configureNode();
const kinds = (fn) => { const k = []; walkStmts(fn.body, (s) => k.push(s.t === 'action' ? s.kind : s.t)); return k; };
const fn = (r, name) => r.program.functions.find((f) => f.name === name);

const titles = (r) => r.scene.nodes.map((n) => n.titleLines?.join(' '));

test('Blink: roles and node kinds', async () => {
  const r = await analyze(SAMPLES.Blink);
  assert.equal(fn(r, 'setup').role, 'setup');
  assert.equal(fn(r, 'loop').role, 'loop');
  assert.deepEqual(kinds(fn(r, 'loop')), ['output', 'wait', 'output', 'wait']);
});

test('kids mode speaks plain English; code mode stays technical', async () => {
  const kid = titles(await analyze(SAMPLES.Blink));
  assert.ok(kid.includes('Turn ON LED (pin 13)'), kid.join(' | '));
  assert.ok(kid.includes('Turn OFF LED (pin 13)'));
  assert.ok(kid.includes('Wait 1 second'));
  const code = titles(await analyze(SAMPLES.Blink, { mode: 'code' }));
  assert.ok(code.includes('Set LED_PIN (13) HIGH'), 'named pin constants are resolved');
  assert.ok(code.includes('Wait 1000 ms'));
});

test('kids mode shows the real code under each block', async () => {
  const r = await analyze(SAMPLES.Blink);
  const n = r.scene.nodes.find((x) => x.titleLines[0].startsWith('Turn ON'));
  assert.equal(n.code, 'digitalWrite(LED_PIN, HIGH)');
  assert.ok(n.explain.length > 20, 'every block carries an explanation');
  assert.equal((await analyze(SAMPLES.Blink, { showCode: false })).scene.nodes.find((x) => x.titleLines[0].startsWith('Turn ON')).code, null);
});

test('conditions are translated to words', async () => {
  const r = await analyze(SAMPLES['Button + LED']);
  const q = r.scene.nodes.find((n) => n.kind === 'decision').titleLines.join(' ');
  assert.equal(q, 'reading is OFF and lastState is ON?');
  const pin = await analyze('const int btn = 2; void loop(){ if (digitalRead(btn) == HIGH) { delay(5); } }');
  assert.equal(pin.scene.nodes.find((n) => n.kind === 'decision').titleLines.join(' '), 'button (pin 2) is ON?');
});

test('counting for-loops become "Repeat N times" and hide the counter plumbing', async () => {
  const src = 'void loop(){ for (int i = 0; i < 3; i++) { digitalWrite(13, HIGH); } }';
  const kid = await analyze(src);
  assert.ok(titles(kid).includes('Repeat 3 times'));
  assert.ok(!kid.scene.nodes.some((n) => /i\+\+|Add 1 to i|int i = 0/.test(n.titleLines.join(' '))));
  const code = await analyze(src, { mode: 'code' });
  assert.ok(code.scene.nodes.some((n) => n.titleLines.join(' ') === 'i++'), 'code mode keeps the step');
});

test('attachInterrupt marks the handler as an ISR and links to its lane', async () => {
  const r = await analyze(SAMPLES['Parking sensor']);
  assert.equal(fn(r, 'onReset').role, 'isr');
  assert.ok(r.scene.links.some((l) => l.toLane === 'fn:onReset' && l.kind === 'trigger'));
  assert.equal(fn(r, 'readDistanceCm').role, 'user');
  assert.deepEqual(fn(r, 'readDistanceCm').calledBy, ['loop', 'loop']);
});

test('consecutive pinMode calls collapse into one card', async () => {
  const code = (await analyze(SAMPLES['Parking sensor'], { mode: 'code' })).scene.nodes.find((n) => n.kind === 'pin-config');
  assert.equal(code.lines.length, 6);
  assert.ok(code.lines[0].includes('TRIG (7) → OUTPUT'));
  const kid = (await analyze(SAMPLES['Parking sensor'])).scene.nodes.find((n) => n.kind === 'pin-config');
  assert.equal(kid.titleLines.join(' '), 'Get the pins ready');
  assert.equal(kid.lines[0], "Trigger (pin 7) sends out");
});

test('switch: shared case labels and trailing breaks', async () => {
  const r = await analyze(SAMPLES['Parking sensor']);
  const sw = fn(r, 'loop').body.find((s) => s.t === 'switch');
  assert.deepEqual(sw.cases.map((c) => c.labels), [['0'], ['1', '2'], ['default']]);
  assert.ok(sw.cases.every((c) => c.body.at(-1)?.t !== 'break' || c.labels.includes('1')));
  assert.ok(r.scene.edges.map((e) => e.label).includes('is 1 or 2'));
  const code = await analyze(SAMPLES['Parking sensor'], { mode: 'code' });
  assert.ok(code.scene.edges.map((e) => e.label).includes('1, 2'));
});

test('ISR lint: delay/Serial inside an interrupt handler are flagged', async () => {
  const r = await analyze('void tick(){ delay(5); Serial.println(1); }\nvoid setup(){ attachInterrupt(0, tick, RISING); }\nvoid loop(){}');
  assert.equal(r.scene.nodes.filter((n) => n.warn).length, 2);
});

test('library objects are labelled by their declared type', async () => {
  const r = await analyze(SAMPLES['Servo sweep']);
  const n = r.scene.nodes.find((x) => x.titleLines?.[0] === 'Turn the servo to pos°');
  assert.equal(n.category, 'SERVO');
});

test('syntax errors are reported with line numbers but the rest still renders', async () => {
  const r = await analyze('void setup(){ pinMode(2, OUTPUT); }\nvoid loop(){ digitalWrite(2, HIGH\n delay(5); }');
  assert.ok(r.diagnostics.length > 0 && r.diagnostics[0].line >= 2);
  assert.ok(r.scene.nodes.some((n) => n.kind === 'pin-config'));
});

test('#ifdef: only the primary branch is drawn and tagged', async () => {
  const r = await analyze('void loop(){\n#ifdef DEBUG\n Serial.println(1);\n#else\n foo();\n#endif\n bar(); }');
  const nodes = r.scene.nodes.filter((n) => n.shape === 'card');
  assert.equal(nodes.length, 2);
  assert.equal(nodes[0].subLines[0], '#ifdef DEBUG');
});

test('unsupported language is rejected clearly', async () => {
  await assert.rejects(() => analyze('x', { language: 'cobol' }), /Unsupported language/);
});
