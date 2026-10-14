// usage: node scripts/dump-scene.mjs sketch.ino [--json]
import fs from 'node:fs';
import { configureNode, analyze } from '../src/node.js';
configureNode();
const file = process.argv[2];
const src = fs.readFileSync(file, 'utf8');
const { program, scene, diagnostics, stats } = await analyze(src);
if (process.argv.includes('--json')) { console.log(JSON.stringify(scene, null, 1)); process.exit(0); }
console.log(stats, diagnostics);
for (const f of program.functions) console.log(f.role.padEnd(8), f.signature, f.calledBy ?? '');
for (const n of scene.nodes) console.log(n.id.padEnd(4), n.kind.padEnd(10), `(${n.x|0},${n.y|0}) ${n.w}x${n.h}`, (n.titleLines?.join(' ') || n.labelLines?.join(' ')), n.lines?.length ? '['+n.lines.join('; ')+']' : '');
