// usage: node scripts/render-svg.mjs sketch.ino out.svg
import fs from 'node:fs';
import { configureNode, analyze, renderSvg } from '../src/node.js';
configureNode();
const { scene } = await analyze(fs.readFileSync(process.argv[2], 'utf8'));
fs.writeFileSync(process.argv[3], renderSvg(scene));
console.log('wrote', process.argv[3], scene.bounds);
