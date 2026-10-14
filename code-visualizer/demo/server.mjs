// Zero-dependency static server for the demo. Serves the package root and maps /vendor/* to the
// tree-sitter runtime + grammars installed in node_modules (so nothing large is committed).
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const tsDir = path.dirname(require.resolve('web-tree-sitter/package.json'));
const gramDir = path.join(path.dirname(require.resolve('tree-sitter-wasms/package.json')), 'out');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.json': 'application/json', '.svg': 'image/svg+xml' };

function resolve(urlPath) {
  const p = decodeURIComponent(urlPath.split('?')[0]);
  if (p === '/') return path.join(root, 'demo/index.html');
  if (p === '/vendor/tree-sitter.js') return path.join(tsDir, 'tree-sitter.js');
  if (p === '/vendor/tree-sitter.wasm') return path.join(tsDir, 'tree-sitter.wasm');
  if (p.startsWith('/vendor/grammars/')) return path.join(gramDir, path.basename(p));
  const abs = path.join(root, p);
  return abs.startsWith(root + path.sep) && !abs.includes('node_modules') ? abs : null;
}

const port = Number(process.env.CV_PORT) || 5178;
http.createServer((req, res) => {
  const file = resolve(req.url);
  fs.readFile(file ?? '', (err, data) => {
    if (err || !file) { res.writeHead(404).end('not found'); return; }
    res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream', 'cache-control': 'no-cache' }).end(data);
  });
}).listen(port, () => console.log(`code-visualizer demo → http://localhost:${port}`));
