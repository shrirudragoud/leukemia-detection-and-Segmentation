// Node.js convenience: wires web-tree-sitter + the bundled grammars from node_modules.
import { createRequire } from 'node:module';
import path from 'node:path';
import Parser from 'web-tree-sitter';
import { configure } from './core/parser.js';

const require = createRequire(import.meta.url);
const wasmDir = path.join(path.dirname(require.resolve('tree-sitter-wasms/package.json')), 'out');

export function configureNode() {
  configure({
    Parser,
    grammars: { cpp: path.join(wasmDir, 'tree-sitter-cpp.wasm'), c: path.join(wasmDir, 'tree-sitter-c.wasm'), python: path.join(wasmDir, 'tree-sitter-python.wasm') },
  });
}

export { analyze, renderSvg } from './index.js';
