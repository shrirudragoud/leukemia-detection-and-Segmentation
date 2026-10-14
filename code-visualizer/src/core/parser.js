// Thin wrapper around web-tree-sitter. The Parser class and wasm locations are injected so the
// same code runs in Node (tests/SSR), a bundler (Vite/webpack) or a plain <script> page.

let ParserClass = null;
let initPromise = null;
let grammarSources = {};
const languages = new Map();
const parsers = new Map();

/**
 * @param {{Parser: any, runtimeWasm?: string, grammars: Record<string,string>}} opts
 *   Parser   – the web-tree-sitter default export (or window.TreeSitter in a browser)
 *   runtimeWasm – URL/path of tree-sitter.wasm (only needed if it is not next to the JS file)
 *   grammars – grammar name -> URL/path of its .wasm (e.g. { cpp: '/vendor/tree-sitter-cpp.wasm' })
 */
export function configure({ Parser, runtimeWasm, grammars }) {
  if (ParserClass && ParserClass !== Parser) {
    languages.clear();
    parsers.clear();
    initPromise = null;
  }
  ParserClass = Parser;
  grammarSources = { ...grammarSources, ...grammars };
  if (runtimeWasm) ParserClass.__runtimeWasm = runtimeWasm;
}

async function init() {
  if (!ParserClass) throw new Error('code-visualizer: call configure({ Parser, grammars }) first');
  const wasm = ParserClass.__runtimeWasm;
  initPromise ??= ParserClass.init(wasm ? { locateFile: () => wasm } : undefined);
  await initPromise;
}

async function parserFor(grammar) {
  await init();
  if (parsers.has(grammar)) return parsers.get(grammar);
  const src = grammarSources[grammar];
  if (!src) throw new Error(`code-visualizer: no wasm configured for grammar "${grammar}"`);
  if (!languages.has(grammar)) languages.set(grammar, await ParserClass.Language.load(src));
  const parser = new ParserClass();
  parser.setLanguage(languages.get(grammar));
  parsers.set(grammar, parser);
  return parser;
}

/** Parse `source`, run `fn(rootNode)` while the tree is alive, then free the wasm tree. */
export async function withTree(grammar, source, fn) {
  const parser = await parserFor(grammar);
  const tree = parser.parse(source);
  try {
    return fn(tree.rootNode);
  } finally {
    tree.delete();
  }
}
