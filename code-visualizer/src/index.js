// Public API.
//
//   import { configure, analyze, renderSvg } from 'code-visualizer';
//   configure({ Parser, grammars: { cpp: '/vendor/tree-sitter-cpp.wasm' } });
//   const { scene, program, diagnostics } = await analyze(source, { language: 'arduino' });
//
// `scene` is plain JSON (nodes with x/y/w/h, edges with routed points, lanes), so it can be drawn by
// the bundled SVG renderer or fed into any other canvas (React Flow, Konva, ...).

import { configure, withTree } from './core/parser.js';
import { extractProgram } from './lang/cfamily.js';
import { arduino } from './profiles/arduino.js';
import { resolveRoles, groupProgram } from './graph/simplify.js';
import { buildScene } from './graph/layout.js';

export { configure } from './core/parser.js';
export { renderSvg } from './render/svg.js';
export { buildTour } from './graph/tour.js';

/** language id -> { profile, extract(root, source, profile) } */
const LANGUAGES = {
  arduino: { profile: arduino, extract: extractProgram },
  ino: { profile: arduino, extract: extractProgram },
  cpp: { profile: arduino, extract: extractProgram },
};

export const supportedLanguages = () => Object.keys(LANGUAGES);

export function languageForFile(name) {
  const lower = name.toLowerCase();
  for (const [id, l] of Object.entries(LANGUAGES)) if (l.profile.extensions.some((e) => lower.endsWith(e))) return id;
  return null;
}

export async function analyze(source, { language = 'arduino', groupConsecutive = true, mode = 'kids', showCode = true } = {}) {
  const lang = LANGUAGES[language];
  if (!lang) throw new Error(`Unsupported language "${language}". Supported: ${supportedLanguages().join(', ')}`);
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();

  const program = await withTree(lang.profile.grammar, source, (root) => lang.extract(root, source, lang.profile));
  resolveRoles(program, lang.profile);
  lang.profile.annotate?.(program);
  if (groupConsecutive) groupProgram(program, lang.profile);
  const scene = buildScene(program, { mode, showCode });

  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  return {
    program,
    scene,
    diagnostics: program.diagnostics,
    stats: { ms: Math.round(t1 - t0), nodes: scene.nodes.length, edges: scene.edges.length, functions: program.functions.length },
  };
}
