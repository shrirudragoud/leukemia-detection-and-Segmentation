# code-visualizer

Paste or load embedded source code and get an auto-laid-out **block flowchart that school kids can read** — like
Scratch, but generated from the real code. Lanes per function, questions as hexagons with green *yes* / red *no*
arrows, loops drawn as tinted "go round again" regions, `switch` as columns, interrupts and calls linked across
lanes, and every block mapped back to its source lines (click a block → highlight code, click code → select block).

Today: **Arduino `.ino` / C++**. Next: C, MicroPython (see [Roadmap](#roadmap)).

### Built for learners

* **Plain English first** – `digitalWrite(LED_PIN, HIGH)` → *"Turn ON LED (pin 13)"*, `delay(1000)` → *"Wait 1 second"*,
  `for (int i=0;i<3;i++)` → *"Repeat 3 times"*, `digitalRead(btn) == HIGH && x` → *"button (pin 2) is ON and x"*.
  `#define`/`const` pin names become words (`ledPin` → "LED", `resetBtn` → "reset button").
* **The real code stays visible** under every block, so kids learn the mapping (toggle "real code" off to hide it).
  **Kid view / Code view** switch re-labels the same graph with technical text (`analyze(src, { mode: 'code' })`).
* **"What does this block do?"** – every block carries an `explain` sentence ("An interrupt is like a doorbell…").
* **▶ Watch it run** – `buildTour(scene)` produces a guided walkthrough (questions alternate yes/no, loops go round,
  calls to your own blocks are followed); the viewer spotlights each block + arrow while the editor highlights the
  line. It is a walkthrough, not an interpreter: it never evaluates conditions.
* **Friendly warnings** – e.g. `delay()` or `Serial` inside an interrupt handler gets a red marker with a kid-readable tip.
* **Scratch-like look** – hat-shaped *"When the board starts"* start block, colour-coded categories (Scratch palette:
  yellow events, orange control, purple output, cyan sensing, pink *My block*), emoji badges, chunky 3D blocks.

```
npm install
npm run demo        # http://localhost:5178  (set CV_PORT to change)
npm test
```

## Research summary

| Question | Options looked at | Decision |
|---|---|---|
| How to parse `.ino` | Regex/line scanning · Arduino CLI preprocessor · **tree-sitter** | **tree-sitter** (`web-tree-sitter` + prebuilt grammars). Error-tolerant (half-typed code in an IDE still yields a tree), incremental, runs in the browser as WASM, and one runtime covers C++, C and Python. `.ino` is C++ with auto-generated prototypes, so the C++ grammar parses it directly. |
| How to lay it out | elkjs · dagre · React Flow + dagre · **custom structured layout** | Source code is *structured* control flow (series-parallel), so a deterministic recursive layout beats a general graph engine: no dependency (elk is ~300 KB WASM), stable output when one line is edited, loops/branches/merges always look the same. A general graph layout is only needed for `goto`-style flow, which embedded code rarely uses. |
| What should a node mean | One node per statement · **one node per *intent*** | Calls are classified by an API profile (`pinMode`, `digitalWrite`, `delay`, `Serial.*`, `attachInterrupt`, `Servo`, `Wire`…) so nodes read "Set LED (13) HIGH", "Wait 1000 ms", "Interrupt: resetBtn FALLING → onReset()". `#define`/`const` pin names are resolved. Runs of `pinMode` collapse into one card. |
| Where does it render | Canvas · **SVG string** | `scene` is plain JSON (positions, routed edge polylines, lanes). An SVG renderer is bundled (works in browser, Node/SSR, export); the same JSON can drive React Flow / Konva if the platform prefers. |

Sources consulted: web-tree-sitter README & npm (WASM runtime, `Language.load`), `tree-sitter-wasms` (prebuilt
C/C++/Python grammars), React Flow layouting guide and an ELK-vs-dagre ADR (elk for nested containment, dagre for
simple trees).

> **Version pin:** `tree-sitter-wasms@0.1.13` grammars are ABI-compatible with `web-tree-sitter@0.22.x`, not
> 0.27 (verified: 0.27 fails to load them). Keep both pinned together.

## Architecture

```
source ──► tree-sitter CST ──► language adapter ──► IR ──► roles/grouping ──► layout ──► route ──► scene JSON ──► SVG / viewer
          (core/parser.js)     (lang/cfamily.js)  (core/ir.js)  (graph/simplify.js) (graph/layout.js) (graph/route.js)   (render/*)
                                      ▲
                          API profile (profiles/arduino.js)
```

* **`core/parser.js`** – injects the `Parser` class and wasm URLs (works in Node, bundlers, or a plain `<script>`).
* **`lang/cfamily.js`** – CST → language-neutral **IR** (`action | if | loop | switch | return | break | continue`).
  Handles `#ifdef` (primary branch tagged), `for` (init / condition / step), range-for, do-while, empty-body
  busy-waits (`while(!digitalRead(2));` → one "Wait" card), `switch` with shared labels, syntax-error diagnostics.
* **`profiles/arduino.js`** – *knowledge*, kept separate from syntax: how a call becomes a node, which functions
  are entry points (`setup`, `loop`), ISR lint (`delay`/`Serial` in an interrupt handler).
* **`profiles/plain.js`** – the plain-English layer: identifier → words, `ms` → "1 second", C conditions → English,
  counting-loop detection, and the per-block explanations.
* **`graph/layout.js`** – blocks laid out around a vertical axis; `if` = yes-column left / no-column right with a
  free merge corridor; loops get a right-hand back-edge lane, left-hand exit lane and a tinted region; `switch` = one
  column per case. Break/continue/return are terminal pills wired to the right target.
* **`graph/tour.js`** – the guided-run step sequence used by ▶ Watch it run.
* **`graph/route.js`** – orthogonal edges (branch bus, merge bar, loop lanes). If a structured route would clip an
  unrelated node, an obstacle-avoiding Dijkstra (bend-penalised) re-routes just that edge.
* **`render/svg.js`** – scene → SVG (also used for export). **`render/viewer.js`** – pan / zoom / select, source
  line ↔ node mapping, dashed link from a call/interrupt node to the target function's lane.

### Scene format (stable contract)

```jsonc
{ "lanes": [{ "id": "fn:loop", "name": "void loop()", "role": "loop", "x":…, "y":…, "w":…, "h":… }],
  "nodes": [{ "id": "n9", "kind": "decision", "shape": "hex|card|pill|hat", "x":…, "y":…, "w":…, "h":…,
              "emoji": "❓", "titleLines": ["button (pin 2) is ON?"], "code": "digitalRead(btn) == HIGH",
              "explain": "The board asks a yes/no question…", "loc": { "start": 49, "end": 49 }, "lane": "fn:loop", "warn": "…" }],
  "edges": [{ "from": "n9", "to": "n10", "kind": "branch|flow|back", "label": "yes", "tone": "yes|no|null", "points": [[x,y],…] }],
  "regions": [{ "kind": "loop", "x":…, "y":…, "w":…, "h":… }],
  "links": [{ "from": "n6", "toLane": "fn:onReset", "kind": "trigger" }] }
```

## Using it in the web IDE

```js
import { configure, analyze } from 'code-visualizer';
import { createViewer } from 'code-visualizer/viewer';
import Parser from 'web-tree-sitter';

configure({ Parser, grammars: { cpp: '/assets/tree-sitter-cpp.wasm' } });   // runtimeWasm: '/assets/tree-sitter.wasm' if not beside the JS
const viewer = createViewer(el, { onSelect: (node) => editor.revealLines(node.loc.start, node.loc.end) });

editor.onDidChangeContent(debounce(async () => {
  const { scene, diagnostics } = await analyze(editor.getValue(), { language: 'arduino', mode: 'kids' /* or 'code' */, showCode: true });
  viewer.setScene(scene);                 // keeps your pan/zoom and selection
  editor.setMarkers(diagnostics);
}, 250));
editor.onDidChangeCursor((p) => viewer.selectByLine(p.lineNumber));
```

Serve `tree-sitter.wasm` (from `web-tree-sitter`) and `tree-sitter-cpp.wasm` (from `tree-sitter-wasms/out`) as static
assets; `demo/server.mjs` shows the mapping.

## Roadmap

1. **More plain-English coverage** – more libraries (NeoPixel, LCD, DHT, IR), `millis()` timers ("if 500 ms have passed"),
   variables shown as little labelled boxes with live values during ▶ Watch it run (needs a tiny interpreter or the
   board's serial stream), and translated UI strings for other languages.
2. **C (ESP-IDF / STM32 HAL / AVR)** – grammar `c` is already loadable; add `profiles/esp-idf.js` etc. (GPIO config,
   `vTaskDelay`, FreeRTOS `xTaskCreate` → one lane per task, ISR registration). The adapter is shared.
3. **MicroPython** – new `lang/python.js` (`if/elif/else`, `while/for`, `try/except`, `with`, `def`, `match`) emitting
   the *same IR*, plus `profiles/micropython.js` (`machine.Pin/PWM/ADC/I2C`, `time.sleep`, `Pin.irq`, `asyncio`
   tasks → lanes). Layout/renderer need no changes.
4. **Incremental re-parse** – keep the tree and pass edits (`tree.edit`) for large files; today a full parse takes
   ~100 ms for 100 lines.
5. **Flash/run integration** – highlight the executing node from the serial/debug stream; breakpoints on nodes.
6. **Collapse/expand** – fold loops, long branches and helper calls inline; minimap for very large sketches.
7. **Known limits** – `switch` fall-through is drawn as flowing to the merge; `goto` is a plain card; class
   methods and templates are listed under globals, not as lanes; macros are not expanded.

## Layout of this package

```
src/            core, lang, profiles, graph, render, index.js (public API), node.js (Node helper)
demo/           index.html + app.js + samples.js + server.mjs
test/           layout invariants (no overlaps, orthogonal edges, no edge through a node, determinism) + semantics
scripts/        dump-scene.mjs · render-svg.mjs (CLI helpers)
```
