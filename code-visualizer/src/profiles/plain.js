// Plain-English helpers: turn identifiers, numbers and C expressions into phrases a child can read.

import { collapse } from '../core/text.js';

const WORDS = { led: 'LED', btn: 'button', trig: 'trigger', pwm: 'PWM', lcd: 'LCD', ir: 'IR', rx: 'RX', tx: 'TX', pot: 'knob', ldr: 'light sensor' };

/** ledPin -> "LED", resetBtn -> "reset button", BUZZER -> "buzzer" */
export function humanName(id) {
  const s = id.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase().replace(/\b(pin|pins|no|num)\b/g, '');
  const words = s.split(/\s+/).filter(Boolean).map((w) => WORDS[w] || w);
  return words.length ? words.join(' ') : id;
}

/** "the LED (pin 13)" for a named constant, "pin 7" for a number, the raw text for variables. */
export function thing(ctx, text) {
  const t = collapse(text);
  const v = ctx.consts.get(t);
  if (v !== undefined && v !== t) return `${humanName(t)} (pin ${v})`;
  if (/^\d+$/.test(t)) return `pin ${t}`;
  if (/^A\d+$/.test(t)) return `pin ${t}`;
  return t;
}

export const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

export function duration(ms) {
  const t = collapse(ms);
  if (!/^\d+(\.\d+)?$/.test(t)) return `${t} milliseconds`;
  const n = Number(t);
  if (n >= 1000 && n % 100 === 0) return `${n / 1000} second${n === 1000 ? '' : 's'}`;
  return `${n} millisecond${n === 1 ? '' : 's'}`;
}

/** C condition -> English: `digitalRead(btn) == HIGH` -> "button (pin 2) is ON", `a && b` -> "a and b". */
export function plainCondition(text, ctx) {
  let s = collapse(text);
  s = s.replace(/!\s*digitalRead\(([^()]*)\)/g, (_, p) => `${thing(ctx, p)} is OFF`);
  s = s.replace(/digitalRead\(([^()]*)\)\s*==\s*(HIGH|LOW|1|0)/g, (_, p, v) => `${thing(ctx, p)} is ${v === 'HIGH' || v === '1' ? 'ON' : 'OFF'}`);
  s = s.replace(/digitalRead\(([^()]*)\)\s*!=\s*(HIGH|LOW)/g, (_, p, v) => `${thing(ctx, p)} is ${v === 'HIGH' ? 'OFF' : 'ON'}`);
  s = s.replace(/digitalRead\(([^()]*)\)/g, (_, p) => `${thing(ctx, p)} is ON`);
  s = s.replace(/analogRead\(([^()]*)\)/g, (_, p) => `the reading of ${thing(ctx, p)}`);
  s = s.replace(/&&/g, ' and ').replace(/\|\|/g, ' or ');
  s = s.replace(/>=/g, ' is at least ').replace(/<=/g, ' is at most ').replace(/!=/g, ' is not ').replace(/==/g, ' is ');
  s = s.replace(/(?<![-<>=!])<(?![<=])/g, ' is less than ').replace(/(?<![-<>=!])>(?![>=])/g, ' is more than ');
  s = s.replace(/(^|[\s(])!\s*([A-Za-z_(])/g, '$1not $2');
  s = s.replace(/\bHIGH\b/g, 'ON').replace(/\bLOW\b/g, 'OFF').replace(/\btrue\b/g, 'true').replace(/\s+/g, ' ').trim();
  return s;
}

/** Recognise `for (i = A; i < N; i++)` style counting loops. Returns null when the pattern is not simple. */
export function countingLoop(initText, cond, stepText) {
  const init = /(?:\w+\s+)?(\w+)\s*=\s*(-?\d+)\s*;?$/.exec(collapse(initText || ''));
  const c = /^(\w+)\s*(<=|<|>=|>)\s*(-?\d+|\w+)$/.exec(collapse(cond || ''));
  const st = /^(\w+)\s*(\+\+|--|\+=\s*(\d+)|-=\s*(\d+))$|^(\+\+|--)(\w+)$/.exec(collapse(stepText || ''));
  if (!init || !c || !st || init[1] !== c[1]) return null;
  const v = init[1];
  const from = Number(init[2]);
  const stepVar = st[1] || st[6];
  if (stepVar !== v) return null;
  const up = (st[2] || st[5] || '').startsWith('+') || st[5] === '++';
  const by = Number(st[3] || st[4] || 1);
  const bound = c[3];
  if (!/^-?\d+$/.test(bound)) return { v, from, to: bound, up, by, times: null };
  const to = Number(bound);
  const incl = c[2].includes('=');
  const span = up ? to - from + (incl ? 1 : 0) : from - to + (incl ? 1 : 0);
  return { v, from, to, up, by, times: span > 0 ? Math.ceil(span / by) : 0 };
}

export const EXPLAIN = {
  start: 'Every program starts somewhere. This block shows where a group of instructions begins.',
  'pin-config': 'Before using a pin, tell the board what it should do: send signals OUT (like to an LED) or listen for signals IN (like from a button).',
  output: 'A pin can be ON (HIGH, 5 volts) or OFF (LOW, 0 volts). This block flips it — like flicking a light switch.',
  'output:PWM out': 'analogWrite makes a pin flick on and off super fast. 0 is off, 255 is full power, and numbers in between make things dimmer or slower.',
  'output:Buzzer': 'This makes the buzzer vibrate at a pitch. A higher number of Hertz (Hz) makes a higher sound.',
  input: 'The board can sense the world! This block reads a pin and remembers the answer in a variable.',
  'input:Digital in': 'digitalRead asks "is this pin ON or OFF?" — perfect for buttons and switches.',
  'input:Analog in': 'analogRead measures a changing voltage from 0 to 1023 — great for light sensors, knobs and temperature.',
  wait: 'The board pauses and does nothing else while it waits. 1000 milliseconds is 1 second.',
  serial: 'The Serial Monitor is a text window on your computer. The board can send messages there so you can see what is happening inside it.',
  bus: 'Some chips (screens, sensors) talk to the board over just two wires. These blocks send and receive that conversation.',
  call: 'This runs another group of instructions that you wrote (a function), then comes back here and carries on.',
  trigger: 'An interrupt is like a doorbell. When the pin changes, the board drops what it is doing, runs that function right away, then goes back.',
  callback: 'This tells a library: "when the time comes, run this function for me".',
  library: 'A library is code someone else wrote so you do not have to. This block uses one of its gadgets.',
  declare: 'A variable is a labelled box that remembers a value. You can look inside it or change it later.',
  process: 'This block changes a variable — it remembers a new value (or does some maths).',
  decision: 'The board asks a yes/no question. If the answer is YES it follows the green arrow, otherwise the red one.',
  loop: 'A loop repeats the blocks inside it. The board keeps going round until the question says stop.',
  switch: 'Like a signpost with many roads: the board looks at one value and takes the road that matches.',
  return: 'This finishes the block of instructions and (maybe) hands an answer back to whoever asked.',
  break: 'Stop the loop right now and carry on after it.',
  continue: 'Skip the rest of this round and start the next one.',
  end: 'The end of this group of instructions.',
  globals: 'These are things the whole program knows about: libraries it uses, names for pins, and variables everyone can see.',
};
