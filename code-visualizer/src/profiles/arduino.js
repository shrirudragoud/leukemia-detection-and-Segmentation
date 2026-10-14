// Arduino API knowledge: turns recognised calls into meaningful, kid-readable flowchart blocks.
// A profile is pure data + small functions, so MicroPython / ESP-IDF / STM32 HAL profiles can be
// added next to this file without touching the parser, layout or renderer.
//
// Every descriptor carries:  kind, category, title (technical) and friendly (plain English).

import { collapse, truncate } from '../core/text.js';
import { walkStmts } from '../core/ir.js';
import { thing, cap, duration, plainCondition } from './plain.js';

const SERIAL = /^Serial\d?$|^SerialUSB$|^Serial_$/;

function pin(ctx, text) {
  const t = collapse(text);
  const v = ctx.consts.get(t);
  return v !== undefined && v !== t ? `${t} (${v})` : t;
}

function interruptPin(ctx, text) {
  const m = /^digitalPinToInterrupt\((.*)\)$/.exec(collapse(text));
  return m ? m[1] : text;
}

const EDGE = { RISING: 'turns ON', FALLING: 'turns OFF', CHANGE: 'changes', LOW: 'is OFF', HIGH: 'is ON' };

const FREE = {
  pinMode: (a, ctx) => {
    const mode = a[1] ?? '?';
    const says = mode === 'OUTPUT' ? 'sends signals out' : mode === 'INPUT_PULLUP' ? 'listens (button-style, with pull-up)' : mode === 'INPUT' ? 'listens for signals' : `works as ${mode}`;
    const short = mode === 'OUTPUT' ? 'sends out' : mode === 'INPUT_PULLUP' || mode === 'INPUT' ? 'listens' : mode;
    return {
      kind: 'pin-config',
      title: `${pin(ctx, a[0])} → ${mode}`,
      friendly: cap(`${thing(ctx, a[0])} ${says}`),
      shortFriendly: cap(`${thing(ctx, a[0])} ${short}`),
    };
  },
  digitalWrite: (a, ctx) => ({
    kind: 'output',
    category: 'Digital out',
    title: /^(HIGH|LOW)$/.test(a[1]) ? `Set ${pin(ctx, a[0])} ${a[1]}` : `Write ${a[1]} to ${pin(ctx, a[0])}`,
    friendly: a[1] === 'HIGH' ? `Turn ON ${thing(ctx, a[0])}` : a[1] === 'LOW' ? `Turn OFF ${thing(ctx, a[0])}` : `Set ${thing(ctx, a[0])} to ${a[1]}`,
  }),
  analogWrite: (a, ctx) => ({
    kind: 'output', category: 'PWM out',
    title: `${pin(ctx, a[0])} PWM = ${a[1]}`,
    friendly: `Set power of ${thing(ctx, a[0])} to ${a[1]} (0 to 255)`,
  }),
  digitalRead: (a, ctx, i) => ({
    kind: 'input', category: 'Digital in', title: `Read ${pin(ctx, a[0])}`,
    friendly: `Check if ${thing(ctx, a[0])} is ON or OFF${i.save}`,
  }),
  analogRead: (a, ctx, i) => ({
    kind: 'input', category: 'Analog in', title: `Read ${pin(ctx, a[0])}`,
    friendly: `Measure ${thing(ctx, a[0])} (0 to 1023)${i.save}`,
  }),
  pulseIn: (a, ctx, i) => ({
    kind: 'input', category: 'Pulse in', title: `Measure pulse on ${pin(ctx, a[0])}`,
    friendly: `Time how long ${thing(ctx, a[0])} stays ${a[1] === 'LOW' ? 'OFF' : 'ON'}${i.save}`,
  }),
  delay: (a) => ({ kind: 'wait', category: 'Wait', title: `Wait ${a[0]} ms`, friendly: `Wait ${duration(a[0])}`, blocking: true }),
  delayMicroseconds: (a) => ({ kind: 'wait', category: 'Wait', title: `Wait ${a[0]} µs`, friendly: `Wait ${a[0]} microseconds (super short!)`, blocking: true }),
  tone: (a, ctx) => ({
    kind: 'output', category: 'Buzzer', title: `Tone ${a[1] ?? ''} Hz on ${pin(ctx, a[0])}`.replace('  ', ' '),
    friendly: `Play a ${a[1] ?? ''} Hz sound on ${thing(ctx, a[0])}`.replace('  ', ' '),
  }),
  noTone: (a, ctx) => ({ kind: 'output', category: 'Buzzer', title: `Silence ${pin(ctx, a[0])}`, friendly: `Stop the sound on ${thing(ctx, a[0])}` }),
  shiftOut: (a, ctx) => ({ kind: 'output', category: 'Shift out', title: `Shift ${a[3] ?? ''} out on ${pin(ctx, a[0])}`, friendly: `Send ${a[3] ?? 'a number'} out, one bit at a time, on ${thing(ctx, a[0])}` }),
  attachInterrupt: (a, ctx) => {
    const fn = collapse(a[1] ?? '?');
    const edge = EDGE[a[2]] || a[2] || 'changes';
    const what = thing(ctx, interruptPin(ctx, a[0]));
    return {
      kind: 'trigger',
      category: 'Interrupt',
      title: `${pin(ctx, interruptPin(ctx, a[0]))} ${a[2] ?? ''} → ${fn}()`.replace('  ', ' '),
      friendly: `Watch ${what}. When it ${edge}, run ${fn}() right away`,
      when: `When ${what} ${edge}`,
      ref: ctx.userFns.has(a[1]) ? a[1] : null,
      refRole: 'isr',
    };
  },
  detachInterrupt: (a, ctx) => ({ kind: 'trigger', category: 'Interrupt', title: `Detach ${pin(ctx, interruptPin(ctx, a[0]))}`, friendly: `Stop watching ${thing(ctx, interruptPin(ctx, a[0]))}` }),
  noInterrupts: () => ({ kind: 'trigger', category: 'Interrupt', title: 'Disable interrupts', friendly: 'Ignore all doorbells for a moment' }),
  interrupts: () => ({ kind: 'trigger', category: 'Interrupt', title: 'Enable interrupts', friendly: 'Listen for doorbells again' }),
};

const SERIAL_METHODS = {
  begin: (a) => ({ title: `Start @ ${a[0] ?? ''} baud`, friendly: `Connect to the computer (speed ${a[0] ?? '?'})` }),
  print: (a) => ({ title: `Print ${a.join(', ')}`, friendly: `Show ${a.join(' and ')} on the computer` }),
  println: (a) => ({ title: a.length ? `Print ${a.join(', ')}` : 'Print newline', friendly: a.length ? `Show ${a.join(' and ')} on the computer` : 'Start a new line on the computer' }),
  printf: (a) => ({ title: `Print ${a.join(', ')}`, friendly: `Show ${a.join(' and ')} on the computer` }),
  write: (a) => ({ title: `Write ${a.join(', ')}`, friendly: `Send ${a.join(' and ')} to the computer` }),
  read: () => ({ title: 'Read byte', friendly: 'Read a letter from the computer' }),
  readString: () => ({ title: 'Read string', friendly: 'Read a message from the computer' }),
  readStringUntil: (a) => ({ title: `Read until ${a[0]}`, friendly: `Read a message until ${a[0]}` }),
  parseInt: () => ({ title: 'Parse integer', friendly: 'Read a number from the computer' }),
  available: () => ({ title: 'Bytes available?', friendly: 'Has the computer sent anything?' }),
  flush: () => ({ title: 'Flush', friendly: 'Wait until everything is sent' }),
};

const WIRE_METHODS = {
  begin: () => ['Start I²C', 'Get ready to talk to chips (I²C)'],
  beginTransmission: (a) => [`Begin write to ${a[0]}`, `Start talking to the chip at address ${a[0]}`],
  endTransmission: () => ['End write', 'Finish talking'],
  write: (a) => [`Write ${a.join(', ')}`, `Send ${a.join(' and ')} to the chip`],
  requestFrom: (a) => [`Request ${a[1] ?? '?'} bytes from ${a[0]}`, `Ask the chip at ${a[0]} for ${a[1] ?? '?'} bytes of data`],
  read: () => ['Read byte', 'Read a byte from the chip'],
  available: () => ['Bytes available?', 'Has the chip sent anything?'],
};

const LIB_IDIOMS = {
  Servo: {
    attach: (a, ctx) => [`Attach to ${pin(ctx, a[0])}`, `Connect the servo to ${thing(ctx, a[0])}`],
    write: (a) => [`Move to ${a[0]}°`, `Turn the servo to ${a[0]}°`],
    writeMicroseconds: (a) => [`Pulse ${a[0]} µs`, `Send the servo a ${a[0]} µs pulse`],
    detach: () => ['Detach', 'Let the servo relax'],
  },
};

/** Describe one recognised call, or null when the profile does not know it. */
function describeCall(call, ctx, info) {
  const a = call.args;
  if (!call.object) return FREE[call.name] ? FREE[call.name](a, ctx, info) : null;

  if (SERIAL.test(call.object)) {
    const m = (SERIAL_METHODS[call.method] || (() => ({ title: `${call.method}(${a.join(', ')})`, friendly: `Serial: ${call.method}` })))(a);
    return { kind: 'serial', category: call.object === 'Serial' ? 'Talk to computer' : call.object, ...m };
  }
  if (call.object === 'Wire' || call.object === 'Wire1') {
    const [title, friendly] = WIRE_METHODS[call.method]?.(a) ?? [`${call.method}(${a.join(', ')})`, `Wire: ${call.method}`];
    return { kind: 'bus', category: 'I²C', title, friendly };
  }
  if (call.object === 'SPI') return { kind: 'bus', category: 'SPI', title: `${call.method}(${a.join(', ')})`, friendly: `SPI chip talk: ${call.method}` };

  const type = ctx.objectTypes.get(call.object);
  if (type) {
    const idiom = LIB_IDIOMS[type]?.[call.method]?.(a, ctx);
    return {
      kind: 'library',
      category: type,
      title: idiom?.[0] ?? `${call.object}.${call.method}(${a.join(', ')})`,
      friendly: idiom?.[1] ?? `Tell ${call.object} to ${call.method}${a.length ? ` ${a.join(', ')}` : ''}`,
      sub: idiom ? `${call.object}.${call.method}()` : undefined,
    };
  }
  return null;
}

export const arduino = {
  id: 'arduino',
  grammar: 'cpp',
  extensions: ['.ino', '.pde'],

  entryPoints: {
    setup: { role: 'setup', description: 'Runs ONE time when the board turns on', startLabel: 'When the board starts', startEmoji: '🏁', endTitle: 'Now go to loop()' },
    loop: { role: 'loop', description: 'Runs again and again, forever', startLabel: 'Forever, again and again', startEmoji: '🔁', repeats: true },
  },
  groupKinds: ['pin-config'],
  groupTitles: { 'pin-config': 'Get the pins ready' },

  plainCondition: (text, ctx) => plainCondition(text, ctx),

  describeStatement(info, ctx) {
    info.save = info.target ? `, and remember it as “${info.target}”` : '';
    for (const call of info.calls) {
      const d = describeCall(call, ctx, info);
      if (!d) continue;
      if (!d.sub && info.target && d.kind !== 'input') {
        d.sub = info.exprIsBareCall ? `→ ${info.target}` : truncate(`${info.target} = ${info.exprText}`, 40);
      }
      if (info.target && !/remember it/.test(d.friendly) && d.kind !== 'input' && d.kind !== 'pin-config') d.friendly += `, and remember it as “${info.target}”`;
      return d;
    }
    return null;
  },

  describeGlobal(g) {
    if (g.kind === 'include') return `Uses the ${g.text.replace(/#include\s*[<"]|\.h[>"]/g, '').trim()} library`;
    if (g.kind === 'define') { const m = /#define\s+(\w+)\s+(.*)/.exec(g.text); return m ? `${m[1]} means ${m[2]}` : g.text; }
    return g.text;
  },

  /** Post-pass over the IR: lint-style hints rendered as red markers on nodes. */
  annotate(program) {
    for (const fn of program.functions) {
      if (fn.role !== 'isr') continue;
      walkStmts(fn.body, (s) => {
        if (s.t !== 'action') return;
        if (s.blocking) s.warn = 'delay() blocks inside an interrupt handler — interrupt code should be super quick!';
        else if (s.kind === 'serial') s.warn = 'Serial printing is unsafe inside an interrupt handler — set a variable instead and print in loop().';
      });
    }
  },
};
