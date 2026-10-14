import { walkStmts } from '../core/ir.js';

/** Assign function roles (setup/loop/isr/callback/user) from the profile and from registration calls. */
export function resolveRoles(program, profile) {
  const byName = new Map(program.functions.map((f) => [f.name, f]));
  for (const [name, entry] of Object.entries(profile.entryPoints || {})) {
    const f = byName.get(name);
    if (f) Object.assign(f, { role: entry.role, description: entry.description, startLabel: entry.startLabel, repeats: !!entry.repeats, endTitle: entry.endTitle });
  }
  for (const f of program.functions) {
    walkStmts(f.body, (s) => {
      if (s.t !== 'action' || !s.ref) return;
      const target = byName.get(s.ref);
      if (!target) return;
      (target.calledBy ??= []).push(f.name);
      if (s.refRole && target.role === 'user') {
        target.role = s.refRole;
        target.description = s.refRole === 'isr' ? 'Jumps in RIGHT AWAY when something happens' : 'Runs later, when a library asks for it';
        target.startLabel = s.refRole === 'isr' ? s.when || 'When the doorbell rings' : 'When it is time';
        target.startEmoji = '⚡';
      }
    });
  }
  for (const f of program.functions) {
    if (f.role === 'user') {
      f.description = f.calledBy?.length ? `My own block, used by ${[...new Set(f.calledBy)].join(', ')}` : 'My own block';
      f.startLabel = 'When this block is used';
      f.startEmoji = '🧩';
    }
  }
  return program;
}

const GROUPS = {
  'pin-config': { category: 'Get ready', title: (n) => `Configure ${n} pins`, friendly: () => 'Get the pins ready' },
  declare: { category: 'Variables', title: (n) => `Declare ${n} variables`, friendly: (n) => `Make ${n} variables` },
};

/** Merge runs of adjacent pinMode(...) calls / plain declarations into one multi-line card. */
export function groupStatements(stmts, groupKinds = ['pin-config']) {
  const kinds = new Set([...groupKinds, 'declare']);
  const out = [];
  for (let i = 0; i < stmts.length; ) {
    const s = stmts[i];
    if (s.t === 'action' && kinds.has(s.kind) && !s.error) {
      let j = i;
      while (j < stmts.length && stmts[j].t === 'action' && stmts[j].kind === s.kind && !stmts[j].error) j++;
      const run = stmts.slice(i, j);
      if (run.length > 1) {
        const g = GROUPS[s.kind];
        out.push({
          t: 'action', kind: s.kind, category: g.category, title: g.title(run.length), friendly: g.friendly(run.length),
          lines: run.map((r) => r.title), friendlyLines: run.map((r) => r.shortFriendly || r.friendly || r.title), text: run.map((r) => r.text).join(' '),
          loc: { start: run[0].loc.start, end: run[run.length - 1].loc.end },
        });
        i = j;
        continue;
      }
    }
    out.push(s);
    i++;
  }
  return out;
}

export function groupProgram(program, profile) {
  const rec = (list) => {
    const grouped = groupStatements(list, profile.groupKinds);
    for (const s of grouped) {
      if (s.t === 'if') { s.then = rec(s.then); s.else = rec(s.else); }
      else if (s.t === 'loop') s.body = rec(s.body);
      else if (s.t === 'switch') for (const c of s.cases) c.body = rec(c.body);
    }
    return grouped;
  };
  for (const f of program.functions) f.body = rec(f.body);
  return program;
}
