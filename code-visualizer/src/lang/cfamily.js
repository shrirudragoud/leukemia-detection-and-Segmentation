// tree-sitter C / C++ concrete syntax tree  ->  language-neutral IR.
// Arduino sketches are C++, so this adapter serves .ino today and plain C later; the
// API-specific meaning of calls comes from the injected `profile`.

import { collapse, truncate } from '../core/text.js';
import { countingLoop } from '../profiles/plain.js';

const loc = (n) => ({ start: n.startPosition.row + 1, end: n.endPosition.row + 1 });
const COMMENT = (n) => n.type === 'comment';

function stripParens(n) {
  if (!n) return '';
  if (n.type === 'condition_clause') return stripParens(n.childForFieldName('value') ?? n.namedChild(0));
  if (n.type === 'parenthesized_expression') return collapse(n.namedChild(0)?.text ?? '');
  return collapse(n.text);
}

function declaratorName(d) {
  while (d && !['identifier', 'field_identifier', 'qualified_identifier'].includes(d.type)) {
    d = d.childForFieldName('declarator') ?? d.namedChildren.find((c) => !COMMENT(c));
  }
  return d?.text ?? null;
}

function parseCall(n) {
  const fn = n.childForFieldName('function');
  const args = (n.childForFieldName('arguments')?.namedChildren ?? []).filter((c) => !COMMENT(c)).map((c) => collapse(c.text));
  if (fn?.type === 'field_expression') {
    const object = collapse(fn.childForFieldName('argument')?.text ?? '');
    const method = fn.childForFieldName('field')?.text ?? '';
    return { name: `${object}.${method}`, object, method, args };
  }
  return { name: collapse(fn?.text ?? ''), object: null, method: null, args };
}

export function extractProgram(root, source, profile) {
  const program = { language: profile.id, globals: [], functions: [], diagnostics: [] };
  const consts = new Map();
  const objectTypes = new Map();
  const userFns = new Set();
  const ctx = { consts, objectTypes, userFns };
  const plain = (t) => (profile.plainCondition ? profile.plainCondition(t, ctx) : t);
  const fnNodes = [];

  // ---- pass 1: top level -------------------------------------------------------------------
  const visitTop = (node) => {
    for (const c of node.namedChildren) {
      switch (c.type) {
        case 'function_definition': {
          const decl = findFunctionDeclarator(c.childForFieldName('declarator'));
          const name = declaratorName(decl?.childForFieldName('declarator'));
          if (name) { userFns.add(name); fnNodes.push({ node: c, decl, name }); }
          break;
        }
        case 'preproc_include':
          program.globals.push({ kind: 'include', text: `#include ${c.childForFieldName('path')?.text ?? ''}`, line: loc(c).start });
          break;
        case 'preproc_def': {
          const name = c.childForFieldName('name')?.text;
          const value = collapse(c.childForFieldName('value')?.text ?? '');
          if (name && value) consts.set(name, value);
          program.globals.push({ kind: 'define', text: truncate(`#define ${name} ${value}`, 48), line: loc(c).start });
          break;
        }
        case 'preproc_function_def':
          program.globals.push({ kind: 'macro', text: truncate(collapse(c.text), 48), line: loc(c).start });
          break;
        case 'declaration':
          topDeclaration(c);
          break;
        case 'preproc_if': case 'preproc_ifdef': case 'preproc_elif': case 'preproc_else':
        case 'namespace_definition': case 'declaration_list': case 'linkage_specification':
          visitTop(c);
          break;
        default:
      }
    }
  };

  function topDeclaration(c) {
    const type = collapse(c.childForFieldName('type')?.text ?? '');
    const isConst = c.namedChildren.some((x) => x.type === 'type_qualifier' && x.text === 'const');
    for (const d of c.namedChildren) {
      if (!['init_declarator', 'identifier', 'array_declarator', 'pointer_declarator', 'function_declarator'].includes(d.type)) continue;
      if (d.type === 'function_declarator' && !d.childForFieldName('declarator')?.text.match(/^\w+$/)) continue;
      const name = declaratorName(d);
      if (!name) continue;
      const value = d.type === 'init_declarator' ? d.childForFieldName('value') : null;
      if (d.type === 'function_declarator') {
        // `Foo bar(1,2);` is parsed as a prototype; a type_identifier return type means a constructed object.
        if (c.childForFieldName('type')?.type === 'type_identifier') objectTypes.set(name, type);
        else continue; // genuine prototype
      } else if (c.childForFieldName('type')?.type === 'type_identifier') {
        objectTypes.set(name, type);
      }
      if (value && /^-?\d+(\.\d+)?$|^0x[0-9a-f]+$|^A\d+$/i.test(collapse(value.text))) consts.set(name, collapse(value.text));
      const initText = value ? collapse(value.text) : null;
      const friendly = objectTypes.get(name) ? `${name} is a ${type}` : initText ? `${name} ${isConst ? 'is' : 'starts as'} ${initText}` : `${name} is a variable`;
      program.globals.push({ kind: 'var', text: truncate(collapse(c.text).replace(/;$/, ''), 48) + ';', friendly, line: loc(c).start });
      return;
    }
  }

  visitTop(root);

  // ---- pass 2: statements ------------------------------------------------------------------
  for (const { node, decl, name } of fnNodes) {
    const params = (decl.childForFieldName('parameters')?.namedChildren ?? []).map((p) => collapse(p.text));
    const returnType = collapse(node.childForFieldName('type')?.text ?? 'void');
    const body = node.childForFieldName('body');
    program.functions.push({
      name,
      params,
      returnType,
      signature: `${returnType} ${name}(${params.join(', ')})`,
      body: body ? stmts(body) : [],
      loc: loc(node),
      role: 'user',
    });
  }

  for (const g of program.globals) g.friendly ??= profile.describeGlobal?.(g) ?? g.text;
  collectErrors(root, program.diagnostics);
  return program;

  // ---- statement conversion ----------------------------------------------------------------
  function stmts(node) {
    if (!node) return [];
    switch (node.type) {
      case 'compound_statement':
        return node.namedChildren.filter((c) => !COMMENT(c)).flatMap(stmts);
      case 'comment': case 'empty_statement':
        return [];
      case 'preproc_ifdef': case 'preproc_if': {
        // Take the primary branch only: inlining #else would show both variants executing.
        const skip = new Set([node.childForFieldName('name')?.id, node.childForFieldName('condition')?.id]);
        const tag = `#${node.type === 'preproc_ifdef' ? (node.text.startsWith('#ifndef') ? 'ifndef' : 'ifdef') : 'if'} ${collapse(
          (node.childForFieldName('name') ?? node.childForFieldName('condition'))?.text ?? '')}`;
        return node.namedChildren
          .filter((c) => !skip.has(c.id) && !COMMENT(c) && !['preproc_else', 'preproc_elif'].includes(c.type))
          .flatMap(stmts)
          .map((s) => (s.t === 'action' && !s.sub ? { ...s, sub: tag } : s));
      }
      case 'labeled_statement':
        return node.namedChildren.filter((c) => c.type !== 'statement_identifier').flatMap(stmts);
      case 'if_statement': return [ifStmt(node)];
      case 'for_statement': return forStmt(node);
      case 'for_range_loop': return [rangeStmt(node)];
      case 'while_statement': return [whileStmt(node)];
      case 'do_statement': return [doStmt(node)];
      case 'switch_statement': return [switchStmt(node)];
      case 'return_statement': return [{ t: 'return', value: collapse(node.namedChild(0)?.text ?? ''), loc: loc(node) }];
      case 'break_statement': return [{ t: 'break', loc: loc(node) }];
      case 'continue_statement': return [{ t: 'continue', loc: loc(node) }];
      case 'ERROR':
        return [{ t: 'action', kind: 'process', category: 'Oops', title: truncate(node.text, 40), friendly: `Something is wrong here: ${truncate(node.text, 30)}`, text: collapse(node.text), error: 'Syntax error', loc: loc(node) }];
      default:
        return [action(node)];
    }
  }

  function ifStmt(n) {
    const cond = n.childForFieldName('condition');
    const alt = n.childForFieldName('alternative');
    const altBody = alt ? (alt.type === 'else_clause' ? alt.namedChildren.filter((c) => !COMMENT(c)).flatMap(stmts) : stmts(alt)) : [];
    return {
      t: 'if',
      cond: stripParens(cond),
      friendly: `${plain(stripParens(cond))}?`,
      then: stmts(n.childForFieldName('consequence')),
      else: altBody,
      loc: loc(n),
      headLoc: { start: loc(n).start, end: cond ? loc(cond).end : loc(n).start },
    };
  }

  function whileStmt(n) {
    const cond = n.childForFieldName('condition');
    const body = stmts(n.childForFieldName('body'));
    const c = stripParens(cond);
    if (!body.length) {
      return { t: 'action', kind: 'wait', category: 'Wait', title: `Wait while ${c}`, friendly: `Wait here while ${plain(c)}`, text: collapse(n.text), loc: loc(n), blocking: true };
    }
    const forever = /^(true|1)$/.test(c);
    return {
      t: 'loop', form: 'while', cond: c, body, loc: loc(n),
      friendly: forever ? 'Repeat forever' : `Repeat while ${plain(c)}`,
      headLoc: { start: loc(n).start, end: cond ? loc(cond).end : loc(n).start },
    };
  }

  function doStmt(n) {
    const cond = n.childForFieldName('condition');
    return {
      t: 'loop', form: 'do', cond: stripParens(cond), body: stmts(n.childForFieldName('body')), loc: loc(n),
      friendly: `Again, while ${plain(stripParens(cond))}`, friendlySub: 'do it once first, then check',
      headLoc: cond ? loc(cond) : loc(n),
    };
  }

  function forStmt(n) {
    const init = n.childForFieldName('initializer');
    const cond = n.childForFieldName('condition');
    const upd = n.childForFieldName('update');
    const initStmts = init ? [{ ...action(init), category: 'Init', role: 'loop-init' }] : [];
    const body = stmts(n.childForFieldName('body'));
    if (upd) body.push({ ...action(upd), category: 'Step', role: 'loop-step' });
    const condText = cond ? collapse(cond.text) : 'forever';
    const count = init && cond && upd ? countingLoop(init.text, condText, upd.text) : null;
    let friendly = condText === 'forever' ? 'Repeat forever' : `Repeat while ${plain(condText)}`;
    let friendlySub = null;
    if (count) {
      const last = count.times != null ? count.from + (count.up ? 1 : -1) * (count.times - 1) * count.by : null;
      friendly = count.times != null ? `Repeat ${count.times} time${count.times === 1 ? '' : 's'}` : `Count ${count.v} ${count.up ? 'up' : 'down'} to ${count.to}`;
      friendlySub = count.times != null ? `${count.v} counts ${count.from} … ${last}${count.by > 1 ? ` (by ${count.by})` : ''}` : `starting at ${count.from}`;
    }
    return [
      ...initStmts,
      {
        t: 'loop',
        form: 'for',
        cond: condText,
        body,
        count: !!count,
        friendly,
        friendlySub,
        loc: loc(n),
        headLoc: { start: loc(n).start, end: loc(n).start },
      },
    ];
  }

  function rangeStmt(n) {
    const v = collapse(n.childForFieldName('declarator')?.text ?? '?');
    const r = collapse(n.childForFieldName('right')?.text ?? '?');
    return {
      t: 'loop', form: 'range', cond: `each ${v} in ${r}`, friendly: `For each ${v} in ${r}`, body: stmts(n.childForFieldName('body')),
      loc: loc(n), headLoc: { start: loc(n).start, end: loc(n).start },
    };
  }

  function switchStmt(n) {
    const cond = n.childForFieldName('condition');
    const cases = [];
    let pending = [];
    const kids = (n.childForFieldName('body')?.namedChildren ?? []).filter((c) => c.type === 'case_statement');
    kids.forEach((cs, i) => {
      const value = cs.childForFieldName('value');
      const isDefault = !value;
      pending.push(isDefault ? 'default' : collapse(value.text));
      const body = cs.namedChildren.filter((c) => c.id !== value?.id && !COMMENT(c)).flatMap(stmts);
      if (!body.length && i < kids.length - 1) return; // `case 1: case 2:` share the next body
      const trailingBreak = body.length && body[body.length - 1].t === 'break';
      if (trailingBreak) body.pop();
      const nice = pending.includes('default') ? 'anything else' : `is ${pending.join(' or ')}`;
      cases.push({ labels: pending, friendlyLabel: nice, isDefault: pending.includes('default'), body, fallsThrough: !trailingBreak && !endsTerminal(body) && i < kids.length - 1 });
      pending = [];
    });
    return { t: 'switch', expr: stripParens(cond), friendly: `Look at ${plain(stripParens(cond))}`, cases, loc: loc(n), headLoc: { start: loc(n).start, end: cond ? loc(cond).end : loc(n).start } };
  }

  function endsTerminal(body) {
    return body.length > 0 && ['return', 'continue', 'break'].includes(body[body.length - 1].t);
  }

  // ---- simple statements -------------------------------------------------------------------
  function action(node) {
    const info = analyse(node);
    const text = collapse(node.text);
    const base = { t: 'action', text, loc: loc(node) };
    const save = info.target ? `, and remember it as “${info.target}”` : '';

    const d = profile.describeStatement(info, ctx);
    if (d) return { ...base, ...d, ref: d.ref ?? null };

    // user function call / callback registration
    for (const call of info.calls) {
      if (userFns.has(call.name)) {
        const args = truncate(call.args.join(', '), 24);
        return {
          ...base, kind: 'call', category: 'My block', title: `${call.name}(${args})`,
          friendly: `Run my block “${call.name}”${call.args.length ? ` using ${args}` : ''}${save}`,
          sub: info.target ? `→ ${info.target}` : undefined, ref: call.name,
        };
      }
      const cb = call.args.find((a) => userFns.has(a));
      if (cb) {
        return {
          ...base, kind: 'callback', category: 'Callback', title: `${call.name}(…, ${cb})`, friendly: `Ask ${call.object || call.name} to run ${cb}() later`,
          sub: `registers ${cb}()`, ref: cb, refRole: 'callback',
        };
      }
    }
    const first = info.calls[0];
    if (first?.object) {
      const type = objectTypes.get(first.object);
      return {
        ...base, kind: 'library', category: type || first.object, title: truncate(`${first.method}(${first.args.join(', ')})`, 40),
        friendly: `Tell ${first.object} to ${first.method}${first.args.length ? ` ${truncate(first.args.join(', '), 24)}` : ''}${save}`,
        sub: info.target ? `→ ${info.target}` : undefined,
      };
    }
    if (first && info.exprIsBareCall) {
      return { ...base, kind: 'library', category: 'Call', title: truncate(`${first.name}(${first.args.join(', ')})`, 44), friendly: `Run ${first.name}(${truncate(first.args.join(', '), 20)})${save}` };
    }
    if (info.isDecl && !info.calls.length) {
      const friendly = info.exprText ? `Make a variable “${info.target}” and set it to ${plain(info.exprText)}` : `Make a variable “${info.target ?? '?'}”`;
      return { ...base, kind: 'declare', category: 'Variable', title: truncate(text.replace(/;$/, ''), 44), friendly };
    }
    return { ...base, kind: 'process', category: 'Do', title: truncate(text.replace(/;$/, ''), 44), friendly: friendlyProcess(node, info) };
  }

  function friendlyProcess(node, info) {
    const e = node.type === 'expression_statement' ? node.namedChild(0) : node;
    if (e?.type === 'update_expression') {
      const arg = collapse(e.childForFieldName('argument')?.text ?? '');
      return e.text.includes('++') ? `Add 1 to ${arg}` : `Take 1 away from ${arg}`;
    }
    if (e?.type === 'assignment_expression') {
      const op = e.childForFieldName('operator')?.text;
      const l = collapse(e.childForFieldName('left')?.text ?? '');
      const r = collapse(e.childForFieldName('right')?.text ?? '');
      switch (op) {
        case '+=': return `Add ${r} to ${l}`;
        case '-=': return `Take ${r} away from ${l}`;
        case '*=': return `Multiply ${l} by ${r}`;
        case '/=': return `Divide ${l} by ${r}`;
        default: return r === `!${l}` ? `Flip “${l}” (ON ↔ OFF)` : `Set ${l} to ${plain(r)}`;
      }
    }
    return truncate(collapse(node.text).replace(/;$/, ''), 44);
  }

  function analyse(node) {
    let expr = node;
    let target = null;
    let isDecl = false;
    if (node.type === 'expression_statement') expr = node.namedChild(0);
    if (node.type === 'declaration') {
      isDecl = true;
      const d = node.namedChildren.find((c) => c.type === 'init_declarator');
      if (d) { target = declaratorName(d.childForFieldName('declarator')); expr = d.childForFieldName('value'); }
      else { target = declaratorName(node.namedChildren.find((c) => c.id !== node.childForFieldName('type')?.id && !COMMENT(c))); expr = null; }
    } else if (expr?.type === 'assignment_expression' && expr.childForFieldName('operator')?.text === '=') {
      target = collapse(expr.childForFieldName('left')?.text ?? '');
      expr = expr.childForFieldName('right');
    }
    const calls = [];
    if (expr) {
      if (expr.type === 'call_expression') calls.push(expr);
      calls.push(...expr.descendantsOfType('call_expression'));
    }
    return {
      calls: calls.map(parseCall),
      target,
      isDecl,
      exprText: collapse(expr?.text ?? ''),
      exprIsBareCall: expr?.type === 'call_expression',
    };
  }
}

function findFunctionDeclarator(d) {
  while (d && d.type !== 'function_declarator') d = d.childForFieldName('declarator');
  return d;
}

function collectErrors(root, out) {
  if (!root.hasError) return;
  const visit = (n) => {
    if (!n.hasError && !n.isMissing) return;
    if (n.isMissing) {
      out.push({ severity: 'error', line: n.startPosition.row + 1, column: n.startPosition.column + 1, message: `Missing "${n.type}"` });
      return;
    }
    if (n.type === 'ERROR') {
      out.push({ severity: 'error', line: n.startPosition.row + 1, column: n.startPosition.column + 1, message: `Syntax error near "${truncate(n.text, 30)}"` });
      return;
    }
    for (const c of n.children) visit(c);
  };
  visit(root);
}
