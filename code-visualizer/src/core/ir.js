// Language-neutral intermediate representation.
//
// Program  { language, globals: Global[], functions: Func[], diagnostics: Diag[] }
// Func     { name, signature, params, returnType, body: Stmt[], loc, role, startLabel, ... }
// Stmt     = action | if | loop | switch | return | break | continue
//   action   { t:'action', kind, title, sub?, lines?, ref?, text, loc, warn?, error? }
//   if       { t:'if', cond, then: Stmt[], else: Stmt[], loc, headLoc }
//   loop     { t:'loop', form:'while'|'for'|'do'|'range', cond, body, loc, headLoc }
//   switch   { t:'switch', expr, cases:[{labels, isDefault, body}], loc, headLoc }
//   return   { t:'return', value?, loc }   break / continue { t:'break'|'continue', loc }
// loc = { start, end }  (1-based inclusive source lines)

export function walkStmts(stmts, fn) {
  for (const s of stmts || []) {
    fn(s);
    if (s.t === 'if') { walkStmts(s.then, fn); walkStmts(s.else, fn); }
    else if (s.t === 'loop') walkStmts(s.body, fn);
    else if (s.t === 'switch') for (const c of s.cases) walkStmts(c.body, fn);
  }
}
