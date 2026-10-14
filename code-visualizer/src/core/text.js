export const collapse = (s) => (s || '').replace(/\s+/g, ' ').trim();

export function truncate(s, n) {
  s = collapse(s);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

/** Greedy word wrap (breaking over-long tokens) to at most `maxLines` lines, ellipsising overflow. */
export function wrap(text, maxChars, maxLines) {
  const words = collapse(text).split(' ').filter(Boolean);
  const lines = [];
  let cur = '';
  for (let w of words) {
    while (w.length > maxChars) {
      if (cur) { lines.push(cur); cur = ''; }
      lines.push(w.slice(0, maxChars));
      w = w.slice(maxChars);
    }
    if (!cur) cur = w;
    else if ((cur + ' ' + w).length <= maxChars) cur += ' ' + w;
    else { lines.push(cur); cur = w; }
  }
  if (cur) lines.push(cur);
  if (lines.length > maxLines) {
    const kept = lines.slice(0, maxLines);
    const last = kept[maxLines - 1];
    kept[maxLines - 1] = last.length >= maxChars ? last.slice(0, maxChars - 1) + '…' : last + '…';
    return kept;
  }
  return lines;
}

export const textWidth = (s, px = 13) => Math.ceil(s.length * px * 0.56);
