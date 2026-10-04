// Renders paper/content.json into a Word document that follows the Bates "How to Write a Paper in
// Scientific Journal Style and Format" guide: 12-pt Times, double spaced, centred capitalised headings,
// left bold-italic subheadings, Table legends ABOVE tables (no gridlines), Figure legends BELOW figures,
// tables/figures on their own pages, hanging-indent Literature Cited.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, Footer, PageNumber, LineRuleType, ShadingType, PageBreak,
} = require("docx");

const [, , inFile, outFile] = process.argv;
const content = JSON.parse(fs.readFileSync(inFile, "utf8"));
const FONT = "Times New Roman";
const DOUBLE = { line: 480, lineRule: LineRuleType.AUTO };
const SINGLE = { line: 240, lineRule: LineRuleType.AUTO };
const TEXT_W = 9360; // 6.5 in

// **bold**, *italic*, ^sup^ , ~sub~ , plain
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (tok.startsWith("^")) out.push(new TextRun({ text: tok.slice(1, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), subScript: true, font: FONT, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = { style: BorderStyle.SINGLE, size: 8, color: "000000" };

function tableBlock(b) {
  const total = b.cols.reduce((s, c) => s + c.w, 0);
  const widths = b.cols.map((c) => Math.round((c.w / total) * TEXT_W));
  widths[widths.length - 1] += TEXT_W - widths.reduce((s, w) => s + w, 0);
  const align = (a) => (a === "right" ? AlignmentType.RIGHT : a === "center" ? AlignmentType.CENTER : AlignmentType.LEFT);
  const cell = (txt, i, borders, bold) =>
    new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      borders: { top: borders.top || NONE, bottom: borders.bottom || NONE, left: NONE, right: NONE },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ alignment: align(b.cols[i].align), spacing: SINGLE,
        children: runs(String(txt), { size: 20, bold })})],
    });
  const header = new TableRow({ tableHeader: true,
    children: b.header.map((h, i) => cell(h, i, { top: RULE, bottom: RULE }, true)) });
  const body = b.rows.map((r, ri) => new TableRow({
    children: r.map((v, i) => cell(v, i, ri === b.rows.length - 1 ? { bottom: RULE } : {}, false)) }));
  const out = [
    new Paragraph({ pageBreakBefore: !String(b.num).startsWith("A"), keepNext: true, spacing: { ...SINGLE, before: String(b.num).startsWith("A") ? 240 : 0, after: 120 }, alignment: AlignmentType.LEFT,
      children: [new TextRun({ text: `Table ${b.num}. `, font: FONT, size: 22 }), ...runs(b.caption, { size: 22 })] }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows: [header, ...body] }),
  ];
  if (b.foot) out.push(new Paragraph({ spacing: { ...SINGLE, before: 120 }, children: runs(b.foot, { size: 20 }) }));
  return out;
}

function figureBlock(b) {
  const { w, h } = pngSize(b.path);
  const widthIn = b.width_in || 6.0;
  const maxH = 7.2;
  let wi = widthIn, hi = (widthIn * h) / w;
  if (hi > maxH) { hi = maxH; wi = (maxH * w) / h; }
  return [
    new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { ...SINGLE, after: 160 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.path),
        transformation: { width: Math.round(wi * 96), height: Math.round(hi * 96) },
        altText: { title: `Figure ${b.num}`, description: b.caption.slice(0, 200), name: `fig${b.num}` } })] }),
    new Paragraph({ spacing: SINGLE, alignment: AlignmentType.LEFT,
      children: [new TextRun({ text: `Figure ${b.num}. `, font: FONT, size: 22 }), ...runs(b.caption, { size: 22 })] }),
  ];
}

const children = [];
for (const b of content.blocks) {
  switch (b.t) {
    case "title":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { ...DOUBLE },
        children: runs(b.text, { bold: true, size: 28 }) }));
      break;
    case "authors":
      for (const l of b.lines)
        children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: SINGLE, children: runs(l) }));
      children.push(new Paragraph({ spacing: SINGLE, children: [] }));
      break;
    case "h":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { ...DOUBLE }, keepNext: true,
        children: runs(b.text.toUpperCase(), { bold: true }) }));
      break;
    case "sh":
      children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { ...DOUBLE }, keepNext: true,
        children: runs(b.text, { bold: true, italics: true }) }));
      break;
    case "p":
      children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { ...DOUBLE },
        indent: { firstLine: b.noindent ? 0 : 720 }, children: runs(b.text) }));
      break;
    case "ref":
      children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { ...SINGLE, after: 120 },
        indent: { left: 720, hanging: 720 }, children: runs(b.text) }));
      break;
    case "code":
      for (const line of b.lines)
        children.push(new Paragraph({ spacing: SINGLE, indent: { left: 360 },
          children: [new TextRun({ text: line, font: "Courier New", size: 18 })] }));
      children.push(new Paragraph({ spacing: SINGLE, children: [] }));
      break;
    case "small":
      children.push(new Paragraph({ spacing: { ...SINGLE, after: 120 }, children: runs(b.text, { size: 22 }) }));
      break;
    case "pb":
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
    case "table":
      children.push(...tableBlock(b));
      break;
    case "fig":
      children.push(...figureBlock(b));
      break;
    default:
      throw new Error("unknown block type " + b.t);
  }
}

const doc = new Document({
  creator: content.meta.creator || "Authors",
  title: content.meta.title,
  styles: { default: { document: { run: { font: FONT, size: 24 } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outFile, buf); console.log("wrote", outFile, buf.length, "bytes"); });
