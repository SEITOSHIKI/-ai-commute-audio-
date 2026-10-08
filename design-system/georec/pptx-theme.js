// GEOREC theme constants for pptxgenjs, derived from tokens.json.
// Colors: 6-digit hex without "#". Lengths: inches. Font sizes: pt.
const color = {
  ink: "1A1A1A",
  gray700: "4D4D4D",
  gray500: "8C8C8C",
  rule: "BFBFBF",
  fill: "F2F2F2",
  paper: "FFFFFF",
};

const font = { sans: "Meiryo", latin: "Arial" };

const size = {
  taglineDisplay: 36,
  coverTitle: 24,
  slideTitle: 20,
  figure: 28,
  body: 14,
  lead: 12,
  table: 9.5,
  footer: 8.5,
  note: 8,
};

const space = { 1: 0.05, 2: 0.1, 3: 0.15, 4: 0.2, 6: 0.3, 8: 0.4 };

const stroke = { rule: 0.75, strong: 1.5 };

const slide = { layout: "LAYOUT_16x9", w: 10, h: 5.625, cornerMark: 0.25 };

// pptxgenjs mutates option objects, so every helper returns a fresh one.
const text = {
  title: () => ({ fontFace: font.sans, fontSize: size.slideTitle, bold: true, color: color.ink, isTextBox: true }),
  lead: () => ({ fontFace: font.sans, fontSize: size.lead, color: color.gray700, isTextBox: true }),
  body: () => ({ fontFace: font.sans, fontSize: size.body, color: color.ink, isTextBox: true }),
  table: () => ({ fontFace: font.sans, fontSize: size.table, color: color.ink, isTextBox: true }),
  note: () => ({ fontFace: font.sans, fontSize: size.note, color: color.gray700, isTextBox: true }),
  footer: () => ({ fontFace: font.sans, fontSize: size.footer, color: color.gray500, align: "right", isTextBox: true }),
  onInk: () => ({ fontFace: font.sans, fontSize: size.body, color: color.paper, isTextBox: true }),
};

const line = {
  rule: () => ({ color: color.rule, width: stroke.rule }),
  strong: () => ({ color: color.ink, width: stroke.strong }),
};

module.exports = { color, font, size, space, stroke, slide, text, line };
