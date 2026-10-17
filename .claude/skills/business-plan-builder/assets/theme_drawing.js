// スライドの見た目（図面と工程表の書き方）。pptxgenjs 用。
// 図面の外枠と右下の表題欄（章｜ロゴ｜頁）、紺の見出しと表、現場の安全色（進む＝緑・注意＝黄・止まる＝赤）。
// 使い方：
//   const TH = require("./theme_drawing.js");
//   const add = await TH.setup(pres, { title: "株式会社〇〇 事業計画書", tagline: "一行の旗印", logo: "logo.svg", notes });
//   const s = add(2); TH.heading(s, "① 事業概要：〜を〜する", "リード文"); TH.table(s, rows, { x, y, w, colW });
const fs = require("fs");

const F = "BIZ UDPGothic"; // Windows標準のユニバーサルデザイン書体（無い環境では代替書体で表示）
const NAVY = "1B365D", BLUE = "2F6DB5", INK = "1F2933", PAPER = "FFFFFF", PALE = "EAF0F6", LINE = "B8C4D0";
const GRAY7 = "4A5563", GRAY5 = "7B8794", YELLOW = "F5C400", PALEY = "FFF4C2", GREEN = "2E7D32", RED = "C62828";
const X0 = 0.4, W = 9.2;
// 見出しの丸数字と、表題欄に書く章の名前（課題の設問に合わせて変える）
let SECTION = { "①": "① 事業", "②": "② スタッフ", "③": "③ 経営理念", "④": "④ 経営計画", "⑤": "⑤ 課題" };

function txt(slide, text, o) {
  slide.addText(text, {
    fontFace: F, fontSize: o.size || 11, color: o.color || INK, bold: !!o.bold,
    x: o.x, y: o.y, w: o.w, h: o.h, margin: o.margin ?? 0, valign: o.valign || "top",
    align: o.align || "left", lineSpacingMultiple: o.lsm || 1.15, paraSpaceAfter: o.psa || 0,
    fill: o.fill ? { color: o.fill } : undefined, isTextBox: true, objectName: o.name,
  });
}
function box(slide, o) {
  slide.addShape(o.shape || "rect", { x: o.x, y: o.y, w: o.w, h: o.h, fill: { color: o.fill || PAPER },
    line: o.line ? { color: o.line, width: o.lw || 0.75, dashType: o.dash } : { type: "none" }, objectName: o.name });
}
const hline = (slide, x, y, w, c = NAVY, lw = 0.75) => slide.addShape("line", { x, y, w, h: 0, line: { color: c, width: lw } });
const arrow = (slide, x, y, w, h, o = {}) => slide.addShape("line", { x, y, w, h, line: { color: o.color || NAVY, width: o.lw || 2, endArrowType: "triangle" } });
const run = (text, o = {}) => ({ text, options: { fontFace: F, ...o } });

// 見出し：番号を紺の四角に入れ、表題の下に図面の表題線を引く。右下の表題欄に章を書く。
// 見出しは「項目：結論」の形で、1行（全角で約28字）に収める
function heading(slide, title, lead) {
  const m = /^([①②③④⑤⑥⑦⑧⑨])\s*(.*)$/.exec(title);
  let tx = X0;
  if (m) {
    box(slide, { x: X0, y: 0.3, w: 0.44, h: 0.44, fill: NAVY });
    txt(slide, String("①②③④⑤⑥⑦⑧⑨".indexOf(m[1]) + 1), { x: X0, y: 0.3, w: 0.44, h: 0.44, size: 17, bold: true, color: PAPER, align: "center", valign: "middle" });
    title = m[2]; tx = X0 + 0.54;
    if (SECTION[m[1]]) txt(slide, SECTION[m[1]], { x: 6.62, y: 5.205, w: 1.3, h: 0.29, size: 7.5, color: NAVY, bold: true, align: "center", valign: "middle" });
  }
  txt(slide, title, { x: tx, y: 0.28, w: W - (tx - X0), h: 0.48, size: 18.5, bold: true, color: NAVY, valign: "middle", name: "title" });
  if (lead) txt(slide, lead, { x: X0, y: 0.76, w: W, h: 0.26, size: 10.5, color: GRAY7, valign: "middle", name: "lead" });
  hline(slide, X0, 1.04, W, NAVY, 1.25);
}

// 表：紺の見出し行に白文字、薄い罫線。強調行は薄い青。セルは文字列か { text, color, bold, fill }
function table(slide, rows, o) {
  const fs = o.size || 9;
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    const head = i === 0, strong = o.strongRows && o.strongRows.includes(i);
    return { text: cell.text, options: {
      fontFace: F, fontSize: fs, color: head ? PAPER : (cell.color || INK), bold: head || strong || !!cell.bold,
      align: j === 0 || (o.leftCols || []).includes(j) ? "left" : (o.align || "right"), valign: "middle",
      fill: { color: head ? NAVY : cell.fill || (strong ? PALE : PAPER) }, margin: [0.02, 0.06, 0.02, 0.06],
      border: [{ type: "none" }, { type: "none" }, { type: "solid", color: head ? NAVY : LINE, pt: 0.75 }, { type: "none" }],
    } };
  }));
  slide.addTable(body, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, objectName: o.name });
}

// 判断基準の色：進む＝緑、直す＝黄、止まる＝赤。赤字の数字は赤
const go = (text) => ({ text, color: GREEN, bold: true });
const fix = (text) => ({ text, fill: PALEY });
const stop = (text) => ({ text, color: RED, bold: true });
const yen = (v, f = (x) => Math.round(x).toLocaleString("ja-JP")) => ({ text: v < 0 ? "▲" + f(-v) : f(v), color: v < 0 ? RED : INK });

// ロゴ：png はそのまま、svg は sharp があれば png にする。無ければ文字で代用
async function logoData(logo) {
  if (!logo) return null;
  if (logo.startsWith("image/") || logo.startsWith("data:")) return logo.replace(/^data:/, "");
  const buf = fs.readFileSync(logo);
  if (logo.endsWith(".png")) return "image/png;base64," + buf.toString("base64");
  try { const sharp = require("sharp"); return "image/png;base64," + (await sharp(buf, { density: 600 }).png().toBuffer()).toString("base64"); }
  catch { return null; }
}

// 図面の枠と表題欄（右下）。表題欄＝[章][ロゴか社名][頁]
// opt: { title, tagline, logo（png/svg のパス）, logoRatio（高さ÷幅）, company（ロゴが無いときの社名）, sections, notes }
async function setup(pres, opt) {
  pres.layout = "LAYOUT_16x9";
  pres.title = opt.title;
  pres.theme = { headFontFace: F, bodyFontFace: F };
  if (opt.sections) SECTION = opt.sections;
  const logo = await logoData(opt.logo);
  const frame = [
    { rect: { x: 0.12, y: 0.12, w: 9.76, h: 5.385, fill: { color: PAPER, transparency: 100 }, line: { color: NAVY, width: 0.75 } } },
    { rect: { x: 6.6, y: 5.2, w: 3.28, h: 0.305, fill: { color: PAPER }, line: { color: NAVY, width: 0.75 } } },
    { line: { x: 7.92, y: 5.2, w: 0, h: 0.305, line: { color: NAVY, width: 0.75 } } },
    { line: { x: 9.3, y: 5.2, w: 0, h: 0.305, line: { color: NAVY, width: 0.75 } } },
    logo ? { image: { x: 8.135, y: 5.252, w: 0.95, h: 0.95 * (opt.logoRatio || 120 / 560), data: logo } }
      : { text: { text: opt.company || "", options: { x: 7.92, y: 5.2, w: 1.38, h: 0.305, fontFace: F, fontSize: 8, bold: true, color: NAVY, align: "center", valign: "middle", margin: 0 } } },
  ];
  if (opt.tagline) frame.push({ text: { text: opt.tagline, options: { x: 0.4, y: 5.22, w: 3.5, h: 0.26, fontFace: F, fontSize: 8, color: GRAY5, margin: 0, valign: "middle" } } });
  pres.defineSlideMaster({ title: "PLAN", background: { color: PAPER }, objects: frame,
    slideNumber: { x: 9.3, y: 5.22, w: 0.58, h: 0.27, fontFace: F, fontSize: 9, bold: true, color: NAVY, align: "center", margin: 0 } });
  pres.defineSlideMaster({ title: "PLAN_COVER", background: { color: PAPER }, objects: frame.slice(0, 1) });
  const notes = opt.notes || {};
  return (n) => { const s = pres.addSlide({ masterName: n === 1 ? "PLAN_COVER" : "PLAN" }); if (notes[n]) s.addNotes(notes[n]); return s; };
}

// 表紙の表題欄（件名｜頁）。表紙だけは大きめに描く
function coverTitleBlock(slide, subject, page) {
  box(slide, { x: 5.6, y: 4.9, w: 4.28, h: 0.605, line: NAVY });
  hline(slide, 5.6, 5.2, 4.28);
  txt(slide, "件名", { x: 5.66, y: 4.92, w: 0.6, h: 0.26, size: 7.5, color: GRAY5, valign: "middle" });
  txt(slide, subject, { x: 6.2, y: 4.92, w: 3.6, h: 0.26, size: 9, bold: true, color: NAVY, valign: "middle" });
  txt(slide, page, { x: 5.66, y: 5.22, w: 4.1, h: 0.26, size: 8.5, color: NAVY, align: "right", valign: "middle" });
}

module.exports = { F, NAVY, BLUE, INK, PAPER, PALE, LINE, GRAY7, GRAY5, YELLOW, PALEY, GREEN, RED, X0, W, ACC: BLUE,
  txt, box, hline, arrow, run, heading, table, go, fix, stop, yen, setup, coverTitleBlock };
