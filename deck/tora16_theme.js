// 16枚版の見た目（図面と工程表の書き方）。
// 図面の枠と表題欄、紺の見出し、現場の安全色（進む＝緑・注意＝黄・止まる＝赤）、紺ヘッダーの表。
const L = require("./tora_lib.js");

const F = "BIZ UDPGothic"; // Windows標準のユニバーサルデザイン書体（無い環境では代替書体で表示）
const NAVY = "1B365D", BLUE = "2F6DB5", INK = "1F2933", PAPER = "FFFFFF", PALE = "EAF0F6", LINE = "B8C4D0";
const GRAY7 = "4A5563", GRAY5 = "7B8794", YELLOW = "F5C400", PALEY = "FFF4C2", GREEN = "2E7D32", RED = "C62828";
const K = { ink: NAVY, paper: PAPER, fill: PALE, gray700: GRAY7, gray500: GRAY5, rule: LINE };
const X0 = 0.4, W = 9.2;
const SECTION = { "①": "① 事業", "②": "② スタッフ", "③": "③ 経営理念", "④": "④ 経営計画", "⑤": "⑤ 課題" };

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
const hline = (slide, x, y, w, c, lw) => slide.addShape("line", { x, y, w, h: 0, line: { color: c, width: lw } });
const arrow = (slide, x, y, w, h, o = {}) => slide.addShape("line", { x, y, w, h, line: { color: o.color || NAVY, width: o.lw || 2, endArrowType: "triangle" } });
const run = (text, o = {}) => ({ text, options: { fontFace: F, ...o } });

// 見出し：番号を紺の四角に入れ、表題の下に図面の表題線を引く。右下の表題欄に章を書く
function heading(slide, title, lead) {
  const m = /^([①②③④⑤])\s*(.*)$/.exec(title);
  let tx = X0;
  if (m) {
    box(slide, { x: X0, y: 0.3, w: 0.44, h: 0.44, fill: NAVY });
    txt(slide, String("①②③④⑤".indexOf(m[1]) + 1), { x: X0, y: 0.3, w: 0.44, h: 0.44, size: 17, bold: true, color: PAPER, align: "center", valign: "middle" });
    title = m[2]; tx = X0 + 0.54;
    txt(slide, SECTION[m[1]], { x: 6.62, y: 5.205, w: 1.3, h: 0.29, size: 7.5, color: NAVY, bold: true, align: "center", valign: "middle" });
  }
  txt(slide, title, { x: tx, y: 0.28, w: W - (tx - X0), h: 0.48, size: 18.5, bold: true, color: NAVY, valign: "middle", name: "title" });
  if (lead) txt(slide, lead, { x: X0, y: 0.76, w: W, h: 0.26, size: 10.5, color: GRAY7, valign: "middle", name: "lead" });
  hline(slide, X0, 1.04, W, NAVY, 1.25);
}

// 表：紺の見出し行、薄い罫線。強調行は薄い青で塗る
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

// 図面の枠と表題欄（右下）。表題欄＝[章][件名][頁]
async function setup(pres, title, notes) {
  pres.layout = "LAYOUT_16x9";
  pres.title = title;
  pres.theme = { headFontFace: F, bodyFontFace: F };
  const logo = await L.png("kantsugi-logotype-color");
  const frame = [
    { rect: { x: 0.12, y: 0.12, w: 9.76, h: 5.385, fill: { color: PAPER, transparency: 100 }, line: { color: NAVY, width: 0.75 } } },
    { rect: { x: 6.6, y: 5.2, w: 3.28, h: 0.305, fill: { color: PAPER }, line: { color: NAVY, width: 0.75 } } },
    { line: { x: 7.92, y: 5.2, w: 0, h: 0.305, line: { color: NAVY, width: 0.75 } } },
    { line: { x: 9.3, y: 5.2, w: 0, h: 0.305, line: { color: NAVY, width: 0.75 } } },
    { image: { x: 8.135, y: 5.252, w: 0.95, h: 0.95 * 120 / 560, data: logo } },
    { text: { text: "現場の勘を次の担い手へ", options: { x: 0.4, y: 5.22, w: 3.5, h: 0.26, fontFace: F, fontSize: 8, color: GRAY5, margin: 0, valign: "middle" } } },
  ];
  pres.defineSlideMaster({ title: "KT", background: { color: PAPER }, objects: frame,
    slideNumber: { x: 9.3, y: 5.22, w: 0.58, h: 0.27, fontFace: F, fontSize: 9, bold: true, color: NAVY, align: "center", margin: 0 } });
  pres.defineSlideMaster({ title: "KT_COVER", background: { color: PAPER }, objects: frame.slice(0, 1) });
  return (n) => { const s = pres.addSlide({ masterName: n === 1 ? "KT_COVER" : "KT" }); if (notes[n]) s.addNotes(notes[n]); return s; };
}

module.exports = { F, NAVY, BLUE, INK, PAPER, PALE, LINE, GRAY7, GRAY5, YELLOW, PALEY, GREEN, RED, K, X0, W, ACC: BLUE, txt, box, hline, arrow, run, heading, table, setup, png: L.png };
