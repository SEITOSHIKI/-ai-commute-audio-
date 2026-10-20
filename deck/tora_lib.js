// Shared primitives for the カンツギ business plan decks (easy manga version and proof version).
// Default look: black/white tokens, one copper accent, logotype bottom-right.
// drawing() switches every primitive to the 16枚版 look (tora16_theme.js): drawing frame and title block, navy, safety colors.
const path = require("path");
const sharp = require("sharp");
const T = require("../design-system/georec/pptx-theme.js");

const { color: K, size: S } = T;
let F = T.font.sans;
const X0 = 0.4, W = 9.2;
let ACC = "B8692E";
let DRAWING = false;
// Colors used inside the primitives (text, frames, line art). drawing() replaces them.
const P = { text: K.ink, frame: K.ink, title: K.ink, line: "#1A1A1A", acc: "#B8692E", hi: "#B8692E", suit: "#1A1A1A", glass: "#B8692E", helmet: "#fff", badge: "#B8692E", bad: "B8692E" };
const LOGOS = path.join(__dirname, "../design-system/georec/logos");
const TAGLINE = "現場の勘を次の担い手へ";

async function png(svgOrFile, density = 600) {
  const src = svgOrFile.startsWith("<") ? Buffer.from(svgOrFile) : path.join(LOGOS, svgOrFile + ".svg");
  const buf = await sharp(src, { density }).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

function txt(slide, text, o) {
  slide.addText(text, {
    fontFace: F, fontSize: o.size || S.body, color: o.color || P.text, bold: !!o.bold,
    x: o.x, y: o.y, w: o.w, h: o.h, margin: o.margin ?? 0, valign: o.valign || "top",
    align: o.align || "left", lineSpacingMultiple: o.lsm || 1.15, paraSpaceAfter: o.psa || 0,
    fill: o.fill ? { color: o.fill } : undefined, isTextBox: true, objectName: o.name, rotate: o.rotate,
  });
}
function box(slide, o) {
  slide.addShape(o.shape || "rect", {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: o.r,
    fill: { color: o.fill || K.paper },
    line: o.line ? { color: o.line, width: o.lw || T.stroke.rule, dashType: o.dash } : { type: "none" },
    objectName: o.name, flipH: o.flipH, flipV: o.flipV, rotate: o.rotate,
  });
}
function hline(slide, x, y, w, c, lw) {
  slide.addShape("line", { x, y, w, h: 0, line: { color: c, width: lw } });
}
function arrow(slide, x, y, w, h, o = {}) {
  slide.addShape("line", { x, y, w, h, flipH: o.flipH, flipV: o.flipV, line: { color: o.color || K.ink, width: o.lw || 2, endArrowType: "triangle", dashType: o.dash } });
}
const run = (text, o = {}) => ({ text, options: { fontFace: F, ...o } });

function heading(slide, title, lead) {
  if (DRAWING && typeof title === "string") title = title.replace(" ─ ", "：");
  const m = /^([①②③④⑤⑥⑦]|第\d話|お金の話\d|根拠\d+|見直し\d+)\s*(.*)$/.exec(title);
  if (m) title = [run(m[1] + " ", { color: ACC }), run(m[2])];
  txt(slide, title, { x: X0, y: 0.28, w: 8.7, h: DRAWING ? 0.48 : 0.42, size: DRAWING ? 18.5 : S.slideTitle, bold: true, color: P.title, valign: "middle", name: "title" });
  if (lead) txt(slide, lead, { x: X0, y: DRAWING ? 0.76 : 0.72, w: 8.7, h: DRAWING ? 0.26 : 0.28, size: DRAWING ? 10.5 : S.lead, color: K.gray700, valign: "middle", name: "lead" });
  if (DRAWING) hline(slide, X0, 1.04, W, P.frame, 1.25);
}

function table(slide, rows, o) {
  if (DRAWING) return require("./tora16_theme.js").table(slide, rows, o);
  const fs = o.size || S.table;
  const body = rows.map((r, i) =>
    r.map((c, j) => {
      const cell = typeof c === "object" ? c : { text: String(c) };
      const head = i === 0;
      const strong = o.strongRows && o.strongRows.includes(i);
      return {
        text: cell.text,
        options: {
          fontFace: F, fontSize: fs, color: cell.color || K.ink,
          bold: head || strong || !!cell.bold,
          align: j === 0 || (o.leftCols || []).includes(j) ? "left" : (o.align || "right"),
          valign: "middle",
          fill: { color: cell.fill || (head ? K.fill : K.paper) },
          margin: [0.02, 0.06, 0.02, 0.06],
          border: [
            { type: strong ? "solid" : "none", color: K.ink, pt: T.stroke.strong },
            { type: "none" },
            head ? { type: "solid", color: K.ink, pt: T.stroke.strong } : { type: "solid", color: K.rule, pt: T.stroke.rule },
            { type: "none" },
          ],
        },
      };
    })
  );
  slide.addTable(body, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, objectName: o.name });
}

// ---------- manga parts ----------
// Panel (koma): thick ink frame, optional narration box at top-left.
function panel(slide, o) {
  box(slide, { x: o.x, y: o.y, w: o.w, h: o.h, fill: o.fill || K.paper, line: P.frame, lw: 2.25, name: o.name });
  if (o.cap) {
    const cw = Math.min(o.w - 0.1, 0.22 + o.cap.length * 0.135);
    box(slide, { x: o.x, y: o.y, w: cw, h: 0.27, fill: P.frame });
    txt(slide, o.cap, { x: o.x + 0.07, y: o.y, w: cw - 0.1, h: 0.27, size: 9, bold: true, color: K.paper, valign: "middle" });
  }
}
// Speech bubble. tail: "bl" | "br" | "tl" | "tr" (where the speaker is, relative to the bubble).
function bubble(slide, text, o) {
  const flipH = o.tail === "br" || o.tail === "tr";
  const flipV = o.tail === "tl" || o.tail === "tr";
  slide.addShape(o.shout ? "wedgeEllipseCallout" : "wedgeRoundRectCallout", {
    x: o.x, y: o.y, w: o.w, h: o.h, flipH, flipV,
    fill: { color: o.fill || K.paper }, line: { color: P.frame, width: 1.5 }, objectName: o.name,
  });
  txt(slide, text, { x: o.x + 0.08, y: o.y + 0.04, w: o.w - 0.16, h: o.h - 0.08, size: o.size || 10, bold: o.bold, color: o.color || P.text, align: "center", valign: "middle", lsm: 1.1 });
}
// Sound effect / emphasis lettering.
function sfx(slide, text, o) {
  txt(slide, text, { x: o.x, y: o.y, w: o.w || 1.6, h: o.h || 0.5, size: o.size || 22, bold: true, color: o.color || ACC, rotate: o.rotate ?? -8, valign: "middle", align: o.align || "left" });
}

// ---------- characters (SVG, black line art, copper accent) ----------
// role: veteran | junior | manager | founder | client ; face: normal | happy | worried | shock | think
function charSVG(role, face = "normal") {
  const ink = P.line, acc = P.acc, gray = "#8C8C8C", light = "#F2F2F2";
  const parts = [];
  // body
  const suit = role === "manager" || role === "founder" || role === "client";
  parts.push(`<path d="M28 238 Q30 172 100 166 Q170 172 172 238 Z" fill="${suit ? P.suit : light}" stroke="${ink}" stroke-width="5"/>`);
  if (suit) {
    parts.push(`<path d="M80 168 L100 196 L120 168 Z" fill="#fff" stroke="${ink}" stroke-width="3"/>`);
    parts.push(`<path d="M94 176 L106 176 L110 222 L100 234 L90 222 Z" fill="${role === "founder" ? acc : gray}"/>`);
  } else {
    parts.push(`<path d="M70 170 L100 196 L130 170" fill="none" stroke="${ink}" stroke-width="4"/>`);
    parts.push(`<rect x="118" y="200" width="26" height="16" rx="2" fill="#fff" stroke="${ink}" stroke-width="3"/>`);
  }
  parts.push(`<rect x="86" y="146" width="28" height="26" fill="#fff" stroke="${ink}" stroke-width="5"/>`);
  // head
  parts.push(`<circle cx="100" cy="100" r="56" fill="#fff" stroke="${ink}" stroke-width="5"/>`);
  // hair / helmet
  if (role === "veteran" || role === "junior") {
    parts.push(`<path d="M40 82 A60 60 0 0 1 160 82 Z" fill="${P.helmet}" stroke="${ink}" stroke-width="5"/>`);
    parts.push(`<rect x="32" y="76" width="136" height="11" rx="5" fill="${ink}"/>`);
    parts.push(`<path d="M100 24 L100 78" stroke="${ink}" stroke-width="4"/>`);
    if (role === "veteran") {
      parts.push(DRAWING ? `<path d="M95 40 h10 v8 h8 v10 h-8 v8 h-10 v-8 h-8 v-10 h8 Z" fill="${P.badge}"/>` : `<rect x="88" y="42" width="24" height="20" fill="${acc}"/>`); // 緑十字
      parts.push(`<path d="M46 104 Q44 122 50 130 M154 104 Q156 122 150 130" stroke="${gray}" stroke-width="8" fill="none"/>`);
    }
  } else if (role === "manager") {
    parts.push(`<path d="M44 98 Q40 42 100 42 Q160 42 156 98 Q150 66 112 62 Q80 70 56 74 Q48 82 44 98 Z" fill="${ink}"/>`);
  } else if (role === "founder") {
    parts.push(`<path d="M44 96 Q42 40 100 40 Q160 40 156 96 Q148 60 100 62 Q70 58 52 74 Z" fill="${ink}"/>`);
  } else {
    parts.push(`<path d="M46 92 Q46 46 100 44 Q154 46 154 92 Q130 70 100 70 Q70 70 46 92 Z" fill="${gray}"/>`);
  }
  // eyes & brows
  const ey = 108;
  const brow = (dx, tilt) => `<path d="M${100 + dx - 12} ${ey - 20 + tilt} L${100 + dx + 12} ${ey - 20 - tilt}" stroke="${ink}" stroke-width="${role === "veteran" ? 7 : 4}" stroke-linecap="round"/>`;
  if (face === "happy") {
    parts.push(`<path d="M68 ${ey} Q78 ${ey - 12} 88 ${ey}" stroke="${ink}" stroke-width="5" fill="none" stroke-linecap="round"/>`);
    parts.push(`<path d="M112 ${ey} Q122 ${ey - 12} 132 ${ey}" stroke="${ink}" stroke-width="5" fill="none" stroke-linecap="round"/>`);
  } else if (face === "shock") {
    parts.push(`<circle cx="78" cy="${ey}" r="10" fill="#fff" stroke="${ink}" stroke-width="4"/><circle cx="78" cy="${ey}" r="3" fill="${ink}"/>`);
    parts.push(`<circle cx="122" cy="${ey}" r="10" fill="#fff" stroke="${ink}" stroke-width="4"/><circle cx="122" cy="${ey}" r="3" fill="${ink}"/>`);
  } else {
    parts.push(`<circle cx="78" cy="${ey}" r="6" fill="${ink}"/><circle cx="122" cy="${ey}" r="6" fill="${ink}"/>`);
  }
  if (face === "worried") { parts.push(brow(-22, -6), brow(22, 6)); }
  else if (face === "think") { parts.push(brow(-22, 4), brow(22, 4)); }
  else { parts.push(brow(-22, 3), brow(22, -3)); }
  // glasses: スポーツ型のAIグラス（顔に沿って巻き込む一枚レンズ。カメラは右上の角に小さく埋め込む）
  if (role === "junior") {
    parts.push(`<path d="M48 94 Q100 82 152 94 L148 112 Q142 126 122 126 Q110 126 105 116 Q100 108 95 116 Q90 126 78 126 Q58 126 52 112 Z" fill="${P.glass}" fill-opacity="0.28" stroke="${ink}" stroke-width="4" stroke-linejoin="round"/>`);
    parts.push(`<path d="M46 95 Q100 81 154 95" fill="none" stroke="${ink}" stroke-width="9" stroke-linecap="round"/>`); // 上のフレーム（眉の線）
    parts.push(`<path d="M60 100 L74 98 M126 98 L140 100" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity="0.8"/>`); // レンズの映り込み
    parts.push(`<path d="M46 96 L38 100 M154 96 L162 100" stroke="${ink}" stroke-width="6" stroke-linecap="round"/>`); // つる
    parts.push(`<circle cx="143" cy="92" r="5.5" fill="${ink}"/><circle cx="143" cy="92" r="2.2" fill="${P.glass}"/>`); // カメラ
    parts.push(`<circle cx="57" cy="92" r="2.4" fill="#fff" stroke="${ink}" stroke-width="1.5"/>`); // 撮影中のランプ
  }
  if (role === "manager") {
    parts.push(`<circle cx="78" cy="${ey}" r="15" fill="none" stroke="${ink}" stroke-width="4"/><circle cx="122" cy="${ey}" r="15" fill="none" stroke="${ink}" stroke-width="4"/><path d="M93 ${ey} L107 ${ey}" stroke="${ink}" stroke-width="4"/>`);
  }
  // mouth
  const my = 136;
  if (face === "happy") parts.push(`<path d="M80 ${my - 4} Q100 ${my + 18} 120 ${my - 4} Z" fill="${ink}"/>`);
  else if (face === "worried") parts.push(`<path d="M84 ${my + 6} Q100 ${my - 6} 116 ${my + 6}" stroke="${ink}" stroke-width="5" fill="none" stroke-linecap="round"/>`);
  else if (face === "shock") parts.push(`<ellipse cx="100" cy="${my + 2}" rx="10" ry="13" fill="${ink}"/>`);
  else if (face === "think") parts.push(`<path d="M88 ${my} L112 ${my - 3}" stroke="${ink}" stroke-width="5" stroke-linecap="round"/>`);
  else parts.push(`<path d="M86 ${my - 2} Q100 ${my + 8} 114 ${my - 2}" stroke="${ink}" stroke-width="5" fill="none" stroke-linecap="round"/>`);
  if (role === "veteran") parts.push(`<path d="M80 ${my - 10} Q100 ${my - 18} 120 ${my - 10}" stroke="${gray}" stroke-width="7" fill="none" stroke-linecap="round"/>`);
  // effects
  if (face === "worried") parts.push(`<path d="M166 70 Q176 86 166 94 Q156 86 166 70 Z" fill="#fff" stroke="${ink}" stroke-width="3"/>`);
  if (face === "shock") parts.push(`<path d="M30 40 L44 58 M20 66 L40 72 M170 40 L156 58 M180 66 L160 72" stroke="${ink}" stroke-width="4" stroke-linecap="round"/>`);
  if (face === "happy") parts.push(`<path d="M170 46 L174 58 L186 62 L174 66 L170 78 L166 66 L154 62 L166 58 Z" fill="${P.hi}"/>`);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 240" width="200" height="240">${parts.join("")}</svg>`;
}

// Simple pictograms (factory, coin, server, film, report, person-hours)
function iconSVG(kind) {
  const ink = P.line, acc = P.acc, hi = P.hi;
  const s = (b) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">${b}</svg>`;
  switch (kind) {
    case "factory": return s(`<path d="M20 180 L20 90 L70 120 L70 90 L120 120 L120 40 L150 40 L150 120 L180 120 L180 180 Z" fill="#fff" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><rect x="40" y="140" width="22" height="22" fill="${ink}"/><rect x="88" y="140" width="22" height="22" fill="${ink}"/><rect x="136" y="140" width="22" height="22" fill="${ink}"/><path d="M135 30 Q150 10 170 18" stroke="${ink}" stroke-width="6" fill="none"/>`);
    case "office": return s(`<rect x="34" y="40" width="132" height="140" fill="#fff" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><path d="M20 180 L180 180" stroke="${ink}" stroke-width="8"/><rect x="56" y="64" width="24" height="22" fill="${ink}"/><rect x="120" y="64" width="24" height="22" fill="${ink}"/><rect x="56" y="104" width="24" height="22" fill="${ink}"/><rect x="120" y="104" width="24" height="22" fill="${ink}"/><rect x="86" y="140" width="28" height="40" fill="${ink}"/><path d="M106 6 L84 40 L102 40 L92 66 L120 30 L102 30 L112 6 Z" fill="${hi}" stroke="${ink}" stroke-width="4" stroke-linejoin="round"/>`);
    case "coin": return s(`<circle cx="100" cy="100" r="78" fill="${DRAWING ? "#C9A000" : acc}" stroke="${ink}" stroke-width="8"/><circle cx="100" cy="100" r="58" fill="none" stroke="#fff" stroke-width="5"/><path d="M70 62 L100 100 L130 62 M100 100 L100 146 M74 108 L126 108 M74 128 L126 128" stroke="#fff" stroke-width="10" fill="none" stroke-linecap="round"/>`);
    case "server": return s(`<rect x="40" y="24" width="120" height="44" rx="6" fill="#fff" stroke="${ink}" stroke-width="8"/><rect x="40" y="78" width="120" height="44" rx="6" fill="#fff" stroke="${ink}" stroke-width="8"/><rect x="40" y="132" width="120" height="44" rx="6" fill="#fff" stroke="${ink}" stroke-width="8"/><circle cx="64" cy="46" r="7" fill="${ink}"/><circle cx="64" cy="100" r="7" fill="${ink}"/><circle cx="64" cy="154" r="7" fill="${ink}"/><path d="M90 46 L140 46 M90 100 L140 100 M90 154 L140 154" stroke="${ink}" stroke-width="7"/>`);
    case "film": return s(`<rect x="20" y="40" width="160" height="120" rx="10" fill="#fff" stroke="${ink}" stroke-width="8"/><path d="M84 74 L84 126 L128 100 Z" fill="${ink}"/><path d="M20 62 L180 62 M20 138 L180 138" stroke="${ink}" stroke-width="5"/>`);
    case "report": return s(`<path d="M44 20 L130 20 L160 50 L160 180 L44 180 Z" fill="#fff" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><path d="M130 20 L130 50 L160 50" fill="none" stroke="${ink}" stroke-width="6"/><rect x="62" y="64" width="44" height="34" fill="${ink}"/><path d="M116 70 L142 70 M116 90 L142 90 M62 118 L142 118 M62 138 L142 138 M62 158 L120 158" stroke="${ink}" stroke-width="6"/>`);
    case "glasses": return s(`<path d="M10 92 L2 104 M190 92 L198 104" stroke="${ink}" stroke-width="9" stroke-linecap="round"/><path d="M14 86 Q100 62 186 86 L180 116 Q172 142 138 142 Q116 142 108 124 Q100 110 92 124 Q84 142 62 142 Q28 142 20 116 Z" fill="${P.glass}" fill-opacity="0.3" stroke="${ink}" stroke-width="7" stroke-linejoin="round"/><path d="M10 88 Q100 60 190 88" fill="none" stroke="${ink}" stroke-width="14" stroke-linecap="round"/><path d="M38 100 L66 94 M134 94 L162 100" stroke="#fff" stroke-width="6" stroke-linecap="round" opacity="0.85"/><circle cx="168" cy="80" r="11" fill="${ink}"/><circle cx="168" cy="80" r="4.5" fill="${P.glass}"/><circle cx="32" cy="80" r="4.5" fill="#fff" stroke="${ink}" stroke-width="2.5"/>`); // スポーツ型のAIグラス（右上にカメラ、左上に撮影中のランプ）
    case "clock": return s(`<circle cx="100" cy="100" r="76" fill="#fff" stroke="${ink}" stroke-width="8"/><path d="M100 50 L100 100 L136 120" stroke="${ink}" stroke-width="9" fill="none" stroke-linecap="round"/>`);
    case "book": return s(`<path d="M100 50 Q60 30 20 44 L20 164 Q60 150 100 170 Q140 150 180 164 L180 44 Q140 30 100 50 Z" fill="#fff" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><path d="M100 50 L100 170" stroke="${ink}" stroke-width="6"/><path d="M40 74 L80 80 M40 98 L80 104 M120 80 L160 74 M120 104 L160 98" stroke="${ink}" stroke-width="5"/>`);
    case "bait": return s(`<path d="M100 20 L100 120 Q100 160 70 160 Q44 160 44 134" stroke="${ink}" stroke-width="9" fill="none" stroke-linecap="round"/><path d="M44 134 L36 150 M44 134 L58 144" stroke="${ink}" stroke-width="8" stroke-linecap="round"/><circle cx="100" cy="30" r="12" fill="${hi}"/>`);
    case "fish": return s(`<path d="M30 100 Q90 30 150 100 Q90 170 30 100 Z" fill="${acc}" stroke="${ink}" stroke-width="8"/><path d="M150 100 L186 70 L186 130 Z" fill="${acc}" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><circle cx="62" cy="92" r="9" fill="${ink}"/>`);
  }
  throw new Error("unknown icon " + kind);
}

// Masters + slide factory. notes: {n: string}
async function setup(pres, title, notes) {
  if (DRAWING) return require("./tora16_theme.js").setup(pres, title, notes);
  pres.layout = T.slide.layout;
  pres.title = title;
  pres.theme = { headFontFace: F, bodyFontFace: F };
  const logoColor = await png("kantsugi-logotype-color");
  pres.defineSlideMaster({
    title: "GR_CONTENT",
    background: { color: K.paper },
    objects: [
      { image: { x: 8.15, y: 5.2, w: 1.45, h: 1.45 * 120 / 560, data: logoColor } },
      { text: { text: TAGLINE, options: { x: 0.8, y: 5.29, w: 3.5, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "left", margin: 0, valign: "middle" } } },
    ],
    slideNumber: { x: 0.4, y: 5.29, w: 0.35, h: 0.22, fontFace: F, fontSize: S.footer, color: K.gray500, align: "left", margin: 0 },
  });
  pres.defineSlideMaster({ title: "GR_COVER", background: { color: K.paper }, objects: [{ image: { x: 8.15, y: 5.2, w: 1.45, h: 1.45 * 120 / 560, data: logoColor } }] });
  return (n) => { const s = pres.addSlide({ masterName: n === 1 ? "GR_COVER" : "GR_CONTENT" }); if (notes[n]) s.addNotes(notes[n]); return s; };
}

const fmt = (v) => Math.round(v).toLocaleString("en-US");
const tri = (v) => (v < 0 ? "▲" + fmt(-v) : fmt(v));

// "黒字化の条件と撤退ライン": shortfall table (10台 vs 15台, hiring linked) and the go / fix / stop gates. Used by every deck.
function gates(slide, V, o = {}) {
  const top = o.top || 1.12, mil = (v) => Math.round(v / 100), be = Math.round(V.customerValue().breakEvenMin);
  const d15 = { devices: 15, mix: { glass: 12, display: 3 } };
  const sc = (k, plan) => V.run({ plan, newSites: V.GROWTH.newSites.map((n) => Math.round(n * k)), contract: V.GROWTH.contract.map((c) => c * k), hireK: k < 1 ? k : undefined });
  const yr = (r) => (r.firstBlack ? r.firstBlack + "年目" : "5年でしない");
  const cell = (r) => [{ text: yr(r), color: r.firstBlack ? (DRAWING ? "2E7D32" : K.ink) : P.bad, bold: !r.firstBlack }, { text: tri(mil(r.y5.op)), color: r.y5.op < 0 ? P.bad : K.ink }, { text: String(mil(r.minCash)), color: r.minCash < 0 ? P.bad : K.ink }];
  txt(slide, "新規拠点が計画に届かなかったら（採用も拠点数に合わせて遅らせる・百万円）", { x: X0, y: top, w: W, h: 0.26, size: 10, bold: true });
  const rows = [1, 0.9, 0.8, 0.7, 0.6].map((k) => {
    const a = sc(k), b = sc(k, d15);
    return [{ text: k === 1 ? "計画どおり" : `計画の${k * 10}割`, bold: k === 1 }, { text: a.rows.map((r) => r.nw).join("・"), color: K.gray700 }, ...cell(a), ...cell(b)];
  });
  table(slide, [["新規の達成度", "新規拠点（1〜5年目）", "黒字化", "5年目利益", "最低現金", "黒字化", "5年目利益", "最低現金"], ...rows],
    { x: X0, y: top + 0.55, w: W, colW: [1.25, 1.75, 1.05, 1.05, 0.95, 1.05, 1.15, 0.95], rowH: 0.25, size: 8.5, align: "center", strongRows: [1], name: "shortfall" });
  txt(slide, "1拠点10台のまま", { x: X0 + 3.0, y: top + 0.3, w: 3.05, h: 0.23, size: 8.5, bold: true, align: "center", valign: "middle", fill: K.fill });
  txt(slide, "1拠点15台に増やせた場合", { x: X0 + 6.05, y: top + 0.3, w: 3.15, h: 0.23, size: 8.5, bold: true, color: ACC, align: "center", valign: "middle", fill: K.fill });
  const gy = top + 2.12;
  // drawing(): 進む＝緑、直す＝薄い黄、止まる＝赤
  const go = (t) => (DRAWING ? { text: t, color: "2E7D32", bold: true } : t), fix = (t) => (DRAWING ? { text: t, fill: "FFF4C2" } : t), stop = (t) => (DRAWING ? { text: t, color: "C62828", bold: true } : t);
  txt(slide, "進む・直す・止まるの線（案）", { x: X0, y: gy, w: W, h: 0.26, size: 10, bold: true });
  table(slide, [["時期", "測るもの", "進む", "直す", "止まる"],
    ["実証（準備期）", "1台1日の削減時間", go(`${be}分以上`), fix(`15〜${be}分：機能を写真台帳に絞る`), stop("15分未満：撤退")],
    ["1年目末", "試験導入3社の本契約", go("2社以上"), fix("1社：値段と顧客を見直す"), stop("0社：撤退か方向転換")],
    ["2年目末", "新規拠点・1拠点の台数", go("5拠点以上・平均12台"), fix("5拠点未満：採用を止める"), "─"],
    ["3年目", "追加の5,000万円を入れるか", go("継続率90%以上"), fix("未達：追加額と採用を絞る"), "─"]],
    { x: X0, y: gy + 0.28, w: W, colW: [1.2, 1.9, 1.6, 2.5, 2.0], rowH: 0.25, size: 8.5, leftCols: [1, 2, 3, 4], name: "gates" });
}

// Switch to the 16枚版 look. Call before destructuring the exports: const L = require("./tora_lib.js").drawing();
function drawing() {
  DRAWING = true;
  F = "BIZ UDPGothic"; ACC = "2F6DB5";
  Object.assign(K, { ink: "1B365D", gray700: "4A5563", gray500: "7B8794", rule: "B8C4D0", fill: "EAF0F6" });
  Object.assign(P, { text: "1F2933", frame: "1B365D", title: "1B365D", line: "#1F2933", acc: "#2F6DB5", hi: "#F5C400", suit: "#1B365D", glass: "#2F6DB5", helmet: "#F5C400", badge: "#2E7D32", bad: "C62828" });
  Object.assign(module.exports, { F, ACC, DRAWING: true });
  return module.exports;
}

module.exports = { T, K, S, F, X0, W, ACC, TAGLINE, png, txt, box, hline, arrow, run, heading, table, panel, bubble, sfx, charSVG, iconSVG, setup, fmt, tri, gates, drawing, DRAWING: false,
  YELLOW: "F5C400", PALEY: "FFF4C2", GREEN: "2E7D32", RED: "C62828", NAVY: "1B365D" };
