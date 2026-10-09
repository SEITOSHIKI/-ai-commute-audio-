// Shared primitives for the 令和の虎 decks (easy manga version and proof version).
// Same look as build_tora.js: black/white tokens, one copper accent, logotype bottom-right.
const path = require("path");
const sharp = require("sharp");
const T = require("../design-system/georec/pptx-theme.js");

const { color: K, size: S } = T;
const F = T.font.sans;
const X0 = 0.4, W = 9.2;
const ACC = "B8692E";
const LOGOS = path.join(__dirname, "../design-system/georec/logos");
const TAGLINE = "現場の勘を、次の担い手へ。";

async function png(svgOrFile, density = 600) {
  const src = svgOrFile.startsWith("<") ? Buffer.from(svgOrFile) : path.join(LOGOS, svgOrFile + ".svg");
  const buf = await sharp(src, { density }).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

function txt(slide, text, o) {
  slide.addText(text, {
    fontFace: F, fontSize: o.size || S.body, color: o.color || K.ink, bold: !!o.bold,
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
  const m = /^([①②③④⑤⑥⑦]|第\d話|お金の話\d|根拠\d+)\s*(.*)$/.exec(title);
  if (m) title = [run(m[1] + " ", { color: ACC }), run(m[2])];
  txt(slide, title, { x: X0, y: 0.28, w: 8.7, h: 0.42, size: S.slideTitle, bold: true, valign: "middle", name: "title" });
  if (lead) txt(slide, lead, { x: X0, y: 0.72, w: 8.7, h: 0.28, size: S.lead, color: K.gray700, valign: "middle", name: "lead" });
}

function table(slide, rows, o) {
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
  box(slide, { x: o.x, y: o.y, w: o.w, h: o.h, fill: o.fill || K.paper, line: K.ink, lw: 2.25, name: o.name });
  if (o.cap) {
    const cw = Math.min(o.w - 0.1, 0.22 + o.cap.length * 0.135);
    box(slide, { x: o.x, y: o.y, w: cw, h: 0.27, fill: K.ink });
    txt(slide, o.cap, { x: o.x + 0.07, y: o.y, w: cw - 0.1, h: 0.27, size: 9, bold: true, color: K.paper, valign: "middle" });
  }
}
// Speech bubble. tail: "bl" | "br" | "tl" | "tr" (where the speaker is, relative to the bubble).
function bubble(slide, text, o) {
  const flipH = o.tail === "br" || o.tail === "tr";
  const flipV = o.tail === "tl" || o.tail === "tr";
  slide.addShape(o.shout ? "wedgeEllipseCallout" : "wedgeRoundRectCallout", {
    x: o.x, y: o.y, w: o.w, h: o.h, flipH, flipV,
    fill: { color: o.fill || K.paper }, line: { color: K.ink, width: 1.5 }, objectName: o.name,
  });
  txt(slide, text, { x: o.x + 0.08, y: o.y + 0.04, w: o.w - 0.16, h: o.h - 0.08, size: o.size || 10, bold: o.bold, color: o.color || K.ink, align: "center", valign: "middle", lsm: 1.1 });
}
// Sound effect / emphasis lettering.
function sfx(slide, text, o) {
  txt(slide, text, { x: o.x, y: o.y, w: o.w || 1.6, h: o.h || 0.5, size: o.size || 22, bold: true, color: o.color || ACC, rotate: o.rotate ?? -8, valign: "middle", align: o.align || "left" });
}

// ---------- characters (SVG, black line art, copper accent) ----------
// role: veteran | junior | manager | founder | client ; face: normal | happy | worried | shock | think
function charSVG(role, face = "normal") {
  const ink = "#1A1A1A", acc = "#" + ACC, gray = "#8C8C8C", light = "#F2F2F2";
  const parts = [];
  // body
  const suit = role === "manager" || role === "founder" || role === "client";
  parts.push(`<path d="M28 238 Q30 172 100 166 Q170 172 172 238 Z" fill="${suit ? ink : light}" stroke="${ink}" stroke-width="5"/>`);
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
    parts.push(`<path d="M40 82 A60 60 0 0 1 160 82 Z" fill="#fff" stroke="${ink}" stroke-width="5"/>`);
    parts.push(`<rect x="32" y="76" width="136" height="11" rx="5" fill="${ink}"/>`);
    parts.push(`<path d="M100 24 L100 78" stroke="${ink}" stroke-width="4"/>`);
    if (role === "veteran") {
      parts.push(`<rect x="88" y="42" width="24" height="20" fill="${acc}"/>`);
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
  // glasses
  if (role === "junior") {
    parts.push(`<rect x="60" y="94" width="36" height="26" rx="4" fill="${ink}" fill-opacity="0.15" stroke="${acc}" stroke-width="5"/>`);
    parts.push(`<rect x="104" y="94" width="36" height="26" rx="4" fill="${ink}" fill-opacity="0.15" stroke="${acc}" stroke-width="5"/>`);
    parts.push(`<path d="M96 104 L104 104 M60 102 L46 98 M140 102 L154 98" stroke="${acc}" stroke-width="5"/>`);
    parts.push(`<circle cx="148" cy="96" r="5" fill="${acc}"/>`);
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
  if (face === "happy") parts.push(`<path d="M170 46 L174 58 L186 62 L174 66 L170 78 L166 66 L154 62 L166 58 Z" fill="${acc}"/>`);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 240" width="200" height="240">${parts.join("")}</svg>`;
}

// Simple pictograms (factory, coin, server, film, report, person-hours)
function iconSVG(kind) {
  const ink = "#1A1A1A", acc = "#" + ACC;
  const s = (b) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">${b}</svg>`;
  switch (kind) {
    case "factory": return s(`<path d="M20 180 L20 90 L70 120 L70 90 L120 120 L120 40 L150 40 L150 120 L180 120 L180 180 Z" fill="#fff" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><rect x="40" y="140" width="22" height="22" fill="${ink}"/><rect x="88" y="140" width="22" height="22" fill="${ink}"/><rect x="136" y="140" width="22" height="22" fill="${ink}"/><path d="M135 30 Q150 10 170 18" stroke="${ink}" stroke-width="6" fill="none"/>`);
    case "coin": return s(`<circle cx="100" cy="100" r="78" fill="${acc}" stroke="${ink}" stroke-width="8"/><circle cx="100" cy="100" r="58" fill="none" stroke="#fff" stroke-width="5"/><path d="M70 62 L100 100 L130 62 M100 100 L100 146 M74 108 L126 108 M74 128 L126 128" stroke="#fff" stroke-width="10" fill="none" stroke-linecap="round"/>`);
    case "server": return s(`<rect x="40" y="24" width="120" height="44" rx="6" fill="#fff" stroke="${ink}" stroke-width="8"/><rect x="40" y="78" width="120" height="44" rx="6" fill="#fff" stroke="${ink}" stroke-width="8"/><rect x="40" y="132" width="120" height="44" rx="6" fill="#fff" stroke="${ink}" stroke-width="8"/><circle cx="64" cy="46" r="7" fill="${ink}"/><circle cx="64" cy="100" r="7" fill="${ink}"/><circle cx="64" cy="154" r="7" fill="${ink}"/><path d="M90 46 L140 46 M90 100 L140 100 M90 154 L140 154" stroke="${ink}" stroke-width="7"/>`);
    case "film": return s(`<rect x="20" y="40" width="160" height="120" rx="10" fill="#fff" stroke="${ink}" stroke-width="8"/><path d="M84 74 L84 126 L128 100 Z" fill="${ink}"/><path d="M20 62 L180 62 M20 138 L180 138" stroke="${ink}" stroke-width="5"/>`);
    case "report": return s(`<path d="M44 20 L130 20 L160 50 L160 180 L44 180 Z" fill="#fff" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><path d="M130 20 L130 50 L160 50" fill="none" stroke="${ink}" stroke-width="6"/><rect x="62" y="64" width="44" height="34" fill="${ink}"/><path d="M116 70 L142 70 M116 90 L142 90 M62 118 L142 118 M62 138 L142 138 M62 158 L120 158" stroke="${ink}" stroke-width="6"/>`);
    case "glasses": return s(`<path d="M14 84 L186 84" stroke="${ink}" stroke-width="8"/><rect x="26" y="78" width="66" height="50" rx="10" fill="#1A1A1A" fill-opacity="0.12" stroke="${acc}" stroke-width="9"/><rect x="108" y="78" width="66" height="50" rx="10" fill="#1A1A1A" fill-opacity="0.12" stroke="${acc}" stroke-width="9"/><path d="M92 96 L108 96" stroke="${acc}" stroke-width="9"/><circle cx="170" cy="64" r="10" fill="${ink}"/><circle cx="170" cy="64" r="4" fill="#fff"/>`);
    case "clock": return s(`<circle cx="100" cy="100" r="76" fill="#fff" stroke="${ink}" stroke-width="8"/><path d="M100 50 L100 100 L136 120" stroke="${ink}" stroke-width="9" fill="none" stroke-linecap="round"/>`);
    case "book": return s(`<path d="M100 50 Q60 30 20 44 L20 164 Q60 150 100 170 Q140 150 180 164 L180 44 Q140 30 100 50 Z" fill="#fff" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><path d="M100 50 L100 170" stroke="${ink}" stroke-width="6"/><path d="M40 74 L80 80 M40 98 L80 104 M120 80 L160 74 M120 104 L160 98" stroke="${ink}" stroke-width="5"/>`);
    case "bait": return s(`<path d="M100 20 L100 120 Q100 160 70 160 Q44 160 44 134" stroke="${ink}" stroke-width="9" fill="none" stroke-linecap="round"/><path d="M44 134 L36 150 M44 134 L58 144" stroke="${ink}" stroke-width="8" stroke-linecap="round"/><circle cx="100" cy="30" r="12" fill="${acc}"/>`);
    case "fish": return s(`<path d="M30 100 Q90 30 150 100 Q90 170 30 100 Z" fill="${acc}" stroke="${ink}" stroke-width="8"/><path d="M150 100 L186 70 L186 130 Z" fill="${acc}" stroke="${ink}" stroke-width="8" stroke-linejoin="round"/><circle cx="62" cy="92" r="9" fill="${ink}"/>`);
  }
  throw new Error("unknown icon " + kind);
}

// Masters + slide factory. notes: {n: string}
async function setup(pres, title, notes) {
  pres.layout = T.slide.layout;
  pres.title = title;
  pres.theme = { headFontFace: F, bodyFontFace: F };
  const logoColor = await png("gen3-logotype-color");
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

module.exports = { T, K, S, F, X0, W, ACC, TAGLINE, png, txt, box, hline, arrow, run, heading, table, panel, bubble, sfx, charSVG, iconSVG, setup, fmt, tri };
