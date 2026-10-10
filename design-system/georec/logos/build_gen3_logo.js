// GEN³ Works logo: an AR viewfinder scans the cube of 三現 (現場・現物・現実) from every side.
// The two side faces are the physical site; the copper top face is the know-how, captured as data points
// that stream out of the open corner to the next generation.
// Usage: node build_gen3_logo.js   (writes the gen3-cube-* and gen3-logotype-* SVGs next to this file)
const fs = require("fs");
const path = require("path");

const INK = "#1A1A1A", G7 = "#4D4D4D", ACC = "#B8692E", PAPER = "#FFFFFF", LIGHT = "#BFBFBF";
const f = (n) => n.toFixed(1);
const THEMES = {
  color: { ink: INK, side: G7, acc: ACC, sub: G7 },
  mono: { ink: INK, side: G7, acc: INK, sub: G7 },
  reverse: { ink: PAPER, side: LIGHT, acc: ACC, sub: "#D9D9D9" },
};

function iso(cx, cy, s) {
  const h = s * 0.866, v = s * 0.5;
  return { T: [cx, cy - s], R: [cx + h, cy - v], C: [cx, cy], L: [cx - h, cy - v], RB: [cx + h, cy + v], LB: [cx - h, cy + v], B: [cx, cy + s] };
}
const poly = (pts, attrs) => `<polygon points="${pts.map((p) => f(p[0]) + "," + f(p[1])).join(" ")}" ${attrs}/>`;

function mark(theme) {
  const t = THEMES[theme];
  const c = iso(58, 66, 31);
  // viewfinder: three closed corners, the top-right one left open for the data stream
  const br = (d) => `<path d="${d}" fill="none" stroke="${t.ink}" stroke-width="7" stroke-linecap="square"/>`;
  let s = br("M8 28 L8 8 L28 8") + br("M8 92 L8 112 L28 112") + br("M112 92 L112 112 L92 112");
  s += poly([c.L, c.C, c.B, c.LB], `fill="${t.ink}"`);
  s += poly([c.C, c.R, c.RB, c.B], `fill="${t.side}"`);
  s += poly([c.T, c.R, c.C, c.L], `fill="none" stroke="${t.acc}" stroke-width="2.4" stroke-linejoin="round"`);
  // top face as a 4×4 grid of data points
  const n = 4;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const u = (i + 0.5) / n, v = (j + 0.5) / n;
    const x = c.L[0] + (c.T[0] - c.L[0]) * u + (c.C[0] - c.L[0]) * v;
    const y = c.L[1] + (c.T[1] - c.L[1]) * u + (c.C[1] - c.L[1]) * v;
    s += `<circle cx="${f(x)}" cy="${f(y)}" r="2.7" fill="${t.acc}"/>`;
  }
  // the stream: points leave the cube and pass through the open corner
  [[90, 32, 3.8], [99, 23, 3.2], [107, 15, 2.6], [114, 8, 2.0]].forEach(([x, y, r]) => (s += `<circle cx="${x}" cy="${y}" r="${r}" fill="${t.acc}"/>`));
  return s;
}

const svg = (inner, w, h, label) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${label}">${inner}</svg>\n`;

function logotype(theme, tagline) {
  const t = THEMES[theme];
  let s = mark(theme);
  s += `<text x="134" y="66" fill="${t.ink}" font-family="Arial, Helvetica, Liberation Sans, sans-serif" font-size="54" font-weight="700" letter-spacing="2">GEN<tspan font-size="34" dy="-20" fill="${t.acc}">3</tspan><tspan dy="20" dx="14">WORKS</tspan></text>`;
  s += `<text x="136" y="100" fill="${t.sub}" font-family="IPAGothic, Meiryo, Hiragino Sans, sans-serif" font-size="22" letter-spacing="6">三現ワークス</text>`;
  if (tagline) s += `<text x="136" y="138" fill="${t.acc}" font-family="IPAGothic, Meiryo, Hiragino Sans, sans-serif" font-size="17" letter-spacing="1">現場の勘を、データ資産として次の担い手へ。</text>`;
  return svg(s, 560, tagline ? 150 : 120, "GEN3 WORKS 三現ワークス");
}

const out = (name, body) => fs.writeFileSync(path.join(__dirname, name), body);
for (const theme of Object.keys(THEMES)) {
  out(`gen3-cube-${theme}.svg`, svg(mark(theme), 120, 120, "GEN3 Works"));
  out(`gen3-logotype-${theme}.svg`, logotype(theme, false));
}
out("gen3-logotype-tagline.svg", logotype("color", true));
out("gen3-logotype-tagline-reverse.svg", logotype("reverse", true));
console.log("wrote gen3 logo SVGs");
