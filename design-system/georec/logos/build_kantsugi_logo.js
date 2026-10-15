// KANTSUGI (カンツギ／勘継) logo.
// An AR viewfinder scans the cube of the site (現場・現物・現実) from every side. A copper seam runs
// across the cube like kintsugi: where know-how would break off, it is joined again with gold — here, data.
// The copper top face is that know-how as data points, streaming out of the open corner to the next generation.
// Usage: node build_kantsugi_logo.js   (writes the kantsugi-* SVGs next to this file)
const fs = require("fs");
const path = require("path");

const INK = "#1A1A1A", G7 = "#4D4D4D", ACC = "#B8692E", PAPER = "#FFFFFF", LIGHT = "#BFBFBF";
const f = (n) => n.toFixed(1);
const THEMES = {
  color: { ink: INK, side: G7, acc: ACC, sub: G7 },
  mono: { ink: INK, side: G7, acc: INK, sub: G7, seam: PAPER },
  reverse: { ink: PAPER, side: LIGHT, acc: ACC, sub: "#D9D9D9" },
};

function iso(cx, cy, s) {
  const h = s * 0.866, v = s * 0.5;
  return { T: [cx, cy - s], R: [cx + h, cy - v], C: [cx, cy], L: [cx - h, cy - v], RB: [cx + h, cy + v], LB: [cx - h, cy + v], B: [cx, cy + s] };
}
const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const poly = (pts, attrs) => `<polygon points="${pts.map((p) => f(p[0]) + "," + f(p[1])).join(" ")}" ${attrs}/>`;

function mark(theme) {
  const t = THEMES[theme];
  const c = iso(58, 66, 31);
  const br = (d) => `<path d="${d}" fill="none" stroke="${t.ink}" stroke-width="7" stroke-linecap="square"/>`;
  let s = br("M8 28 L8 8 L28 8") + br("M8 92 L8 112 L28 112") + br("M112 92 L112 112 L92 112");
  s += poly([c.L, c.C, c.B, c.LB], `fill="${t.ink}"`);
  s += poly([c.C, c.R, c.RB, c.B], `fill="${t.side}"`);
  s += poly([c.T, c.R, c.C, c.L], `fill="none" stroke="${t.acc}" stroke-width="2.4" stroke-linejoin="round"`);
  // kintsugi seam: a joined crack across both side faces
  const p0 = lerp(c.L, c.LB, 0.52), p4 = lerp(c.C, c.B, 0.6), p8 = lerp(c.R, c.RB, 0.5);
  const onLeft = (u, v) => [c.L[0] + (c.C[0] - c.L[0]) * u + (c.LB[0] - c.L[0]) * v, c.L[1] + (c.C[1] - c.L[1]) * u + (c.LB[1] - c.L[1]) * v];
  const onRight = (u, v) => [c.C[0] + (c.R[0] - c.C[0]) * u + (c.B[0] - c.C[0]) * v, c.C[1] + (c.R[1] - c.C[1]) * u + (c.B[1] - c.C[1]) * v];
  const seam = [p0, onLeft(0.26, 0.7), onLeft(0.5, 0.48), onLeft(0.75, 0.74), p4, onRight(0.28, 0.42), onRight(0.52, 0.7), onRight(0.76, 0.44), p8];
  s += `<polyline points="${seam.map((p) => f(p[0]) + "," + f(p[1])).join(" ")}" fill="none" stroke="${t.seam || t.acc}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>`;
  // top face: know-how as a 4×4 grid of data points
  const n = 4;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const u = (i + 0.5) / n, v = (j + 0.5) / n;
    const x = c.L[0] + (c.T[0] - c.L[0]) * u + (c.C[0] - c.L[0]) * v;
    const y = c.L[1] + (c.T[1] - c.L[1]) * u + (c.C[1] - c.L[1]) * v;
    s += `<circle cx="${f(x)}" cy="${f(y)}" r="2.7" fill="${t.acc}"/>`;
  }
  [[90, 32, 3.8], [99, 23, 3.2], [107, 15, 2.6], [114, 8, 2.0]].forEach(([x, y, r]) => (s += `<circle cx="${x}" cy="${y}" r="${r}" fill="${t.acc}"/>`));
  return s;
}

const svg = (inner, w, h, label) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${label}">${inner}</svg>\n`;

function logotype(theme, tagline) {
  const t = THEMES[theme];
  let s = mark(theme);
  s += `<text x="134" y="66" fill="${t.ink}" font-family="Arial, Helvetica, Liberation Sans, sans-serif" font-size="52" font-weight="700" letter-spacing="3">KAN<tspan fill="${t.acc}">TSUGI</tspan></text>`;
  s += `<text x="136" y="100" fill="${t.sub}" font-family="IPAGothic, Meiryo, Hiragino Sans, sans-serif" font-size="22" letter-spacing="6">カンツギ｜勘継</text>`;
  if (tagline) s += `<text x="136" y="138" fill="${t.acc}" font-family="IPAGothic, Meiryo, Hiragino Sans, sans-serif" font-size="17" letter-spacing="2">現場の勘を、次の担い手へ。</text>`;
  return svg(s, 560, tagline ? 150 : 120, "KANTSUGI カンツギ");
}

const out = (name, body) => fs.writeFileSync(path.join(__dirname, name), body);
for (const theme of Object.keys(THEMES)) {
  out(`kantsugi-cube-${theme}.svg`, svg(mark(theme), 120, 120, "KANTSUGI"));
  out(`kantsugi-logotype-${theme}.svg`, logotype(theme, false));
}
out("kantsugi-logotype-tagline.svg", logotype("color", true));
out("kantsugi-logotype-tagline-reverse.svg", logotype("reverse", true));
console.log("wrote kantsugi logo SVGs");
