// 本編 p.8 の流れ図（assets/shot-overview.png）を、16枚版と同じ配色（紺・青・安全色の黄）で描き直す。
// 使い方：node tools/overview_svg.js
const path = require("path");
const sharp = require("sharp");
const NAVY = "#1B365D", INK = "#1F2933", GRAY = "#4A5563", PALEY = "#FFF4C2", YEL = "#C9A000";
const FONT = "IPAPGothic, 'BIZ UDPGothic', sans-serif";
const t = (x, y, s, size, o = {}) => `<text x="${x}" y="${y}" font-family="${FONT}" font-size="${size}" font-weight="${o.bold ? 700 : 400}" fill="${o.fill || INK}" text-anchor="middle" dominant-baseline="middle">${s}</text>`;
const node = (x, y, w, h, title, sub, o = {}) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${o.fill || "#fff"}" stroke="${o.stroke || NAVY}" stroke-width="4"/>` +
  t(x + w / 2, y + h / 2 - 22, title, 36, { bold: true, fill: o.title || NAVY }) + t(x + w / 2, y + h / 2 + 30, sub, 26, { fill: o.sub || GRAY });
const arrow = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${NAVY}" stroke-width="4" marker-end="url(#a)"/>`;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="2228" height="938" viewBox="0 0 2228 938">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="${NAVY}"/></marker></defs>
<rect width="2228" height="938" fill="#fff"/>
${node(45, 335, 335, 178, "現場の作業者", "グラス・カメラ着用")}
${node(512, 314, 377, 220, "① 自動で記録", "一人称動画・会話", { fill: NAVY, title: "#fff", sub: "#fff" })}
${node(1023, 314, 377, 220, "② 整理する", "生成AI＋現場経験者", { fill: NAVY, title: "#fff", sub: "#fff" })}
${node(1567, 68, 600, 178, "若手へ（その場で）", "図面・手順を現物に重ね、差異を表示")}
${node(1567, 335, 600, 178, "管理者へ（自動報告）", "写真を抽出し、当日の報告書を作成")}
${node(1567, 602, 600, 178, "次の担い手へ（後から）", "解説動画・音声番組・研修資料")}
${node(512, 668, 888, 200, "③ 蓄積した作業データで AI が学ぶ", "フィジカルAI：ロボット・検査AIへ手順を渡す（5年目〜）", { fill: PALEY, stroke: YEL })}
${arrow(380, 424, 500, 424)}${arrow(889, 424, 1011, 424)}${arrow(1400, 424, 1555, 424)}
${arrow(1400, 390, 1555, 160)}${arrow(1400, 458, 1555, 690)}${arrow(1211, 534, 1211, 656)}
${t(1030, 603, "記録が増えるほど賢くなる", 26, { fill: GRAY })}
</svg>`;
sharp(Buffer.from(svg), { density: 72 }).png().toFile(path.join(__dirname, "../assets/shot-overview.png")).then(() => console.log("wrote assets/shot-overview.png"));
