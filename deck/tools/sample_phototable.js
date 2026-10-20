// 個社設定で整える「報告書の型」の見本（工事写真台帳）。架空の工事。
// 使い方：node tools/sample_phototable.js → assets/sample-phototable-row.png（横長・1行）／ assets/sample-phototable-page.png（A4縦・3枚）
const path = require("path");
const sharp = require("sharp");
const NAVY = "#1B365D", INK = "#1F2933", GRAY = "#4A5563", LINE = "#B8C4D0", PALE = "#EAF0F6", BLUE = "#2F6DB5", RED = "#C62828", GREEN = "#2E7D32";
const FONT = "IPAPGothic, 'BIZ UDPGothic', sans-serif";
const t = (x, y, s, size, o = {}) => `<text x="${x}" y="${y}" font-family="${FONT}" font-size="${size}" font-weight="${o.bold ? 700 : 400}" fill="${o.fill || INK}" text-anchor="${o.anchor || "start"}" dominant-baseline="middle">${s}</text>`;
// 写真の代わりの絵：ケーブルの接続部（シース・半導電層・絶縁体）と、スケールの当て方
function photo(x, y, w, h, board) {
  const cy = y + h * 0.55;
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#2B3138"/>` +
    `<rect x="${x + w * 0.05}" y="${cy - h * 0.09}" width="${w * 0.32}" height="${h * 0.18}" fill="#3A3F44"/>` +
    `<rect x="${x + w * 0.37}" y="${cy - h * 0.075}" width="${w * 0.2}" height="${h * 0.15}" fill="#1E1E1E"/>` +
    `<rect x="${x + w * 0.57}" y="${cy - h * 0.065}" width="${w * 0.26}" height="${h * 0.13}" fill="#E8EDF2"/>` +
    `<rect x="${x + w * 0.83}" y="${cy - h * 0.03}" width="${w * 0.12}" height="${h * 0.06}" fill="#B87333"/>` +
    `<rect x="${x + w * 0.36}" y="${cy + h * 0.12}" width="${w * 0.23}" height="${h * 0.06}" fill="#F5C400"/>` +
    [0, 1, 2, 3, 4, 5].map((k) => `<line x1="${x + w * (0.36 + k * 0.046)}" y1="${cy + h * 0.12}" x2="${x + w * (0.36 + k * 0.046)}" y2="${cy + h * 0.15}" stroke="#1F2933" stroke-width="2"/>`).join("") +
    `<rect x="${x + w * 0.05}" y="${y + h * 0.06}" width="${w * 0.62}" height="${h * 0.22}" fill="#F7F7F2" stroke="#9AA5B1" stroke-width="1.5"/>` +
    t(x + w * 0.08, y + h * 0.125, board[0], Math.min(h * 0.055, w * 0.045), {}) + t(x + w * 0.08, y + h * 0.215, board[1], Math.min(h * 0.055, w * 0.045), {});
}
function rowBlock(x, y, w, h, rows, board, opt = {}) {
  const pw = w * 0.42;
  let s = photo(x, y, pw, h, board);
  const tx = x + pw, tw = w - pw, rh = h / rows.length;
  s += `<rect x="${tx}" y="${y}" width="${tw}" height="${h}" fill="#fff" stroke="${NAVY}" stroke-width="2"/>`;
  rows.forEach(([k, v, c], i) => {
    const ry = y + i * rh;
    s += `<rect x="${tx}" y="${ry}" width="${tw * 0.3}" height="${rh}" fill="${PALE}" stroke="${LINE}" stroke-width="1"/>`;
    s += `<rect x="${tx + tw * 0.3}" y="${ry}" width="${tw * 0.7}" height="${rh}" fill="#fff" stroke="${LINE}" stroke-width="1"/>`;
    s += t(tx + 14, ry + rh / 2, k, opt.fs || 26, { fill: NAVY, bold: true });
    s += t(tx + tw * 0.3 + 14, ry + rh / 2, v, opt.fs || 26, { fill: c || INK, bold: !!c });
  });
  return s;
}
const R1 = [["工種", "電線路工事 ／ ケーブル接続"], ["位置", "No.12 マンホール 直線接続（1回目）"], ["撮影日", "2026年10月20日 10:42（自動）"], ["確認項目", "半導電層の端部寸法"], ["設計／実測", "120mm ／ 112mm → 是正後 120mm", RED], ["処置", "テープ基準で測り直し（熟練者の指示を記録）"], ["撮影者", "職長（AIグラスで自動記録）"]];
const svgRow = `<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="720" viewBox="0 0 2000 720"><rect width="2000" height="720" fill="#fff"/>` +
  t(30, 40, "工事写真台帳（見本・架空の工事）", 34, { bold: true, fill: NAVY }) + t(1970, 40, "○○電力 指定様式に合わせた例", 26, { fill: GRAY, anchor: "end" }) +
  `<line x1="30" y1="72" x2="1970" y2="72" stroke="${NAVY}" stroke-width="3"/>` + rowBlock(30, 92, 1940, 560, R1, ["No.12 直線接続", "半導電層 端部 120mm"]) +
  t(30, 690, "AIが手順ごとに写真を割り当て、黒板の項目を会話と図面から埋める。書式（項目・並び・様式）は発注者ごとに個社設定で合わせる", 24, { fill: GRAY }) + `</svg>`;
const rowsFor = (k) => [
  [["工種", "電線路工事 ／ ケーブル接続"], ["位置", "No.12 直線接続 1回目"], ["確認項目", "シースのはぎ取り長さ"], ["設計／実測", "250mm ／ 250mm", GREEN], ["撮影", "10:21 職長（自動）"]],
  [["工種", "電線路工事 ／ ケーブル接続"], ["位置", "No.12 直線接続 1回目"], ["確認項目", "半導電層の端部寸法"], ["設計／実測", "120mm ／ 112mm → 120mm", RED], ["処置", "測り直し（指示を記録）"]],
  [["工種", "電線路工事 ／ ケーブル接続"], ["位置", "No.12 直線接続 1回目"], ["確認項目", "絶縁体の清掃・テーピング"], ["結果", "良（手順どおり）", GREEN], ["撮影", "11:05 職長（自動）"]],
][k];
let page = `<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1980" viewBox="0 0 1400 1980"><rect width="1400" height="1980" fill="#fff"/>` +
  t(60, 70, "工事写真台帳", 46, { bold: true, fill: NAVY }) + t(1340, 70, "1／12", 28, { fill: GRAY, anchor: "end" }) +
  `<line x1="60" y1="110" x2="1340" y2="110" stroke="${NAVY}" stroke-width="4"/>` +
  [["工事件名", "○○線 地中ケーブル布設替工事（架空）"], ["工事場所", "○○市○○町 地内"], ["請負者", "○○電設株式会社"], ["工期", "2026年10月1日〜2027年1月31日"]].map(([k, v], i) =>
    `<rect x="60" y="${130 + i * 52}" width="260" height="52" fill="${PALE}" stroke="${LINE}"/><rect x="320" y="${130 + i * 52}" width="1020" height="52" fill="#fff" stroke="${LINE}"/>` + t(80, 156 + i * 52, k, 26, { bold: true, fill: NAVY }) + t(340, 156 + i * 52, v, 26)).join("");
[0, 1, 2].forEach((k) => { page += rowBlock(60, 370 + k * 520, 1280, 480, rowsFor(k), [["No.12 直線接続 ①", "シース はぎ取り 250mm"], ["No.12 直線接続 ②", "半導電層 端部 120mm"], ["No.12 直線接続 ③", "絶縁体 清掃・テープ"]][k], { fs: 25 }); });
page += t(60, 1940, "書式は「デジタル写真管理情報基準」（電子納品）や発注者の指定様式に合わせて出力できる（個社設定）", 24, { fill: GRAY }) + `</svg>`;
(async () => {
  await sharp(Buffer.from(svgRow), { density: 72 }).png().toFile(path.join(__dirname, "../assets/sample-phototable-row.png"));
  await sharp(Buffer.from(page), { density: 72 }).png().toFile(path.join(__dirname, "../assets/sample-phototable-page.png"));
  console.log("wrote assets/sample-phototable-row.png, assets/sample-phototable-page.png");
})();
