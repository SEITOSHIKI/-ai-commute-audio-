// Fermi estimate of the target market, segmented on two axes (industry × size).
// Each cell: customers × annual price × frequency (1/yr subscription). Money in 万円.
// Inputs marked conf "C" are assumptions to replace with e-Stat / MLIT tables.
const SEG = [
  // industry, size, customers, price, inSAM, conf, note
  ["電線・ケーブル工場", "大（300人以上）", 40, 500, true, "C", "15ライン相当。353事業所の内訳は仮定"],
  ["電線・ケーブル工場", "中（30〜299人）", 150, 300, true, "C", "標準プラン"],
  ["電線・ケーブル工場", "小（30人未満）", 163, 120, false, "C", "回収条件（月105人時）を満たしにくい"],
  ["電気工事会社", "大（300人以上）", 300, 1500, true, "C", "支店5拠点×300万円"],
  ["電気工事会社", "中（30〜299人）", 2100, 300, true, "C", "資本金5,000万円以上 約2,400社の内訳は仮定"],
  ["電気工事会社", "小（30人未満）", 63097, 60, false, "C", "65,497社の残り。ライト版を想定"],
  ["電気通信工事会社", "大（300人以上）", 100, 1500, true, "C", "支店5拠点×300万円"],
  ["電気通信工事会社", "中（30〜299人）", 600, 300, true, "C", "同規模層 約700社の内訳は仮定"],
];

// SOM: demand-side funnel for the cells targeted in years 1-5.
const SOM = [
  // cell index, region share, reach rate, close rate, note
  [1, 1.0, 0.8, 0.25, "業界が狭く、顧問・展示会で8割に会える"],
  [4, 0.35, 0.3, 0.25, "関東の比率35%（川崎拠点から日帰り圏）"],
];

function estimate() {
  const cells = SEG.map(([ind, size, n, price, inSAM, conf, note]) => ({ ind, size, n, price, inSAM, conf, note, value: n * price }));
  const tamSeg = cells.reduce((a, c) => a + c.value, 0);
  const sam = cells.filter((c) => c.inSAM).reduce((a, c) => a + c.value, 0);
  const samSites = cells.filter((c) => c.inSAM).reduce((a, c) => a + c.n, 0);
  const som = SOM.map(([i, region, reach, close, note]) => {
    const c = cells[i];
    const won = c.n * region * reach * close;
    return { cell: c.ind + "・" + c.size, n: c.n, region, reach, close, won, value: won * c.price, note };
  });
  const somSites = som.reduce((a, s) => a + s.won, 0);
  const somValue = som.reduce((a, s) => a + s.value, 0);
  return { cells, tamSeg, sam, samSites, som, somSites, somValue };
}

module.exports = { SEG, SOM, estimate };

if (require.main === module) {
  const e = estimate();
  console.table(e.cells.map((c) => ({ 業種: c.ind, 規模: c.size, 顧客数: c.n, 単価: c.price, 年額億: (c.value / 1e4).toFixed(1), SAM: c.inSAM })));
  console.log("全セル合計", (e.tamSeg / 1e4).toFixed(0), "億円 / SAM", (e.sam / 1e4).toFixed(0), "億円", e.samSites, "社");
  console.table(e.som.map((s) => ({ セル: s.cell, 顧客数: s.n, 地域: s.region, 到達: s.reach, 成約: s.close, 獲得: s.won.toFixed(1) })));
  console.log("SOM", e.somSites.toFixed(0), "拠点", (e.somValue / 1e4).toFixed(2), "億円");
}
