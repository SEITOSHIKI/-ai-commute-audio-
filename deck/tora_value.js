// What the customer gains, in yen, and how much loss it prevents nationally. Money in 万円.
// Customer = a mid-size electrical contractor (the first target): 30 site managers + 70 electricians.
// Inputs marked conf "C" are assumptions to verify in the pilot; "B" are taken from published research ranges.
const CUSTOMER = {
  people: 100,            // 施工管理者30＋電工70（C）
  hourlyCost: 0.4,        // 1時間の人件費 4,000円＝780万円÷1,950時間（A）
  docHoursPerMonth: 6,    // 1人の書類・報告の削減 月6時間（C：建設業の書類電子化の実績を準用）
  sales: 300000,          // 年間完成工事高 30億円（C：中堅100人規模の目安）
  reworkRate: 0.02,       // 手戻りが工事費に占める割合 2%（B：海外研究で0.4〜6%、RC造の事例で1.85〜2.1%）
  reworkCut: 0.2,         // 手戻りの削減率 20%（C：実証で測る）
  newHires: 5,            // 年間の新人（C）
  coachHoursPerHire: 240, // 新人1人に先輩が付く時間 年240時間＝週5時間（C）
  coachCut: 0.3,          // 指導時間の削減率 30%（C）
  price: 300,             // 当社の年額
  annualHours: 1950,
  overtimeCap: 360,       // 時間外の上限 年360時間（原則、A）
};

// Since 2026-10-16: the customer's gain is counted per device (tora_v2.js). Only the document time is in the plan;
// rework is an upside to measure, coaching is not priced.
const V = require("./tora_v2.js");
function customerValue(o = {}) {
  const c = V.customerValue(o);
  return {
    ...c,
    levers: [
      { key: "docs", name: "書類・写真整理の時間", value: c.docs, formula: "10台×240日×（職長15分＋施工管理20分）", inPlan: true },
      { key: "rework", name: "手戻り（上振れ）", value: c.rework, formula: "完成工事高30億円×2%×対象工種30%×削減20%", inPlan: false },
      { key: "coach", name: "新人の指導", value: 0, formula: "金額にせず、独り立ちの月数で測る", inPlan: false },
    ],
    total: c.docs, multiple: c.ratio, freedPeople: c.freedHours / V.WAGE.annualHours,
  };
}

// National view: rework that the industry pays every year, and the value our customers would gain.
const NATION = {
  constructionInvestment: 7557e6,  // 建設投資 2025年度見通し 75.57兆円（国土交通省、解説記事で確認）
  electricalSales: 1196.23e6,      // 電気工事業の完成工事高 2024年度 約11.96兆円（建設工事施工統計の二次集計）
  reworkRate: 0.02,
  workers: 4770000,                // 建設業就業者 477万人（2024年）
  over55: 0.367,                   // 55歳以上 36.7%
  samCompanies: 3100,              // ターゲット市場の社数（tora_market.js）
  somSites: V.run().y5.sites,      // 5年目の拠点数（81）
};
function nationValue(o = {}) {
  const n = { ...NATION, ...o };
  const c = customerValue();
  return {
    n,
    reworkAll: n.constructionInvestment * n.reworkRate,
    reworkElectrical: n.electricalSales * n.reworkRate,
    retiring: n.workers * n.over55,
    samValue: n.samCompanies * c.total,
    somValue: n.somSites * c.total,
    somRevenue: n.somSites * V.priceSite(),
  };
}

module.exports = { CUSTOMER, customerValue, NATION, nationValue };

if (require.main === module) {
  const c = customerValue();
  c.levers.forEach((l) => console.log(l.name, Math.round(l.value), "万円", l.formula));
  console.log("費用", Math.round(c.cost), "倍率", c.ratio.toFixed(2), "上振れ込み", c.ratioUp.toFixed(2), "元が取れる分", c.breakEvenMin.toFixed(1));
  const v = nationValue();
  console.log("建設全体の手戻り", (v.reworkAll / 1e8).toFixed(2), "兆円 / 電気工事", (v.reworkElectrical / 10000).toFixed(0), "億円 / 55歳以上", Math.round(v.retiring / 10000), "万人");
  console.log("5年目に届ける価値", (v.somValue / 10000).toFixed(1), "億円 / 売上", (v.somRevenue / 10000).toFixed(2), "億円");
}
