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

function customerValue(o = {}) {
  const a = { ...CUSTOMER, ...o };
  const docs = a.people * a.docHoursPerMonth * 12 * a.hourlyCost;
  const rework = a.sales * a.reworkRate * a.reworkCut;
  const coach = a.newHires * a.coachHoursPerHire * a.coachCut * a.hourlyCost;
  const total = docs + rework + coach;
  const freedHours = a.people * a.docHoursPerMonth * 12;
  return {
    a, levers: [
      { key: "docs", name: "書類・報告の時間", value: docs, formula: `${a.people}人×月${a.docHoursPerMonth}時間×12か月×4,000円` },
      { key: "rework", name: "手戻り・施工ミス", value: rework, formula: `完成工事高${a.sales / 10000}億円×手戻り${a.reworkRate * 100}%×削減${a.reworkCut * 100}%` },
      { key: "coach", name: "新人の指導時間", value: coach, formula: `新人${a.newHires}人×年${a.coachHoursPerHire}時間×削減${a.coachCut * 100}%×4,000円` },
    ],
    total, multiple: total / a.price,
    freedHours, freedPeople: freedHours / a.annualHours, capShare: a.docHoursPerMonth * 12 / a.overtimeCap,
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
  somSites: 85,                    // 5年目の拠点数
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
    somRevenue: n.somSites * 300,
  };
}

module.exports = { CUSTOMER, customerValue, NATION, nationValue };

if (require.main === module) {
  const c = customerValue();
  c.levers.forEach((l) => console.log(l.name, Math.round(l.value), "万円", l.formula));
  console.log("合計", Math.round(c.total), "万円 / 価格の", c.multiple.toFixed(1), "倍 / 浮く時間", c.freedHours, "h =", c.freedPeople.toFixed(1), "人分 / 上限の", Math.round(c.capShare * 100), "%");
  const v = nationValue();
  console.log("建設全体の手戻り", (v.reworkAll / 1e8).toFixed(2), "兆円 / 電気工事", (v.reworkElectrical / 10000).toFixed(0), "億円 / 引退予備軍", Math.round(v.retiring / 10000), "万人");
  console.log("ターゲット市場の顧客価値", (v.samValue / 10000).toFixed(0), "億円 / 5年目に届ける価値", (v.somValue / 10000).toFixed(1), "億円 / 売上", v.somRevenue / 10000, "億円");
}
