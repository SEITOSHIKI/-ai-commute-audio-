// Financial model behind the 令和の虎 deck. All money in 万円.
// The base case reproduces the plan on slides 8〜11 exactly; scenarios reuse the same formulas.
const BASE = {
  price: 300,          // 年額利用料／拠点
  setup: 200,          // 導入支援（初年度のみ）／新規拠点
  licenseMargin: 0.85, // ライセンス粗利率（原価15％＝サーバ・AI推論・保守）
  serviceMargin: 1 / 3, // 導入支援・受託の粗利率（原価＝担当者2か月分の人件費 800×2/12≒133万円を振替）
  newSites: [3, 9, 18, 27, 33],
  churn: [0, 0, 0, 2, 3],
  contract: [600, 750, 1100, 550, 950], // 受託開発
  staff: [4, 7, 11, 16, 21], // 4年目から営業を3名に（年100件超の商談に2名では足りない）
  salary: 800,         // 1人あたり人件費（社会保険込み）
  other: [1200, 1530, 1900, 2270, 2900], // 家賃・広告・交通・端末・専門家・雑費
  funding: [12500, 0, 0, 0, 0],          // 公庫1,000万＋CVC・VC 1億1,500万（1年目）
  trialYear1: true,    // 1年目の3拠点は有償試験導入（ライセンス課金なし）
};

const SALES = 3; // 5年目の営業人数

// Shortfall scenario: new sites (and contract work) reach only k of the plan.
// linked = true slows hiring with them: staff = 4 + (plan − 4) × k.
function shortfall(k, linked = true) {
  const r = run({
    newSites: BASE.newSites.map((n) => Math.round(n * k)),
    contract: BASE.contract.map((c) => c * k),
    staff: linked ? BASE.staff.map((n) => 4 + Math.round((n - 4) * k)) : BASE.staff,
  });
  return { k, linked, ...r, minCash: Math.min(...r.rows.map((x) => x.cash)) };
}

function run(over = {}) {
  const a = { ...BASE, ...over };
  const rows = [];
  let sites = 0, cash = 0, cum = 0;
  for (let y = 0; y < 5; y++) {
    const prev = sites;
    const nw = a.newSites[y], ch = a.churn[y];
    sites = prev + nw - ch;
    const existing = prev * a.price;
    const newLic = a.trialYear1 && y === 0 ? 0 : nw * a.price * 0.5;
    const setup = nw * a.setup;
    const contract = a.contract[y];
    const license = existing + newLic, service = setup + contract;
    const sales = license + service;
    const cogs = license * (1 - a.licenseMargin) + service * (1 - a.serviceMargin);
    const gp = sales - cogs;
    const payroll = a.staff[y] * a.salary;
    const transfer = service * (1 - a.serviceMargin);
    const sga = payroll - transfer + a.other[y];
    const op = gp - sga;
    cum += op;
    cash += a.funding[y] + op;
    rows.push({ year: y + 1, prev, nw, ch, sites, existing, newLic, setup, contract, license, service, sales, cogs, gp,
      gpRate: gp / sales, payroll, transfer, other: a.other[y], sga, op, cum, cash, staff: a.staff[y],
      bep: sga / (gp / sales), sitesPerStaff: sites / a.staff[y] });
  }
  const firstBlack = rows.find((r) => r.op > 0);
  return { a, rows, firstBlack: firstBlack ? firstBlack.year : null, minCash: Math.min(...rows.map((r) => r.cash)), y5: rows[4] };
}

// Unit economics (per site)
const unit = (a = BASE) => {
  const gpYear = a.price * a.licenseMargin;              // 255
  const setupGp = a.setup * a.serviceMargin;              // 67
  const cac = (SALES * a.salary + 500) / 33;              // 営業3名＋広告500万 ÷ 5年目新規33
  return { gpYear, setupGp, cac, ltv: (years) => gpYear * years + setupGp, paybackMonths: cac / (gpYear / 12) };
};

module.exports = { BASE, SALES, run, unit, shortfall };

if (require.main === module) {
  const r = run();
  console.table(r.rows.map((x) => ({ y: x.year, sites: x.sites, sales: x.sales, gp: Math.round(x.gp), sga: Math.round(x.sga), op: Math.round(x.op), cum: Math.round(x.cum), cash: Math.round(x.cash), bep: Math.round(x.bep), sps: x.sitesPerStaff.toFixed(2) })));
}
