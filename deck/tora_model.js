// Financial model used by every deck. Money in 万円.
// Since 2026-10-16 the plan runs on tora_v2.js (wage from statistics, savings per device, base + per-device pricing,
// per-device AI cost, role-based team, churn 10%/yr). This file keeps the names the deck builders import.
const V = require("./tora_v2.js");

const BASE = {
  price: V.priceSite(),                 // 年300万円／拠点（基盤60万＋端末24万×10台）
  setup: V.PLAN.setup,                  // 導入支援 200万円
  licenseMargin: V.marginSite(),        // 利用料の粗利率 72%
  setupMargin: V.unit().setupGp / V.PLAN.setup, // 導入支援の粗利率 約30%
  newSites: V.GROWTH.newSites,
  churnRate: V.GROWTH.churnRate,
  contract: V.GROWTH.contract,
  funding: V.GROWTH.funding,
  staff: V.team().map((t) => t.staff),
};
const SALES = V.SALES_HEADS;

module.exports = { BASE, SALES, run: V.run, unit: V.unit, shortfall: V.shortfall, V };

if (require.main === module) {
  const r = V.run();
  console.table(r.rows.map((x) => ({ y: x.year, sites: x.sites, sales: Math.round(x.sales), gp: Math.round(x.gp), sga: Math.round(x.sga), op: Math.round(x.op), cum: Math.round(x.cum), cash: Math.round(x.cash), sps: x.sitesPerStaff.toFixed(2) })));
}
