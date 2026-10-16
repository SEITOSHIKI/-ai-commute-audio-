// 見直し案（v2）: the numbers the 補足資料 proposes after questioning every assumption of tora_model.js (v1).
// Money in 万円 unless noted. conf: 統計＝公的統計・制度 / 公表＝メーカー・クラウドの公表価格 / 見積＝社内見積 / 仮説＝実証で測る
// Usage: node tora_v2.js  (prints the tables)

// ---------- 1. お客様の1時間あたりの人件費 ----------
const WAGE = {
  electrician: 551,     // 電気工事従事者の年収（統計：令和5年賃金構造基本統計調査 月36.74万×12＋賞与110.06万）
  manager: 620,         // 施工管理者の年収（統計で代用：建築技術者 約642万〔令和6年〕と土木技術者 約604万〔令和5年〕の間）
  employerLoad: 1.17,   // 会社負担の法定福利費 約17%（制度：健保約5.0%＋厚年9.15%＋雇用（建設）1.1%＋子ども・子育て拠出金0.36%＋労災 約1.2%）
  annualHours: 1938,    // 建設業の年間総実労働時間（統計：毎月勤労統計 令和6年 月161.5時間×12。残業を含む）
  overtimePremium: 1.25 // 残業が減る場合の割増（制度：時間外25%以上）
};
const hourly = (annual) => (annual * WAGE.employerLoad) / WAGE.annualHours; // 万円／時間
const H = { electrician: hourly(WAGE.electrician), manager: hourly(WAGE.manager) };

// ---------- 2. 端末と値段（1拠点） ----------
const PLAN = {
  devices: 10,          // 1拠点のグラスの数（見積：1班に1台、職長が掛ける。5現場×2班）
  workDays: 240,        // 端末を使う日数／年（見積）
  baseFee: 60,          // 基盤 月5万円（見積）
  deviceFee: 24,        // 端末ライセンス 月2万円／台（見積）。録画 月80時間まで、超過は従量
  setup: 200,           // 導入支援（初年度のみ）
  lease: { glass: 8 * 0.03 * 12, display: 12 * 0.03 * 12 }, // お客様が払うグラスのリース 年額：カメラ付きAIグラス約8万円（Ray-Ban Meta 7.4〜8.9万円）・表示付きARグラス約12万円（Meta Ray-Ban Display $799）×3年リース料率約3%
  mix: { glass: 8, display: 2 },  // 記録はカメラ付きAIグラス8台、手元に図面を出す班は表示付きARグラス2台（見積）
};
const priceSite = (p = PLAN) => p.baseFee + p.deviceFee * p.devices; // 300
const leaseSite = (p = PLAN) => Object.keys(p.mix).reduce((a, k) => a + p.mix[k] * p.lease[k], 0); // 31.7

// ---------- 3. お客様の得 ----------
const SAVE = { electricianMin: 15, managerMin: 20 }; // 1台1日あたりの削減（仮説）：職長の日報・撮影15分、施工管理の写真整理・台帳20分
function customerValue(o = {}) {
  const s = { ...SAVE, ...o }, p = { ...PLAN, ...(o.plan || {}) };
  const perDeviceDay = (s.electricianMin / 60) * H.electrician + (s.managerMin / 60) * H.manager; // 万円
  const docs = perDeviceDay * p.devices * p.workDays;
  const rework = 300000 * 0.02 * 0.3 * 0.2; // 完成工事高30億×手戻り2%×対象工種30%×削減20%（上振れ・計画に入れない）
  const cost = priceSite(p) + leaseSite(p);
  const breakEvenMin = (cost / (p.devices * p.workDays)) / ((H.electrician * 15 / 35 + H.manager * 20 / 35) / 60); // 同じ配分で何分減れば元が取れるか
  return { perDeviceDay, docs, rework, cost, ratio: docs / cost, ratioUp: (docs + rework) / cost, breakEvenMin,
    freedHours: ((s.electricianMin + s.managerMin) / 60) * p.devices * p.workDays };
}

// ---------- 4. 当社の原価 ----------
const CLOUD = { // 1台あたり 円／月（公表単価から見積。$1＝150円、1か月20日・1日4時間録画）
  transcribe: 2200, // 音声の文字起こし 4,800分×$0.003
  vision: 1000,     // 大事な場面の画像解析 3,000枚×約1,000トークン
  report: 200,      // 日報・報告書の文章生成
  storage: 700,     // 切り出した動画 月10GBを3年保存（S3相当 $0.025/GB）
  ops: 1900,        // 転送・監視・バックアップ・端末管理（MDM）・余裕
};
const cloudPerDevice = Object.values(CLOUD).reduce((a, b) => a + b, 0) / 10000 * 12; // 万円／台／年 = 7.2
const FULL_UPLOAD_PER_DEVICE = 5.0 * 12;  // 全編を送り毎秒解析した場合 約5万円／台／月（見積）
const siteInfra = 12;                     // 拠点あたりのデータ基盤 月1万円
const cogsSite = (p = PLAN, perDev = cloudPerDevice) => siteInfra + perDev * p.devices;
const marginSite = (p = PLAN, perDev = cloudPerDevice) => 1 - cogsSite(p, perDev) / priceSite(p);

// ---------- 5. 導入支援の中身 ----------
const SETUP_TASKS = [
  ["現状調査・業務の流れの整理", 5], ["端末の設定・アカウント・端末管理", 3], ["図面・手順・報告書の型の登録", 10],
  ["職長・施工管理者への研修（2回）", 4], ["現場での立ち会い・初期の手直し（週2日×8週）", 16], ["導入前後の効果測定レポート", 4],
];
const setupDays = SETUP_TASKS.reduce((a, [, d]) => a + d, 0); // 42人日
const setupTravel = 15; // 交通費（16回訪問×約1万円）

// ---------- 6. チーム（期末人数）と年収 ----------
const ROLES = [ // [役割, 年収, 1〜5年目の人数, 何をするか, 求める特性]
  ["代表（創業者）", [480, 600, 720, 720, 720], [1, 1, 1, 1, 1], "最初の営業・現場監修の統括・理念と判断の基準づくり", "施工管理の経験とAIの実装"],
  ["CTO", 1000, [1, 1, 1, 1, 1], "映像AI・データ基盤の設計と開発の責任者", "映像AIとクラウドを設計した経験"],
  ["エンジニア", 800, [1, 2, 3, 4, 5], "アプリ・AI処理・クラウド・グラス連携", "現場に足を運ぶのをいとわない"],
  ["現場監修（施工管理経験者）", 650, [1, 1, 1, 1, 2], "映像の意味づけ（ラベル付け）・工種の型・品質の確認", "電気工事の施工管理10年以上"],
  ["導入コンサルタント", 600, [1, 2, 4, 6, 8], "導入支援・各社の帳票と連携の設定・定着支援（当社のコンサル部隊）", "現場と事務所の両方の言葉で話せる"],
  ["営業（技術営業）", 650, [0, 1, 1, 3, 3], "商談・試験導入の提案。年30商談／人", "電設業界の人脈と技術の理解"],
  ["管理（人事・経理・総務）", 550, [0, 0, 1, 1, 2], "採用・労務・経理・契約", "少人数で幅広く回せる"],
  ["戦略企画", 700, [0, 0, 0, 1, 1], "価格・提携・大手顧客の企画", "数字で提携と価格を設計できる"],
];
const LOAD = 1.2;          // 会社負担 法定福利16%＋福利厚生・研修4%（見積）
const RECRUIT_RATE = 0.15; // 採用費：年収の30%（人材紹介の相場）×紹介会社経由の割合50%（見積）
const ADVISORS = 180;      // 社外技術顧問（大学・映像AI）月10万＋業界顧問（電気工事業OB）月5万
// 6〜10年目の延長の前提（見積）。5年目までの計画は変えずに、その先を同じ式で計算する
const EXT = {
  newSites: [36, 38, 40, 40, 40],              // 関東の中堅で頭打ち。大手の支店と電気通信工事で少しずつ伸ばす（電線工場・海外は入れない）
  contract: [1000, 1100, 1200, 1300, 1400],    // 個社設定（売上の5%以下）
  heads: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [6, 6, 7, 7, 8], [2, 3, 3, 3, 4], [8, 9, 9, 9, 9], [3, 4, 4, 4, 4], [2, 2, 3, 3, 3], [1, 1, 1, 2, 2]], // ROLES の順
  other: { office: [600, 700, 700, 800, 800], ads: [500, 550, 600, 600, 600], travel: [550, 600, 650, 700, 700], tools: [400, 400, 450, 450, 500],
    experts: [200, 220, 220, 250, 250], misc: [200, 220, 240, 260, 280], ml: [400, 440, 480, 520, 560] },
  funding: [0, 0, 0, 0, 0],
};
const headsAt = (i, y) => (y < 5 ? ROLES[i][2][y] : EXT.heads[i][y - 5]);
function team(years = 5) {
  return Array.from({ length: years }, (_, y) => y).map((y) => {
    let n = 0, pay = 0, hires = 0;
    const rows = ROLES.map(([name, sal], i) => {
      const s = Array.isArray(sal) ? sal[Math.min(y, 4)] : sal, h = headsAt(i, y), prev = y ? headsAt(i, y - 1) : 0;
      n += h; pay += s * h * LOAD; hires += Math.max(0, h - prev) * s;
      return { name, salary: s, heads: h };
    });
    return { year: y + 1, rows, staff: n, payroll: pay, recruit: hires * RECRUIT_RATE };
  });
}

// ---------- 7. 拠点数・売上・損益 ----------
const GROWTH = {
  newSites: [3, 9, 18, 27, 33], // 導入担当1人＝年6拠点の上限から（見積）
  churnRate: 0.10,              // 年10%（見積：B2B SaaSの年間解約は大企業5〜10%・中小30〜50%。乗り換えにくさを見て中間より低め）
  contract: [600, 750, 1100, 550, 950], // 個社設定：その会社の帳票・既存システムとの連携の設定（当社の導入コンサルタントが行う。件数×単価の仮置き）
  contractMargin: 1 / 3,
  other: { // その他の経費（万円／年）
    office: [240, 300, 360, 480, 600],   // オフィス賃料＋共益費（川崎 15坪→40坪、坪1.25〜1.3万円）
    ads: [200, 250, 300, 400, 500],      // 展示会・業界団体・Web・営業資料
    travel: [150, 250, 300, 400, 500],   // 営業・定着支援の交通費（導入支援の交通費は原価）
    tools: [200, 250, 300, 350, 400],    // 開発用の端末・検証機・ソフト
    experts: [100, 120, 150, 180, 200],  // 税理士・弁護士・弁理士・社労士
    misc: [100, 120, 190, 200, 200],
    ml: [0, 180, 240, 300, 360],         // AIの学習・評価の計算費（クラウドGPU 月15〜30万円、2年目から）
  },
  funding: [20000, 0, 5000, 0, 0], // 自己資金（課題の前提「豊富な資金」）：1年目2億円、3年目に撤退ラインを越えていれば5,000万円を追加。準備期の500万円は別
};
const SALES_HEADS = 3; // 5年目の営業人数
const CSDAY = 600 * LOAD / 240; // 導入コンサルタントの日額（万円）

// o: { newSites, contract, churnRate, funding, plan, perDevice, priceK, hireK (採用を拠点数に連動), salaryAdd (年収への上乗せ) }
function run(o = {}) {
  const years = o.years || 5, ext = (a, e) => (a.length >= years ? a : a.concat(e));
  const g = { ...GROWTH, ...o }, p = { ...PLAN, ...(o.plan || {}) };
  if (years > 5) {
    g.newSites = ext(g.newSites, EXT.newSites); g.contract = ext(g.contract, EXT.contract); g.funding = ext(g.funding, EXT.funding);
    g.other = Object.fromEntries(Object.entries(g.other).map(([k, v]) => [k, ext(v, EXT.other[k])]));
  }
  if (o.priceK) { p.baseFee *= o.priceK; p.deviceFee *= o.priceK; }
  const perDev = o.perDevice ?? cloudPerDevice;
  const T = team(years);
  let sites = 0, cash = 0, cum = 0;
  const rows = T.map((t, y) => {
    const planStaff = t.staff;
    const staff = o.hireK ? 4 + Math.round((planStaff - 4) * o.hireK) : planStaff;
    const scale = staff / planStaff;
    const payroll = (t.payroll + (o.salaryAdd || 0) * LOAD * planStaff) * scale, recruit = t.recruit * scale;
    const open = sites, churn = y === 0 ? 0 : Math.round(open * g.churnRate), nw = g.newSites[y];
    sites = open - churn + nw;
    const trial = y === 0;
    const existing = (open - churn) * priceSite(p);
    const newLic = trial ? 0 : nw * priceSite(p) * 0.5;
    const license = existing + newLic;
    const billed = (open - churn) + (trial ? 0 : nw * 0.5);
    const licCost = billed * cogsSite(p, perDev) + (trial ? nw * cogsSite(p, perDev) * 0.25 : 0); // 試験導入の3か月分も原価はかかる
    const setup = nw * p.setup;
    const setupLabor = nw * setupDays * CSDAY, setupCost = setupLabor + nw * setupTravel;
    const contract = g.contract[y], contractLabor = contract * (1 - g.contractMargin);
    const service = setup + contract, sales = license + service;
    const cogs = licCost + setupCost + contractLabor, gp = sales - cogs;
    const other = Object.values(g.other).reduce((a, v) => a + v[y], 0) + ADVISORS;
    const transfer = setupLabor + contractLabor;
    const sga = payroll - transfer + recruit + other;
    const op = gp - sga;
    cum += op; cash += g.funding[y] + op;
    return { year: y + 1, open, prev: open, churn, ch: churn, nw, sites, existing, stock: existing, newLic, license, setup, setupRev: setup, contract, service, sales,
      licCost, setupCost, setupLabor, contractLabor, cogs, gp, gpRate: gp / sales, payroll, recruit, other, transfer, sga, op, cum, cash, staff,
      perStaff: sga / staff, bep: sga / (gp / sales), sitesPerStaff: sites / staff, licMargin: license ? 1 - licCost / license : 0, funding: g.funding[y] };
  });
  const fb = rows.find((r) => r.op > 0);
  const fc = rows.find((r) => r.cum > 0);
  return { rows, firstBlack: fb ? fb.year : null, cumBack: fc ? fc.year : null, minCash: Math.min(...rows.map((r) => r.cash)), y5: rows[4], cumPeak: Math.min(...rows.map((r) => r.cum)) };
}

// 新規が計画のk割しか取れなかったら。linked＝採用も拠点数に合わせて遅らせる
function shortfall(k, linked = true) {
  return { k, linked, ...run({ newSites: GROWTH.newSites.map((n) => Math.round(n * k)), contract: GROWTH.contract.map((c) => c * k), hireK: linked ? k : undefined }) };
}

// 1拠点で見た損得（万円）
function unit(p = PLAN) {
  const gpYear = priceSite(p) - cogsSite(p);                    // 216
  const setupGp = p.setup - (setupDays * CSDAY + setupTravel);   // 59
  const salesPay = ROLES.find((r) => r[0].startsWith("営業"))[1] * LOAD; // 780
  const cac = (SALES_HEADS * salesPay + GROWTH.other.ads[4]) / GROWTH.newSites[4];
  return { gpYear, setupGp, cac, salesPay, ltv: (years) => gpYear * years + setupGp, paybackMonths: cac / (gpYear / 12) };
}

module.exports = { WAGE, H, PLAN, priceSite, leaseSite, SAVE, customerValue, CLOUD, cloudPerDevice, FULL_UPLOAD_PER_DEVICE, siteInfra, cogsSite, marginSite,
  SETUP_TASKS, setupDays, setupTravel, ROLES, EXT, LOAD, RECRUIT_RATE, ADVISORS, team, GROWTH, SALES_HEADS, CSDAY, run, shortfall, unit };

if (require.main === module) {
  console.log("hourly", H, "price", priceSite(), "lease", leaseSite(), "cloud/dev", cloudPerDevice, "margin", marginSite());
  for (const m of [[10, 15], [15, 20], [25, 35]]) console.log("save", m, customerValue({ electricianMin: m[0], managerMin: m[1] }));
  console.log("setupDays", setupDays);
  console.table(team().map((t) => ({ y: t.year, staff: t.staff, payroll: Math.round(t.payroll), recruit: Math.round(t.recruit) })));
  console.table(run().rows.map((r) => ({ y: r.year, open: r.open, ch: r.churn, nw: r.nw, sites: r.sites, sales: Math.round(r.sales), gp: Math.round(r.gp), gpr: (r.gpRate * 100).toFixed(0), lm: (r.licMargin * 100).toFixed(0), sga: Math.round(r.sga), op: Math.round(r.op), cum: Math.round(r.cum), cash: Math.round(r.cash) })));
}
