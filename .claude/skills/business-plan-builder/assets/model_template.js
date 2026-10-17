// 事業計画の計算モデル（ひな形）。このファイルをコピーして、事業ごとに数字を置き換える。
// 資料・台本・想定問答の数字は、すべてこのファイルから出す（手で書き換えない）。
// 金額は万円。conf（確度）：統計＝公的統計・制度／公表＝各社の公表価格／見積＝社内の見積／仮説＝実証で測る
// 例の数字は「AIグラス×電気工事」の事業（1拠点＝1社の営業所、単位＝端末）を簡単にしたもの（役割を6つに減らしてある）。
// 使い方：node model_template.js（主な表を出す）／ node model_template.js --json > plan_summary.json

// ---------- 1. お客様の1時間あたりの人件費 ----------
const WAGE = {
  worker: 551,       // 使う人Aの年収（統計：賃金構造基本統計調査の職種別。区分が無ければ近い職種で代用し、代用と書く）
  manager: 620,      // 使う人Bの年収（統計）
  employerLoad: 1.17, // 会社負担：健保 約5%＋厚年 9.15%＋雇用 約1%＋子ども・子育て拠出金 0.36%＋労災（制度）
  annualHours: 1938, // 年間総実労働時間（統計：毎月勤労統計の業種別「総実労働時間」×12。残業込み。所定の時間で割らない）
};
const hourly = (annual) => (annual * WAGE.employerLoad) / WAGE.annualHours;
const H = { worker: hourly(WAGE.worker), manager: hourly(WAGE.manager) };

// ---------- 2. 値段（1拠点） ----------
// 「会社に1つ（基盤）＋ 使う単位（台・人・席）」に分けると、台数が増えても粗利率が崩れない
const PLAN = {
  units: 10,     // 1拠点で使う単位の数（見積：使う人の分だけ。全社員にしない）
  workDays: 240, // 使う日数／年（見積）
  baseFee: 60,   // 基盤 年額（見積）
  unitFee: 24,   // 1単位 年額（見積）
  setup: 200,    // 導入支援（初年度のみ）。作業を人日で積み上げ、原価×1.4前後
  extraCost: 31.7, // お客様が別に払う周辺の費用 年額（端末のリースなど）。お客様の支払いに入れる
};
const priceSite = (p = PLAN) => p.baseFee + p.unitFee * p.units;

// ---------- 3. お客様の効果 ----------
const SAVE = { workerMin: 15, managerMin: 20 }; // 1単位1日あたりの削減（仮説：実証で測る）
const UPSIDE = 360; // 計画に入れない上振れ（例：手戻りの削減）。実証で測れてから足す
function customerValue(o = {}) {
  const s = { ...SAVE, ...o }, p = { ...PLAN, ...(o.plan || {}) };
  const perUnitDay = (s.workerMin / 60) * H.worker + (s.managerMin / 60) * H.manager;
  const value = perUnitDay * p.units * p.workDays;
  const cost = priceSite(p) + p.extraCost;
  // 同じ配分のまま、1単位1日に何分減れば元が取れるか（実証の合格ライン）
  const mix = s.workerMin + s.managerMin;
  const perMin = (H.worker * s.workerMin / mix + H.manager * s.managerMin / mix) / 60;
  const breakEvenMin = cost / (p.units * p.workDays) / perMin;
  return { perUnitDay, value, cost, ratio: value / cost, ratioUp: (value + UPSIDE) / cost, breakEvenMin };
}

// ---------- 4. 当社の原価 ----------
const COGS = { // 1単位あたり 円／月（公表単価から積み上げる）
  ai: 3400,      // 文字起こし・画像解析・文章生成
  storage: 700,  // 保存
  ops: 1900,     // 転送・監視・バックアップ・端末管理・余裕
};
const perUnitCost = Object.values(COGS).reduce((a, b) => a + b, 0) / 10000 * 12; // 万円／単位／年
const siteInfra = 12; // 拠点ごとのデータ基盤 年額
const cogsSite = (p = PLAN, perUnit = perUnitCost) => siteInfra + perUnit * p.units;
const marginSite = (p = PLAN) => 1 - cogsSite(p) / priceSite(p);

const SETUP_DAYS = 42, SETUP_TRAVEL = 15; // 導入支援の人日と交通費（作業ごとに積み上げた合計）

// ---------- 5. 体制 ----------
// [役割, 年収（数値か1〜5年目の配列）, 1〜5年目の期末人数, 任せる仕事, 求める特性]
// 個社ごとの設定や連携は外注にしない（知見が社外に残る）。社内の担当で行い、人件費は原価に振り替える
const ROLES = [
  ["代表（創業者）", [480, 600, 720, 720, 720], [1, 1, 1, 1, 1], "最初の営業・理念と判断の基準", "その業界の現場経験"],
  ["CTO", 1000, [1, 1, 1, 1, 1], "設計と開発の責任者", "同じ規模の設計経験"],
  ["エンジニア", 800, [1, 2, 3, 4, 5], "アプリ・AI処理・クラウド", "現場に足を運ぶのをいとわない"],
  ["導入コンサルタント", 600, [1, 2, 4, 6, 8], "導入支援・個社の設定と連携・定着支援", "現場と事務所の両方の言葉で話せる"],
  ["営業", 650, [0, 1, 1, 3, 3], "商談・試験導入の提案", "業界の人脈と技術の理解"],
  ["管理", 550, [0, 0, 1, 1, 2], "採用・労務・経理・契約", "少人数で幅広く回せる"],
];
const LOAD = 1.2;          // 会社負担と福利厚生
const RECRUIT_RATE = 0.15; // 採用費：年収の30% × 紹介会社経由の割合50%
const ADVISORS = 180;      // 社外顧問 年額
const CS_DAY = 600 * LOAD / 240; // 導入コンサルタントの日額（原価への振り替えに使う）

// ---------- 6. 成長と経費（1〜5年目） ----------
const GROWTH = {
  newSites: [3, 9, 18, 27, 33],          // 新規拠点（見積：導入担当1人が年に回せる数から）
  churnRate: 0.10,                       // 年間解約率（見積：根拠が海外の目安なら、そう書く）
  contract: [600, 750, 1100, 550, 950],  // 個社設定の売上（件数×単価の仮置き）
  contractMargin: 1 / 3,
  other: { // その他の経費 年額
    office: [240, 300, 360, 480, 600], ads: [200, 250, 300, 400, 500], travel: [150, 250, 300, 400, 500],
    tools: [200, 250, 300, 350, 400], experts: [100, 120, 150, 180, 200], misc: [100, 120, 190, 200, 200], ml: [0, 180, 240, 300, 360],
  },
  funding: [20000, 0, 5000, 0, 0],       // 資金の投入（自己資金・融資・出資）
};

// ---------- 7. 6〜10年目の延長（5年目までの計画は変えずに、同じ式で延ばす） ----------
const EXT = {
  newSites: [36, 38, 40, 40, 40],
  contract: [1000, 1100, 1200, 1300, 1400],
  heads: [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [6, 6, 7, 7, 8], [8, 9, 9, 9, 9], [3, 4, 4, 4, 4], [2, 2, 3, 3, 3]], // ROLES の順
  other: { office: [600, 700, 700, 800, 800], ads: [500, 550, 600, 600, 600], travel: [550, 600, 650, 700, 700],
    tools: [400, 400, 450, 450, 500], experts: [200, 220, 220, 250, 250], misc: [200, 220, 240, 260, 280], ml: [400, 440, 480, 520, 560] },
  funding: [0, 0, 0, 0, 0],
};

const headsAt = (i, y) => (y < 5 ? ROLES[i][2][y] : EXT.heads[i][y - 5]);
function team(years = 5) {
  return Array.from({ length: years }, (_, y) => {
    let staff = 0, payroll = 0, hires = 0;
    ROLES.forEach(([, sal], i) => {
      const s = Array.isArray(sal) ? sal[Math.min(y, 4)] : sal, h = headsAt(i, y), prev = y ? headsAt(i, y - 1) : 0;
      staff += h; payroll += s * h * LOAD; hires += Math.max(0, h - prev) * s;
    });
    return { year: y + 1, staff, payroll, recruit: hires * RECRUIT_RATE };
  });
}

// ---------- 8. 収支 ----------
// o: { years, newSites, contract, churnRate, funding, plan, perUnit, priceK（値段の倍率）, hireK（採用を拠点数に連動）, salaryAdd }
function run(o = {}) {
  const years = o.years || 5, ext = (a, e) => (a.length >= years ? a : a.concat(e));
  const g = { ...GROWTH, ...o, other: { ...GROWTH.other } }, p = { ...PLAN, ...(o.plan || {}) };
  if (years > 5) {
    g.newSites = ext(g.newSites, EXT.newSites); g.contract = ext(g.contract, EXT.contract); g.funding = ext(g.funding, EXT.funding);
    for (const k of Object.keys(g.other)) g.other[k] = ext(g.other[k], EXT.other[k]);
  }
  if (o.priceK) { p.baseFee *= o.priceK; p.unitFee *= o.priceK; }
  const perUnit = o.perUnit ?? perUnitCost;
  let sites = 0, cash = 0, cum = 0;
  const rows = team(years).map((t, y) => {
    const staff = o.hireK ? 4 + Math.round((t.staff - 4) * o.hireK) : t.staff, scale = staff / t.staff;
    const payroll = (t.payroll + (o.salaryAdd || 0) * LOAD * t.staff) * scale, recruit = t.recruit * scale;
    const open = sites, churn = y === 0 ? 0 : Math.round(open * g.churnRate), nw = g.newSites[y];
    sites = open - churn + nw;
    const trial = y === 0; // 1年目は試験導入（利用料なし、原価は3か月分）
    const license = (open - churn) * priceSite(p) + (trial ? 0 : nw * priceSite(p) * 0.5); // 新規は半年分
    const billed = (open - churn) + (trial ? 0 : nw * 0.5);
    const licCost = billed * cogsSite(p, perUnit) + (trial ? nw * cogsSite(p, perUnit) * 0.25 : 0);
    const setupLabor = nw * SETUP_DAYS * CS_DAY, setupCost = setupLabor + nw * SETUP_TRAVEL;
    const contract = g.contract[y], contractLabor = contract * (1 - g.contractMargin);
    const sales = license + nw * p.setup + contract;
    const cogs = licCost + setupCost + contractLabor, gp = sales - cogs;
    const other = Object.values(g.other).reduce((a, v) => a + v[y], 0) + ADVISORS;
    const sga = payroll - (setupLabor + contractLabor) + recruit + other; // 原価に振り替えた人件費を引く
    const op = gp - sga;
    cum += op; cash += g.funding[y] + op;
    return { year: y + 1, open, churn, nw, sites, license, sales, cogs, gp, gpRate: gp / sales, payroll, recruit, other, sga, op, cum, cash, staff,
      bepSales: sga / (gp / sales) }; // 損益分岐点の売上
  });
  const fb = rows.find((r) => r.op > 0), fc = rows.find((r) => r.cum > 0);
  return { rows, firstBlack: fb ? fb.year : null, cumBack: fc ? fc.year : null, minCash: Math.min(...rows.map((r) => r.cash)), last: rows[rows.length - 1] };
}

// ---------- 9. 崩れたとき ----------
const STRESS = [
  ["計画どおり", {}],
  ["値下げ −20%", { priceK: 0.8 }],
  ["解約 2倍", { churnRate: GROWTH.churnRate * 2 }],
  ["原価 2倍", { perUnit: perUnitCost * 2 }],
  ["年収 +100万円", { salaryAdd: 100 }],
  ["新規 −30%（採用も遅らせる）", { newSites: GROWTH.newSites.map((n) => Math.round(n * 0.7)), contract: GROWTH.contract.map((c) => c * 0.7), hireK: 0.7 }],
  ["新規 −30%（採用はそのまま）", { newSites: GROWTH.newSites.map((n) => Math.round(n * 0.7)), contract: GROWTH.contract.map((c) => c * 0.7) }],
];
const stress = (years = 5) => STRESS.map(([name, o]) => ({ name, ...run({ ...o, years }) }));

// 新規が計画の何割でも黒字になるか（黒字化の条件を探す。黒字になる値を逆算して計画に入れない）
const shortfall = (k, plan) => run({ plan, newSites: GROWTH.newSites.map((n) => Math.round(n * k)), contract: GROWTH.contract.map((c) => c * k), hireK: k < 1 ? k : undefined });

// 1拠点で見た損得（LTV と CAC）
function unit(p = PLAN) {
  const gpYear = priceSite(p) - cogsSite(p), setupGp = p.setup - (SETUP_DAYS * CS_DAY + SETUP_TRAVEL);
  const sales = ROLES.find((r) => r[0].startsWith("営業")), salesPay = sales[1] * LOAD * sales[2][4];
  const cac = (salesPay + GROWTH.other.ads[4]) / GROWTH.newSites[4]; // 創業者の営業時間を入れるなら、ここに足す
  const life = 1 / GROWTH.churnRate;
  return { gpYear, setupGp, cac, ltv: gpYear * life + setupGp, paybackMonths: cac / (gpYear / 12) };
}

// ---------- 10. 進む・直す・止まるの線（撤退ライン） ----------
const GATES = () => [
  ["実証（準備期）", "1単位1日の削減時間", `${Math.round(customerValue().breakEvenMin)}分以上`, "未満なら機能を絞る", "大きく下回れば撤退"],
  ["1年目末", "試験導入の本契約", "2社以上", "1社：値段と顧客を見直す", "0社：撤退か方向転換"],
  ["2年目末", "新規拠点", "5拠点以上", "未満：採用を止める", "─"],
  ["3年目", "追加の資金を入れるか", "継続率90%以上", "未達：追加額と採用を絞る", "─"],
];

// ---------- 11. 確度の表（計算詳細版の最初のほうに載せる） ----------
const CONFIDENCE = [ // [前提, 値, 確度, 懸念, 確かめ方]
  ["使う人の時給", `${H.worker.toFixed(2)}万円`, "統計", "職種の区分が近い職種の代用", "最新年の原表"],
  ["1日の削減時間", `${SAVE.workerMin}＋${SAVE.managerMin}分`, "仮説", "現場で測っていない", "実証で測る"],
  ["値段", `${priceSite()}万円／拠点・年`, "見積", "お客様に聞いていない", "試験導入で聞く"],
  ["原価", `${perUnitCost.toFixed(1)}万円／単位・年`, "公表", "使い方で増える", "上限の設計"],
  ["解約率", `${GROWTH.churnRate * 100}%`, "見積", "海外の目安", "1年目の継続率"],
];

// 記事の種に渡す要約（note_bridge.md の plan_summary.json）
function summary(meta = {}) {
  const cv = customerValue(), u = unit(), r5 = run(), r10 = run({ years: 10 });
  return { ...meta,
    effect: { yen: Math.round(cv.value), ratio: +cv.ratio.toFixed(2), breakEvenMin: Math.round(cv.breakEvenMin) },
    unit: { grossProfit: Math.round(u.gpYear), margin: Math.round(marginSite() * 100), ltv: Math.round(u.ltv), cac: Math.round(u.cac) },
    plan5: { sites: r5.last.sites, salesOku: +(r5.last.sales / 10000).toFixed(1), op5: Math.round(r5.last.op), black: r5.firstBlack },
    plan10: { black: r10.firstBlack, cumBack: r10.cumBack, op10: Math.round(r10.last.op) } };
}

module.exports = { WAGE, H, PLAN, priceSite, SAVE, UPSIDE, customerValue, COGS, perUnitCost, siteInfra, cogsSite, marginSite, SETUP_DAYS, SETUP_TRAVEL,
  ROLES, LOAD, RECRUIT_RATE, ADVISORS, CS_DAY, GROWTH, EXT, team, run, STRESS, stress, shortfall, unit, GATES, CONFIDENCE, summary };

if (require.main === module) {
  if (process.argv.includes("--json")) { console.log(JSON.stringify(summary({ slug: "example" }), null, 2)); process.exit(0); }
  const r = (v) => Math.round(v);
  console.log("時給", H, "値段", priceSite(), "粗利率", (marginSite() * 100).toFixed(0) + "%", "お客様の効果", customerValue());
  for (const years of [5, 10]) {
    const m = run({ years });
    console.table(m.rows.map((x) => ({ 年: x.year, 拠点: x.sites, 売上: r(x.sales), 粗利: r(x.gp), 販管費: r(x.sga), 利益: r(x.op), 累計: r(x.cum), 現金: r(x.cash), 人: x.staff })));
    const y = (v) => (v ? v + "年目" : "期間内にしない");
    console.log(`${years}年：黒字化 ${y(m.firstBlack)}／累計回収 ${y(m.cumBack)}／最低現金 ${r(m.minCash)}`);
  }
  console.table(stress().map((s) => ({ 場合: s.name, 黒字化: s.firstBlack ?? "─", "5年目利益": r(s.last.op), 最低現金: r(s.minCash) })));
  console.table([1, 0.9, 0.8, 0.7].map((k) => ({ 達成度: k, "10台": shortfall(k).firstBlack ?? "─", "15台": shortfall(k, { units: 15 }).firstBlack ?? "─" })));
  console.log("1拠点", unit());
}
