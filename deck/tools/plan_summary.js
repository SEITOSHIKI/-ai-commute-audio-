// カンツギの計算モデル（tora_v2.js）から、note記事の種に渡す要約をつくる。
// 使い方：node tools/plan_summary.js > plan_summary.json
const V = require("../tora_v2.js");
const cv = V.customerValue(), u = V.unit(), r5 = V.run(), r10 = V.run({ years: 10 });
const d15 = V.run({ plan: { devices: 15, mix: { glass: 12, display: 3 } } });
const r = Math.round;
console.log(JSON.stringify({
  slug: "kantsugi",
  business: "AIグラスで電気工事の作業を記録し、書類と育成を自動化する",
  customer: "関東の中堅電気工事会社",
  price: `1拠点${V.PLAN.devices}台で年${V.priceSite()}万円（基盤＋台数）`,
  effect: { yen: r(cv.docs), ratio: +cv.ratio.toFixed(2), breakEvenMin: r(cv.breakEvenMin) },
  unit: { grossProfit: r(u.gpYear), margin: r(V.marginSite() * 100), ltv: r(u.ltv(7)), cac: r(u.cac) },
  plan5: { sites: r5.y5.sites, salesOku: +(r5.y5.sales / 10000).toFixed(1), op5: r(r5.y5.op), black: r5.firstBlack },
  plan10: { black: r10.firstBlack, cumBack: r10.cumBack, op10: r(r10.rows[9].op) },
  blackCondition: `1拠点の台数を${V.PLAN.devices}台から15台に増やせれば、${d15.firstBlack}年目に黒字になる`,
  competitor: "日立グループ（AI Safety・日立パワーソリューションズの保守向けAI）",
  position: "大手の保守・安全ではなく、中堅の電気工事会社の施工の書類と技能の引き継ぎに絞り、月額で売る",
  weakPoints: ["普段使いのAIグラスは、落下・粉じん・安全規格の面で現場に弱い", "値段をお客様に確かめていない", `1拠点${V.PLAN.devices}台のままでは5年で黒字にならない`, "大手が同じ領域に来たら、ブランドで負ける"],
  lessons: ["お客様の効果は、使う人の分だけ数える（全社員が得をする計算にしない）", "時給は、統計の年収×1.17÷残業込みの年間総実労働時間で出す", "黒字になる値を逆算しない。赤字ならそのまま出して、黒字化の条件を示す", "同じ方向の大手を先に探し、稼ぎ方まで調べる"],
  assumptions: [
    ["電気工事従事者の年収", "551万円", "統計"], ["施工管理者の年収", "620万円", "統計"], ["年間総実労働時間", "1,938時間", "統計"],
    ["会社負担", "×1.17", "統計"], ["AIの原価", `1台 年${V.cloudPerDevice.toFixed(1)}万円`, "公表"], ["グラスの値段", "約8万円", "公表"],
    ["値段", `1拠点 年${V.priceSite()}万円`, "見積"], ["解約率", "年10%", "見積"], ["新規拠点", V.GROWTH.newSites.join("・"), "見積"],
    ["1日の削減時間", "職長15分＋施工管理者20分", "仮説"], ["手戻りの削減", "上振れとして別に示す", "仮説"],
  ],
  making: {
    angle: "研修の事業計画を、10分・16枚・A4用紙2枚にまとめるまで",
    conclusion: "詳しい本編（33枚）を先につくり、2〜3枚ずつ1枚に統合した。式と表を残したので、削っても浅くならなかった。",
    steps: ["設問①〜⑤の順に本編をつくる", "計算の過程は別冊の詳細版に分ける", "16枚に統合し、1ページ4枚で表裏2枚に印刷する", "台本を1分300字で計り、9分半に収める", "AIっぽい言い回しとデザインを、自分の言葉と図面の書き方に直す"],
    numbers: ["本編33枚 → 16枚", "台本 9分25秒", "A4用紙2枚（表裏4ページ・1ページ4枚）"],
  },
  ideas: ["保安法人と組んだ、点検記録の自動化", "書類の仕上げだけを請け負うサービス", "電気通信工事への横展開"],
  references: [],
}, null, 2));
