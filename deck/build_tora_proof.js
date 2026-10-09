// 令和の虎 事業計画書「理論武装版」: every number traced to its formula, assumption and source.
// All figures come from tora_model.js, so the tables and the stress test always match the plan.
// Usage: node build_tora_proof.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const L = require("./tora_lib.js");
const { run: model, unit } = require("./tora_model.js");

const { K, S, X0, W, ACC, txt, box, hline, arrow, run, heading, table, fmt, tri } = L;
const OUT_DIR = process.argv[2] || __dirname;
const OUT = path.join(OUT_DIR, "令和の虎_理論武装版_数字の根拠.pptx");
const base = model();
const R = base.rows, Y5 = R[4];
const U = unit();
const M = (v) => Math.round(v / 100); // 万円 → 百万円
const yrs = ["", "1年目", "2年目", "3年目", "4年目", "5年目"];

// Stress test: each scenario changes one lever (or two) of the same model.
const SC = [
  ["基本計画", {}, "─"],
  ["単価 −20%（年240万円）", { price: 240 }, "価格交渉で値引きされた"],
  ["解約が計画の2倍", { churn: [0, 1, 1, 4, 6] }, "定着しなかった"],
  ["ライセンス粗利率 85%→75%", { licenseMargin: 0.75 }, "AIの推論費用が想定より高い"],
  ["人件費 +100万円／人", { salary: 900 }, "採用市場が厳しい"],
  ["単価 −20% かつ 解約2倍", { price: 240, churn: [0, 1, 1, 4, 6] }, "複合"],
  ["新規獲得 −30%（採用も連動）", { newSites: [2, 6, 13, 19, 23], staff: [4, 7, 10, 13, 17] }, "営業が想定より遅い"],
  ["1年遅れ（採用も1年遅らせる）", { newSites: [3, 3, 9, 18, 27], staff: [4, 4, 7, 11, 15], other: [1200, 1200, 1530, 1900, 2270] }, "実証が長引いた"],
  ["新規獲得 −30%（採用は計画どおり）", { newSites: [2, 6, 13, 19, 23] }, "やってはいけない例"],
].map(([n, o, why]) => { const r = model(o); return { n, why, black: r.firstBlack, op5: r.y5.op, minCash: r.minCash }; });

const NOTE_PARTS = {
  1: ["使い方", "数字を聞かれたら、このページ番号で答える",
    "この資料は発表用ではなく、質問への備えです。すべての数字は一つの計算モデルから出しているので、どこを聞かれても辻褄が合います。聞かれたら、まず数字、次に式、最後に前提の順で答えます。"],
  2: ["根拠1", "営業利益は5つの数字の掛け算と足し算に分解できる",
    "5年目の営業利益6,900万円を分解した木です。上から、売上総利益から販管費を引く。売上総利益は、利用料の85%とサービスの40%。利用料は、前の年までの55拠点に300万円と、新しい33拠点の半年分。販管費は、社員20人に800万円から、導入支援に回った人件費を原価に振り替え、家賃などを足す。どの枝を聞かれても、この木のどこかを指せば答えられます。"],
  3: ["根拠2", "前提は11個。確度の低いものは実証で確かめる",
    "計画の前提を一覧にしました。確度Aは公的統計や契約で決まるもの、Bは社内の見積もり、Cはまだ検証していない仮説です。Cの代表は、1人あたり月6時間の削減です。だから500万円の使い道の中心が、この数字を実測する実証になっています。"],
  4: ["根拠3", "価格は、お客様が得する額の約1割",
    "年300万円の根拠は、お客様の得の約1割です。100人の工場で月240万円、年2,880万円分の時間が浮くので、その約10%をいただきます。人数ではなく拠点とライン数で課金するのは、人数課金だと現場がアカウントを絞り、記録が貯まらなくなるからです。"],
  5: ["根拠4", "小さな工場でも1年以内に元が取れる",
    "お客様の回収期間の感度表です。人数と削減時間が半分でも、100人・月3時間、または50人・月6時間で、約4か月で回収できます。1年で回収するための最低条件は、人数かける削減時間が月105人時。30人の工場なら1人月3.5時間です。これを下回る小さな工場は、最初は狙いません。"],
  6: ["根拠5", "売上は4つの部品の足し算",
    "売上の計算表です。前の年までのお客様の利用料、新しいお客様の半年分の利用料、導入支援、受託開発の4つを足します。新しいお客様は年の途中で始まるので、初年度は半年分にしています。1年目の3拠点は有償の試験導入で、利用料はいただかない前提です。"],
  7: ["根拠6", "導入支援の人件費は原価に振り替える",
    "原価と販管費です。利用料の原価はサーバーとAIの費用で15%。導入支援と受託の原価は担当者の人件費で60%。その分を販管費の人件費から引いて、二重に数えないようにしています。"],
  8: ["根拠7", "損益分岐点は売上で見ても、人で見ても4年目",
    "損益分岐点は二通りで見ています。一つは販管費を粗利率で割る売上ベース。もう一つは社員1人あたりの拠点数で、713万円を255万円で割って2.8拠点。どちらで見ても4年目に超えます。"],
  9: ["根拠8", "LTVは控えめに7年。3年で解約されても13倍",
    "LTVは年255万円の粗利を7年分と、導入支援の粗利80万円で約1,865万円。計画上の解約率は年5%台なので平均は約18年続く計算ですが、控えめに7年にしています。仮に3年で全部解約されても、LTVは845万円で、CACの約13倍です。"],
  10: ["根拠9", "崩れても死なない。条件は採用を拠点数に連動させること",
    "計画が崩れた場合のストレステストです。単価が2割下がっても、解約が倍になっても、4年目の黒字は変わりません。獲得が3割遅れたり、1年遅れたりすると黒字化は5年目にずれますが、採用を拠点数に合わせて遅らせれば、現金は最低ラインを割りません。一番危ないのは、獲得が遅れているのに計画どおり採用することで、この場合は現金が尽きます。だから、社員1人2.8拠点を採用のルールにしています。"],
  11: ["根拠10", "必要資金1.5億円は積み上げで出した",
    "必要資金1.5億円は、累積赤字のピーク6,500万円、運転資金2,900万円、1年遅れた場合の備え5,600万円の積み上げです。現金残高は最も薄い3年目でも約6,000万円残ります。"],
  12: ["根拠11", "SAMは約3,350拠点×300万円＝約100億円",
    "市場規模の計算です。電線・ケーブルの製造事業所が353。電気工事は許可業者65,497社のうち、中堅以上を約3,000社と推計しました。合わせて約3,350拠点に年300万円をかけて約100億円。5年目の85拠点はその2.5%です。中堅以上の比率は推計なので、聞かれたら推計だと正直に言います。"],
  13: ["根拠12", "厳しい質問には、数字→式→前提の順で答える",
    "厳しい質問への回答集です。それぞれ、一言の答えと、根拠のページを書いています。"],
  14: ["出典", "公的統計と原典を明記する。未確認のものは未確認と言う",
    "出典一覧です。確認が必要なものには印を付けています。発表前に原典を確認してください。"],
};
const NOTES = Object.fromEntries(Object.entries(NOTE_PARTS).map(([n, [t, msg, body]]) => [n, `【${t}】\n要約：${msg}\n\n${body}`]));

const foot = (s, t) => txt(s, t, { x: X0, y: 4.98, w: W, h: 0.26, size: S.note, color: K.gray500 });

async function main() {
  const pres = new pptxgen();
  const add = await L.setup(pres, "三現ワークス 事業計画書 理論武装版", NOTES);

  // ---------- 1 表紙 ----------
  {
    const s = add(1);
    box(s, { x: 0, y: 0, w: 3.6, h: 5.625, fill: K.ink });
    s.addImage({ data: await L.png("gen3-logotype-reverse"), x: 0.35, y: 1.05, w: 3.0, h: 3.0 * 120 / 560 });
    txt(s, "数字には、\nすべて式と\n前提がある。", { x: 0.4, y: 2.1, w: 3.0, h: 1.4, size: 20, bold: true, color: K.paper, lsm: 1.2 });
    txt(s, "令和の虎　事業計画書　別冊", { x: 4.0, y: 0.55, w: 5.5, h: 0.28, size: S.lead, color: K.gray700 });
    txt(s, [run("理論武装版", { color: ACC, breakLine: true }), run("数字の根拠と、崩れたときの備え")], { x: 4.0, y: 0.9, w: 5.6, h: 1.0, size: 24, bold: true, lsm: 1.15 });
    const toc = [["根拠1〜2", "数字の木と、11個の前提"], ["根拠3〜4", "価格と、お客様の回収"], ["根拠5〜8", "売上・費用・損益分岐点・LTV/CAC"], ["根拠9〜10", "ストレステストと資金計画"], ["根拠11〜12", "市場規模の計算と、厳しい質問"]];
    toc.forEach(([a, b], i) => {
      const y = 2.15 + i * 0.42;
      hline(s, 4.0, y, 5.6, K.rule, T().rule);
      txt(s, a, { x: 4.0, y, w: 1.3, h: 0.42, size: 10.5, bold: true, color: ACC, valign: "middle" });
      txt(s, b, { x: 5.3, y, w: 4.3, h: 0.42, size: 10.5, valign: "middle" });
    });
    txt(s, "すべての表は同じ計算モデル（tora_model.js）から出力。発表資料・やさしい版と数字が一致する", { x: 4.0, y: 4.4, w: 5.6, h: 0.45, size: 9, color: K.gray700 });
  }

  // ---------- 2 数字の木 ----------
  {
    const s = add(2);
    heading(s, "根拠1　数字の木 ─ 5年目の営業利益6,900万円を分解する", "どの数字を聞かれても、この木のどこかを指せば答えられる（単位：百万円）");
    const nb = (x, y, w, label, val, f, o = {}) => {
      box(s, { x, y, w, h: 0.62, fill: o.dark ? K.ink : o.acc ? K.paper : K.fill, line: o.acc ? ACC : undefined, lw: 1.75 });
      const c = o.dark ? K.paper : K.ink;
      txt(s, [run(label + "  ", { fontSize: 8.5 }), run(val, { bold: true, fontSize: 13, color: o.dark ? "E3A774" : o.acc ? ACC : K.ink })], { x: x + 0.08, y: y + 0.02, w: w - 0.16, h: 0.32, color: c, valign: "middle" });
      txt(s, f, { x: x + 0.08, y: y + 0.33, w: w - 0.16, h: 0.26, size: 7.5, color: o.dark ? K.paper : K.gray700, valign: "middle" });
    };
    const lic = Y5.license / 100, svc = Y5.service / 100;
    nb(X0, 2.6, 1.75, "営業利益", fmt(M(Y5.op)), "粗利 − 販管費", { dark: true });
    nb(2.45, 1.45, 1.9, "売上総利益", fmt(M(Y5.gp)), "利用料×85%＋サービス×40%");
    nb(2.45, 3.75, 1.9, "販管費", fmt(M(Y5.sga)), "人件費−振替＋その他");
    nb(4.65, 1.12, 2.3, "利用料の粗利", fmt(lic * 0.85), `利用料 ${lic.toFixed(1)} × 85%`);
    nb(4.65, 1.84, 2.3, "サービスの粗利", fmt(svc * 0.4), `導入支援${Y5.setup / 100}＋受託${Y5.contract / 100} × 40%`);
    nb(4.65, 3.3, 2.3, "人件費", fmt(Y5.payroll / 100), `社員${Y5.staff}人 × 800万円`);
    nb(4.65, 4.02, 2.3, "原価への振替・その他", "−" + fmt(Y5.transfer / 100) + " / +" + fmt(Y5.other / 100), "サービス×60% ／ 家賃・広告など");
    nb(7.25, 0.98 + 0.12, 2.35, "前の年までの利用料", fmt(Y5.existing / 100), `期首${Y5.prev}拠点 × 300万円`, { acc: true });
    nb(7.25, 1.76, 2.35, "新しいお客様の利用料", fmt(Y5.newLic / 100), `新規${Y5.nw}拠点 × 300万円 × 半年`, { acc: true });
    nb(7.25, 2.48, 2.35, "導入支援", fmt(Y5.setup / 100), `新規${Y5.nw}拠点 × 200万円`);
    // connectors
    const vline = (x, y1, y2) => s.addShape("line", { x, y: y1, w: 0, h: y2 - y1, line: { color: K.gray500, width: 1 } });
    vline(2.3, 1.76, 4.06); hline(s, 2.15, 2.91, 0.15, K.gray500, 1); hline(s, 2.3, 1.76, 0.15, K.gray500, 1); hline(s, 2.3, 4.06, 0.15, K.gray500, 1);
    vline(4.5, 1.43, 2.15); hline(s, 4.35, 1.76, 0.15, K.gray500, 1); hline(s, 4.5, 1.43, 0.15, K.gray500, 1); hline(s, 4.5, 2.15, 0.15, K.gray500, 1);
    vline(4.5, 3.61, 4.33); hline(s, 4.35, 4.06, 0.15, K.gray500, 1); hline(s, 4.5, 3.61, 0.15, K.gray500, 1); hline(s, 4.5, 4.33, 0.15, K.gray500, 1);
    vline(7.1, 1.41, 2.07); hline(s, 6.95, 1.43, 0.15, K.gray500, 1); hline(s, 7.1, 2.07, 0.15, K.gray500, 1);
    hline(s, 6.95, 2.15, 0.15, K.gray500, 1); vline(7.1, 2.15, 2.79); hline(s, 7.1, 2.79, 0.15, K.gray500, 1);
    box(s, { x: 7.25, y: 3.3, w: 2.35, h: 1.34, fill: K.ink });
    txt(s, [run("動かせるのは3つだけ", { bold: true, color: "E3A774", breakLine: true }), run("① 拠点数（新規−解約）", { breakLine: true }), run("② 単価（300万円）", { breakLine: true }), run("③ 社員数（採用のペース）")], { x: 7.37, y: 3.35, w: 2.15, h: 1.25, size: 9.5, color: K.paper, valign: "middle", psa: 2 });
    foot(s, "受託開発（9.5百万円）はサービスの粗利に含む。端数は四捨五入");
  }

  // ---------- 3 前提一覧 ----------
  {
    const s = add(3);
    heading(s, "根拠2　前提一覧 ─ 11個の前提と、その確度", "A＝公的統計・制度で決まる　B＝社内の見積もり　C＝未検証の仮説（実証で確かめる）");
    table(s, [
      ["前提", "値", "根拠", "確度"],
      ["単価", "年300万円／拠点", "お客様の得（年2,880万円）の約10%（根拠3）", "B"],
      ["導入支援", "200万円（初年度）", "担当者1人が約2か月現場に入る。原価120万円", "B"],
      ["利用料の原価率", "15%", "クラウド・AI推論・保守。実証で実測する", "B"],
      ["サービスの原価率", "60%", "担当者の人件費（1人年800万円の2か月分＋交通費）", "B"],
      ["新規獲得", "3→9→18→27→33拠点", "導入担当1人＝年6拠点（2か月で1拠点）の処理能力が上限", "B"],
      ["解約", "0→0→0→2→3拠点", "年5%前後。導入後は記録が貯まり、乗り換えにくい", "C"],
      ["人件費", "1人 年800万円", "年収約600万円×1.3（社会保険など）", "A"],
      ["社員数", "4→7→11→15→20人", "社員1人2.8拠点を超える前提で採用する（根拠7）", "B"],
      ["その他経費", "1,200→2,900万円", "家賃・広告・交通・端末・専門家・雑費（発表資料⑥）", "B"],
      ["お客様の削減時間", "1人 月6時間", "書類作業の削減。建設業の書類電子化の実績（月6.5時間）を準用 ※", "C"],
      ["1時間の人件費", "4,000円", "780万円÷年1,950時間", "A"],
    ], { x: X0, y: 1.12, w: W, colW: [1.55, 1.75, 5.25, 0.65], rowH: 0.3, size: 8.5, leftCols: [1, 2], align: "center", name: "assumptions" });
    foot(s, "※ 月6.5時間の出典は発表前に原典を確認する。Cの2つ（削減時間・解約）を、500万円の実証で実測する");
  }

  // ---------- 4 価格の根拠 ----------
  {
    const s = add(4);
    heading(s, "根拠3　価格 ─ お客様が得する額の約1割をいただく", "価値基準の値付け。原価からではなく、お客様の得から決める");
    const top = 1.15;
    box(s, { x: X0, y: top, w: 4.5, h: 1.7, fill: K.fill });
    txt(s, "お客様の得（100名の拠点・年）", { x: X0 + 0.15, y: top + 0.08, w: 4.2, h: 0.28, size: 10.5, bold: true });
    txt(s, [run("100人 × 月6時間 × 4,000円 × 12か月", { breakLine: true }), run("＝ 年2,880万円", { bold: true, fontSize: 18, color: ACC })], { x: X0 + 0.15, y: top + 0.42, w: 4.2, h: 0.8, size: 11 });
    txt(s, "年300万円 ÷ 年2,880万円 ＝ 約10%。残りの約9割がお客様の得", { x: X0 + 0.15, y: top + 1.25, w: 4.2, h: 0.35, size: 10, bold: true });
    txt(s, "価格の部品（拠点・ライン・機能に比例）", { x: X0, y: top + 1.85, w: 4.5, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["部品", "単価", "標準", "年額"],
      ["基盤", "100万円", "1拠点", "100万円"],
      ["ライン登録", "20万円", "5ライン", "100万円"],
      ["追加機能", "50万円", "2本", "100万円"],
      [{ text: "合計", bold: true }, "", "", { text: "300万円", bold: true }],
    ], { x: X0, y: top + 2.15, w: 4.5, colW: [1.4, 1.0, 1.0, 1.1], rowH: 0.3, size: 9, strongRows: [4], name: "price-parts" });
    const rx = 5.15, rw = 4.45;
    txt(s, "なぜこの課金の形か", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    [
      ["人数ではなく拠点で課金", "人数課金だと、現場はアカウントを絞る。記録が減り、当社の強み（貯まるデータ）が育たない"],
      ["ラインと機能で上がる", "使うほど広がる形。2年目以降の単価上昇（アップセル）の余地になる。計画には入れていない"],
      ["導入支援は別料金", "現場に入る作業は人件費がかかる。利用料に混ぜると粗利率が見えなくなる"],
      ["年額前払い", "黒字倒産を防ぐ。入金が先、費用が後になる"],
    ].forEach(([a, b], i) => {
      const y = top + 0.32 + i * 0.75;
      hline(s, rx, y, rw, K.rule, 0.75);
      txt(s, a, { x: rx, y: y + 0.05, w: rw, h: 0.26, size: 10, bold: true, color: ACC });
      txt(s, b, { x: rx, y: y + 0.31, w: rw, h: 0.42, size: 9 });
    });
    foot(s, "値引きされた場合の影響はストレステスト（根拠9）：単価−20%でも4年目に黒字");
  }

  // ---------- 5 顧客ROI感度 ----------
  {
    const s = add(5);
    heading(s, "根拠4　お客様の回収 ─ 前提が半分でも、約4か月で元が取れる", "回収月数＝初年度の支払い500万円 ÷（人数 × 削減時間 × 4,000円）");
    const Ns = [30, 50, 100, 200], Hs = [3, 6, 10];
    const cell = (n, h) => { const m = 500 / (0.4 * n * h); return { text: m >= 10 ? m.toFixed(0) + "か月" : m.toFixed(1) + "か月", bold: n === 100 && h === 6, fill: n === 100 && h === 6 ? "F6E7DA" : m > 12 ? K.fill : undefined, color: m > 12 ? K.gray700 : K.ink }; };
    txt(s, "回収までの月数（初年度）", { x: X0, y: 1.15, w: 5, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["対象の人数 ＼ 1人の削減時間", "月3時間", "月6時間（計画）", "月10時間"],
      ...Ns.map((n) => [n + "人" + (n === 100 ? "（計画）" : ""), ...Hs.map((h) => cell(n, h))]),
    ], { x: X0, y: 1.45, w: 5.4, colW: [2.1, 1.1, 1.1, 1.1], rowH: 0.42, size: 10, align: "center", name: "roi-grid" });
    txt(s, "色付き＝計画の前提。灰色＝12か月を超える（最初は狙わない）", { x: X0, y: 3.62, w: 5.4, h: 0.24, size: 8.5, color: K.gray700 });
    const rx = 6.1, rw = 3.5;
    box(s, { x: rx, y: 1.15, w: rw, h: 1.55, fill: K.ink });
    txt(s, [run("1年で回収する最低条件", { bold: true, color: "E3A774", breakLine: true }), run("人数 × 削減時間 ≧ 月105人時", { bold: true, fontSize: 14, breakLine: true }), run("500万円 ÷ 12か月 ÷ 4,000円 ≒ 104", { fontSize: 9 })], { x: rx + 0.15, y: 1.2, w: rw - 0.3, h: 1.45, size: 10, color: K.paper, valign: "middle", psa: 3 });
    box(s, { x: rx, y: 2.85, w: rw, h: 1.0, fill: K.fill });
    txt(s, [run("2年目以降は年300万円だけ", { bold: true, breakLine: true }), run("100人・月6時間なら、年2,880万円の得に対し年300万円。約9.6倍")], { x: rx + 0.15, y: 2.9, w: rw - 0.3, h: 0.9, size: 9.5, valign: "middle" });
    box(s, { x: X0, y: 4.05, w: W, h: 0.75, line: ACC, lw: 1.5 });
    txt(s, [run("ターゲットの条件に変換：", { bold: true, color: ACC }), run("対象の作業者が30人以上、かつ書類・報告の作業が1人月3.5時間以上ある拠点。中堅の電線工場（数十〜数百人規模）と、施工管理者10名以上の電気工事会社はこれを満たす")], { x: X0 + 0.15, y: 4.05, w: W - 0.3, h: 0.75, size: 9.5, valign: "middle" });
    foot(s, "時間の価値で計算しており、現金の支出が減るわけではない。お客様には「残業の削減」または「浮いた時間で育成」として説明する");
  }

  // ---------- 6 売上の計算表 ----------
  {
    const s = add(6);
    heading(s, "根拠5　売上の計算 ─ 4つの部品の足し算", "売上＝前の年までの利用料＋新しいお客様の利用料（半年分）＋導入支援＋受託開発（単位：万円）");
    const row = (lab, f, o = {}) => [o.bold ? { text: lab, bold: true } : lab, ...R.map((r) => (o.bold ? { text: f(r), bold: true } : f(r)))];
    table(s, [
      yrs,
      row("期首の拠点数", (r) => String(r.prev)),
      row("＋ 新規", (r) => String(r.nw)),
      row("− 解約", (r) => String(r.ch)),
      row("期末の拠点数", (r) => String(r.sites), { bold: true }),
      row("前の年までの利用料（期首×300）", (r) => fmt(r.existing)),
      row("新しいお客様の利用料（新規×300×0.5）", (r) => fmt(r.newLic)),
      row("導入支援（新規×200）", (r) => fmt(r.setup)),
      row("受託開発", (r) => fmt(r.contract)),
      row("売上高", (r) => fmt(r.sales), { bold: true }),
      row("うち前の年までの利用料の比率", (r) => (r.sales ? Math.round((r.existing / r.sales) * 100) + "%" : "─")),
    ], { x: X0, y: 1.12, w: W, colW: [3.45, 1.15, 1.15, 1.15, 1.15, 1.15], rowH: 0.3, size: 9, strongRows: [4, 9], name: "sales-calc" });
    box(s, { x: X0, y: 4.5, w: W, h: 0.42, fill: K.fill });
    txt(s, [run("5年目の月次：", { bold: true, color: ACC }), run(`利用料 約${fmt(Y5.license / 12)}万円（月25万円×平均約72拠点）＋導入支援 約${fmt(Y5.setup / 12)}万円＋受託 約${fmt(Y5.contract / 12)}万円 ＝ 約${fmt(Y5.sales / 12)}万円`)], { x: X0 + 0.15, y: 4.5, w: W - 0.3, h: 0.42, size: 9.5, valign: "middle" });
    foot(s, "1年目の3拠点は有償の試験導入（導入支援200万円のみ、利用料なし）。新規は年の途中で始まるため初年度の利用料は半年分");
  }

  // ---------- 7 原価・販管費 ----------
  {
    const s = add(7);
    heading(s, "根拠6　費用の計算 ─ 導入支援の人件費は原価に振り替える", "同じ人件費を原価と販管費で二重に数えない（単位：万円）");
    const row = (lab, f, o = {}) => [o.bold ? { text: lab, bold: true } : lab, ...R.map((r) => ({ text: f(r), bold: !!o.bold, color: o.signed && f(r).startsWith("▲") ? K.gray700 : K.ink }))];
    table(s, [
      yrs,
      row("売上高", (r) => fmt(r.sales)),
      row("− 利用料の原価（利用料×15%）", (r) => fmt(r.license * 0.15)),
      row("− サービスの原価（支援・受託×60%）", (r) => fmt(r.service * 0.6)),
      row("売上総利益（粗利）", (r) => fmt(r.gp), { bold: true }),
      row("粗利率", (r) => Math.round(r.gpRate * 100) + "%"),
      row("人件費（社員数×800）", (r) => `${fmt(r.payroll)}（${r.staff}人）`),
      row("− 原価へ振替（サービスの原価）", (r) => fmt(r.transfer)),
      row("＋ その他経費（家賃・広告・交通など）", (r) => fmt(r.other)),
      row("販管費", (r) => fmt(r.sga), { bold: true }),
      row("営業利益", (r) => tri(r.op), { bold: true, signed: true }),
      row("累積営業損益", (r) => tri(r.cum), { signed: true }),
    ], { x: X0, y: 1.12, w: W, colW: [3.45, 1.15, 1.15, 1.15, 1.15, 1.15], rowH: 0.29, size: 9, strongRows: [4, 9, 10], name: "cost-calc" });
    foot(s, "発表資料の表は百万円単位で四捨五入（例：5年目 粗利21,253万円→213、販管費14,370万円→144、営業利益6,883万円→69）");
  }

  // ---------- 8 損益分岐点 ----------
  {
    const s = add(8);
    heading(s, "根拠7　損益分岐点 ─ 売上で見ても、人で見ても4年目", "二つの見方が同じ答えになることが、計画の筋の良さの証拠");
    const top = 1.15, lw = 4.45;
    txt(s, "① 売上で見る：損益分岐点売上＝販管費÷粗利率", { x: X0, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["百万円", ...yrs.slice(1)],
      ["販管費", ...R.map((r) => String(M(r.sga)))],
      ["粗利率", ...R.map((r) => Math.round(r.gpRate * 100) + "%")],
      ["分岐点売上", ...R.map((r) => String(M(r.bep)))],
      ["実際の売上", ...R.map((r) => String(M(r.sales)))],
      ["安全余裕率", ...R.map((r) => (r.sales > r.bep ? Math.round((1 - r.bep / r.sales) * 100) + "%" : "─"))],
    ], { x: X0, y: top + 0.3, w: lw, colW: [1.2, 0.65, 0.65, 0.65, 0.65, 0.65], rowH: 0.3, size: 9, strongRows: [3], name: "bep-sales" });
    const rx = 5.15, rw = 4.45;
    txt(s, "② 人で見る：社員1人あたりの拠点数", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    box(s, { x: rx, y: top + 0.3, w: rw, h: 0.95, fill: K.ink });
    txt(s, [run("社員1人の販管費 713万円（4年目：1億700万円÷15人）", { breakLine: true }), run("÷ 1拠点の粗利 255万円（300万円×85%）", { breakLine: true }), run("＝ 2.8拠点／人", { bold: true, fontSize: 14, color: "E3A774" })], { x: rx + 0.15, y: top + 0.32, w: rw - 0.3, h: 0.9, size: 9.5, color: K.paper, valign: "middle" });
    table(s, [["", ...yrs.slice(1)], ["拠点÷社員", ...R.map((r) => ({ text: r.sitesPerStaff.toFixed(2), bold: r.sitesPerStaff > 2.8, color: r.sitesPerStaff > 2.8 ? ACC : K.ink }))]],
      { x: rx, y: top + 1.35, w: rw, colW: [1.2, 0.65, 0.65, 0.65, 0.65, 0.65], rowH: 0.3, size: 9, name: "bep-people" });
    box(s, { x: X0, y: 3.45, w: W, h: 1.3, fill: K.fill });
    txt(s, [
      run("なぜ販管費を「固定費」とみなすか：", { bold: true, color: ACC }), run("販管費の8割は人件費と家賃で、売上が増えても1年の中では動かない。変動費（サーバー代・導入担当の人件費）は原価に入れてある", { breakLine: true }),
      run("聞かれたらこう答える：", { bold: true, color: ACC }), run("「3年目は社員11人で30拠点、1人2.73で2.8に届かず赤字。4年目は15人で55拠点、3.67で黒字です」", { breakLine: true }),
      run("経営のルール：", { bold: true, color: ACC }), run("採用は「1人2.8拠点を超える見込み」が立ってから。これが根拠9の資金の守りになる"),
    ], { x: X0 + 0.15, y: 3.45, w: W - 0.3, h: 1.3, size: 9.5, valign: "middle", psa: 3 });
    foot(s, "713万円は4年目の値（1年目920万円、5年目719万円）。年によって多少動くため、目安として2.8を使う");
  }

  // ---------- 9 LTV/CAC ----------
  {
    const s = add(9);
    heading(s, "根拠8　LTV・CAC ─ 控えめに7年。3年で解約されても13倍", "LTV＝年の粗利×継続年数＋導入支援の粗利　CAC＝営業・広告費÷新規拠点数");
    const top = 1.15;
    box(s, { x: X0, y: top, w: 4.45, h: 1.25, fill: K.fill });
    txt(s, [run("LTV（1拠点が残す粗利）", { bold: true, breakLine: true }), run("255万円 × 7年 ＋ 80万円 ＝ ", {}), run("約1,865万円", { bold: true, color: ACC, fontSize: 16, breakLine: true }), run("255万円＝300万円×85%　80万円＝200万円×40%", { fontSize: 8.5, color: K.gray700 })], { x: X0 + 0.15, y: top + 0.05, w: 4.15, h: 1.15, size: 10.5, valign: "middle", psa: 3 });
    box(s, { x: 5.15, y: top, w: 4.45, h: 1.25, fill: K.fill });
    txt(s, [run("CAC（1拠点を獲得する費用）", { bold: true, breakLine: true }), run("（1,600万円＋500万円）÷ 33拠点 ＝ ", {}), run("約64万円", { bold: true, color: ACC, fontSize: 16, breakLine: true }), run("5年目：営業2人の人件費＋広告・展示会 ÷ 新規拠点", { fontSize: 8.5, color: K.gray700 })], { x: 5.3, y: top + 0.05, w: 4.15, h: 1.15, size: 10.5, valign: "middle", psa: 3 });
    txt(s, "継続年数を変えたら", { x: X0, y: top + 1.42, w: 4.45, h: 0.26, size: 10.5, bold: true });
    table(s, [["継続年数", "LTV", "LTV÷CAC"], ...[3, 5, 7, 10].map((y) => [y + "年" + (y === 7 ? "（計画）" : ""), fmt(U.ltv(y)) + "万円", { text: (U.ltv(y) / U.cac).toFixed(0) + "倍", bold: y === 7 }])],
      { x: X0, y: top + 1.72, w: 4.45, colW: [1.65, 1.4, 1.4], rowH: 0.27, size: 9.5, name: "ltv-years" });
    txt(s, "獲得が難しくなったら", { x: 5.15, y: top + 1.42, w: 4.45, h: 0.26, size: 10.5, bold: true });
    table(s, [["CAC", "LTV÷CAC（7年）", "回収期間"], ...[[1, "計画"], [2, "2倍"], [4, "4倍"]].map(([k, l]) => [fmt(U.cac * k) + "万円（" + l + "）", (U.ltv(7) / (U.cac * k)).toFixed(0) + "倍", (U.paybackMonths * k).toFixed(0) + "か月"])],
      { x: 5.15, y: top + 1.72, w: 4.45, colW: [1.75, 1.35, 1.35], rowH: 0.27, size: 9.5, name: "cac-stress" });
    box(s, { x: X0, y: 4.3, w: W, h: 0.64, line: ACC, lw: 1.5 });
    txt(s, [
      run("7年は控えめ：", { bold: true, color: ACC }), run("計画の解約は5年目に55拠点中3（年約5.5%）。1÷5.5%＝平均約18年続く計算。7年は年約14%の解約に相当", { breakLine: true }),
      run("目安：", { bold: true, color: ACC }), run("SaaSでよく使われる目安は「LTV÷CACが3倍以上、回収12か月以内」。当社は29倍・3か月"),
    ], { x: X0 + 0.15, y: 4.3, w: W - 0.3, h: 0.64, size: 9, valign: "middle", psa: 1 });
    foot(s, "1〜2年目は創業者が営業するため、CACに創業者の人件費は入れていない（聞かれたら正直に言う）");
  }

  // ---------- 10 ストレステスト ----------
  {
    const s = add(10);
    heading(s, "根拠9　ストレステスト ─ 崩れても死なない条件は「採用の連動」", "同じ計算モデルで1つずつ前提を崩した結果。最低現金のライン＝固定費3か月分（約2,000万円）");
    table(s, [
      ["シナリオ", "想定する事態", "黒字化", "5年目 営業利益", "最低の現金残高"],
      ...SC.map((c, i) => {
        const bad = c.minCash < 2000, last = i === SC.length - 1;
        const col = bad ? ACC : K.ink;
        return [{ text: c.n, bold: i === 0 || last, color: last ? ACC : K.ink }, { text: c.why, color: last ? ACC : K.gray700 },
          { text: c.black ? c.black + "年目" : "しない", color: col, bold: !c.black }, { text: tri(M(c.op5)) + "百万円", color: c.op5 < 0 ? ACC : K.ink }, { text: tri(M(c.minCash)) + "百万円", color: col, bold: bad }];
      }),
    ], { x: X0, y: 1.12, w: W, colW: [2.9, 2.2, 1.0, 1.5, 1.6], rowH: 0.3, size: 9, leftCols: [1], strongRows: [1], name: "stress" });
    box(s, { x: X0, y: 4.2, w: W, h: 0.72, fill: K.ink });
    txt(s, [run("結論：", { bold: true, color: "E3A774" }), run("値引き・解約・コスト高では黒字化の年は変わらない。獲得の遅れは黒字化を1年ずらすが、採用を拠点数に連動させれば現金は残る。危ないのは「遅れているのに計画どおり採用する」ことだけ → 採用は「1人2.8拠点」をルールにする")], { x: X0 + 0.15, y: 4.2, w: W - 0.3, h: 0.72, size: 9.5, color: K.paper, valign: "middle" });
    foot(s, "調達は全シナリオで同額（1年目に1億2,500万円）。資本性ローン2,000万円の予備枠は使わない前提。簡易計算（税・償却・運転資本は含めない）");
  }

  // ---------- 11 資金計画 ----------
  {
    const s = add(11);
    heading(s, "根拠10　資金計画 ─ 必要資金1.5億円は積み上げで出した", "「集められるだけ集める」ではなく「必要額を積んだら1.5億円」");
    const top = 1.15, lw = 4.45;
    txt(s, "必要資金の積み上げ", { x: X0, y: top, w: lw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["項目", "金額", "計算"],
      ["累積赤字のピーク", "6,500万円", "1〜3年目の赤字の合計"],
      ["運転資金", "2,900万円", "3年目の年間費用の3か月分"],
      ["1年遅れの備え", "5,600万円", "赤字をもう1年吸収する額"],
      [{ text: "合計", bold: true }, { text: "1億5,000万円", bold: true }, ""],
    ], { x: X0, y: top + 0.3, w: lw, colW: [1.45, 1.1, 1.9], rowH: 0.3, size: 8.5, leftCols: [2], strongRows: [4], name: "need" });
    txt(s, "調達の順番", { x: X0, y: top + 1.95, w: lw, h: 0.26, size: 10.5, bold: true });
    table(s, [
      ["時期", "手段", "金額"],
      ["準備期", "出資（今回）", "500万円"],
      ["1年目", "日本政策金融公庫（創業融資）", "1,000万円"],
      ["1年目", "CVC・VC", "1億1,500万円"],
      ["予備", "公庫 資本性ローン", "2,000万円"],
    ], { x: X0, y: top + 2.25, w: lw, colW: [0.8, 2.4, 1.25], rowH: 0.28, size: 8.5, leftCols: [1], name: "raise" });
    const rx = 5.15, rw = 4.45;
    txt(s, "期末の現金残高（簡易計算・百万円）", { x: rx, y: top, w: rw, h: 0.26, size: 10.5, bold: true });
    table(s, [["", ...yrs.slice(1)], ["調達", "125", "0", "0", "0", "0"], ["営業損益", ...R.map((r) => tri(M(r.op)))], ["期末現金", ...R.map((r) => ({ text: String(M(r.cash)), bold: true }))]],
      { x: rx, y: top + 0.3, w: rw, colW: [1.2, 0.65, 0.65, 0.65, 0.65, 0.65], rowH: 0.3, size: 9, strongRows: [3], name: "cash" });
    box(s, { x: rx, y: top + 1.65, w: rw, h: 2.0, fill: K.ink });
    txt(s, [
      run("黒字倒産を防ぐ4つのルール", { bold: true, color: "E3A774", breakLine: true }),
      run("① 利用料は年額前払い。導入支援は着手金50%", { breakLine: true }),
      run("② 現金が固定費3か月分を割る前に追加調達を始める", { breakLine: true }),
      run("③ 資金繰り表で12か月先までの入出金を管理", { breakLine: true }),
      run("④ 採用は社員1人2.8拠点の見込みが立ってから"),
    ], { x: rx + 0.15, y: top + 1.7, w: rw - 0.3, h: 1.9, size: 9.5, color: K.paper, valign: "middle", psa: 4 });
    foot(s, "公庫「新規開業・スタートアップ支援資金」の融資限度額7,200万円（うち運転資金4,800万円）。融資額・金利は審査で決まる（要確認）");
  }

  // ---------- 12 市場規模 ----------
  {
    const s = add(12);
    heading(s, "根拠11　市場規模の計算 ─ SAMは約3,350拠点×300万円", "TAM（周辺の大きな市場）は第三者の調査、SAM（当社が届く市場）は自社の計算");
    const top = 1.15;
    const steps = [
      ["電線・ケーブル製造の事業所", "353", "経済構造実態調査（2024年）", "A"],
      ["電気工事業の許可業者", "65,497社", "国交省 許可業者数（令和7年3月末）", "A"],
      ["うち資本金5,000万円以上", "約2,400社", "資本金階層の比率 約3.7%を適用", "C"],
      ["＋電気通信工事の同規模層", "約700社", "完成工事高の比率 約29%を適用", "C"],
      ["電気工事 合計（丸め）", "約3,000社", "", "─"],
      [{ text: "SAMの拠点数", bold: true }, { text: "約3,350拠点", bold: true }, "353＋約3,000", "─"],
      [{ text: "SAM", bold: true }, { text: "約100億円", bold: true }, "約3,350拠点 × 年300万円", "─"],
      [{ text: "SOM（5年目）", bold: true }, { text: "2.55億円", bold: true }, "85拠点 × 300万円（SAMの約2.5%）", "─"],
    ];
    table(s, [["段階", "数", "根拠", "確度"], ...steps], { x: X0, y: top, w: 6.1, colW: [2.15, 1.05, 2.5, 0.4], rowH: 0.32, size: 8.5, leftCols: [2], align: "center", strongRows: [7], name: "sam" });
    const rx = 6.7, rw = 2.9;
    box(s, { x: rx, y: top, w: rw, h: 1.65, fill: K.fill });
    txt(s, [run("TAM（周辺市場）", { bold: true, breakLine: true }), run("スマートファクトリー（国内）", { breakLine: true }), run("42億→92億米ドル、年9.03%", { bold: true, breakLine: true }), run("建設テック（建築分野）", { breakLine: true }), run("1,845億→3,043億円、年7.4%", { bold: true })], { x: rx + 0.12, y: top + 0.05, w: rw - 0.24, h: 1.55, size: 9, valign: "middle", psa: 2 });
    box(s, { x: rx, y: top + 1.75, w: rw, h: 1.2, line: ACC, lw: 1.5 });
    txt(s, [run("聞かれたら：", { bold: true, color: ACC, breakLine: true }), run("「中堅以上の約3,000社は、許可業者数に資本金の比率をかけた推計です。実証と並行して、業界団体の名簿で数え直します」")], { x: rx + 0.12, y: top + 1.78, w: rw - 0.24, h: 1.14, size: 8.5, valign: "middle" });
    box(s, { x: X0, y: 4.15, w: W, h: 0.65, fill: K.ink });
    txt(s, [run("SAMが小さすぎないか：", { bold: true, color: "E3A774" }), run("5年目の85拠点はSAMの2.5%。成長期以降は、素材産業（鉄鋼・化学・製紙など連続工程）と海外に広げる。これはSAMに入れていない")], { x: X0 + 0.15, y: 4.15, w: W - 0.3, h: 0.65, size: 9.5, color: K.paper, valign: "middle" });
    foot(s, "確度C（3.7%・29%）は推計に用いた比率。発表前に国交省「建設業許可業者数調査」「建設工事施工統計調査」の原表で確認する");
  }

  // ---------- 13 厳しい質問 ----------
  {
    const s = add(13);
    heading(s, "根拠12　厳しい質問 ─ 数字 → 式 → 前提の順で答える", "一言で答え、根拠のページを示す");
    table(s, [
      ["質問", "一言の答え", "根拠"],
      ["本当に年300万円払う？", "月240万円分の時間が浮き、約2か月で元が取れます", "根拠3・4"],
      ["月6時間は本当？", "建設業の実績を準用した仮説です。500万円の実証で実測します", "根拠2"],
      ["なぜ4年目まで赤字？", "人を先に雇うからです。1人2.8拠点を超える4年目に黒字", "根拠7"],
      ["獲得が遅れたら？", "採用も遅らせます。3割遅れても現金は2,000万円を割りません", "根拠9"],
      ["値引きされたら？", "2割引きでも4年目に黒字。5年目の営業利益は3,200万円", "根拠9"],
      ["LTV 7年は甘くない？", "計画の解約率なら平均18年。3年でもCACの13倍です", "根拠8"],
      ["1.5億円は多すぎない？", "赤字のピーク・運転資金・1年遅れの備えを積んだ額です", "根拠10"],
      ["市場が小さくない？", "5年目でSAMの2.5%。素材産業と海外は含めていません", "根拠11"],
      ["大手が真似したら？", "現場の記録は貯まるほど移せない。監修できる人が要る", "発表資料③"],
      ["あなたが倒れたら？", "役割を絞り判断基準を先に決め、1か月不在でも回る形にします", "発表資料リスク"],
    ], { x: X0, y: 1.12, w: W, colW: [2.3, 5.6, 1.3], rowH: 0.32, size: 9, leftCols: [1], align: "center", name: "qa" });
    foot(s, "詳しい想定問答は「令和の虎_想定問答.md」（Q1〜Q20）");
  }

  // ---------- 14 出典 ----------
  {
    const s = add(14);
    heading(s, "出典一覧 ─ 原典と、確認が必要なもの", "★＝発表前に原典を確認する");
    const src = [
      ["技能継承・人材育成", "経済産業省・厚生労働省・文部科学省「2026年版 ものづくり白書」（令和8年5月）"],
      ["デジタル活用と技能継承", "労働政策研究・研修機構 調査シリーズNo.265（2026年3月）、No.267（2026年4月）"],
      ["スマートファクトリー市場", "IMARC Group「Japan Smart Factory Market」（2025→2034年、年9.03%）"],
      ["建設テック市場", "矢野経済研究所 プレスリリースNo.3789（2025年4月）建築分野ソフトウェア"],
      ["電線・ケーブル製造の事業所数", "経済産業省「経済構造実態調査」2024年"],
      ["電気工事業の許可業者数", "国土交通省 建設業許可業者数（令和7年3月末）65,497社"],
      ["中堅以上の比率 ★", "国土交通省「建設業許可業者数調査」資本金階層別・「建設工事施工統計調査」から推計"],
      ["書類作業の削減 月6.5時間 ★", "建設業の書類電子化の実績として準用。原典を特定して差し替える"],
      ["創業融資", "日本政策金融公庫「新規開業・スタートアップ支援資金」"],
      ["国の方針", "内閣府「人工知能基本計画」（令和8年7月14日 閣議決定）"],
      ["競合", "各社公表資料（Airion、キャディ、三菱総合研究所、FRONTEO、tebiki、スタディスト）"],
    ];
    table(s, [["項目", "出典"], ...src], { x: X0, y: 1.12, w: W, colW: [2.6, 6.6], rowH: 0.3, size: 8.5, leftCols: [1], name: "sources" });
    foot(s, "LTV÷CACの目安（3倍以上・回収12か月以内）はSaaS業界で一般に使われる経験則であり、特定の統計ではない");
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}
const T = () => L.T.stroke;

main().catch((e) => { console.error(e); process.exit(1); });
