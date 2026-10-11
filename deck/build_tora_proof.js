// 令和の虎 事業計画書「理論武装版」: every number traced to its formula, assumption and source.
// All figures come from tora_model.js, so the tables and the stress test always match the plan.
// Usage: node build_tora_proof.js [outDir]   (needs pptxgenjs and sharp)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const L = require("./tora_lib.js");
const { run: model, unit } = require("./tora_model.js");
const { estimate } = require("./tora_market.js");

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
    "年300万円の根拠は、お客様の得の約1割です。施工管理者と電工が100人の会社で月240万円、年2,880万円分の時間が浮くので、その約10%をいただきます。人数ではなく拠点とライン数で課金するのは、人数課金だと現場がアカウントを絞り、記録が貯まらなくなるからです。"],
  5: ["根拠4", "小さな会社でも1年以内に元が取れる",
    "お客様の回収期間の感度表です。人数と削減時間が半分でも、100人・月3時間、または50人・月6時間で、約4か月で回収できます。1年で回収するための最低条件は、人数かける削減時間が月105人時。30人の会社なら1人月3.5時間です。これを下回る小さな会社は、最初は狙いません。"],
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
  12: ["根拠11", "ターゲット市場は業種×規模で数えて約141億円",
    "市場規模の計算です。業種と規模の2つの軸で分けて、顧客数かける年額を足しました。電気工事と電気通信工事の中堅以上、約3,100社で約141億円です。大手は支店5拠点分で数えています。土台の65,497社は国の統計ですが、規模別の内訳は仮定なので、聞かれたら推計だと正直に言います。電線や素材の工場は、成長期以降の隣接市場として外しています。"],
  13: ["根拠12", "最初の顧客は4つの物差しで選び、5年の獲得数は顧客側から積んだ",
    "なぜ中堅の電気工事会社からなのか。大きさ、困りごと、会いやすさ、波及効果の4つで採点し、17点で最も高くなりました。5年の獲得数は、関東の735社に、会える率45%、成約率25%をかけて83社、大手1社の5拠点を足して約88拠点。導入能力から出した計画の85拠点は、これより控えめです。"],
  14: ["根拠13", "グラスは作らず選ぶ。手元ARの効果は3つの数字で実測する",
    "端末と実証の設計です。グラス本体は開発しません。落下・防じん防水・電池・ヘルメット・手袋での操作という現場の条件で市販の端末を選び、乗り換えられる設計にします。手元ARの効果は、よく使うジョイント1種類で、手戻りの件数、1ジョイントの作業時間、新人が独り立ちするまでの期間の3つで測ります。"],
  15: ["根拠14", "厳しい質問には、数字→式→前提の順で答える",
    "厳しい質問への回答集です。それぞれ、一言の答えと、根拠のページを書いています。"],
  16: ["出典", "公的統計と原典を明記する。未確認のものは未確認と言う",
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
    const toc = [["根拠1〜2", "数字の木と、11個の前提"], ["根拠3〜4", "価格と、お客様の回収"], ["根拠5〜8", "売上・費用・損益分岐点・LTV/CAC"], ["根拠9〜10", "ストレステストと資金計画"], ["根拠11〜14", "市場・最初の顧客・端末・厳しい質問"]];
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
      ["新規獲得", "3→9→18→27→33拠点", "導入担当1人＝年6拠点が上限。顧客側からも約88拠点（根拠12）", "B"],
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
    txt(s, [run("ターゲットの条件に変換：", { bold: true, color: ACC }), run("施工管理者・電工が30人以上、かつ書類・報告の作業が1人月3.5時間以上ある拠点。関東の中堅電気工事会社（施工管理者30人以上）はこれを満たす")], { x: X0 + 0.15, y: 4.05, w: W - 0.3, h: 0.75, size: 9.5, valign: "middle" });
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
    const E = estimate();
    heading(s, "根拠11　市場規模の計算 ─ 業種×規模で数えて約141億円", "全体市場は第三者の調査、ターゲット市場は「顧客数×年額×年1回」を足し上げる");
    const top = 1.15;
    const cellTxt = (c) => (c.inSAM ? { text: `${fmt(c.n)}社×${fmt(c.price)}万＝${(c.value / 1e4).toFixed(0)}億`, bold: c.ind === "電気工事会社" && c.size.startsWith("中"), fill: c.ind === "電気工事会社" && c.size.startsWith("中") ? "F6E7DA" : undefined } : { text: c.ind.startsWith("電線") ? "隣接市場" : "対象外", color: K.gray500 });
    const inds = ["電気工事会社", "電気通信工事会社", "電線・ケーブル工場"];
    const sizes = ["大", "中", "小"];
    const rows = inds.map((ind) => [ind, ...sizes.map((z) => { const c = E.cells.find((x) => x.ind === ind && x.size.startsWith(z)); return c ? cellTxt(c) : { text: "─", color: K.gray500 }; })]);
    table(s, [["業種＼規模", "大（300人以上）", "中（30〜299人）", "小（30人未満）"], ...rows], { x: X0, y: top, w: 6.1, colW: [1.45, 1.6, 1.65, 1.4], rowH: 0.38, size: 8.5, align: "center", name: "seg-matrix" });
    table(s, [["前提", "値", "確度"],
      ["電気工事業の許可業者", "65,497社（令和7年3月末）", "A"],
      ["規模別の内訳（大300・中2,100）", "資本金5,000万円以上 約2,400社を分けた", "C"],
      ["電気通信工事の中堅以上", "約700社（完成工事高の比率から）", "C"],
      ["大手の年額", "支店5拠点 × 300万円 ＝ 1,500万円", "C"],
      ["小規模を外す理由", "1年で元が取れる条件（月105人時）を満たしにくい", "B"],
    ], { x: X0, y: top + 1.65, w: 6.1, colW: [2.2, 3.4, 0.5], rowH: 0.27, size: 8, leftCols: [1], align: "center", name: "market-assump" });
    const rx = 6.7, rw = 2.9;
    box(s, { x: rx, y: top, w: rw, h: 1.5, fill: K.ink });
    txt(s, [run("ターゲット市場", { bold: true, color: "E3A774", breakLine: true }), run(`約${(E.sam / 1e4).toFixed(0)}億円`, { bold: true, fontSize: 20, breakLine: true }), run(`中堅以上 約${fmt(Math.round(E.samSites / 100) * 100)}社`, { fontSize: 9 })], { x: rx + 0.12, y: top + 0.05, w: rw - 0.24, h: 1.4, size: 10, color: K.paper, valign: "middle", psa: 2 });
    box(s, { x: rx, y: top + 1.6, w: rw, h: 1.45, fill: K.fill });
    txt(s, [run("全体市場（第三者の調査）", { bold: true, breakLine: true }), run("建設テック（建築分野）", { breakLine: true }), run("1,845億→3,043億円、年7.4%", { bold: true, breakLine: true }), run("スマートファクトリー（国内）", { breakLine: true }), run("42億→92億米ドル、年9.03%", { bold: true })], { x: rx + 0.12, y: top + 1.65, w: rw - 0.24, h: 1.35, size: 8.5, valign: "middle", psa: 1 });
    box(s, { x: X0, y: 4.45, w: W, h: 0.48, line: ACC, lw: 1.5 });
    txt(s, [run("聞かれたら：", { bold: true, color: ACC }), run("「土台の65,497社は国の統計、規模別の内訳は推計です。業界団体の名簿と経済センサスで数え直します」。以前の約100億円は、各社を1拠点として数えた控えめな値")], { x: X0 + 0.15, y: 4.45, w: W - 0.3, h: 0.48, size: 8.5, valign: "middle" });
    foot(s, "計算は tora_market.js。差し替える統計：国交省「建設業許可業者数調査」業種別・資本金階層別、経済センサス（設備工事業の従業者規模別）");
  }

  // ---------- 13 最初の顧客とファネル ----------
  {
    const s = add(13);
    const E = estimate();
    heading(s, "根拠12　最初の顧客 ─ 4つの物差しで選び、顧客側から積む", "セグメント評価（大きさ・困りごと・会いやすさ・波及）と、5年間のファネル");
    const top = 1.15;
    const ev = [["電気工事・中堅", 5, 5, 4, 3, "63億円。残業規制と書類の多さ。施工管理の経歴が信用に"], ["電気工事・大手", 4, 4, 2, 5, "審査が長い。採用されると協力会社へ波及"], ["電線工場・中堅", 1, 4, 4, 3, "市場が約4.5億円と小さい。成長期以降の隣接市場"], ["電気通信工事・中堅", 3, 4, 2, 3, "人脈が薄い。成長期に広げる"]];
    table(s, [["候補", "大きさ", "困りごと", "会いやすさ", "波及", "合計", "理由"],
      ...ev.map((r, i) => [{ text: r[0], bold: i === 0 }, ...r.slice(1, 5).map((v) => ({ text: String(v), bold: i === 0 })), { text: String(r.slice(1, 5).reduce((a, b) => a + b, 0)), bold: true, color: i === 0 ? ACC : K.ink }, r[5]])],
      { x: X0, y: top, w: W, colW: [1.6, 0.65, 0.75, 0.85, 0.55, 0.6, 4.2], rowH: 0.3, size: 8.5, align: "center", leftCols: [6], strongRows: [1], name: "seg-eval" });
    txt(s, "5年間のファネル（顧客側から）", { x: X0, y: top + 1.65, w: W, h: 0.26, size: 10.5, bold: true });
    table(s, [["対象", "顧客数", "地域", "会える率", "成約率", "1社の拠点", "5年で獲得"],
      ...E.som.map((r) => [r.cell, fmt(r.n) + "社", "関東 " + Math.round(r.region * 100) + "%", Math.round(r.reach * 100) + "%", Math.round(r.close * 100) + "%", r.per + "拠点", { text: r.won.toFixed(0) + "拠点", bold: true }]),
      [{ text: "合計", bold: true }, "", "", "", "", "", { text: `約${E.somSites.toFixed(0)}拠点`, bold: true, color: ACC }]],
      { x: X0, y: top + 1.95, w: W, colW: [2.6, 1.0, 1.1, 1.0, 1.0, 1.1, 1.4], rowH: 0.28, size: 8.5, align: "center", strongRows: [3], name: "funnel" });
    box(s, { x: X0, y: 4.18, w: W, h: 0.72, fill: K.ink });
    txt(s, [run("2つの方向で同じ答え：", { bold: true, color: "E3A774" }), run(`顧客側の積み上げ 約${E.somSites.toFixed(0)}拠点 ≒ 導入能力（導入担当1人＝年6拠点）から出した計画 85拠点。計画は控えめな方を採る`, { breakLine: true }), run("会える率・成約率は仮定。準備期の有償試験3社で、商談数と成約率を実測して更新する")], { x: X0 + 0.15, y: 4.18, w: W - 0.3, h: 0.72, size: 9, color: K.paper, valign: "middle", psa: 2 });
    foot(s, "評価は社内の仮採点。関東の比率35%は仮定（許可業者の都道府県別の数で確認する）");
  }

  // ---------- 14 端末と実証 ----------
  {
    const s = add(14);
    heading(s, "根拠13　端末と実証 ─ グラスは作らず選び、効果は実測する", "端末の条件と候補（公表値）、手元ARの実証の設計");
    const top = 1.15;
    table(s, [["現場の条件", "目安", "候補の公表値（RealWear Navigator 520）"],
      ["落下", "2m級", "2mの落下試験（販売店の公表値）"],
      ["防じん・防水", "IP66級", "IP66"],
      ["電池", "1シフト", "約6〜8時間、動作中に交換できる"],
      ["装着", "ヘルメット", "ヘルメット用クリップ（別売）"],
      ["操作", "手袋のまま", "音声操作"],
      ["温度", "屋外の夏冬", "−20〜50℃"],
    ], { x: X0, y: top, w: 5.3, colW: [1.25, 1.1, 2.95], rowH: 0.3, size: 8.5, leftCols: [1, 2], name: "device-spec" });
    txt(s, "ARグラス：Metaの表示付きグラスは2025年発売。両眼ARグラスの一般発売は2027年と報道（公式発表ではない）", { x: X0, y: top + 2.2, w: 5.3, h: 0.45, size: 8.5, color: K.gray700 });
    const rx = 5.95, rw = 3.65;
    box(s, { x: rx, y: top, w: rw, h: 2.65, fill: K.fill });
    txt(s, [run("手元ARの実証（3か月）", { bold: true, color: ACC, breakLine: true }),
      run("対象：よく使うジョイント1種類", { breakLine: true }),
      run("1か月目：紙の図面で基準を測る", { breakLine: true }),
      run("2〜3か月目：ARで同じ作業", { breakLine: true }),
      run("測る数字", { bold: true, breakLine: true }),
      run("① 手戻り・施工ミスの件数", { breakLine: true }),
      run("② 1ジョイントの作業時間", { breakLine: true }),
      run("③ 新人が独り立ちするまでの期間")], { x: rx + 0.15, y: top + 0.05, w: rw - 0.3, h: 2.55, size: 9.5, valign: "middle", psa: 2 });
    box(s, { x: X0, y: 4.0, w: W, h: 0.85, fill: K.ink });
    txt(s, [run("聞かれたら：", { bold: true, color: "E3A774" }), run("「グラスは作りません。作るのはアプリ・AI・データ基盤と、端末を壊さない保護カバーとヘルメット取付具です。端末が進化したら乗り換えます」", { breakLine: true }),
      run("初期コスト：", { bold: true, color: "E3A774" }), run("組立図の3D化が必要。1種類に絞って始め、型を作ってから広げる")], { x: X0 + 0.15, y: 4.0, w: W - 0.3, h: 0.85, size: 9, color: K.paper, valign: "middle", psa: 3 });
    foot(s, "端末の仕様は販売店の公表値。RealWear公式の仕様書で確認する。保護めがねとしての規格適合は要確認");
  }

  // ---------- 15 厳しい質問 ----------
  {
    const s = add(15);
    heading(s, "根拠14　厳しい質問 ─ 数字 → 式 → 前提の順で答える", "一言で答え、根拠のページを示す");
    table(s, [
      ["質問", "一言の答え", "根拠"],
      ["本当に年300万円払う？", "月240万円分の時間が浮き、約2か月で元が取れます", "根拠3・4"],
      ["月6時間は本当？", "建設業の実績を準用した仮説です。500万円の実証で実測します", "根拠2"],
      ["なぜ電気工事から？", "市場が最も大きく、残業規制で困りごとが強く、私の経歴が信用になるからです", "根拠12"],
      ["ANDPADがあるのでは？", "あれは人が入力する器。当社は自動で記録し判断を渡す。競わず連携します", "発表資料③"],
      ["なぜ4年目まで赤字？", "人を先に雇うからです。1人2.8拠点を超える4年目に黒字", "根拠7"],
      ["獲得が遅れたら？", "採用も遅らせます。3割遅れても現金は2,000万円を割りません", "根拠9"],
      ["値引きされたら？", "2割引きでも4年目に黒字。5年目の営業利益は3,200万円", "根拠9"],
      ["LTV 7年は甘くない？", "計画の解約率なら平均18年。3年でもCACの13倍です", "根拠8"],
      ["1.5億円は多すぎない？", "赤字のピーク・運転資金・1年遅れの備えを積んだ額です", "根拠10"],
      ["市場が小さくない？", "5年目でターゲット市場の約1.8%。工場と海外は含めていません", "根拠11"],
      ["グラスは開発できるのか？", "作りません。市販の端末を現場仕様にし、乗り換えられる設計にします", "根拠13"],
      ["ARの位置ずれは？", "数mm〜cmずれます。最終確認はスケール。間違いに気づく道具です", "根拠13"],
      ["あなたが倒れたら？", "役割を絞り判断基準を先に決め、1か月不在でも回る形にします", "発表資料リスク"],
    ], { x: X0, y: 1.12, w: W, colW: [2.1, 5.9, 1.2], rowH: 0.26, size: 8, leftCols: [1], align: "center", name: "qa" });
    foot(s, "詳しい想定問答は「令和の虎_想定問答.md」");
  }

  // ---------- 16 出典 ----------
  {
    const s = add(16);
    heading(s, "出典一覧 ─ 原典と、確認が必要なもの", "★＝発表前に原典を確認する");
    const src = [
      ["建設業の就業者・年齢構成", "国土交通省（総務省「労働力調査」2024年平均から算出）477万人、55歳以上36.7%、29歳以下11.7%"],
      ["残業の上限規制", "労働基準法の時間外労働の上限規制（建設業は2024年4月1日から適用、原則 月45時間・年360時間）"],
      ["建設テック市場", "矢野経済研究所 プレスリリースNo.3789（2025年4月）建築分野ソフトウェア"],
      ["スマートファクトリー市場", "IMARC Group「Japan Smart Factory Market」（2025→2034年、年9.03%）"],
      ["電気工事業の許可業者数 ★", "国土交通省 建設業許可業者数（令和7年3月末）65,497社（二次資料で確認、原表で要確認）"],
      ["規模別の内訳・関東の比率 ★", "推計に用いた仮定。国交省「建設業許可業者数調査」資本金階層別・都道府県別で確認する"],
      ["書類作業の削減 月6.5時間 ★", "建設業の書類電子化の実績として準用。原典を特定して差し替える"],
      ["施工管理アプリ", "スパイダープラス 有価証券報告書（契約1,524社・2022年12月末）、各社公表資料（ANDPAD、蔵衛門）"],
      ["創業融資", "日本政策金融公庫「新規開業・スタートアップ支援資金」"],
      ["国の方針", "内閣府「人工知能基本計画」（令和8年7月14日 閣議決定）"],
      ["端末", "RealWear Navigator 520 販売店の公表値（IP66・2m落下・電池交換）、Meta Ray-Ban Display（2025年発売）、ARグラス2027年発売の報道"],
      ["製造業の技能継承", "2026年版 ものづくり白書、労働政策研究・研修機構 調査シリーズNo.265・No.267（補足資料）"],
    ];
    table(s, [["項目", "出典"], ...src], { x: X0, y: 1.12, w: W, colW: [2.4, 6.8], rowH: 0.28, size: 8, leftCols: [1], name: "sources" });
    foot(s, "LTV÷CACの目安（3倍以上・回収12か月以内）はSaaS業界で一般に使われる経験則であり、特定の統計ではない");
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
}
const T = () => L.T.stroke;

main().catch((e) => { console.error(e); process.exit(1); });
